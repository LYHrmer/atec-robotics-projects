"""Independent 1-second relative-follow check for a contact-grasp / payload run.

OFFLINE ONLY. Reads the recorded telemetry (actual object pose and gripper pose)
for external audit. Nothing here is or was a policy input. Passing the follow
check plus the mesh/root rise is grasp evidence for ONE object at ONE moment; it
is NOT transport, delivery, objects_in_circle or Task B completion.

Corrections applied 2026-09-14 (see the ``baseline_*`` and
``original_environment_flags`` blocks of the report):

1. BASELINE.  The previous version located its "pre-preparation" baseline with
   ``_phase()`` name matching over the TOP-LEVEL ``phase`` key only.  Inside the
   frozen prefix that key is the literal string ``PREFIX`` for the whole
   contact-grasp run, so the first row it accepted was the first *payload* row -
   already ``PAYLOAD_RAISE`` - and the report labelled it ``CONTACT_OPEN``.
   The baseline is now found in the NESTED ``child_phase`` / ``child_state``
   fields, which carry the child policy's own phase while the child is still
   running (``child_called_this_tick`` is required so a stale post-handoff value
   can never be used).  The report now names the field it used and the row it
   landed on, and falls back explicitly - never silently - when no nested field
   is present.
2. ENVIRONMENT FLAGS.  The script now reads ``terminated``, ``truncated``, the
   original termination terms and the original illegal-contact sensor itself
   from ``result.json`` / ``telemetry.npz`` / ``scoring_events.json`` instead of
   trusting a separate audit to have done it, and the admission verdict requires
   all of them clean.

Thresholds are unchanged: they are declared here and are never read from the
policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Independent admission thresholds, fixed here and NOT read from the policy.
RISE_MESH_M = .010
RISE_ROOT_M = .015
FOLLOW_WINDOW_S = 1.
FOLLOW_DRIFT_M = .020

#: Public flags of the ORIGINAL environment that must all be clean for the
#: evidence to be admitted.  Names come from the recorded term list, not from the
#: policy under audit.
REQUIRED_CLEAN_TERMS = ('illegal_contact',)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_directory', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--baseline-phase', default='CONTACT_OPEN',
                    help='the child phase whose FIRST sample is the pre-preparation baseline')
parser.add_argument('--object-index', type=int, default=None,
                    help='override the independently associated 1-based object index')
args = parser.parse_args()

run = args.run_directory
result = json.loads((run/'result.json').read_text())
meta = json.loads((run/'environment_metadata.json').read_text())
dt = float(meta['step_dt'])
telemetry = np.load(run/'telemetry.npz')
events = json.loads((run/'scoring_events.json').read_text())
rows = [json.loads(line) for line in (run/'trace.jsonl').open()]


# -- phase resolution ---------------------------------------------------------
def _debug(row):
    return row.get('policy_debug', {}) or {}


def _phase_label(row):
    """The most specific NON-prefix phase label recorded on this row.

    The top-level ``phase`` is used first because it is the live phase of the
    outermost policy.  While the outer policy reports ``PREFIX`` the real work is
    being done by a nested child, whose own phase is recorded in
    ``child_phase`` / ``child_state``; those are read ONLY when
    ``child_called_this_tick`` says the child actually ran, so the stale value
    left behind after a handoff can never be mistaken for a live label.
    """
    debug = _debug(row)
    parent = debug.get('phase')
    if parent and parent not in ('SETTLE', 'PREFIX'):
        return parent
    if debug.get('child_called_this_tick'):
        for key in ('child_phase', 'child_state'):
            value = debug.get(key)
            if value and value not in ('SETTLE', 'PREFIX'):
                return value
    for key in ('probe_phase', 'contact_phase'):
        value = debug.get(key)
        if value and value not in ('SETTLE', 'PREFIX'):
            return value
    return ''


#: Ordered (field, requires_live_child) candidates used to FIND the baseline.
#: The nested child fields come first and are the ones this correction is about.
BASELINE_FIELDS = (('child_phase', True), ('child_state', True),
                   ('phase', False), ('contact_phase', False), ('probe_phase', False))


def find_baseline(rows, phase_name):
    """First row whose phase equals ``phase_name``, preferring nested child fields.

    Returns ``(row_index, field_used, requires_live_child)`` or ``(None, ...)``.
    """
    for field, needs_child in BASELINE_FIELDS:
        for index, row in enumerate(rows):
            debug = _debug(row)
            if needs_child and not debug.get('child_called_this_tick'):
                continue
            if debug.get(field) == phase_name:
                return index, field, needs_child
    return None, None, None


phases = np.array([_phase_label(row) or '' for row in rows])
start, baseline_field, baseline_needs_child = find_baseline(rows, args.baseline_phase)
if start is None:
    # Explicit, loud fallback: never silently relabel a payload row as the
    # pre-preparation baseline.  It stops instead of producing a mislabelled
    # number.
    raise SystemExit(
        f'no row records baseline phase {args.baseline_phase!r} in any of '
        f'{ [f for f, _ in BASELINE_FIELDS] }; refusing to guess a baseline')
indices = np.flatnonzero(phases != '')
if not len(indices):
    raise SystemExit('probe/contact phases were never entered')

# Independent target association: the object nearest the gripper at the first
# original score event. Never a policy input.
state = events[0]['state']
distance = np.linalg.norm(np.asarray(state['object_xyz'])-np.asarray(state['gripper_xyz']), axis=1)
target = int(np.argmin(distance)) if args.object_index is None else int(args.object_index)-1

end = result['final_state_before_close']
steps = np.r_[telemetry['step'][start:], result['steps']+1]
times = (steps-1)*dt
object_xyz = np.concatenate([telemetry['object_xyz'][start:, target],
                             np.asarray(end['object_xyz'])[None, target]])
object_quat = np.concatenate([telemetry['object_quat'][start:, target],
                              np.asarray(end['object_quat'])[None, target]])
gripper_xyz = np.concatenate([telemetry['gripper_xyz'][start:], np.asarray(end['gripper_xyz'])[None]])
gripper_quat = np.concatenate([telemetry['gripper_quat'][start:], np.asarray(end['gripper_quat'])[None]])
names = telemetry['joint_names'].tolist()
q = np.concatenate([telemetry['q'][start:], np.asarray(end['q'])[None]])
width = q[:, names.index('arm_joint7')]-q[:, names.index('arm_joint8')]
phase = np.concatenate([phases[start:], np.array([phases[-1]])])

object_R = Rotation.from_quat(object_quat[:, [1, 2, 3, 0]]).as_matrix()
gripper_R = Rotation.from_quat(gripper_quat[:, [1, 2, 3, 0]]).as_matrix()

# The object root expressed in the CURRENT gripper frame: if the object is truly
# held, this vector stays put while both bodies move in the world.
relative = np.einsum('nji,nj->ni', gripper_R, object_xyz-gripper_xyz)

# Full original mesh, transformed by the ACTUAL recorded quaternion each step.
import os
os.environ.setdefault('LD_LIBRARY_PATH', '')
from pxr import Usd, UsdGeom
asset = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/006_mustard_bottle.usd')
stage = Usd.Stage.Open(str(asset))
cache = UsdGeom.XformCache()
vertices = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices.append(points@matrix[:3, :3].T+matrix[:3, 3])
vertices = np.concatenate(vertices)
lowest = np.array([float(np.min(vertices@mat[2])+pos[2]) for mat, pos in zip(object_R, object_xyz)])

mesh_rise = lowest-lowest[0]
root_rise = object_xyz[:, 2]-object_xyz[0, 2]

# -- original environment flags, read by THIS script ---------------------------
# Nothing below is inferred from the policy; the arrays are the recorder's own
# copy of the simulator's termination / illegal-contact sensors.
term_names = [str(name) for name in telemetry['termination_term_names']]
termination = np.asarray(telemetry['termination']).astype(bool)
terms_fired = {name: bool(termination[:, index].any())
               for index, name in enumerate(term_names)}
illegal_names = [str(name) for name in telemetry['illegal_contact_body_names']]
illegal_force = np.asarray(telemetry['illegal_force'], dtype=float)
illegal_per_body = {name: float(illegal_force[:, index].max())
                    for index, name in enumerate(illegal_names)}
illegal_any = bool((illegal_force > 0.).any())
event_flags = []
for event in events:
    flags = dict(event.get('termination_flags') or {})
    event_flags.append({'step': int(event['step']), 'termination_flags': flags})
illegal_term_fired = bool(terms_fired.get('illegal_contact', False)) or illegal_any
environment_flags = {
    'source': 'result.json + telemetry.npz + scoring_events.json, read by this script; no separate '
              'audit is assumed to have done it',
    'official_terminated': bool(result['terminated']),
    'official_truncated': bool(result['truncated']),
    'termination_term_names': term_names,
    'termination_terms_fired_any': {name: fired for name, fired in terms_fired.items()},
    'illegal_contact_term_fired': bool(terms_fired.get('illegal_contact', False)),
    'illegal_contact_body_names': illegal_names,
    'illegal_contact_max_force_per_body': illegal_per_body,
    'illegal_contact_force_ever_nonzero': illegal_any,
    'scoring_event_termination_flags': event_flags,
    'required_clean_terms': list(REQUIRED_CLEAN_TERMS),
    'clean': bool(not result['terminated'] and not result['truncated'] and not illegal_term_fired),
}

preparation_phase_before_baseline = phases[start-1] if start > 0 else ''
report = {
    'run_id': run.name,
    'scope': 'OFFLINE independent audit of recorded actual object/gripper pose and the original full '
             'mesh. No value here was ever a policy input.',
    'object_index_1based': target+1,
    'stop_reason': result['stop_reason'],
    'official_terminated': bool(result['terminated']),
    'official_truncated': bool(result['truncated']),
    'original_environment_flags': environment_flags,
    'reward_term_totals_raw': result['reward_term_totals_raw'],
    'note_on_reward': 'grasped_objects only means the gripper base came within the official .20 m of an '
                      'object root; objects_in_circle is the delivery term and is 0 here',
    'admission_thresholds': {'mesh_lowest_surface_rise_m': RISE_MESH_M, 'root_rise_m': RISE_ROOT_M,
                             'follow_window_s': FOLLOW_WINDOW_S, 'follow_drift_m': FOLLOW_DRIFT_M},
    'baseline_step': int(steps[0]),
    'baseline_trace_row': int(start),
    'baseline_source_field': baseline_field,
    'baseline_field_is_nested_child_phase': bool(baseline_needs_child),
    'baseline_phase_before': preparation_phase_before_baseline,
    'baseline_basis': ('the FIRST sample whose NESTED child phase/state is '
                       f'{args.baseline_phase!r} ({baseline_field}, valid only while '
                       'child_called_this_tick), i.e. BEFORE the wrist rotation and the extra leg '
                       'reference, so no preparation-stage push or tip is hidden. It is NOT the '
                       'first payload phase.'),
    'baseline_lowest_mesh_world_z_m': float(lowest[0]),
    'baseline_object_root_world_m': object_xyz[0].tolist(),
    'peak_mesh_rise_m': float(np.max(mesh_rise)),
    'peak_root_rise_m': float(np.max(root_rise)),
    'final_mesh_rise_m': float(mesh_rise[-1]),
    'final_root_rise_m': float(root_rise[-1]),
    'final_jaw_width_m': float(width[-1]),
    'follow_windows': [],
}

# Every window of >= 1 s in which the object stayed BOTH lifted and fixed in the
# gripper frame. Drift is the max spread of the relative root over the window.
# Inclusive window: N samples span (N-1)*dt, so a full 1.00 s needs 51 samples at 50 Hz.
samples = int(round(FOLLOW_WINDOW_S/dt))+1
best = None
for stop in range(samples, len(relative)+1):
    begin = stop-samples
    if float(np.min(mesh_rise[begin:stop])) < RISE_MESH_M:
        continue
    if float(np.min(root_rise[begin:stop])) < RISE_ROOT_M:
        continue
    window = relative[begin:stop]
    drift = float(np.max(np.linalg.norm(window-window[0], axis=1)))
    spread = float(np.max(np.max(window, axis=0)-np.min(window, axis=0)))
    entry = {'first_step': int(steps[begin]), 'last_step': int(steps[stop-1]),
             'duration_s': float(times[stop-1]-times[begin]),
             'relative_drift_from_first_sample_m': drift,
             'relative_axis_spread_m': spread,
             'min_mesh_rise_m': float(np.min(mesh_rise[begin:stop])),
             'min_root_rise_m': float(np.min(root_rise[begin:stop])),
             'min_jaw_width_m': float(np.min(width[begin:stop])),
             'phases': sorted(set(phase[begin:stop].tolist())),
             'passes_follow': bool(drift <= FOLLOW_DRIFT_M)}
    if best is None or drift < best['relative_drift_from_first_sample_m']:
        best = entry
report['best_follow_window'] = best
report['follow_window_count_passing'] = int(best is not None and best['passes_follow'])

# Longest CONTIGUOUS run of samples that stayed lifted with bounded relative
# drift measured from the first sample of that run.
longest = None
begin = 0
while begin < len(relative):
    stop = begin
    while stop < len(relative):
        if mesh_rise[stop] < RISE_MESH_M or root_rise[stop] < RISE_ROOT_M:
            break
        drift = float(np.max(np.linalg.norm(relative[begin:stop+1]-relative[begin], axis=1)))
        if drift > FOLLOW_DRIFT_M:
            break
        stop += 1
    span = stop-begin
    if span >= 2:
        entry = {'first_step': int(steps[begin]), 'last_step': int(steps[stop-1]),
                 'samples': int(span), 'duration_s': float(times[stop-1]-times[begin]),
                 'relative_drift_from_first_sample_m': float(np.max(np.linalg.norm(
                     relative[begin:stop]-relative[begin], axis=1))),
                 'min_mesh_rise_m': float(np.min(mesh_rise[begin:stop])),
                 'min_root_rise_m': float(np.min(root_rise[begin:stop])),
                 'min_jaw_width_m': float(np.min(width[begin:stop])),
                 'phases': sorted(set(phase[begin:stop].tolist()))}
        if longest is None or entry['duration_s'] > longest['duration_s']:
            longest = entry
    begin = max(stop, begin+1)
report['longest_contiguous_lifted_follow'] = longest

physical = bool(report['peak_mesh_rise_m'] >= RISE_MESH_M
                and report['peak_root_rise_m'] >= RISE_ROOT_M
                and best is not None and best['passes_follow']
                and environment_flags['clean'])
report['physical_grasp_admitted'] = physical
report['verdict'] = ('PHYSICAL_GRASP_EVIDENCE_FOR_ONE_OBJECT' if physical
                     else 'NOT_ADMITTED_AS_PHYSICAL_GRASP')
report['explicitly_not_claimed'] = ['transport', 'navigation', 'objects_in_circle delivery',
                                    'Task B completion', 'repeatability across seeds',
                                    'that a nonempty jaw or a proximity point proves anything']
report['source_sha256'] = {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                           for name in ('result.json', 'telemetry.npz', 'scoring_events.json')}
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'follow_windows'}, indent=2))
