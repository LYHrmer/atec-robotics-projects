"""Independent 1-second relative-follow check for a contact-grasp run.

OFFLINE ONLY. Reads the recorded telemetry (actual object pose and gripper pose)
for external audit. Nothing here is or was a policy input. Passing the follow
check plus the mesh/root rise is grasp evidence for ONE object at ONE moment; it
is NOT transport, delivery, objects_in_circle or Task B completion.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_directory', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

run = args.run_directory
result = json.loads((run/'result.json').read_text())
meta = json.loads((run/'environment_metadata.json').read_text())
dt = float(meta['step_dt'])
telemetry = np.load(run/'telemetry.npz')
events = json.loads((run/'scoring_events.json').read_text())
rows = [json.loads(line) for line in (run/'trace.jsonl').open()]

phases = np.array([row.get('policy_debug', {}).get('probe_phase') or '' for row in rows])
indices = np.flatnonzero(phases != '')
if not len(indices):
    raise SystemExit('probe/contact phases were never entered')

# Independent target association: the object nearest the gripper at the first
# original score event. Never a policy input.
state = events[0]['state']
distance = np.linalg.norm(np.asarray(state['object_xyz'])-np.asarray(state['gripper_xyz']), axis=1)
target = int(np.argmin(distance))

start = int(indices[0])
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

# Inclusive window: N samples span (N-1)*dt, so a full 1.00 s needs 51 samples at 50 Hz.
samples = int(round(FOLLOW_WINDOW_S/dt))+1
report = {
    'run_id': run.name,
    'scope': 'OFFLINE independent audit of recorded actual object/gripper pose and the original full '
             'mesh. No value here was ever a policy input.',
    'object_index_1based': target+1,
    'stop_reason': result['stop_reason'],
    'official_terminated': bool(result['terminated']),
    'official_truncated': bool(result['truncated']),
    'reward_term_totals_raw': result['reward_term_totals_raw'],
    'note_on_reward': 'grasped_objects only means the gripper base came within the official .20 m of an '
                      'object root; objects_in_circle is the delivery term and is 0 here',
    'admission_thresholds': {'mesh_lowest_surface_rise_m': RISE_MESH_M, 'root_rise_m': RISE_ROOT_M,
                             'follow_window_s': FOLLOW_WINDOW_S, 'follow_drift_m': FOLLOW_DRIFT_M},
    'baseline_step': int(steps[0]),
    'baseline_basis': 'the first added-phase (CONTACT_OPEN) sample, i.e. BEFORE the wrist rotation and '
                      'the extra leg reference, so no preparation-stage push or tip is hidden',
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
                and not result['terminated'])
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
