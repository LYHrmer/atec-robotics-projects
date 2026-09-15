"""Measured audit of one carry-probe run (offline, external evidence only).

Answers the questions the P1 gate actually asks, from the recorded run:

1. did the base move, in the direction it was commanded, and how far;
2. how did the heading respond to the bounded +-10 deg yaw probes;
3. was the SAME load carried continuously through the movement segments, measured
   as the whole original mesh relative to the TRUE pre-preparation baseline;
4. how much did the load drift relative to the gripper over the whole movement;
5. did the original failure channels fire at all.

The baseline is the first frame whose policy state is the real preparation state,
found from the recorded state column rather than assumed by name - the p13 audit
mislabeled this and used an already-raising frame instead.

Nothing here is or was a policy input. Passing is carry evidence for ONE object in
ONE run; it is not transport, clearance, delivery, objects_in_circle or completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RISE_MESH_M = .010
RISE_ROOT_M = .015
FOLLOW_WINDOW_S = 1.
FOLLOW_DRIFT_M = .020
MUSTARD_INDEX = 9
PREPARATION_STATE = 'CONTACT_OPEN'


def quat(value):
    value = np.asarray(value, dtype=float)
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object-index', type=int, default=MUSTARD_INDEX)
    args = parser.parse_args()
    run = args.run_directory

    telemetry = np.load(run/'telemetry.npz')
    states, phases = [], []
    with (run/'trace.jsonl').open() as handle:
        for line in handle:
            record = json.loads(line)
            debug = record.get('policy_debug') or {}
            states.append(str(debug.get('state') or record.get('policy_state') or ''))
            phases.append(str(debug.get('phase') or ''))

    total = min(len(states), telemetry['step'].shape[0])
    states, phases = states[:total], phases[:total]
    preparation = [i for i, name in enumerate(states) if name == PREPARATION_STATE]
    if not preparation:
        raise SystemExit('no %s frame recorded; cannot establish the true baseline' % PREPARATION_STATE)
    baseline = preparation[0]

    asset = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/'
                 'objects/task_b/006_mustard_bottle.usd')
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(asset))
    mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    scale = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
    points = points@scale[:3, :3].T+scale[:3, 3]

    objective = args.object_index
    mesh_z = np.empty(total)
    root_z = np.empty(total)
    relative = np.empty((total, 3))
    for index in range(total):
        object_pose = np.eye(4)
        object_pose[:3, :3] = quat(telemetry['object_quat'][index, objective])
        object_pose[:3, 3] = telemetry['object_xyz'][index, objective]
        gripper = np.eye(4)
        gripper[:3, :3] = quat(telemetry['gripper_quat'][index])
        gripper[:3, 3] = telemetry['gripper_xyz'][index]
        world = points@object_pose[:3, :3].T+object_pose[:3, 3]
        mesh_z[index] = world[:, 2].min()
        root_z[index] = object_pose[2, 3]
        relative[index] = (np.linalg.inv(gripper)@object_pose)[:3, 3]

    report = {
        'scope': 'measured offline audit of a carry probe; external evidence only, never a policy input',
        'run': run.name, 'samples': int(total),
        'baseline': {'state': PREPARATION_STATE, 'sample': int(baseline),
                     'step': int(telemetry['step'][baseline]),
                     'note': 'the TRUE preparation frame, taken from the recorded state column'},
        'object_index': objective, 'mesh_vertices': int(len(points)),
        'failure_channels': {
            'terminated_any': bool(telemetry['termination'][:total].any()),
            'illegal_force_max_n': float(np.abs(telemetry['illegal_force'][:total]).max()),
            'env_reward_total': float(np.sum(telemetry['reward_raw_total'][:total])),
            'score_total': float(np.sum(telemetry['score'][:total])),
            'note': 'score and reward are read here for the audit only and drove nothing',
        },
        'segments': {},
    }

    movement_phases = ('PROBE_DRIVE', 'PROBE_BRAKE_DRIVE', 'PROBE_YAW_POS', 'PROBE_BRAKE_YAW_POS',
                       'PROBE_YAW_NEG', 'PROBE_BRAKE_YAW_NEG', 'PROBE_HOLD')
    for phase in movement_phases:
        rows = np.array([i for i, name in enumerate(phases) if name == phase], dtype=int)
        if not len(rows):
            continue
        rise = mesh_z[rows]-mesh_z[baseline]
        root_rise = root_z[rows]-root_z[baseline]
        drift = np.linalg.norm(relative[rows]-relative[rows[0]], axis=1)
        report['segments'][phase] = {
            'samples': int(len(rows)), 'seconds': float(len(rows)*float(telemetry['dt'])),
            'mesh_rise_min_m': float(rise.min()), 'mesh_rise_max_m': float(rise.max()),
            'root_rise_min_m': float(root_rise.min()), 'root_rise_max_m': float(root_rise.max()),
            'relative_drift_max_m': float(drift.max()),
            'any_sample_below_mesh_gate': bool((rise < RISE_MESH_M).any()),
            'any_sample_below_root_gate': bool((root_rise < RISE_ROOT_M).any()),
        }

    # Longest unbroken stretch that would satisfy the carry gate at every sample.
    rows = np.array([i for i, name in enumerate(phases) if name in movement_phases], dtype=int)
    if len(rows):
        good = ((mesh_z[rows]-mesh_z[baseline] >= RISE_MESH_M)
                & (root_z[rows]-root_z[baseline] >= RISE_ROOT_M))
        best = run_length = 0
        for flag in good:
            run_length = run_length+1 if flag else 0
            best = max(best, run_length)
        rise = mesh_z[rows]-mesh_z[baseline]
        report['carry_gate'] = {
            'movement_samples': int(len(rows)), 'movement_seconds': float(len(rows)*float(telemetry['dt'])),
            'longest_unbroken_pass_samples': int(best),
            'longest_unbroken_pass_seconds': float(best*float(telemetry['dt'])),
            'mesh_rise_min_over_movement_m': float(rise.min()),
            'mesh_rise_max_over_movement_m': float(rise.max()),
            'passed_every_movement_sample': bool(good.all()),
            'thresholds': {'mesh_m': RISE_MESH_M, 'root_m': RISE_ROOT_M},
        }

    # Motion, from the PUBLIC twist only (the same signal the policy used).
    try:
        import sys
        sys.path.insert(0, '/home/lybm/ATEC_Robotics_Projects_20260910')
        from task_b.public_odometry import PublicPlanarOdometry
        odometry = PublicPlanarOdometry(dt=float(telemetry['dt']))
        xy, yaw = [], []
        for index in range(total):
            odometry.update(telemetry['proprio'][index])
            xy.append(odometry.position_xy.copy())
            yaw.append(odometry.yaw_unwrapped_rad)
        xy, yaw = np.asarray(xy), np.asarray(yaw)
        if len(rows):
            report['motion'] = {
                'basis': 'public proprio twist integrated once per control tick; no ground truth',
                'xy_at_movement_start': xy[rows[0]].tolist(),
                'xy_at_movement_end': xy[rows[-1]].tolist(),
                'net_displacement_m': float(np.linalg.norm(xy[rows[-1]]-xy[rows[0]])),
                'path_length_m': float(np.linalg.norm(np.diff(xy[rows], axis=0), axis=1).sum()),
                'yaw_at_movement_start_rad': float(yaw[rows[0]]),
                'yaw_at_movement_end_rad': float(yaw[rows[-1]]),
                'net_yaw_change_rad': float(yaw[rows[-1]]-yaw[rows[0]]),
            }
            yaw_rate = np.gradient(yaw[rows], float(telemetry['dt']))
            report['motion']['peak_yaw_rate_rad_s'] = float(np.abs(yaw_rate).max())
    except Exception as error:
        report['motion'] = {'error': repr(error)}

    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in
                      ('run', 'baseline', 'failure_channels', 'carry_gate', 'motion')
                      if key in report}, indent=2))
    print('\nper-movement-phase:')
    for name, entry in report['segments'].items():
        print('  %-20s %5.1fs  mesh_rise %+.4f..%+.4f m  root %+.4f..%+.4f  rel_drift %.4f m  broken=%s'
              % (name, entry['seconds'], entry['mesh_rise_min_m'], entry['mesh_rise_max_m'],
                 entry['root_rise_min_m'], entry['root_rise_max_m'],
                 entry['relative_drift_max_m'], entry['any_sample_below_mesh_gate']))


if __name__ == '__main__':
    main()
