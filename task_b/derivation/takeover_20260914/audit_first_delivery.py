"""Independent delivery audit: did the SAME object actually leave the gripper and stay in the barrel?

This is the acceptance the policy deliberately cannot make for itself. The policy
never reads the score, so `delivery_release_sequence_complete` is a SEQUENCE
claim; these four questions are what turn it into a delivery claim, and each is
answered from recorded telemetry and the official scoring events only:

1. did the carried object actually LEAVE the gripper (relative pose diverging and
   the object falling away), rather than merely being commanded open;
2. did the official `objects_in_circle` term fire, and at which step;
3. is the object inside the SOLID barrel - the inscribed inner wall and between
   the barrel floor top and the rim - not merely inside the 1 m scoring radius;
4. did it STAY there, for how long.

The carried object is identified from the data (the one closest to the gripper at
the carry handoff), never from an assumed index.

Nothing here is a policy input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BARREL_CENTRE_XY = np.array([-3., -10.])
#: the wall is a 32-segment annulus: .98 is the inner VERTEX radius, and the
#: conservative circular bound inside it is the inscribed radius
INNER_WALL_INSCRIBED_M = .98*np.cos(np.pi/32.)
BARREL_FLOOR_TOP_Z = .05
RIM_Z = .55
SCORING_RADIUS_M = 1.0


def quat(value):
    value = np.asarray(value, dtype=float)
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gripper-separation_m', type=float, default=.30,
                        help='relative displacement above which the object counts as no longer carried')
    args = parser.parse_args()
    run = args.run_directory

    telemetry = np.load(run/'telemetry.npz')
    result = json.loads((run/'result.json').read_text())
    phases = []
    with (run/'trace.jsonl').open() as handle:
        for line in handle:
            debug = (json.loads(line).get('policy_debug') or {})
            phases.append(str(debug.get('phase') or ''))
    total = min(len(phases), telemetry['step'].shape[0])
    phases = phases[:total]

    handoff = next((i for i, name in enumerate(phases) if name == 'CARRY_READY'), None)
    if handoff is None:
        raise SystemExit('no CARRY_READY frame: this run never took the carry handoff')

    gripper_xyz = telemetry['gripper_xyz'][:total]

    def gripper_pose(index):
        pose = np.eye(4)
        pose[:3, :3] = quat(telemetry['gripper_quat'][index])
        pose[:3, 3] = gripper_xyz[index]
        return pose

    def object_pose(index, obj):
        pose = np.eye(4)
        pose[:3, :3] = quat(telemetry['object_quat'][index, obj])
        pose[:3, 3] = telemetry['object_xyz'][index, obj]
        return pose

    # Identify the carried object from ONE sample, then track only that object:
    # converting all 18 objects on every sample is needlessly slow.
    handoff_gripper = gripper_pose(handoff)
    carried = int(np.argmin([
        np.linalg.norm((np.linalg.inv(handoff_gripper)@object_pose(handoff, obj))[:3, 3])
        for obj in range(18)]))

    relative = np.empty((total, 3))
    for index in range(total):
        relative[index] = (np.linalg.inv(gripper_pose(index))
                           @ object_pose(index, carried))[:3, 3]

    distance_to_axis = np.linalg.norm(
        telemetry['object_xyz'][:total, carried, :2]-BARREL_CENTRE_XY, axis=1)
    object_z = telemetry['object_xyz'][:total, carried, 2]
    inside_barrel = ((distance_to_axis <= INNER_WALL_INSCRIBED_M)
                     & (object_z >= BARREL_FLOOR_TOP_Z) & (object_z <= RIM_Z))
    inside_scoring = ((distance_to_axis <= SCORING_RADIUS_M)
                      & (object_z >= 0.) & (object_z <= .5))

    held = np.linalg.norm(relative[:total], axis=1)
    separation = held-held[handoff]
    # Search only AT OR AFTER the handoff: before it the object is not carried, so
    # its distance to the gripper is large for reasons that have nothing to do
    # with release, and an earlier version wrongly reported a release at sample 0.
    candidates = np.flatnonzero(separation[handoff:] > args.gripper_separation_m)
    release_sample = int(candidates[0]+handoff) if len(candidates) else None

    # Once it leaves, the object must fall: check it does not keep rising with the gripper.
    post = slice(release_sample, total) if release_sample is not None else slice(total, total)
    release_phases = [name for name in phases if name.startswith('RELEASE')]
    events = result.get('scoring_events') or []

    longest = best = 0
    for flag in inside_barrel:
        best = best+1 if flag else 0
        longest = max(longest, best)
    longest_after = best = 0
    start = release_sample if release_sample is not None else total
    for flag in inside_barrel[start:]:
        best = best+1 if flag else 0
        longest_after = max(longest_after, best)

    report = {
        'scope': 'independent measured audit of one delivery attempt; telemetry and official '
                 'scoring events only, never a policy input',
        'run': run.name, 'samples': int(total), 'carry_handoff_sample': int(handoff),
        'carried_object_index': carried,
        'carried_object_identified_by': 'closest to the gripper at the carry handoff, from the data',
        'phases_reached': sorted(set(phases)),
        'reached_a_release_phase': bool(release_phases),
        'answer_1_object_left_the_gripper': {
            'released': release_sample is not None,
            'release_sample': release_sample,
            'separation_threshold_m': args.gripper_separation_m,
            'relative_displacement_at_end_m': float(held[total-1]),
            'relative_displacement_at_handoff_m': float(held[handoff]),
            'highest_object_z_after_release_m': (float(object_z[post].max())
                                                 if release_sample is not None else None),
        },
        'answer_2_official_circle_event': {
            'events': len(events), 'first_event_step': (events[0].get('step')
                                                        if events and isinstance(events[0], dict)
                                                        else None),
            'reward_term_totals_raw': result.get('reward_term_totals_raw'),
            'score_raw_total': result.get('score_raw_total'),
        },
        'answer_3_inside_the_solid_barrel': {
            'samples_inside_solid_barrel': int(inside_barrel.sum()),
            'samples_inside_scoring_radius_only': int((inside_scoring & ~inside_barrel).sum()),
            'final_distance_to_axis_m': float(distance_to_axis[total-1]),
            'final_object_z_m': float(object_z[total-1]),
            'inner_wall_inscribed_m': float(INNER_WALL_INSCRIBED_M),
            'floor_top_z': BARREL_FLOOR_TOP_Z, 'rim_z': RIM_Z,
            'note': 'the solid barrel uses the inscribed inner wall, not the 1 m scoring radius',
        },
        'answer_4_did_it_stay': {
            'longest_continuous_inside_s_barrel': float(longest*float(telemetry['dt'])),
            'longest_continuous_after_release_s': float(longest_after*float(telemetry['dt'])),
            'still_inside_at_end': bool(inside_barrel[total-1]),
        },
        'original_failure': {
            'terminated': bool(result.get('terminated')),
            'truncated': bool(result.get('truncated')),
            'illegal_force_max_n': float(np.abs(telemetry['illegal_force'][:total]).max()),
            'task_physics_modified': result.get('task_physics_modified'),
        },
        'verdict': None,
    }
    a = report['answer_1_object_left_the_gripper']['released']
    b = report['answer_2_official_circle_event']['events'] > 0
    c = report['answer_4_did_it_stay']['still_inside_at_end']
    d = not report['original_failure']['terminated'] and not report['original_failure']['truncated']
    report['verdict'] = ('DELIVERY_EVIDENCE_COMPLETE' if (a and b and c and d) else
                         'SEQUENCE_ONLY_NOT_A_DELIVERY')

    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in
                      ('run', 'carried_object_index', 'reached_a_release_phase', 'verdict',
                       'answer_1_object_left_the_gripper', 'answer_2_official_circle_event',
                       'answer_4_did_it_stay')}, indent=2))


if __name__ == '__main__':
    main()
