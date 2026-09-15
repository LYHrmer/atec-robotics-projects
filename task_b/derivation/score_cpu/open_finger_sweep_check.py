"""Open-finger sweep geometry at the D low-place candidates (static CPU diagnostic).

`next_delivery_geometry_2006.md` computed the C->D descent with the fingers in
their recorded grip, and says plainly that the open-finger sweep is NOT covered
("必须补做从实际夹宽到70mm的手指完整开合扫掠"). This fills exactly that gap: it
sweeps the two prismatic finger joints across their full travel and reports where
the finger geometry goes relative to the barrel rim and wall.

Two facts drive the numbers, both read from the original robot USD and the p13
telemetry rather than assumed:

* the finger joints are PRISMATIC with hard limits ``arm_joint7 in [0, .035]``
  and ``arm_joint8 in [-.035, 0]``, so the total jaw gap spans 0 to .07 m and no
  command can place either finger outside that travel;
* p13 ended at a measured gap of .0575 m, so a release to full open is about
  12.5 mm of real finger travel, not the .11 m of command-space travel that the
  written contract assumed from the preload command.

Geometry convention: docked with the barrel on body +Y at the standoff, the
barrel axis sits at body (0, standoff, 0). A point clears the outer wall iff its
body-frame radial distance to that axis is >= 1.0, and the rim is at WORLD
z = .55, so the vertical comparison uses the recorded base pose.

Static geometry on recorded telemetry. Not collision simulation, not a carry,
clearance or delivery result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

REPO = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(REPO))

ASSETS = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model')
GEOMETRY = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score_cpu/next_delivery_geometry_2006.json')

FINGER_BODIES = ('arm_link7', 'arm_link8')
GRIPPER_BODY = 'gripper_base'
ARM_CHAIN = tuple('arm_link' + str(index) for index in range(1, 9))
OUTER_RADIUS_M = 1.0
INNER_INSCRIBED_RADIUS_M = .98*np.cos(np.pi/32.)     # 32-segment wall, conservative
RIM_Z_M = .55
STANDOFF_M = 1.45
JOINT7_LIMITS = (0., .035)
JOINT8_LIMITS = (-.035, 0.)


def quat(value):
    if hasattr(value, 'GetReal'):
        return Rotation.from_quat([*value.GetImaginary(), value.GetReal()]).as_matrix()
    value = np.asarray(value)
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def transform(position, rotation):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, position
    return out


def apply(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def load_tree(stage):
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            continue
        axis_attr = prim.GetAttribute('physics:axis')
        rows.append(dict(
            name=prim.GetName(), parent=parents[0].name, child=children[0].name,
            kind=prim.GetTypeName(), axis=axis_attr.Get() if axis_attr else None,
            parent_joint=transform(joint.GetLocalPos0Attr().Get(), quat(joint.GetLocalRot0Attr().Get())),
            child_joint=transform(joint.GetLocalPos1Attr().Get(), quat(joint.GetLocalRot1Attr().Get()))))
    return rows


def chain(row_list, qmap):
    poses = {'base_link': np.eye(4)}
    pending = list(row_list)
    while pending:
        previous = len(pending)
        for row in list(pending):
            if row['parent'] not in poses:
                continue
            origin = poses[row['parent']] @ row['parent_joint']
            motion = np.eye(4)
            if row['axis']:
                axis = np.eye(3)['XYZ'.index(row['axis'])]
                value = qmap.get(row['name'], 0.)
                if row['kind'] == 'PhysicsRevoluteJoint':
                    motion[:3, :3] = Rotation.from_rotvec(axis*value).as_matrix()
                elif row['kind'] == 'PhysicsPrismaticJoint':
                    motion[:3, 3] = axis*value
            poses[row['child']] = origin @ motion @ np.linalg.inv(row['child_joint'])
            pending.remove(row)
        if len(pending) == previous:
            raise RuntimeError('Unresolved rigid-body tree')
    return poses


def collision_clouds(stage):
    cache = UsdGeom.XformCache()
    collected = {}
    for collision in stage.Traverse():
        if not collision.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = collision.GetParent()
        pieces = []
        for prim in Usd.PrimRange(collision, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
                points = np.array(list(itertools.product((-half, half), repeat=3)))
            else:
                extent = np.asarray(UsdGeom.Boundable(prim).GetExtentAttr().Get(), dtype=float)
                if extent.shape != (2, 3) or not np.isfinite(extent).all():
                    raise RuntimeError('Missing explicit shape extent: ' + str(prim.GetPath()))
                points = np.array(list(itertools.product(*zip(extent[0], extent[1]))))
            relative = np.asarray(cache.ComputeRelativeTransform(prim, body)[0]).T
            pieces.append(apply(relative, points))
        if pieces:
            collected[body.GetName()] = np.concatenate(pieces)
    return collected


def sweep_case(rows, clouds, base_world, joints, command7, command8):
    qmap = {'arm_joint'+str(index+1): float(joints[index]) for index in range(6)}
    qmap['arm_joint7'], qmap['arm_joint8'] = float(command7), float(command8)
    poses = chain(rows, qmap)
    finger = np.concatenate([apply(poses[name], clouds[name]) for name in FINGER_BODIES])
    arm = np.concatenate([apply(poses[name], clouds[name]) for name in ARM_CHAIN])
    world_z = apply(base_world, finger)[:, 2]
    radial = np.hypot(finger[:, 0], finger[:, 1]-STANDOFF_M)
    arm_radial = np.hypot(arm[:, 0], arm[:, 1]-STANDOFF_M)
    arm_world_z = apply(base_world, arm)[:, 2]
    return {
        'command_joint7_rad': float(command7), 'command_joint8_rad': float(command8),
        'jaw_command_gap_m': float(command7-command8),
        'finger_lowest_body_z_m': float(finger[:, 2].min()),
        'finger_lowest_world_z_m': float(world_z.min()),
        'finger_clear_of_rim_by_m': float(world_z.min()-RIM_Z_M),
        'finger_min_radial_to_barrel_axis_m': float(radial.min()),
        'finger_wall_margin_vs_inscribed_m': float(radial.min()-INNER_INSCRIBED_RADIUS_M),
        'finger_points_below_rim_inside_wall': int(np.sum((world_z < RIM_Z_M)
                                                          & (radial < OUTER_RADIUS_M))),
        'arm_lowest_world_z_m': float(arm_world_z.min()),
        'arm_min_radial_to_barrel_axis_m': float(arm_radial.min()),
        'arm_points_below_rim_inside_wall': int(np.sum((arm_world_z < RIM_Z_M)
                                                       & (arm_radial < OUTER_RADIUS_M))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=9)
    args = parser.parse_args()

    geometry = json.loads(GEOMETRY.read_text())
    base_world = transform(np.asarray(geometry['base_xyz_actual'], dtype=float),
                           quat(np.asarray(geometry['base_quat_wxyz_actual'], dtype=float)))
    cases = geometry['lower_inside_bin_D_candidates']['cases']

    robot_stage = Usd.Stage.Open(str(ASSETS/'robot/b2w/b2w_piper.usda'))
    rows = load_tree(robot_stage)
    clouds = collision_clouds(robot_stage)

    # The recorded p13 ending gap, read from its own telemetry rather than assumed.
    telemetry = np.load('/home/lybm/ATEC_Experiments_20260910/task_b_score/'
                        'plan_p13_payload_denyquist_seed42_01/telemetry.npz')
    names = telemetry['joint_names'].tolist()
    recorded_gap = float(telemetry['q'][-1, names.index('arm_joint7')]
                         - telemetry['q'][-1, names.index('arm_joint8')])

    report = {
        'scope': 'static CPU geometry of the OPEN-finger sweep at the D candidates; not collision '
                 'simulation, not a carry, clearance or delivery result',
        'source': str(GEOMETRY), 'base_pose_is_the_recorded_p13_pose': True,
        'finger_joints_are_prismatic_with_hard_limits': {
            'arm_joint7_limits_rad': list(JOINT7_LIMITS), 'arm_joint8_limits_rad': list(JOINT8_LIMITS),
            'total_jaw_gap_range_m': [0., float(JOINT7_LIMITS[1]-JOINT8_LIMITS[0])]},
        'recorded_p13_ending_gap_m': recorded_gap,
        'release_travel_from_recorded_gap_to_full_open_m': float(.07-recorded_gap),
        'barrel': {'standoff_m': STANDOFF_M, 'outer_radius_m': OUTER_RADIUS_M,
                   'inner_inscribed_radius_m': float(INNER_INSCRIBED_RADIUS_M), 'rim_z_m': RIM_Z_M},
        'cases': [],
    }
    for index, case in enumerate(cases):
        joints = np.asarray(case['q_rad'], dtype=float)
        command7 = np.linspace(JOINT7_LIMITS[0], JOINT7_LIMITS[1], args.steps)
        command8 = np.linspace(JOINT8_LIMITS[0], JOINT8_LIMITS[1], args.steps)
        sweep = [sweep_case(rows, clouds, base_world, joints, a, b)
                 for a, b in zip(command7, command8)]
        report['cases'].append({
            'index': index,
            'target_gripper_body_xyz_m': case['target_gripper_body_xyz_m'],
            'joints_rad': joints.tolist(),
            'closed_reference': case['endpoint'],
            'open_sweep': sweep,
            'worst_finger_rim_clearance_m': min(entry['finger_clear_of_rim_by_m'] for entry in sweep),
            'worst_finger_wall_margin_m': min(entry['finger_wall_margin_vs_inscribed_m']
                                              for entry in sweep),
            'any_finger_below_rim_inside_wall': any(
                entry['finger_points_below_rim_inside_wall'] for entry in sweep),
            'any_arm_below_rim_inside_wall': any(
                entry['arm_points_below_rim_inside_wall'] for entry in sweep),
        })

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('recorded p13 ending gap: %.5f m   travel to full open: %.5f m'
          % (recorded_gap, .07-recorded_gap))
    print('inscribed inner radius (32-segment wall): %.6f m' % INNER_INSCRIBED_RADIUS_M)
    print()
    print('%-6s %-22s %-12s %-14s %-12s %-10s %s' % ('case', 'gripper body xyz', 'worst rim',
                                                     'worst wall margin', 'finger<rim', 'arm<rim',
                                                     'verdict'))
    for entry in report['cases']:
        print('%-6d %-22s %+12.4f %+14.4f %-12s %-10s %s' % (
            entry['index'], np.round(entry['target_gripper_body_xyz_m'], 3),
            entry['worst_finger_rim_clearance_m'], entry['worst_finger_wall_margin_m'],
            entry['any_finger_below_rim_inside_wall'], entry['any_arm_below_rim_inside_wall'],
            'OK' if (entry['worst_finger_rim_clearance_m'] > 0
                     and not entry['any_finger_below_rim_inside_wall']
                     and not entry['any_arm_below_rim_inside_wall']) else 'REJECT'))


if __name__ == '__main__':
    main()
