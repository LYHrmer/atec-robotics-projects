"""Docking clearance recheck in the REAL carried pose (static CPU diagnostic).

The earlier docking study assumed a rigid attachment at the p4 closure pose and
the old C goal. This redoes it against the pose that was ACTUALLY achieved and
held for 44.5 s in the p13 run: the recorded object quaternion at the hold
sample gives the true in-hand mesh placement, and the recorded leg joints give
the true chassis envelope. Neither the old 74 mm side margin nor the old 1.45 m
standoff is assumed here - both are recomputed.

Geometry: the barrel is a vertical cylinder, centre C, scored radius 1 m, wall
.02 m, wall height .5 m, so the rim is at z ~ .55 m and the inner wall radius is
.98 m. Docked with the barrel on body +Y, place the barrel axis at body
(0, standoff, 0). A point p in body frame then clears the outer wall iff
hypot(p_x, p_y - standoff) >= 1.0, and is over the mouth iff it is above the rim
AND inside radius .98.

Everything here is static geometry on recorded telemetry. It is a planning
diagnostic, not collision simulation and not evidence of a successful carry.
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
DEFAULT_RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_p13_payload_denyquist_seed42_01')

BARREL_CENTRE = np.array([-3., -10.])
OUTER_RADIUS_M = 1.0
INNER_RADIUS_M = .98
RIM_Z_M = .55
MUSTARD_INDEX = 9


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
    """Every physics joint as a (parent, child, local frames, axis) row."""
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
    """Body-frame pose of every rigid body for one joint configuration."""
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
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif row['kind'] == 'PhysicsPrismaticJoint':
                    motion[:3, 3] = axis * value
            poses[row['child']] = origin @ motion @ np.linalg.inv(row['child_joint'])
            pending.remove(row)
        if len(pending) == previous:
            raise RuntimeError('Unresolved rigid-body tree')
    return poses


def collision_points(stage, row_list, qmap):
    """Collision-geometry point cloud per body, in the body's own frame."""
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
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
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


def bench(points_body, standoff, base_world):
    """Radial clearance of body-frame points against the docked barrel.

    The radial term is body-frame because the docking geometry is defined that
    way, but the rim height is a WORLD quantity: a body-frame z compared against
    .55 m would silently answer a different question whenever the base is not at
    the origin.
    """
    lateral = np.hypot(points_body[:, 0], points_body[:, 1] - standoff)
    world_z = apply(base_world, points_body)[:, 2]
    above = world_z >= RIM_Z_M
    return {
        'min_distance_to_barrel_axis_m': float(lateral.min()),
        'max_body_y_m': float(points_body[:, 1].max()),
        'points_below_rim_inside_outer_wall': int(np.sum((~above) & (lateral < OUTER_RADIUS_M))),
        'points_above_rim_outside_inner_wall': int(np.sum(above & (lateral > INNER_RADIUS_M))),
        'min_radial_margin_to_outer_wall_m': float(lateral.min() - OUTER_RADIUS_M),
        'points_above_rim': int(np.sum(above)),
        'radial_range_of_points_above_rim_m': ([float(lateral[above].min()), float(lateral[above].max())]
                                               if above.any() else None),
        'lowest_world_z_m': float(world_z.min()),
        'highest_world_z_m': float(world_z.max()),
        'lowest_body_z_m': float(points_body[:, 2].min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, nargs='?', default=DEFAULT_RUN)
    parser.add_argument('--phase', default='PAYLOAD_HOLD')
    parser.add_argument('--approach', choices=('side', 'front'), default='side',
                        help='side: barrel on body +Y as the contract plans. '
                             'front: barrel on body +X, which needs no final turn.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    telemetry = np.load(args.run / 'telemetry.npz')
    names = telemetry['joint_names'].tolist()
    phases = []
    with (args.run / 'trace.jsonl').open() as handle:
        for line in handle:
            record = json.loads(line)
            debug = record.get('policy_debug') or {}
            phases.append(str(debug.get('phase') or record.get('policy_state') or ''))
    hits = [i for i, name in enumerate(phases) if name == args.phase]
    if not hits:
        raise SystemExit('phase %r not found in %s' % (args.phase, args.run.name))
    index = hits[-1]
    qmap = dict(zip(names, telemetry['q'][index].astype(float)))

    robot_stage = Usd.Stage.Open(str(ASSETS / 'robot/b2w/b2w_piper.usda'))
    rows = load_tree(robot_stage)
    poses = chain(rows, qmap)
    clouds = collision_points(robot_stage, rows, qmap)

    object_stage = Usd.Stage.Open(str(ASSETS / 'objects/task_b/006_mustard_bottle.usd'))
    mesh = next(UsdGeom.Mesh(p) for p in object_stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    mesh_scale = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
    mesh_points = apply(mesh_scale, mesh_points)

    gripper_world = transform(telemetry['gripper_xyz'][index], quat(telemetry['gripper_quat'][index]))
    object_world = transform(telemetry['object_xyz'][index, MUSTARD_INDEX],
                             quat(telemetry['object_quat'][index, MUSTARD_INDEX]))
    in_hand = np.linalg.inv(gripper_world) @ object_world
    mesh_gripper = apply(in_hand, mesh_points)

    body_clouds, arm_clouds = {}, {}
    for name, points in clouds.items():
        target = arm_clouds if (name.startswith('arm_') or name == 'gripper_base') else body_clouds
        target[name] = apply(poses[name], points)
    chassis = np.concatenate(list(body_clouds.values()))
    arm = np.concatenate(list(arm_clouds.values()))
    gripper_pose = poses['gripper_base']
    payload = apply(gripper_pose, mesh_gripper)

    if args.approach == 'front':
        # Body +X becomes body +Y, so the same radial math answers the frontal
        # question. The barrel then sits ahead of the robot, which needs no
        # final heading change - decisive because measured yaw authority is ~0.
        spin = Rotation.from_rotvec([0., 0., np.pi/2]).as_matrix()
        chassis = chassis @ spin.T
        arm = arm @ spin.T
        payload = payload @ spin.T
        gripper_pose = gripper_pose.copy()
        gripper_pose[:3, :3] = gripper_pose[:3, :3] @ spin

    report = {
        'scope': 'static CPU geometry on recorded telemetry; not collision simulation, '
                 'not a carry, clearance or delivery result',
        'run': args.run.name, 'phase': args.phase, 'sample': int(index),
        'approach': args.approach,
        'barrel': {'centre': BARREL_CENTRE.tolist(), 'outer_radius_m': OUTER_RADIUS_M,
                   'inner_radius_m': INNER_RADIUS_M, 'rim_z_m': RIM_Z_M},
        'attachment': 'mesh placed from the RECORDED object quaternion at this sample, so this is the '
                      'in-hand pose actually achieved, not a rigid-attachment assumption',
        'mesh_vertices': int(len(mesh_points)),
        'payload_body_aabb_min': payload.min(axis=0).tolist(),
        'payload_body_aabb_max': payload.max(axis=0).tolist(),
        'gripper_base_body_m': gripper_pose[:3, 3].tolist(),
        'chassis_max_body_y_m': float(chassis[:, 1].max()),
        'chassis_min_body_y_m': float(chassis[:, 1].min()),
        'minimum_safe_standoff_m': float(1.0 + chassis[:, 1].max() + .03),
        'standoffs': {},
    }
    base_world = transform(telemetry['base_xyz'][index], quat(telemetry['base_quat'][index]))
    for standoff in (1.35, 1.40, 1.45, 1.50, 1.55):
        payload_bench = bench(payload, standoff, base_world)
        entry = {'chassis': bench(chassis, standoff, base_world),
                 'payload': payload_bench,
                 'arm': bench(arm, standoff, base_world)}
        entry['chassis_wall_gap_m'] = float(standoff - OUTER_RADIUS_M - chassis[:, 1].max())
        entry['payload_entirely_above_rim'] = bool(payload_bench['lowest_world_z_m'] >= RIM_Z_M)
        entry['payload_clear_of_wall_below_rim'] = bool(
            payload_bench['points_below_rim_inside_outer_wall'] == 0)
        entry['payload_reaches_over_mouth'] = bool(
            payload_bench['points_above_rim'] > 0
            and payload_bench['radial_range_of_points_above_rim_m'][0] < INNER_RADIUS_M)
        report['standoffs'][str(standoff)] = entry
    report['recommended_standoff_m'] = next(
        (float(s) for s in (1.35, 1.40, 1.45, 1.50, 1.55)
         if report['standoffs'][str(s)]['chassis_wall_gap_m'] > 0.
         and report['standoffs'][str(s)]['payload_clear_of_wall_below_rim']), None)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'run', 'phase', 'sample', 'chassis_max_body_y_m', 'minimum_safe_standoff_m',
        'payload_body_aabb_min', 'payload_body_aabb_max', 'gripper_base_body_m',
        'recommended_standoff_m')}, indent=2))
    print('\nstandoff | chassis gap | below-rim wall hits | payload lowest world z | '
          'above-rim radial min | over mouth')
    for standoff, entry in report['standoffs'].items():
        radial = entry['payload']['radial_range_of_points_above_rim_m']
        print('%8s | %+11.4f | %19d | %21.4f | %19s | %s' % (
            standoff, entry['chassis_wall_gap_m'],
            entry['payload']['points_below_rim_inside_outer_wall'],
            entry['payload']['lowest_world_z_m'],
            'n/a' if radial is None else '%.4f' % radial[0],
            entry['payload_reaches_over_mouth']))


if __name__ == '__main__':
    main()
