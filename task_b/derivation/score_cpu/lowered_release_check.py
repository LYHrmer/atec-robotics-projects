"""Can the release be lowered enough to contain a restitution-1 bounce?

The bounce arithmetic is unforgiving: the barrel rim is at world z = .55 m and the
floor top at .05 m, and the original object material has restitution 1, so a
release whose bottle bottom sits above the rim can return to its release height
and leave. The recorded A-pose release in plan_d4 put the bottle bottom at
.6948 m, i.e. .145 m above the rim.

This solves IK for the SAME gripper orientation and the SAME forward reach, with
the gripper body z lowered in steps, and reports for each candidate:

* the bottle's lowest world z, using the RECORDED gripper->object transform (an
  assumption carried forward, not a new measurement);
* whether the bottle is still over the barrel mouth at the frontal standoff;
* whether any finger or arm link dips below the rim and inside the outer wall;
* whether the solution stays inside the original joint limits.

Static CPU geometry on recorded data. Not collision simulation, not a carry or
delivery result.
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

from task_b.arm_kinematics import fk, solve_ik          # noqa: E402
from task_e_geometry import JOINT_LOWER, JOINT_UPPER    # noqa: E402

ASSETS = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model')
RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_d4_first_delivery_seed42_01')
OUTER_RADIUS_M = 1.0
INNER_INSCRIBED_M = .98*np.cos(np.pi/32.)
RIM_Z_M = .55
BARREL_FLOOR_TOP_Z = .05
STANDOFF_M = 1.545
ARM_CHAIN = tuple('arm_link'+str(i) for i in range(1, 9))


def quat(v):
    if hasattr(v, 'GetReal'):
        return Rotation.from_quat([*v.GetImaginary(), v.GetReal()]).as_matrix()
    v = np.asarray(v, dtype=float)
    return Rotation.from_quat(v[[1, 2, 3, 0]]).as_matrix()


def xform(p, r):
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = p; return out


def apply(pose, pts):
    return np.asarray(pts)@pose[:3, :3].T+pose[:3, 3]


def load_tree(stage):
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if len(b0) != 1 or len(b1) != 1:
            continue
        ax = prim.GetAttribute('physics:axis')
        rows.append(dict(name=prim.GetName(), parent=b0[0].name, child=b1[0].name,
                         kind=prim.GetTypeName(), axis=ax.Get() if ax else None,
                         pj=xform(j.GetLocalPos0Attr().Get(), quat(j.GetLocalRot0Attr().Get())),
                         cj=xform(j.GetLocalPos1Attr().Get(), quat(j.GetLocalRot1Attr().Get()))))
    return rows


def chain(rows, qmap):
    poses = {'base_link': np.eye(4)}
    pending = list(rows)
    while pending:
        prev = len(pending)
        for r in list(pending):
            if r['parent'] not in poses:
                continue
            origin = poses[r['parent']]@r['pj']
            motion = np.eye(4)
            if r['axis']:
                axis = np.eye(3)['XYZ'.index(r['axis'])]
                val = qmap.get(r['name'], 0.)
                if r['kind'] == 'PhysicsRevoluteJoint':
                    motion[:3, :3] = Rotation.from_rotvec(axis*val).as_matrix()
                elif r['kind'] == 'PhysicsPrismaticJoint':
                    motion[:3, 3] = axis*val
            poses[r['child']] = origin@motion@np.linalg.inv(r['cj'])
            pending.remove(r)
        if len(pending) == prev:
            raise RuntimeError('unresolved tree')
    return poses


def clouds(stage):
    cache = UsdGeom.XformCache(); out = {}
    for coll in stage.Traverse():
        if not coll.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = coll.GetParent(); pieces = []
        for prim in Usd.PrimRange(coll, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if prim.IsA(UsdGeom.Mesh):
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            elif prim.IsA(UsdGeom.Cube):
                h = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
                pts = np.array(list(itertools.product((-h, h), repeat=3)))
            else:
                e = np.asarray(UsdGeom.Boundable(prim).GetExtentAttr().Get(), dtype=float)
                pts = np.array(list(itertools.product(*zip(e[0], e[1]))))
            rel = np.asarray(cache.ComputeRelativeTransform(prim, body)[0]).T
            pieces.append(apply(rel, pts))
        if pieces:
            out[body.GetName()] = np.concatenate(pieces)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    tel = np.load(RUN/'telemetry.npz')
    names = tel['joint_names'].tolist()
    handoff = None
    with (RUN/'trace.jsonl').open() as fh:
        for i, line in enumerate(fh):
            if (json.loads(line).get('policy_debug') or {}).get('phase') == 'CARRY_READY':
                handoff = i; break
    qmap = dict(zip(names, tel['q'][handoff].astype(float)))
    q_arm = np.array([qmap['arm_joint'+str(i)] for i in range(1, 7)])
    base_world = xform(tel['base_xyz'][handoff], quat(tel['base_quat'][handoff]))
    gripper_world = xform(tel['gripper_xyz'][handoff], quat(tel['gripper_quat'][handoff]))
    object_world = xform(tel['object_xyz'][handoff, 9], quat(tel['object_quat'][handoff, 9]))

    asset = ASSETS/'objects/task_b/006_mustard_bottle.usd'
    stage_o = Usd.Stage.Open(str(asset))
    mesh = next(UsdGeom.Mesh(p) for p in stage_o.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    scale = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
    points = points@scale[:3, :3].T+scale[:3, 3]
    in_hand = np.linalg.inv(gripper_world)@object_world
    mesh_gripper = apply(in_hand, points)

    robot = Usd.Stage.Open(str(ASSETS/'robot/b2w/b2w_piper.usda'))
    rows = load_tree(robot); body_clouds = clouds(robot)

    pose0 = fk(q_arm)
    start_xy = pose0[:2, 3]
    rotation0 = pose0[:3, :3]
    report = {'scope': 'static CPU geometry of a lowered release pose; the recorded '
                       'gripper->object transform is carried forward as an ASSUMPTION',
              'run': RUN.name, 'handoff_sample': int(handoff),
              'current_gripper_body_z_m': float(pose0[2, 3]),
              'current_bottle_bottom_world_z_m': float(apply(base_world, apply(
                  np.linalg.inv(base_world)@gripper_world, mesh_gripper))[:, 2].min()),
              'rim_z_m': RIM_Z_M, 'inner_inscribed_m': float(INNER_INSCRIBED_M),
              'candidates': []}

    for delta in (0., .05, .10, .15, .20, .25, .30):
        target = np.array([start_xy[0], start_xy[1], pose0[2, 3]-delta])
        fit = solve_ik(target, rotation=rotation0, seed=q_arm, max_nfev=120)
        entry = {'lowering_m': delta, 'target_body_z_m': float(target[2]),
                 'ik_success': bool(fit.success),
                 'ik_position_error_m': float(fit.position_error),
                 'ik_orientation_error_rad': float(fit.orientation_error)}
        if fit.success:
            joints = np.asarray(fit.joints, dtype=float)
            entry['outside_joint_limits'] = [int(i+1) for i in range(6)
                                             if not (JOINT_LOWER[i]-1e-6 <= joints[i]
                                                     <= JOINT_UPPER[i]+1e-6)]
            q = dict(qmap)
            for i in range(6):
                q['arm_joint'+str(i+1)] = float(joints[i])
            gpose = fk(joints)
            pose = chain(rows, q)
            # Rotate in the BODY frame, then take world z from the world frame:
            # applying base_world first and rotating after mixes frames and makes
            # the radial test meaningless.
            spin = Rotation.from_rotvec([0., 0., np.pi/2]).as_matrix()
            bottle_body = apply(gpose, mesh_gripper)@spin.T
            bottle_world = apply(base_world, bottle_body)
            radial = np.hypot(bottle_body[:, 0], bottle_body[:, 1]-STANDOFF_M)
            above = bottle_world[:, 2] >= RIM_Z_M
            # The wall is an ANNULUS between the inscribed inner radius and the
            # outer radius, from the floor top up to the rim. A point below the
            # rim but INSIDE the inner radius is the barrel's empty interior -
            # exactly where the bottle belongs - and is not a collision. Testing
            # "below the rim and within the outer radius" instead flagged the
            # interior and made every lowering look like a wall strike.
            def wall_strikes(points_body, points_world):
                radial = np.hypot(points_body[:, 0], points_body[:, 1]-STANDOFF_M)
                z = points_world[:, 2]
                in_annulus = ((z >= BARREL_FLOOR_TOP_Z) & (z <= RIM_Z_M)
                              & (radial >= INNER_INSCRIBED_M) & (radial <= OUTER_RADIUS_M))
                below_floor = (z < BARREL_FLOOR_TOP_Z) & (radial <= OUTER_RADIUS_M)
                return int(np.sum(in_annulus | below_floor))

            entry['bottle_lowest_world_z_m'] = float(bottle_world[:, 2].min())
            entry['rim_clearance_m'] = float(bottle_world[:, 2].min()-RIM_Z_M)
            entry['bottle_wall_strikes'] = wall_strikes(bottle_body, bottle_world)
            entry['bottle_below_rim_inside_outer_wall'] = entry['bottle_wall_strikes']
            entry['bottle_above_rim_radial_min_m'] = (float(radial[above].min())
                                                      if above.any() else None)
            entry['bottle_over_mouth'] = bool(above.any()
                                              and radial[above].min() < INNER_INSCRIBED_M)
            arm_body = np.concatenate([apply(pose[n], body_clouds[n]) for n in ARM_CHAIN])@spin.T
            arm_world = apply(base_world, arm_body)
            entry['arm_wall_strikes'] = wall_strikes(arm_body, arm_world)
            entry['arm_below_rim_inside_outer_wall'] = entry['arm_wall_strikes']
            entry['ideal_bounce_apex_m'] = entry['bottle_lowest_world_z_m']
            entry['bounce_can_leave_barrel'] = bool(entry['bottle_lowest_world_z_m'] > RIM_Z_M)
            entry['joints_rad'] = joints.tolist()
        report['candidates'].append(entry)

    ok = [e for e in report['candidates']
          if e.get('ik_success') and not e.get('outside_joint_limits')
          and e.get('bottle_over_mouth') and not e.get('bounce_can_leave_barrel')
          and not e.get('bottle_below_rim_inside_outer_wall')
          and not e.get('arm_below_rim_inside_outer_wall')]
    report['deepest_safe_lowering_m'] = (max(e['lowering_m'] for e in ok) if ok else None)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('current gripper body z %.4f  bottle bottom %.4f m  (rim %.2f)'
          % (report['current_gripper_body_z_m'], report['current_bottle_bottom_world_z_m'], RIM_Z_M))
    print()
    print('%-8s %-9s %-11s %-11s %-9s %-11s %-9s %s' % (
        'lower_m', 'body_z', 'bottle_low_z', 'rim_clear', 'over_mouth', 'bounce_out',
        'wall_hits', 'verdict'))
    for e in report['candidates']:
        if not e.get('ik_success'):
            print('%-8.2f %-9.3f  IK FAILED  pos_err=%.4f' % (e['lowering_m'], e['target_body_z_m'],
                                                             e['ik_position_error_m'])); continue
        bad = (e['outside_joint_limits'] or e['bottle_below_rim_inside_outer_wall']
               or e['arm_below_rim_inside_outer_wall'] or not e['bottle_over_mouth'])
        print('%-8.2f %-9.3f %-11.4f %-11.4f %-9s %-11s %-9d %s' % (
            e['lowering_m'], e['target_body_z_m'], e['bottle_lowest_world_z_m'],
            e['rim_clearance_m'], e['bottle_over_mouth'], e['bounce_can_leave_barrel'],
            e['bottle_below_rim_inside_outer_wall']+e['arm_below_rim_inside_outer_wall'],
            'REJECT' if bad else ('SAFE' if not e['bounce_can_leave_barrel'] else 'bounce risk')))
    print()
    print('deepest safe lowering:', report['deepest_safe_lowering_m'])


if __name__ == '__main__':
    main()
