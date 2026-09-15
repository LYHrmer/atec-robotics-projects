"""Open-finger release geometry for the FRONTAL dock (static CPU diagnostic).

Why this exists. plan_d2/d3 measured the chassis yaw authority at ~0.0016 rad/s in
both the in-place and the rolling-arc form. The contract's side dock needs
2.064 rad of heading change, which is ~1290 s at that rate and therefore exceeds
the 1200 s episode limit on its own. A FRONTAL dock needs only 0.493 rad, which
fits. But the sideways EXTEND pose the side dock relies on cannot reach a barrel
that is dead ahead, so a frontal dock has to release from the RAISE pose, and that
release geometry has never been checked.

The arm pose used here is the REAL recorded raise pose (the arm q at the carry
handoff of plan_d3), not a re-solved IK. The finger sweep covers the whole
prismatic travel, and the barrel is placed on body +X at the standoff.

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
RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_d3_carry_probe_seed42_01')

ARM_CHAIN = tuple('arm_link'+str(i) for i in range(1, 9))
GRIPPER = 'gripper_base'
OUTER_RADIUS_M = 1.0
INNER_INSCRIBED_M = .98*np.cos(np.pi/32.)
RIM_Z_M = .55
JOINT7 = (0., .035)
JOINT8 = (-.035, 0.)


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
    ap.add_argument('--steps', type=int, default=9)
    ap.add_argument('--swing_rad', type=float, default=0.0,
                    help='place the barrel this far off the body +X axis, so the ARM '
                         'swings instead of the chassis having to turn')
    args = ap.parse_args()

    tel = np.load(RUN/'telemetry.npz')
    names = tel['joint_names'].tolist()
    # the REAL raise pose: the arm q at the carry handoff
    handoff = None
    with (RUN/'trace.jsonl').open() as fh:
        for i, line in enumerate(fh):
            d = (json.loads(line).get('policy_debug') or {})
            if d.get('phase') == 'CARRY_READY':
                handoff = i; break
    qmap = dict(zip(names, tel['q'][handoff].astype(float)))
    base_world = xform(tel['base_xyz'][handoff], quat(tel['base_quat'][handoff]))

    stage = Usd.Stage.Open(str(ASSETS/'robot/b2w/b2w_piper.usda'))
    rows = load_tree(stage); body_clouds = clouds(stage)

    report = {'scope': 'static CPU geometry of the FRONTAL-dock release from the recorded RAISE '
                       'pose; not collision simulation, not a carry or delivery result',
              'run': RUN.name, 'handoff_sample': int(handoff),
              'arm_q_rad_at_handoff': [qmap['arm_joint'+str(i)] for i in range(1, 7)],
              'barrel_centre_body_axis': '+X (frontal dock)',
              'inner_inscribed_radius_m': float(INNER_INSCRIBED_M), 'rim_z_m': RIM_Z_M,
              'standoffs': {}}
    report['swing_rad'] = args.swing_rad
    for standoff in (1.45, 1.50, 1.52, 1.55, 1.60, 1.65):
        entries = []
        for c7, c8 in zip(np.linspace(*JOINT7, args.steps), np.linspace(*JOINT8, args.steps)):
            q = dict(qmap); q['arm_joint7'], q['arm_joint8'] = float(c7), float(c8)
            poses = chain(rows, q)
            finger = np.concatenate([apply(poses[n], body_clouds[n]) for n in ('arm_link7', 'arm_link8')])
            arm = np.concatenate([apply(poses[n], body_clouds[n]) for n in ARM_CHAIN])
            chassis = np.concatenate([apply(poses[n], body_clouds[n]) for n in body_clouds
                                      if not n.startswith('arm_') and n != GRIPPER])
            # rotate body +X onto +Y so the same radial maths answers the frontal case
            spin = Rotation.from_rotvec([0., 0., np.pi/2+args.swing_rad]).as_matrix()
            f = finger@spin.T; a = arm@spin.T; ch = chassis@spin.T
            fz = apply(base_world, f)[:, 2]; az = apply(base_world, a)[:, 2]
            fr = np.hypot(f[:, 0], f[:, 1]-standoff); ar = np.hypot(a[:, 0], a[:, 1]-standoff)
            entries.append({
                'jaw_gap_m': float(c7-c8),
                'finger_lowest_world_z_m': float(fz.min()),
                'arm_lowest_world_z_m': float(az.min()),
                'finger_below_rim_inside_wall': int(np.sum((fz < RIM_Z_M) & (fr < OUTER_RADIUS_M))),
                'arm_below_rim_inside_wall': int(np.sum((az < RIM_Z_M) & (ar < OUTER_RADIUS_M))),
                'finger_min_radial_to_axis_m': float(fr.min()),
            })
        report['standoffs'][str(standoff)] = {
            'chassis_forward_extent_m': float(
                np.concatenate([apply(chain(rows, qmap)[n], body_clouds[n]) for n in body_clouds
                                if not n.startswith('arm_') and n != GRIPPER])[:, 0].max()),
            'sweep': entries,
            'worst_finger_rim_clearance_m': min(e['finger_lowest_world_z_m'] for e in entries) - RIM_Z_M,
            'any_finger_below_rim_inside_wall': any(e['finger_below_rim_inside_wall'] for e in entries),
            'any_arm_below_rim_inside_wall': any(e['arm_below_rim_inside_wall'] for e in entries),
            'finger_min_radial_over_sweep_m': min(e['finger_min_radial_to_axis_m'] for e in entries),
        }
    report['chassis_forward_extent_m'] = report['standoffs'][str(1.45)]['chassis_forward_extent_m']
    report['minimum_safe_standoff_m'] = report['chassis_forward_extent_m'] + OUTER_RADIUS_M + .03

    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('recorded RAISE pose (arm q):', np.round(report['arm_q_rad_at_handoff'], 4))
    print('chassis forward extent: %.4f m -> minimum safe standoff %.4f m'
          % (report['chassis_forward_extent_m'], report['minimum_safe_standoff_m']))
    print()
    print('%-9s %-14s %-16s %-16s %s' % ('standoff', 'chassis gap', 'finger rim clr',
                                         'finger min radial', 'finger/arm below rim in wall'))
    for s, e in report['standoffs'].items():
        print('%-9s %+14.4f %+16.4f %+16.4f  %s / %s' % (
            s, float(s)-OUTER_RADIUS_M-report['chassis_forward_extent_m'],
            e['worst_finger_rim_clearance_m'], e['finger_min_radial_over_sweep_m'],
            e['any_finger_below_rim_inside_wall'], e['any_arm_below_rim_inside_wall']))


if __name__ == '__main__':
    main()
