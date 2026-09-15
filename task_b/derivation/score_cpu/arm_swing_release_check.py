"""Does swinging ONLY the arm reach a barrel that is dead ahead? (static CPU)

The chassis cannot turn: measured on plan_d8 the wheel differential turns about
1 degree and then stalls at 0.00004 rad/s, so a dock that needs 0.49 rad of
heading change is out of reach. But the residual heading error after backing
away is only ~0.15-0.26 rad, and joint1 of the Piper arm rotates the whole arm
about its mount with a range of -150..+124 degrees.

So the question is whether the ARM can absorb the residual heading error instead
of the chassis. This tests exactly that: the barrel is placed dead ahead at the
standoff, the RECORDED raise pose is used, and only arm_joint1 is offset.

An earlier version rotated the barrel cloud about the body origin instead, which
is NOT equivalent - the arm mount is offset 0.2 m from the origin, so rotating
the arm about the mount and rotating the barrel about the origin differ by about
0.2*phi. That test was wrong and this one replaces it.

Static geometry on recorded telemetry. Not collision simulation, not a delivery.
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

REPO = Path("/home/lybm/ATEC_Robotics_Projects_20260910")
sys.path.insert(0, str(REPO))

ASSETS = Path("/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model")
RUN = Path("/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_d3_carry_probe_seed42_01")
ARM_CHAIN = tuple("arm_link"+str(i) for i in range(1, 9))
GRIPPER = "gripper_base"
OUTER_RADIUS_M = 1.0
INNER_INSCRIBED_M = .98*np.cos(np.pi/32.)
RIM_Z_M = .55
JOINT7 = (0., .035)
JOINT8 = (-.035, 0.)


def quat(v):
    if hasattr(v, "GetReal"):
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
        ax = prim.GetAttribute("physics:axis")
        rows.append(dict(name=prim.GetName(), parent=b0[0].name, child=b1[0].name,
                         kind=prim.GetTypeName(), axis=ax.Get() if ax else None,
                         pj=xform(j.GetLocalPos0Attr().Get(), quat(j.GetLocalRot0Attr().Get())),
                         cj=xform(j.GetLocalPos1Attr().Get(), quat(j.GetLocalRot1Attr().Get()))))
    return rows


def chain(rows, qmap):
    poses = {"base_link": np.eye(4)}
    pending = list(rows)
    while pending:
        prev = len(pending)
        for r in list(pending):
            if r["parent"] not in poses:
                continue
            origin = poses[r["parent"]]@r["pj"]
            motion = np.eye(4)
            if r["axis"]:
                axis = np.eye(3)["XYZ".index(r["axis"])]
                val = qmap.get(r["name"], 0.)
                if r["kind"] == "PhysicsRevoluteJoint":
                    motion[:3, :3] = Rotation.from_rotvec(axis*val).as_matrix()
                elif r["kind"] == "PhysicsPrismaticJoint":
                    motion[:3, 3] = axis*val
            poses[r["child"]] = origin@motion@np.linalg.inv(r["cj"])
            pending.remove(r)
        if len(pending) == prev:
            raise RuntimeError("unresolved tree")
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
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    tel = np.load(RUN/"telemetry.npz")
    names = tel["joint_names"].tolist()
    handoff = None
    with (RUN/"trace.jsonl").open() as fh:
        for i, line in enumerate(fh):
            if (json.loads(line).get("policy_debug") or {}).get("phase") == "CARRY_READY":
                handoff = i; break
    qmap = dict(zip(names, tel["q"][handoff].astype(float)))
    base_world = xform(tel["base_xyz"][handoff], quat(tel["base_quat"][handoff]))
    stage = Usd.Stage.Open(str(ASSETS/"robot/b2w/b2w_piper.usda"))
    rows = load_tree(stage); body_clouds = clouds(stage)
    base_q1 = qmap["arm_joint1"]

    # the HELD BOTTLE, placed from the recorded object pose at the handoff so it
    # moves with the arm exactly as it did in the run
    from task_b.arm_kinematics import fk
    from scipy.spatial.transform import Rotation as _R
    obj_path = ASSETS/"objects/task_b/006_mustard_bottle.usd"
    ostage = Usd.Stage.Open(str(obj_path))
    mesh = next(UsdGeom.Mesh(pp) for pp in ostage.Traverse() if pp.IsA(UsdGeom.Mesh))
    mpts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    ms = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
    mpts = mpts@ms[:3, :3].T+ms[:3, 3]
    gw = xform(tel["gripper_xyz"][handoff], quat(tel["gripper_quat"][handoff]))
    ow = xform(tel["object_xyz"][handoff, 9], quat(tel["object_quat"][handoff, 9]))
    in_hand = np.linalg.inv(gw)@ow
    mesh_gripper = apply(in_hand, mpts)

    report = {"scope": "static CPU geometry: can the ARM absorb the residual heading error? "
                       "Not collision simulation, not a delivery result",
              "note": "the barrel is DEAD AHEAD at the standoff; only arm_joint1 is offset",
              "arm_joint1_at_handoff_rad": float(base_q1),
              "swings": []}
    for swing in (0.0, -0.15, -0.256, -0.35, -0.493):
        entry = {"swing_rad": swing, "standoffs": {}}
        for standoff in (1.45, 1.50, 1.52, 1.55, 1.60):
            worst_finger_rim = 1e9; worst_finger_radial = 1e9
            worst_bottle_rim = 1e9; worst_bottle_radial = 1e9
            finger_hits = 0; arm_hits = 0; bottle_hits = 0
            for c7, c8 in zip(np.linspace(*JOINT7, 7), np.linspace(*JOINT8, 7)):
                q = dict(qmap)
                q["arm_joint1"] = base_q1+swing
                q["arm_joint7"], q["arm_joint8"] = float(c7), float(c8)
                poses = chain(rows, q)
                finger = np.concatenate([apply(poses[n], body_clouds[n])
                                         for n in ("arm_link7", "arm_link8")])
                arm = np.concatenate([apply(poses[n], body_clouds[n]) for n in ARM_CHAIN])
                fz = apply(base_world, finger)[:, 2]
                az = apply(base_world, arm)[:, 2]
                # barrel axis is at body (standoff, 0) - dead ahead
                fr = np.hypot(finger[:, 0]-standoff, finger[:, 1])
                ar = np.hypot(arm[:, 0]-standoff, arm[:, 1])
                gp = poses["gripper_base"]
                # frame discipline: radial is a BODY-frame quantity, world z is
                # not. Mixing them put the bottle 11.8 m from the axis.
                bottle_body = apply(gp, mesh_gripper)
                bz = apply(base_world, bottle_body)[:, 2]
                br = np.hypot(bottle_body[:, 0]-standoff, bottle_body[:, 1])
                worst_bottle_rim = min(worst_bottle_rim, float(bz.min())-RIM_Z_M)
                worst_bottle_radial = min(worst_bottle_radial, float(br[bz >= RIM_Z_M].min())
                                          if (bz >= RIM_Z_M).any() else 1e9)
                bottle_hits += int(np.sum((bz < RIM_Z_M) & (br >= INNER_INSCRIBED_M)
                                          & (br <= OUTER_RADIUS_M)))
                worst_finger_rim = min(worst_finger_rim, float(fz.min())-RIM_Z_M)
                worst_finger_radial = min(worst_finger_radial, float(fr.min()))
                finger_hits += int(np.sum((fz < RIM_Z_M) & (fr >= INNER_INSCRIBED_M)
                                          & (fr <= OUTER_RADIUS_M)))
                arm_hits += int(np.sum((az < RIM_Z_M) & (ar >= INNER_INSCRIBED_M)
                                       & (ar <= OUTER_RADIUS_M)))
            entry["standoffs"][str(standoff)] = {
                "finger_rim_clearance_m": worst_finger_rim,
                "finger_min_radial_to_axis_m": worst_finger_radial,
                "finger_wall_strikes": finger_hits, "arm_wall_strikes": arm_hits,
                "bottle_rim_clearance_m": worst_bottle_rim,
                "bottle_min_radial_above_rim_m": worst_bottle_radial,
                "bottle_body_xy_at_rest": [round(float(bottle_body[:, 0].mean()), 4),
                                           round(float(bottle_body[:, 1].mean()), 4)],
                "bottle_wall_strikes": bottle_hits,
                "bottle_over_mouth": bool(worst_bottle_radial < INNER_INSCRIBED_M),
                "fingers_inside_mouth": bool(worst_finger_radial < INNER_INSCRIBED_M),
            }
        report["swings"].append(entry)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("barrel DEAD AHEAD; only arm_joint1 offset.  inner inscribed radius %.4f" % INNER_INSCRIBED_M)
    print()
    print("%-9s %-9s %-11s %-11s %-9s %-11s %-9s %s" % ("swing", "standoff", "bottle rim",
          "bottle rad", "over mouth", "finger rim", "in mouth", "strikes b/f/a"))
    for entry in report["swings"]:
        for standoff, e in entry["standoffs"].items():
            print("%-9.3f %-9s %+11.4f %+11.4f %-9s %+11.4f %-9s %d/%d/%d" % (
                entry["swing_rad"], standoff, e["bottle_rim_clearance_m"],
                e["bottle_min_radial_above_rim_m"], e["bottle_over_mouth"],
                e["finger_rim_clearance_m"], e["fingers_inside_mouth"],
                e["bottle_wall_strikes"], e["finger_wall_strikes"], e["arm_wall_strikes"]))
        print()


if __name__ == "__main__":
    main()
