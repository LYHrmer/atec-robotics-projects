"""Can the arm reach ~0.1 m further forward, so a LOW release can clear the rim?

Why. The bottle cannot descend past the 0.55 m rim from the current raise pose: its
radial minimum above the rim is 0.9447 m inside a 0.9753 m mouth, but its outer part
sits further out, so any lowering that would contain a restitution-1 bounce clips the
wall annulus (measured: 31 mesh points at 0.15 m of lowering, 553 at 0.20 m). Moving
the CHASSIS closer is not available either - the front extent is 0.481 m, so the body
cannot come nearer than 1.511 m.

That leaves the ARM: reaching further forward moves the whole bottle toward the barrel
axis, which is what lets it descend. This solves IK for a range of forward reaches at
the rim-height release heights and reports whether the bottle would then be clear.

Static CPU geometry on the recorded pose. Not collision simulation, not a delivery.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

REPO = Path("/home/lybm/ATEC_Robotics_Projects_20260910")
sys.path.insert(0, str(REPO))
from task_b.arm_kinematics import fk, solve_ik          # noqa: E402
from task_e_geometry import JOINT_LOWER, JOINT_UPPER    # noqa: E402

ASSETS = Path("/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model")
RUN = Path("/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_d3_carry_probe_seed42_01")
ARM_CHAIN = tuple("arm_link"+str(i) for i in range(1, 9))
OUTER, INNER, RIM = 1.0, .98*np.cos(np.pi/32.), .55


def quat(v):
    if hasattr(v, "GetReal"):
        return Rotation.from_quat([*v.GetImaginary(), v.GetReal()]).as_matrix()
    v = np.asarray(v, dtype=float)
    return Rotation.from_quat(v[[1, 2, 3, 0]]).as_matrix()


def xform(p, r):
    o = np.eye(4); o[:3, :3] = r; o[:3, 3] = p; return o


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
    poses = {"base_link": np.eye(4)}; pending = list(rows)
    while pending:
        prev = len(pending)
        for r in list(pending):
            if r["parent"] not in poses:
                continue
            origin = poses[r["parent"]]@r["pj"]; m = np.eye(4)
            if r["axis"]:
                ax = np.eye(3)["XYZ".index(r["axis"])]; val = qmap.get(r["name"], 0.)
                if r["kind"] == "PhysicsRevoluteJoint":
                    m[:3, :3] = Rotation.from_rotvec(ax*val).as_matrix()
                elif r["kind"] == "PhysicsPrismaticJoint":
                    m[:3, 3] = ax*val
            poses[r["child"]] = origin@m@np.linalg.inv(r["cj"]); pending.remove(r)
        if len(pending) == prev:
            raise RuntimeError("unresolved tree")
    return poses


def collision_clouds(stage):
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

    tel = np.load(RUN/"telemetry.npz"); names = tel["joint_names"].tolist()
    handoff = None
    with (RUN/"trace.jsonl").open() as fh:
        for i, line in enumerate(fh):
            if (json.loads(line).get("policy_debug") or {}).get("phase") == "CARRY_READY":
                handoff = i; break
    qmap = dict(zip(names, tel["q"][handoff].astype(float)))
    base_world = xform(tel["base_xyz"][handoff], quat(tel["base_quat"][handoff]))
    q_arm = np.array([qmap["arm_joint"+str(i)] for i in range(1, 7)])
    pose0 = fk(q_arm); rot0 = pose0[:3, :3]; start = pose0[:3, 3]

    ost = Usd.Stage.Open(str(ASSETS/"objects/task_b/006_mustard_bottle.usd"))
    mesh = next(UsdGeom.Mesh(p) for p in ost.Traverse() if p.IsA(UsdGeom.Mesh))
    mp = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    ms = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
    mp = mp@ms[:3, :3].T+ms[:3, 3]
    gw = xform(tel["gripper_xyz"][handoff], quat(tel["gripper_quat"][handoff]))
    ow = xform(tel["object_xyz"][handoff, 9], quat(tel["object_quat"][handoff, 9]))
    in_hand = np.linalg.inv(gw)@ow
    mesh_gripper = apply(in_hand, mp)

    robot = Usd.Stage.Open(str(ASSETS/"robot/b2w/b2w_piper.usda"))
    rows = load_tree(robot)
    clouds = collision_clouds(robot)

    report = {"scope": "static CPU IK sweep for a further-forward release pose; not a delivery",
              "current_gripper_body_xyz": start.tolist(),
              "rim_z_m": RIM, "inner_inscribed_m": float(INNER), "candidates": []}
    for reach, dz in itertools.product((.60, .65, .70, .75), (0., -.10, -.15, -.20)):
        target = np.array([reach, start[1], start[2]+dz])
        fit = solve_ik(target, rotation=rot0, seed=q_arm, max_nfev=120)
        e = {"reach_x_m": float(reach), "dz_m": float(dz), "target": target.tolist(),
             "ik_success": bool(fit.success), "pos_err_m": float(fit.position_error)}
        if fit.success:
            j = np.asarray(fit.joints, dtype=float)
            e["outside_limits"] = [i+1 for i in range(6)
                                   if not (JOINT_LOWER[i]-1e-6 <= j[i] <= JOINT_UPPER[i]+1e-6)]
            bb = apply(fk(j), mesh_gripper)
            bz = apply(base_world, bb)[:, 2]
            # ON-AXIS dock: barrel axis at body (standoff, 0); use the chassis minimum
            radial = np.hypot(bb[:, 0]-1.511, bb[:, 1])
            above = bz >= RIM
            e["bottle_lowest_world_z"] = float(bz.min())
            e["bottle_radial_max_above_rim"] = float(radial[above].max()) if above.any() else None
            e["bottle_radial_max_all"] = float(radial.max())
            e["whole_bottle_inside_mouth"] = bool(radial.max() < INNER)
            e["wall_strikes"] = int(np.sum((bz < RIM) & (radial >= INNER) & (radial <= OUTER)))
            e["bounce_can_leave"] = bool(bz.min() > RIM)
            # The ARM must clear the rim and wall too - the bottle being fine says
            # nothing about the links carrying it.
            pose = chain(rows, dict(qmap, **{"arm_joint"+str(i+1): float(j[i]) for i in range(6)}))
            arm = np.concatenate([apply(pose[n], clouds[n]) for n in ARM_CHAIN])
            az = apply(base_world, arm)[:, 2]
            ar = np.hypot(arm[:, 0]-1.511, arm[:, 1])
            e["arm_lowest_world_z"] = float(az.min())
            e["arm_wall_strikes"] = int(np.sum((az < RIM) & (ar >= INNER) & (ar <= OUTER)))
        report["candidates"].append(e)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print("dock standoff fixed at the chassis minimum 1.511 m; barrel axis at body (1.511, 0)")
    print("%-8s %-7s %-9s %-11s %-13s %-10s %s" % ("reach_x", "dz", "ik", "bottle_low",
          "radial_max", "in mouth", "verdict"))
    for e in report["candidates"]:
        if not e["ik_success"]:
            print("%-8.2f %-7.2f IK FAILED (pos err %.4f)" % (e["reach_x_m"], e["dz_m"], e["pos_err_m"])); continue
        ok = (not e["outside_limits"] and e["whole_bottle_inside_mouth"]
              and e["wall_strikes"] == 0 and not e["bounce_can_leave"]
              and e.get("arm_wall_strikes", 1) == 0)
        print("%-8.2f %-7.2f %-9s %-11.4f %-13s %-10s %s" % (
            e["reach_x_m"], e["dz_m"], "ok" if not e["outside_limits"] else "limits",
            e["bottle_lowest_world_z"], "%.4f" % e["bottle_radial_max_all"],
            e["whole_bottle_inside_mouth"],
            "SAFE" if ok else ("ARM hits %d" % e.get("arm_wall_strikes", -1)
                               if e.get("arm_wall_strikes") else
                               ("bounce risk" if e["wall_strikes"] == 0
                                else "bottle wall %d" % e["wall_strikes"]))))


if __name__ == "__main__":
    main()
