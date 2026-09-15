"""Grasp evidence from an oracle grasp probe's telemetry -- no new recorder.

Re-derives the Piper jaw midpoint from the recorded joint vector and base pose with
the verified static FK, then reports whether the target object left the ground and
whether it tracked the jaws. Everything comes from telemetry.npz (q / base_xyz /
base_quat / object_xyz) that evaluate.py already writes.

"Leaves the ground" is a rise of the object's root above its own pre-descent height.
"Moves with the jaws" is the object staying a near-constant, small distance from the
jaw midpoint while it rises -- a pinch, not a nudge or a scoop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from task_b.arm_kinematics import GRASP_DEPTH, fk  # noqa: E402

ARM = tuple(f"arm_joint{i}" for i in range(1, 9))
PHASE_OF = {"REACH": "reach", "LOWER": "descend", "LOWER_HOLD": "descend", "CLOSE": "close",
            "CLOSE_HOLD": "close", "LIFT": "lift", "LIFT_HOLD": "lift", "DONE": "lift"}


def jaw_midpoints(joint_names, q, base_xyz, base_quat):
    ids = [joint_names.index(name) for name in ARM]
    out = np.empty((len(q), 3))
    for step in range(len(q)):
        pose = fk(q[step][ids])
        body = pose[:3, 3] + GRASP_DEPTH * pose[:3, 2]
        rotation = Rotation.from_quat(base_quat[step][[1, 2, 3, 0]]).as_matrix()
        out[step] = base_xyz[step] + rotation @ body
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = json.loads((args.run / "result.json").read_text())
    plan = result.get("grasp_probe") or {}
    target = int(plan.get("target_object", 1))
    index = target - 1

    tel = np.load(args.run / "telemetry.npz", allow_pickle=False)
    joint_names = [str(n) for n in tel["joint_names"]]
    q, base, quat = tel["q"], tel["base_xyz"], tel["base_quat"]
    objects, steps = tel["object_xyz"], tel["step"]
    phases = []
    with (args.run / "trace.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            phases.append(((row.get("policy_debug") or {}).get("phase"), row["pre_step_base_xyz"][2]))

    jaw = jaw_midpoints(joint_names, q, base, quat)
    obj = objects[:, index]
    rest_z = float(obj[0, 2])
    rise = obj[:, 2] - rest_z
    distance = np.linalg.norm(obj - jaw, axis=1)

    peak = int(np.argmax(rise))
    report = {
        "run": str(args.run), "target_object": target, "target_kind": plan.get("target_kind"),
        "steps": int(len(steps)), "stop_reason": result["stop_reason"],
        "terminated": result["terminated"], "truncated": result["truncated"],
        "active_termination_terms_at_stop": result["active_termination_terms_at_stop"],
        "oracle_plan": plan,
        "object_rest_z_m": rest_z,
        "object_peak_rise_m": float(rise[peak]),
        "object_final_rise_m": float(rise[-1]),
        "object_peak_rise_step": int(steps[peak]),
        "left_the_ground": bool(rise.max() > 0.03),
        "min_jaw_to_object_m": float(distance.min()),
        "jaw_distance_at_peak_rise_m": float(distance[peak]),
        "jaw_distance_first_m": float(distance[0]),
        "gripper_base_min_z_m": float(base[:, 2].min()),
        "base_z_start_m": float(base[0, 2]),
    }
    held = bool(rise.max() > 0.03 and distance[peak] < 0.08)
    report["verdict"] = "real_grasp_observed" if held else "no_grasp_observed"
    report["verdict_basis"] = ("object rose >30 mm with the jaw midpoint within 80 mm of it at the "
                               "peak" if held else
                               "either the object never rose >30 mm or the jaw was not on it")

    sampled = []
    every = max(1, len(steps) // 30)
    for i in range(0, len(steps), every):
        sampled.append({"step": int(steps[i]), "phase": phases[i][0], "base_z_m": round(phases[i][1], 4),
                        "object_rise_m": round(float(rise[i]), 4),
                        "jaw_to_object_m": round(float(distance[i]), 4)})
    report["trace_sample"] = sampled
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
