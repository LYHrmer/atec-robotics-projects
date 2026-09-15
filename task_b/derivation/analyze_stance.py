"""Read a stance_descend run and report the stance envelope.

Reports, per step: the commanded drop, the measured body height, the drop actually
achieved against the settled stance, each official contact body's force, and the
leg tracking error. Then derives the settled object poses from the same telemetry,
so the "objects rest at these heights" assumption is measured rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OBJECT_TYPES = {**{f"object_{i}": "sugar" for i in range(1, 7)},
                **{f"object_{i}": "mustard" for i in range(7, 13)},
                **{f"object_{i}": "banana" for i in range(13, 19)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--settle_calls", type=int, default=200)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = json.loads((args.run / "result.json").read_text())
    meta = json.loads((args.run / "environment_metadata.json").read_text())
    tel = np.load(args.run / "telemetry.npz", allow_pickle=False)
    steps = tel["step"]
    base_z = tel["base_xyz"][:, 2]
    obj = tel["object_xyz"]                      # (steps, 18, 3)
    illegal = tel["illegal_force"]               # (steps, n_bodies)
    bodies = [str(b) for b in tel["illegal_contact_body_names"]]
    dt = float(tel["dt"])

    drops, tracking = [], []
    with (args.run / "trace.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            debug = row.get("policy_debug") or {}
            drops.append(debug.get("drop_cmd_m"))
            tracking.append(debug.get("leg_tracking_error_rad"))
    drops = np.array([np.nan if d is None else d for d in drops], dtype=float)
    tracking = np.array([np.nan if t is None else t for t in tracking], dtype=float)

    settled = int(args.settle_calls)
    settled_base_z = float(np.median(base_z[max(0, settled - 20):settled]))
    achieved = settled_base_z - base_z

    report = {
        "run": str(args.run), "steps": int(len(steps)), "stop_reason": result["stop_reason"],
        "terminated": result["terminated"], "truncated": result["truncated"],
        "active_termination_terms_at_stop": result["active_termination_terms_at_stop"],
        "camera_free": result.get("camera_free"), "enable_cameras": result.get("enable_cameras"),
        "settled_base_z_m": settled_base_z,
        "commanded_drop_max_m": float(np.nanmax(drops)) if np.isfinite(drops).any() else None,
        "achieved_drop_at_stop_m": float(achieved[-1]),
        "max_achieved_drop_m": float(np.max(achieved)),
        "illegal_contact_body_names": bodies,
        "max_illegal_force_per_body": dict(zip(bodies, np.max(illegal, axis=0).tolist()))
        if illegal.size else {},
        "illegal_contact_threshold": (meta.get("illegal_contact") or {}).get("threshold"),
        "leg_tracking_error_max_rad": float(np.nanmax(tracking)) if np.isfinite(tracking).any() else None,
        "base_planar_drift_m": float(np.max(np.linalg.norm(
            tel["base_xyz"][:, :2] - tel["base_xyz"][settled - 1, :2], axis=1))),
    }

    # Settled object poses, measured from the same run.
    rest = obj[settled - 1] if settled <= len(obj) else obj[-1]
    tops = {}
    for index in range(18):
        name = f"object_{index + 1}"
        tops[name] = {"type": OBJECT_TYPES[name], "root_z_m": float(rest[index][2])}
    report["settled_object_root_z_m"] = tops
    by_type: dict[str, list] = {}
    for entry in tops.values():
        by_type.setdefault(entry["type"], []).append(entry["root_z_m"])
    report["settled_root_z_by_type"] = {k: {"min": float(np.min(v)), "max": float(np.max(v)),
                                            "mean": float(np.mean(v)), "n": len(v)}
                                        for k, v in by_type.items()}

    # What the measured envelope unlocks. The Piper's fingers reach at most
    # REACH_BELOW_BODY m below the body origin (global optimum over the bounded
    # joint space, cross-checked against this repository's own ~0.294 m note and
    # against M1's measured jaw-midpoint position). A top grasp wants the
    # fingertip ~30 mm below the object's top so the pads flank its upper body.
    REACH_BELOW_BODY = 0.2937
    half_extent = {"sugar": 0.04635, "mustard": 0.09565, "banana": 0.01930}
    envelope = {}
    for kind, stats in report["settled_root_z_by_type"].items():
        top = stats["mean"] + half_extent[kind]
        required_body_z = top - 0.03 + REACH_BELOW_BODY
        deepest = settled_base_z - report["max_achieved_drop_m"]
        envelope[kind] = {
            "settled_top_m": round(top, 4),
            "required_body_z_m": round(required_body_z, 4),
            "required_drop_m": round(settled_base_z - required_body_z, 4),
            "deepest_body_z_reached_m": round(deepest, 4),
            "graspable_at_measured_envelope": bool(deepest <= required_body_z),
        }
    report["grasp_envelope_by_type"] = envelope
    report["fingertip_floor_model"] = {
        "reach_below_body_origin_m": REACH_BELOW_BODY,
        "note": "fingertip world z = body origin z - 0.2937 for a level body; verified by "
                "global joint-space optimisation and consistent with M1's measured geometry",
    }

    # The stance envelope: the deepest drop reached before any illegal contact.
    contact_steps = np.where(illegal.max(axis=1) > 0)[0] if illegal.size else np.array([], dtype=int)
    report["first_nonzero_contact_force_step"] = int(contact_steps[0] + 1) if len(contact_steps) else None
    report["achieved_drop_at_first_contact_m"] = (float(achieved[contact_steps[0]])
                                                  if len(contact_steps) else None)

    # Sampled depth curve.
    curve = []
    for index in range(settled, len(steps), max(1, (len(steps) - settled) // 25)):
        curve.append({"step": int(steps[index]), "sim_s": round(float(steps[index] * dt), 2),
                      "drop_cmd_m": None if not np.isfinite(drops[index]) else round(float(drops[index]), 4),
                      "achieved_drop_m": round(float(achieved[index]), 4),
                      "base_z_m": round(float(base_z[index]), 4),
                      "max_illegal_force_n": round(float(illegal[index].max()), 3) if illegal.size else None})
    report["depth_curve"] = curve

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
