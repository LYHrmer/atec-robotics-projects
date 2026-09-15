"""Re-derive a Task B run's official score from the run's own telemetry.

No simulator, no GPU. The evaluator already records the state each reward term
reads -- every object's world root position, the gripper body's position and the
official contact forces -- so both scored terms can be rebuilt from scratch and
compared against what the run claims. If the evaluator's accounting were wrong,
or the telemetry were not the state the score came from, the two would disagree.

Term geometry, from the official `atec_rl_lab/tasks/task_b/mdp/rewards.py`:

* ``objects_in_circle``  object_xy within 1.0 m of (-3, -10) AND 0.0 <= root_z
  <= 0.5, counted once per object per episode, 1.0 each.
* ``grasped_objects``    distance from ``gripper_base`` to an object root
  <= 0.20 m, counted once per object per episode, 1.0 each.

Usage:  python task_b/audit_delivery_score.py RUN_DIR [--report OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OBJECT_COUNT = 18
TARGET_CENTER = (-3.0, -10.0)
TARGET_RADIUS = 1.0
TARGET_Z_MIN = 0.0
TARGET_Z_MAX = 0.5
GRASP_DISTANCE = 0.20


def first_entries(inside):
    """One-time-per-key flags: True only on the step a key first turns inside."""
    inside = np.asarray(inside, dtype=bool)
    counted = np.zeros(inside.shape[1], dtype=bool)
    fired = np.zeros_like(inside)
    for step in range(len(inside)):
        newly = inside[step] & ~counted
        fired[step] = newly
        counted |= inside[step]
    return fired


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    run = args.run
    telemetry = np.load(run / "telemetry.npz")
    result = json.loads((run / "result.json").read_text())
    scoring = json.loads((run / "scoring_events.json").read_text())
    term_names = [str(name) for name in telemetry["reward_term_names"]]
    objects = telemetry["object_xyz"]
    if objects.shape != (len(telemetry["step"]), OBJECT_COUNT, 3):
        raise SystemExit(f"object_xyz has shape {objects.shape}, expected (steps, {OBJECT_COUNT}, 3)")

    xy = np.linalg.norm(objects[:, :, :2] - np.asarray(TARGET_CENTER), axis=2)
    inside = ((xy <= TARGET_RADIUS)
              & (objects[:, :, 2] >= TARGET_Z_MIN)
              & (objects[:, :, 2] <= TARGET_Z_MAX))
    circle = first_entries(inside).sum(axis=1).astype(float)

    gripper = telemetry["gripper_xyz"]
    reach = np.linalg.norm(objects - gripper[:, None, :], axis=2)
    grasp = first_entries(reach <= GRASP_DISTANCE).sum(axis=1).astype(float)

    rebuilt = {"grasped_objects": grasp, "objects_in_circle": circle}
    recorded = {name: telemetry["reward_terms"][:, index]
                for index, name in enumerate(term_names)}
    if set(term_names) != set(rebuilt):
        raise SystemExit(f"telemetry records terms {term_names}; this audit rebuilds {list(rebuilt)}")

    # telemetry[i] is the state BEFORE step i+1, while the terms recorded at
    # telemetry[i] are read AFTER that step, so a recorded term belongs to the
    # geometry one sample later: recorded[i] should equal rebuilt[i + lag]. Check
    # both directions rather than assume, and keep the one that agrees.
    agree = {}
    for lag in (0, 1):
        stop = len(circle) - lag
        agree[lag] = {name: int(np.sum(recorded[name][:stop] != rebuilt[name][lag:lag + stop]))
                      for name in rebuilt}
    best = min(agree, key=lambda lag: sum(agree[lag].values()))

    totals = {name: float(rebuilt[name].sum()) for name in rebuilt}
    reported_totals = result.get("reward_term_totals_raw") or {}
    score_total = float(sum(totals.values()))
    seconds_per_step = float(telemetry["dt"])
    events = []
    for name in rebuilt:
        for step in np.flatnonzero(rebuilt[name] > 0):
            index = int(step) - best          # the sample the term was read at
            events.append({"term": name, "step": index + 1,
                           "geometry_sample_index": int(step),
                           "policy_seconds": round((index + 1) * seconds_per_step, 3)})
    events.sort(key=lambda item: item["step"])

    recorded_events = [(int(e["step"]), name)
                       for e in scoring for name, value in (e["reward_terms"] or {}).items() if value > 0]
    rebuilt_events = [(int(e["step"]), e["term"]) for e in events]

    contact = telemetry["illegal_force"]
    checks = {
        "term_values_match_the_recorded_terms": sum(agree[best].values()) == 0,
        "score_total_matches": abs(score_total - float(result.get("score_raw_total", -1))) < 1e-6,
        "per_term_totals_match": all(
            abs(totals[name] - float(reported_totals.get(name, -1))) < 1e-6 for name in totals),
        "scoring_events_match": rebuilt_events == recorded_events,
        "no_illegal_contact_all_run": float(contact.max()) == 0.0,
        "no_official_termination": not result.get("terminated") and not result.get("truncated"),
    }
    report = {
        "run_directory": str(run),
        "steps": int(len(telemetry["step"])),
        "step_dt_s": seconds_per_step,
        "rebuilt_score_raw_total": score_total,
        "recorded_score_raw_total": result.get("score_raw_total"),
        "rebuilt_term_totals": totals,
        "recorded_term_totals": reported_totals,
        "rebuilt_scoring_events": rebuilt_events,
        "recorded_scoring_events": recorded_events,
        "state_term_lag_steps": best,
        "mismatches_at_each_lag": {str(lag): agree[lag] for lag in agree},
        "max_official_illegal_contact_force_n": float(contact.max()),
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "Re-derives the two scored terms from the recorded object and gripper positions and "
                 "the official term geometry, then checks them against the run's own accounting. It "
                 "does not re-simulate, so it validates the scoring, not the physics.",
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
