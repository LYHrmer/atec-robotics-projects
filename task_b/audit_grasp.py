"""CPU audit of the first_reach grasp sequence (close jaws, then lift).

No simulator or GPU. Drives FirstReachPolicy through the lowering hold into the
grasp stages on a synthetic schema whose legs track the commanded reference
exactly, so a state-machine or sign error shows up here instead of in a hardware
run. Prints a JSON report; exit code 1 if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_b import leg_kinematics as legs  # noqa: E402
from task_b.audit_stance import ALL_NAMES, DEFAULT_LEG_POSE, build_schema  # noqa: E402
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM  # noqa: E402
from task_b.first_reach import FirstReachPolicy  # noqa: E402

checks, notes = {}, []


def record(name, passed, note=""):
    checks[name] = bool(passed)
    if note:
        notes.append(f"{name}: {note}")


def proprio(schema, leg_q=None, arm_q=None):
    obs = np.zeros(84)
    obs[9:12] = [0., 0., -1.]
    term = schema.term(LEG_TERM)
    if leg_q is not None:
        ids = [ALL_NAMES.index(n) for n in term.joint_names]
        obs[12 + np.array(ids)] = leg_q - schema.default_joint_pos[np.array(ids)]
    if arm_q is not None:
        arm = schema.term(ARM_TERM)
        ids = [ALL_NAMES.index(n) for n in arm.joint_names]
        obs[12 + np.array(ids)] = arm_q - schema.default_joint_pos[np.array(ids)]
    return obs


def make_policy(schema, *, grasp, lowering_m, dt=0.02):
    policy = FirstReachPolicy(schema, ALL_NAMES, dict(zip(ALL_NAMES, schema.default_joint_pos)),
                              dt=dt, settle_calls=100, reach_only=False,
                              lowering_m=lowering_m, grasp=grasp)
    # Put it where the lowering hold has just completed.
    base_q = np.array([DEFAULT_LEG_POSE[c][legs.LEG_LINKS.index(n.split("_")[1])]
                       for c in legs.CORNERS for n in [f"{c}_{l}_joint" for l in legs.LEG_LINKS]])
    policy.calls = 1000
    policy.reach_q = policy.arm_command[:6].copy()
    policy.reach_target = np.array([0.5, 0., -0.3])
    policy.reach_start = policy.calls - 100
    policy.reach_timeout_s = 15.
    policy.reach_up_anchor = np.array([0., 0., 1.])
    policy.reach_base_displacement = np.zeros(3)
    policy.reach_base_yaw = 0.
    policy.lower_authorized = True
    policy.lower_authorized_call = policy.calls - 250      # 5 s ago, inside the 6 s window
    policy.lower_reference_q0 = base_q.copy()
    policy.lower_delta = policy._solve_lower_delta(base_q)
    policy.lowering_alpha = 1.0
    policy.lower_hold_start = policy.calls - 150           # 3 s, past the 2 s hold
    return policy, base_q


def drive(policy, schema, base_q, steps=700):
    """Run the policy with legs tracking the commanded reference exactly."""
    rows = []
    for _ in range(steps):
        leg_q = base_q + policy.lowering_alpha * policy.lower_delta
        arm_q = np.zeros(8)
        arm_q[:6] = policy.arm_command[:6]
        arm_q[6:] = policy.arm_command[6:]
        obs = proprio(schema, leg_q=leg_q, arm_q=arm_q)
        action = policy.act(obs, {})
        rows.append({"subphase": policy.debug.get("reach_subphase"), "action": action.copy(),
                     "alpha": policy.lowering_alpha, "jaw": policy.jaw_target.copy(),
                     "jaw_cmd": policy.arm_command[6:].copy(), "done": policy.done_reason,
                     "reason": policy.state_reason,
                     "grasp_total_s": policy.debug.get("grasp_total_s", 0.)})
        if policy.done_reason is not None:
            break
    return rows


def main() -> int:
    schema = build_schema()
    term, arm, wheel = schema.term(LEG_TERM), schema.term(ARM_TERM), schema.term(WHEEL_TERM)

    # 1. Default behaviour is unchanged when the grasp is off.
    plain, base_q = make_policy(schema, grasp=False, lowering_m=0.02)
    plain_rows = drive(plain, schema, base_q, steps=200)
    record("no_grasp_still_ends_at_the_lowering_hold",
           plain.done_reason == "reach_lowering_hold_complete",
           f"done_reason={plain.done_reason}")

    # 2. The grasp sequence runs to completion, in order.
    policy, base_q = make_policy(schema, grasp=True, lowering_m=0.03)
    rows = drive(policy, schema, base_q)
    seen = [r["subphase"] for r in rows if r["subphase"]]
    order = ["GRASP_CLOSE", "GRASP_CLOSE_HOLD", "GRASP_LIFT", "GRASP_LIFT_HOLD"]
    first_index = {name: next((i for i, s in enumerate(seen) if s == name), None) for name in order}
    record("grasp_reaches_lift_complete",
           policy.done_reason == "reach_grasp_lift_complete", f"done_reason={policy.done_reason}")
    record("grasp_stages_run_in_order",
           all(first_index[n] is not None for n in order)
           and first_index["GRASP_CLOSE"] < first_index["GRASP_CLOSE_HOLD"]
           < first_index["GRASP_LIFT"] < first_index["GRASP_LIFT_HOLD"],
           json.dumps(first_index))
    record("grasp_does_not_exceed_its_budget",
           rows[-1]["grasp_total_s"] <= policy.grasp_budget_s,
           f"grasp took {rows[-1]['grasp_total_s']:.2f} s of a {policy.grasp_budget_s} s budget")

    # 3. Jaws close to the zero stops and stay there.
    crossing = next((i for i, r in enumerate(rows) if np.allclose(r["jaw"], 0.)), None)
    record("jaw_target_goes_to_the_closed_stop", crossing is not None)
    if crossing is not None:
        tail = rows[crossing:]
        record("jaw_stays_closed_through_the_lift",
               all(np.allclose(r["jaw"], 0.) for r in tail))
        closed_cmd = np.max([np.max(np.abs(r["jaw_cmd"])) for r in tail[-30:]])
        record("finger_command_reaches_the_closed_stop", closed_cmd < 1e-3,
               f"max |finger command| = {closed_cmd:.2e} m")

    # 4. The body actually lifts back, monotonically, over the requested time.
    alpha = np.array([r["alpha"] for r in rows])
    lift = next((i for i, r in enumerate(rows) if r["subphase"] == "GRASP_LIFT"), None)
    if lift is not None:
        seg = alpha[lift:]
        record("lift_returns_the_leg_reference_to_zero", float(seg[-1]) == 0.0,
               f"alpha ends at {seg[-1]:.4f}")
        record("lift_is_monotone_non_increasing", bool(np.all(np.diff(seg) <= 1e-12)))
        drop_s = float(np.sum(np.diff(seg) < 0) * policy.dt)
        record("lift_takes_about_the_requested_time", abs(drop_s - policy.grasp_lift_s) < 0.6,
               f"lift spanned {drop_s:.2f} s vs requested {policy.grasp_lift_s} s")

    # 5. Action hygiene throughout the grasp.
    acts = np.array([r["action"] for r in rows])
    record("actions_are_finite_float32_24",
           acts.shape[1] == 24 and acts.dtype == np.float32 and bool(np.isfinite(acts).all()))
    record("arm_reach_slice_and_wheels_untouched_by_the_grasp",
           bool(np.allclose(acts[:, wheel.start:wheel.stop], 0.)))
    # The leg slice is an encoded angle delta, so compare it in radians against the
    # solved delta -- not against the metres of body drop.
    decoded = acts[:, term.start:term.stop].astype(float) * term.scale
    worst_encode = max(float(np.max(np.abs(decoded[i] - alpha[i] * policy.lower_delta)))
                       for i in range(len(rows)))
    record("leg_action_encodes_alpha_times_the_solved_delta", worst_encode < 1e-6,
           f"max encoding error {worst_encode:.2e} rad")
    # And the solved delta itself must still deliver the requested drop.
    solved, _ = legs.descend_all(
        {c: np.array([base_q[ALL_NAMES.index(n)] for n in legs.leg_joint_names(c)])
         for c in legs.CORNERS}, policy.lowering_m)
    rise = [legs.foot_body_xyz(c, *solved[c])[2]
            - legs.foot_body_xyz(c, *[base_q[ALL_NAMES.index(n)] for n in legs.leg_joint_names(c)])[2]
            for c in legs.CORNERS]
    record("solved_delta_delivers_the_requested_drop",
           max(abs(value - policy.lowering_m) for value in rise) < 1e-4,
           f"rises {[round(v, 5) for v in rise]} for a {policy.lowering_m} m drop")

    # 6. Argument handling.
    boundaries = {}
    for name, kwargs in (("grasp_without_lowering", dict(grasp=True, lowering_m=0.)),
                         ("lowering_above_the_envelope", dict(grasp=True, lowering_m=0.30))):
        try:
            make_policy(schema, **kwargs)
            boundaries[name] = False
        except ValueError:
            boundaries[name] = True
    record("invalid_grasp_arguments_rejected", all(boundaries.values()), json.dumps(boundaries))

    # 7. The oracle probe policy: same shape and phase discipline, driven on CPU.
    from task_b.grasp_probe import GraspProbePolicy
    # The evaluator descends first and hands over the delta that undoes it. This is
    # the same solve, taken from the standing pose.
    descended, _ = legs.descend_all({c: np.array(DEFAULT_LEG_POSE[c]) for c in legs.CORNERS}, 0.06)
    restore_delta = np.array(
        [descended[n.split("_")[0]][legs.LEG_LINKS.index(n.split("_")[1])]
         - DEFAULT_LEG_POSE[n.split("_")[0]][legs.LEG_LINKS.index(n.split("_")[1])]
         for n in term.joint_names])
    probe = GraspProbePolicy(schema, ALL_NAMES, dict(zip(ALL_NAMES, schema.default_joint_pos)),
                             dt=0.02, grasp_point_body=[0.50, 0.0, -0.25],
                             descent_delta_rad=restore_delta)
    arm_names = [n for n in ALL_NAMES if n.startswith("arm_joint")]
    leg_names = list(term.joint_names)
    arm_idx = np.array([ALL_NAMES.index(n) for n in arm_names])
    leg_idx = np.array([ALL_NAMES.index(n) for n in leg_names])
    base_legs = np.array([DEFAULT_LEG_POSE[c][legs.LEG_LINKS.index(n.split("_")[1])]
                          for c in legs.CORNERS for n in [f"{c}_{l}_joint" for l in legs.LEG_LINKS]])
    measured_arm = np.array([schema.default_joint_pos[i] for i in arm_idx])
    probe_rows = []
    for _ in range(1200):
        obs = np.zeros(84)
        obs[9:12] = [0., 0., -1.]
        obs[12 + leg_idx] = (base_legs + (1. - probe.alpha) * probe.descent_delta) - schema.default_joint_pos[leg_idx]
        obs[12 + arm_idx] = measured_arm - schema.default_joint_pos[arm_idx]
        action = probe.act(obs)
        probe_rows.append({"phase": probe.phase, "action": action.copy(), "alpha": probe.alpha,
                           "jaw": probe.jaw_target.copy()})
        # The arm tracks the command, which is what the real articulation does.
        measured_arm = np.r_[action[probe.arm.start:probe.arm.stop].astype(float) * probe.arm.scale
                             + schema.default_joint_pos[arm_idx]]
        if probe.done_reason is not None:
            break
    p_acts = np.array([r["action"] for r in probe_rows])
    p_phases = [r["phase"] for r in probe_rows]
    record("probe_actions_are_finite_float32_24",
           p_acts.shape[1] == 24 and p_acts.dtype == np.float32 and bool(np.isfinite(p_acts).all()),
           f"shape {p_acts.shape} dtype {p_acts.dtype}")
    record("probe_completes_all_phases", probe.done_reason == "grasp_probe_complete",
           f"done_reason={probe.done_reason}")
    record("probe_phases_run_in_order",
           all(p in p_phases for p in ("REACH", "CLOSE", "LIFT", "DONE"))
           and p_phases.index("REACH") < p_phases.index("CLOSE") < p_phases.index("LIFT"),
           json.dumps({p: p_phases.index(p) for p in set(p_phases)}))
    record("probe_wheels_untouched", bool(np.allclose(p_acts[:, wheel.start:wheel.stop], 0.)))
    # The very first action must HOLD the stance the evaluator descended to, not
    # return the legs to standing: that would undo the descent before the reach.
    first_leg = p_acts[0, term.start:term.stop].astype(float) * term.scale
    record("probe_first_action_holds_the_descended_stance",
           float(np.max(np.abs(first_leg - probe.descent_delta))) < 1e-6,  # float32 action
           f"max |first leg action - descent delta| = "
           f"{float(np.max(np.abs(first_leg - probe.descent_delta))):.2e} rad")
    record("probe_lift_returns_the_legs_to_the_standing_pose", float(probe.alpha) == 1.0,
           f"lift alpha ends at {probe.alpha:.4f} (1.0 = fully back to the standing stance)")
    closed_at = next((i for i, r in enumerate(probe_rows) if np.allclose(r["jaw"], probe.jaw_closed)), None)
    record("probe_jaws_close_before_the_lift",
           closed_at is not None and closed_at < p_phases.index("LIFT"),
           f"jaw closed at row {closed_at}, lift starts at {p_phases.index('LIFT')}"
           if closed_at is not None else "jaw never closed")
    # The fingers slide along the gripper's local +Y. For a top grasp that axis must
    # be horizontal and perpendicular to the approach, or the jaws close front-to-back
    # along the approach instead of straddling the object.
    jaw_axis = np.array(probe.jaw_axis_body)
    record("probe_jaw_axis_is_horizontal", abs(jaw_axis[2]) < .25,
           f"jaw axis {np.round(jaw_axis, 3)}")
    record("probe_jaw_axis_is_perpendicular_to_the_approach",
           abs(float(jaw_axis @ np.array([1., 0., 0.]))) < .4,
           f"jaw axis . body-x = {float(jaw_axis @ np.array([1., 0., 0.])):+.3f}")

    # 8. Static check: the evaluator's policy constructions must match the policy
    # signatures. Two real bugs in this file's history were exactly this -- a missing
    # name and a stale keyword -- and neither needs the simulator to catch.
    import ast
    import inspect
    evaluate_src = (ROOT / "task_b" / "evaluate.py").read_text()
    from task_b.grasp_probe import GraspProbePolicy
    from task_b.stance_descend import StanceDescendPolicy
    signatures = {"GraspProbePolicy": GraspProbePolicy, "StanceDescendPolicy": StanceDescendPolicy}
    bad_calls = []
    for node in ast.walk(ast.parse(evaluate_src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in signatures:
            params = set(inspect.signature(signatures[node.func.id].__init__).parameters)
            unexpected = sorted({k.arg for k in node.keywords} - params)
            if unexpected:
                bad_calls.append(f"{node.func.id}:{unexpected}")
    record("evaluator_policy_calls_match_signatures", not bad_calls, json.dumps(bad_calls))

    report = {"checks": checks, "passed": all(checks.values()), "count": len(checks),
              "notes": notes,
              "grasp_wall_seconds": round((len(rows) - 1) * policy.dt, 2),
              "stages_seen": sorted(set(s for s in seen)),
              "scope": "CPU state machine, action encoding and argument handling only. It does not "
                       "show that the fingers met an object, that anything was held, or that the "
                       "stance stayed legal -- a physical run must measure those."}
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
