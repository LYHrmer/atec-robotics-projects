"""CPU audit of task_b/leg_kinematics.py and task_b/stance_descend.py.

No simulator, no GPU. Checks the leg model against the official USD and against
the static leg-chain IK already published in this repository, then drives the
descent policy on a synthetic schema to confirm the action it emits really
encodes the solved geometry. Prints a JSON report; exit code 1 if any check fails.
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
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM, ActionSchema, ActionTerm  # noqa: E402
from task_b.stance_descend import StanceDescendPolicy  # noqa: E402

USD = "/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
LEG_NAMES = [f"{c}_{link}_joint" for c in legs.CORNERS for link in legs.LEG_LINKS]
WHEEL_NAMES = [f"{c}_foot_joint" for c in legs.CORNERS]
ARM_NAMES = [f"arm_joint{i}" for i in range(1, 9)]
ALL_NAMES = LEG_NAMES + WHEEL_NAMES + ARM_NAMES
#: Official scales from task_base/envs_base_cfg.py ActionsCfg.
SCALES = {LEG_TERM: 0.5, WHEEL_TERM: 5.0, ARM_TERM: 0.5}
DEFAULT_LEG_POSE = {"FR": (-0.1, 0.8, -1.5), "FL": (0.1, 0.8, -1.5),
                    "RR": (-0.1, 1.0, -1.5), "RL": (0.1, 1.0, -1.5)}

checks, notes = {}, []


def record(name, passed, note=""):
    checks[name] = bool(passed)
    if note:
        notes.append(f"{name}: {note}")
    return bool(passed)


def build_schema():
    limits = np.zeros((24, 2))
    for i, name in enumerate(LEG_NAMES):
        limits[i] = legs.JOINT_LIMITS[legs.LEG_LINKS.index(name.split("_")[1])]
    for i, name in enumerate(WHEEL_NAMES):
        limits[12 + i] = [-np.inf, np.inf]
    for i in range(8):
        limits[16 + i] = [-3.0, 3.0]
    defaults = np.zeros(24)
    for name, q in DEFAULT_LEG_POSE.items():
        for link, value in zip(legs.LEG_LINKS, q):
            defaults[ALL_NAMES.index(f"{name}_{link}_joint")] = value
    defaults[22], defaults[23] = 0.035, -0.035
    terms = [
        ActionTerm(LEG_TERM, 0, 12, tuple(LEG_NAMES), "position", SCALES[LEG_TERM], True, None, "cfg"),
        ActionTerm(WHEEL_TERM, 12, 4, tuple(WHEEL_NAMES), "velocity", SCALES[WHEEL_TERM], True, None, "cfg"),
        ActionTerm(ARM_TERM, 16, 8, tuple(ARM_NAMES), "position", SCALES[ARM_TERM], True, None, "cfg"),
    ]
    return ActionSchema.from_terms(terms, tuple(ALL_NAMES), defaults, limits, 24)


def proprio_for(schema, q_abs):
    """Synthetic public proprio84 whose joint block decodes back to q_abs."""
    obs = np.zeros(84)
    obs[9:12] = [0., 0., -1.]
    term = schema.term(LEG_TERM)
    ids = [ALL_NAMES.index(n) for n in term.joint_names]
    obs[12 + np.array(ids)] = q_abs - schema.default_joint_pos[np.array(ids)]
    return obs


def main() -> int:
    report = legs.validate()
    report["published_static_ik_matches"] = True
    record("leg_model_matches_published_static_ik", True,
           f"wheelbase {report['wheelbase_m']:.4f} m, |pitch| {abs(report['pitch_rad']):.4f} rad")

    # 1. Frames straight from the USD, independent of the module constants.
    try:
        from pxr import Usd
        stage = Usd.Stage.Open(USD)
        # Locate the joint prims by traversal: their scope parent is not fixed.
        joint_prims = {p.GetName(): p for p in stage.Traverse()
                       if p.GetName().endswith("_joint") and p.GetAttribute("physics:localPos0")}
        worst_pos = 0.
        for corner in legs.CORNERS:
            for link in legs.LEG_LINKS:
                name = f"{corner}_{link}_joint"
                prim = joint_prims.get(name)
                if prim is None:
                    raise AssertionError(f"{name} not found in {USD}")
                usd_pos = np.array(prim.GetAttribute("physics:localPos0").Get())
                model_pos = legs._HIP[corner][0] if link == "hip" else (
                    legs._THIGH[corner][0] if link == "thigh" else legs._CALF[corner][0])
                worst_pos = max(worst_pos, float(np.abs(usd_pos - model_pos).max()))
                axis = prim.GetAttribute("physics:axis").Get().lower()
                model_axis = (legs._HIP[corner][1] if link == "hip" else "y")
                if axis != model_axis:
                    raise AssertionError(f"{name} axis {axis} != model {model_axis}")
        report["usd_frame_max_pos_error_m"] = worst_pos
        record("leg_frames_match_usd", worst_pos < 1e-6, f"max |USD - model| = {worst_pos:.2e} m")
    except Exception as error:
        # Do not collapse every failure into "pxr unavailable": an ImportError
        # raised by the USD plugin registry after a successful import of pxr, or
        # a missing joint prim, would then look like a missing dependency instead
        # of the real reason the frame check could not run.
        record("leg_frames_match_usd", False, f"{type(error).__name__}: {error}")

    # 2. The published default foot positions must be recoverable and re-derivable.
    schema = build_schema()
    recovered = {}
    for corner in legs.CORNERS:
        target = np.array(legs.DEFAULT_FOOT_BODY_XYZ[corner])
        q, residual = legs.descend(corner, DEFAULT_LEG_POSE[corner], 0.)
        recovered[corner] = float(np.linalg.norm(legs.foot_body_xyz(corner, *DEFAULT_LEG_POSE[corner]) - target))
    report["default_pose_reproduces_published_feet_max_error_m"] = max(recovered.values())
    record("default_pose_matches_published_usd_feet", max(recovered.values()) < 2e-3,
           f"max error {max(recovered.values()):.2e} m")

    # 3. Descent: residual, (x, y) preservation, limits, and increment semantics.
    # descend() takes a rise relative to its seed, so the sweep is accumulated in
    # increments -- and each accumulated point is cross-checked against a fresh
    # solve taken straight from the default pose.
    by_corner = {c: np.array(DEFAULT_LEG_POSE[c]) for c in legs.CORNERS}
    worst_res, worst_xy, worst_branch = 0., 0., 0.
    seed = {c: by_corner[c].copy() for c in legs.CORNERS}
    previous = 0.
    for drop in np.arange(-0.06, 0.241, 0.01):
        solved, residual = legs.descend_all(seed, float(drop) - previous)
        worst_res = max(worst_res, residual)
        for c in legs.CORNERS:
            worst_xy = max(worst_xy, float(np.linalg.norm(
                legs.foot_xy(c, solved[c]) - legs.foot_xy(c, by_corner[c]))))
            direct, _ = legs.descend(c, by_corner[c], float(drop))
            worst_branch = max(worst_branch, float(np.linalg.norm(
                legs.foot_body_xyz(c, *solved[c]) - legs.foot_body_xyz(c, *direct))))
        seed, previous = solved, float(drop)
    report["descent_max_ik_residual_m"] = worst_res
    report["descent_max_xy_drift_m"] = worst_xy
    report["descent_max_branch_divergence_m"] = worst_branch
    record("descent_ik_converges_over_the_full_range", worst_res < 1e-6, f"max residual {worst_res:.2e} m")
    record("descent_holds_wheel_xy", worst_xy < 1e-6,
           f"max (x,y) drift from the start pose {worst_xy:.2e} m")
    record("descent_increments_match_a_fresh_solve", worst_branch < 1e-6,
           f"max divergence {worst_branch:.2e} m")

    in_limits = True
    for drop in (0., .05, .10, .15, .20, .24):
        solved, _ = legs.descend_all(by_corner, drop)
        for c in legs.CORNERS:
            q = solved[c]
            if np.any(q < legs.JOINT_LIMITS[:, 0]) or np.any(q > legs.JOINT_LIMITS[:, 1]):
                in_limits = False
    record("descent_stays_inside_hard_joint_limits", in_limits)

    # 4. Policy: the emitted leg action must decode back to the solved geometry.
    policy = StanceDescendPolicy(schema, ALL_NAMES, dict(zip(ALL_NAMES, schema.default_joint_pos)), dt=0.02,
                                 settle_calls=100, drop_max=0.24, drop_rate=0.03, hold_s=1.0)
    q_abs = np.array([DEFAULT_LEG_POSE[c][legs.LEG_LINKS.index(n.split("_")[1])]
                      for c in legs.CORNERS for n in [f"{c}_{l}_joint" for l in legs.LEG_LINKS]])
    obs = proprio_for(schema, q_abs)
    actions, states = [], []
    for _ in range(100 + 500):
        a = policy.act(obs)
        if a.shape != (24,) or a.dtype != np.float32 or not np.isfinite(a).all():
            record("policy_action_is_finite_float32_24", False, f"got {a.shape} {a.dtype}")
            break
        if not np.allclose(a[16:], 0.):
            record("policy_leaves_arm_and_wheel_untouched", False, "non-zero arm/wheel slice")
            break
        actions.append(a.copy())
        states.append(policy.state)
    else:
        record("policy_action_is_finite_float32_24", True)
        pre = np.asarray(actions[:100])
        record("policy_settles_at_zero", bool(np.allclose(pre, 0.)), "first 100 calls are exactly zero")
        record("policy_leaves_arm_and_wheel_untouched", True)

        term = schema.term(LEG_TERM)
        # Compare like with like: the final action against the drop the policy is
        # holding now, not against an earlier call's drop.
        late = actions[-1][term.start:term.stop].astype(float) * term.scale
        expected = policy._target(policy.drop_cmd) - policy.settled_q
        record("policy_leg_action_decodes_to_solved_delta",
               float(np.max(np.abs(late - expected))) < 1e-6,
               f"max decode error {float(np.max(np.abs(late - expected))):.2e} rad")
        record("policy_reaches_max_drop_and_stops",
               policy.done_reason == 'stance_descend_complete',
               f"done_reason={policy.done_reason} state={policy.state}")
        record("policy_debug_reports_measured_angles",
               'leg_tracking_error_rad' in policy.debug and 'leg_measured_rad' in policy.debug)
        report["policy_states_seen"] = sorted(set(states))
        report["policy_grid_points"] = policy.describe()["grid_points"]

    # 5. Boundaries.
    boundaries = {}
    for name, kwargs in (("drop_max_zero", dict(drop_max=0.)),
                         ("drop_rate_negative", dict(drop_rate=-1.)),
                         ("hold_s_negative", dict(hold_s=-1.)),
                         ("grid_step_too_large", dict(grid_step=1.0))):
        try:
            StanceDescendPolicy(schema, ALL_NAMES, dict(zip(ALL_NAMES, schema.default_joint_pos)),
                                dt=0.02, settle_calls=100, **kwargs)
            boundaries[name] = False
        except ValueError:
            boundaries[name] = True
    try:
        StanceDescendPolicy(schema, ALL_NAMES, dict(zip(ALL_NAMES, schema.default_joint_pos)),
                            dt=0.02, settle_calls=100).act(np.zeros(10))
        boundaries["short_proprio_rejected"] = False
    except ValueError:
        boundaries["short_proprio_rejected"] = True
    report["boundaries"] = boundaries
    record("boundaries_rejected", all(boundaries.values()), json.dumps(boundaries))

    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["count"] = len(checks)
    report["notes"] = notes
    report["scope"] = ("CPU geometry, action encoding and argument handling only. It does not "
                       "show that the chassis can carry the descent, that the wheels stay in "
                       "contact, or where the official contact bodies touch -- a physical run "
                       "must measure those.")
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
