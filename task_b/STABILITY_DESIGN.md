# Task B stability controller — design, evidence and predictions

> 本文前半部分保留实现时的设计状态与待验证预测；当前 CPU/GPU 实测补充见文末及 [实验记录](../docs/TASK_B_EXPERIMENTS.md)。

`task_b/stability.py` implements `StabilityController`, an **untested experimental
candidate** that filters an incoming normalized action before it reaches the
official action manager. It is not a policy: it never originates a drive or turn
direction, and it does not touch physics, gains, assets, rewards or terminations.

Status: the original candidate completed one 500-step turn test without a
termination, but yawed only about **0.526°**, so useful steering was not
demonstrated. The subsequent boundary fixes passed CPU checks; their physical
paired test is separate. The gains and claimed stabilizing effects below
remain hypotheses.

## API

```python
from task_b.stability import StabilityController

controller = StabilityController(
    schema,                      # live task_b.control.ActionSchema
    observation_joint_names,     # proprio joint order (preserve_order=True)
    default_joint_positions,     # defaults aligned with those names
    dt=0.02,
    leg_mode="raise_low_corners",  # or "off" for direction-independent only
    yaw_assist_max=0.0,            # opt-in magnitude-only turn escalation
    strict=True,
)

safe_action = controller.apply(incoming_action, proprio)  # float32, shape (24,)
record = controller.last_debug()   # or controller.debug
spec = controller.describe()       # JSON-serializable
controller.reset()                 # between episodes
```

`apply` returns a finite `float32` array of the same length as the incoming
action. All eight `joint_arm` entries are copied through unchanged (arm safety
stays with the caller). Only `joint_leg` and `joint_wheel` are adjusted.

`strict=True` raises `StabilityError` on malformed proprio. With `strict=False`,
malformed proprio forces the gate to zero, retracts wheel commands and appears
in `debug["errors"]`. Invalid actions always raise: preserving a non-finite arm
request would violate the finite-output contract. Finite float64 actions that
overflow float32 are rejected before casting, including during settling.

## Inputs

Legal inputs only: the public 84-element proprio vector and the static
joint/action schema. Term order is read verbatim from the official
`source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py`
(`ProprioObservationsCfg`, concatenated, `enable_corruption = False`):

| slice | term | used here |
| --- | --- | --- |
| 0:3 | `base_lin_vel` | no (debug only) |
| 3:6 | `base_ang_vel` (root frame) | yes — roll/pitch rate gate, yaw rate |
| 6:9 | `velocity_commands` | no |
| 9:12 | `projected_gravity` | yes — tilt magnitude and downhill direction |
| 12:36 | `joint_pos` (relative to defaults) | no (layout validation only) |
| 36:60 | `joint_vel` | no |
| 60:84 | `actions` (last action) | no |

`projected_gravity` is `asset.data.projected_gravity_b`, i.e. `Rᵀ·(0,0,−1)`. Its
horizontal part therefore points **downhill in the base frame**, which is the
only geometric fact the leg correction relies on.

Root pose, object poses and contact truth from the recorded logs were used to
choose constants offline. They are never runtime inputs.

## Evidence used (ATEC_Experiments_20260910/task_b_initial, seed 42)

| run | command | outcome |
| --- | --- | --- |
| default settle | zero action | settles in ~100 calls, base z ≈ 0.52 m |
| `forward_seed42_02` | all wheels +0.10 (0.5 rad/s) | survived 500 steps, +0.275 m x in 8 s |
| `turn_seed42_01` | right +0.10 / left −0.10 | survived, yawed only ~0.53° in 8 s (stalled) |
| `turn_seed42_03` | differential 0.6 (3 rad/s) | **illegal contact at step 322 (6.44 s)** |
| `crouch_seed42_02` | hip 0 / thigh 1.0 / calf −2.0 | **illegal RL_thigh 1082 N at step 195**, z 0.326 |
| `crouch_seed42_03` | same at fraction 0.6 | **illegal at step 231**, z ≈ 0.366 |

Read: drive at 0.10 survived this short single-seed trial; ±0.10 produced little
steering, while 0.6 and both tested lower stances produced illegal contacts.
These observations do not establish a general safe command envelope. The
candidate therefore tests small differentials, slow changes and local leg
extension, without a success guarantee.

## What the controller does

**Settle (first 100 calls).** Returned equal to the incoming action cast to
float32. The bootstrap supplies zero for these calls; the wrapper does not
replace arbitrary incoming actions with zero. Filters warm up meanwhile.

**Wheels.** The four requests are decomposed into common `c = mean(w)`,
half-differential `d = (mean(right) − mean(left))/2` (right positive, matching
`task_b.control.wheel_side`) and per-wheel residual. Each part is clipped, then
multiplied by `gate · authority`, then slew limited, then recombined and hard
clipped to `±0.35` per wheel. A request with **all four wheels exactly zero**
clears wheel/assist memory and stops immediately, preserving an upstream visual
controller's loss-of-target stop. Continuous wheels have no usable soft limits,
so this cap is a constant, not something read from the articulation. Non-None
leg/wheel action clips are currently rejected rather than guessed.

**Gate.** `tilt = |(g_x, g_y)|` of the unit projected gravity (= sin of the angle
from vertical) and `tilt_rate = |(ω_x, ω_y)|`, both EMA filtered (τ = 0.15 s and
0.20 s). Each maps through a linear ramp: 1.0 inside the deadband, 0.0 at the
ceiling. Their product is the gate. Above `sin(tilt) = 0.30` (~17.46°) the controller aborts
— wheel targets go to zero at 4× slew and the leg correction retracts.
Raw gravity must have finite norm in `[.5,1.5]`. Upward/horizontal gravity and a
degenerate gravity filter force abort, since `|g_xy|` alone aliases an inverted
robot with an upright one. These boundaries leave normal-pose gains unchanged.

**Authority.** A scalar multiplying all wheel caps. Every gated call multiplies
it by 0.98; every quiet call adds 0.002; floor 0.30. Repeated near-misses make
the run progressively more conservative rather than re-entering the same command.

**Leg levelling (`leg_mode="raise_low_corners"`).** One "corner unit" is a
synthetic common direction: thigh +0.20 rad and calf −0.50 rad, converted through
the live leg scale. It is **not the exact failed crouch action**: front thighs
changed from .8 to 1.0, rear thighs stayed at 1.0 (delta zero), and hips also
moved toward zero. Corners downhill receive negative units in `[−0.12,0]`.
Final leg actions are clipped to `(soft_limit − default) / scale`.

Direct CPU FK of the actual B2w USD confirms the local signs: default front
wheel centers are at body x≈+.303 m, rear centers at x≈−.455 m, and left/right
y signs match their names. Applying −.12 units lowers the front wheel centers
relative to the body by **13.8275 mm** and the rear centers by **12.7895 mm**;
with ground contact this is the geometric extension direction. It also shifts
the feet horizontally by about 3.9–6.5 mm. This supports corner selection and
local extension near default, **not dynamic stabilization, contact preservation
or collision freedom**.

**Yaw assist (off by default).** With `yaw_assist_max > 0`, while the gate is
fully open, the caller is requesting a differential, and `|yaw rate| < 0.15`
rad/s, the controller adds 0.002 per call to the **magnitude** of the caller's
differential, up to `yaw_assist_max·authority`, retracting at 4× on any gate
event plus a 50-call cooldown. It never chooses a direction, so no verified
steering sign is required.

## Constants

| constant | value | why |
| --- | --- | --- |
| `SETTLE_CALLS` | 100 | matches the bootstrap; preserves the initial condition |
| `WHEEL_MAX_COMMON` | 0.15 | 1.5× the magnitude observed to survive one 500-step drive trial |
| `WHEEL_MAX_DIFF` | 0.30 | half the differential that failed at 6.44 s |
| `WHEEL_MAX_RESIDUAL` | 0.05 | small per-wheel asymmetry for a visual-approach caller |
| `WHEEL_MAX_ABS` | 0.35 | final per-wheel clip (1.75 rad/s) |
| slew (common / diff / residual) | 0.002 / 0.003 / 0.004 per call | ≈ the bootstrap's 100-call ramp rate |
| `TILT_DEADBAND` / `TILT_MAX` / `TILT_ABORT` | 0.06 / 0.20 / 0.30 | ≈3.4° / 11.5° / 17° |
| `TILT_RATE_DEADBAND` / `_MAX` | 0.5 / 1.5 rad/s | above the settled-stance noise floor |
| `GRAVITY_TAU` / `ANGVEL_TAU` | 0.15 / 0.20 s | the configured (though disabled) proprio noise is ±0.05 on gravity and ±0.2 rad/s on angular rate; physical vibration remains |
| authority decay / recover / floor | 0.98 / 0.002 / 0.30 | ~35 gated calls halve authority; ~500 quiet calls restore it |
| `LEG_MAX_UNITS` / slew / kp | 0.12 / 0.004 per call / 1.2 | ≤0.024 rad thigh and 0.06 rad calf; full deflection in 0.6 s |
| `YAW_ASSIST_STEP` / deadband / cooldown | 0.002 / 0.15 rad/s / 50 calls | 0.10 of extra differential takes 1 s to build |

## Testable predictions

Run each against the uncorrected same-seed bootstrap.

1. **Neutral on the safe forward run.** With `forward` at 0.10 and default
   settings, `apply` should be a no-op within float32 rounding for the whole
   500 steps (`debug["gates"]["combined"] == 1.0`, `applied_common → 0.10`,
   `corner_units` all 0). Base displacement should match the uncorrected run to
   within a few millimetres. If it does not, the deadbands are too tight.
2. **The 0.6-differential run does not produce an illegal contact at step 322.**
   The differential is capped at 0.30 from the start; the prediction is survival
   to 500 steps. If it still fails, the failure is not differential magnitude and
   H1/H2 are wrong.
3. **The 0.6-crouch run behaves identically.** The controller cannot undo a
   commanded crouch — it only adds raise-only corrections capped at 0.12 units,
   ~0.06 rad of calf. Predicted outcome: still illegal, a few steps later at
   most. It is not a crouch fix and should not be reported as one.
4. **Turn at the cap is likely still stalled.** `turn` with a 0.30 differential
   predicts a yaw rate under ~0.5 rad/s and possibly under the 0.15 rad/s
   stall threshold. If `debug["wheels"]["applied_diff"]` sits at the cap and the
   yaw rate stays below 0.15 rad/s, the correct report is **stalled**, not pass.
5. **`yaw_rate_per_diff_estimate`** in the wheel debug is a running least-squares
   slope of yaw rate against applied differential. After a turn run with real
   excitation, its sign identifies the steering convention that the bootstrap
   listed as a hypothesis. It is diagnostic output only; nothing acts on it.
6. **Gate leads failure.** In any run that still ends in an illegal contact,
   `debug["gates"]["combined"]` should drop below 1.0 at least ~0.5 s before the
   terminating step. If the gate only fires at the same step as the contact, tilt
   is not a usable predictor here and the whole gating premise needs replacing.
7. **`leg_mode="off"` ablation.** Same seed, same caps, no leg correction. Any
   claimed benefit of H3 must show up as a difference between predictions 2/6
   under `"raise_low_corners"` and `"off"`; otherwise report the wheel gating
   alone as the only supported effect.

## Limitations

- Nothing here is verified. No pass may be claimed without runtime evidence.
- The differential→yaw sign is still unknown. The controller only scales
  magnitudes and reports a slope estimate.
- H3's corner mapping and near-default extension sign are now supported by
  direct static USD FK. Dynamic levelling still needs the `leg_mode="off"`
  comparison; static geometry cannot establish that benefit.
- Raising the low corners raises the centre of mass slightly. The 0.12-unit cap
  is the only protection against that being counterproductive.
- Constants come from five recorded runs at a single seed; nothing was swept.
- The controller cannot prevent a failure caused by the *commanded* posture, only
  by wheel commands and small tilt corrections.
- Arm entries are unchecked by contract.

## Independent CPU audit and first trial outcome

`results/task_b_stability_cpu_audit.json` records 14 CPU checks, including
reordered observation/action joints, non-unit scales, arm preservation,
continuous-wheel position-limit handling, bad-gravity/overflow rejection,
explicit stop behavior and four roll/pitch corner-selection cases. Across a
frozen 500-frame normal-pose sequence, boundary fixes leave all float32 actions
**bit-identical**. The 600 random bounded-pose cases stayed finite; maximum
physical joint-bound discrepancy was 5.15e-8 from float32 rounding.

The original `turn_stabilized_seed42_01` finished 500 steps with no termination
and zero score, but about 0.526° yaw change. Settled `sin(tilt)≈.0633` was already
above the .06 deadband, causing repeated authority decay to .30 and final
differential about .0879. Thus the proposed neutral behavior at the baseline
stance was not observed; useful steering is still unproven. Baseline tilt
calibration/gain changes require a separate version and comparison. This audit
changed boundary handling only, not those gains.
