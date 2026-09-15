# Task B heading schedule (turn-priority speed scheduling)

`task_b/heading_schedule.py` — new, independent, no other file changed. Nothing in
this document has been executed in simulation, and no motion, alignment or Task B
score is claimed. All constants are hypotheses, not verified safety limits.

## 1. The recorded failure this addresses

All seed 42, official `ATEC-TaskB-B2wPiper`, physics/actuators/rewards unchanged:

| run | outcome |
| --- | --- |
| `first_reach_seed42_01` | 1800 steps, stable, score 0, ~1.6 m forward, **almost no yaw** at common ~0.10 / half-diff ~0.20 |
| `first_reach_gain3_seed42_01` (no stance hold) | terminated step 442 (8.84 s), RR_thigh illegal contact 610 N; step-400 diagnostics base z ~0.4797, quat (wxyz) `[.99198,.10024,-.06637,.03902]`, roll ~0.20 rad |
| pulse variant | wheels stopped near raw `sin(tilt)=0.12`, still terminated step 447 — **a wheel stop is not physical recovery**, the legs kept folding |
| `stance_hold` + gain 3 | survived 2400 steps / 48 s, yaw only ~1.5 deg |
| `stance_hold` + gain 8 (`task_b_score/first_reach_stance_gain8_seed42_01`) | stable so far, yaw ~6.5 deg, **but forward motion is fast enough that the target leaves the camera FOV**; not a score or pass |

The world poses above are retrospective diagnostics from result artifacts. They are
not inputs to this helper: it reads only the caller's public RGB-D bearing/distance
and public proprio `[3:6]` body rates / `[9:12]` projected gravity.

The last row is the failure being attacked. Raising total drive gain finally
produced measurable yaw, but the same gain also spends the approach distance, and
once the object is out of the field of view the bearing estimate is gone before
the turn finishes.

## 2. The trade-off

This chassis rolls far better than it yaws (measured: same-direction commands
track and translate; differentials up to ±0.2 normalized yielded ~0 yaw). So
alignment must be bought with time, not distance: hold the forward request at a
small floor — or exactly zero once the target is close — while `|bearing|` is
large, and keep the caller's full differential. This schedule never makes the turn
faster; it stops the approach from consuming the alignment budget.

Cost, stated honestly: near-zero common with a large differential is close to the
counter-rotating pattern that preceded the step-322 (0.6 pure differential),
step-442 and step-447 illegal contacts. This schedule does not make that pattern
safe. It is only defensible while root's `stance_hold.py` is holding the settled
leg geometry, and its own tilt/tilt-rate stop zeroes wheels only — the pulse run
shows that is not leg recovery.

## 3. Where it may be used

* **Yes** — root's existing direct differential path in `first_reach.py:190`,
  which forms `forward + side_sign * turn` itself and clips per wheel.
* **Yes** — `task_b.locomotion.LocomotionController` in **passthrough** mode
  (`bearing_error=None`), which only applies its physical per-wheel bound and slew.
* **No** — `LocomotionController` arc mode. That mode enforces
  `|differential| <= curvature_ratio * |common|` (0.8 default) so every wheel keeps
  one sign; a 4%-forward schedule would have its differential cut to ~3% of the
  forward request. Same-direction arc limiting and turn-priority near-spin are
  mutually exclusive intents. This helper claims the second and says so.

It is also not a second drive-gain controller: root keeps ownership of the total
gain, and this module only multiplies by a bounded scalar schedule (no torque or
speed integrator).

## 4. API and units

```python
from task_b.heading_schedule import HeadingSchedule, HeadingScheduleError

schedule = HeadingSchedule(dt=0.02)            # bounded keyword parameters, all documented
record = schedule.update(bearing_error,        # rad, body frame, + toward +y (left)
                         forward_request,      # normalized joint_wheel action units
                         turn_request,         # normalized half-differential, sign preserved
                         target_valid=True,
                         observed_yaw_rate=0.0, # rad/s, proprio[5]
                         projected_gravity=None,# proprio[9:12], or 6 values with proprio[3:6]
                         distance_m=None)       # detector forward planar distance, m
record["common"], record["differential"], record["mode"]
schedule.debug, schedule.calls, schedule.reset(), schedule.describe()
```

* Normalized action units throughout. Wheel speed is **never** converted to m/s;
  the wheel radius is not verified here.
* `common = forward_factor * forward_request`, `forward_factor ∈ [0, 1]`.
* `differential = turn_factor * turn_request`, `turn_factor = 1.0` unless
  `turn_boost_max > 1` **and** `turn_boost_per_s > 0` explicitly authorize a
  bounded boost (default: disabled, so neither request can be amplified).
* Modes: `TURN` (`|bearing| >= turn_enter`), `BLEND` (hysteresis band), `DRIVE`
  (`|bearing| <= align_enter`), `STOP`.
* Immediate zero + memory cleared: `target_valid=False`, `bearing_error=None`,
  both requests zero, `tilt >= tilt_stop_rad`, or `|(w_x,w_y)| >= tilt_rate_stop`.
  A stop zeroes this module's two outputs only; it never resets, terminates or
  scores anything.
* Inputs are validated before any gate: non-finite or over-bounded scalars raise
  `HeadingScheduleError` (including huge finite requests, `|request| > 5.0`), as
  does a gravity vector with norm outside `[0.5, 1.5]` or a 4-element input (a root
  quaternion is rejected by shape and by contract).
* The caller's turn sign is passed through and never inferred from drift;
  `turn_sign_matches_bearing` is reported as a diagnostic only.

## 5. Parameters — hypotheses, not safety

| parameter | default | hypothesis |
| --- | --- | --- |
| `turn_enter` / `turn_exit` | 0.30 / 0.22 rad | a 0.08 rad hysteresis band plus `min_mode_calls=10` (0.2 s) removes chatter a single smooth gate shows on a noisy bearing |
| `align_enter` / `align_exit` | 0.10 / 0.16 rad | below ~6 deg the residual bearing is worth trading for approach speed |
| `turn_forward_floor` | 0.04 | same floor root is testing, so the two schedules differ only in structure |
| `align_before_close_m` | 1.00 m | inside 1 m an unaligned target must be turned to first, or it leaves the FOV — this is the rule the gain-8 run lacked |
| `stop_distance_m` / `approach_band_m` | 0.50 / 0.60 m | forward tapers to zero at 0.5 m; root still owns the real standoff and brake |
| `per_wheel_abs_max` | 0.60 | matches root's own per-wheel clip; when it binds, **forward yields first** |
| `tilt_stop_rad` / `tilt_rate_stop` | 0.12 rad / 0.45 rad/s | same trip points root already uses, so this is a redundant floor, not a new policy |
| `losing_bearing_rate` | 0.05 rad/s | if `|bearing|` is growing, pin forward to the floor regardless of mode |
| `turn_boost_max` / `turn_boost_per_s` | 1.0 / 0.0 (off) | a weak yaw response is a **diagnostic**; escalation must be opted into and stays bounded |

`yaw_response_weak` latches after `yaw_stall_latch_calls` (100 calls ≈ 2 s) of
`|turn_request| >= 0.10` with `|filtered yaw rate| < 0.03 rad/s`. It proves nothing
about the cause — drive authority, per-wheel normal load and ground friction are
not observable from these inputs.

## 6. CPU tests to run (not yet executed by me)

No test file is included in this delivery; these are the checks I would assert,
all synthetic and Isaac-free:

1. **Immediate parking** — `target_valid=False`, `bearing_error=None`, and
   `forward_request=turn_request=0` each return `common == differential == 0.0`,
   `mode == "STOP"`, and clear memory (a following normal call must start from
   `mode_dwell_calls == 0`, `boost == 1.0`, no bearing-rate carry-over).
2. **No amplification** — over random finite requests and bearings,
   `|common| <= |forward_request|` and `|differential| <= |turn_request|` with the
   default `turn_boost_max=1.0`; `record["request"]["amplified"] is False`.
3. **Turn priority** — at `|bearing| = 0.4`, `forward_factor == 0.04`; with
   `distance_m = 0.9` and `|bearing| = 0.4`, `common == 0.0` while
   `differential == turn_request`.
4. **Headroom yields forward** — `forward_request=0.5, turn_request=0.5`:
   `|differential| == 0.5` and `|common| <= 0.1` (`per_wheel_abs_max=0.6`).
5. **Hysteresis / no chatter** — bearing dithering across 0.22 rad by ±0.005 for
   400 calls produces at most a couple of mode changes, and never a `TURN→DRIVE`
   transition without passing through `BLEND`.
6. **Sign preservation** — `sign(differential) == sign(turn_request)` for both
   signs, including when the bearing sign disagrees.
7. **Validation before gates** — NaN, ±inf and `1e30` in any scalar raise
   `HeadingScheduleError` *even when* `target_valid=False`; a 4-element
   `projected_gravity`, a zero vector and a norm-3 vector all raise.
8. **Posture stop** — `projected_gravity` at 0.13 rad of tilt, or 6-element input
   with `|(w_x,w_y)| = 0.5`, returns both outputs zero with a populated
   `stop_reasons`.
9. **Bounded boost** — with `turn_boost_max=1.5, turn_boost_per_s=0.5`, a stalled
   yaw in `TURN` mode raises `turn_factor` monotonically to exactly 1.5 and no
   further, and it decays back to 1.0 once the yaw responds or the mode leaves
   `TURN`.
10. **JSON safety** — `json.dumps(record, allow_nan=False)` and
    `json.dumps(schedule.describe(), allow_nan=False)` succeed on every call above.
11. **Constructor validation** — mis-ordered thresholds, `turn_boost_max < 1`,
    negative `dt` and `align_before_close_m < stop_distance_m` all raise.

## 7. Minimal root comparison

Two 500-step runs, seed 42, official physics, `stance_hold.py` active, identical
drive gain and identical detector, differing only in the schedule:

* **A (control)** root's current fixed multiplicative gate — `forward *= clip(1 -
  |bearing|/0.25, 0.04, 1)` at `first_reach.py:188`.
* **B (candidate)** the same `forward`/`turn` values passed through
  `HeadingSchedule.update(...)` with `distance_m` from the selected detection, the
  result driving the same direct differential path.

Read from the traces: cumulative `|yaw|` change, `|bearing|` at the last valid
detection, number of calls with a valid target (FOV retention), planar distance
travelled, mode-change count, `yaw_response_weak` latch, max `sin(tilt)`, and
whether either run terminates. B is worth keeping only if it retains a valid
target for materially more calls and ends with a smaller `|bearing|` at a
comparable or lower travelled distance, without a new termination. If B terminates
or shows a higher max tilt, the near-spin risk in §2 is the first suspect and the
correct next step is lowering `turn_request` upstream, not raising the floor here.

## 8. Stance geometry (hypothesis only, not implemented)

Changing stance width changes both the drive-force moment arm and the contact
scrub resistance. A wider stance does not by itself reduce average per-wheel
normal load: total vertical support still balances the same weight, and its
distribution depends on body attitude and load location. Neither narrowing nor
widening can be recommended from the existing wheel-speed means alone. Any such
experiment belongs to the leg-reference owner and requires static geometry and
collision checks followed by a separate bounded physical comparison. This helper
implements no leg changes and provides no validated abduction offset.

## 9. Independent CPU review after the Claude delivery

The real Claude CLI call completed successfully in the existing session
`5dd69995-f42f-4c09-a02e-4ccb322d7283` (6 turns). It delivered this design and the
original helper, SHA `cb776c7365c2513831dd62fe3b5009dda04cef409c5722a1254baab2d813f7e2`.
No tests were executed by that CLI call. Codex independently reviewed and tested
the helper, then corrected two demonstrated boundary failures:

- A target switching from +0.4 to -0.4 rad made the signed-bearing EMA pass through
  zero. The original helper briefly allowed full distance-tapered forward speed,
  even at range0.9m while the raw bearing remained large. Forward constraints now
  use `max(abs(raw), abs(filtered))`; mode dwell also cannot postpone reducing
  forward authority. The 20-frame regression holds common exactly zero.
- Parking previously retained filtered yaw from the old phase. It now clears
  that filter with the rest of the scheduling memory.

The earlier stance-width paragraph was also corrected: moment-arm changes alone
do not prove easier yaw or lower wheel normal loads.

`task_b/audit_heading_schedule.py` now executes **29 passing synthetic CPU checks**,
recorded in `results/task_b_heading_schedule_cpu_audit.json`. Coverage includes
700 bounded random requests with no default amplification or sign reversal,
headroom allocation, missing targets, invalid/huge input, gravity and raw-rate
stops, hysteresis, explicit-only boost, bearing side-switches and restart state.
This is still not a physical alignment, stability or scoring result.

Integration must explicitly place this helper relative to the caller's drive
gain. Default `per_wheel_abs_max=.60` is a normalized request cap, not a promise
to preserve a gain8 command applied before it. Do not silently multiply by8 after
claiming the helper's final bound. The scalar distance argument must use a
documented body-frame range definition from the public detector. The caller owns
the final stop before arm motion and must provide zero common and turn requests
for that phase.
