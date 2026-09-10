# Neutral-reference stability profile — design, acceptance criteria, limitations

> 本文前半部分保留实现时的设计状态与待验证预测；当前 CPU/GPU 实测补充见文末及 [实验记录](../docs/TASK_B_EXPERIMENTS.md)。

`task_b/stability_profiles.py` adds one opt-in profile on top of the frozen
`task_b/stability.py`. The baseline file and `task_b/STABILITY_DESIGN.md` are
untouched: `NeutralAwareStabilityController` subclasses `StabilityController`,
feeds it a derived copy of the proprio vector, and keeps every native boundary
safeguard where it is.

**Status: unverified.** No simulation run has been executed for this profile.
Nothing below is a claim of success.

## The problem it addresses

`turn_stabilized_seed42_01` (500 steps, no termination, ~0.526° of post-settle
yaw) shows the baseline gating on a *healthy* pose:

| observation | value |
| --- | --- |
| settled body `sin(tilt)` | 0.0633 |
| baseline `TILT_DEADBAND` | 0.06 |
| combined gate | 0.976, firing on 366 calls |
| authority | pinned at the 0.30 floor |
| applied differential | 0.08788 of a requested 0.6 |

The nominal stance sits just outside the deadband, so the tilt gate never fully
opened, authority decayed monotonically, and the turn was throttled by a
constant pitch offset rather than by any stability event. This falsifies the
earlier "neutral on the safe forward run" prediction: the baseline is not
neutral on a healthy stance.

## API

```python
from task_b.stability_profiles import NeutralAwareStabilityController, build_controller

controller = build_controller(
    "neutral",                    # or "baseline" -> plain StabilityController
    schema, observation_joint_names, default_joint_positions, dt=0.02,
    leg_mode="raise_low_corners", yaw_assist_max=0.0, strict=True,
    # neutral-only, all optional:
    enable_neutral_reference=True, calib_window=20, calib_max_spread=0.03,
    calib_max_rate=0.15, calib_max_reference_tilt=0.15, abs_tilt_deadband=0.18,
)

safe = controller.apply(action, proprio)   # float32 (24,), same contract as the baseline
controller.debug["neutral_reference"]      # absolute + reference-relative diagnostics
controller.profile_debug()                 # same record, standalone copy
controller.describe()                      # baseline spec + neutral_reference section
controller.reset()                         # also drops the frozen reference
```

The constructor is positionally identical to the baseline; every extra argument
is keyword-only with a default. `enable_neutral_reference=False` reproduces the
baseline exactly, so one class can serve both arms of an A/B run. Selecting the
profile is the only opt-in; nothing auto-enables.

## Mechanism

**Calibration (once, then frozen).** During the 100-call zero-action settle the
profile buffers the last 20 usable samples of public `projected_gravity` (unit
normalized) with the raw `|(ω_x, ω_y)|` of each. At the end of the settle window
it accepts a reference only if all of:

| check | threshold | why |
| --- | --- | --- |
| usable samples | == `calib_window` (20) | need a full window |
| max raw `|(ω_x, ω_y)|` in window | ≤ 0.15 rad/s | the body must be stationary |
| `|mean(unit samples)|` | ≥ 0.999 | direction-free coherence |
| max angle to the mean | ≤ 0.03 rad (1.7°) | the stance must have stopped moving |
| reference `sin(tilt)` | ≤ 0.15 and `g_z < 0` | a slope start must not be absorbed |

On acceptance the reference `r` is the normalized mean, and the frozen rotation
`R` is the **shortest proper rotation** taking `r` to `(0,0,−1)` (Rodrigues,
checked for orthonormality, `det = +1` and `R·r = target`). On refusal the
controller runs as the uncalibrated baseline and reports the reason in
`describe()` and in every debug record. Calibration is attempted once; `reset()`
drops it so a new episode recalibrates.

**Normal operation.** The baseline receives a copy of the proprio vector whose
gravity slice is replaced by `R·g_raw` and nothing else. `R` is orthonormal, so
the norm the baseline validates is unchanged. All baseline feedback and gating
(tilt deadband/max/abort, leg levelling, authority, yaw assist) then acts on
deviation from the neutral stance. The caller's array is never written; the
angular-velocity slice is never rewritten, so the baseline's rate gate and yaw
readings stay raw.

**Independent absolute envelope.** Computed from the raw gravity in parallel,
never from the rotated copy:

- `sin(tilt)_abs` below `abs_tilt_deadband` (0.18, ~10.4°): wheel request passed
  through bit-exact; the profile is a no-op on the request.
- Between 0.18 and 0.30: the **wheel request only** is scaled by a linear ramp
  before the baseline sees it. Legs and arms are bit-identical.
- At or above `ABS_TILT_ABORT` (0.30, the baseline's own `TILT_ABORT`), or with
  gravity outside the downward hemisphere, or on unusable proprio: the baseline
  is fed the **raw, unrotated** proprio plus a zero wheel request, which trips
  its native immediate-stop path; the baseline abort flag separately depends on its gravity checks and filtered tilt. A trip latches the raw
  frame for 50 calls and until the filtered absolute tilt is back under the
  deadband.

So a dangerous absolute posture is never accepted because calibration happened
to match it: the reference can shift the trip point by at most the reference
tilt itself (≤ 0.15 by construction, 0.0633 for the recorded stance), and above
0.30 the reference is bypassed entirely.

**Diagnostics.** `debug["neutral_reference"]` always carries both frames:
`absolute` (raw `sin(tilt)`, degrees, filtered value, unit gravity, raw tilt
rate, gate, danger flag and reasons) and `relative` (what the baseline saw).
`debug["tilt"]["frame"]` states which frame the baseline gains ran on, and
`debug["tilt"]["absolute_sin_tilt"]` repeats the true value inline. True tilt is
never hidden or replaced.

## Preserved from the frozen baseline

Arm entries are the incoming bytes. An all-zero incoming wheel request is
forwarded untouched and still stops immediately. Float32 representability,
finiteness, gravity-norm `[0.5, 1.5]`, inverted-gravity abort, non-`None` action
clip rejection and leg soft-limit clipping are all the baseline's, unmodified.
No gain, cap, slew rate, physics, actuator, asset, reward or termination is
changed. Runtime inputs remain the public 84-vector and the static schema; no
contact, object or root truth is read.

## Acceptance criteria (for root's CPU and GPU runs)

CPU:

1. **Bit-exact parity when disabled.** `enable_neutral_reference=False`
   reproduces the baseline output for a recorded 500-frame sequence, max error 0.
2. **Bit-exact parity while healthy and calibrated.** With calibration accepted
   and absolute tilt under 0.18, the wheel request is forwarded unscaled; the
   only difference from the baseline is the rotated gravity the baseline sees.
3. **Refusal path.** A synthetic settle window that is moving
   (spread > 0.03 rad), rotating (> 0.15 rad/s), short (< 20 samples), or tilted
   past 0.15 leaves `calibrated == False`, records a reason, and produces
   baseline-identical output.
4. **Absolute override.** Injected gravity at `sin(tilt) ≥ 0.30`, `(0,0,+1)` and
   `(1,0,0)` must abort and stop even when the reference-relative tilt is small
   — including a reference deliberately calibrated near the limit.
5. **Reference correctness.** `R·r == (0,0,−1)` within 1e-9, `RᵀR = I`,
   `det R = +1`, and `‖R·g‖ == ‖g‖`, so the baseline's norm check is unaffected.
6. **No caller mutation.** The supplied `action` and `proprio` arrays are
   unchanged after `apply`; arm bytes and zero-wheel stop preserved.

GPU (paired, same seed, against the recorded baseline runs):

7. **The nominal-pose gate disappears.** For a calibrated healthy run,
   `gates.combined == 1.0` for the great majority of post-settle calls and
   `authority` stays at 1.0 instead of sinking to 0.30. Failure to reproduce
   this means the settled stance is not constant and a single frozen reference
   is the wrong model.
8. **Turn activity increases.** `turn` at a 0.6 request should apply a
   differential near the 0.30 cap instead of 0.0879, and post-settle yaw should
   exceed the 0.526° of `turn_stabilized_seed42_01` by a clear margin. Whether
   that is *useful* steering is a separate question; if yaw stays under ~1°, the
   correct report is still **stalled**.
9. **Survival is not lost.** The paired turn and visual runs must still reach
   500 steps without an illegal contact. If the un-gated authority reintroduces
   the step-322 style failure, the honest conclusion is that the baseline's
   throttling — not its tilt model — was what kept the earlier run alive, and
   the profile should be rejected.
10. **Visual approach.** `visual_stabilized` with the neutral profile should
    hold ALIGN at least as long as the current run does at the 0.30 authority
    floor, with more command headroom. Regression in alignment is a rejection.

## Limitations

- Unverified: no simulation run exists for this profile.
- Reference-relative tilt is **not** true tilt. Any report must quote the
  `absolute` block, which is why both are always emitted.
- The leg-levelling downhill direction is computed in the rotated frame, so
  corner selection is approximate to within the reference tilt (~3.6° for the
  recorded stance) and can mis-select near the deadband. The correction stays
  raise-only and capped at 0.12 units, so a mis-selection extends a slightly
  wrong corner rather than lowering anything.
- The rotation fixes no yaw; it is a stance offset, not an orientation estimate.
- A genuine slope start up to `calib_max_reference_tilt` would be absorbed into
  the reference. Only the absolute envelope still sees it.
- A single frozen reference cannot track a stance that changes with load, arm
  extension or a commanded crouch; such changes appear as reference-relative
  tilt and will gate, which is intended but is not the same as tracking.
- After an absolute-envelope trip the robot is held stopped for the 50-call
  latch even if tilt recovers immediately.
- Thresholds are chosen from one seed's recorded runs and are not swept.

## 本地实测补充（2026-09-10）

主代理已完成 18 项 CPU 检查，以及实际记录的初始 100 帧观测回放：校准接受末 20 帧，参考 sin(tilt)=0.05082。真实 ±0.60 差速试验存活 500 步，静止期后偏航约 1.366°，末尾 authority=1、差速=0.30。与视觉接近组合运行 1500 步未终止，偏航约 −1.227°，仍在 ALIGN，累计分为 0。参考校准消除了正常姿态持续限速，但没有打通有效转向或接近；上述设计预测应与这些实测一起阅读。见 [完整实验](../docs/TASK_B_EXPERIMENTS.md) 与 [CPU 检查](../results/task_b_neutral_profile_cpu_audit.json)。

发布前输入校验补充：绝对倾角 gate=0 时，原始巨大 wheel 请求也必须交由基类拒绝，不能先乘零隐藏溢出。修复后 CPU 共21项通过，其中500帧有界输入与修复前逐位一致。已跑GPU的源文件SHA `1bdadc29…` 快照保存在 `results/task_b_bootstrap/source_snapshots/`；此边界补丁没有重跑GPU，也不改变已记录回合的结果。
