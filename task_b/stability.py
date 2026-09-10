"""Conservative closed-loop stability wrapper for Task B B2wPiper experiments.

This module owns nothing but a *filter* that sits between whatever policy is
driving the robot (the open-loop bootstrap in :mod:`task_b.control`, or a future
visual approach controller) and the environment's action manager:

    controller = StabilityController(schema, observation_joint_names, defaults)
    safe_action = controller.apply(incoming_action, proprio)

It reads only the public proprioception vector and the static joint/action
schema. Root pose, object poses and contact truth are *never* inputs; they are
evidence used offline to choose the constants below.

What it does, in order of confidence:

1. **Direction-independent slowing and slew limiting of the wheels.** Wheel
   velocity requests are split into a common (drive) part, a left/right
   differential (steer) part and a per-wheel residual; each is capped, rate
   limited and multiplied by a tilt/tilt-rate gate plus an adaptive authority
   that shrinks every time the gate fires. No sign knowledge is required.
2. **Optional magnitude-only yaw assist.** When enabled it grows the *caller's
   own* differential direction while the base is quiet and the yaw rate is
   under-responding, and retracts it immediately when the gate fires. It never
   chooses a turn direction, so it needs no verified yaw sign.
3. **Raise-only leg levelling.** The in-base-frame horizontal component of
   ``projected_gravity`` points downhill, so the corners on that side are the
   low ones; those corners are extended slightly. Corrections are clipped to be
   raise-only, so the controller never walks toward the crouch posture that
   produced illegal contacts in the recorded runs.

Every output is hard clipped: legs through the *actual* defaults, term scale and
soft position limits, wheels through a finite constant cap (continuous wheels
have no usable soft limits). Arm entries are passed through untouched.

None of this is proven. The constants are hypotheses fitted to the recorded
``task_b_initial`` runs and must be tested against an uncorrected same-seed run.
See ``task_b/STABILITY_DESIGN.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import ARM_TERM, LEG_JOINT_PATTERN, LEG_TERM, WHEEL_TERM, wheel_side

#: Calls returned completely unmodified so the recorded initial condition (the
#: official settle to the default stance, base z ~0.52 m after ~100 calls) is
#: reproduced bit-for-bit before any correction can act.
SETTLE_CALLS = 100

# ---------------------------------------------------------------------------
# Wheel limits. Evidence from ATEC_Experiments_20260910/task_b_initial:
#   * all four wheels at +0.10 normalized (0.5 rad/s) survived 500 steps;
#   * right +0.10 / left -0.10 survived but yawed only ~0.53 deg in 8 s;
#   * a 0.6 differential produced an illegal contact at step 322 (6.44 s).
# The caps below stay at or under half of the one differential magnitude that is
# known to fail, and only modestly above the one drive magnitude known to hold.
# ---------------------------------------------------------------------------
WHEEL_MAX_COMMON = 0.15      #: normalized drive command per wheel (0.75 rad/s)
WHEEL_MAX_DIFF = 0.30        #: normalized half-differential (1.5 rad/s split)
WHEEL_MAX_RESIDUAL = 0.05    #: per-wheel asymmetry a caller may add on top
WHEEL_MAX_ABS = 0.35         #: final hard clip on any single wheel command
WHEEL_SLEW_COMMON = 0.002    #: normalized per call (0.10/s, ~ the bootstrap ramp)
WHEEL_SLEW_DIFF = 0.003      #: normalized per call (0.15/s)
WHEEL_SLEW_RESIDUAL = 0.004
ABORT_SLEW_GAIN = 4.0        #: faster retraction while aborting

# ---------------------------------------------------------------------------
# Tilt gating. `tilt` is |(g_x, g_y)| of the unit projected gravity, i.e.
# sin(total tilt from vertical); `tilt_rate` is |(w_x, w_y)| of the base angular
# velocity. Both are low-pass filtered before use.
# ---------------------------------------------------------------------------
TILT_DEADBAND = 0.06         #: ~3.4 deg: below this the gate is fully open
TILT_MAX = 0.20              #: ~11.5 deg: gate closed, commands driven to zero
TILT_ABORT = 0.30            #: ~17 deg: hard abort, wheels ramped to zero fast
TILT_RATE_DEADBAND = 0.5     #: rad/s
TILT_RATE_MAX = 1.5          #: rad/s
GRAVITY_TAU = 0.15           #: s, low-pass time constant for projected gravity
ANGVEL_TAU = 0.20            #: s, low-pass time constant for base angular rates

#: Adaptive authority: multiplies the caps. Every gated call shrinks it, quiet
#: calls slowly restore it. This makes repeated near-misses progressively more
#: conservative instead of letting the robot re-enter the same command.
AUTHORITY_DECAY = 0.98
AUTHORITY_RECOVER = 0.002
AUTHORITY_FLOOR = 0.30

# ---------------------------------------------------------------------------
# Leg levelling uses a common synthetic direction: thigh +0.20 / calf -0.50
# rad. The failed crouch had that thigh delta only in front (rear delta was 0),
# and additionally moved the hips toward zero. This vector is not that exact
# action. Direct B2w USD FK confirms its negative direction extends each leg
# near the default stance; this is geometric evidence, not dynamic stability.
# ---------------------------------------------------------------------------
LEG_LOWER_DIRECTION_RAD = {"hip": 0.0, "thigh": 0.20, "calf": -0.50}
LEG_MAX_UNITS = 0.12         #: <= 0.024 rad thigh, 0.06 rad calf per corner
LEG_SLEW_UNITS = 0.004       #: units per call (0.2 units/s)
LEG_KP_UNITS_PER_TILT = 1.2  #: units of raise per unit of sin(tilt) beyond the deadband

#: Yaw assist (opt-in). Magnitude only: it scales the caller's own differential.
YAW_ASSIST_STEP = 0.002      #: normalized per call added while under-responding
YAW_RATE_DEADBAND = 0.15     #: rad/s; below this the turn counts as stalled
YAW_ASSIST_COOLDOWN_CALLS = 50

LEG_MODES = ("raise_low_corners", "off")


class StabilityError(ValueError):
    """The controller cannot be built or driven against the live schema."""


@dataclass(frozen=True)
class ProprioLayout:
    """Index layout of the official Task B proprio group.

    Term order is taken verbatim from the official
    ``task_base/envs_base_cfg.py::ProprioObservationsCfg``: ``base_lin_vel(3)``,
    ``base_ang_vel(3)``, ``velocity_commands(3)``, ``projected_gravity(3)``,
    ``joint_pos(N)``, ``joint_vel(N)``, ``actions(A)``; concatenated, corruption
    disabled. With N = A = 24 this is the public 84-vector.
    """

    num_joints: int
    num_actions: int

    @property
    def base_lin_vel(self) -> slice:
        return slice(0, 3)

    @property
    def base_ang_vel(self) -> slice:
        return slice(3, 6)

    @property
    def velocity_commands(self) -> slice:
        return slice(6, 9)

    @property
    def projected_gravity(self) -> slice:
        return slice(9, 12)

    @property
    def joint_pos(self) -> slice:
        return slice(12, 12 + self.num_joints)

    @property
    def joint_vel(self) -> slice:
        return slice(12 + self.num_joints, 12 + 2 * self.num_joints)

    @property
    def actions(self) -> slice:
        start = 12 + 2 * self.num_joints
        return slice(start, start + self.num_actions)

    @property
    def size(self) -> int:
        return 12 + 2 * self.num_joints + self.num_actions

    def to_dict(self) -> dict:
        return {
            "expected_size": self.size,
            "base_lin_vel": [0, 3], "base_ang_vel": [3, 6], "velocity_commands": [6, 9],
            "projected_gravity": [9, 12],
            "joint_pos_rel": [self.joint_pos.start, self.joint_pos.stop],
            "joint_vel_rel": [self.joint_vel.start, self.joint_vel.stop],
            "last_actions": [self.actions.start, self.actions.stop],
            "source": "official ProprioObservationsCfg term order (concatenated, corruption disabled)",
        }


def _ema_alpha(dt: float, tau: float) -> float:
    return float(dt / (tau + dt))


def _ramp_gate(value: float, deadband: float, ceiling: float) -> float:
    """1.0 below `deadband`, linearly to 0.0 at `ceiling`."""
    if value <= deadband:
        return 1.0
    if value >= ceiling:
        return 0.0
    return float((ceiling - value) / (ceiling - deadband))


def _slew(previous: float, target: float, step: float) -> float:
    delta = target - previous
    if delta > step:
        return previous + step
    if delta < -step:
        return previous - step
    return target


class StabilityController:
    """Cautious closed-loop wrapper around an incoming normalized action.

    Parameters
    ----------
    schema:
        The live :class:`task_b.control.ActionSchema` (or anything exposing the
        same ``term``/``joint_index``/``default_joint_pos``/
        ``soft_joint_pos_limits``/``total_dim`` interface).
    observation_joint_names:
        Joint order of the proprio ``joint_pos``/``joint_vel`` blocks
        (``preserve_order=True`` over the articulation).
    default_joint_positions:
        Default positions aligned with ``observation_joint_names``; used to
        rebuild absolute joint angles from the relative proprio block.
    dt:
        Control period in seconds (0.02 for the official task).
    """

    def __init__(self, schema, observation_joint_names, default_joint_positions, dt: float = 0.02,
                 *, leg_mode: str = "raise_low_corners", yaw_assist_max: float = 0.0,
                 wheel_max_common: float = WHEEL_MAX_COMMON, wheel_max_diff: float = WHEEL_MAX_DIFF,
                 settle_calls: int = SETTLE_CALLS, strict: bool = True):
        if leg_mode not in LEG_MODES:
            raise StabilityError(f"leg_mode must be one of {list(LEG_MODES)}, got {leg_mode!r}")
        if not np.isfinite(dt) or dt <= 0:
            raise StabilityError(f"dt must be positive and finite, got {dt!r}")
        if not np.isfinite(yaw_assist_max) or yaw_assist_max < 0:
            raise StabilityError("yaw_assist_max must be finite and >= 0")
        for name, value in (("wheel_max_common", wheel_max_common), ("wheel_max_diff", wheel_max_diff)):
            if not np.isfinite(value) or value < 0 or value > WHEEL_MAX_ABS:
                raise StabilityError(f"{name} must be finite in [0, {WHEEL_MAX_ABS}], got {value!r}")
        if int(settle_calls) != settle_calls or settle_calls < 0:
            raise StabilityError("settle_calls must be a nonnegative integer")

        self.schema = schema
        self.dt = float(dt)
        self.leg_mode = leg_mode
        self.yaw_assist_max = float(yaw_assist_max)
        self.wheel_max_common = float(wheel_max_common)
        self.wheel_max_diff = float(wheel_max_diff)
        self.settle_calls = int(settle_calls)
        self.strict = bool(strict)
        self.warnings: list[str] = []

        self.observation_joint_names = tuple(str(name) for name in observation_joint_names)
        if len(self.observation_joint_names) != 24 or len(set(self.observation_joint_names)) != 24:
            raise StabilityError("Expected 24 distinct observation joint names")
        self.observation_defaults = np.asarray(default_joint_positions, dtype=np.float64).reshape(-1)
        if self.observation_defaults.shape != (len(self.observation_joint_names),):
            raise StabilityError(
                f"default_joint_positions {self.observation_defaults.shape} does not match "
                f"{len(self.observation_joint_names)} observation joint names"
            )
        if not np.isfinite(self.observation_defaults).all():
            raise StabilityError("Non-finite default joint positions")

        self.layout = ProprioLayout(num_joints=len(self.observation_joint_names),
                                    num_actions=int(schema.total_dim))

        leg = schema.term(LEG_TERM)
        wheel = schema.term(WHEEL_TERM)
        arm = schema.term(ARM_TERM)
        if callable(getattr(schema, "validate", None)):
            schema.validate()
        if leg.mode != "position" or not leg.use_default_offset or wheel.mode != "velocity":
            raise StabilityError("Requires default-offset position legs and velocity wheels")
        if leg.clip is not None or wheel.clip is not None:
            raise StabilityError("Non-None leg/wheel action clips are not supported")
        if not np.isfinite(leg.scale) or leg.scale <= 0:
            raise StabilityError("Leg scale must be positive and finite")
        self._leg_slice = slice(leg.start, leg.stop)
        self._arm_slice = slice(arm.start, arm.stop)
        self._leg_scale = float(leg.scale)

        # Cross-check the two default sources; disagreement is reported, not fatal.
        by_name = dict(zip(self.observation_joint_names, self.observation_defaults))
        for joint in leg.joint_names:
            if joint not in by_name:
                raise StabilityError(f"Leg joint '{joint}' missing from observation_joint_names")
            expected = float(schema.default_joint_pos[schema.joint_index(joint)])
            if abs(by_name[joint] - expected) > 1e-3:
                self.warnings.append(
                    f"default mismatch for {joint}: observation {by_name[joint]:.6f} vs schema {expected:.6f}"
                )

        # Static hard clip for the leg term, from the real defaults/scale/limits.
        lower, upper = [], []
        for joint in leg.joint_names:
            index = schema.joint_index(joint)
            low, high = (float(x) for x in schema.soft_joint_pos_limits[index])
            default = float(schema.default_joint_pos[index])
            if not (np.isfinite(low) and np.isfinite(high)) or low > high:
                raise StabilityError(f"Leg joint '{joint}' has unusable soft limits ({low}, {high})")
            lower.append((low - default) / self._leg_scale)
            upper.append((high - default) / self._leg_scale)
        self._leg_action_lower = np.asarray(lower, dtype=np.float64)
        self._leg_action_upper = np.asarray(upper, dtype=np.float64)
        if not np.isfinite(self._leg_action_lower).all() or not np.isfinite(self._leg_action_upper).all():
            raise StabilityError("Leg scale produces non-finite normalized limits")

        # One synthetic corner direction, converted to actual normalized units.
        self._leg_corner_names = ("FR", "FL", "RR", "RL")
        self._leg_corner_index = {corner: i for i, corner in enumerate(self._leg_corner_names)}
        self._leg_unit = np.zeros(leg.dim, dtype=np.float64)
        self._leg_corner_of_entry = np.full(leg.dim, -1, dtype=np.int64)
        for offset, joint in enumerate(leg.joint_names):
            match = LEG_JOINT_PATTERN.match(joint)
            if match is None:
                raise StabilityError(f"Unrecognized leg joint '{joint}'")
            corner, link = match.group("corner"), match.group("link")
            if corner not in self._leg_corner_index:
                raise StabilityError(f"Unexpected leg corner '{corner}'")
            self._leg_corner_of_entry[offset] = self._leg_corner_index[corner]
            self._leg_unit[offset] = LEG_LOWER_DIRECTION_RAD[link] / self._leg_scale

        # Wheel geometry: side signs match task_b.control ('right' positive).
        self._wheel_indices = np.asarray(
            [wheel.start + offset for offset in range(wheel.dim)], dtype=np.int64)
        self._wheel_names = tuple(wheel.joint_names)
        self._wheel_sign = np.asarray(
            [1.0 if wheel_side(name) == "right" else -1.0 for name in self._wheel_names],
            dtype=np.float64)
        if not np.any(self._wheel_sign > 0) or not np.any(self._wheel_sign < 0):
            raise StabilityError(f"Wheel joints {list(self._wheel_names)} do not span both sides")

        self.debug: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear filters, applied commands and the adaptive authority."""
        self.calls = 0
        self._gravity_f = np.array([0.0, 0.0, -1.0])
        self._angvel_f = np.zeros(3)
        self._filters_primed = False
        self._common = 0.0
        self._diff = 0.0
        self._residual = np.zeros(len(self._wheel_names))
        self._leg_units = np.zeros(len(self._leg_corner_names))
        self._leg_correction = np.zeros(self._leg_unit.size)
        self._leg_clipped = 0
        self._request_summary = (0.0, 0.0, 0.0)
        self.authority = 1.0
        self._assist = 0.0
        self._cooldown = 0
        self._gate_events = 0
        self._abort_events = 0
        # Running least squares of yaw rate against applied differential, so the
        # untested steering sign can be read off a run instead of guessed.
        self._yaw_sum_dd = 0.0
        self._yaw_sum_dy = 0.0
        self.debug = {"call": 0, "state": "reset", "errors": []}

    # ------------------------------------------------------------------- main
    def apply(self, action, proprio) -> np.ndarray:
        """Return a finite normalized action of the same size, arms untouched."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.schema.total_dim:
            raise StabilityError(f"Action of size {action.size} != schema total {self.schema.total_dim}")
        if not np.isfinite(action).all():
            raise StabilityError("Incoming action contains non-finite entries")
        if np.any(np.abs(action) > np.finfo(np.float32).max):
            raise StabilityError("Incoming action is not representable as finite float32")
        arm_in = action[self._arm_slice].copy()

        self.calls += 1
        errors: list[str] = []
        measurement = self._read_proprio(proprio, errors)
        if measurement is None:
            gravity, angvel = self._gravity_f, self._angvel_f
        else:
            gravity, angvel = self._filter(*measurement)

        # sin(tilt) alone aliases upright and inverted poses. Keep the existing
        # filtered gains for normal motion, but reject an upward gravity vector
        # or a near-zero EMA caused by opposing orientations.
        gravity_guard = (gravity[2] >= 0.0 or np.linalg.norm(gravity) < 0.5
                         or (measurement is not None and measurement[0][2] >= 0.0))
        if gravity_guard:
            errors.append("gravity direction outside downward hemisphere or degenerate filtered gravity")

        tilt_xy = gravity[:2]
        tilt = float(np.hypot(tilt_xy[0], tilt_xy[1]))
        tilt_rate = float(np.hypot(angvel[0], angvel[1]))
        yaw_rate = float(angvel[2])

        if self.calls <= self.settle_calls:
            # Bootstrap settle is reproduced exactly; filters still warm up.
            out = action.astype(np.float32)
            self.debug = self._make_debug("settle", tilt, tilt_xy, tilt_rate, yaw_rate,
                                          1.0, 1.0, action, out, errors)
            return out

        gate_tilt = _ramp_gate(tilt, TILT_DEADBAND, TILT_MAX)
        gate_rate = _ramp_gate(tilt_rate, TILT_RATE_DEADBAND, TILT_RATE_MAX)
        if errors:
            gate_tilt = gate_rate = 0.0
        gate = gate_tilt * gate_rate
        aborting = tilt >= TILT_ABORT or bool(errors)
        if aborting:
            self._abort_events += 1

        if gate < 1.0:
            self._gate_events += 1
            self.authority = max(AUTHORITY_FLOOR, self.authority * AUTHORITY_DECAY)
            self._cooldown = YAW_ASSIST_COOLDOWN_CALLS
        else:
            self.authority = min(1.0, self.authority + AUTHORITY_RECOVER)
            self._cooldown = max(0, self._cooldown - 1)

        out = action.copy()
        out[self._wheel_indices] = self._wheels(action, gate, yaw_rate, aborting)
        out[self._leg_slice] = self._legs(action[self._leg_slice], tilt_xy, tilt, gate, aborting)
        out[self._arm_slice] = arm_in  # arms are pass-through by contract

        if not np.isfinite(out).all():
            raise StabilityError("Stability controller produced a non-finite action")
        out = out.astype(np.float32)
        if not np.isfinite(out).all():
            raise StabilityError("Stability output is not representable as finite float32")
        self.debug = self._make_debug("abort" if aborting else "active", tilt, tilt_xy, tilt_rate,
                                      yaw_rate, gate_tilt, gate_rate, action, out, errors)
        return out

    # -------------------------------------------------------------- internals
    def _read_proprio(self, proprio, errors: list[str]):
        proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio.size != self.layout.size:
            message = f"proprio size {proprio.size} != expected {self.layout.size}"
            if self.strict:
                raise StabilityError(message)
            errors.append(message)
            return None
        gravity = proprio[self.layout.projected_gravity]
        angvel = proprio[self.layout.base_ang_vel]
        if not np.isfinite(gravity).all() or not np.isfinite(angvel).all():
            if self.strict:
                raise StabilityError("Non-finite projected gravity or base angular velocity")
            errors.append("non-finite proprio slice")
            return None
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or not 0.5 <= norm <= 1.5:
            # A unit gravity direction is expected; anything else is unusable.
            if self.strict:
                raise StabilityError(f"Projected gravity has implausible norm {norm:.4f}")
            errors.append(f"gravity norm {norm:.4f}")
            return None
        return gravity / norm, angvel

    def _filter(self, gravity, angvel):
        if not self._filters_primed:
            self._gravity_f = np.array(gravity, dtype=np.float64)
            self._angvel_f = np.array(angvel, dtype=np.float64)
            self._filters_primed = True
        else:
            ag = _ema_alpha(self.dt, GRAVITY_TAU)
            aw = _ema_alpha(self.dt, ANGVEL_TAU)
            self._gravity_f += ag * (np.asarray(gravity) - self._gravity_f)
            self._angvel_f += aw * (np.asarray(angvel) - self._angvel_f)
        return self._gravity_f, self._angvel_f

    def _wheels(self, action, gate: float, yaw_rate: float, aborting: bool) -> np.ndarray:
        """Cap, gate and slew-limit the wheel request; return the applied commands."""
        request = action[self._wheel_indices]
        if np.all(request == 0.0):
            # An upstream visual controller uses zero for loss of target or an
            # explicit stop. Do not reintroduce blind motion with slew memory.
            self._common = self._diff = self._assist = 0.0
            self._residual[:] = 0.0
            self._request_summary = (0.0, 0.0, 0.0)
            return np.zeros_like(request)
        common_req = float(np.mean(request))
        right = request[self._wheel_sign > 0]
        left = request[self._wheel_sign < 0]
        diff_req = 0.5 * float(np.mean(right) - np.mean(left))
        residual_req = request - common_req - self._wheel_sign * diff_req

        authority = self.authority
        headroom = gate * authority
        assist = self._update_assist(diff_req, gate, yaw_rate)
        diff_cmd = float(np.sign(diff_req) * (abs(diff_req) + assist)) if diff_req != 0.0 else 0.0

        if aborting:
            common_t = diff_t = 0.0
            residual_t = np.zeros_like(residual_req)
            slew = ABORT_SLEW_GAIN
        else:
            common_t = float(np.clip(common_req, -self.wheel_max_common * headroom,
                                     self.wheel_max_common * headroom))
            diff_t = float(np.clip(diff_cmd, -self.wheel_max_diff * headroom,
                                   self.wheel_max_diff * headroom))
            residual_t = np.clip(residual_req, -WHEEL_MAX_RESIDUAL * headroom,
                                 WHEEL_MAX_RESIDUAL * headroom)
            slew = 1.0

        self._common = _slew(self._common, common_t, WHEEL_SLEW_COMMON * slew)
        self._diff = _slew(self._diff, diff_t, WHEEL_SLEW_DIFF * slew)
        for i in range(self._residual.size):
            self._residual[i] = _slew(float(self._residual[i]), float(residual_t[i]),
                                      WHEEL_SLEW_RESIDUAL * slew)

        # Diagnostic only: accumulate d(yaw)/d(differential) so the steering sign
        # can be read from a real run instead of assumed.
        self._yaw_sum_dd += self._diff * self._diff
        self._yaw_sum_dy += self._diff * yaw_rate

        self._request_summary = (common_req, diff_req, float(np.max(np.abs(residual_req))))
        return np.clip(self._common + self._wheel_sign * self._diff + self._residual,
                       -WHEEL_MAX_ABS, WHEEL_MAX_ABS)

    def _update_assist(self, diff_req: float, gate: float, yaw_rate: float) -> float:
        """Grow the caller's own differential magnitude while a turn is stalled."""
        if self.yaw_assist_max <= 0.0 or abs(diff_req) < 1e-6:
            self._assist = max(0.0, self._assist - YAW_ASSIST_STEP * ABORT_SLEW_GAIN)
            return self._assist
        quiet = gate >= 0.999 and self._cooldown == 0
        stalled = abs(yaw_rate) < YAW_RATE_DEADBAND
        if quiet and stalled:
            self._assist = min(self.yaw_assist_max * self.authority, self._assist + YAW_ASSIST_STEP)
        elif not quiet:
            self._assist = max(0.0, self._assist - YAW_ASSIST_STEP * ABORT_SLEW_GAIN)
        return self._assist

    def _legs(self, leg_request: np.ndarray, tilt_xy, tilt: float, gate: float,
              aborting: bool) -> np.ndarray:
        """Add a raise-only levelling correction, then hard clip to real limits."""
        target = np.zeros_like(self._leg_units)
        if self.leg_mode == "raise_low_corners" and not aborting:
            # projected_gravity is R^T (0,0,-1): its horizontal part points
            # downhill in the base frame, so a positive x component means the
            # front corners are the low ones.
            gx, gy = float(tilt_xy[0]), float(tilt_xy[1])
            excess = max(0.0, tilt - TILT_DEADBAND)
            if tilt > 1e-9 and excess > 0.0:
                gain = LEG_KP_UNITS_PER_TILT * excess / tilt
                front_low, rear_low = max(0.0, gx) * gain, max(0.0, -gx) * gain
                left_low, right_low = max(0.0, gy) * gain, max(0.0, -gy) * gain
                for corner, index in self._leg_corner_index.items():
                    longitudinal = front_low if corner[0] == "F" else rear_low
                    lateral = left_low if corner[1] == "L" else right_low
                    # Negative units = extension = raise this corner.
                    target[index] = -min(LEG_MAX_UNITS, longitudinal + lateral)
            target *= gate

        slew = LEG_SLEW_UNITS * (ABORT_SLEW_GAIN if aborting else 1.0)
        for index in range(target.size):
            self._leg_units[index] = _slew(float(self._leg_units[index]), float(target[index]), slew)
        self._leg_units = np.clip(self._leg_units, -LEG_MAX_UNITS, 0.0)

        correction = self._leg_unit * self._leg_units[self._leg_corner_of_entry]
        commanded = np.asarray(leg_request, dtype=np.float64) + correction
        clipped = np.clip(commanded, self._leg_action_lower, self._leg_action_upper)
        self._leg_clipped = int(np.count_nonzero(np.abs(clipped - commanded) > 1e-9))
        self._leg_correction = correction
        return clipped

    def _make_debug(self, state, tilt, tilt_xy, tilt_rate, yaw_rate, gate_tilt, gate_rate,
                    request, applied, errors) -> dict:
        wheel_request = request[self._wheel_indices]
        wheel_applied = applied[self._wheel_indices]
        summary = getattr(self, "_request_summary", (0.0, 0.0, 0.0))
        yaw_slope = (self._yaw_sum_dy / self._yaw_sum_dd) if self._yaw_sum_dd > 1e-9 else None
        return {
            "call": self.calls,
            "state": state,
            "errors": list(errors),
            "warnings": list(self.warnings),
            "tilt": {
                "sin_tilt": tilt,
                "tilt_deg": float(np.degrees(np.arctan2(tilt, -self._gravity_f[2]))),
                "gravity_xy_filtered": [float(tilt_xy[0]), float(tilt_xy[1])],
                "tilt_rate_rad_s": tilt_rate,
                "yaw_rate_rad_s": yaw_rate,
            },
            "gates": {
                "tilt": float(gate_tilt), "rate": float(gate_rate),
                "combined": float(gate_tilt * gate_rate), "authority": float(self.authority),
                "gate_events": self._gate_events, "abort_events": self._abort_events,
                "assist": float(self._assist), "assist_cooldown": int(self._cooldown),
            },
            "wheels": {
                "joint_names": list(self._wheel_names),
                "requested": [float(x) for x in wheel_request],
                "applied": [float(x) for x in wheel_applied],
                "requested_common": float(summary[0]),
                "requested_diff": float(summary[1]),
                "requested_residual_max": float(summary[2]),
                "applied_common": float(self._common),
                "applied_diff": float(self._diff),
                "saturated": bool(np.any(np.abs(wheel_applied) >= WHEEL_MAX_ABS - 1e-9)),
                "yaw_rate_per_diff_estimate": None if yaw_slope is None else float(yaw_slope),
            },
            "legs": {
                "corner_units": {corner: float(self._leg_units[index])
                                 for corner, index in self._leg_corner_index.items()},
                "correction_normalized": [float(x) for x in getattr(
                    self, "_leg_correction", np.zeros(self._leg_unit.size))],
                "entries_clipped_by_limits": int(getattr(self, "_leg_clipped", 0)),
                "action_lower": self._leg_action_lower.tolist(),
                "action_upper": self._leg_action_upper.tolist(),
            },
            "arm_preserved": bool(np.array_equal(
                np.asarray(applied[self._arm_slice], dtype=np.float32),
                np.asarray(request[self._arm_slice], dtype=np.float32))),
        }

    def last_debug(self) -> dict:
        """Copy of the most recent per-call debug record."""
        return dict(self.debug)

    def describe(self) -> dict:
        """JSON-serializable description of the controller and its hypotheses."""
        return {
            "controller": "StabilityController",
            "status": "experimental candidate; untested in simulation",
            "dt": self.dt,
            "settle_calls": self.settle_calls,
            "settle_behaviour": "first calls returned byte-identical to the incoming action so the "
                                "recorded initial condition is preserved",
            "inputs": "public proprio observation and the static joint/action schema only",
            "proprio_layout": self.layout.to_dict(),
            "leg_mode": self.leg_mode,
            "strict": self.strict,
            "wheel": {
                "joint_names": list(self._wheel_names),
                "action_indices": [int(i) for i in self._wheel_indices],
                "side_sign": {name: float(sign) for name, sign in
                              zip(self._wheel_names, self._wheel_sign)},
                "max_common": self.wheel_max_common,
                "max_diff": self.wheel_max_diff,
                "max_residual": WHEEL_MAX_RESIDUAL,
                "max_abs": WHEEL_MAX_ABS,
                "slew_per_call": {"common": WHEEL_SLEW_COMMON, "diff": WHEEL_SLEW_DIFF,
                                  "residual": WHEEL_SLEW_RESIDUAL, "abort_gain": ABORT_SLEW_GAIN},
                "yaw_assist_max": self.yaw_assist_max,
                "explicit_zero_request": "all four requested wheel actions zero clears slew/assist and stops immediately",
                "yaw_assist": "magnitude only; grows the caller's own differential while the base is "
                              "quiet and |yaw rate| < %.2f rad/s. It never picks a turn direction." %
                              YAW_RATE_DEADBAND,
                "note": "continuous wheels have no usable soft position limits; the caps above are "
                        "constants, not limits read from the articulation",
            },
            "gating": {
                "tilt_deadband": TILT_DEADBAND, "tilt_max": TILT_MAX, "tilt_abort": TILT_ABORT,
                "tilt_rate_deadband": TILT_RATE_DEADBAND, "tilt_rate_max": TILT_RATE_MAX,
                "gravity_tau_s": GRAVITY_TAU, "angvel_tau_s": ANGVEL_TAU,
                "authority": {"decay": AUTHORITY_DECAY, "recover": AUTHORITY_RECOVER,
                              "floor": AUTHORITY_FLOOR},
                "definition": "tilt = |horizontal part of the unit projected gravity| = sin(angle "
                              "from vertical); tilt_rate = |(w_x, w_y)| of the base angular velocity",
                "gravity_boundary": "raw norm in [.5,1.5]; upward or degenerate filtered gravity forces abort",
            },
            "legs": {
                "corner_unit_rad": dict(LEG_LOWER_DIRECTION_RAD),
                "corner_unit_normalized": self._leg_unit.tolist(),
                "leg_scale": self._leg_scale,
                "max_units": LEG_MAX_UNITS, "slew_units_per_call": LEG_SLEW_UNITS,
                "kp_units_per_sin_tilt": LEG_KP_UNITS_PER_TILT,
                "sign_convention": "positive units follow a synthetic thigh+.2/calf-.5 direction; "
                                   "negative units extend legs near the default stance in static USD FK",
                "action_lower": self._leg_action_lower.tolist(),
                "action_upper": self._leg_action_upper.tolist(),
                "clip_definition": "(soft_limit - default) / leg_scale from the live articulation",
            },
            "hypotheses": [
                "H1 (wheel slew + caps): capping the drive command at %.2f and the half-differential "
                "at %.2f normalized, with slew limiting, keeps the base within the survivable envelope "
                "seen at 0.10/0.10 while staying below the 0.6 differential that produced an illegal "
                "contact at step 322." % (self.wheel_max_common, self.wheel_max_diff),
                "H2 (tilt gating): filtered sin(tilt) and base roll/pitch rate lead loss of stability "
                "by enough calls that shrinking wheel commands before %.2f rad of tilt prevents the "
                "illegal contact. Direction-independent." % TILT_MAX,
                "H3 (leg levelling): gravity points downhill and USD FK verifies F/L corner positions "
                "and extension direction at the default stance. Whether %.2f synthetic corner units "
                "reduce dynamic tilt still requires a physical test." % LEG_MAX_UNITS,
                "H4 (yaw assist, opt-in): a stalled turn can be escalated safely if the escalation is "
                "slow, magnitude-only and retracted whenever the tilt gate fires.",
            ],
            "limitations": [
                "No simulation run has been executed for this controller; nothing here is verified.",
                "The differential-to-yaw sign is unknown; the controller never chooses a turn "
                "direction and only reports yaw_rate_per_diff_estimate for offline identification.",
                "Static USD FK verifies H3's corner mapping and local extension, not its dynamic "
                "stabilizing effect; compare leg_mode='off' to test that effect.",
                "Raising the low corners raises the centre of mass slightly; the tiny cap is the only "
                "protection against that being harmful.",
                "Arm entries are passed through unchecked, by contract; arm safety is the caller's.",
                "Gains were chosen from five recorded runs at one seed, not fitted or swept.",
            ],
        }
