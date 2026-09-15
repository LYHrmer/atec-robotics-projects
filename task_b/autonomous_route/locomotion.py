"""Low-curvature forward/arc locomotion candidate for the original ATEC-TaskB-B2wPiper.

This is an *independent* module. It does not import, subclass or modify
:mod:`task_b.stability`, :mod:`task_b.stability_profiles`,
:mod:`task_b.visual_approach` or :mod:`task_b.control` beyond the shared static
schema helpers (``WHEEL_TERM``/``LEG_TERM``/``ARM_TERM``/``wheel_side``). It
changes no physics, asset, actuator, reward or termination.

Why a new candidate
-------------------
Recorded ATEC-TaskB-B2wPiper runs at seed 42 (see ``LOCOMOTION_DESIGN.md`` for the
numbers) show a consistent split:

* Same-direction (common) wheel commands *do* move the base. 0.10 normalized
  (0.5 rad/s target) tracked to ~0.19-0.22 rad/s measured and moved +0.2748 m in
  400 steps; root's ``first_reach_seed42_01`` covered ~1.6 m over 1800 steps.
* Differential (yaw) commands do not. +/-0.30 normalized (+/-1.5 rad/s) yielded
  ~1.4 deg of yaw in 500 steps with measured wheel speeds near zero; +/-0.6
  (+/-3 rad/s) tracked on only one diagonal (FL, RR) while the other diagonal
  (FR, RL) was dragged backwards, and terminated on illegal contact at step 322.
  Root's 0.2 differential while driving forward produced almost no yaw either.

The B2w wheels are ``ImplicitActuatorCfg(stiffness=0.0, damping=1.0,
effort_limit_sim=20.0, velocity_limit_sim=50.0)`` (original
``assets/robots/b2w.py``). With zero stiffness the implicit drive is a *damped
velocity source*. Isaac Lab estimates its torque as
``damping * (target - measured)`` and clips that estimate at 20 N*m. It does not
expose the actual PhysX drive torque. The recorded mean speed errors suggest
roughly 0.3 N*m (forward) to 1.5-3 N*m (turns) in this approximation; means alone
do not establish peak torque, saturation, normal load or the cause of a stall.

Hypotheses this module is built to test (all falsifiable, none verified):

H1  Insufficient drive response may contribute to weak yaw authority. Escalating
    the differential *while the wheels are already rolling forward* (an arc)
    may produce measurable yaw where a rotate-in-place did not. Forward rolling
    changes the contact motion; a lower resistance is a hypothesis to test.
H2  The FL/RR vs FR/RL split in ``turn_seed42_03`` may reflect per-wheel normal-load
    asymmetry (diagonal rocking of a stiff four-wheel frame plus the offset
    Piper arm mass), not a uniform actuator lag. Then the wheels that spin are
    the unloaded ones, which produce little traction, and per-wheel tracking
    ratios plus a diagonal asymmetry metric can flag it, but cannot measure load.
H3  Curvature limiting avoids a commanded pure spin in arc mode: keeping
    ``|differential| <= curvature_ratio * |common|`` with ``curvature_ratio <=
    1`` means every wheel keeps the sign of the common command, so this module
    does not command that counter-rotating pattern in arc mode. It does not
    establish physical stability; the optional passthrough mode can request it.

What it does
------------
``apply()`` takes the caller's full 24-entry normalized action, preserves every
non-wheel entry exactly, and rewrites only the wheel slice:

* ``bearing_error=None`` -> **bounded pass-through**. The caller's wheel request
  is kept as-is apart from the physical per-wheel bound and the slew limit. No
  feedback is added and every integrator is cleared, so this path cannot
  silently become a different behaviour.
* ``bearing_error`` supplied -> **arc tracking**. The common (forward) part is
  inferred from the caller's own wheel request; the caller's differential and
  per-wheel residual are discarded and replaced by a bounded yaw correction
  built from the public body yaw rate and the named measured wheel velocities.
* ``enabled=False`` or an all-zero wheel request -> wheels are zeroed on the
  same call and all integrators/slew memory are cleared, so parking, target loss
  and stationary arm phases can never be crept through.

Inputs are the public 84-element proprio vector and the static joint/action
schema only. Root pose, object pose, contact buffers, layout coordinates and any
truth parameter are never read.

Integration order and the existing stability filter
---------------------------------------------------
This module is an *intent generator* and must run **before** any stability
filter. Stacking it under :class:`task_b.stability.StabilityController` as
currently configured will mask it: that filter caps the common command at 0.15
normalized, the half-differential at 0.30, the per-wheel residual at 0.05 and
slews the common term at 0.002/call, all multiplied by an adaptive authority that
decays to a 0.30 floor. A 0.30 common / 0.24 differential arc request would be
delivered as at most 0.15 / 0.30 and as little as 0.045 / 0.09 once the authority
floor is reached. For the paired probe, run this controller *without* that
wrapper and rely on its own tilt/tilt-rate/gravity guards; re-introduce the
wrapper only after the arc gain has been identified. Nothing in this module
edits, wraps or re-tunes ``stability.py``/``stability_profiles.py``.

Nothing here has been run in simulation. No claim is made that the base turns,
that an arc tracks, or that Task B scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import ARM_TERM, LEG_TERM, WHEEL_TERM, wheel_side

# ---------------------------------------------------------------------------
# Wheel command envelope, in normalized action units unless stated otherwise.
# Physical target = schema.term('joint_wheel').scale * normalized (velocity term
# with a zero default-velocity offset), so the physical bound below is the one
# that actually matters and is derived through the live scale.
#
# Evidence used to pick the defaults (all seed 42, official task):
#   * +/-1.5 rad/s per wheel (0.30 normalized) survived 500 steps with no
#     termination, in the *counter-rotating* pattern;
#   * all-wheels +0.5 rad/s survived 500 steps and moved the base;
#   * +/-3 rad/s counter-rotating (0.6 normalized) terminated at step 322.
# The official action is not bounded at 0.6: the wheel actuator declares
# velocity_limit_sim=50 rad/s and effort_limit_sim=20 N*m. MAX_WHEEL_RAD_S below
# is this module's own conservative test cap, not an official limit.
# ---------------------------------------------------------------------------
MAX_WHEEL_RAD_S = 3.0          #: default physical per-wheel bound (0.60 normalized at scale 5.0)
HARD_MAX_WHEEL_RAD_S = 10.0    #: module refuses to be configured above this (actuator declares 50)
COMMON_MAX = 0.30              #: normalized common (forward) command, 1.5 rad/s at scale 5.0
DIFF_MAX = 0.30                #: normalized half-differential ceiling, 1.5 rad/s at scale 5.0
CURVATURE_RATIO = 0.80         #: |diff| <= ratio * |common|; <= 1.0 keeps every wheel's sign
SLEW_PER_CALL = 0.004          #: normalized per call (0.2/s): ~1.5 s from rest to 0.30

#: Forward tracking assist. The damped drive always settles below its target, so
#: a bounded integral raises the target until the torque suffices. Bounded,
#: anti-windup protected and disabled by a stall latch.
FORWARD_ASSIST_MAX = 0.10      #: normalized (0.5 rad/s at scale 5.0)
FORWARD_ASSIST_KI = 1.0        #: normalized per second per normalized shortfall

#: Yaw correction. bearing_error is a body-frame bearing in rad, positive toward
#: +y (left), matching visual_approach.py's arctan2(y, x).
BEARING_KP = 0.20              #: normalized half-differential per rad of bearing
BEARING_TO_YAW_RATE = 1.0      #: desired body yaw rate (rad/s) per rad of bearing
YAW_RATE_MAX = 0.6             #: rad/s ceiling on the desired yaw rate
YAW_KI = 0.20                  #: normalized per second per (rad/s) of yaw-rate error
DIFF_I_MAX = 0.20              #: bound on the yaw integral alone
INTEGRAL_LEAK = 0.02           #: fraction of the integral shed per call when not integrating
BEARING_ABS_MAX = float(np.pi) #: rad; a supplied bearing outside +/-pi is a caller bug

# ---------------------------------------------------------------------------
# Safety guards. These are hard stops, not ramps: ramp gating is the business of
# the separate stability filter. Thresholds sit above the measured nominal
# settled stance (sin(tilt) = 0.0633) so a healthy pose is not gated.
# ---------------------------------------------------------------------------
TILT_STOP = 0.12               #: sin(tilt) ~6.9 deg -> zero wheels; not a stability guarantee
TILT_RATE_STOP = 1.5           #: |(w_x, w_y)| rad/s -> zero wheels
MIN_DOWN_COMPONENT = 0.5       #: unit gravity z must be <= -0.5 (within 60 deg of down)
GRAVITY_NORM_MIN, GRAVITY_NORM_MAX = 0.5, 1.5
YAW_TAU = 0.10                 #: s, EMA on the body yaw rate
WHEEL_VEL_TAU = 0.10           #: s, EMA on measured wheel velocities

# ---------------------------------------------------------------------------
# Tracking-failure indicators. Telemetry and feedback-shaping only: they never
# terminate an episode, never trigger a reset and never touch scoring.
# ---------------------------------------------------------------------------
STALL_RATIO = 0.15             #: measured/target below this counts as not tracking
STALL_MIN_TARGET_RAD_S = 0.25  #: below this target magnitude no ratio is judged
YAW_STALL_RATE = 0.05          #: rad/s; |yaw rate| under this counts as no yaw response
YAW_STALL_MIN_DIFF_RAD_S = 0.5 #: only judge yaw stall while commanding at least this differential
STALL_LATCH_CALLS = 75         #: 1.5 s at dt = 0.02

MODES = ("passthrough", "arc", "stop")


class LocomotionError(ValueError):
    """The candidate cannot be built or driven against the live schema/inputs."""


@dataclass(frozen=True)
class ProprioLayout:
    """Index layout of the official Task B proprio group.

    Term order is the official
    ``task_base/envs_base_cfg.py::ObservationsCfg.ProprioObservationsCfg``:
    ``base_lin_vel(3)``, ``base_ang_vel(3)``, ``velocity_commands(3)``,
    ``projected_gravity(3)``, ``joint_pos(N)``, ``joint_vel(N)``, ``actions(A)``,
    concatenated with ``enable_corruption = False``. With N = A = 24 this is the
    public 84-vector.

    Only ``base_ang_vel``, ``projected_gravity`` and ``joint_vel`` are read here.
    The 6:9 block is the *velocity command*, not an acceleration, and is not used
    as feedback. ``joint_vel`` is ``mdp.joint_vel_rel`` with no scale, so it is
    rad/s relative to the default joint velocities (zero for this articulation).
    """

    num_joints: int
    num_actions: int

    @property
    def base_ang_vel(self) -> slice:
        return slice(3, 6)

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
    def size(self) -> int:
        return 12 + 2 * self.num_joints + self.num_actions

    def to_dict(self) -> dict:
        return {
            "expected_size": self.size,
            "base_ang_vel": [3, 6],
            "velocity_commands_not_used": [6, 9],
            "projected_gravity": [9, 12],
            "joint_pos_rel": [self.joint_pos.start, self.joint_pos.stop],
            "joint_vel_rel": [self.joint_vel.start, self.joint_vel.stop],
            "joint_vel_units": "rad/s, mdp.joint_vel_rel with no scale term; default joint "
                               "velocities are zero for B2wPiper",
            "source": "official ProprioObservationsCfg term order (concatenated, corruption disabled)",
        }


def _num(value):
    """JSON-safe float: ``None`` for missing or non-finite values."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _ema_alpha(dt: float, tau: float) -> float:
    return float(dt / (tau + dt))


def _slew(previous: float, target: float, step: float) -> float:
    delta = target - previous
    if delta > step:
        return previous + step
    if delta < -step:
        return previous - step
    return target


class LocomotionController:
    """Conservative forward/arc wheel controller with bounded yaw feedback.

    Parameters
    ----------
    schema:
        Live :class:`task_b.control.ActionSchema` (or anything exposing the same
        ``term``/``total_dim``/``validate`` interface). Action indices, wheel
        order and the wheel scale are read from it; nothing is assumed.
    observation_joint_names:
        Joint order of the proprio ``joint_pos``/``joint_vel`` blocks, as
        resolved by the live observation manager (``preserve_order=True``). Wheel
        velocities are looked up **by name**, so this order may differ from both
        the articulation order and the action order.
    dt:
        Control period in seconds (0.02 for the official task).

    Keyword arguments are the bounded, configurable defaults documented at module
    level. ``max_wheel_rad_s`` is the master bound: every returned wheel entry
    satisfies ``|scale * entry| <= max_wheel_rad_s``.
    """

    def __init__(self, schema, observation_joint_names, dt: float = 0.02, *,
                 max_wheel_rad_s: float = MAX_WHEEL_RAD_S, common_max: float = COMMON_MAX,
                 diff_max: float = DIFF_MAX, curvature_ratio: float = CURVATURE_RATIO,
                 slew_per_call: float = SLEW_PER_CALL, turn_sign: float = 1.0,
                 bearing_kp: float = BEARING_KP, bearing_to_yaw_rate: float = BEARING_TO_YAW_RATE,
                 yaw_rate_max: float = YAW_RATE_MAX, yaw_ki: float = YAW_KI,
                 diff_i_max: float = DIFF_I_MAX, forward_assist_max: float = FORWARD_ASSIST_MAX,
                 forward_assist_ki: float = FORWARD_ASSIST_KI,
                 tilt_stop: float = TILT_STOP, tilt_rate_stop: float = TILT_RATE_STOP,
                 stall_latch_calls: int = STALL_LATCH_CALLS,
                 wheel_default_vel_rad_s=0.0, strict: bool = True):
        if not np.isfinite(dt) or dt <= 0.0:
            raise LocomotionError(f"dt must be positive and finite, got {dt!r}")
        if turn_sign not in (-1.0, 1.0, -1, 1):
            raise LocomotionError(f"turn_sign must be +1 or -1, got {turn_sign!r}")
        positive = {"max_wheel_rad_s": max_wheel_rad_s, "slew_per_call": slew_per_call,
                    "yaw_rate_max": yaw_rate_max, "tilt_stop": tilt_stop,
                    "tilt_rate_stop": tilt_rate_stop}
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise LocomotionError(f"{name} must be finite and > 0, got {value!r}")
        nonnegative = {"common_max": common_max, "diff_max": diff_max, "bearing_kp": bearing_kp,
                       "bearing_to_yaw_rate": bearing_to_yaw_rate, "yaw_ki": yaw_ki,
                       "diff_i_max": diff_i_max, "forward_assist_max": forward_assist_max,
                       "forward_assist_ki": forward_assist_ki}
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise LocomotionError(f"{name} must be finite and >= 0, got {value!r}")
        if not np.isfinite(curvature_ratio) or not 0.0 <= curvature_ratio <= 1.0:
            raise LocomotionError(
                f"curvature_ratio must be in [0, 1] so no wheel reverses against the common "
                f"command, got {curvature_ratio!r}")
        if max_wheel_rad_s > HARD_MAX_WHEEL_RAD_S:
            raise LocomotionError(
                f"max_wheel_rad_s {max_wheel_rad_s} exceeds this module's hard test cap "
                f"{HARD_MAX_WHEEL_RAD_S} rad/s; the actuator's velocity_limit_sim is 50 rad/s but "
                "no run supports commands that large")
        if tilt_stop > 1.0:
            raise LocomotionError("tilt_stop is sin(tilt) and must be <= 1.0")
        if int(stall_latch_calls) != stall_latch_calls or stall_latch_calls < 1:
            raise LocomotionError("stall_latch_calls must be a positive integer")

        if callable(getattr(schema, "validate", None)):
            schema.validate()
        self.schema = schema
        self.total_dim = int(schema.total_dim)
        wheel = schema.term(WHEEL_TERM)
        leg = schema.term(LEG_TERM)
        arm = schema.term(ARM_TERM)
        if wheel.mode != "velocity":
            raise LocomotionError(f"Wheel term is '{wheel.mode}', expected a velocity term")
        if wheel.clip is not None:
            raise LocomotionError("A non-None wheel action clip is not supported")
        if not np.isfinite(wheel.scale) or wheel.scale <= 0.0:
            raise LocomotionError(f"Wheel scale must be positive and finite, got {wheel.scale!r}")
        self.wheel_scale = float(wheel.scale)

        self._wheel_names = tuple(wheel.joint_names)
        self._wheel_indices = np.asarray([wheel.start + offset for offset in range(wheel.dim)],
                                        dtype=np.int64)
        self._wheel_sign = np.asarray(
            [1.0 if wheel_side(name) == "right" else -1.0 for name in self._wheel_names],
            dtype=np.float64)
        if not np.any(self._wheel_sign > 0) or not np.any(self._wheel_sign < 0):
            raise LocomotionError(f"Wheel joints {list(self._wheel_names)} do not span both sides")
        # Diagonal grouping for H2: 'a' = FL/RR, 'b' = FR/RL.
        self._diagonal = np.asarray(
            [1.0 if name.split("_")[0] in ("FL", "RR") else -1.0 for name in self._wheel_names],
            dtype=np.float64)
        self._leg_slice = slice(leg.start, leg.stop)
        self._arm_slice = slice(arm.start, arm.stop)
        self._non_wheel = np.ones(self.total_dim, dtype=bool)
        self._non_wheel[self._wheel_indices] = False

        self.observation_joint_names = tuple(str(name) for name in observation_joint_names)
        if (len(self.observation_joint_names) != 24
                or set(self.observation_joint_names) != set(schema.joint_names)):
            raise LocomotionError("Expected all 24 articulation joints in resolved observation order")
        if len(set(self.observation_joint_names)) != len(self.observation_joint_names):
            raise LocomotionError("observation_joint_names contains duplicates")
        missing = [name for name in self._wheel_names if name not in self.observation_joint_names]
        if missing:
            raise LocomotionError(f"Wheel joints absent from observation_joint_names: {missing}")
        self._wheel_obs_indices = np.asarray(
            [self.observation_joint_names.index(name) for name in self._wheel_names],
            dtype=np.int64)
        self.layout = ProprioLayout(num_joints=len(self.observation_joint_names),
                                    num_actions=self.total_dim)

        defaults = np.asarray(wheel_default_vel_rad_s, dtype=np.float64).reshape(-1)
        if defaults.size == 1:
            defaults = np.repeat(defaults, len(self._wheel_names))
        if defaults.size != len(self._wheel_names) or not np.isfinite(defaults).all():
            raise LocomotionError(
                "wheel_default_vel_rad_s must be a finite scalar or one finite value per wheel")
        if np.any(defaults != 0.0):
            raise LocomotionError("Only the official zero default wheel velocity is supported")
        self._wheel_default_vel = defaults

        self.dt = float(dt)
        self.max_wheel_rad_s = float(max_wheel_rad_s)
        #: Normalized bound derived through the *live* scale.
        self.wheel_cap = self.max_wheel_rad_s / self.wheel_scale
        self.common_max = float(common_max)
        self.diff_max = float(diff_max)
        if self.common_max + self.diff_max > self.wheel_cap + 1e-12:
            raise LocomotionError(
                f"common_max + diff_max ({self.common_max + self.diff_max:.4f}) exceeds the physical "
                f"bound {self.wheel_cap:.4f} normalized ({self.max_wheel_rad_s} rad/s at scale "
                f"{self.wheel_scale}); raise max_wheel_rad_s deliberately or lower the caps")
        self.curvature_ratio = float(curvature_ratio)
        self.slew_per_call = float(slew_per_call)
        self.turn_sign = float(turn_sign)
        self.bearing_kp = float(bearing_kp)
        self.bearing_to_yaw_rate = float(bearing_to_yaw_rate)
        self.yaw_rate_max = float(yaw_rate_max)
        self.yaw_ki = float(yaw_ki)
        self.diff_i_max = float(min(diff_i_max, self.diff_max))
        self.forward_assist_max = float(forward_assist_max)
        self.forward_assist_ki = float(forward_assist_ki)
        self.tilt_stop = float(tilt_stop)
        self.tilt_rate_stop = float(tilt_rate_stop)
        self.stall_latch_calls = int(stall_latch_calls)
        self.strict = bool(strict)

        self.debug: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear every integrator, filter, slew memory and latched indicator."""
        self.calls = 0
        self._applied = np.zeros(len(self._wheel_names), dtype=np.float64)
        self._assist = 0.0
        self._diff_i = 0.0
        self._yaw_f = 0.0
        self._wheel_f = np.zeros(len(self._wheel_names), dtype=np.float64)
        self._filters_primed = False
        self._common_stall_calls = 0
        self._yaw_stall_calls = 0
        self.common_tracking_failed = False
        self.yaw_tracking_failed = False
        self._common_stall_events = 0
        self._yaw_stall_events = 0
        self._stop_events = 0
        self._arc_calls = 0
        self._tracking = {}
        self.debug = {"call": 0, "mode": "stop", "state": "reset", "stop_reasons": []}

    def _clear_feedback(self) -> None:
        """Zero commands and integrators; drop latched indicators lacking evidence."""
        self._applied[:] = 0.0
        self._assist = 0.0
        self._diff_i = 0.0
        self._common_stall_calls = 0
        self._yaw_stall_calls = 0
        self.common_tracking_failed = False
        self.yaw_tracking_failed = False

    # ------------------------------------------------------------------- main
    def apply(self, incoming_action, proprio, *, bearing_error=None,
              enabled: bool = True) -> np.ndarray:
        """Return a finite float32 action of the schema size, wheels only rewritten.

        Every non-wheel entry of ``incoming_action`` (legs and arm) is preserved
        exactly. Raw-action validation happens before any gate, scaling or early
        exit, so a malformed request is reported even when the wheels are about
        to be parked.
        """
        action = self._validate_action(incoming_action)
        bearing = self._validate_bearing(bearing_error)
        measurement, errors = self._read_proprio(proprio)

        self.calls += 1
        out = action.copy()
        request = action[self._wheel_indices]
        stop_reasons = list(errors)
        if not bool(enabled):
            stop_reasons.append("disabled by caller")
        if np.all(request == 0.0):
            stop_reasons.append("zero wheel request")
        elif bearing is not None and float(np.mean(request)) == 0.0:
            stop_reasons.append("zero common request in arc mode")

        gravity = angvel = None
        tilt = tilt_rate = yaw_rate = None
        if measurement is not None:
            gravity, angvel, wheel_vel = measurement
            self._update_filters(angvel[2], wheel_vel)
            tilt = float(np.hypot(gravity[0], gravity[1]))
            tilt_rate = float(np.hypot(angvel[0], angvel[1]))
            yaw_rate = float(angvel[2])
            if gravity[2] > -MIN_DOWN_COMPONENT:
                stop_reasons.append(
                    f"projected gravity outside the downward cone (g_z {gravity[2]:.4f})")
            if tilt >= self.tilt_stop:
                stop_reasons.append(f"sin(tilt) {tilt:.4f} >= {self.tilt_stop}")
            if tilt_rate >= self.tilt_rate_stop:
                stop_reasons.append(f"|(w_x,w_y)| {tilt_rate:.4f} >= {self.tilt_rate_stop} rad/s")

        if stop_reasons:
            self._stop_events += 1
            self._clear_feedback()
            self._tracking = {}
            out[self._wheel_indices] = 0.0
            return self._finish(out, action, "stop", stop_reasons, bearing, tilt, tilt_rate,
                                yaw_rate, request, {}, None if gravity is None else float(gravity[2]))

        # ------------------------------------------------------------ tracking
        # Measured response is compared against the *previously applied* target,
        # which is the command that produced it.
        tracking = self._tracking_metrics(request)
        self._tracking = tracking

        common_req = float(np.mean(request))
        right = request[self._wheel_sign > 0]
        left = request[self._wheel_sign < 0]
        diff_req = 0.5 * float(np.mean(right) - np.mean(left))

        if bearing is None:
            mode = "passthrough"
            self._assist = 0.0
            self._diff_i = 0.0
            target = np.clip(request, -self.wheel_cap, self.wheel_cap)
            detail = {"discarded_request_diff": 0.0, "assist": 0.0, "diff_p": 0.0,
                      "diff_i": 0.0, "diff_applied": 0.0, "common_applied": common_req,
                      "curvature_cap": None, "yaw_rate_desired": None, "yaw_rate_error": None,
                      "integral_frozen": None, "clipped_by_physical_bound":
                          bool(np.any(np.abs(request) > self.wheel_cap + 1e-12))}
        else:
            mode = "arc"
            self._arc_calls += 1
            common_applied, assist = self._forward_command(common_req, tracking)
            diff_applied, detail = self._yaw_command(bearing, common_applied, tracking)
            detail.update(discarded_request_diff=diff_req, assist=assist,
                          common_applied=common_applied,
                          clipped_by_physical_bound=False)
            target = common_applied + self._wheel_sign * diff_applied

        if mode == "arc":
            # An arc is a convex cone of same-direction wheel commands. One
            # interpolation factor preserves its curvature bound; independent
            # per-wheel slew can violate the bound during deceleration.
            # Entering from a different mode or reversing direction first
            # clears old wheel requests, so an old spin cannot leak into arc.
            common_target = float(np.mean(target))
            if (self.debug.get("mode") != "arc"
                    or np.any(self._applied * common_target < 0.0)):
                self._applied[:] = 0.0
            delta = target - self._applied
            magnitude = float(np.max(np.abs(delta)))
            fraction = min(1.0, self.slew_per_call / magnitude) if magnitude > 0.0 else 1.0
            self._applied += fraction * delta
        else:
            for index in range(self._applied.size):
                self._applied[index] = _slew(float(self._applied[index]), float(target[index]),
                                             self.slew_per_call)
        self._applied = np.clip(self._applied, -self.wheel_cap, self.wheel_cap)
        out[self._wheel_indices] = self._applied
        return self._finish(out, action, mode, [], bearing, tilt, tilt_rate, yaw_rate,
                            request, detail, None if gravity is None else float(gravity[2]))

    # -------------------------------------------------------------- internals
    def _validate_action(self, incoming_action) -> np.ndarray:
        """Reject a malformed raw action before any gate, scale or early exit."""
        try:
            action = np.asarray(incoming_action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise LocomotionError(f"Incoming action is not array-like: {error}") from error
        if action.size != self.total_dim:
            raise LocomotionError(f"Action of size {action.size} != schema total {self.total_dim}")
        if not np.isfinite(action).all():
            raise LocomotionError("Incoming action contains non-finite entries")
        if np.any(np.abs(action) > np.finfo(np.float32).max):
            raise LocomotionError("Incoming action is not representable as finite float32")
        return action

    def _validate_bearing(self, bearing_error):
        """``None`` or a finite, bounded scalar bearing in radians."""
        if bearing_error is None:
            return None
        value = np.asarray(bearing_error, dtype=np.float64).reshape(-1)
        if value.size != 1:
            raise LocomotionError(f"bearing_error must be a scalar, got size {value.size}")
        bearing = float(value[0])
        if not np.isfinite(bearing):
            raise LocomotionError("bearing_error must be finite")
        if abs(bearing) > BEARING_ABS_MAX:
            raise LocomotionError(
                f"bearing_error {bearing} rad is outside +/-{BEARING_ABS_MAX}; a body-frame bearing "
                "cannot exceed pi")
        return bearing

    def _read_proprio(self, proprio):
        """Return ``(unit_gravity, angvel, wheel_vel_rad_s)`` or ``(None, reasons)``."""
        def fail(message):
            if self.strict:
                raise LocomotionError(message)
            return None, [message]

        try:
            values = np.asarray(proprio, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            return fail(f"proprio is not array-like: {error}")
        if values.size != self.layout.size:
            return fail(f"proprio size {values.size} != expected {self.layout.size}")
        gravity = values[self.layout.projected_gravity]
        angvel = values[self.layout.base_ang_vel]
        wheel_vel = values[self.layout.joint_vel][self._wheel_obs_indices] + self._wheel_default_vel
        if not (np.isfinite(gravity).all() and np.isfinite(angvel).all()
                and np.isfinite(wheel_vel).all()):
            return fail("non-finite projected gravity, body angular velocity or wheel velocity")
        # Official observations are float32. Reject a wider caller's finite
        # extremes before subtraction/EMA/means can overflow and poison state.
        limit = np.finfo(np.float32).max
        if np.any(np.abs(angvel) > limit) or np.any(np.abs(wheel_vel) > limit):
            return fail("body angular velocity or wheel velocity is not finite-float32 representable")
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or not GRAVITY_NORM_MIN <= norm <= GRAVITY_NORM_MAX:
            return fail(f"projected gravity has implausible norm {norm:.4f}")
        return (gravity / norm, angvel, wheel_vel), []

    def _update_filters(self, yaw_rate: float, wheel_vel: np.ndarray) -> None:
        if not self._filters_primed:
            self._yaw_f = float(yaw_rate)
            self._wheel_f = np.array(wheel_vel, dtype=np.float64)
            self._filters_primed = True
            return
        self._yaw_f += _ema_alpha(self.dt, YAW_TAU) * (float(yaw_rate) - self._yaw_f)
        self._wheel_f += _ema_alpha(self.dt, WHEEL_VEL_TAU) * (
            np.asarray(wheel_vel, dtype=np.float64) - self._wheel_f)

    def _tracking_metrics(self, request: np.ndarray) -> dict:
        """Measured-vs-previously-commanded tracking, and the stall latches.

        Physical units throughout. ``target_*`` is what was *applied* on the
        previous call; ``measured_*`` is the filtered public wheel velocity. The
        two are kept separate everywhere, including in the debug record.
        """
        scale = self.wheel_scale
        previous = self._applied
        target = previous * scale
        measured = self._wheel_f
        target_common = float(np.mean(target))
        measured_common = float(np.mean(measured))
        target_diff = 0.5 * float(np.mean(target[self._wheel_sign > 0])
                                  - np.mean(target[self._wheel_sign < 0]))
        measured_diff = 0.5 * float(np.mean(measured[self._wheel_sign > 0])
                                    - np.mean(measured[self._wheel_sign < 0]))

        ratios = []
        for index in range(target.size):
            if abs(target[index]) >= STALL_MIN_TARGET_RAD_S:
                ratios.append(float(measured[index] / target[index]))
            else:
                ratios.append(None)

        # H2: does one diagonal carry the motion? Positive means FL/RR spin more.
        magnitude = np.abs(measured)
        group_a = float(np.mean(magnitude[self._diagonal > 0])) if np.any(self._diagonal > 0) else 0.0
        group_b = float(np.mean(magnitude[self._diagonal < 0])) if np.any(self._diagonal < 0) else 0.0

        common_ratio = None
        if abs(target_common) >= STALL_MIN_TARGET_RAD_S:
            common_ratio = float(measured_common / target_common)
            if common_ratio < STALL_RATIO:
                self._common_stall_calls += 1
            else:
                self._common_stall_calls = max(0, self._common_stall_calls - 1)
        else:
            self._common_stall_calls = max(0, self._common_stall_calls - 1)
        if self._common_stall_calls >= self.stall_latch_calls and not self.common_tracking_failed:
            self.common_tracking_failed = True
            self._common_stall_events += 1
        elif self._common_stall_calls == 0:
            self.common_tracking_failed = False

        diff_ratio = None
        if abs(target_diff) >= YAW_STALL_MIN_DIFF_RAD_S:
            diff_ratio = float(measured_diff / target_diff)
            if abs(self._yaw_f) < YAW_STALL_RATE:
                self._yaw_stall_calls += 1
            else:
                self._yaw_stall_calls = max(0, self._yaw_stall_calls - 1)
        else:
            self._yaw_stall_calls = max(0, self._yaw_stall_calls - 1)
        if self._yaw_stall_calls >= self.stall_latch_calls and not self.yaw_tracking_failed:
            self.yaw_tracking_failed = True
            self._yaw_stall_events += 1
        elif self._yaw_stall_calls == 0:
            self.yaw_tracking_failed = False

        return {
            "requested_common_normalized": float(np.mean(request)),
            "target_common_rad_s": target_common,
            "measured_common_rad_s": measured_common,
            "common_tracking_ratio": common_ratio,
            "target_diff_rad_s": target_diff,
            "measured_diff_rad_s": measured_diff,
            "diff_tracking_ratio": diff_ratio,
            "per_wheel_target_rad_s": [float(x) for x in target],
            "per_wheel_measured_rad_s": [float(x) for x in measured],
            "per_wheel_tracking_ratio": ratios,
            "diagonal_mean_abs_rad_s": {"FL_RR": group_a, "FR_RL": group_b},
            "diagonal_asymmetry_rad_s": group_a - group_b,
            "yaw_rate_filtered_rad_s": float(self._yaw_f),
            "common_stall_calls": int(self._common_stall_calls),
            "yaw_stall_calls": int(self._yaw_stall_calls),
            "common_tracking_failed": bool(self.common_tracking_failed),
            "yaw_tracking_failed": bool(self.yaw_tracking_failed),
            "definition": "target_* is the command applied on the previous call; measured_* is the "
                          "filtered public joint_vel_rel of the named wheel joints",
        }

    def _forward_command(self, common_req: float, tracking: dict):
        """Bounded common command plus the anti-windup forward tracking assist."""
        if common_req == 0.0 or self.forward_assist_max <= 0.0:
            self._assist = max(0.0, self._assist * (1.0 - INTEGRAL_LEAK))
            if common_req == 0.0:
                self._assist = 0.0
            return float(np.clip(common_req, -self.common_max, self.common_max)), 0.0

        direction = 1.0 if common_req > 0.0 else -1.0
        # Track the caller's nominal velocity, not our assisted command. Using
        # the latter would keep increasing assistance under a constant load
        # even after actual wheel speed exceeds the caller's request.
        nominal = float(np.clip(common_req, -self.common_max, self.common_max))
        shortfall_norm = direction * (nominal
                                      - tracking["measured_common_rad_s"] / self.wheel_scale)
        saturated = abs(common_req) + self._assist >= self.common_max - 1e-12
        if self.common_tracking_failed:
            # The wheels are not responding at all: escalating the target only
            # raises torque into a blocked wheel. Shed the assist instead.
            self._assist = max(0.0, self._assist * (1.0 - INTEGRAL_LEAK))
        elif saturated and shortfall_norm > 0.0:
            pass  # clamping anti-windup: hold, do not accumulate into the cap
        else:
            self._assist = float(np.clip(self._assist + self.dt * self.forward_assist_ki
                                         * shortfall_norm, 0.0, self.forward_assist_max))
        common = direction * (abs(common_req) + self._assist)
        return float(np.clip(common, -self.common_max, self.common_max)), direction * self._assist

    def _yaw_command(self, bearing: float, common_applied: float, tracking: dict):
        """Bounded yaw correction from the bearing, the gyro and wheel tracking."""
        diff_p = self.turn_sign * self.bearing_kp * bearing
        yaw_desired = float(np.clip(self.bearing_to_yaw_rate * bearing,
                                    -self.yaw_rate_max, self.yaw_rate_max))
        yaw_error = yaw_desired - float(self._yaw_f)

        curvature_cap = self.curvature_ratio * abs(common_applied)
        limit = min(self.diff_max, curvature_cap)
        integrate = (self.yaw_ki > 0.0 and limit > 0.0 and not self.yaw_tracking_failed)
        frozen = False
        if integrate:
            increment = self.dt * self.yaw_ki * self.turn_sign * yaw_error
            candidate = float(np.clip(self._diff_i + increment, -self.diff_i_max, self.diff_i_max))
            total_candidate = diff_p + candidate
            total_current = diff_p + self._diff_i
            if abs(total_candidate) > limit and abs(total_candidate) > abs(total_current):
                frozen = True  # saturated and pushing further out
            else:
                self._diff_i = candidate
        else:
            self._diff_i *= (1.0 - INTEGRAL_LEAK)
            if abs(self._diff_i) < 1e-9:
                self._diff_i = 0.0

        diff = float(np.clip(diff_p + self._diff_i, -limit, limit))
        return diff, {
            "diff_p": float(diff_p),
            "diff_i": float(self._diff_i),
            "diff_applied": diff,
            "curvature_cap": float(curvature_cap),
            "diff_limit": float(limit),
            "yaw_rate_desired": yaw_desired,
            "yaw_rate_error": float(yaw_error),
            "integral_frozen": bool(frozen),
            "integral_active": bool(integrate),
        }

    def _finish(self, out: np.ndarray, action: np.ndarray, mode: str, stop_reasons,
                bearing, tilt, tilt_rate, yaw_rate, request, detail, gravity_z=None) -> np.ndarray:
        if not np.isfinite(out).all():
            raise LocomotionError("Locomotion controller produced a non-finite action")
        result = out.astype(np.float32)
        if not np.isfinite(result).all():
            raise LocomotionError("Locomotion output is not representable as finite float32")
        preserved = np.array_equal(result[self._non_wheel],
                                   action.astype(np.float32)[self._non_wheel])
        if not preserved:
            raise LocomotionError("Non-wheel action entries were not preserved")
        physical = result[self._wheel_indices].astype(np.float64) * self.wheel_scale
        if np.any(np.abs(physical) > self.max_wheel_rad_s + 1e-6):
            raise LocomotionError("Wheel target exceeded the physical bound")
        self.debug = {
            "call": self.calls,
            "mode": mode,
            "state": mode,
            "stop_reasons": list(stop_reasons),
            "enabled_output_zero": bool(np.all(result[self._wheel_indices] == 0.0)),
            "bearing_error_rad": _num(bearing),
            "bearing_error_supplied": bearing is not None,
            "posture": {
                "sin_tilt": _num(tilt),
                "tilt_deg": None if tilt is None or gravity_z is None else
                            _num(np.degrees(np.arctan2(tilt, -gravity_z))),
                "tilt_rate_rad_s": _num(tilt_rate),
                "yaw_rate_rad_s": _num(yaw_rate),
                "yaw_rate_filtered_rad_s": _num(self._yaw_f),
                "tilt_stop": self.tilt_stop,
                "tilt_rate_stop": self.tilt_rate_stop,
            },
            "wheels": {
                "joint_names": list(self._wheel_names),
                "action_indices": [int(index) for index in self._wheel_indices],
                "side_sign": {name: float(sign) for name, sign
                              in zip(self._wheel_names, self._wheel_sign)},
                "requested_normalized": [float(x) for x in request],
                "applied_normalized": [float(x) for x in result[self._wheel_indices]],
                "applied_target_rad_s": [float(x) for x in physical],
                "physical_bound_rad_s": self.max_wheel_rad_s,
                "normalized_bound": self.wheel_cap,
                "saturated": bool(np.any(np.abs(physical) >= self.max_wheel_rad_s - 1e-6)),
                "wheel_scale": self.wheel_scale,
            },
            "command": {key: _num(value) if not isinstance(value, (bool, np.bool_))
                        and isinstance(value, (int, float, np.floating))
                        else value for key, value in detail.items()},
            "tracking": self._tracking,
            "counters": {
                "calls": self.calls, "arc_calls": self._arc_calls,
                "stop_events": self._stop_events,
                "common_tracking_failed_events": self._common_stall_events,
                "yaw_tracking_failed_events": self._yaw_stall_events,
            },
            "non_wheel_entries_preserved": bool(preserved),
            "note": "tracking-failure indicators are telemetry and feedback shaping only; they "
                    "never reset the episode, never terminate and never touch scoring",
        }
        return result

    # ----------------------------------------------------------------- output
    def last_debug(self) -> dict:
        """Copy of the most recent per-call debug record."""
        return dict(self.debug)

    def describe(self) -> dict:
        """JSON-serializable description of the candidate and its hypotheses."""
        return {
            "controller": "LocomotionController",
            "status": "experimental candidate; never executed in simulation",
            "dt": self.dt,
            "strict": self.strict,
            "inputs": "public proprio observation, the static joint/action schema, and an optional "
                      "bearing error supplied by the caller's vision stage",
            "never_reads": ["robot root or world pose", "object poses", "contact or force buffers",
                            "seed/layout coordinates", "simulator objects", "any truth parameter"],
            "proprio_layout": self.layout.to_dict(),
            "modes": {
                "passthrough": "bearing_error is None: the caller's wheel request is preserved apart "
                               "from the physical per-wheel bound and the slew limit; no feedback is "
                               "added and integrators are cleared",
                "arc": "bearing_error supplied: the common part of the caller's wheel request is "
                       "kept (optionally assisted), the caller's differential and residual are "
                       "discarded, and a bounded yaw correction is added",
                "stop": "enabled=False, an all-zero wheel request, or a guard trip: wheels are zero "
                        "on the same call and all integrators and slew memory are cleared",
            },
            "wheel": {
                "joint_names": list(self._wheel_names),
                "action_indices": [int(index) for index in self._wheel_indices],
                "observation_indices": [int(index) for index in self._wheel_obs_indices],
                "side_sign": {name: float(sign) for name, sign
                              in zip(self._wheel_names, self._wheel_sign)},
                "diagonal_groups": {"FL_RR": [name for name, value
                                              in zip(self._wheel_names, self._diagonal) if value > 0],
                                    "FR_RL": [name for name, value
                                              in zip(self._wheel_names, self._diagonal) if value < 0]},
                "scale": self.wheel_scale,
                "physical_bound_rad_s": self.max_wheel_rad_s,
                "normalized_bound": self.wheel_cap,
                "common_max": self.common_max,
                "diff_max": self.diff_max,
                "curvature_ratio": self.curvature_ratio,
                "slew_per_call": self.slew_per_call,
                "target_definition": "physical wheel velocity target = scale * normalized action "
                                     "(velocity term, zero default-velocity offset)",
                "no_linear_speed_conversion": "wheel rad/s is never converted to m/s; the wheel "
                                              "radius and track width are not verified here",
                "cannot_rotate_in_place": "the differential is capped at curvature_ratio * |common|, "
                                          "so a zero common command yields a zero differential",
            },
            "feedback": {
                "forward_assist": {
                    "max": self.forward_assist_max, "ki_per_s": self.forward_assist_ki,
                    "rationale": "the configured damping-only implicit drive estimates torque "
                                 "from target-minus-measured velocity. Assistance tracks the caller's "
                                 "nominal velocity; it decreases when measured velocity exceeds that "
                                 "request. Actual PhysX torque is not observed.",
                    "anti_windup": "held at the common_max clamp, shed by INTEGRAL_LEAK while the "
                                   "common tracking-failure latch is set, zeroed on any stop",
                },
                "yaw": {
                    "bearing_kp": self.bearing_kp,
                    "bearing_to_yaw_rate": self.bearing_to_yaw_rate,
                    "yaw_rate_max": self.yaw_rate_max,
                    "yaw_ki": self.yaw_ki,
                    "diff_i_max": self.diff_i_max,
                    "turn_sign": self.turn_sign,
                    "bearing_convention": "body-frame bearing in rad, positive toward +y (left), "
                                          "matching visual_approach.py's arctan2(y, x)",
                    "turn_sign_hypothesis": "turn_sign=+1 assumes a positive right-minus-left "
                                            "differential produces a positive (CCW, toward +y) body "
                                            "yaw rate. This is a configurable hypothesis, not a "
                                            "measurement; it is never auto-calibrated from drift.",
                    "anti_windup": "clamping: the integral is not accumulated when the total "
                                   "differential is already at min(diff_max, curvature_cap) and the "
                                   "increment pushes further out; it leaks toward zero when "
                                   "integration is disabled and is zeroed on any stop",
                },
            },
            "guards": {
                "action": "size, finiteness and float32 representability are checked before any "
                          "gate, scaling or early exit",
                "bearing": f"finite and |bearing| <= {BEARING_ABS_MAX} rad, else LocomotionError",
                "proprio": f"size must be {self.layout.size}; non-finite gravity/gyro/wheel "
                           f"velocities and a gravity norm outside "
                           f"[{GRAVITY_NORM_MIN}, {GRAVITY_NORM_MAX}] raise in strict mode and "
                           "force a stop otherwise; gyro/wheel rates must be finite-float32 "
                           "representable before filter updates",
                "gravity_cone": f"unit gravity z must be <= -{MIN_DOWN_COMPONENT}; inversion or a "
                                "horizontal reading stops the wheels",
                "tilt": f"sin(tilt) >= {self.tilt_stop} or |(w_x,w_y)| >= {self.tilt_rate_stop} "
                        "rad/s stops the wheels",
                "legs_and_arm": "never modified; no posture change and no leg mode of any kind",
                "history": "no actor history or environment state is reset by this module",
            },
            "tracking_failure_indicators": {
                "common": f"signed measured common / previously applied common target < {STALL_RATIO} "
                          f"raises a hysteretic counter to {self.stall_latch_calls} while that target is at "
                          f"least {STALL_MIN_TARGET_RAD_S} rad/s",
                "yaw": f"|filtered yaw rate| < {YAW_STALL_RATE} rad/s for {self.stall_latch_calls} "
                       f"net hysteretic counts while the applied half-differential is at least "
                       f"{YAW_STALL_MIN_DIFF_RAD_S} rad/s",
                "effect": "disables the corresponding integrator and sheds it; the latch clears "
                          "when the hysteresis counter returns to zero or on any stop",
                "scope": "telemetry and feedback shaping only; no reset, termination or scoring "
                         "effect whatsoever",
            },
            "integration": {
                "order": "run this controller BEFORE any stability filter; it produces the intended "
                         "wheel request, it is not a safety wrapper for someone else's request",
                "stability_wrapper": "task_b.stability.StabilityController as configured caps the "
                                     "common command at 0.15, the half-differential at 0.30, the "
                                     "per-wheel residual at 0.05, slews the common term at "
                                     "0.002/call and multiplies all of it by an adaptive authority "
                                     "with a 0.30 floor. Stacking it on top of this candidate would "
                                     "deliver at most 0.15 common and as little as 0.045, so the "
                                     "arc experiment must be run with that wrapper disabled.",
                "ownership": "wheel feedback is owned entirely by this module for the duration of a "
                             "call; the caller owns mode selection, bearing estimation, parking, "
                             "leg posture and the arm",
            },
            "hypotheses": [
                "H1 insufficient drive response may limit yaw. The configured damping and mean "
                "speed errors suggest an approximate 0.3-3 N*m scale, but do not measure peak "
                "PhysX torque or prove saturation. Forward arcs require a new physical test.",
                "H2 the FL/RR versus FR/RL speed split may reflect uneven support or contact; "
                "diagonal tracking metrics flag the symptom without measuring normal loads.",
                "H3 limiting |differential| to curvature_ratio * |common| keeps every wheel rolling "
                "in one direction, which avoids the counter-rotating pattern that preceded the "
                "step-322 illegal contact.",
            ],
            "limitations": [
                "No simulation run exists for this controller; no motion, yaw, arc or Task B score "
                "is claimed.",
                "The differential-to-yaw sign is a configurable hypothesis (turn_sign); it is never "
                "inferred from noisy drift.",
                "Static friction, rolling resistance, wheel radius, track width and the true "
                "per-wheel normal loads are all unknown here; the actuator reasoning comes from the "
                "configured ImplicitActuatorCfg gains and the recorded target-vs-measured wheel "
                "velocities, and contact or drive telemetry is not available to this module.",
                "max_wheel_rad_s defaults to 3.0 rad/s per wheel. Sustained commands at that "
                "magnitude have never been run on this task in any pattern; the only 500-step "
                "survival evidence is at 0.5 rad/s common and +/-1.5 rad/s counter-rotating.",
                "The forward assist assumes the caller's normalized common request is a velocity "
                "target. If the caller means something else, the assist is meaningless.",
                "Tilt and tilt-rate guards use the current raw public sample; feedback yaw and "
                "wheel rates use EMAs. Stopping wheels cannot arrest all ongoing leg collapse. "
                "The separate gain3 and .12-stop pulse probes both terminated on RR_thigh contact; "
                "neither early wheel stop nor same-direction commands prove stability.",
            ],
        }
