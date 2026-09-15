"""Turn-priority speed schedule for the Task B first-reach approach.

Independent helper. It owns *no* action vector, no schema, no joint and no
environment state: it takes the caller's already-chosen nominal wheel requests
and returns a scheduled ``(common, differential)`` pair in the same normalized
action units. It never changes motor gains, effort limits, physics, rewards or
terminations, never grasps, never resets any actor or environment history, and
reads no simulator, root, object or seed truth. Bearing and distance come from
the caller's public RGB-D detector; yaw rate and gravity come from the public
proprio vector.

Recorded failure it addresses (seed 42, official task)
------------------------------------------------------
``first_reach_seed42_01`` drove ~1.6 m in 1800 steps with almost no yaw at
common ~0.10 / half-differential ~0.20. Scaling those requests by 3 without
stance holding terminated at step 442 on an RR_thigh illegal contact (610 N,
base z ~0.4797 and roll ~0.20 rad at step 400 - retrospective diagnostics, not
inputs). A pulse variant that stopped the wheels near raw ``sin(tilt) = 0.12``
still terminated at step 447 because the legs kept folding: **stopping the wheels
is not physical recovery**. With root's ``stance_hold.py`` holding the settled
leg geometry, gain 3 survived 2400 steps but yawed only ~1.5 deg, and gain 8
survived with ~6.5 deg of yaw but drove forward fast enough that the target left
the camera field of view. That last failure - closing distance before the heading
is fixed - is exactly what this schedule is for.

The trade-off it encodes
------------------------
This chassis yaws far more slowly than it rolls, so the only way to keep a
usable bearing is to spend distance sparingly: hold the forward request near a
small floor (or at zero) while the bearing is large, keep the differential at the
caller's full authority, and release forward speed only as the bearing collapses
and the target is still far enough away. Nothing here makes the turn faster; it
stops the approach from consuming the alignment budget.

Where it may be used - and where it must not
--------------------------------------------
This schedule produces a *low-common / high-differential* pattern, which is
close to a counter-rotating command. It therefore belongs to the existing direct
differential path in root's ``first_reach.py`` (which forms
``forward + side_sign * turn`` itself), or to
:class:`task_b.locomotion.LocomotionController` in **passthrough** mode only
(``bearing_error=None``).

It must **not** be fed into ``LocomotionController``'s arc mode: that mode
enforces ``|differential| <= curvature_ratio * |common|`` (0.8 by default) to keep
every wheel rolling in one direction, so a 4%-forward schedule would have its
differential cut to ~3% of the caller's forward request. Same-direction arc
limiting and turn-priority near-spin are mutually exclusive intents; only one of
them can be in force on a given call, and this module claims the second.

Known risks, stated plainly
---------------------------
* A near-spin at a large differential is the family of commands that produced the
  step-322 (0.6 pure differential) and step-442/447 (gain 3, no stance hold)
  illegal contacts. Nothing here makes that pattern safe.
* This schedule is only defensible while root's ``stance_hold.py`` is active. It
  has no leg authority and cannot stop a leg from folding; its tilt and tilt-rate
  stops zero the wheels, which the recorded pulse run shows is not recovery.
* Every constant below is a hypothesis fitted to a handful of seed-42 runs. None
  of it is verified safety, and no motion, yaw, alignment or Task B score is
  claimed. No simulation has been run with this module.
"""
from __future__ import annotations

import numpy as np

#: Bearing thresholds in rad, with hysteresis: TURN is entered at
#: ``turn_enter`` and left at ``turn_exit``; DRIVE is entered at ``align_enter``
#: and left at ``align_exit``. Root's fixed gate is a smooth multiplier that
#: reaches its 4% floor at |bearing| = 0.24 rad.
TURN_ENTER, TURN_EXIT = 0.30, 0.22
ALIGN_ENTER, ALIGN_EXIT = 0.10, 0.16
#: Forward factor held while turning. 0.04 matches the floor root is testing, so
#: the two schedules can be compared at the same forward authority.
TURN_FORWARD_FLOOR = 0.04
MIN_MODE_CALLS = 10          #: dwell before a mode may change again (0.2 s at dt=0.02)

#: Distance handling, metres, from the caller's public RGB-D detector.
ALIGN_BEFORE_CLOSE_M = 1.00  #: inside this range, an unaligned target gets zero forward
STOP_DISTANCE_M = 0.50       #: forward taper reaches zero here (root owns the real standoff)
APPROACH_BAND_M = 0.60       #: taper width above the stop distance

#: Public-proprio guards. Thresholds match the checks root already applies in
#: first_reach.py, so this module is a redundant floor rather than a new policy.
TILT_STOP_RAD = 0.12         #: angle between -projected_gravity and body z
TILT_RATE_STOP = 0.45        #: |(w_x, w_y)| rad/s
GRAVITY_NORM_MIN, GRAVITY_NORM_MAX = 0.5, 1.5

BEARING_TAU, YAW_TAU = 0.10, 0.10   #: s, EMA time constants
LOSING_BEARING_RATE = 0.05   #: rad/s of |bearing| growth that pins forward to the floor
YAW_STALL_RATE = 0.03        #: rad/s; below this the yaw response counts as absent
YAW_PROBE_DIFF = 0.10        #: only judge yaw response above this |differential|
YAW_STALL_LATCH_CALLS = 100  #: 2 s of no response before the diagnostic latches

PER_WHEEL_ABS_MAX = 0.60     #: |common| + |differential| ceiling; forward yields first
MAX_INPUT_REQUEST = 5.0      #: a larger finite request is a caller bug, not a command
MAX_YAW_RATE_INPUT = 50.0    #: rad/s; beyond this the gyro reading is rejected
MODES = ("STOP", "TURN", "BLEND", "DRIVE")


class HeadingScheduleError(ValueError):
    """The schedule was given inputs it refuses to interpret."""


def _scalar(name, value, limit, *, allow_none=False):
    """Finite scalar within ``+/-limit``, or ``None`` when explicitly allowed."""
    if value is None:
        if allow_none:
            return None
        raise HeadingScheduleError(f"{name} must not be None")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise HeadingScheduleError(f"{name} must be a scalar, got size {array.size}")
    result = float(array[0])
    if not np.isfinite(result):
        raise HeadingScheduleError(f"{name} must be finite, got {value!r}")
    if abs(result) > limit:
        raise HeadingScheduleError(f"|{name}| = {abs(result)} exceeds the accepted bound {limit}")
    return result


class HeadingSchedule:
    """Schedule nominal common/differential wheel requests, turn first.

    All request values in and out are **normalized action units** for the
    official ``joint_wheel`` velocity term; ``bearing_error`` is a body-frame
    bearing in rad (positive toward +y/left, matching ``first_reach.py``'s
    ``arctan2(y, x)``), ``observed_yaw_rate`` is rad/s from public proprio
    ``[3:6]``, and ``distance_m`` is the detector's forward planar distance in
    metres. Wheel speeds are never converted to linear m/s: the wheel radius is
    not verified here.

    The returned ``common`` is ``forward_factor * forward_request`` with
    ``forward_factor`` in ``[0, 1]``, and ``differential`` is
    ``turn_factor * turn_request`` with ``turn_factor`` in
    ``[0, turn_boost_max]``. ``turn_boost_max`` defaults to 1.0, so by default
    **neither request can be amplified**; a value above 1.0 is the only way to
    authorize extra turn authority, and it is applied only while the yaw response
    is diagnosed as absent in TURN mode.
    """

    def __init__(self, dt: float = 0.02, *, turn_enter: float = TURN_ENTER,
                 turn_exit: float = TURN_EXIT, align_enter: float = ALIGN_ENTER,
                 align_exit: float = ALIGN_EXIT, turn_forward_floor: float = TURN_FORWARD_FLOOR,
                 min_mode_calls: int = MIN_MODE_CALLS,
                 align_before_close_m: float = ALIGN_BEFORE_CLOSE_M,
                 stop_distance_m: float = STOP_DISTANCE_M, approach_band_m: float = APPROACH_BAND_M,
                 per_wheel_abs_max: float = PER_WHEEL_ABS_MAX, tilt_stop_rad: float = TILT_STOP_RAD,
                 tilt_rate_stop: float = TILT_RATE_STOP, turn_boost_max: float = 1.0,
                 turn_boost_per_s: float = 0.0, yaw_stall_rate: float = YAW_STALL_RATE,
                 yaw_probe_diff: float = YAW_PROBE_DIFF,
                 yaw_stall_latch_calls: int = YAW_STALL_LATCH_CALLS,
                 losing_bearing_rate: float = LOSING_BEARING_RATE,
                 max_input_request: float = MAX_INPUT_REQUEST):
        if not np.isfinite(dt) or dt <= 0.0:
            raise HeadingScheduleError(f"dt must be positive and finite, got {dt!r}")
        for name, value in (("turn_enter", turn_enter), ("turn_exit", turn_exit),
                            ("align_enter", align_enter), ("align_exit", align_exit),
                            ("align_before_close_m", align_before_close_m),
                            ("stop_distance_m", stop_distance_m),
                            ("approach_band_m", approach_band_m),
                            ("per_wheel_abs_max", per_wheel_abs_max),
                            ("tilt_stop_rad", tilt_stop_rad), ("tilt_rate_stop", tilt_rate_stop),
                            ("yaw_stall_rate", yaw_stall_rate), ("yaw_probe_diff", yaw_probe_diff),
                            ("losing_bearing_rate", losing_bearing_rate),
                            ("max_input_request", max_input_request)):
            if not np.isfinite(value) or value <= 0.0:
                raise HeadingScheduleError(f"{name} must be finite and > 0, got {value!r}")
        if not 0.0 < align_enter < align_exit <= turn_exit < turn_enter <= np.pi:
            raise HeadingScheduleError(
                "bearing thresholds must satisfy 0 < align_enter < align_exit <= turn_exit < "
                f"turn_enter <= pi, got {align_enter}, {align_exit}, {turn_exit}, {turn_enter}")
        if not 0.0 <= turn_forward_floor <= 1.0:
            raise HeadingScheduleError("turn_forward_floor is a factor and must lie in [0, 1]")
        if not np.isfinite(turn_boost_max) or turn_boost_max < 1.0:
            raise HeadingScheduleError(
                "turn_boost_max must be >= 1.0; values above 1.0 explicitly authorize amplifying "
                "the caller's turn request")
        if not np.isfinite(turn_boost_per_s) or turn_boost_per_s < 0.0:
            raise HeadingScheduleError("turn_boost_per_s must be finite and >= 0")
        if tilt_stop_rad > 0.5 * np.pi:
            raise HeadingScheduleError("tilt_stop_rad must be below pi/2")
        if align_before_close_m < stop_distance_m:
            raise HeadingScheduleError("align_before_close_m must be >= stop_distance_m")
        for name, value in (("min_mode_calls", min_mode_calls),
                            ("yaw_stall_latch_calls", yaw_stall_latch_calls)):
            if int(value) != value or value < 1:
                raise HeadingScheduleError(f"{name} must be a positive integer")

        self.dt = float(dt)
        self.turn_enter, self.turn_exit = float(turn_enter), float(turn_exit)
        self.align_enter, self.align_exit = float(align_enter), float(align_exit)
        self.turn_forward_floor = float(turn_forward_floor)
        self.min_mode_calls = int(min_mode_calls)
        self.align_before_close_m = float(align_before_close_m)
        self.stop_distance_m = float(stop_distance_m)
        self.approach_band_m = float(approach_band_m)
        self.per_wheel_abs_max = float(per_wheel_abs_max)
        self.tilt_stop_rad = float(tilt_stop_rad)
        self.tilt_rate_stop = float(tilt_rate_stop)
        self.turn_boost_max = float(turn_boost_max)
        self.turn_boost_per_s = float(turn_boost_per_s)
        self.yaw_stall_rate = float(yaw_stall_rate)
        self.yaw_probe_diff = float(yaw_probe_diff)
        self.yaw_stall_latch_calls = int(yaw_stall_latch_calls)
        self.losing_bearing_rate = float(losing_bearing_rate)
        self.max_input_request = float(max_input_request)
        self.debug: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear mode, filters, the yaw diagnostic and any authorized boost."""
        self.calls = 0
        self.mode = "STOP"
        self._mode_calls = 0
        self._bearing_f = None
        self._bearing_rate = 0.0
        self._yaw_f = 0.0
        self._yaw_stall_calls = 0
        self.yaw_response_weak = False
        self._boost = 1.0
        self._stop_events = 0
        self._yaw_stall_events = 0
        self.debug = {"call": 0, "mode": "STOP", "state": "reset", "stop_reasons": []}

    def _clear_memory(self) -> None:
        """Drop scheduling memory so a stop cannot be crept out of."""
        self.mode = "STOP"
        self._mode_calls = 0
        self._bearing_f = None
        self._bearing_rate = 0.0
        self._yaw_f = 0.0
        self._boost = 1.0
        self._yaw_stall_calls = 0
        self.yaw_response_weak = False

    # ------------------------------------------------------------------- main
    def update(self, bearing_error, forward_request, turn_request, *, target_valid: bool = True,
               observed_yaw_rate: float = 0.0, projected_gravity=None, distance_m=None) -> dict:
        """Return the scheduled ``{'common', 'differential', 'mode', ...}`` record.

        Inputs are validated before any gate or scaling. ``target_valid=False``,
        a ``None`` bearing, or an all-zero request pair returns both outputs zero
        and clears scheduling memory on the same call.
        """
        bearing = _scalar("bearing_error", bearing_error, np.pi + 1e-6, allow_none=True)
        forward_request = _scalar("forward_request", forward_request, self.max_input_request)
        turn_request = _scalar("turn_request", turn_request, self.max_input_request)
        yaw_rate = _scalar("observed_yaw_rate", observed_yaw_rate, MAX_YAW_RATE_INPUT)
        distance = _scalar("distance_m", distance_m, 1.0e3, allow_none=True)
        if distance is not None and distance < 0.0:
            raise HeadingScheduleError(f"distance_m must be >= 0, got {distance}")
        gravity, tilt, tilt_rate = self._posture(projected_gravity)

        self.calls += 1
        reasons = []
        if not bool(target_valid):
            reasons.append("target_valid is False")
        if bearing is None:
            reasons.append("no bearing supplied")
        if forward_request == 0.0 and turn_request == 0.0:
            reasons.append("zero forward and turn request")
        if tilt is not None and tilt >= self.tilt_stop_rad:
            reasons.append(f"tilt {tilt:.4f} rad >= {self.tilt_stop_rad}")
        if tilt_rate is not None and tilt_rate >= self.tilt_rate_stop:
            reasons.append(f"|(w_x,w_y)| {tilt_rate:.4f} >= {self.tilt_rate_stop} rad/s")
        if reasons:
            self._stop_events += 1
            self._clear_memory()
            return self._record(0.0, 0.0, "STOP", reasons, bearing, forward_request, turn_request,
                               yaw_rate, tilt, tilt_rate, distance, 0.0, 0.0, {})

        # ------------------------------------------------------------ filters
        self._yaw_f += (self.dt / (YAW_TAU + self.dt)) * (yaw_rate - self._yaw_f)
        if self._bearing_f is None:
            self._bearing_f, self._bearing_rate = bearing, 0.0
        else:
            previous = self._bearing_f
            self._bearing_f += (self.dt / (BEARING_TAU + self.dt)) * (bearing - self._bearing_f)
            rate = (abs(self._bearing_f) - abs(previous)) / self.dt
            self._bearing_rate += (self.dt / (BEARING_TAU + self.dt)) * (rate - self._bearing_rate)
        # A signed EMA may pass through zero when a target changes sides. It
        # must not authorize forward motion while the current target is still
        # far off-axis. Release uses the conservative of raw and filtered size.
        magnitude = max(abs(bearing), abs(self._bearing_f))

        # ------------------------------------------- mode, with hysteresis
        mode = self.mode if self.mode in ("TURN", "BLEND", "DRIVE") else "BLEND"
        if self._mode_calls >= self.min_mode_calls or self.mode == "STOP":
            if magnitude >= self.turn_enter:
                mode = "TURN"
            elif magnitude <= self.align_enter:
                mode = "DRIVE"
            elif self.mode == "TURN" and magnitude <= self.turn_exit:
                mode = "BLEND"
            elif self.mode == "DRIVE" and magnitude >= self.align_exit:
                mode = "BLEND"
            elif self.mode == "STOP":
                mode = "BLEND"
        self._mode_calls = self._mode_calls + 1 if mode == self.mode else 0
        self.mode = mode

        # -------------------------------------------------- forward schedule
        detail = {"forward_factor_distance_taper": None, "hold_for_alignment": False,
                  "common_reduced_for_headroom": False}
        if mode == "TURN":
            forward_factor = self.turn_forward_floor
        elif mode == "DRIVE":
            forward_factor = 1.0
        else:
            span = max(self.turn_exit - self.align_enter, 1e-9)
            blend = float(np.clip((self.turn_exit - magnitude) / span, 0.0, 1.0))
            forward_factor = self.turn_forward_floor + (1.0 - self.turn_forward_floor) * blend
        # Mode dwell may delay a label change, but never delay a reduction in
        # forward authority after the current bearing exceeds the align band.
        guard_factor = 1.0
        if magnitude >= self.align_exit:
            blend = float(np.clip((self.turn_exit - magnitude)
                                  / (self.turn_exit - self.align_enter), 0.0, 1.0))
            guard_factor = self.turn_forward_floor + (1.0 - self.turn_forward_floor) * blend
            forward_factor = min(forward_factor, guard_factor)
        detail["bearing_guard_magnitude_rad"] = magnitude
        detail["forward_factor_guard_limit"] = guard_factor
        detail["forward_factor_bearing"] = forward_factor

        if distance is not None:
            taper = float(np.clip((distance - self.stop_distance_m) / self.approach_band_m, 0.0, 1.0))
            forward_factor *= taper
            detail["forward_factor_distance_taper"] = taper
            # Turn priority: do not spend the remaining approach distance while
            # the heading is still wrong, which is how the gain-8 run drove the
            # target out of the camera field of view.
            if distance <= self.align_before_close_m and magnitude > self.align_exit:
                forward_factor = 0.0
                detail["hold_for_alignment"] = True
        losing = self._bearing_rate > self.losing_bearing_rate
        if losing:
            forward_factor = min(forward_factor, self.turn_forward_floor)
        detail["bearing_growing"] = bool(losing)
        detail["bearing_rate_abs_rad_s"] = float(self._bearing_rate)
        forward_factor = float(np.clip(forward_factor, 0.0, 1.0))

        # ----------------------------------------------------- turn schedule
        turn_factor = self._turn_factor(mode, turn_request)
        common = forward_factor * forward_request
        differential = turn_factor * turn_request

        # Per-wheel headroom: the forward term yields first, so a saturated
        # command loses translation rather than heading authority.
        if abs(differential) > self.per_wheel_abs_max:
            differential = float(np.sign(differential) * self.per_wheel_abs_max)
        headroom = self.per_wheel_abs_max - abs(differential)
        if abs(common) > headroom:
            common = float(np.sign(common) * max(0.0, headroom))
            detail["common_reduced_for_headroom"] = True
        return self._record(common, differential, mode, [], bearing, forward_request, turn_request,
                            yaw_rate, tilt, tilt_rate, distance, forward_factor, turn_factor, detail)

    # -------------------------------------------------------------- internals
    def _posture(self, projected_gravity):
        """Validate and normalize an optional public projected-gravity vector."""
        if projected_gravity is None:
            return None, None, None
        values = np.asarray(projected_gravity, dtype=np.float64).reshape(-1)
        if values.size not in (3, 6):
            raise HeadingScheduleError(
                "projected_gravity must be the 3-vector from proprio[9:12], optionally followed by "
                "the 3 body angular rates from proprio[3:6]; a root quaternion is not accepted")
        gravity, rates = values[:3], (values[3:6] if values.size == 6 else None)
        if not np.isfinite(values).all():
            raise HeadingScheduleError("projected_gravity/angular rates must be finite")
        norm = float(np.linalg.norm(gravity))
        if not GRAVITY_NORM_MIN <= norm <= GRAVITY_NORM_MAX:
            raise HeadingScheduleError(f"projected gravity has implausible norm {norm:.4f}")
        unit = gravity / norm
        tilt = float(np.arccos(np.clip(-unit[2], -1.0, 1.0)))
        tilt_rate = None if rates is None else float(np.hypot(rates[0], rates[1]))
        return unit, tilt, tilt_rate

    def _turn_factor(self, mode: str, turn_request: float) -> float:
        """Caller authority (1.0), plus an explicitly authorized bounded boost."""
        stalled = (abs(turn_request) >= self.yaw_probe_diff
                   and abs(self._yaw_f) < self.yaw_stall_rate)
        self._yaw_stall_calls = (self._yaw_stall_calls + 1 if stalled
                                 else max(0, self._yaw_stall_calls - 1))
        if self._yaw_stall_calls >= self.yaw_stall_latch_calls and not self.yaw_response_weak:
            self.yaw_response_weak = True
            self._yaw_stall_events += 1
        elif self._yaw_stall_calls == 0:
            self.yaw_response_weak = False

        if self.turn_boost_max <= 1.0 or self.turn_boost_per_s <= 0.0:
            self._boost = 1.0
        elif mode == "TURN" and self.yaw_response_weak:
            self._boost = float(min(self.turn_boost_max,
                                    self._boost + self.turn_boost_per_s * self.dt))
        else:
            self._boost = float(max(1.0, self._boost - self.turn_boost_per_s * self.dt))
        return self._boost

    def _record(self, common, differential, mode, reasons, bearing, forward_request, turn_request,
                yaw_rate, tilt, tilt_rate, distance, forward_factor, turn_factor, detail) -> dict:
        """Build the JSON-safe return value and store it as ``self.debug``."""
        def number(value):
            if value is None:
                return None
            value = float(value)
            return value if np.isfinite(value) else None

        matches = None
        if bearing is not None and bearing != 0.0 and turn_request != 0.0:
            matches = bool(np.sign(turn_request) == np.sign(bearing))
        result = {
            "common": float(common),
            "differential": float(differential),
            "mode": mode,
            "call": self.calls,
            "stop_reasons": list(reasons),
            "units": "normalized joint_wheel action units; bearing rad; yaw rate rad/s; distance m",
            "request": {"forward": float(forward_request), "turn": float(turn_request),
                        "forward_factor": number(forward_factor), "turn_factor": number(turn_factor),
                        "amplified": bool(turn_factor > 1.0 + 1e-12),
                        "turn_sign_preserved": True,
                        "turn_sign_matches_bearing": matches},
            "bearing": {"raw_rad": number(bearing), "filtered_rad": number(self._bearing_f),
                        "abs_rate_rad_s": number(self._bearing_rate),
                        "turn_enter": self.turn_enter, "turn_exit": self.turn_exit,
                        "align_enter": self.align_enter, "align_exit": self.align_exit,
                        "mode_dwell_calls": int(self._mode_calls)},
            "distance_m": number(distance),
            "posture": {"tilt_rad": number(tilt), "tilt_rate_rad_s": number(tilt_rate),
                        "tilt_stop_rad": self.tilt_stop_rad,
                        "tilt_rate_stop_rad_s": self.tilt_rate_stop,
                        "supplied": tilt is not None},
            "yaw": {"raw_rad_s": number(yaw_rate), "filtered_rad_s": number(self._yaw_f),
                    "stall_calls": int(self._yaw_stall_calls),
                    "response_weak": bool(self.yaw_response_weak),
                    "boost": float(self._boost), "boost_max": self.turn_boost_max,
                    "note": "a weak yaw response is reported, not escalated, unless "
                            "turn_boost_max > 1 explicitly authorizes a bounded boost"},
            "schedule": dict(detail),
            "counters": {"calls": self.calls, "stop_events": self._stop_events,
                         "yaw_response_weak_events": self._yaw_stall_events},
            "claims": "scheduling hypothesis only; no verified motion, alignment or Task B score",
        }
        self.debug = result
        return result

    def describe(self) -> dict:
        """JSON-safe description of the schedule, its intent and its limits."""
        return {
            "helper": "HeadingSchedule",
            "status": "experimental; never executed in simulation",
            "dt": self.dt,
            "owns": "nominal common and half-differential wheel requests only",
            "does_not_own": ["action vector assembly", "arm or grasp FSM (root)",
                             "leg reference stabilization (root's stance_hold.py)",
                             "motor gains, effort limits, physics, rewards, terminations",
                             "environment or actor history"],
            "inputs": "caller bearing/distance from public RGB-D, public proprio yaw rate and "
                      "projected gravity; no root, object, contact or seed truth",
            "modes": {"TURN": "|bearing| >= turn_enter: forward held at turn_forward_floor",
                      "BLEND": "hysteresis band: forward interpolated up from the floor",
                      "DRIVE": "|bearing| <= align_enter: full caller forward request",
                      "STOP": "invalid target, zero request pair, or a tilt/tilt-rate trip"},
            "thresholds": {"turn_enter": self.turn_enter, "turn_exit": self.turn_exit,
                           "align_enter": self.align_enter, "align_exit": self.align_exit,
                           "min_mode_calls": self.min_mode_calls,
                           "turn_forward_floor": self.turn_forward_floor,
                           "align_before_close_m": self.align_before_close_m,
                           "stop_distance_m": self.stop_distance_m,
                           "approach_band_m": self.approach_band_m,
                           "per_wheel_abs_max": self.per_wheel_abs_max},
            "guards": {"validation": "all scalars finite and bounded before any gate; "
                                     f"|request| <= {self.max_input_request}",
                       "stop": "target_valid False, bearing None, zero request pair, "
                               f"tilt >= {self.tilt_stop_rad} rad or |(w_x,w_y)| >= "
                               f"{self.tilt_rate_stop} rad/s -> both outputs zero and memory cleared",
                       "no_env_reset": "a stop zeroes this module's outputs only; it never resets, "
                                       "terminates or scores anything",
                       "gravity": "optional 3-vector (or 3+3 with body rates) with norm in "
                                  f"[{GRAVITY_NORM_MIN}, {GRAVITY_NORM_MAX}]; a root quaternion is "
                                  "rejected",
                       "amplification": "forward_factor in [0,1]; turn_factor is 1.0 unless "
                                        "turn_boost_max > 1 and turn_boost_per_s > 0",
                       "turn_sign": "the caller's turn sign is passed through and never inferred "
                                    "from drift or gyro noise"},
            "integration": {
                "use_with": "root's direct differential path in first_reach.py, or "
                            "LocomotionController passthrough (bearing_error=None)",
                "never_with": "LocomotionController arc mode, which caps |differential| at "
                              "curvature_ratio * |common| and would erase the turn authority this "
                              "schedule exists to protect",
                "requires": "root's stance_hold.py active; wheel stops are not leg recovery",
            },
            "hypotheses": [
                "H1 holding forward at ~4% (or zero inside align_before_close_m) while |bearing| is "
                "large keeps the target in the camera field of view long enough to finish the turn; "
                "the gain-8 run lost it by closing distance first.",
                "H2 hysteresis plus a mode dwell removes the chatter a single smooth gate can show "
                "when the bearing estimate is noisy near its threshold.",
                "H3 yielding the forward term first when the per-wheel ceiling binds preserves yaw "
                "authority at the cost of translation.",
            ],
            "limitations": [
                "Low-common/high-differential commands are close to the counter-rotating pattern "
                "recorded before the step-322, step-442 and step-447 illegal contacts.",
                "No leg authority: the tilt stop zeroes wheels, and the recorded pulse run shows "
                "that this alone did not prevent the legs from folding.",
                "Constants are fitted to a few seed-42 runs; they are hypotheses, not safety limits.",
                "distance_m is a monocular RGB-D detector estimate; a wrong distance directly "
                "mis-schedules the approach.",
                "The weak-yaw diagnostic proves nothing about its cause: drive authority, per-wheel "
                "normal load and ground friction are not observable here.",
            ],
            "stance_geometry_note": "If a narrower or longer stance is ever tried to raise yaw "
                                    "authority, it must be a bounded static hip-abduction offset "
                                    "chosen offline (e.g. |delta hip| <= 0.10 rad) and owned by the "
                                    "leg module: narrowing shortens the yaw moment arm per unit "
                                    "wheel force while raising the per-wheel load and the "
                                    "self-collision risk between thigh and body. This helper "
                                    "implements no leg change of any kind.",
        }
