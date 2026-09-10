"""Neutral-reference stability profile for Task B.

This module adds one opt-in profile on top of the frozen
:class:`task_b.stability.StabilityController`. It does not modify or copy the
baseline implementation: it subclasses it, feeds it a *derived* copy of the
proprio vector, and leaves every native boundary safeguard (float32 rejection,
gravity norm/finiteness checks, inverted-gravity abort, zero-wheel immediate
stop, arm pass-through, leg limit clipping) exactly where it is.

Why this exists
---------------
The measured settled B2wPiper stance is not level in the base frame. The
``turn_stabilized_seed42_01`` run recorded ``sin(tilt) = 0.0633`` for a healthy
settled body, above the baseline's ``TILT_DEADBAND = 0.06``. The baseline
therefore gated on 366 calls of a *nominal* posture, drove its authority to the
0.30 floor and applied only 0.0879 of a requested 0.6 differential. That is a
false positive against the nominal pose, not a stability event.

The profile calibrates that nominal pose once, from public projected gravity
during the tail of the zero-action settle window, freezes it, and then feeds the
baseline a gravity vector expressed relative to that reference. Deviation from
the neutral stance drives the normal feedback and gating; the *absolute* posture
keeps an independent, wider safety envelope that can slow or stop the robot even
when the reference-relative error is small.

Runtime inputs remain the public 84-element proprio vector and the static
joint/action schema. No contact, object or root truth is read. No gain, physics,
asset or reward is changed.

Nothing here is verified in simulation. See ``task_b/NEUTRAL_REFERENCE_DESIGN.md``.
"""
from __future__ import annotations

import numpy as np

from .stability import (
    GRAVITY_TAU,
    SETTLE_CALLS,
    StabilityController,
    StabilityError,
    TILT_ABORT,
    WHEEL_MAX_COMMON,
    WHEEL_MAX_DIFF,
    _ema_alpha,
    _ramp_gate,
)

PROFILES = ("baseline", "neutral")

#: Number of trailing settle samples averaged into the neutral reference.
CALIB_WINDOW = 20
#: Max angle (rad) between any window sample and their mean. Above this the
#: stance was still moving and calibration is refused.
CALIB_MAX_SPREAD = 0.03
#: Max raw |(w_x, w_y)| (rad/s) allowed in the window.
CALIB_MAX_RATE = 0.15
#: Max sin(tilt) the frozen reference itself may have. The recorded nominal is
#: 0.0633; a start on a real slope must not be normalized away as "neutral".
CALIB_MAX_REFERENCE_TILT = 0.15
#: Minimum |mean(unit samples)|; a second, direction-free coherence check.
CALIB_MIN_COHERENCE = 0.999

#: Absolute-posture envelope, independent of the reference. Below the deadband
#: the profile is a bit-exact pass-through to the baseline; between deadband and
#: abort the wheel *request* is scaled down before the baseline sees it.
ABS_TILT_DEADBAND = 0.18   #: ~10.4 deg, well above the 0.0633 nominal stance
ABS_TILT_ABORT = float(TILT_ABORT)  #: ~17.46 deg, same trip point as the baseline
#: Once the absolute envelope trips, stay in the raw (uncalibrated) frame for at
#: least this many calls, and until the absolute tilt is back inside the deadband.
DANGER_LATCH_CALLS = 50

_DOWN = np.array([0.0, 0.0, -1.0])


def _json_float(value):
    """Plain float, or None when the value is missing or non-finite."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


class NeutralAwareStabilityController(StabilityController):
    """:class:`StabilityController` with an optional frozen neutral reference.

    Constructor is compatible with the baseline; the extra arguments are keyword
    only and all have defaults, so ``--stability_profile neutral`` can construct
    it exactly like the baseline.

    Parameters (in addition to the baseline's)
    ------------------------------------------
    enable_neutral_reference:
        ``False`` reproduces the baseline bit-for-bit (no calibration, no
        rotation, no request scaling). Present so one class can serve both arms
        of an A/B run; selecting this class is already the opt-in.
    calib_window, calib_max_spread, calib_max_rate, calib_max_reference_tilt:
        Calibration acceptance thresholds, exposed so they can be retuned
        without editing this file.
    abs_tilt_deadband:
        Start of the independent absolute-posture slowdown ramp.
    """

    def __init__(self, schema, observation_joint_names, default_joint_positions, dt: float = 0.02,
                 *, leg_mode: str = "raise_low_corners", yaw_assist_max: float = 0.0,
                 wheel_max_common: float = WHEEL_MAX_COMMON, wheel_max_diff: float = WHEEL_MAX_DIFF,
                 settle_calls: int = SETTLE_CALLS, strict: bool = True,
                 enable_neutral_reference: bool = True, calib_window: int = CALIB_WINDOW,
                 calib_max_spread: float = CALIB_MAX_SPREAD, calib_max_rate: float = CALIB_MAX_RATE,
                 calib_max_reference_tilt: float = CALIB_MAX_REFERENCE_TILT,
                 abs_tilt_deadband: float = ABS_TILT_DEADBAND):
        super().__init__(schema, observation_joint_names, default_joint_positions, dt,
                         leg_mode=leg_mode, yaw_assist_max=yaw_assist_max,
                         wheel_max_common=wheel_max_common, wheel_max_diff=wheel_max_diff,
                         settle_calls=settle_calls, strict=strict)
        if int(calib_window) != calib_window or calib_window < 2:
            raise StabilityError("calib_window must be an integer >= 2")
        for name, value in (("calib_max_spread", calib_max_spread),
                            ("calib_max_rate", calib_max_rate),
                            ("calib_max_reference_tilt", calib_max_reference_tilt),
                            ("abs_tilt_deadband", abs_tilt_deadband)):
            if not np.isfinite(value) or value <= 0.0:
                raise StabilityError(f"{name} must be finite and > 0, got {value!r}")
        if not calib_max_reference_tilt < ABS_TILT_ABORT:
            raise StabilityError("calib_max_reference_tilt must stay below the absolute abort tilt")
        if not abs_tilt_deadband < ABS_TILT_ABORT:
            raise StabilityError("abs_tilt_deadband must stay below the absolute abort tilt")

        self.enable_neutral_reference = bool(enable_neutral_reference)
        self.calib_window = int(calib_window)
        self.calib_max_spread = float(calib_max_spread)
        self.calib_max_rate = float(calib_max_rate)
        self.calib_max_reference_tilt = float(calib_max_reference_tilt)
        self.abs_tilt_deadband = float(abs_tilt_deadband)
        self._reset_profile_state()

    # ------------------------------------------------------------------ state
    def _reset_profile_state(self) -> None:
        self._samples: list[tuple[np.ndarray, float]] = []
        self.calibrated = False
        self.calibration_reason = "not attempted yet"
        self._rotation = np.eye(3)
        self._reference = None          # unit gravity of the frozen neutral stance
        self._reference_tilt = None     # sin(tilt) of that reference
        self._reference_spread = None
        self._calibrated_at_call = None
        self._abs_tilt_f = None
        self._danger_latch = 0
        self._abs_gate_events = 0
        self._danger_events = 0
        self._last_profile_debug = {"state": "reset"}

    def reset(self) -> None:
        """Reset the baseline and drop the frozen reference.

        The reference describes one episode's settled stance, so a new episode
        must recalibrate rather than inherit the previous one.
        """
        super().reset()
        self._reset_profile_state()

    # ------------------------------------------------------------------- main
    def apply(self, action, proprio) -> np.ndarray:
        """Baseline ``apply`` on a reference-relative view of the same proprio."""
        if not self.enable_neutral_reference:
            out = super().apply(action, proprio)
            self._annotate(out, disabled=True)
            return out

        snapshot = self._snapshot(proprio)
        call_index = self.calls + 1  # the baseline increments self.calls itself

        if call_index <= self.settle_calls:
            if snapshot is not None:
                self._samples.append(snapshot)
                if len(self._samples) > self.calib_window:
                    del self._samples[:-self.calib_window]
            out = super().apply(action, proprio)
            if self.calls >= self.settle_calls:
                self._calibrate()
            self._annotate(out, snapshot=snapshot, settling=True)
            return out

        if self.settle_calls == 0 and self.calibration_reason == "not attempted yet":
            self.calibration_reason = "no settle window: settle_calls == 0"

        abs_tilt, abs_tilt_filtered, danger, reasons = self._absolute_state(snapshot)
        use_reference = self.calibrated and not danger

        if snapshot is None:
            # Malformed proprio: hand the caller's own object to the baseline so
            # its strict/non-strict boundary behaviour is untouched.
            derived_proprio = proprio
            abs_gate = 0.0
        else:
            derived_proprio = self._derived_proprio(proprio, use_reference)
            abs_gate = 0.0 if danger else _ramp_gate(abs_tilt_filtered, self.abs_tilt_deadband,
                                                     ABS_TILT_ABORT)

        derived_action, scaled = self._derived_action(action, abs_gate)
        if abs_gate < 1.0:
            self._abs_gate_events += 1
        out = super().apply(derived_action, derived_proprio)
        self._annotate(out, snapshot=snapshot, abs_tilt=abs_tilt, abs_tilt_filtered=abs_tilt_filtered,
                       abs_gate=abs_gate, danger=danger, reasons=reasons,
                       use_reference=use_reference, wheels_scaled=scaled)
        return out

    # -------------------------------------------------------------- internals
    def _snapshot(self, proprio):
        """Read raw unit gravity and raw |(w_x, w_y)| without touching the caller's array."""
        try:
            raw = np.asarray(proprio, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if raw.size != self.layout.size:
            return None
        gravity = raw[self.layout.projected_gravity]
        angvel = raw[self.layout.base_ang_vel]
        if not np.isfinite(gravity).all() or not np.isfinite(angvel).all():
            return None
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or not 0.5 <= norm <= 1.5:
            return None
        return np.array(gravity, dtype=np.float64) / norm, float(np.hypot(angvel[0], angvel[1]))

    def _calibrate(self) -> None:
        """Freeze the neutral reference once, or record why it was refused."""
        if self.calibrated or self._calibrated_at_call is not None:
            return
        self._calibrated_at_call = self.calls
        if len(self._samples) < self.calib_window:
            self.calibration_reason = (
                f"only {len(self._samples)} usable settle samples, need {self.calib_window}")
            return

        window = self._samples[-self.calib_window:]
        vectors = np.stack([sample[0] for sample in window])
        rates = np.array([sample[1] for sample in window])
        max_rate = float(np.max(rates))
        if max_rate > self.calib_max_rate:
            self.calibration_reason = (
                f"settle window not calm: max |(w_x,w_y)| {max_rate:.4f} > {self.calib_max_rate}")
            return

        mean = vectors.mean(axis=0)
        coherence = float(np.linalg.norm(mean))
        if coherence < CALIB_MIN_COHERENCE:
            self.calibration_reason = f"gravity samples incoherent: |mean| {coherence:.6f}"
            return
        reference = mean / coherence
        spread = float(np.max(np.arccos(np.clip(vectors @ reference, -1.0, 1.0))))
        if spread > self.calib_max_spread:
            self.calibration_reason = (
                f"gravity still moving: spread {spread:.4f} rad > {self.calib_max_spread}")
            return

        tilt = float(np.hypot(reference[0], reference[1]))
        if reference[2] >= 0.0 or tilt > self.calib_max_reference_tilt:
            self.calibration_reason = (
                f"reference is not a near-down vector: sin(tilt) {tilt:.4f}, g_z {reference[2]:.4f}")
            return

        rotation = _shortest_rotation(reference, _DOWN)
        if rotation is None:
            self.calibration_reason = "could not build a proper rotation to level"
            return

        self._rotation = rotation
        self._reference = reference
        self._reference_tilt = tilt
        self._reference_spread = spread
        self.calibrated = True
        self.calibration_reason = (
            f"frozen at call {self.calls} from {self.calib_window} samples: "
            f"sin(tilt) {tilt:.4f}, spread {spread:.4f} rad, max rate {max_rate:.4f} rad/s")

    def _absolute_state(self, snapshot):
        """Absolute tilt, its EMA, and whether the independent envelope trips."""
        reasons: list[str] = []
        if snapshot is None:
            self._danger_latch = DANGER_LATCH_CALLS
            self._danger_events += 1
            reasons.append("unusable proprio")
            previous = self._abs_tilt_f if self._abs_tilt_f is not None else float("nan")
            return float("nan"), previous, True, reasons

        gravity, _ = snapshot
        abs_tilt = float(np.hypot(gravity[0], gravity[1]))
        if self._abs_tilt_f is None:
            self._abs_tilt_f = abs_tilt
        else:
            self._abs_tilt_f += _ema_alpha(self.dt, GRAVITY_TAU) * (abs_tilt - self._abs_tilt_f)
        filtered = float(self._abs_tilt_f)

        if gravity[2] >= 0.0:
            reasons.append("gravity outside downward hemisphere (inverted or horizontal)")
        if abs_tilt >= ABS_TILT_ABORT:
            reasons.append(f"absolute sin(tilt) {abs_tilt:.4f} >= {ABS_TILT_ABORT}")
        if filtered >= ABS_TILT_ABORT:
            reasons.append(f"filtered absolute sin(tilt) {filtered:.4f} >= {ABS_TILT_ABORT}")

        if reasons:
            self._danger_latch = DANGER_LATCH_CALLS
            self._danger_events += 1
        elif self._danger_latch > 0:
            # Latched: hold the raw frame until it is quiet again and the latch runs out.
            self._danger_latch -= 1
            if filtered > self.abs_tilt_deadband:
                self._danger_latch = DANGER_LATCH_CALLS
            reasons.append("absolute-envelope latch still active")
        return abs_tilt, filtered, bool(reasons), reasons

    def _derived_proprio(self, proprio, use_reference: bool):
        """A copy with only the gravity slice rotated; the caller's array is untouched."""
        if not use_reference:
            return proprio
        raw = np.array(proprio, dtype=np.float64).reshape(-1)
        gravity = raw[self.layout.projected_gravity]
        # The rotation is orthonormal, so the norm the baseline validates is preserved.
        raw[self.layout.projected_gravity] = self._rotation @ gravity
        return raw

    def _derived_action(self, action, abs_gate: float):
        """Scale only the wheel request by the absolute gate; arms/legs are bit-identical."""
        try:
            derived = np.array(action, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return action, False
        if derived.size != self.schema.total_dim or not np.isfinite(derived).all():
            return action, False  # let the baseline raise its own error on the original
        if np.any(np.abs(derived) > np.finfo(np.float32).max):
            # Validate the original request before a zero absolute gate can
            # hide an unrepresentable wheel value. The baseline owns errors.
            return action, False
        if abs_gate >= 1.0:
            return action, False  # bit-exact baseline path
        request = derived[self._wheel_indices]
        if np.all(request == 0.0):
            return action, False  # keep the native zero-request immediate stop intact
        derived[self._wheel_indices] = request * abs_gate
        return derived, True

    def _annotate(self, out, *, snapshot=None, settling: bool = False, disabled: bool = False,
                  abs_tilt=None, abs_tilt_filtered=None, abs_gate=1.0, danger=False,
                  reasons=None, use_reference=False, wheels_scaled=False) -> None:
        """Attach both absolute and reference-relative diagnostics to ``self.debug``."""
        if snapshot is not None and abs_tilt is None:
            abs_tilt = float(np.hypot(snapshot[0][0], snapshot[0][1]))
        relative = self.debug.get("tilt", {}) if isinstance(self.debug, dict) else {}
        record = {
            "profile": "neutral" if self.enable_neutral_reference else "baseline",
            "enabled": self.enable_neutral_reference,
            "settling": bool(settling),
            "frame_used_by_baseline": "reference_relative" if use_reference else "absolute_raw",
            "calibrated": self.calibrated,
            "calibration_reason": self.calibration_reason,
            "calibration_samples": len(self._samples),
            "reference": {
                "unit_gravity": None if self._reference is None else self._reference.tolist(),
                "sin_tilt": self._reference_tilt,
                "tilt_deg": (None if self._reference_tilt is None
                             else float(np.degrees(np.arcsin(min(1.0, self._reference_tilt))))),
                "sample_spread_rad": self._reference_spread,
                "frozen_at_call": self._calibrated_at_call,
                "rotation_to_level": self._rotation.tolist(),
            },
            "absolute": {
                # Non-finite values are reported as null so the record stays
                # strictly JSON-serializable.
                "sin_tilt": _json_float(abs_tilt),
                "tilt_deg": (None if snapshot is None else float(np.degrees(np.arctan2(
                    np.hypot(snapshot[0][0], snapshot[0][1]), -snapshot[0][2])))),
                "sin_tilt_filtered": _json_float(abs_tilt_filtered),
                "unit_gravity": None if snapshot is None else snapshot[0].tolist(),
                "tilt_rate_rad_s": None if snapshot is None else float(snapshot[1]),
                "deadband": self.abs_tilt_deadband,
                "abort": ABS_TILT_ABORT,
                "gate": float(abs_gate),
                "danger": bool(danger),
                "reasons": list(reasons or []),
                "latch_calls_left": int(self._danger_latch),
                "gate_events": self._abs_gate_events,
                "danger_events": self._danger_events,
            },
            "relative": {
                "sin_tilt": relative.get("sin_tilt"),
                "tilt_deg": relative.get("tilt_deg"),
                "gravity_xy_filtered": relative.get("gravity_xy_filtered"),
            },
            "wheel_request_scaled_by_absolute_gate": bool(wheels_scaled),
            "note": "baseline tilt/gate fields are reference-relative whenever "
                    "frame_used_by_baseline == 'reference_relative'; the true absolute posture is "
                    "always reported under 'absolute'",
        }
        if disabled:
            record["note"] = "neutral reference disabled; identical to the baseline controller"
        if isinstance(self.debug, dict):
            self.debug["neutral_reference"] = record
            if isinstance(self.debug.get("tilt"), dict):
                self.debug["tilt"]["frame"] = record["frame_used_by_baseline"]
                self.debug["tilt"]["absolute_sin_tilt"] = record["absolute"]["sin_tilt"]
        self._last_profile_debug = record

    # ----------------------------------------------------------------- output
    def profile_debug(self) -> dict:
        """Copy of the most recent profile-level record (also inside ``debug``)."""
        return dict(self._last_profile_debug)

    def describe(self) -> dict:
        spec = super().describe()
        spec["controller"] = "NeutralAwareStabilityController"
        spec["profile"] = "neutral" if self.enable_neutral_reference else "baseline"
        spec["base_controller"] = "StabilityController (frozen; unmodified)"
        spec["neutral_reference"] = {
            "enabled": self.enable_neutral_reference,
            "purpose": "the settled B2wPiper stance measured sin(tilt) = 0.0633, above the baseline "
                       "TILT_DEADBAND of 0.06, so the baseline gated 366 calls of a healthy pose and "
                       "sank to the 0.30 authority floor",
            "calibration": {
                "window": self.calib_window,
                "source": "public projected_gravity during the tail of the zero-action settle window",
                "frozen_once": True,
                "max_spread_rad": self.calib_max_spread,
                "max_rate_rad_s": self.calib_max_rate,
                "max_reference_sin_tilt": self.calib_max_reference_tilt,
                "min_coherence": CALIB_MIN_COHERENCE,
                "on_refusal": "runs as the uncalibrated baseline and reports the reason",
                "status": self.calibration_reason,
                "calibrated": self.calibrated,
                "reference_unit_gravity": None if self._reference is None else self._reference.tolist(),
                "reference_sin_tilt": self._reference_tilt,
            },
            "absolute_envelope": {
                "deadband": self.abs_tilt_deadband,
                "abort": ABS_TILT_ABORT,
                "latch_calls": DANGER_LATCH_CALLS,
                "action": "wheel request scaled by a ramp between deadband and abort; on abort or "
                          "inverted gravity the baseline is fed the raw (unrotated) proprio and a "
                          "zero wheel request for immediate wheel stop; the baseline abort state "
                          "is determined independently by its raw-gravity checks and filtered tilt",
                "rates": "base angular velocity is never rewritten, so the baseline's rate gate and "
                         "yaw readings stay raw",
            },
            "unchanged": [
                "all baseline gains, caps, slew rates, leg limits and abort thresholds",
                "arm entries are the incoming bytes",
                "an all-zero wheel request still stops immediately and clears wheel memory",
                "float32, finiteness, gravity-norm and inverted-gravity boundaries",
            ],
        }
        spec.setdefault("limitations", []).append(
            "Neutral profile: unverified. The frozen reference removes a constant pitch offset, so "
            "reference-relative tilt is NOT the true tilt; read 'absolute' in the debug record.")
        spec["limitations"].extend([
            "Neutral profile: the leg-levelling downhill direction is computed in the rotated "
            "reference frame, so corner selection is approximate to within the reference tilt "
            "(~3.6 deg for the recorded stance) and can mis-select near the deadband.",
            "Neutral profile: the rotation is the shortest proper rotation to level, which fixes no "
            "yaw; it is a stance offset, not an orientation estimate.",
            "Neutral profile: calibration assumes the settle window ends on flat, stationary ground. "
            "A start on a real slope up to the reference-tilt limit would be absorbed into the "
            "reference; only the absolute envelope still sees it.",
        ])
        return spec


def _shortest_rotation(source, target):
    """Shortest proper rotation matrix taking unit ``source`` to unit ``target``."""
    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    axis = np.cross(source, target)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(np.dot(source, target))
    if sin_angle < 1e-12:
        if cos_angle <= 0.0:
            return None  # antipodal: no shortest rotation, and not a near-down reference anyway
        rotation = np.eye(3)
    else:
        skew = np.array([[0.0, -axis[2], axis[1]],
                         [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cos_angle) / (sin_angle ** 2))
    if not np.isfinite(rotation).all():
        return None
    if np.max(np.abs(rotation @ rotation.T - np.eye(3))) > 1e-9:
        return None
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-9:
        return None
    if float(np.linalg.norm(rotation @ source - target)) > 1e-9:
        return None
    return rotation


def build_controller(profile: str, schema, observation_joint_names, default_joint_positions,
                     dt: float = 0.02, **kwargs):
    """Construct the controller for ``--stability_profile {baseline,neutral}``."""
    if profile not in PROFILES:
        raise StabilityError(f"Unknown stability profile {profile!r}; expected one of {list(PROFILES)}")
    if profile == "baseline":
        for name in ("enable_neutral_reference", "calib_window", "calib_max_spread",
                     "calib_max_rate", "calib_max_reference_tilt", "abs_tilt_deadband"):
            if name in kwargs:
                raise StabilityError(f"'{name}' is not a baseline argument")
        return StabilityController(schema, observation_joint_names, default_joint_positions, dt,
                                   **kwargs)
    return NeutralAwareStabilityController(schema, observation_joint_names, default_joint_positions,
                                           dt, **kwargs)
