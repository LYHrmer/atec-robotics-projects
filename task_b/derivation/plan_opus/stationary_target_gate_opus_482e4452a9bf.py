"""Post-stop visual target confirmation gate for the Task B BRAKE phase.

The gate answers one question: *may the caller trust a detected target point as
a settled, post-stop observation?* It owns no action, no IK, no navigation and
no state machine. It reads only what the caller passes in: a control step index,
a public "quiet" flag the caller computes from public body/joint velocities, and
optionally one raw detection point in the body frame with its camera id and, if
the sensor exposes one, a frame token. It never touches ground truth, object
roots, rewards or the simulator.

Why it exists: at 10 Hz a detection consumed right after the chassis stops can
still be an image captured while the body was moving, and a single frame cannot
be distinguished from a re-read of the same frame. So the gate requires
(1) continuous quiet for ``quiet_required_s``, (2) one further full
``sensor_period`` so the last in-motion image is flushed, and (3) two accepted
samples from the *same* camera, at least one sensor period apart, within
``match_tol_m`` of each other. A spatial jump restarts confirmation rather than
being averaged into a fictitious target.

Freshness honesty: when no ``frame_token`` is supplied the gate cannot prove a
frame is new. It does not invent a frame id and it does not require the pixels
to change (a stationary scene legitimately produces identical images). It states
its basis as ``'sensor_period_elapsed_assumption'``: the only evidence is that
the caller handed in a new detection sample and that enough control time has
elapsed for the sensor to have produced a new frame.

Scope limit: this gate is for BRAKE-phase standstill localization only. It must
**not** be used during REACH: while the arm sweeps down the target is expected
to leave the field of view or fall below the detector's minimum depth, and
in-motion loss monitoring belongs to the main state machine.

No simulation has been run with this module.
"""
from __future__ import annotations

import math
import operator
from statistics import median

QUIET_REQUIRED_S = 0.20      #: continuous quiet before the settle wait starts
MATCH_TOL_M = 0.04           #: max distance between two confirming samples
STALE_S = 0.50               #: age at which a confirmation stops being usable
MIN_CONFIRMATIONS = 2        #: accepted same-source samples needed for ready
MAX_ABS_COORD_M = 1.0e3      #: a larger finite coordinate is a caller bug
MAX_SAMPLES = 5              #: samples kept for the target estimate
TOL = 1.0e-9


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


class StationaryTargetGate:
    """Confirm a stationary target point observed after the chassis has stopped.

    ``update`` always returns a JSON-safe record with ``ready``,
    ``target_body`` (a fresh ``list`` of 3 floats or ``None``),
    ``confirmations``, ``last_fresh_step``, ``age_s``, ``reason`` and
    ``freshness_basis``, plus diagnostics. Nothing is shared with the caller:
    input sequences are copied into tuples and every returned list is new.
    """

    def __init__(self, dt: float = 0.02, sensor_period: float = 0.1, *,
                 quiet_required_s: float = QUIET_REQUIRED_S, match_tol_m: float = MATCH_TOL_M,
                 stale_s: float = STALE_S, min_confirmations: int = MIN_CONFIRMATIONS):
        self.dt = _positive("dt", dt)
        self.sensor_period = _positive("sensor_period", sensor_period)
        self.quiet_required_s = _positive("quiet_required_s", quiet_required_s)
        self.match_tol_m = _positive("match_tol_m", match_tol_m)
        self.stale_s = _positive("stale_s", stale_s)
        if int(min_confirmations) != min_confirmations or min_confirmations < 2:
            raise ValueError("min_confirmations must be an integer >= 2")
        if self.sensor_period < self.dt:
            raise ValueError("sensor_period must be >= dt")
        if self.stale_s < self.sensor_period:
            raise ValueError("stale_s must be >= sensor_period")
        self.min_confirmations = int(min_confirmations)
        self.reset(0)

    # ---------------------------------------------------------------- validation
    @staticmethod
    def _as_step(step) -> int:
        try:
            value = operator.index(step)
        except TypeError as error:
            raise ValueError(f"step must be an integer control call index, got {step!r}") from error
        if value < 0:
            raise ValueError(f"step must be >= 0, got {value}")
        return value

    @staticmethod
    def _as_point(point_body) -> tuple:
        """Copy a 3-element finite point into a tuple; never keep the caller's object."""
        if isinstance(point_body, (str, bytes, dict)):
            raise ValueError(f"point_body must be a 3-element sequence, got {type(point_body).__name__}")
        try:
            values = list(point_body)
        except TypeError as error:
            raise ValueError(f"point_body must be a 3-element sequence, got {point_body!r}") from error
        if len(values) != 3:
            raise ValueError(f"point_body must have 3 elements, got {len(values)}")
        out = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"point_body elements must be real numbers, got {value!r}") from error
            if not math.isfinite(number) or abs(number) > MAX_ABS_COORD_M:
                raise ValueError(f"point_body elements must be finite and bounded, got {value!r}")
            out.append(number)
        return tuple(out)

    @staticmethod
    def _as_source(source) -> str:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("a detection point requires a non-empty string source (camera id)")
        return source

    # --------------------------------------------------------------------- state
    def reset(self, step) -> None:
        """Clear every in-motion target, count and time. Call on BRAKE entry."""
        step = self._as_step(step)
        self._min_step = step
        self._last_step = None
        self._quiet_since = None
        self._source = None
        self._token = None
        self._clear_evidence()
        self._last_reason = "reset"
        self._last_quiet_s = 0.0

    def _clear_evidence(self) -> None:
        self._samples: list = []          # (step, point tuple, had_token)
        self._confirmations = 0
        self._target = None
        self._ready = False
        self._last_fresh_step = None

    def _expire(self, step) -> bool:
        """Drop a confirmation whose newest accepted sample is older than ``stale_s``."""
        if self._last_fresh_step is None:
            return False
        if (step - self._last_fresh_step) * self.dt > self.stale_s + TOL:
            had = self._samples or self._ready
            self._samples, self._confirmations, self._target, self._ready = [], 0, None, False
            return bool(had) or True
        return False

    # ---------------------------------------------------------------------- main
    def update(self, step, quiet, point_body=None, source=None, frame_token=None) -> dict:
        """Advance one control call and return the gate record.

        ``point_body`` is passed only when this 10 Hz tick produced a new raw
        detection; ``None`` is an ordinary non-detection tick and never resets an
        existing confirmation. A repeated ``step`` is ignored (state unchanged);
        a decreasing ``step`` raises ``ValueError``.
        """
        step = self._as_step(step)
        if step < self._min_step:
            raise ValueError(f"step {step} precedes the reset step {self._min_step}")
        if self._last_step is not None:
            if step < self._last_step:
                raise ValueError(f"step went backwards: {step} after {self._last_step}")
            if step == self._last_step:
                return self._record("duplicate_step_ignored", step)
        self._last_step = step
        quiet = bool(quiet)
        point = None if point_body is None else self._as_point(point_body)
        if point is not None:
            source = self._as_source(source)

        if not quiet:
            self._clear_evidence()
            self._quiet_since = None
            self._source = None
            self._token = None
            self._last_quiet_s = 0.0
            return self._record("not_quiet", step)

        if self._quiet_since is None:
            self._quiet_since = step
        self._last_quiet_s = (step - self._quiet_since) * self.dt
        if self._last_quiet_s + TOL < self.quiet_required_s + self.sensor_period:
            phase = ("settling_quiet" if self._last_quiet_s + TOL < self.quiet_required_s
                     else "settling_sensor_period")
            return self._record("pre_settle_point_discarded" if point is not None else phase, step)

        if point is None:
            stale = self._expire(step)
            return self._record("stale_evidence" if stale
                                else ("confirmed_hold" if self._ready else "no_candidate"), step)

        switched = source != self._source
        if switched:
            self._clear_evidence()
            self._source = source
            self._token = None
        if frame_token is not None and self._token is not None and frame_token == self._token:
            stale = self._expire(step)
            return self._record("stale_evidence" if stale else "repeated_frame_token", step)

        jumped = False
        if self._samples:
            prev_step, prev_point, _ = self._samples[-1]
            if (step - prev_step) * self.dt + TOL < self.sensor_period:
                if frame_token is not None:
                    self._token = frame_token
                self._expire(step)
                return self._record("sample_too_soon", step)
            if math.dist(point, prev_point) > self.match_tol_m:
                self._clear_evidence()
                jumped = True
        if frame_token is not None:
            self._token = frame_token
        self._samples.append((step, point, frame_token is not None))
        del self._samples[:-MAX_SAMPLES]
        self._confirmations += 1
        self._last_fresh_step = step
        self._ready = self._confirmations >= self.min_confirmations
        self._target = self._estimate() if self._ready else None
        if self._ready:
            reason = "confirmed"
        elif jumped:
            reason = "spatial_jump_restart"
        elif switched:
            reason = "source_changed"
        else:
            reason = "first_sample"
        return self._record(reason, step)

    # ----------------------------------------------------------------- internals
    def _estimate(self) -> tuple:
        """Per-axis median of the kept accepted samples (mean of the middle pair)."""
        return tuple(float(median([sample[1][axis] for sample in self._samples])) for axis in range(3))

    def _record(self, reason: str, step: int) -> dict:
        """Build a fresh JSON-safe record; no state is shared with the caller."""
        self._last_reason = reason
        age = (None if self._last_fresh_step is None
               else float((step - self._last_fresh_step) * self.dt))
        tokened = bool(self._samples) and all(sample[2] for sample in self._samples)
        return {
            "ready": bool(self._ready),
            "target_body": None if self._target is None else [float(v) for v in self._target],
            "confirmations": int(self._confirmations),
            "last_fresh_step": None if self._last_fresh_step is None else int(self._last_fresh_step),
            "age_s": age,
            "reason": reason,
            "freshness_basis": "distinct_frame_token" if tokened else "sensor_period_elapsed_assumption",
            "step": int(step),
            "source": self._source,
            "quiet_s": float(self._last_quiet_s),
            "settled": bool(self._last_quiet_s + TOL >= self.quiet_required_s + self.sensor_period),
            "sample_steps": [int(sample[0]) for sample in self._samples],
            "params": {"dt": self.dt, "sensor_period": self.sensor_period,
                       "quiet_required_s": self.quiet_required_s, "match_tol_m": self.match_tol_m,
                       "stale_s": self.stale_s, "min_confirmations": self.min_confirmations},
            "scope": "BRAKE standstill confirmation only; not valid during REACH arm motion",
            "claims": "confirms a post-stop detection sample only; no ground truth, IK, action or "
                      "pixel-change requirement, and no verified Task B score",
        }
