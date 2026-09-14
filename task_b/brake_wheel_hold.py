"""Wheel-angle position hold for the brake phase of a Task B mobility episode.

Context (from the real P1 failure): the wheel actuators are configured with
stiffness=0 and damping=1, so a velocity command of 0 does not lock the wheel
position.  After the base is braked to a stop the chassis still rolls backwards
by 3.6 / 10.5 / 11.8 cm.  This module closes a small proportional-derivative
loop on the *publicly available wheel joint angles* only, and emits a single
physical wheel speed target in rad/s.  It does not touch the official actuator
configuration, and it reads no world pose, ground truth, reward, camera or
Isaac state.

The emitted command is always in physical rad/s.  Any action-space
normalization (schema scale, wheel gain, ...) is the caller's job and lives in
``task_b/evaluate.py``; this module never applies it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

NUM_WHEELS = 4


def _positive_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}")
    return out


def _non_negative_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float, got {value!r}")
    return out


def _wheel_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Return a strict ``(4,)`` finite float array, rejecting nested shapes."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric length-{NUM_WHEELS} vector") from exc
    if arr.ndim != 1 or arr.shape[0] != NUM_WHEELS:
        raise ValueError(f"{name} must have shape ({NUM_WHEELS},), got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite, got {arr.tolist()}")
    return arr


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi] so raw wheel angles jumping by +-2pi/+-4pi are equal."""
    return math.atan2(math.sin(angle), math.cos(angle))


class BrakeWheelHold:
    """Hold the four wheel angles at the position captured when braking ended.

    Control law, per :meth:`update` call::

        e_i          = wrap(anchor_i - q_i)                 # per wheel, radians
        raw_desired  = kp * mean(e) - kd * mean(qdot)
        desired      = clip(raw_desired, -max_speed, +max_speed)
        command     += clip(desired - command, -slew*dt, +slew*dt)

    ``command`` is one scalar shared by all four wheels with the same sign; the
    module has no notion of left/right sides because it only resists rolling.
    """

    def __init__(
        self,
        dt: float = 0.02,
        kp: float = 2.0,
        kd: float = 0.25,
        max_speed: float = 0.6,
        slew: float = 2.0,
    ) -> None:
        self.dt = _positive_float(dt, "dt")
        self.kp = _positive_float(kp, "kp")
        self.kd = _non_negative_float(kd, "kd")
        self.max_speed = _positive_float(max_speed, "max_speed")
        self.slew = _positive_float(slew, "slew")

        self._anchor: Optional[np.ndarray] = None
        self._active = False
        self._last_command = 0.0
        self._debug: Dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self._active

    @property
    def anchor(self) -> Optional[List[float]]:
        return None if self._anchor is None else self._anchor.tolist()

    @property
    def last_command(self) -> float:
        return self._last_command

    @property
    def debug(self) -> Dict[str, Any]:
        return dict(self._debug)

    def engage(self, wheel_positions: Sequence[float]) -> None:
        """Latch the current wheel angles as the hold target.

        Calling :meth:`engage` while already active is a no-op: the anchor is
        never re-latched mid-hold, otherwise the accumulated rollback would be
        silently forgiven.
        """
        positions = _wheel_vector(wheel_positions, "wheel_positions")
        if self._active:
            return
        self._anchor = positions.copy()
        self._active = True
        self._last_command = 0.0
        self._debug = {}

    def release(self) -> None:
        """Drop the anchor and stop commanding; later updates return 0.0."""
        self._anchor = None
        self._active = False
        self._last_command = 0.0
        self._debug = {}

    def update(self, wheel_positions: Sequence[float], wheel_velocities: Sequence[float]) -> float:
        """Return the shared physical wheel speed target in rad/s."""
        if not self._active or self._anchor is None:
            return 0.0

        positions = _wheel_vector(wheel_positions, "wheel_positions")
        velocities = _wheel_vector(wheel_velocities, "wheel_velocities")

        errors = [_wrap(float(a) - float(q)) for a, q in zip(self._anchor, positions)]
        mean_error = float(np.mean(errors))
        mean_qdot = float(np.mean(velocities))

        raw_desired = self.kp * mean_error - self.kd * mean_qdot
        desired = float(np.clip(raw_desired, -self.max_speed, self.max_speed))
        step = float(np.clip(desired - self._last_command, -self.slew * self.dt, self.slew * self.dt))
        self._last_command = self._last_command + step

        self._debug = {
            "wrapped_errors": [float(e) for e in errors],
            "mean_error": mean_error,
            "mean_qdot": mean_qdot,
            "raw_desired": float(raw_desired),
            "command_rad_s": float(self._last_command),
            "saturated": bool(abs(raw_desired) > self.max_speed),
            "active": True,
        }
        return float(self._last_command)

    def describe(self) -> Dict[str, Any]:
        """Human-readable summary; JSON-serializable, claims no score."""
        return {
            "name": "brake_wheel_hold",
            "problem": (
                "wheel actuators use stiffness=0 / damping=1, so a zero velocity "
                "command does not lock wheel position; the base rolled back "
                "3.6 / 10.5 / 11.8 cm after braking in the real P1 run"
            ),
            "parameters": {
                "dt": self.dt,
                "kp": self.kp,
                "kd": self.kd,
                "max_speed": self.max_speed,
                "slew": self.slew,
            },
            "output": {
                "quantity": "wheel angular velocity target",
                "unit": "rad/s (physical, never normalized here)",
                "shape": "scalar shared by all four wheels, identical sign",
                "range_rad_s": [-self.max_speed, self.max_speed],
                "slew_per_step_rad_s": self.slew * self.dt,
            },
            "inputs": {
                "wheel_positions": f"public wheel joint angles, shape ({NUM_WHEELS},), radians, may jump by +-2pi",
                "wheel_velocities": f"public wheel joint velocities, shape ({NUM_WHEELS},), rad/s",
            },
            "not_used": [
                "world pose / root state",
                "ground truth targets",
                "reward signals",
                "camera images",
                "Isaac Sim internals",
                "official actuator configuration (unmodified)",
            ],
            "normalization": (
                "none applied here; action-space scale and wheel gain belong to "
                "task_b/evaluate.py"
            ),
            "status": (
                "experimental brake-phase hold; not yet validated in simulation "
                "and no scored result is claimed"
            ),
        }
