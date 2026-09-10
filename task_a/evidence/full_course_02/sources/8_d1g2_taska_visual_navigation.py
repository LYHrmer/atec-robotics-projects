"""Visual pose anchors with short-term proprioceptive heading prediction.

The visual estimates use the initial camera/robot frame supplied by visual
odometry. No simulator world pose or body-linear-velocity odometry is used.
"""

from __future__ import annotations

import numpy as np


class VisualNavigator:
    """Follow initial-frame y=0 and stop when visual updates become stale.

    Pass a visual result to ``observe_visual`` before calling ``command`` for
    the same observation timestamp. An accepted visual anchor already includes
    motion up to that timestamp, so the next command only seeds the gyro sample;
    it does not integrate that interval again. Subsequent samples integrate yaw
    using the ZYX Euler yaw rate and trapezoidal integration. XY changes only on
    accepted visual updates. The first visual initialization anchors known zero.
    """

    def __init__(self):
        self.metadata = {
            "kind": "visual_anchors_with_proprioceptive_heading_prediction",
            "visual_fields": ["accepted", "reason", "estimated_xy", "estimated_yaw"],
            "gyro_indices": [3, 6],
            "gravity_indices": [9, 12],
            "initial_xy": [0.0, 0.0],
            "initial_yaw": 0.0,
            "slow_after_seconds": 0.3,
            "stop_after_seconds": 0.8,
            "slowdown": "linear speed reduction between 0.3 and 0.8 seconds",
            "uses_simulator_world_pose": False,
            "uses_body_linear_velocity": False,
        }
        self.reset()

    def reset(self):
        self.estimated_xy = np.zeros(2, dtype=np.float64)
        self.estimated_yaw = 0.0
        self.target_yaw = 0.0
        self.yaw_rate = 0.0
        self.visual_age_seconds = 0.0
        self.visual_initialized = False
        self.visual_update_count = 0
        self.last_visual_reason = "not_initialized"
        self.speed_scale = 0.0
        self.stopped_for_visual_loss = True
        self._previous_yaw_rate = None
        self._heading_anchor_fresh = False
        self.independent_heading_updates = 0

    def observe_visual(self, result):
        """Accept a same-timestamp VO anchor; rejected VO never updates XY.

        Returns True for an accepted anchor or known-zero initialization.
        ``reason='initialized'`` is the start-of-episode visual reference and
        anchors zero; arbitrary nonzero pose fields are not treated as known.
        """
        reason = str(result.get("reason", ""))
        initialized = reason == "initialized"
        if not initialized and not bool(result.get("accepted", False)):
            self.last_visual_reason = reason
            return False
        if initialized:
            xy = np.zeros(2, dtype=np.float64)
            yaw = 0.0
        else:
            xy = np.asarray(result["estimated_xy"], dtype=np.float64)
            yaw = float(result["estimated_yaw"])
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(yaw):
                raise ValueError("Accepted visual pose must contain finite XY and yaw")
        self.estimated_xy = xy.copy()
        self.estimated_yaw = yaw
        self.visual_age_seconds = 0.0
        self._previous_yaw_rate = None
        self._heading_anchor_fresh = False
        self.visual_initialized = True
        self.visual_update_count += 1
        self.last_visual_reason = reason
        self.stopped_for_visual_loss = False
        return True

    def observe_heading(self, result):
        """Use a same-timestamp image heading without refreshing XY or its age."""
        if not result or not result.get('accepted', False):
            return False
        yaw = float(result['heading_rad'])
        if not np.isfinite(yaw):
            raise ValueError('Accepted heading must be finite')
        self.estimated_yaw = yaw
        self._heading_anchor_fresh = True
        self.independent_heading_updates += 1
        return True

    def command(self, proprio, speed, dt):
        """Return float32 [forward_speed, 0, yaw_command] for one new sample."""
        observation = np.asarray(proprio, dtype=np.float64)
        if observation.shape != (81,):
            raise ValueError(f"Expected proprio shape (81,), got {observation.shape}")
        speed, dt = float(speed), float(dt)
        if not np.isfinite(speed) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("speed must be finite and dt must be positive and finite")
        angular = observation[3:6]
        gravity = observation[9:12]
        if not np.isfinite(np.r_[angular, gravity]).all():
            raise ValueError("Gyro and gravity observations must be finite")
        gravity_norm = float(np.linalg.norm(gravity))
        if not np.isfinite(gravity_norm) or gravity_norm < 1.0e-9:
            raise ValueError("Projected gravity must have a finite nonzero norm")
        _, gy, gz = gravity / gravity_norm
        denominator = float(gy * gy + gz * gz)
        if denominator < 1.0e-4:
            raise ValueError("Proprioceptive yaw is singular near vertical pitch")
        rate = float((-gy * angular[1] - gz * angular[2]) / denominator)
        yaw = self.estimated_yaw
        age = self.visual_age_seconds
        if self._previous_yaw_rate is not None:
            if not self._heading_anchor_fresh:
                yaw += 0.5 * (self._previous_yaw_rate + rate) * dt
            age += dt
        self._heading_anchor_fresh = False
        target_yaw = float(np.clip(
            np.arctan2(-0.15 * self.estimated_xy[1], max(speed, 0.2)), -0.25, 0.25
        ))
        error = float((target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
        stopped = not self.visual_initialized or age > 0.8
        scale = 0.0 if stopped else float(np.clip((0.8 - age) / 0.5, 0.0, 1.0))
        yaw_command = 0.0 if stopped else float(np.clip(error, -0.5, 0.5))
        action = np.asarray([speed * scale, 0.0, yaw_command], dtype=np.float32)
        if not np.isfinite(np.r_[yaw, age, rate, action]).all():
            raise ValueError("Visual navigation produced a nonfinite result")
        self.estimated_yaw = float(yaw)
        self.yaw_rate = rate
        self.visual_age_seconds = float(age)
        self.target_yaw = target_yaw
        self._previous_yaw_rate = rate
        self.speed_scale = scale
        self.stopped_for_visual_loss = stopped
        return action

    def state_dict(self):
        """Return independent, JSON-compatible estimates and visual status."""
        return {
            "estimated_xy": self.estimated_xy.tolist(),
            "estimated_yaw": float(self.estimated_yaw),
            "target_yaw": float(self.target_yaw),
            "yaw_rate": float(self.yaw_rate),
            "visual_age_seconds": float(self.visual_age_seconds),
            "visual_initialized": self.visual_initialized,
            "visual_update_count": self.visual_update_count,
            "last_visual_reason": self.last_visual_reason,
            "speed_scale": float(self.speed_scale),
            "stopped_for_visual_loss": self.stopped_for_visual_loss,
            "independent_heading_updates": self.independent_heading_updates,
        }
