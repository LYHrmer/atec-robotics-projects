"""Proprioceptive heading and lateral odometry for D1+G2 Task A.

The inertial frame starts at the robot's known initial position and zero yaw.
Only body velocity, body angular velocity and projected gravity are consumed;
this module does not read simulator poses, terrain geometry or scoring state.
Odometry can drift and is an estimate, not an absolute position measurement.
"""

from __future__ import annotations

import numpy as np


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class D1G2ProprioNavigator:
    """Keep the robot near initial-frame y=0 using its supplied proprioception.

    Call once for each observation, before applying that observation's command.
    The first call establishes the t=0 velocity/rate samples without integrating
    an interval. Subsequent calls integrate the interval since the previous
    sample, using trapezoidal integration for both yaw and planar velocity.
    ``reset()`` is required when the environment starts a new episode.
    """

    def __init__(self):
        self.metadata = {
            "kind": "proprioceptive_heading_and_lateral_odometry",
            "observation_size": 81,
            "inputs": {
                "body_linear_velocity": [0, 3],
                "body_angular_velocity": [3, 6],
                "projected_gravity": [9, 12],
            },
            "initial_xy": [0.0, 0.0],
            "initial_yaw": 0.0,
            "frame": "inertial_frame_at_episode_start",
            "yaw_convention": "ZYX Euler yaw, stored unwrapped",
            "integration": "trapezoidal; first sample establishes time zero",
            "lateral_target": 0.0,
            "lateral_gain": 0.15,
            "heading_gain": 1.0,
            "target_yaw_limit": 0.25,
            "yaw_command_limit": 0.5,
            "minimum_speed_for_heading": 0.2,
            "minimum_cos_pitch_squared": 1.0e-4,
            "uses_simulator_world_pose": False,
            "uses_terrain_truth": False,
        }
        self.reset()

    def reset(self):
        """Reset estimates to the known episode start; retain no old samples."""
        self.estimated_xy = np.zeros(2, dtype=np.float64)
        self.estimated_yaw = 0.0
        self.target_yaw = 0.0
        self.yaw_rate = 0.0
        self._previous_yaw_rate = None
        self._previous_velocity_xy = None
        self._elapsed_seconds = 0.0
        self._samples = 0

    def state_dict(self):
        """Return a JSON-compatible diagnostic snapshot with independent values."""
        return {
            "estimated_xy": self.estimated_xy.tolist(),
            "estimated_yaw": float(self.estimated_yaw),
            "target_yaw": float(self.target_yaw),
            "yaw_rate": float(self.yaw_rate),
            "elapsed_seconds": float(self._elapsed_seconds),
            "samples": self._samples,
        }

    def command_from_estimate(self, speed):
        """Compute steering without advancing the inertial samples."""
        speed = float(speed)
        if not np.isfinite(speed):
            raise ValueError('speed must be finite')
        self.target_yaw = float(np.clip(np.arctan2(-0.15 * self.estimated_xy[1], max(speed, 0.2)), -0.25, 0.25))
        yaw_command = float(np.clip(_wrap_angle(self.target_yaw - self.estimated_yaw), -0.5, 0.5))
        return np.asarray([speed, 0.0, yaw_command], dtype=np.float32)

    def command(self, proprio, speed, dt):
        """Observe one state and return float32 [speed, 0, yaw_command].

        Gravity is normalized before extracting roll/pitch. Near vertical pitch
        the ZYX yaw coordinate becomes singular; reject that sample instead of
        silently integrating world-z angular velocity as though it were yaw.
        Invalid samples leave the previous navigator state intact.
        """
        observation = np.asarray(proprio, dtype=np.float64)
        if observation.shape != (81,):
            raise ValueError(f"Expected proprio shape (81,), got {observation.shape}")
        speed, dt = float(speed), float(dt)
        if not np.isfinite(speed) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("speed must be finite and dt must be positive and finite")
        velocity_body = observation[:3]
        angular_body = observation[3:6]
        gravity = observation[9:12]
        if not np.isfinite(np.concatenate((velocity_body, angular_body, gravity))).all():
            raise ValueError("Navigator velocity, angular velocity and gravity must be finite")
        gravity_norm = float(np.linalg.norm(gravity))
        if not np.isfinite(gravity_norm) or gravity_norm < 1.0e-9:
            raise ValueError("Projected gravity must have a finite nonzero norm")
        gx, gy, gz = gravity / gravity_norm
        denominator = float(gy * gy + gz * gz)
        if denominator < 1.0e-4:
            raise ValueError("Proprioceptive yaw is singular near vertical pitch")
        yaw_rate = float((-gy * angular_body[1] - gz * angular_body[2]) / denominator)
        yaw = self.estimated_yaw
        if self._previous_yaw_rate is not None:
            yaw += 0.5 * (self._previous_yaw_rate + yaw_rate) * dt

        roll = np.arctan2(-gy, -gz)
        pitch = np.arcsin(np.clip(gx, -1.0, 1.0))
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        # First two rows of Rz(yaw) @ Ry(pitch) @ Rx(roll).
        rotation_xy = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        ])
        velocity_xy = rotation_xy @ velocity_body
        xy = self.estimated_xy.copy()
        if self._previous_velocity_xy is not None:
            xy += 0.5 * (self._previous_velocity_xy + velocity_xy) * dt

        target_yaw = float(np.clip(np.arctan2(-0.15 * xy[1], max(speed, 0.2)), -0.25, 0.25))
        yaw_command = float(np.clip(_wrap_angle(target_yaw - yaw), -0.5, 0.5))
        action = np.asarray([speed, 0.0, yaw_command], dtype=np.float32)
        if not np.isfinite(np.r_[xy, yaw, yaw_rate, velocity_xy, action]).all():
            raise ValueError("Navigator integration produced a nonfinite result")

        self.estimated_xy = xy
        self.estimated_yaw = float(yaw)
        self.target_yaw = target_yaw
        self.yaw_rate = yaw_rate
        self._previous_yaw_rate = yaw_rate
        self._previous_velocity_xy = velocity_xy.copy()
        if self._samples:
            self._elapsed_seconds += dt
        self._samples += 1
        return action
