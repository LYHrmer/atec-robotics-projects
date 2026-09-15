"""Public-twist planar dead reckoning for the Task B base pose.

Independent helper. It owns no action, no gate and no environment state: it reads
one original 84-element public proprio observation per control tick and returns
its own integrated planar pose estimate. It reads **only** base linear velocity
``[0:3]``, base angular velocity ``[3:6]`` and projected gravity ``[9:12]`` -
never simulator or root state, pose snapshots, images, reward, score, object
identifiers or seeded object coordinates - and it never resets or recovers
itself. The one non-proprio input is the *official* public static spawn
XY = (-10, -10), yaw = 0, which is task configuration, not an object map.

This is not a drift budget. ``first_reach.py`` / ``multi_reach.py`` carry local
short-horizon drift guards; this is a separate full-episode odometry estimate
and neither modifies nor imports them.

Timing is pre-step telemetry: call 1 returns the fixed spawn estimate at
observation time 0 and stores that observation's twist; call N integrates the
twist stored at call N-1 over exactly ``dt`` to reach the current observation
time. Row N's estimate has consumed rows 1..N-1 only, and no observation is ever
integrated twice.

The approximation is exactly the already-offline-tested formula, with no
filtering, bias correction, clipping, world-pose calibration or learned
constants::

    up       = -gravity / norm(gravity)
    v_t      = linear_velocity - dot(linear_velocity, up) * up
    w        = dot(angular_velocity, up)
    mid_yaw  = yaw_unwrapped + 0.5 * w * dt
    xy      += dt * Rz2(mid_yaw) @ v_t[:2]
    yaw_unwrapped += w * dt

Public base linear velocity is treated as body-frame and rotated into the plane;
vertical drift is not integrated and z is not tracked. Replaying this exact
formula over two completed original runs gave final XY errors of 16.7 mm and
19.8 mm and a yaw error of ~0.008 rad without GT calibration - finite-run
measurements, not promised accuracy and not an uncertainty guarantee, since
there is no world-pose measurement anywhere in here.
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 84                       #: original public proprio length
LIN_VEL_SLICE = slice(0, 3)        #: base linear velocity, body frame
ANG_VEL_SLICE = slice(3, 6)        #: base angular velocity, body frame
GRAVITY_SLICE = slice(9, 12)       #: projected gravity, body frame
GRAVITY_NORM_MIN = 0.5             #: below this the up axis is not usable
SPAWN_XY = (-10.0, -10.0)          #: official public static spawn
SPAWN_YAW_RAD = 0.0


def _rz2(yaw: float) -> np.ndarray:
    """Planar rotation matrix for ``yaw`` in rad."""
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, -sin], [sin, cos]], dtype=np.float64)


def _wrap(yaw: float) -> float:
    """Wrap ``yaw`` to ``[-pi, pi)``."""
    return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)


class PublicPlanarOdometry:
    """Integrate the public base twist into a planar ``(x, y, yaw)`` estimate."""

    def __init__(self, dt: float = 0.02):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")
        self.dt = float(dt)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Return to the fixed public spawn and drop all stored twist/history."""
        self._xy = np.array(SPAWN_XY, dtype=np.float64)
        self._yaw_unwrapped = float(SPAWN_YAW_RAD)
        self._prev_v_t = None
        self._prev_w = None
        self.update_calls = 0
        self.integrated_intervals = 0
        self.path_length_m = 0.0

    # ------------------------------------------------------------- properties
    @property
    def position_xy(self) -> np.ndarray:
        return self._xy.copy()

    @property
    def yaw_rad(self) -> float:
        return _wrap(self._yaw_unwrapped)

    @property
    def yaw_unwrapped_rad(self) -> float:
        return float(self._yaw_unwrapped)

    @property
    def elapsed_s(self) -> float:
        return self.integrated_intervals * self.dt

    # -------------------------------------------------------------------- main
    def update(self, proprio) -> dict:
        """Consume one public observation and return the pose snapshot."""
        obs = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if obs.size != OBS_DIM:
            raise ValueError(f"proprio must have {OBS_DIM} elements, got {obs.size}")
        if not np.isfinite(obs).all():
            raise ValueError("proprio must be finite")
        gravity = obs[GRAVITY_SLICE]
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < GRAVITY_NORM_MIN:
            raise ValueError(f"projected gravity norm {norm} is below {GRAVITY_NORM_MIN}")

        up = -gravity / norm
        lin_vel, ang_vel = obs[LIN_VEL_SLICE], obs[ANG_VEL_SLICE]
        v_t = lin_vel - float(np.dot(lin_vel, up)) * up
        w = float(np.dot(ang_vel, up))

        # Everything is computed into locals first, so a non-finite or
        # overflowing result cannot leave the pose partially advanced.
        xy, yaw_unwrapped = self._xy, self._yaw_unwrapped
        path_length, intervals = self.path_length_m, self.integrated_intervals
        if self._prev_v_t is not None:
            prev_v_t, prev_w = self._prev_v_t, self._prev_w
            mid_yaw = yaw_unwrapped + 0.5 * prev_w * self.dt
            xy = xy + self.dt * (_rz2(mid_yaw) @ prev_v_t[:2])
            yaw_unwrapped = yaw_unwrapped + prev_w * self.dt
            path_length = path_length + float(np.linalg.norm(prev_v_t)) * self.dt
            intervals += 1
        if not (np.isfinite(xy).all() and np.isfinite(yaw_unwrapped)
                and np.isfinite(path_length) and np.isfinite(v_t).all() and np.isfinite(w)
                and np.isfinite(intervals * self.dt)):
            raise ValueError("integration produced a non-finite pose; state left unchanged")

        self._xy, self._yaw_unwrapped = np.asarray(xy, dtype=np.float64), float(yaw_unwrapped)
        self.path_length_m, self.integrated_intervals = float(path_length), intervals
        self._prev_v_t, self._prev_w = v_t.copy(), w
        self.update_calls += 1
        return self.snapshot()

    def snapshot(self) -> dict:
        """Fresh JSON-safe pose record; mutating it cannot touch internal state."""
        return {
            "position_xy": [float(self._xy[0]), float(self._xy[1])],
            "yaw_rad": self.yaw_rad,
            "yaw_unwrapped_rad": self.yaw_unwrapped_rad,
            "elapsed_s": self.elapsed_s,
            "update_calls": int(self.update_calls),
            "integrated_intervals": int(self.integrated_intervals),
            "path_length_m": float(self.path_length_m),
        }

    # ------------------------------------------------------------------ helper
    def target_body_xy(self, target_xy) -> np.ndarray:
        """Body-frame planar offset of a known static waypoint. Reads no map."""
        target = np.asarray(target_xy, dtype=np.float64).reshape(-1)
        if target.size != 2:
            raise ValueError(f"target_xy must have 2 elements, got {target.size}")
        if not np.isfinite(target).all():
            raise ValueError("target_xy must be finite")
        result = _rz2(-self.yaw_rad) @ (target - self._xy)
        if not np.isfinite(result).all():
            raise ValueError("waypoint transform produced a non-finite offset")
        return result

    def describe(self) -> dict:
        """The snapshot plus the fixed sources, timing and approximation claims."""
        record = self.snapshot()
        record.update({
            "helper": "PublicPlanarOdometry",
            "method": "public-twist planar dead reckoning",
            "dt": self.dt,
            "initialization": {"source": "official Task B public static spawn, fixed in code",
                              "spawn_xy": list(SPAWN_XY), "spawn_yaw_rad": SPAWN_YAW_RAD,
                              "reset_takes_pose_arguments": False},
            "inputs": {"proprio_len": OBS_DIM, "base_linear_velocity": "[0:3]",
                       "base_angular_velocity": "[3:6]", "projected_gravity": "[9:12]",
                       "gravity_norm_min": GRAVITY_NORM_MIN,
                       "reads_nothing_else": "no simulator state, pose snapshot, image, reward, "
                                             "score, object identifier or seeded object coordinate"},
            "timing": "call 1 returns the fixed spawn estimate at observation time 0; call N "
                      "integrates the twist stored at call N-1 over exactly dt, so the estimate "
                      "has consumed rows 1..N-1 (pre-step telemetry) and no row is integrated twice",
            "approximation": "up = -g/|g|; v_t = v - (v.up)up; w = a.up; midpoint yaw over dt; "
                             "xy += dt * Rz2(mid_yaw) @ v_t[:2]; yaw += w*dt. Public base linear "
                             "velocity is body-frame, not world-frame; vertical drift is not "
                             "integrated. No filtering, bias correction, clipping, world-pose "
                             "calibration or learned constants.",
            "path_length_note": "sum of |stored v_t| * dt, using the gravity-projected 3-vector",
            "claims": ["no world-pose measurement of any kind",
                       "no uncertainty guarantee; drift is unbounded and unobservable here",
                       "no navigation, waypoint or action logic; target_body_xy is a stateless "
                       "transform of a caller-supplied static waypoint and retains no map",
                       "no automatic reset and no failure recovery",
                       "offline replay of two completed original runs gave 16.7/19.8 mm final XY "
                       "error and ~0.008 rad yaw error without GT calibration - finite-run "
                       "measurements, not promised accuracy"],
        })
        return record
