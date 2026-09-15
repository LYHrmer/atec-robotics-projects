"""Fixed camera mounts, recovered from the simulator instead of assumed.

The visual policies back-project a pixel and a depth into the body frame with a
camera pose built from joint FK plus a hardcoded mount. Whatever that mount is
wrong by lands in the object estimate with no other symptom, and the grip needs
the estimate to about a centimetre. This module supplies

* the arm sweep that moves the gripper camera through a spread of held joint
  poses while the base stands still, and
* the closed-form fit that recovers both mounts from the camera poses the
  simulator reports for those same steps.

Everything here is numpy/scipy plus the existing FK; no simulator imports, so
the whole fit can be audited on the CPU against synthetic ground truth before a
GPU run is spent on it.

Frames
------
``body`` is the frame the arm FK uses, i.e. the articulation root frame that
``robot.data.root_pos_w`` reports -- the same frame ``solve_ik`` works in.

``optical`` is the ROS optical camera frame the detectors assume: +Z forward
along the view axis, +X right in the image, +Y down in the image. It is the
frame in which ``visual_approach.detect_yellow_candidates`` builds
``camera_point``, so it is the frame the mounts have to be expressed in.

``world camera`` is Isaac's other camera convention: +X forward, +Z up, right =
-Y. ``isaaclab.sensors.CameraData.quat_w_world`` is reported in this convention
(``camera.py::_update_poses`` converts the OpenGL prim pose with
``target="world"``), which is why it is converted here rather than used directly.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from task_b.arm_kinematics import ARM_JOINT_NAMES, arm_targets_to_action, fk
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

#: World-camera axes expressed in optical axes: optical +Z -> world-camera +X,
#: optical +X -> world-camera -Y, optical +Y -> world-camera -Z.
WORLD_CAMERA_FROM_OPTICAL = np.array([[0., 0., 1.],
                                      [-1., 0., 0.],
                                      [0., -1., 0.]])
#: Inverse of the above, used to read a world-camera pose as an optical one.
OPTICAL_FROM_WORLD_CAMERA = WORLD_CAMERA_FROM_OPTICAL.T

#: Arm joint targets the sweep holds. The first is the robot's own default arm
#: pose, so the sweep begins with no transient at all. The rest stay inside the
#: Piper limits and in the upper half of the shoulder range, so the gripper and
#: its camera stay clear of both the ground (body z 0 is 0.474 m above the
#: terrain) and the chassis, while spreading across all six joints so a mount
#: fitted at one pose has to survive the others. The jaw stays open at its
#: default, which leaves the fingers out of the way of everything the cameras
#: look at.
SWEEP_POSES = np.array([
    (0.00, 1.20, -1.50, 0.00, 1.20, 0.00),
    (0.45, 1.20, -1.50, 0.00, 1.00, 0.00),
    (-0.45, 1.20, -1.50, 0.00, 1.00, 0.00),
    (0.00, 0.85, -1.10, 0.00, 1.10, 0.00),
    (0.00, 1.55, -1.85, 0.00, 1.20, 0.00),
    (0.00, 1.20, -1.50, 0.50, 0.60, 0.00),
    (0.00, 1.20, -1.50, -0.50, 0.60, 0.00),
    (0.30, 1.35, -1.65, 0.00, 0.30, -0.45),
    (-0.30, 1.35, -1.65, 0.00, 0.30, 0.45),
])
#: Jaw open targets in metres of joint travel, the same absolute pair
#: grasp_probe closed from in the validated M2/M3 runs. They are targets, not
#: offsets: the Task B environment's own arm_joint7/8 defaults are (0, 0), which
#: differs from Task E's, so the action is derived from the live defaults.
JAW_OPEN = np.array([.035, -.035])


def _as_transform(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-9)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError(f"{name} must be a proper rigid transform")
    return matrix


def world_camera_from_optical_rotation(quat_w_world):
    """``world_from_optical`` rotation from a ``CameraData.quat_w_world`` (wxyz).

    The sensor reports the prim in the world-camera convention; the detectors
    need optical. This is the single place that convention is resolved.
    """
    quaternion = np.asarray(quat_w_world, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.isfinite(quaternion).all():
        raise ValueError("quat_w_world must be four finite values, scalar first")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return rotation @ WORLD_CAMERA_FROM_OPTICAL


def world_from_optical(pos_w, quat_w_world):
    """Simulator camera pose as ``world_from_optical``."""
    position = np.asarray(pos_w, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.isfinite(position).all():
        raise ValueError("pos_w must be three finite metres")
    transform = np.eye(4)
    transform[:3, :3] = world_camera_from_optical_rotation(quat_w_world)
    transform[:3, 3] = position
    return transform


def inverse(transform):
    """Inverse of a rigid 4x4, without a general matrix solve."""
    matrix = _as_transform(transform, "transform")
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def body_from_world(base_xyz, base_quat):
    """``body_from_world`` for the articulation root pose the FK frame uses."""
    position = np.asarray(base_xyz, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(base_quat, dtype=np.float64).reshape(-1)
    if position.size != 3 or quaternion.size != 4:
        raise ValueError("base pose must be three metres and a scalar-first quaternion")
    if not (np.isfinite(position).all() and np.isfinite(quaternion).all()):
        raise ValueError("base pose must be finite")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix().T
    transform[:3, 3] = -transform[:3, :3] @ position
    return transform


def body_from_optical(base_xyz, base_quat, camera_pos_w, camera_quat_w):
    """The mount a camera really has, in the FK body frame, at one step."""
    return body_from_world(base_xyz, base_quat) @ world_from_optical(camera_pos_w, camera_quat_w)


def mean_rotation(matrices):
    """Quaternion eigen-average of a stack of rotation matrices (Markley)."""
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[1:] != (3, 3) or len(stack) == 0:
        raise ValueError("mean_rotation needs a non-empty stack of 3x3 rotations")
    quaternions = Rotation.from_matrix(stack).as_quat()
    reference = quaternions[0]
    quaternions = np.where((quaternions @ reference)[:, None] < 0., -quaternions, quaternions)
    _, vectors = np.linalg.eigh(quaternions.T @ quaternions)
    return Rotation.from_quat(vectors[:, -1]).as_matrix()


def mean_transform(transforms):
    """Mean of rigid transforms: arithmetic mean of position, eigen-mean rotation."""
    stack = np.asarray([_as_transform(value, "transform") for value in transforms])
    result = np.eye(4)
    result[:3, :3] = mean_rotation(stack[:, :3, :3])
    result[:3, 3] = stack[:, :3, 3].mean(axis=0)
    return result


def transform_spread(transforms, reference):
    """Largest position and rotation departure of a stack from one transform."""
    stack = np.asarray([_as_transform(value, "transform") for value in transforms])
    centre = _as_transform(reference, "reference")
    positions = np.linalg.norm(stack[:, :3, 3] - centre[:3, 3], axis=1)
    angles = Rotation.from_matrix(np.einsum("nij,kj->nik", stack[:, :3, :3], centre[:3, :3])
                                  ).magnitude()
    return float(positions.max()), float(angles.max())


def stable_steps(arm_q, base_xyz, base_quat, *, window=25, arm_tolerance=2e-4,
                 position_tolerance=1e-4, angle_tolerance=1.5e-4):
    """Indices whose last ``window`` steps were still, so poses cannot be stale.

    The cameras refresh at 10 Hz (``update_period = 0.1``) while the policy runs
    at 50 Hz, so a recorded camera pose can be up to five steps old. A step is
    kept only if its own step-to-step change AND every change in the preceding
    ``window`` were below tolerance, which makes that staleness irrelevant and
    also rejects the spawn bounce (terrain restitution 1.0) and any slow drift.

    The tolerances are per-step changes, not window ranges: the arm servo settles
    asymptotically, so a range test would reject a pose that is already still to
    well under a millimetre of camera motion. The defaults come from a measured
    sweep -- the last 20 steps of each held pose move the arm by at most 2.3e-4
    rad, the base by 8.5e-5 m and 1.3e-4 rad -- so five steps of lag bias the
    recovered mount by at most about half a millimetre.
    """
    arm = np.asarray(arm_q, dtype=np.float64)
    positions = np.asarray(base_xyz, dtype=np.float64)
    quaternions = np.asarray(base_quat, dtype=np.float64)
    if arm.ndim != 2 or positions.shape != (len(arm), 3) or quaternions.shape != (len(arm), 4):
        raise ValueError("arm_q, base_xyz and base_quat must share a first dimension")
    if window < 1:
        raise ValueError("window must be at least one step")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    still = np.zeros(len(arm), dtype=bool)
    if len(arm) > 1:
        still[1:] = (
            np.abs(np.diff(arm, axis=0)).max(axis=1) < arm_tolerance
        ) & (
            np.abs(np.diff(positions, axis=0)).max(axis=1) < position_tolerance
        ) & (
            Rotation.from_matrix(np.einsum("nij,njk->nik", rotations[1:],
                                           np.transpose(rotations[:-1], (0, 2, 1)))
                                 ).magnitude() < angle_tolerance
        )
    keep = [index for index in range(window, len(arm))
            if bool(still[index - window + 1:index + 1].all())]
    return np.asarray(keep, dtype=int)


def fit_head(samples):
    """Head mount: one constant ``body_from_optical``, because base_link is rigid."""
    return mean_transform([sample["body_from_optical"] for sample in samples])


def fit_ee(samples):
    """Wrist mount: the constant ``gripper_base_from_optical`` behind FK @ offset."""
    offsets = [inverse(fk(sample["arm_q"])) @ sample["body_from_optical"] for sample in samples]
    return mean_transform(offsets), offsets


def induced_error(model, simulator, points_body):
    """Body-frame points as the model places them, minus where they really are.

    ``model`` and ``simulator`` are the same camera expressed twice. Writing the
    model's estimate of a true point ``p`` as ``model @ simulator^-1 @ p`` makes
    the whole mount error a single rigid transform, so the resulting error is
    exactly the localisation error a perfect detector would still inherit.
    """
    error = _as_transform(model, "model") @ inverse(simulator)
    points = np.asarray(points_body, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_body must be a finite (n, 3) stack of body-frame metres")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (error @ homogeneous.T).T[:, :3] - points


class CameraCalibrationPolicy:
    """Hold the arm at a spread of joint targets so the mounts can be fitted.

    Open loop and proprio-only: it commands the arm position term straight to
    each target and leaves the legs and wheels at their defaults, which keeps the
    chassis standing still under the same servo the official environment applies.
    Every target is interpolated from the previous one, never stepped, so the
    gripper camera never sees a transient it did not settle out of.
    """

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=200, ramp_calls=25, hold_calls=140, poses=SWEEP_POSES):
        schema.validate()
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        for name, value in (("settle_calls", settle_calls), ("ramp_calls", ramp_calls),
                            ("hold_calls", hold_calls)):
            if int(value) < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be a non-negative integer")
        if int(ramp_calls) < 1 or int(hold_calls) < 1:
            raise ValueError("ramp_calls and hold_calls must be at least one")
        self.schema = schema
        self.names = tuple(observation_joint_names)
        self.defaults = dict(defaults)
        self.dt = float(dt)
        self.settle_calls, self.ramp_calls, self.hold_calls = (
            int(settle_calls), int(ramp_calls), int(hold_calls))
        self.calls, self.alpha = 0, 0.0
        self.done_reason = None

        self.arm = schema.term(ARM_TERM)
        self.leg = schema.term(LEG_TERM)
        self.wheel = schema.term(WHEEL_TERM)
        self.arm_obs_ids = np.array([self.names.index(name) for name in self.arm.joint_names])
        self.arm_defaults = np.array([self.defaults[name] for name in ARM_JOINT_NAMES])

        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) == 0 or not np.isfinite(poses).all():
            raise ValueError("poses must be a non-empty (n, 6) stack of arm joint radians")
        # The PHYSICAL limit is the hard one. The soft limits are 0.9x these, and
        # the robot's own default arm pose already sits outside them (arm_joint5
        # defaults to 1.20 rad against a 1.098 rad soft bound), so checking soft
        # would reject the very pose the environment resets to. The official arm
        # action term clips to nothing, so a hard-limit pose is commanded as given.
        too_low = poses < JOINT_LOWER - 1e-9
        too_high = poses > JOINT_UPPER + 1e-9
        if too_low.any() or too_high.any():
            index, joint = np.argwhere(too_low | too_high)[0]
            raise ValueError(
                f"sweep pose {index} puts {ARM_JOINT_NAMES[joint]} at {poses[index, joint]:.4f} rad, "
                f"outside the Piper hard limit [{JOINT_LOWER[joint]:.4f}, {JOINT_UPPER[joint]:.4f}]")
        jaw = np.array([self.defaults[name] for name in ("arm_joint7", "arm_joint8")])
        self.jaw_default = jaw
        #: Normalized action targets, one per pose, in the arm term's own order.
        self.pose_actions = np.array([
            arm_targets_to_action(np.r_[joints, JAW_OPEN], self.arm.joint_names,
                                  self.defaults, scale=self.arm.scale) for joints in poses])
        self.poses = poses
        self.open_loop = np.zeros(len(self.arm.joint_names))

    def _target(self):
        """Normalized arm action for the current call, interpolated between poses."""
        index, offset = divmod(max(self.calls - self.settle_calls, 0),
                               self.ramp_calls + self.hold_calls)
        if index >= len(self.poses):
            return self.pose_actions[-1], len(self.poses) - 1, 1.0
        # Interpolate from the previous TARGET, not the previous output, so each
        # pose is a clean ramp out of the last one and the hold is fully settled.
        previous = self.pose_actions[index - 1] if index else self.open_loop
        blend = min(offset / self.ramp_calls, 1.)
        return previous + blend * (self.pose_actions[index] - previous), index, blend

    def act(self, proprio) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio.size == 0 or not np.isfinite(proprio).all():
            raise ValueError(f"Non-finite or empty proprio observation of size {proprio.size}")
        self.calls += 1
        action = np.zeros(self.schema.total_dim, dtype=np.float64)
        target, index, blend = self._target()
        self.alpha = float(blend)
        action[self.arm.start:self.arm.stop] = target
        if not np.isfinite(action).all():
            raise ValueError("Non-finite calibration action")
        if self.calls >= self.settle_calls + len(self.poses) * (self.ramp_calls + self.hold_calls):
            self.done_reason = "camera_calibration_complete"
        return action.astype(np.float32)

    def describe(self) -> dict:
        return {
            "settle_calls": self.settle_calls, "ramp_calls": self.ramp_calls,
            "hold_calls": self.hold_calls,
            "poses_arm_joint_rad": self.poses.tolist(),
            "poses_arm_joint_names": list(ARM_JOINT_NAMES[:6]),
            "jaw_target_m": JAW_OPEN.tolist(),
            "jaw_default_m": self.jaw_default.tolist(),
            "arm_term": {"start": self.arm.start, "dim": self.arm.dim, "scale": self.arm.scale},
            "leg_wheel_action": "held at the schema defaults (zero normalized action)",
            "inputs": "public proprio observation and the static action schema only; no scene state",
            "purpose": "not a policy: it only parks the gripper camera at known joint poses so the "
                       "simulator's own camera poses can be read off and the mounts fitted from them",
        }
