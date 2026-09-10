"""Static B2w-mounted Piper geometry; all Cartesian values are body-frame.

No simulator imports or root/object state. The common Piper chain is shared
with Task E, but every FK call explicitly supplies the Task B arm mount. Task
E's table position, default joint angles and world-space IK are not used.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from task_e_geometry import IKResult, JOINT_LOWER, JOINT_UPPER, fk as _chain_fk

ARM_JOINT_NAMES = tuple(f"arm_joint{i}" for i in range(1, 9))
BODY_FROM_ARM_POSITION = np.array([.2, 0., .1])
BODY_FROM_ARM_QUATERNION = np.array([1., 0., 0., 0.])
GRASP_DEPTH = .115


def _vector(value, size, name):
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def _rotation(value):
    value = np.asarray(value, dtype=float)
    if value.shape == (4,):
        value = Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value), 1., atol=1e-6)):
        raise ValueError("rotation must be a proper 3x3 rotation or wxyz quaternion")
    return value


def fk(joints, *, return_jacobian=False):
    """Return body_from_gripper_base, optionally its body-frame 6x6 Jacobian."""
    joints = np.asarray(joints, dtype=float).reshape(-1)
    if joints.size not in (6, 8) or not np.isfinite(joints).all():
        raise ValueError("fk requires six arm joints or all eight Piper joints")
    return _chain_fk(joints[:6], base_position=BODY_FROM_ARM_POSITION,
                     base_quaternion=BODY_FROM_ARM_QUATERNION,
                     return_jacobian=return_jacobian)


def pinch_position(joints, *, grasp_depth=GRASP_DEPTH):
    """Nominal point on the finger contact slab, along gripper local +Z."""
    if not np.isfinite(grasp_depth) or not 0. <= grasp_depth <= .1358:
        raise ValueError("grasp_depth must lie in [0, .1358] m")
    pose = fk(joints)
    return pose[:3, 3] + float(grasp_depth) * pose[:3, 2]


def solve_ik(position, rotation=None, seed=None, *, orientation_weight=.20,
             multi_start=True, max_nfev=90, position_tolerance=.005,
             orientation_tolerance=.08):
    """Bounded body-frame IK; inspect success/errors before commanding it.

    This solver only enforces joint limits and a requested pose. It does not
    certify collision freedom or balance of the mobile robot.
    """
    desired = _vector(position, 3, 'position')
    target_rotation = None if rotation is None else _rotation(rotation)
    lo, hi = JOINT_LOWER + 1e-6, JOINT_UPPER - 1e-6
    initial = np.zeros(6) if seed is None else _vector(np.asarray(seed).reshape(-1)[:6], 6, 'seed')
    initial = np.clip(initial, lo, hi)
    if not np.isfinite(orientation_weight) or orientation_weight <= 0. or max_nfev < 1:
        raise ValueError('Invalid IK orientation weight or evaluation budget')

    def residual(q):
        pose = fk(q)
        delta = pose[:3, 3] - desired
        if target_rotation is None:
            return delta
        angular = Rotation.from_matrix(pose[:3, :3] @ target_rotation.T).as_rotvec()
        return np.r_[delta, orientation_weight * angular]

    guesses = [initial]
    if multi_start:
        bearing = np.arctan2(desired[1], desired[0] - BODY_FROM_ARM_POSITION[0])
        guesses += [np.array([bearing, j2, j3, j4, j5, 0.])
                    for j2, j3, j4, j5 in [(1.5, -1.8, 0., 1.), (.5, -1., 0., 1.),
                                          (2.5, -1.5, 0., -.8), (1.5, -2., 1.4, -.8),
                                          (1.5, -2., -1.4, -.8)]]
    best, evaluations = None, 0
    for guess in guesses:
        fit = least_squares(residual, np.clip(guess, lo, hi), bounds=(lo, hi),
                            max_nfev=int(max_nfev), ftol=1e-8, xtol=1e-8, gtol=1e-8)
        evaluations += fit.nfev
        pose = fk(fit.x)
        pe = float(np.linalg.norm(pose[:3, 3] - desired))
        oe = 0. if target_rotation is None else float(np.linalg.norm(
            Rotation.from_matrix(pose[:3, :3] @ target_rotation.T).as_rotvec()))
        cost = pe * pe + (orientation_weight * oe) ** 2
        if best is None or cost < best[0]:
            best = (cost, fit.x.copy(), pe, oe)
        if pe < position_tolerance and oe < orientation_tolerance:
            break
    _, joints, pe, oe = best
    return IKResult(joints, pe, oe, pe < position_tolerance and oe < orientation_tolerance, evaluations)


def differential_ik(joints, position, rotation=None, *, damping=.025,
                    max_joint_delta=.025, orientation_weight=.20):
    """One bounded DLS step from measured arm joints, in current body frame."""
    q = _vector(np.asarray(joints).reshape(-1)[:6], 6, 'joints')
    position = _vector(position, 3, 'position')
    if not all(np.isfinite(v) and v > 0. for v in (damping, max_joint_delta, orientation_weight)):
        raise ValueError('DLS damping, delta bound and orientation weight must be positive')
    pose, jacobian = fk(q, return_jacobian=True)
    error = position - pose[:3, 3]
    if rotation is None:
        jacobian = jacobian[:3]
    else:
        angular = Rotation.from_matrix(_rotation(rotation) @ pose[:3, :3].T).as_rotvec()
        error = np.r_[error, orientation_weight * angular]
        jacobian[3:] *= orientation_weight
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping ** 2 * np.eye(len(error)), error)
    return np.clip(q + np.clip(delta, -max_joint_delta, max_joint_delta), JOINT_LOWER, JOINT_UPPER)


def arm_joints_from_proprio(proprio, observation_joint_names, default_joint_positions):
    """Read absolute Piper q from official Task B proprio[12:36].

    Supply observation-manager joint order, not articulation or action order.
    Defaults must be a static name->position mapping, not Task E defaults.
    """
    obs = _vector(proprio, 84, 'Task B proprio')
    names = tuple(observation_joint_names)
    if len(names) != 24 or len(set(names)) != 24 or not set(ARM_JOINT_NAMES).issubset(names):
        raise ValueError('Expected 24 distinct observed joints including the eight Piper joints')
    q = np.array([obs[12 + names.index(name)] + default_joint_positions[name] for name in ARM_JOINT_NAMES])
    return _vector(q, 8, 'absolute arm joints')


def gripper_targets(opening_m=.07, *, preload_m=0.):
    """Return physical [arm_joint7, arm_joint8] target positions in metres.

    opening=.07 is fully open; opening=0 closes. Optional preload up to .025 m
    requests pressure beyond the physical closed stop, as Task E does, without
    modifying joint limits. Actual collision/contact behavior needs simulation.
    """
    if not np.isfinite(opening_m) or not 0. <= opening_m <= .07:
        raise ValueError('opening must lie in [0, .07] m')
    if not np.isfinite(preload_m) or not 0. <= preload_m <= .025:
        raise ValueError('preload must lie in [0, .025] m')
    half = .5 * float(opening_m) - float(preload_m)
    return np.array([half, -half])


def arm_targets_to_action(targets, action_joint_names, default_joint_positions, *, scale):
    """Eight normalized arm-term values in its actual action-manager order.

    Return only the arm slice; the caller preserves the leg/wheel action. The
    current official arm term is position/default-offset/scale=.5/clip=None.
    Verify that schema before using this affine mapping.
    """
    targets = _vector(targets, 8, 'arm targets')
    names = tuple(action_joint_names)
    if len(names) != 8 or set(names) != set(ARM_JOINT_NAMES):
        raise ValueError('Expected each Piper arm joint exactly once')
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError('scale must be positive and finite')
    by_name = dict(zip(ARM_JOINT_NAMES, targets))
    action = np.array([(by_name[name] - default_joint_positions[name]) / scale for name in names])
    return _vector(action, 8, 'normalized arm action')


def ee_camera_transform(joints):
    """body_from_ROS_ee_camera from measured q and the official fixed mount."""
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_euler('z', -np.pi/2).as_matrix()
    offset[:3, 3] = [-.05, 0., .06]
    return fk(joints) @ offset


def head_camera_transform():
    """body_from_ROS_head_camera for the original B2w camera, without state.

    B2w inherits B2's camera on base_link: offset uses the Isaac ``world``
    camera convention (+X forward,+Y left,+Z up), pitched down by +30 deg.
    The final axis conversion maps ROS optical +Z to that forward direction.
    """
    ros_to_world_camera = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', np.pi/6).as_matrix() @ ros_to_world_camera
    transform[:3, 3] = [.4216099977493286, .02500000037252903, .06185099855065346]
    return transform
