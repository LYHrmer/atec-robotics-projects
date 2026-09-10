"""Cartesian-goal residual servo: deployable features, analytic DLS and guards.

The existing IK path still produces a joint-space waypoint, but the student
never sees it. The waypoint is converted through forward kinematics into a
Cartesian gripper_base goal, an analytic damped-least-squares step solves the
bulk of the servo problem, and the network only supplies a bounded joint-delta
residual on top of it. No teacher action and no simulator truth is an input.

Scaling is fixed physical (metres, radians, the commanded joint limit); nothing
is normalized per episode, so training and deployment features are identical
functions of the same pre-action observation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from task_e_geometry import JOINT_LOWER, JOINT_UPPER, differential_ik, fk
from tools.task_e.il.common import PHASES

CARTESIAN_INTERFACE_VERSION = 1
POSITION_SCALE = .10
ROTATION_SCALE = .5
VELOCITY_SCALE = 4.
LIMIT_SCALE = .085
DLS_DAMPING = .025
DLS_ORIENTATION_WEIGHT = .20
# Teacher delta and DLS delta are each bounded by the commanded limit (<=.085),
# so a residual can never honestly need more than twice it.
RESIDUAL_SCALE_FLOOR = .002
RESIDUAL_SCALE_CAP = .17
HIDDEN = 128

CARTESIAN_FEATURE_NAMES = ([f'joint_{i}_over_pi' for i in range(6)]
                           + [f'joint_velocity_{i}_over_4' for i in range(6)]
                           + [f'position_error_{axis}_over_0.10' for axis in 'xyz']
                           + ['position_error_norm_over_0.10']
                           + [f'rotation_error_rotvec_{axis}_over_0.5' for axis in 'xyz']
                           + ['rotation_error_norm_over_0.5']
                           + [f'analytic_dls_delta_{i}_over_limit' for i in range(6)]
                           + ['analytic_dls_delta_max_abs_over_limit', 'limit_over_0.085']
                           + [f'joint_{i}_range_position' for i in range(6)]
                           + [f'phase_{phase}' for phase in PHASES]
                           + [f'object_{i}' for i in range(4)])


def goal_from_joint_target(target):
    """Cartesian gripper_base goal implied by an IK path waypoint."""
    pose = fk(np.asarray(target, dtype=float).reshape(-1)[:6])
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def pose_error(q, goal_position, goal_rotation):
    """Position error in metres and orientation error as a rotation vector."""
    pose = fk(np.asarray(q, dtype=float).reshape(-1)[:6])
    angular = Rotation.from_matrix(np.asarray(goal_rotation, dtype=float) @ pose[:3, :3].T).as_rotvec()
    return np.asarray(goal_position, dtype=float).reshape(-1)[:3] - pose[:3, 3], angular


def analytic_step(q, goal_position, goal_rotation, limit):
    """One DLS update, rate-limited like the teacher and inside joint limits."""
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    updated = differential_ik(q, goal_position, goal_rotation, damping=DLS_DAMPING,
                              max_joint_delta=float(limit),
                              orientation_weight=DLS_ORIENTATION_WEIGHT)
    return updated - q


def cartesian_features(q, qdot, goal_position, goal_rotation, analytic, limit, phase, object_id):
    """One deployable feature vector, all available before commanding the robot."""
    q, qdot, analytic = (np.asarray(value, dtype=float).reshape(-1)
                         for value in (q, qdot, analytic))
    limit = float(limit)
    object_id = int(object_id or 0)
    if (min(q.size, qdot.size) < 6 or analytic.size != 6 or phase not in PHASES
            or not limit > 0. or not 0 <= object_id < 4):
        raise ValueError('Invalid Cartesian servo feature shape, phase, object or limit')
    position_error, rotation_error = pose_error(q, goal_position, goal_rotation)
    result = np.zeros(len(CARTESIAN_FEATURE_NAMES), dtype=np.float32)
    result[0:6] = q[:6] / np.pi
    result[6:12] = np.clip(qdot[:6] / VELOCITY_SCALE, -4., 4.)
    result[12:15] = np.clip(position_error / POSITION_SCALE, -8., 8.)
    result[15] = min(float(np.linalg.norm(position_error)) / POSITION_SCALE, 8.)
    result[16:19] = np.clip(rotation_error / ROTATION_SCALE, -8., 8.)
    result[19] = min(float(np.linalg.norm(rotation_error)) / ROTATION_SCALE, 8.)
    result[20:26] = np.clip(analytic / limit, -1., 1.)
    result[26] = min(float(np.max(np.abs(analytic))) / limit, 1.)
    result[27] = limit / LIMIT_SCALE
    # Signed position inside the physical joint range; -1 and +1 are the stops.
    result[28:34] = np.clip(2. * (q[:6] - JOINT_LOWER) / (JOINT_UPPER - JOINT_LOWER) - 1., -2., 2.)
    result[34 + PHASES.index(phase)] = 1.
    result[41 + object_id] = 1.
    if not np.isfinite(result).all():
        raise ValueError('Non-finite Cartesian servo features')
    return result


def command_from_residual(q, analytic, residual, limit):
    """Apply the residual, then the physical guards: rate limit, joint limits.

    Returns the six-joint command and whether each guard actually clamped it.
    There is no teacher fallback: a clamp is a physical bound, not a correction
    toward a demonstrated action.
    """
    q = np.asarray(q, dtype=float).reshape(-1)[:6]
    raw = np.asarray(analytic, dtype=float).reshape(-1) + np.asarray(residual, dtype=float).reshape(-1)
    if raw.size != 6 or not np.isfinite(raw).all():
        raise ValueError('Invalid analytic/residual servo step')
    step = np.clip(raw, -float(limit), float(limit))
    command = np.clip(q + step, JOINT_LOWER, JOINT_UPPER)
    return (command, bool(np.any(np.abs(raw - step) > 1e-12)),
            bool(np.any(np.abs(command - q - step) > 1e-12)))


def residual_scale_from_labels(labels, quantile=.995):
    """Bound the learned residual by observed label coverage, not by guesswork."""
    magnitude = np.abs(np.asarray(labels, dtype=float).reshape(-1, 6))
    scale = np.quantile(magnitude, float(quantile), axis=0)
    return np.clip(scale, RESIDUAL_SCALE_FLOOR, RESIDUAL_SCALE_CAP).astype(np.float32)


def derive_sample(q, qdot, target, delta, limit, phase, object_id):
    """Offline features, residual label and DLS step from one stored frame.

    Deployment calls the same three functions in the same order, so a training
    row and a runtime row cannot drift apart.
    """
    goal_position, goal_rotation = goal_from_joint_target(target)
    analytic = analytic_step(q, goal_position, goal_rotation, limit)
    features = cartesian_features(q, qdot, goal_position, goal_rotation, analytic,
                                  limit, phase, object_id)
    label = np.asarray(delta, dtype=float).reshape(-1)[:6] - analytic
    return features, label, analytic


def make_cartesian_model():
    """Zero output layer: an untrained student commands exactly the DLS step."""
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(len(CARTESIAN_FEATURE_NAMES), HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, 6), torch.nn.Tanh())
    torch.nn.init.zeros_(model[4].weight)
    torch.nn.init.zeros_(model[4].bias)
    return model
