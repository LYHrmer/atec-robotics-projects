"""Shared deployable features and a learned six-joint servo interface."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

PHASES = ('PREGRASP', 'DESCEND', 'LIFT', 'TRANSPORT', 'PLACE', 'RETRACT', 'HOME')
FEATURE_NAMES = (['joint_error_over_limit', 'joint_position_over_pi',
                  'joint_velocity_over_4', 'limit_over_0.085']
                 + [f'joint_{i}' for i in range(6)]
                 + [f'phase_{phase}' for phase in PHASES]
                 + [f'object_{i}' for i in range(4)])
INTERFACE_VERSION = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_settings(controller, slow):
    if controller.current_object == 3 and controller.state in ('LIFT', 'TRANSPORT'):
        slow = True
    limit = .035 if slow else .085
    if controller.current_object == 2 and controller.state in ('LIFT', 'TRANSPORT', 'PLACE'):
        limit = min(limit, .025)
    return bool(slow), limit


def servo_features(q, qdot, target, limit, phase, object_id):
    """Per-joint inputs, all available before commanding the robot.

    target is the existing IK path waypoint, derived from RGB-D contacts and
    the published basket location. No teacher action or simulator truth enters.
    """
    q, qdot, target = (np.asarray(value, dtype=float).reshape(-1)
                       for value in (q, qdot, target))
    if min(q.size, qdot.size, target.size) < 6 or phase not in PHASES:
        raise ValueError('Invalid servo feature shape or phase')
    result = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
    result[:, 0] = np.clip((target[:6] - q[:6]) / limit, -8., 8.)
    result[:, 1] = q[:6] / np.pi
    result[:, 2] = np.clip(qdot[:6] / 4., -4., 4.)
    result[:, 3] = limit / .085
    result[:, 4:10] = np.eye(6)
    result[:, 10 + PHASES.index(phase)] = 1.
    result[:, 17 + int(object_id or 0)] = 1.
    if not np.isfinite(result).all():
        raise ValueError('Non-finite servo features')
    return result


def make_model():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 1), torch.nn.Tanh())
