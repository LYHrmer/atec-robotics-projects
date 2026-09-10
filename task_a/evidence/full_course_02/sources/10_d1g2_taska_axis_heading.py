"""Independent CPU heading candidate from known Task A grid directions.

This module is not connected to a runtime controller. It reads only a current
RGB-D image, OpenCV intrinsics, the original proprioceptive gravity direction,
camera mounting angle, and an optional visual/inertial yaw prior. No simulator
world pose or terrain geometry enters the estimator.

Task A has unrotated cubic-projected brick texture and a mostly axis-aligned
height-field triangulation. Horizontal line directions therefore often share
an eightfold angular mode. The observation has a 45-degree ambiguity; a prior
selects a branch, and cannot repair an already incorrect ambiguity branch.
Confidence measures line agreement, not semantic proof that lines are ground.
Aligned foreground objects can produce a confident false observation.

Algorithm and gates were frozen before the held-out frame evaluation. The
three development frames were 1000/5000/10000 from residual_1999_rgbd_course_01;
their world-pose diagnostics were read only after estimates were frozen.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


ALGORITHM_VERSION = "axis_heading_cpu_candidate_v1"
SOURCE = "known_course_grid+RGBD+gravity"
PARAMETERS = {
    "minimum_line_pixels": 20.0,
    "depth_range_m": (0.15, 8.0),
    "minimum_projected_line_m": 0.015,
    "mode_bandwidth_deg": 2.0,
    "mode_grid_deg": 0.25,
    "mode_refinement_radius_deg": 3.0,
    "separate_peak_distance_deg": 8.0,
    "minimum_valid_lines": 8,
    "minimum_mode_support_fraction": 0.45,
    "minimum_peak_ratio": 3.0,
    "maximum_prior_innovation_deg": 10.0,
}
_PERIOD = math.pi / 4.0


def _fold45(angle):
    return (angle + _PERIOD / 2.0) % _PERIOD - _PERIOD / 2.0


def _gravity_level_rotation(gravity):
    gx, gy, gz = gravity / np.linalg.norm(gravity)
    roll = math.atan2(-gy, -gz)
    pitch = math.atan2(gx, math.hypot(gy, gz))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])


def estimate(rgb, depth, K, projected_gravity, mount_pitch_deg=30.0, predicted_yaw=None):
    """Return a gated yaw candidate; angles are radians unless named ``deg``.

    ``K`` must already use OpenCV pixel centers (the renderer's raw principal
    point requires minus 0.5). ``projected_gravity`` is original proprio[9:12].
    ``predicted_yaw`` must come from legal visual/inertial estimation. Its nearest
    45-degree branch is used; absent/invalid priors never produce acceptance.
    A rejected result may still expose the folded candidate for diagnostics.
    The function neither reads files nor mutates caller inputs or global state.
    """
    result = {
        "source": SOURCE,
        "algorithm_version": ALGORITHM_VERSION,
        "accepted": False,
        "reason": "invalid_input",
        "heading_rad": None,
        "yaw_mod45_radians": None,
        "ambiguity_degrees": 45.0,
    }
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    K = np.asarray(K, dtype=np.float64)
    gravity = np.asarray(projected_gravity, dtype=np.float64)
    if (rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8
            or depth.shape != rgb.shape[:2] or min(rgb.shape[:2]) < 16
            or K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-8
            or gravity.shape != (3,) or not np.isfinite(gravity).all()
            or np.linalg.norm(gravity) < 1e-8 or not math.isfinite(float(mount_pitch_deg))):
        return result
    gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
    raw = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:
        result.update(reason="insufficient_lines", detected_lines=0, valid_lines=0)
        return result
    lines = raw[:, 0, :]
    angle = math.radians(float(mount_pitch_deg))
    c, s = math.cos(angle), math.sin(angle)
    body_camera = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    level_camera = _gravity_level_rotation(gravity) @ body_camera
    inverse_K = np.linalg.inv(K)
    depth32 = depth.astype(np.float32, copy=False)
    directions, weights = [], []
    for line in lines:
        p, q = line.reshape(2, 2).astype(np.float64)
        pixel_length = float(np.linalg.norm(q - p))
        if pixel_length < PARAMETERS["minimum_line_pixels"]:
            continue
        uv = p[None, :] + np.linspace(0.0, 1.0, 9)[:, None] * (q - p)[None, :]
        z = cv2.remap(depth32, uv[:, 0].astype(np.float32)[None, :],
                      uv[:, 1].astype(np.float32)[None, :], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))[0]
        near, far = PARAMETERS["depth_range_m"]
        if not (np.isfinite(z).all() and (z > near).all() and (z < far).all()):
            continue
        camera_points = (np.c_[uv, np.ones(9)] @ inverse_K.T) * z[:, None]
        level_points = camera_points @ level_camera.T
        direction = level_points[-1] - level_points[0]
        if np.linalg.norm(direction[:2]) < PARAMETERS["minimum_projected_line_m"]:
            continue
        directions.append(math.atan2(direction[1], direction[0]))
        weights.append(pixel_length)
    result.update(detected_lines=len(lines), valid_lines=len(directions))
    if len(directions) < PARAMETERS["minimum_valid_lines"]:
        result["reason"] = "insufficient_lines"
        return result
    theta, weights = np.asarray(directions), np.asarray(weights)
    circular_mean = np.sum(weights * np.exp(8j * theta)) / weights.sum()
    grid = np.deg2rad(np.arange(-22.5, 22.5, PARAMETERS["mode_grid_deg"]))
    delta = _fold45(theta[:, None] - grid[None, :])
    scores = (weights[:, None] * np.exp(-0.5 * (delta / math.radians(
        PARAMETERS["mode_bandwidth_deg"])) ** 2)).sum(0) / weights.sum()
    peak = grid[scores.argmax()]
    members = np.abs(_fold45(theta - peak)) < math.radians(PARAMETERS["mode_refinement_radius_deg"])
    refined = np.angle(np.sum(weights[members] * np.exp(8j * theta[members]))) / 8.0
    folded_yaw = float(_fold45(-refined))
    separated = np.abs(_fold45(grid - peak)) > math.radians(PARAMETERS["separate_peak_distance_deg"])
    ratio = float(scores.max() / max(float(scores[separated].max()), 1e-9))
    support = float(weights[members].sum() / weights.sum())
    result.update(yaw_mod45_radians=folded_yaw, mode_support_fraction=support,
                  mode_peak_ratio=ratio, eightfold_concentration=float(abs(circular_mean)))
    if support < PARAMETERS["minimum_mode_support_fraction"]:
        result["reason"] = "weak_direction_consensus"
        return result
    if ratio < PARAMETERS["minimum_peak_ratio"]:
        result["reason"] = "ambiguous_direction_mode"
        return result
    if predicted_yaw is None or not math.isfinite(float(predicted_yaw)):
        result["reason"] = "missing_or_invalid_prior_yaw"
        return result
    prior_yaw = float(predicted_yaw)
    innovation = float(_fold45(folded_yaw - prior_yaw))
    candidate = prior_yaw + innovation
    result.update(candidate_heading_rad=math.atan2(math.sin(candidate), math.cos(candidate)),
                  prior_innovation_radians=innovation)
    if abs(innovation) > math.radians(PARAMETERS["maximum_prior_innovation_deg"]):
        result["reason"] = "prior_disagreement"
        return result
    result.update(accepted=True, reason="known_grid_direction_candidate",
                  heading_rad=result["candidate_heading_rad"])
    return result
