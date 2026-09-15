"""B2w leg geometry and a wheels-planted vertical-descent solver.

CPU only. Body frame, metres, radians. Nothing here reads simulator state.

Every joint frame below was read from the official ``b2w_piper.usda``
(``physics:localPos0`` and ``physics:axis``), not copied from Task A. The leg
chain is base_link -> hip -> thigh -> calf -> foot, with the wheel carried by
the foot link. ``foot_body_xyz`` returns the wheel *origin* (the axle), so the
wheel radius converts it to a ground clearance.

``descend`` answers one question: which leg angles raise each wheel by a given
amount in the body frame while holding its (x, y)? Holding (x, y) fixes the
wheelbase and track; the body then drops by that amount. This is a static
geometric solution. It says nothing about whether the legs can carry the load,
whether the wheels stay in contact, or whether a link hits the ground -- those
are measured, not solved.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CORNERS = ("FR", "FL", "RR", "RL")
LEG_LINKS = ("hip", "thigh", "calf")

#: Official wheel geometry: ``FR_foot`` is a 0.2258 m cylinder, so r = 0.1129.
WHEEL_RADIUS_M = 0.1129
#: Published static leg-chain IK (task_b/stance_reference.py) for validation.
COMPACT_JOINT_REFERENCE = {
    "FR": (-.3023523168313238, .8130831661377943, -1.8754745945886573),
    "FL": (.30696657668610144, .8173372201608101, -1.8881043202266299),
    "RR": (-.27114297155288936, .9786392186138522, -1.729057945170937),
    "RL": (.2750100973730118, .9858544286612007, -1.7429508632418722),
}
#: Published USD-derived foot positions at the USD default pose
#: (results/task_b_stability_cpu_audit.json:usd_geometry.corners).
DEFAULT_FOOT_BODY_XYZ = {
    "FR": (0.3029015617769021, -0.24310808475268864, -0.496942442391909),
    "FL": (0.3029015617769021, 0.24211308253104266, -0.49704227561354214),
    "RR": (-0.45521590663593814, -0.24158238956323105, -0.48173638094030824),
    "RL": (-0.45521590663593814, 0.24058738535069288, -0.4818362143616969),
}

_HIP = {"FR": (np.array([.3285, -.072, 0.]), 'x'), "FL": (np.array([.3285, .072, 0.]), 'x'),
        "RR": (np.array([-.3285, -.072, 0.]), 'x'), "RL": (np.array([-.3285, .072, 0.]), 'x')}
_THIGH = {"FR": (np.array([0., -.11973, 0.]), 'y'), "FL": (np.array([0., .11973, 0.]), 'y'),
          "RR": (np.array([0., -.11973, 0.]), 'y'), "RL": (np.array([0., .11973, 0.]), 'y')}
_CALF = {c: (np.array([0., 8.821e-05 if c in ("FR", "RR") else -8.821e-05, -.35]), 'y')
         for c in CORNERS}
_FOOT = {c: (np.array([0., -.001 if c in ("FR", "RR") else 0., -.35]), 'y') for c in CORNERS}

#: Hard joint limits in radians, converted from the USD's degree values.
HIP_LIMITS = (-np.deg2rad(49.8473), np.deg2rad(49.8473))
THIGH_LIMITS = (-np.deg2rad(53.8580), np.deg2rad(268.7172))
CALF_LIMITS = (-np.deg2rad(161.5741), -np.deg2rad(24.6372))
JOINT_LIMITS = np.array([HIP_LIMITS, THIGH_LIMITS, CALF_LIMITS])


def leg_joint_names(corner: str) -> tuple[str, str, str]:
    """Action-schema joint names for one leg, in hip/thigh/calf order."""
    if corner not in CORNERS:
        raise ValueError(f"Unknown corner {corner!r}; expected one of {CORNERS}")
    return tuple(f"{corner}_{link}_joint" for link in LEG_LINKS)


def _rot(axis: str, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(np.eye(3)[{"x": 0, "y": 1, "z": 2}[axis]] * angle).as_matrix()


def foot_body_xyz(corner: str, hip: float, thigh: float, calf: float) -> np.ndarray:
    """Wheel (foot link) origin in the base_link frame."""
    if corner not in CORNERS:
        raise ValueError(f"Unknown corner {corner!r}")
    values = np.asarray([hip, thigh, calf], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Leg angles must be finite")
    offset, axis = _HIP[corner]
    return offset + _rot(axis, float(hip)) @ (
        _THIGH[corner][0] + _rot('y', float(thigh)) @ (
            _CALF[corner][0] + _rot('y', float(calf)) @ _FOOT[corner][0]))


def foot_xy(corner: str, q) -> np.ndarray:
    """Wheel origin (x, y) in the base_link frame."""
    return foot_body_xyz(corner, *q)[:2]


def descend(corner: str, q_default, drop_m: float, *, max_nfev: int = 500):
    """Leg angles that raise this wheel by ``drop_m`` while holding its (x, y).

    Positive raises the wheel, which lowers the body; negative lowers the wheel,
    which raises the body, so the same solver returns a stance at a different
    height in either direction.

    Returns ``(q, residual_m)``. The residual is the achieved foot-position error
    against the requested target; callers must inspect it rather than assume the
    solution was reached.
    """
    q0 = np.asarray(q_default, dtype=np.float64).reshape(-1)
    if q0.size != 3 or not np.isfinite(q0).all():
        raise ValueError("q_default must be three finite leg angles")
    if not np.isfinite(drop_m) or not -0.25 <= drop_m <= 0.25:
        raise ValueError("drop_m must be finite and lie in [-0.25, 0.25] m")
    lo, hi = JOINT_LIMITS[:, 0] + 1e-6, JOINT_LIMITS[:, 1] - 1e-6
    if np.any(q0 < lo) or np.any(q0 > hi):
        raise ValueError(f"{corner} default angles {np.round(q0, 4)} already violate the hard limits")
    target = foot_body_xyz(corner, *q0) + np.array([0., 0., float(drop_m)])

    def residual(q):
        return foot_body_xyz(corner, *q) - target

    fit = least_squares(residual, np.clip(q0, lo, hi), bounds=(lo, hi),
                        max_nfev=int(max_nfev), ftol=1e-12, xtol=1e-12)
    return fit.x, float(np.linalg.norm(residual(fit.x)))


def descend_all(q_default_by_corner: dict, drop_m: float) -> tuple[dict, float]:
    """Solve every corner for one drop. Returns ``({corner: q}, worst residual)``."""
    out, worst = {}, 0.0
    for corner in CORNERS:
        q, residual = descend(corner, q_default_by_corner[corner], drop_m)
        out[corner], worst = q, max(worst, residual)
    return out, worst


def validate() -> dict:
    """Self-check the model against the published static leg-chain IK.

    Raises rather than reporting a number when the model does not reproduce the
    geometry that Task B's own stance reference was built from.
    """
    feet = {c: foot_body_xyz(c, *COMPACT_JOINT_REFERENCE[c]) for c in CORNERS}
    wheelbase = abs(float((feet["FR"][0] + feet["FL"][0]) / 2 - (feet["RR"][0] + feet["RL"][0]) / 2))
    track = abs(float((feet["FL"][1] + feet["RL"][1]) / 2 - (feet["FR"][1] + feet["RR"][1]) / 2))
    rear, front = (feet["RR"][2] + feet["RL"][2]) / 2, (feet["FR"][2] + feet["FL"][2]) / 2
    # The published reference quotes only a magnitude ("pitch ~.064 rad"), and the
    # front legs are the shorter pair, so the sign here is a convention: record it
    # rather than assert it.
    pitch = float(np.arctan2(rear - front, wheelbase))
    if not 0.755 < wheelbase < 0.765:
        raise AssertionError(f"wheelbase {wheelbase:.4f} does not match the published 0.76 m")
    if not 0.055 < abs(pitch) < 0.070:
        raise AssertionError(f"|pitch| {abs(pitch):.4f} does not match the published ~0.064 rad")
    return {"wheelbase_m": wheelbase, "track_m": track, "pitch_rad": pitch,
            "foot_z_mean_m": float(np.mean([feet[c][2] for c in CORNERS])),
            "compact_feet_body_xyz": {c: feet[c].tolist() for c in CORNERS},
            "wheel_radius_m": WHEEL_RADIUS_M,
            "joint_limits_rad": JOINT_LIMITS.tolist()}


def check_default_pose(q_default_by_corner: dict, *, tolerance_m: float = 0.002) -> dict:
    """Compare the runtime default leg pose against the published USD foot positions.

    Called by the evaluator once the articulation is live, so a mismatch between
    this model and the loaded asset aborts the run instead of producing a
    plausible-looking descent from the wrong geometry.
    """
    report = {}
    for corner, expected in DEFAULT_FOOT_BODY_XYZ.items():
        actual = foot_body_xyz(corner, *q_default_by_corner[corner])
        error = float(np.linalg.norm(actual - np.asarray(expected)))
        report[corner] = {"model": actual.tolist(), "published": list(expected), "error_m": error}
        if error > tolerance_m:
            raise AssertionError(
                f"{corner} default foot position differs from the published USD value by "
                f"{error:.4f} m (model {np.round(actual, 5)}, published {np.round(expected, 5)})")
    return report
