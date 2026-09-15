"""CPU study: can the Piper put its finger slab around a Task B object, and at what park?

Body frame, metres, radians. No simulator. Verified inputs:
  * joint limits and the arm mount are checked against the real asset below
  * fk reproduced the M1 run's recorded gripper_base to 0.0 m, and predicted the
    measured world gripper_base to 0.0044 m
"""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.arm_kinematics import fk, BODY_FROM_ARM_POSITION, BODY_FROM_ARM_QUATERNION
from task_e_geometry import JOINT_LOWER, JOINT_UPPER, top_grasp_rotation
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# --- preflight -------------------------------------------------------------
assert np.allclose(BODY_FROM_ARM_POSITION, [.2, 0., .1]), "arm mount moved"
assert np.allclose(BODY_FROM_ARM_QUATERNION, [1, 0, 0, 0]), "arm mount rotation moved"
assert np.all(JOINT_LOWER < JOINT_UPPER)
assert abs(JOINT_UPPER[1] - np.deg2rad(179.9087371826172)) < 1e-9, "joint2 limit not the USD value"
FINGER_TIP = 0.1358          # finger origins along gripper local +Z
GRASP_DEPTH = 0.115          # nominal contact-slab centre
JAW_HALF = 0.035             # per-finger travel

def ik_topdown(goal, jaw_xy=(1., 0.), seed=None):
    """Position+orientation IK for a straight-down top grasp. Returns (q, err)."""
    R = top_grasp_rotation(jaw_xy)
    lo, hi = JOINT_LOWER + 1e-6, JOINT_UPPER - 1e-6
    def residual(q):
        p = fk(q)
        return np.r_[p[:3,3] - goal, .2 * Rotation.from_matrix(p[:3,:3] @ R.T).as_rotvec()]
    guesses = [np.zeros(6)] if seed is None else [seed]
    guesses += [np.array([np.arctan2(goal[1], goal[0]-.2), j2, j3, j4, j5, 0.])
                for j2, j3, j4, j5 in [(1.5,-1.8,0.,1.), (1.0,-1.5,0.,1.), (2.0,-1.5,0.,-.8),
                                       (1.5,-2.,1.4,-.8), (1.5,-2.,-1.4,-.8), (2.6,-1.0,0.,0.),
                                       (1.2,-2.2,0.,0.)]]
    best = None
    for g in guesses:
        fit = least_squares(residual, np.clip(g, lo, hi), bounds=(lo, hi), max_nfev=400,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        p = fk(fit.x)
        pe = float(np.linalg.norm(p[:3,3]-goal))
        oe = float(np.linalg.norm(Rotation.from_matrix(p[:3,:3] @ R.T).as_rotvec()))
        if best is None or pe + .2*oe < best[0]:
            best = (pe + .2*oe, fit.x.copy(), pe, oe)
    return best[1], best[2:]

# object geometry, body frame. root_z_rest = height of the object ORIGIN above ground
# when at rest, and half_h = half the object's vertical extent at rest.
OBJECTS = {  # (root_z_rest, half_h, narrow_width, wide_width)
    "mustard": (0.0957, 0.0957, 0.0582, 0.0960),   # root at centre, stood upright by OTHER_QUAT
    "sugar":   (0.0464, 0.0464, 0.0451, 0.1763),   # root at centre, thin axis vertical
    "banana":  (0.0193, 0.0193, 0.0386, 0.1972),   # root at centre, lies down
}

def study(base_h, standoff, lateral=0.0):
    out = {}
    for name, (root_rest, half_h, narrow, wide) in OBJECTS.items():
        root_body_z = -(base_h - root_rest)          # ground at body z = -base_h
        centre = np.array([standoff, lateral, root_body_z])
        goal = centre + np.array([0., 0., GRASP_DEPTH])   # slide the slab centre onto the centre
        q, (pe, oe) = ik_topdown(goal)
        p = fk(q)
        zhat, yhat = p[:3, 2], p[:3, 1]
        slab_lo = p[:3, 3]
        slab_hi = p[:3, 3] + FINGER_TIP * zhat
        s = float((centre - slab_lo) @ zhat)          # depth of object centre along approach
        r = float(np.linalg.norm((centre - slab_lo) - s * zhat))  # radial offset from approach axis
        out[name] = dict(ik_pos_err=pe, ik_ori_err=np.degrees(oe),
                         depth_in_slab=s, radial=r,
                         slab_lo_z=float(slab_lo[2]), slab_hi_z=float(slab_hi[2]),
                         obj_top_z=float(root_body_z + half_h),
                         tips_below_top=float(root_body_z + half_h - slab_hi[2]),
                         jaw_fits=bool(min(narrow, wide) <= 0.07),
                         jaw_axis_horizontality=float(abs(yhat[2])),
                         q=np.round(q, 3).tolist())
    return out

h = 0.5148   # M1's measured base height at the grasp
print(f"base height {h} m   (object rest z world: mustard "
      f"{h-(h-0.0957-(-(h-0.0957)))*0:.4f})")
print(f"{'standoff':>8} {'obj':>8} {'ikpos':>7} {'ikori':>6} {'rdepth':>7} {'radial':>7} "
      f"{'slabhi_z':>9} {'top_z':>8} {'tips_under_top':>15} {'jaw_axis|z|':>11}")
for s in np.arange(0.20, 0.75, 0.05):
    for name, res in study(h, float(s)).items():
        print(f"{s:8.2f} {name:>8} {res['ik_pos_err']:7.4f} {res['ik_ori_err']:6.1f} "
              f"{res['depth_in_slab']:7.4f} {res['radial']:7.4f} {res['slab_hi_z']:9.4f} "
              f"{res['obj_top_z']:8.4f} {res['tips_below_top']:15.4f} {res['jaw_axis_horizontality']:11.4f}")
