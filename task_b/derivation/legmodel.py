"""B2w leg forward kinematics, read from b2w_piper.usda, validated against the
static leg-chain IK constants already published in task_b/stance_reference.py."""
import numpy as np
from scipy.spatial.transform import Rotation

CORNERS = ("FR", "FL", "RR", "RL")
# (joint frame offset in the parent frame, joint axis), from the USD localPos0/axis
HIP   = {"FR": (np.array([ .3285, -.072, 0.]), 'x'), "FL": (np.array([ .3285, .072, 0.]), 'x'),
         "RR": (np.array([-.3285, -.072, 0.]), 'x'), "RL": (np.array([-.3285, .072, 0.]), 'x')}
THIGH = {"FR": (np.array([0., -.11973, 0.]), 'y'), "FL": (np.array([0., .11973, 0.]), 'y'),
         "RR": (np.array([0., -.11973, 0.]), 'y'), "RL": (np.array([0., .11973, 0.]), 'y')}
CALF  = {c: (np.array([0., 8.821e-05 if c in ("FR","RR") else -8.821e-05, -.35]), 'y') for c in CORNERS}
FOOT  = {c: (np.array([0., -.001 if c in ("FR","RR") else 0., -.35]), 'y') for c in CORNERS}
LIMITS = {"hip": np.deg2rad([-49.8473, 49.8473]), "thigh": np.deg2rad([-53.8580, 268.7172]),
          "calf": np.deg2rad([-161.5741, -24.6372])}

def R(axis, q):
    return Rotation.from_rotvec(np.eye(3)[{"x":0,"y":1,"z":2}[axis]] * q).as_matrix()

def foot_body_xyz(corner, hip, thigh, calf):
    """Wheel (foot link) origin in the base_link frame."""
    off, ax = HIP[corner];   p = off + R(ax, hip) @ (
        THIGH[corner][0] + R('y', thigh) @ (
        CALF[corner][0]  + R('y', calf)  @ FOOT[corner][0]))
    return p

def leg_from_mapping(entry):
    return entry   # (hip, thigh, calf)

# --- validation against the published static leg-chain IK -------------------
COMPACT = {  # task_b/stance_reference.py COMPACT_JOINT_REFERENCE
 "FR": (-.3023523168313238, .8130831661377943, -1.8754745945886573),
 "FL": ( .30696657668610144, .8173372201608101, -1.8881043202266299),
 "RR": (-.27114297155288936, .9786392186138522, -1.729057945170937),
 "RL": ( .2750100973730118, .9858544286612007, -1.7429508632418722)}
feet = {c: foot_body_xyz(c, *COMPACT[c]) for c in CORNERS}
for c in CORNERS:
    print(f"  {c} foot body xyz = {np.round(feet[c],5)}")
wb = abs((feet["FR"][0]+feet["FL"][0])/2 - (feet["RR"][0]+feet["RL"][0])/2)
tr = abs((feet["FL"][1]+feet["RL"][1])/2 - (feet["FR"][1]+feet["RR"][1])/2)
zs = np.array([feet[c][2] for c in CORNERS])
print(f"\n  wheelbase  {wb:.4f} m   (published ~0.76)")
print(f"  track      {tr:.4f} m   (published ~0.70)")
print(f"  foot z     {zs.mean():+.4f} m  spread {zs.max()-zs.min():.4f}")
import re, pathlib
src = pathlib.Path("task_b/stance_reference.py").read_text()
print("\n  joint-limit check vs USD:")
for k, v in LIMITS.items(): print(f"    {k}: USD {np.round(v,4)}")
