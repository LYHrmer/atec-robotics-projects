"""Fingertip floor vs nose-down body pitch, arm re-optimised at each pitch."""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.arm_kinematics import fk
from task_e_geometry import JOINT_LOWER, JOINT_UPPER
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

FINGER_TIP = 0.1358
lo, hi = JOINT_LOWER+1e-6, JOINT_UPPER-1e-6
rng = np.random.default_rng(1)
BASE_H = 0.5148

def R_pitch(theta):                     # nose-down about body +y
    return Rotation.from_euler('y', theta).as_matrix()

def floor(theta, ntries=4000):
    R = R_pitch(theta)
    Q = rng.uniform(lo, hi, size=(ntries,6))
    best = None
    for q in Q:
        p = fk(q); v = p[:3,3] + FINGER_TIP*p[:3,2]
        z = (R @ v)[2]
        if best is None or z < best[0]: best = (z, q.copy())
    for scale in (0.02, 0.005):
        r = minimize(lambda q: (R @ (fk(q)[:3,3] + FINGER_TIP*fk(q)[:3,2]))[2], best[1],
                     method="L-BFGS-B", bounds=list(zip(lo,hi)), options=dict(maxiter=3000, ftol=1e-14))
        if (R @ (fk(r.x)[:3,3] + FINGER_TIP*fk(r.x)[:3,2]))[2] < best[0]:
            best = ((R @ (fk(r.x)[:3,3] + FINGER_TIP*fk(r.x)[:3,2]))[2], r.x.copy())
    return float(best[0]), best[1]

print("body origin held at %.4f m; nose-down pitch about the origin" % BASE_H)
print(f"{'pitch_deg':>9} {'fingertip_floor_world_z':>24}  {'reaches':>28}")
tops = {"mustard":0.1913, "sugar":0.0927, "banana":0.0386}
for deg in (0,5,10,15,20,25,30,35,40):
    z, q = floor(np.deg2rad(deg))
    w = BASE_H + z
    reach = ",".join(k for k,v in tops.items() if w <= v-0.03) or "-"
    print(f"{deg:9.0f} {w:24.4f}  {reach:>28}")
