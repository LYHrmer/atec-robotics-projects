"""Lowest world z the Piper fingers can reach, over the whole bounded joint space.

Independent of any IK: multi-start optimisation of fingertip height, plus a dense
random sweep as a cross-check that the optimum is not a solver artefact.
"""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.arm_kinematics import fk
from task_e_geometry import JOINT_LOWER, JOINT_UPPER
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ARM_BASE_BODY = np.array([.2, 0., .1])
FINGER_TIP = 0.1358

def tip_body(q):
    p = fk(q)
    return p[:3,3] + FINGER_TIP * p[:3,2]

def slab_z_range(q, n=9):
    """z of points sampled along the finger slab (gripper_base -> fingertip)."""
    p = fk(q)
    ts = np.linspace(0., FINGER_TIP, n)
    pts = p[:3,3][None,:] + ts[:,None]*p[:3,2][None,:]
    return pts[:,2].min(), pts[:,2].max()

lo, hi = JOINT_LOWER + 1e-6, JOINT_UPPER - 1e-6
rng = np.random.default_rng(0)

# dense random sweep (world frame: body origin at 0.5148, no tilt)
BASE_H = 0.5148
N = 400000
Q = rng.uniform(lo, hi, size=(N, 6))
P = np.empty((N,3))
for i in range(N):
    P[i] = tip_body(Q[i])
world_z = P[:,2] + BASE_H
k = int(np.argmin(world_z))
print("random sweep  n=%d  min fingertip world z = %.4f m   at q=%s" % (N, world_z[k], np.round(Q[k],3)))

# multi-start local optimisation from the best random starts
order = np.argsort(world_z)[:20]
best = None
for i in order:
    r = minimize(lambda q: tip_body(q)[2], Q[i], method="L-BFGS-B", bounds=list(zip(lo,hi)),
                 options=dict(maxiter=2000, ftol=1e-14))
    val = tip_body(r.x)[2]
    if best is None or val < best[0]:
        best = (val, r.x.copy())
print("refined       min fingertip world z = %.4f m   at q=%s" % (best[0]+BASE_H, np.round(best[1],4)))
lo_s, hi_s = slab_z_range(best[1])
print("finger slab spans world z %.4f .. %.4f (lowest fingertip %.4f)" % (lo_s+BASE_H, hi_s+BASE_H, lo_s+BASE_H))
print("arm base is at world z %.4f;  finger tip extends %.4f m below the arm base"
      % (BASE_H + ARM_BASE_BODY[2], BASE_H + ARM_BASE_BODY[2] - (best[0]+BASE_H)))

# object tops, for comparison
print("\nobject tops above ground if at rest: mustard 0.1913  sugar 0.0927  banana 0.0386")
print("reachable fingertip floor above ground: %.4f" % (best[0]+BASE_H))
