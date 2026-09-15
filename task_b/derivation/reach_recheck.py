"""Independent re-derivation of the fingertip floor with a different optimiser path."""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.arm_kinematics import fk
from task_e_geometry import JOINT_LOWER, JOINT_UPPER
from scipy.optimize import differential_evolution
TIP=0.1358
def tip_z(q): 
    p=fk(q); return float((p[:3,3]+TIP*p[:3,2])[2])
res=differential_evolution(tip_z, list(zip(JOINT_LOWER+1e-6, JOINT_UPPER-1e-6)),
                           seed=7, maxiter=400, popsize=40, tol=1e-11, polish=True)
p=fk(res.x)
tip=p[:3,3]+TIP*p[:3,2]
print("differential_evolution  fingertip body-frame:", np.round(tip,5),
      " z=%.4f  (-> %.4f below the body origin)" % (tip[2], -tip[2]))
print("gripper_base body-frame:", np.round(p[:3,3],5), " -> %.4f below" % (-p[:3,3][2]))
print("approach axis vs straight down: %.1f deg" % np.degrees(np.arccos(np.clip(-p[2,2],-1,1))))
print("q:", np.round(res.x,4))
