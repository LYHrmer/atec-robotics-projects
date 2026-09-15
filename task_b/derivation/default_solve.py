import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.leg_kinematics import (CORNERS, DEFAULT_FOOT_BODY_XYZ, JOINT_LIMITS,
                                   foot_body_xyz, descend, foot_xy, validate)
from scipy.optimize import least_squares
print("validate():", {k:(round(v,4) if isinstance(v,float) else v) for k,v in validate().items() if not isinstance(v,dict)})
lo, hi = JOINT_LIMITS[:,0]+1e-6, JOINT_LIMITS[:,1]-1e-6
print("\nrecovering the default leg pose from the published foot positions:")
defaults={}
for c in CORNERS:
    tgt=np.array(DEFAULT_FOOT_BODY_XYZ[c])
    best=None
    for guess in ([0.,.8,-1.6],[0.,1.0,-1.9],[0.,.6,-1.2],[-.3,.9,-1.9],[.3,.9,-1.9],[0.,1.2,-2.0]):
        f=least_squares(lambda q: foot_body_xyz(c,*q)-tgt, np.clip(guess,lo,hi), bounds=(lo,hi), max_nfev=800)
        e=float(np.linalg.norm(foot_body_xyz(c,*f.x)-tgt))
        if best is None or e<best[0]: best=(e,f.x.copy())
    defaults[c]=best[1]
    print(f"  {c}: q={np.round(best[1],4)}  residual={best[0]:.2e} m")
print("\ndescend from those defaults:")
for drop in (0.0,.03,.06,.09,.12,.15,.18,.21,.24):
    worst=0.; lim=0.
    for c in CORNERS:
        q,r=descend(c, defaults[c], drop); worst=max(worst,r)
        lim=max(lim, float(np.max((q-lo)/(hi-lo))), float(np.max((hi-q)/(hi-lo))))
    print(f"  drop {drop:.2f}  worst IK residual {worst:.2e}  closest approach to a limit: {(1-lim)*100:5.1f}% of range left")
