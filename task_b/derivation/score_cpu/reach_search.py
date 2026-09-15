import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.arm_kinematics import fk, pinch_position
from task_e_geometry import JOINT_LOWER, JOINT_UPPER, _JOINT_POS, _JOINT_ROT

HERE = Path(__file__).parent
T = np.load('/home/lybm/ATEC_Robotics_Projects_20260910/results/task_b_bootstrap/hold_seed42_03/telemetry.npz')
R = Rotation.from_quat(T['base_quat'][-1][[1,2,3,0]]).as_matrix()
base_z = float(T['base_xyz'][-100:,2].mean())
lo, hi = JOINT_LOWER[:5] + .01, JOINT_UPPER[:5] - .01
bounds = list(zip(lo, hi))
rng = np.random.default_rng(20260910)
def pose(q): return fk(np.r_[q, 0.])
def points(q):
    p=np.array([.2,0,.1]); r=np.eye(3); ps=[p.copy()]
    for i,a in enumerate(np.r_[q[:5],0.]):
        p=p+r@_JOINT_POS[i]; r=r@_JOINT_ROT[i]
        c,s=np.cos(a),np.sin(a); r=r@np.array([[c,-s,0],[s,c,0],[0,0,1]])
        ps.append(p.copy())
    return np.array(ps)

global_fit=differential_evolution(lambda q: pose(q)[2,3], bounds, seed=22, tol=1e-10, popsize=20,maxiter=400,polish=True)
print('global',global_fit.fun,global_fit.x,pose(global_fit.x)[:3,3],flush=True)
rows=[]
for x,y in [(x,y) for x in [.45,.50,.55,.60,.65,.70,.75,.80] for y in [0.,.2]]:
    cons={'type':'eq','fun':lambda q:pose(q)[:2,3]-[x,y]}
    fits=[]
    for q0 in [global_fit.x]+[rng.uniform(lo,hi) for _ in range(24)]:
        opt=minimize(lambda q: pose(q)[2,3],q0,bounds=bounds,constraints=cons,method='SLSQP',options={'maxiter':160,'ftol':1e-11})
        if np.linalg.norm(cons['fun'](opt.x))<1e-5: fits.append(opt)
    if not fits:
        rows.append(dict(x=x,y=y,feasible=False));continue
    fit=min(fits,key=lambda f:f.fun); q=np.r_[fit.x,0.]; p=fk(q)[:3,3]
    pg=fk(q); gz=base_z+(R@p)[2]
    row=dict(x=x,y=y,feasible=True,q=q.tolist(),gripper_body=p.tolist(),pinch_body=pinch_position(q).tolist(),gripper_world_z_observed_hold=gz,vertical_gap_mustard=gz-(.045+.191301/2),vertical_gap_sugar=gz-(.045+.09268/2),chain_origins=points(q).tolist(),down_axis=pg[:3,2].tolist())
    rows.append(row);print(json.dumps(row),flush=True)
out=dict(base_z=base_z,base_rotation=R.tolist(),bounds=np.array(bounds).tolist(),global_min=dict(q=np.r_[global_fit.x,0.].tolist(),p=pose(global_fit.x)[:3,3].tolist()),rows=rows)
(HERE/'reach_search.json').write_text(json.dumps(out,indent=2)+'\n')
