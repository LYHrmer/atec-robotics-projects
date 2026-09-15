import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.arm_kinematics import fk,solve_ik
from task_e_geometry import JOINT_LOWER,JOINT_UPPER
P=Path(__file__).parent;run=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/first_reach_compact_seed42_02');a=np.load(run/'telemetry.npz')
rng=np.random.default_rng(834);rows=[]
for i in [1999,3999]:
 R=Rotation.from_quat(a['base_quat'][i][[1,2,3,0]]).as_matrix();bz=float(a['base_xyz'][i,2]); oz=float(a['object_xyz'][i,9,2]);
 for x,y in [(.56,0.),(.56,.07),(.56,.10),(.56,.15),(.53,.10),(.50,.10),(.60,.10)]:
  for lowering in [0.,.01,.02,.03]:
   z=(oz-(bz-lowering)-R[2,0]*x-R[2,1]*y)/R[2,2];target=np.array([x,y,z]);seeds=[np.array([0,1.1,-.8,0,.8,0]),np.array([np.arctan2(y,x-.2),3.13999,-1.5,0,-.0884,0])]+[rng.uniform(JOINT_LOWER+1e-5,JOINT_UPPER-1e-5) for _ in range(8)]
   opts=[least_squares(lambda q:fk(q)[:3,3]-target,seed,bounds=(JOINT_LOWER+1e-6,JOINT_UPPER-1e-6),max_nfev=120) for seed in seeds];fit=min(opts,key=lambda f:np.linalg.norm(f.fun));p=fk(fit.x)[:3,3];distance=float(np.linalg.norm(p-target))
   rows.append(dict(snapshot_step=i+1,body_z=bz,pitch_rad=float(Rotation.from_matrix(R).as_euler('xyz')[1]),hypothetical_lowering_m=lowering,target_body=target.tolist(),min_distance_m=distance,within_official_20cm=distance<.2,q=fit.x.tolist(),gripper_body=p.tolist(),delta_body=(p-target).tolist()))
out=dict(run=str(run),note='Offline static reach search with diagnostic body orientation/height and actual upright mustard root height. Object truth never enters policy. Hypothetical x/y/lowering targets do not claim achieved motion.',rows=rows)
(P/'compact_reach_window.json').write_text(json.dumps(out,indent=2)+'\n')
for row in rows:
 if row['hypothetical_lowering_m'] in [0.,.02]: print(row['snapshot_step'],row['target_body'][:2],'lower',row['hypothetical_lowering_m'],'dist',round(row['min_distance_m'],5),'q',np.round(row['q'],4).tolist(),flush=True)
