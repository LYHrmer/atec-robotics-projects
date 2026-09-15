import json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

P=Path(__file__).parent;source=json.loads((P/'stance_geometry.json').read_text()); boxes=source['local_visual_bounds']; frames=source['joint_frames']
a=np.load('/home/lybm/ATEC_Robotics_Projects_20260910/results/task_b_bootstrap/hold_seed42_03/telemetry.npz');names=a['joint_names'].tolist()
R=Rotation.from_quat(a['base_quat'][-1][[1,2,3,0]]).as_matrix(); bz=float(a['base_xyz'][-100:,2].mean())
def fk(c,q):
 t=np.eye(4); ts={}
 for l,v in zip(['hip','thigh','calf','foot'],list(q)+[0.]):
  f=frames[c+'_'+l+'_joint']; r=np.eye(4);r[:3,:3]=Rotation.from_euler(f['axis'].lower(),v).as_matrix()
  t=t@np.array(f['parent_frame'])@r@np.linalg.inv(np.array(f['child_frame']));ts[l]=t.copy()
 return ts
def data(c,q):
 ts=fk(c,q);tf=ts['foot']; lo,hi=np.array(boxes[c+'_foot']);center=(lo+hi)/2;rad=max(hi[0]-lo[0],hi[2]-lo[2])/2
 p=tf[:3,:3]@center+tf[:3,3];rz=(R@tf[:3,1])[2];bottom=bz+(R@p)[2]-rad*np.sqrt(max(0,1-rz*rz))
 lt,ht=np.array(boxes[c+'_thigh']);corners=np.array(list(itertools.product(*zip(lt,ht))));w=(corners@ts['thigh'][:3,:3].T+ts['thigh'][:3,3])@R.T
 return p,bottom,float(w[:,2].min()+bz),float((R@ts['calf'][:3,3])[2]+bz)
out=dict(note='These are desired ACTUAL settled joint references for slow stance adjustment under observed stance-hold feedback, not guarantees for raw open-loop joint target offsets. Static wheel geometry only.',base_z=bz,body_R=R.tolist(),candidates={})
for label,halfx,halfy in [('moderate_short_wide',.38,.35),('stronger_short_wide',.34,.36)]:
 rows={}
 for c in ['FR','FL','RR','RL']:
  idx=[names.index(c+'_'+l+'_joint') for l in ['hip','thigh','calf']];q0=a['q'][-100:,idx].mean(0);xy=np.array([halfx if c[0]=='F' else -halfx,halfy if c[1]=='L' else -halfy])
  fit=least_squares(lambda q:np.r_[data(c,q)[0][:2]-xy,data(c,q)[1]-.047],q0,bounds=([-.7,-.6,-2.7],[.7,3.,-.55]),gtol=1e-12,xtol=1e-12,ftol=1e-12)
  p,bottom,thigh,knee=data(c,fit.x);defaults=np.array([.1 if c[1]=='L' else -.1,.8 if c[0]=='F' else 1.,-1.5])
  rows[c]=dict(actual_q_reference=fit.x.tolist(),observed_hold_q=q0.tolist(),delta_from_hold=(fit.x-q0).tolist(),delta_from_official_default=(fit.x-defaults).tolist(),
    wheel_center_body=p.tolist(),wheel_bottom_world_z=bottom,thigh_visual_box_min_z=thigh,knee_world_z=knee,residual_norm=float(np.linalg.norm(fit.fun)))
 out['candidates'][label]=dict(wheelbase=2*halfx,track=2*halfy,legs=rows)
(P/'stance_ik_candidates.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
