import itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
P=Path(__file__).parent; source=json.loads((P/'stance_geometry.json').read_text());frames=source['joint_frames'];boxes=source['local_visual_bounds']
run=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/first_reach_compact_seed42_02');a=np.load(run/'telemetry.npz');names=a['joint_names'].tolist()
def chain(c,q):
 t=np.eye(4);ts={}
 for l,v in zip(['hip','thigh','calf','foot'],[*q,0.]):
  j=frames[c+'_'+l+'_joint'];r=np.eye(4);r[:3,:3]=Rotation.from_euler(j['axis'].lower(),v).as_matrix();t=t@np.array(j['parent_frame'])@r@np.linalg.inv(np.array(j['child_frame']));ts[l]=t.copy()
 return ts
def wheel_and_clearance(c,q,R,bz):
 ts=chain(c,q);f=ts['foot'];lo,hi=np.array(boxes[c+'_foot']);center=(lo+hi)/2;rad=max(hi[0]-lo[0],hi[2]-lo[2])/2;p=f[:3,:3]@center+f[:3,3];axis=R@f[:3,1];bottom=bz+(R@p)[2]-rad*np.sqrt(max(0,1-axis[2]**2))
 lo,hi=np.array(boxes[c+'_thigh']);corners=np.array(list(itertools.product(*zip(lo,hi))));v=(corners@ts['thigh'][:3,:3].T+ts['thigh'][:3,3])@R.T
 return p,bottom,float(v[:,2].min()+bz),float((R@ts['calf'][:3,3])[2]+bz)
out=dict(run=str(run),note='Static inverse kinematics keeps approximate wheel world positions and wheel-bottom heights while lowering body. Joint increments are a backup slow reference adjustment, not a validated controller or guaranteed height change.',snapshots=[])
for i in [1999,3999]:
 R=Rotation.from_quat(a['base_quat'][i][[1,2,3,0]]).as_matrix();bz=float(a['base_xyz'][i,2]);record=dict(step=i+1,original_body_z=bz,pitch_rad=float(Rotation.from_matrix(R).as_euler('xyz')[1]),lowerings=[])
 for dh in [.01,.02,.03]:
  legs={}
  for c in ['FR','FL','RR','RL']:
   ids=[names.index(c+'_'+l+'_joint') for l in ['hip','thigh','calf']];q=a['q'][i,ids].astype(float);p0,bottom0,_,_=wheel_and_clearance(c,q,R,bz);pgoal=p0+R.T@np.array([0,0,dh]);
   opt=least_squares(lambda v:np.r_[wheel_and_clearance(c,v,R,bz-dh)[0][:2]-pgoal[:2],wheel_and_clearance(c,v,R,bz-dh)[1]-bottom0],q,bounds=([-.78,-.65,-2.70],[.78,4.40,-.55]),gtol=1e-12,xtol=1e-12,ftol=1e-12)
   p,bottom,tmin,knee=wheel_and_clearance(c,opt.x,R,bz-dh);legs[c]=dict(original_actual_q=q.tolist(),desired_actual_q=opt.x.tolist(),q_increment=(opt.x-q).tolist(),normalized_action_increment_scale_half=((opt.x-q)/.5).tolist(),wheel_bottom_z=bottom,thigh_visual_box_min_z=tmin,knee_z=knee,residual=float(np.linalg.norm(opt.fun)))
  record['lowerings'].append(dict(lowering_m=dh,target_body_z=bz-dh,legs=legs))
 out['snapshots'].append(record)
(P/'compact_lowering_ik.json').write_text(json.dumps(out,indent=2)+'\n')
for sn in out['snapshots']:
 for low in sn['lowerings']:
  if low['lowering_m']==.02:print(json.dumps(dict(step=sn['step'],body_z=low['target_body_z'],legs=low['legs']),indent=2))
