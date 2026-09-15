import itertools,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics

P=Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda')
s=Usd.Stage.Open(str(P)); b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
chain={}; box={}
def transform(pos,quat):
 t=np.eye(4);t[:3,3]=list(pos);t[:3,:3]=Rotation.from_quat([*quat.GetImaginary(),quat.GetReal()]).as_matrix();return t
for p in s.Traverse():
 if p.IsA(UsdPhysics.Joint) and any(x in p.GetName() for x in ['hip_joint','thigh_joint','calf_joint','foot_joint']):
  j=UsdPhysics.Joint(p)
  chain[p.GetName()]=(transform(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),transform(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get()),p.GetAttribute('physics:axis').Get())
 if p.GetName() in [c+'_'+link for c in ['FR','FL','RR','RL'] for link in ['hip','thigh','calf','foot']]:
  bb=b.ComputeUntransformedBound(p).ComputeAlignedRange(); box[p.GetName()]=[list(bb.GetMin()),list(bb.GetMax())]

def fk(corner,q):
 t=np.eye(4);ts={}
 for link,v in zip(['hip','thigh','calf','foot'],q):
  a,z,axis=chain[corner+'_'+link+'_joint']; r=np.eye(4);r[:3,:3]=Rotation.from_euler(axis.lower(),v).as_matrix()
  t=t@a@r@np.linalg.inv(z);ts[link]=t.copy()
 return ts
def features(qs,R=np.eye(3),base=np.zeros(3)):
 rows={}
 for c,qs4 in qs.items():
  ts=fk(c,qs4);tf=ts['foot'];lo,hi=np.array(box[c+'_foot']); center=(lo+hi)/2
  local_wheelcenter=tf[:3,:3]@center+tf[:3,3];wheelcenter=R@local_wheelcenter+base
  axis=R@tf[:3,1];radius=max(hi[0]-lo[0],hi[2]-lo[2])/2
  wheelbottom=wheelcenter[2]-radius*np.sqrt(max(0,1-axis[2]**2))
  thigh=ts['thigh']; l,h=np.array(box[c+'_thigh']);corners=np.array(list(itertools.product(*zip(l,h))))
  worldcorners=(corners@thigh[:3,:3].T+thigh[:3,3])@R.T+base
  rows[c]=dict(q=list(qs4),wheel_center_body=local_wheelcenter.tolist(),wheel_center_world=wheelcenter.tolist(),wheel_bottom_world_z=float(wheelbottom),
    hip_origin_body=ts['hip'][:3,3].tolist(),knee_origin_body=ts['calf'][:3,3].tolist(),knee_origin_world=(R@ts['calf'][:3,3]+base).tolist(),
    thigh_box_min_world_z=float(worldcorners[:,2].min()))
 wheelcenters=np.array([r['wheel_center_body'] for r in rows.values()])
 height=max(.045-r['wheel_bottom_world_z'] for r in rows.values()) if np.allclose(base,0) else None
 return dict(legs=rows,wheelbase_x=float(np.mean([rows[c]['wheel_center_body'][0] for c in ['FR','FL']])-np.mean([rows[c]['wheel_center_body'][0] for c in ['RR','RL']])),
      track_y=float(np.mean([rows[c]['wheel_center_body'][1] for c in ['FL','RL']])-np.mean([rows[c]['wheel_center_body'][1] for c in ['FR','RR']])),
      required_level_base_z_for_lowest_wheel_on_ground=height)
def posture(h,tf,tr,cf,cr):
 return {c:[h if c.endswith('L') else -h,tf if c.startswith('F') else tr,cf if c.startswith('F') else cr,0.] for c in ['FR','FL','RR','RL']}
candidates={
 'official_default':posture(.1,.8,1.,-1.5,-1.5),
 'wide_default':posture(.2,.8,1.,-1.5,-1.5),
 'short_wide_A':posture(.2,1.15,.85,-2.,-2.),
 'short_wide_B':posture(.16,1.1,.9,-2.,-2.),
 'short_wide_C':posture(.2,1.05,.75,-1.8,-1.8),
}
out=dict(static_usd_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),joint_frames={k:dict(parent_frame=v[0].tolist(),child_frame=v[1].tolist(),axis=v[2]) for k,v in chain.items()},local_visual_bounds=box,
      static_candidates={k:features(v) for k,v in candidates.items()},runs={})
for name,path in [
 ('hold','/home/lybm/ATEC_Robotics_Projects_20260910/results/task_b_bootstrap/hold_seed42_03'),
 ('gain3','/home/lybm/ATEC_Experiments_20260910/task_b_score/first_reach_gain3_seed42_01'),
 ('pulse','/home/lybm/ATEC_Experiments_20260910/task_b_score/first_reach_pulse_seed42_01')]:
 p=Path(path);a=np.load(p/'telemetry.npz');names=a['joint_names'].tolist()
 def get_qs(row):return {c:[float(row[names.index(c+'_'+l+'_joint')]) for l in ['hip','thigh','calf','foot']] for c in ['FR','FL','RR','RL']}
 r=json.loads((p/'result.json').read_text());ii=[min(j,len(a['q'])-1) for j in [99,199,299,len(a['q'])-1]]
 records=[]
 for i in ii:
  R=Rotation.from_quat(a['base_quat'][i][[1,2,3,0]]).as_matrix(); f=features(get_qs(a['q'][i]),R,a['base_xyz'][i]);f.update(step=int(a['step'][i]),base_xyz=a['base_xyz'][i].tolist(),euler_xyz=Rotation.from_matrix(R).as_euler('xyz').tolist());records.append(f)
 term=r.get('terminal_pre_reset'); terminal=None
 if term:
  state=term['snapshot']['state']; R=Rotation.from_quat(np.array(state['base_quat'])[[1,2,3,0]]).as_matrix();terminal=features(get_qs(state['q']),R,np.array(state['base_xyz']));terminal.update(base_xyz=state['base_xyz'],euler_xyz=Rotation.from_matrix(R).as_euler('xyz').tolist(),illegal_forces=dict(zip(a['illegal_contact_body_names'].tolist(),state['illegal_force'])))
 out['runs'][name]=dict(records=records,terminal=terminal)
dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{'wheelbase':v['wheelbase_x'],'track':v['track_y'],'base_z':v['required_level_base_z_for_lowest_wheel_on_ground'],'wheels':{c:vv['wheel_center_body'] for c,vv in v['legs'].items()}} for k,v in out['static_candidates'].items()},indent=2))
for name,r in out['runs'].items():
 print('RUN',name)
 for f in r['records']:
  print(f['step'],'xyz',np.round(f['base_xyz'],3),'rpy',np.round(f['euler_xyz'],3), 'wheelbase',round(f['wheelbase_x'],3),'track',round(f['track_y'],3),'bottoms',{c:round(v['wheel_bottom_world_z'],3) for c,v in f['legs'].items()},'knee_z',{c:round(v['knee_origin_world'][2],3) for c,v in f['legs'].items()})
 if r['terminal']:print('TERMINAL',json.dumps(r['terminal']))
