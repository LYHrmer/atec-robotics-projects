import json, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.arm_kinematics import ee_camera_transform, head_camera_transform

P=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/first_reach_seed42_01')
a=np.load(P/'telemetry.npz'); traces=[json.loads(l) for l in (P/'trace.jsonl').read_text().splitlines()]
R=Rotation.from_quat(a['base_quat'][:,[1,2,3,0]]).as_matrix()
names=a['joint_names'].tolist(); ids=[names.index('arm_joint'+str(i)) for i in range(1,9)]
rows=[]
for i,t in enumerate(traces):
 d=t.get('policy_debug') or {}
 for c in d.get('candidates',[]):
  p=np.asarray(c['body_point']); pw=R[i]@p+a['base_xyz'][i]
  delta=pw-a['object_xyz'][i]; oi=int(np.argmin(np.linalg.norm(delta[:,:2],axis=1))); e=delta[oi]
  cam=ee_camera_transform(a['q'][i,ids]) if c['source']=='ee' else head_camera_transform()
  recompute=cam[:3,:3]@np.asarray(c['camera_point'])+cam[:3,3]
  row=dict(step=int(t['step']),source=c['source'],selected=c==d.get('selected'),body_point=p.tolist(),world_point=pw.tolist(),nearest_object_index_1based=oi+1,
      nearest_object_world=a['object_xyz'][i,oi].tolist(),delta_xyz=e.tolist(),distance_3d=float(np.linalg.norm(e)),distance_xy=float(np.linalg.norm(e[:2])),
      camera_point=c['camera_point'],pixel_uv=c['pixel_uv'],bbox_xywh=c['bbox_xywh'],height_span_m=c['height_span_m'],
      q5=float(a['q'][i,ids[4]]),body_recompute_error=float(np.linalg.norm(recompute-p)))
  rows.append(row)
summary={}
for source in ['head','ee']:
 rr=[r for r in rows if r['source']==source]; selected=[r for r in rr if r['selected']]
 for label,group in [(source,rr),(source+'_selected',selected)]:
  if group:
   delta=np.array([r['delta_xyz'] for r in group]); xy=np.array([r['distance_xy'] for r in group])
   summary[label]=dict(count=len(group),delta_xyz_median=np.median(delta,axis=0).tolist(),delta_xyz_min=delta.min(0).tolist(),delta_xyz_max=delta.max(0).tolist(),xy_median=float(np.median(xy)),xy_max=float(xy.max()),first=group[0],middle=group[len(group)//2],last=group[-1])
out=dict(run=str(P),diagnostic_only=True,note='RGB-D candidate points are surface statistics, not declared object roots. World root is used only for offline diagnostic matching.',summary=summary,rows=rows,
 input_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['telemetry.npz','trace.jsonl','result.json']})
dest=P.parent/'first_reach_seed42_01_visual_point_audit.json';dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
