"""Offline finger normal-contact diagnostics; never a force-to-object or grasp proof."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
r=json.loads((a.run/'result.json').read_text())
m=json.loads((a.run/'environment_metadata.json').read_text())
t=np.load(a.run/'telemetry.npz')
rows=[json.loads(x) for x in (a.run/'trace.jsonl').open()]
f=np.asarray(t['finger_contact_forces_w'],dtype=float)
assert f.shape==(r['steps'],2,3) and np.isfinite(f).all()
assert np.array_equal(t['step'],[x['step'] for x in rows])
dt=float(t['dt']); phases=np.array([x['policy_debug'].get('probe_phase') or '' for x in rows])
width=np.asarray(t['q'][:,t['joint_names'].tolist().index('arm_joint7')]-t['q'][:,t['joint_names'].tolist().index('arm_joint8')])
mask_probe=phases!=''; first=np.flatnonzero(mask_probe)[0]
norm=np.linalg.norm(f,axis=2); h=np.linalg.norm(f[:,:,:2],axis=2)

def scalar(v):
 return {'mean':float(np.mean(v)),'median':float(np.median(v)),'min':float(np.min(v)),'max':float(np.max(v)),
         'p10':float(np.quantile(v,.1)),'p90':float(np.quantile(v,.9))}

def longest_span(mask, indices):
 longest=length=0
 for k,value in enumerate(mask):
  if k and indices[k] != indices[k-1]+1: length=0
  length=length+1 if value else 0;longest=max(longest,length)
 return {'samples':int(longest),'endpoint_span_s':max(0,longest-1)*dt}

def stats(indices):
 data={'first_step':int(t['step'][indices[0]]),'last_step':int(t['step'][indices[-1]]),
       'sample_count':len(indices),'endpoint_span_s':float(indices[-1]-indices[0])*dt,
       'jaw_width_m':scalar(width[indices]),'fingers':{}}
 for j,name in enumerate(m['physics_diagnostics']['finger_contact_body_names']):
  active=norm[indices,j]>.1
  data['fingers'][name]={'world_xyz_mean_N':f[indices,j].mean(axis=0).tolist(),
       'normal_resultant_norm_N':scalar(norm[indices,j]),'world_horizontal_resultant_N':scalar(h[indices,j]),
       'signed_world_up_normal_force_on_finger_N':scalar(f[indices,j,2]),
       'sample_fraction_above_norm_N':{str(v):float(np.mean(norm[indices,j]>v)) for v in (.1,.5,1.)},
       'normal_elevation_above_horizontal_deg_when_norm_above_point_one':
       scalar(np.rad2deg(np.arctan2(f[indices,j,2][active],h[indices,j][active]))) if np.any(active) else None}
 data['dual_contact_sample_fraction']={str(v):float(np.mean(np.all(norm[indices]>v,axis=1))) for v in (.1,.5,1.)}
 data['longest_consecutive_dual_contact_norm_above_point_one_N']=longest_span(np.all(norm[indices]>.1,axis=1), indices)
 data['opposite_sum_of_finger_normal_Z_N']=scalar(-np.sum(f[indices,:,2],axis=1))
 return data

report={'run_id':a.run.name,'steps':r['steps'],'stop_reason':r['stop_reason'],
        'score_raw_total':r['score_raw_total'],'scope':__doc__,
        'timing':'same-row pre-step cached finger normal force and actual q/object/gripper poses; no action-poststate conflation',
        'sensor_semantics_source':'/home/lybm/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py:74',
        'sensor_semantics':'sum of normal contact forces on each finger body in world coordinates; excludes tangential/friction forces; unfiltered by contacted object',
        'materials':{},'stages':{}}
for name,asset in m['physics_diagnostics']['assets'].items():
 if name not in ('robot','object_10'):continue
 values=np.asarray(asset['material_properties'])
 report['materials'][name]={'shape_count':len(values),'distinct_static_dynamic_restitution_rows':np.unique(values,axis=0).tolist(),'masses':asset['masses']}
for phase in ('PROBE_CLOSE','PROBE_LIFT','PROBE_OBSERVE'):
 ix=np.flatnonzero(phases==phase)
 if not len(ix):continue
 report['stages'][phase]=stats(ix)
 report['stages'][phase+'_first_half_second']=stats(ix[:min(len(ix),26)])
 report['stages'][phase+'_last_half_second']=stats(ix[-min(len(ix),26):])
 if phase=='PROBE_CLOSE':
  issued=ix[[rows[i]['policy_debug'].get('probe_finger_target_commanded',False) for i in ix]]
  if len(issued):report['stages']['PROBE_CLOSE_full_target_commanded']=stats(issued)
 if phase=='PROBE_LIFT':
  nonempty=ix[width[ix]>=.006]
  if len(nonempty):report['stages']['PROBE_LIFT_nonempty_samples']=stats(nonempty)
report['limitations']=[
 'Nonzero net normal force proves sampled body contact, but it is not object-specific. Exact contacts to mustard versus other links require filtered contact data.',
 'Normal force world Z is force ON the finger. Its opposite is only the counterpart normal contribution; omitted tangential friction prevents computing actual object support.',
 'Samples are cached at control ticks, not a continuous contact trace or all physics substeps; zero sampled net force does not prove all contact absent between ticks.',
 'All robot and target shape material coefficients are live PhysX getter values; this does not measure effective grip strength or static-friction utilization.',
 'Read the independently computed actual-quaternion original-mesh clearance report for physical lift/follow.']
report['source_sha256']={name:hashlib.sha256((a.run/name).read_bytes()).hexdigest() for name in ('result.json','environment_metadata.json','telemetry.npz','trace.jsonl')}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
