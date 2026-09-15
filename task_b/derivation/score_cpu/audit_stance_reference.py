import json,sys,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,'/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.control import ActionSchema,ActionTerm
from task_b.stance_reference import StanceReference,COMPACT_JOINT_REFERENCE,COMPACT_SHORT_JOINT_REFERENCE
from task_b.stance_hold import StanceHold

P=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/reach_probe_seed42_01');m=json.loads((P/'environment_metadata.json').read_text());a=np.load(P/'telemetry.npz');n=a['joint_names'].tolist()
cfg=m['action_schema']; defaults=dict(zip(cfg['articulation_joint_names'],cfg['default_joint_pos'])); soft=dict(zip(cfg['articulation_joint_names'],cfg['soft_joint_pos_limits']));hard=dict(zip(n,m['articulation']['joint_pos_limits']))
def fixture(reorder=False,scale=.5):
 names=tuple(n[::-1] if reorder else n); groups=cfg['terms'][::-1] if reorder else cfg['terms'];ts=[];start=0
 for t in groups:
  joints=tuple(t['joint_names'][::-1] if reorder else t['joint_names']);ts.append(ActionTerm(t['name'],start,t['dim'],joints,t['mode'],scale if t['name']=='joint_leg' else t['scale'],t['use_default_offset'],None,'resolved'));start+=t['dim']
 s=ActionSchema.from_terms(ts,names,[defaults[v] for v in names],[[np.nan if b is None else b for b in soft[v]] for v in names],24)
 obs_names=tuple(np.roll(names,7));return s,obs_names
def observation(q_by_name,names):
 obs=np.zeros(84);obs[9:12]=[.064,0,-np.sqrt(1-.064**2)];obs[12:36]=[q_by_name[v]-defaults[v] for v in names];return obs
def raises(fn):
 try:fn()
 except (ValueError,TypeError):return True
 return False
checks={};evidence={};s,on=fixture();obs=observation(dict(zip(n,a['q'][99])),on);random=np.linspace(-.4,.4,24)
off=StanceReference(s,on);checks['off_preserves_entire_action_exactly']=np.array_equal(off.apply(random,obs),random) and not off.requires_wheel_stop
checks['input_action_unmodified']=np.array_equal(random,np.linspace(-.4,.4,24))
checks['duplicate_observation_names_rejected']=raises(lambda:StanceReference(s,on[:-1]+(on[0],)))
checks['compact_requires_twenty_settle_samples']=raises(lambda:StanceReference(s,on,profile='compact',settle_calls=19))
checks['unsupported_profile_rejected']=raises(lambda:StanceReference(s,on,profile='bad'))
checks['duration_outside_4_to_6_rejected']=raises(lambda:StanceReference(s,on,transition_seconds=3.))
checks['missing_hard_joint_mapping_rejected']=raises(lambda:StanceReference(s,on,profile='compact',hard_joint_pos_limits={}))
checks['nan_action_rejected']=raises(lambda:off.apply(np.full(24,np.nan),obs))
checks['nonfinite_proprio_rejected']=raises(lambda:off.apply(np.zeros(24),np.full(84,np.inf)))
checks['float32_overflow_action_rejected']=raises(lambda:off.apply(np.full(24,1e100),obs))
checks['calibration_rejects_nonzero_leg_request']=raises(lambda:StanceReference(s,on,profile='compact').apply(np.ones(24),obs))
checks['unexpected_action_clipping_rejected']=raises(lambda:StanceReference(replace(s,terms=tuple(replace(t,clip={'.*':[-1,1]}) if t.name=='joint_leg' else t for t in s.terms)),on))
for profile,targets in [('compact',COMPACT_JOINT_REFERENCE),('compact_short',COMPACT_SHORT_JOINT_REFERENCE)]:
 for reorder,scale in [(False,.5),(True,.23)]:
  key=f'{profile}_reorder{reorder}_scale{scale}';s,on=fixture(reorder,scale);l=s.term('joint_leg');w=s.term('joint_wheel');arm=s.term('joint_arm');keep=np.r_[np.arange(w.start,w.stop),np.arange(arm.start,arm.stop)]
  r=StanceReference(s,on,profile=profile,hard_joint_pos_limits=hard);holder=StanceHold(s,on)
  inp=np.linspace(-.4,.4,24);inp[l.start:l.stop]=0.;cal=[]
  for j in range(100):
   qb=dict(zip(n,a['q'][j]));ob=observation(qb,on);out=r.apply(inp,ob);held=holder.apply(out,ob);assert np.array_equal(out,inp);assert r.requires_wheel_stop;cal.append([qb[v] for v in l.joint_names])
  ref=np.mean(cal[-20:],axis=0);assert np.allclose(r.reference,ref,atol=1e-12) and np.allclose(holder.reference,ref,atol=1e-12)
  target=np.array([targets[v.split('_')[0]][('hip','thigh','calf').index(v.split('_')[1])] for v in l.joint_names]);desireds=[];maxpreserve=0.;dq=np.zeros(12)
  for j in range(250):
   qb=dict(zip(n,a['q'][99]));qb.update(dict(zip(l.joint_names,ref+dq)));ob=observation(qb,on);out=r.apply(inp,ob);held=holder.apply(out,ob)
   maxpreserve=max(maxpreserve,float(np.max(abs(out[keep]-inp[keep]))));dq=out[l.start:l.stop]*l.scale;actual_ref=ref+dq;desireds.append(actual_ref)
   if j<249:assert r.requires_wheel_stop
   assert np.isfinite(held).all()
  assert r.trajectory_complete and not r.requires_wheel_stop;assert np.allclose(actual_ref,target,atol=2e-15);assert maxpreserve==0
  deltas=np.diff(np.vstack([ref,desireds]),axis=0);expected_delta=target-ref
  assert np.all(deltas*np.sign(expected_delta)[None,:]>=-1e-12)
  checked_hard=np.array([hard[v] for v in l.joint_names]);assert np.all(np.array(desireds)>=checked_hard[:,0]) and np.all(np.array(desireds)<=checked_hard[:,1])
  checks[key+'_mapping_scale_profile_final_target_and_holder_composition']=True
  checks[key+'_wheel_arm_exact_preservation_and_wheel_gate']=True
  checks[key+'_monotone_hard_bounded_5s_transition']=True
  evidence[key]=dict(max_reference_velocity_rad_s=float(abs(deltas).max()/.02),final_target_by_joint=dict(zip(l.joint_names,target.tolist())),max_wheel_arm_error=maxpreserve)
mod=Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_b/stance_reference.py')
out=dict(verdict='pass' if all(checks.values()) else 'fail',checks=checks,check_count=len(checks),module_sha256=hashlib.sha256(mod.read_bytes()).hexdigest(),evidence=evidence,limits_source=str(P/'environment_metadata.json'),reference_observations_source=str(P/'telemetry.npz'),claim='CPU reference/mapping/bounds/composition checks only; no dynamic success proven')
dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['verdict','check_count','module_sha256','checks']},indent=2))
