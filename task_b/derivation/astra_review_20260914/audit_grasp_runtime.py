"""Read-only independent geometry check of a completed grasp probe."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom

parser=argparse.ArgumentParser();parser.add_argument('run_directory',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
p=args.run_directory;r=json.loads((p/'result.json').read_text());t=np.load(p/'telemetry.npz');rows=[json.loads(x) for x in (p/'trace.jsonl').open()];events=json.loads((p/'scoring_events.json').read_text());dt=float(json.loads((p/'environment_metadata.json').read_text())['step_dt'])
report={'run_id':p.name,'scope':'Offline actual recorded pose/mesh check; no policy input, no reward-derived grasp claim.','steps':r['steps'],'stop_reason':r['stop_reason'],'score_raw_total':r['score_raw_total'],'reward_terms':r['reward_term_totals_raw'],'official_terminated':r['terminated'],'official_truncated':r['truncated'],'probe_states':{},'actual_object_quaternion_used':True,'verified_lift_and_follow':False}
phases=np.array([x.get('policy_debug',{}).get('probe_phase') or '' for x in rows]);indices=np.flatnonzero(phases!='')
if not len(indices):
 report['reason']='probe_not_entered';args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(0)
if not events:raise ValueError('No original score event available to independently associate a target')
s=events[0]['state'];dist=np.linalg.norm(np.asarray(s['object_xyz'])-np.asarray(s['gripper_xyz']),axis=1);obj=int(np.argmin(dist));report['object_index_1based']=obj+1
if not 6<=obj<12:raise ValueError('This mesh review is scoped to the mustard asset; do not reuse the wrong mesh')
asset=Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/006_mustard_bottle.usd');stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();verts=[]
for prim in stage.Traverse():
 if prim.IsA(UsdGeom.Mesh):
  v=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());M=np.asarray(cache.GetLocalToWorldTransform(prim)).T;verts.append(v@M[:3,:3].T+M[:3,3])
verts=np.concatenate(verts)
start=int(indices[0]);end=r['final_state_before_close'];names=t['joint_names'].tolist();j7=names.index('arm_joint7');j8=names.index('arm_joint8')
steps=np.r_[t['step'][start:],r['steps']+1];times=(steps-1)*dt
objpos=np.concatenate([t['object_xyz'][start:,obj],np.asarray(end['object_xyz'])[None,obj]])
objquat=np.concatenate([t['object_quat'][start:,obj],np.asarray(end['object_quat'])[None,obj]])
gpos=np.concatenate([t['gripper_xyz'][start:],np.asarray(end['gripper_xyz'])[None]])
gquat=np.concatenate([t['gripper_quat'][start:],np.asarray(end['gripper_quat'])[None]])
q=np.concatenate([t['q'][start:],np.asarray(end['q'])[None]])
phase=np.concatenate([phases[start:],np.array([phases[-1]])])
if not np.isfinite(objquat).all() or np.any(np.linalg.norm(objquat,axis=1)<.5):raise ValueError('Invalid measured object quaternion')
OR=Rotation.from_quat(objquat[:,[1,2,3,0]]).as_matrix();GR=Rotation.from_quat(gquat[:,[1,2,3,0]]).as_matrix()
lowest=np.array([np.min(verts@mat[2])+pos[2] for mat,pos in zip(OR,objpos)])
relative=np.einsum('nji,nj->ni',GR,objpos-gpos)
width=q[:,j7]-q[:,j8]
report['pre_close_baseline']={'step':int(steps[0]),'state_timing':'pre_step','object_root_world_m':objpos[0].tolist(),'lowest_original_mesh_world_z_m':float(lowest[0]),'gripper_world_m':gpos[0].tolist(),'object_quaternion_wxyz':objquat[0].tolist()}
report['object_quaternion_norm_max_error']=float(np.max(np.abs(np.linalg.norm(objquat,axis=1)-1)))
for name in ['PROBE_CLOSE','PROBE_LIFT','PROBE_OBSERVE']:
 ii=np.flatnonzero(phase==name)
 if not len(ii):continue
 report['probe_states'][name]={'first_step':int(steps[ii[0]]),'last_state_step':int(steps[ii[-1]]),'samples':int(len(ii)),'state_span_s':float(times[ii[-1]]-times[ii[0]]),'jaw_width_min_max_m':[float(width[ii].min()),float(width[ii].max())],'object_root_rise_min_max_from_preclose_m':[float((objpos[ii,2]-objpos[0,2]).min()),float((objpos[ii,2]-objpos[0,2]).max())],'mesh_surface_clearance_min_max_from_preclose_m':[float((lowest[ii]-lowest[0]).min()),float((lowest[ii]-lowest[0]).max())],'gripper_z_rise_min_max_m':[float((gpos[ii,2]-gpos[0,2]).min()),float((gpos[ii,2]-gpos[0,2]).max())]}
report['final_actual']={'object_root_rise_m':float(objpos[-1,2]-objpos[0,2]),'minimum_mesh_surface_rise_m':float(lowest[-1]-lowest[0]),'gripper_z_rise_m':float(gpos[-1,2]-gpos[0,2]),'object_world_xyz':objpos[-1].tolist(),'object_quaternion_wxyz':objquat[-1].tolist(),'jaw_width_m':float(width[-1]),'relative_root_in_gripper_m':relative[-1].tolist()}
window=np.flatnonzero((phase=='PROBE_OBSERVE')&(times>=times[-1]-1.-1e-8))
if len(window)>=2:
 duration=float(times[window[-1]]-times[window[0]]);motion=float(np.max(np.linalg.norm(relative[window]-relative[window[0]],axis=1)))
 checks={'at_least_1s_actual_observe_span':duration>=1.-1e-8,'object_root_rise_at_least_15mm':bool(np.all(objpos[window,2]-objpos[0,2]>=.015)),'lowest_mesh_surface_clears_old_support_by_10mm':bool(np.all(lowest[window]-lowest[0]>=.01)),'jaws_nonempty':bool(np.all(width[window]>=.006)),'relative_root_motion_below_20mm':motion<=.02,'no_official_failure':not r['terminated'] and not r['truncated']}
 report['last_observation_window']={'state_span_s':duration,'samples':len(window),'relative_root_motion_max_m':motion,'checks':checks};report['verified_lift_and_follow']=all(checks.values())
report['maximum_illegal_force_N']=float(np.max(np.abs(t['illegal_force'])))
report['limits']=['Recorded actual object quaternions are used, not initial OTHER_QUAT.','Mesh lower-surface rise is relative to its measured pre-close support level; no unverified zero-height floor assumption.','Original mesh geometry does not attest exact cooked collision hull or friction/contact forces.','Nonempty jaws alone are not grasp proof; positive proximity score alone is not grasp proof.','Passing this local lift/follow check would not prove transport, delivery or full Task B completion.']
report['source_sha256']={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['result.json','telemetry.npz','scoring_events.json','trace.jsonl']};report['asset_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest();args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
