import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.arm_kinematics import fk

P=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/reach_probe_seed42_01')
a=np.load(P/'telemetry.npz'); result=json.loads((P/'result.json').read_text())
trace=[json.loads(l) for l in (P/'trace.jsonl').read_text().splitlines()]
names=a['joint_names'].tolist()
ids=[names.index('arm_joint'+str(i)) for i in range(1,9)]
q=a['q'][:,ids]
poses=np.array([fk(row) for row in q])
R=Rotation.from_quat(a['base_quat'][:,[1,2,3,0]]).as_matrix()
world=a['base_xyz']+np.einsum('nij,nj->ni',R,poses[:,:3,3])
err=np.linalg.norm(world-a['gripper_xyz'],axis=1)
distance=np.linalg.norm(a['object_xyz']-a['gripper_xyz'][:,None,:],axis=2)
trace_distance=np.array([r['diagnostic_min_gripper_object_distance_m'] for r in trace])
worldrot=np.einsum('nij,njk->nik',R,poses[:,:3,:3])
zrow=worldrot[:,2,:]
opening=q[:,6]-q[:,7]
finger_bottom=world[:,2]-abs(zrow[:,0])*.028-abs(zrow[:,1])*(opening/2+.0265)+np.minimum(zrow[:,2]*.0593,zrow[:,2]*.1358)
probedebug=result['policy']['debug']
targets=np.array(probedebug['arm_target'])
final_gravity=a['proprio'][-1,9:12]
reach_idx=np.array([i for i,t in enumerate(trace) if t['policy_state']=='REACH'])
z=a['gripper_xyz'][:,2]
checks={
 'exactly_1000_samples':len(a['step'])==1000 and len(trace)==1000,
 'step_numbers_contiguous':bool(np.array_equal(a['step'],np.arange(1,1001))),
 'no_termination_term_fired':not bool(a['termination'].any()),
 'no_trace_termination_or_truncation':not any(t['terminated'] or t['truncated'] for t in trace),
 'reported_max_steps_stop':result['stop_reason']=='max_steps' and not result['terminated'] and not result['truncated'],
 'zero_score_confirmed':bool(np.max(abs(a['score']))==0 and np.max(abs(a['reward_terms']))==0 and result['score_raw_total']==0),
 'independent_fk_agrees_with_simulator_20um':bool(err.max()<2e-5),
 'independent_object_distance_agrees_with_trace_2um':bool(np.max(abs(distance.min(1)-trace_distance))<2e-6),
 'all_action_values_finite':bool(np.isfinite(a['action']).all()),
 'all_wheel_commands_zero':bool(np.max(abs(a['action'][:,12:16]))==0),
 'open_fingers_final':bool(q[-1,6]>.033 and q[-1,7]<-.033),
}
out=dict(run=str(P),verdict='independent static-reach runtime checks passed' if all(checks.values()) else 'inspect failed checks',checks=checks,
 input_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['telemetry.npz','trace.jsonl','result.json','environment_metadata.json','source_manifest.json']},
 kinematics_source_sha256=hashlib.sha256(Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_b/arm_kinematics.py').read_bytes()).hexdigest(),
 evidence=dict(steps=int(len(a['step'])),first_reach_step=int(reach_idx[0]+1),max_illegal_force_n=float(a['illegal_force'].max()),
  fk_max_world_error_m=float(err.max()),final_q=q[-1].tolist(),final_target_q=targets.tolist(),final_tracking_error_rad_or_m=(q[-1]-targets).tolist(),
  final_gripper_body_fk=poses[-1,:3,3].tolist(),final_gripper_world_diagnostic=a['gripper_xyz'][-1].tolist(),final_gripper_world_fk=world[-1].tolist(),
  final_base_z=float(a['base_xyz'][-1,2]),final_projected_gravity=final_gravity.tolist(),
  gripper_min_world_z=float(z.min()),final_gripper_world_z=float(z[-1]),
  final_expected_upright_mustard_vertical_gap=float(z[-1]-(.045+.191301/2)),
  final_horizontal_radius_for_20cm_proximity_on_upright_mustard=float(np.sqrt(max(0,.2**2-(z[-1]-(.045+.191301/2))**2))),
  nearest_actual_object_min_distance_m=float(distance.min()),nearest_actual_object_final_distance_m=float(distance[-1].min()),
  nominal_finger_box_min_ground_clearance_m=float((finger_bottom-.045).min()),
  last_100_body_velocity_mean=a['proprio'][-100:,:3].mean(0).tolist(),last_100_world_position_difference_velocity=np.diff(a['base_xyz'][-100:],axis=0).mean(0).tolist()),
 limitations=['Reach probe uses a synthetic body-frame target, not an actual visually approached object.',
  'No score, no physical grasp, no delivery, and no full Task B pass occurred.',
  'Upright mustard score margin is a static geometric diagnostic only, not current actual-object proximity.',
  'FK and nominal finger box do not certify complete arm-mesh, terrain or object collision freedom; official illegal-contact terms remained false.',
  'Published trace records pre-step geometry with post-step reward/termination flags; each alignment is preserved here.',
  'Probe debug target continues body-twist propagation during stationary reach, drifting vertically. It must not be treated as a current visual target.'],
 path=[dict(step=int(a['step'][i]),q=q[i].tolist(),gripper_body=poses[i,:3,3].tolist(),gripper_world=a['gripper_xyz'][i].tolist()) for i in [0,99,199,399,599,799,999]])
# Difference velocity above must divide by the logged control dt.
out['evidence']['last_100_world_position_difference_velocity']=(np.diff(a['base_xyz'][-100:],axis=0).mean(0)/float(a['dt'])).tolist()
dest=P.parent/'reach_probe_seed42_01_independent_audit.json'
dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['verdict','checks','evidence']},indent=2))
