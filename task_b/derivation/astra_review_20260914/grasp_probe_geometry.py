"""CPU-only nominal convex-hull/FK review, never imported by a policy."""
from pathlib import Path
import hashlib, json
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from task_b.arm_kinematics import fk

ROOT=Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model')
RUNS=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score')
OUT=Path(__file__).with_suffix('.json')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mesh_world(stage,path,cache):
    prim=stage.GetPrimAtPath(path)
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
    mat=np.asarray(cache.GetLocalToWorldTransform(prim)).T
    return points@mat[:3,:3].T+mat[:3,3]

bottle_path=ROOT/'objects/task_b/006_mustard_bottle.usd'
b=Usd.Stage.Open(str(bottle_path));bc=UsdGeom.XformCache()
bottle=mesh_world(b,'/Root/_06_mustard_bottle',bc)
bottle_R=Rotation.from_quat([0,-.707,.707,0]).as_matrix()
piper_path=ROOT/'robot/piper/piper.usd'
s=Usd.Stage.Open(str(piper_path));c=UsdGeom.XformCache()
g=np.asarray(c.GetLocalToWorldTransform(s.GetPrimAtPath('/piper/gripper_base'))).T
fingers={}
for link in ['link7','link8']:
    v=mesh_world(s,'/piper/'+link+'/collisions/'+link+'/mesh',c)
    fingers[link]=(v-g[:3,3])@g[:3,:3]
report={
 'scope':'CPU-only geometry. World/object values are retrospective diagnostics and must never enter control.',
 'assumptions':['Old runs record object root positions but no actual object quaternion. OTHER_QUAT is a nominal assumption, not measured final orientation.','Ideal convex hulls of original USD vertices are not the cooked PhysX hulls; contact offsets, friction, dynamics and centering are not solved.','Intersection possibility is not proof of bilateral contact force, grasp or lift.'],
 'nominal_object_quaternion_wxyz':[0,0,-.707,.707],
 'finger_closed_bounds_gripper_m':{k:[v.min(0).tolist(),v.max(0).tolist()] for k,v in fingers.items()},
 'source_sha256':{str(x):digest(x) for x in [bottle_path,piper_path,ROOT/'robot/piper/configuration/piper_base.usd',ROOT/'robot/piper/configuration/piper_physics.usd',Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_b/arm_kinematics.py'),Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_e_geometry.py')]},
 'runs':[]}
for run_id in ['plan_p2_lower02_seed42_01','plan_p2_fullhold_seed42_01']:
    run=RUNS/run_id;r=json.loads((run/'result.json').read_text());state=r['final_state_before_close'];z=np.load(run/'telemetry.npz');names=z['joint_names'].tolist()
    q=np.asarray(state['q'])[[names.index('arm_joint'+str(i)) for i in range(1,9)]]
    W=np.eye(4);W[:3,:3]=Rotation.from_quat(np.asarray(state['base_quat'])[[1,2,3,0]]).as_matrix();W[:3,3]=state['base_xyz'];B=fk(q);G=W@B
    bottle_world=bottle@bottle_R.T+np.asarray(state['object_xyz'][9]);v=(bottle_world-G[:3,3])@G[:3,:3];H=ConvexHull(v).equations
    slab=v[(v[:,2]>=.0593)&(v[:,2]<=.1358)]
    item={'run_id':run_id,'steps':r['steps'],'stop_reason':r['stop_reason'],'actual_arm_q':q.tolist(),'actual_world_gripper_from_B_FK':G.tolist(),'FK_position_vs_record_m':float(np.linalg.norm(G[:3,3]-state['gripper_xyz'])),'object_root_world_m':state['object_xyz'][9],
      'nominal_bottle_world_bounds_m':[bottle_world.min(0).tolist(),bottle_world.max(0).tolist()],
      'axis_points_world_m':{str(d):(G[:3,3]+d*G[:3,2]).tolist() for d in [.0593,.115,.1358]},
      'bottle_vertices_in_finger_depth_slab':{'count':len(slab),'bounds_gripper_m':[slab.min(0).tolist(),slab.max(0).tolist()]},
      'ideal_convex_hull_contact_possible':{},'paired_joint_lifts':[],
      'source_sha256':{n:digest(run/n) for n in ['result.json','telemetry.npz','environment_metadata.json']}}
    for link,sign in [('link7',-1),('link8',1)]:
        F=ConvexHull(fingers[link]).equations;tests=[]
        for opening in [.07,.065,.06,.055,.05,.045,.04]:
            J=F.copy();J[:,3]-=J[:,:3]@np.array([0,sign*opening/2,0]);E=np.concatenate([H,J]);fit=linprog(np.zeros(3),A_ub=E[:,:3],b_ub=-E[:,3],bounds=[(None,None)]*3,method='highs')
            tests.append({'symmetric_opening_m':opening,'intersects':bool(fit.success),'witness_gripper_m':fit.x.tolist() if fit.success else None})
        item['ideal_convex_hull_contact_possible'][link]=tests
    for delta in [.05,.10,.12]:
        qq=q.copy();qq[1]-=delta;qq[2]+=delta;N=fk(qq);motion=W[:3,:3]@(N[:3,3]-B[:3,3]);angle=Rotation.from_matrix(N[:3,:3]@B[:3,:3].T).magnitude()
        item['paired_joint_lifts'].append({'q2_delta_rad':-delta,'q3_delta_rad':delta,'predicted_world_translation_m':motion.tolist(),'orientation_change_rad':float(angle)})
    report['runs'].append(item)
report['recommendation']='One bounded physical probe is justified: normal first-reach completion, keep lowered legs/wheel anchor, close gently with finite preload, then measured q2-.10/q3+.10 and observe. Do not infer grasp from proximity reward or static geometry.'
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(OUT)
for r in report['runs']:
    print(r['run_id'],r['stop_reason'],'tip_z',r['axis_points_world_m']['0.1358'][2],'nominal_top',r['nominal_bottle_world_bounds_m'][1][2],'lift_10',r['paired_joint_lifts'][1]['predicted_world_translation_m'])
