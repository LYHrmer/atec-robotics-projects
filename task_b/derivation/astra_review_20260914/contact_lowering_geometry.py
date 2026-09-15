"""CPU-only candidate geometry; every world quantity is offline diagnosis."""
import itertools, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from pxr import Usd, UsdGeom
from task_b.arm_kinematics import fk as arm_fk

root = Path('/home/lybm/ATEC_Experiments_20260910')
run = root/'task_b_score/plan_p4_grasp_feedback_seed42_01'
t = np.load(run/'telemetry.npz')
rows = [json.loads(x) for x in (run/'trace.jsonl').open()]
i = next(i for i,x in enumerate(rows) if x.get('policy_debug',{}).get('probe_phase') == 'PROBE_CLOSE')
names = t['joint_names'].tolist()
geo = json.loads((root/'task_b_score_cpu/stance_geometry.json').read_text())
frames, boxes = geo['joint_frames'], geo['local_visual_bounds']
corners = ['FR','FL','RR','RL']
delta = dict(FR=[-.01361,.02854,-.05819], FL=[.01298,.02856,-.05866],
             RR=[-.00992,.03810,-.06400], RL=[.00944,.03883,-.06426])
q0 = {c: np.array([t['q'][i,names.index(c+'_'+l+'_joint')] for l in ['hip','thigh','calf','foot']]) for c in corners}
base = t['base_xyz'][i]
R0 = Rotation.from_quat(t['base_quat'][i][[1,2,3,0]]).as_matrix()
qa = np.array([t['q'][i,names.index('arm_joint'+str(j))] for j in range(1,9)])
Ga = arm_fk(qa)

def transforms(c, q):
    T = np.eye(4); out = {}
    for link,v in zip(['hip','thigh','calf','foot'],q):
        f = frames[c+'_'+link+'_joint']; A = np.asarray(f['parent_frame']); B = np.asarray(f['child_frame'])
        M = np.eye(4); M[:3,:3] = Rotation.from_euler(f['axis'].lower(),v).as_matrix()
        T = T@A@M@np.linalg.inv(B); out[link] = T.copy()
    return out

def centers(qs):
    out = []
    for c in corners:
        T = transforms(c,qs[c])['foot']; center = np.mean(boxes[c+'_foot'],axis=0)
        out.append(T[:3,:3]@center+T[:3,3])
    return np.array(out)

old_centers = centers(q0)@R0.T+base
def align(points, world):
    a,b = points.mean(0),world.mean(0)
    U,_,V = np.linalg.svd((points-a).T@(world-b)); S=np.eye(3);S[-1,-1]=np.linalg.det(V.T@U.T)
    R=V.T@S@U.T; translation=b-R@a
    residual=np.linalg.norm(points@R.T+translation-world,axis=1)
    return R,translation,residual

asset = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model')
s=Usd.Stage.Open(str(asset/'objects/task_b/006_mustard_bottle.usd'));cache=UsdGeom.XformCache();verts=[]
for prim in s.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        v=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());M=np.asarray(cache.GetLocalToWorldTransform(prim)).T
        verts.append(v@M[:3,:3].T+M[:3,3])
v=np.concatenate(verts);OR=Rotation.from_quat(t['object_quat'][i,9][[1,2,3,0]]).as_matrix()
worldv=v@OR.T+t['object_xyz'][i,9]
s=Usd.Stage.Open(str(asset/'robot/piper/piper.usd'));cache=UsdGeom.XformCache()
Tg=np.asarray(cache.GetLocalToWorldTransform(s.GetPrimAtPath('/piper/gripper_base'))).T
finger_hulls={}
for link in ['link7','link8']:
    prim=s.GetPrimAtPath('/piper/'+link+'/collisions/'+link+'/mesh')
    vv=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());M=np.asarray(cache.GetLocalToWorldTransform(prim)).T
    vv=(vv@M[:3,:3].T+M[:3,3]-Tg[:3,3])@Tg[:3,:3]
    finger_hulls[link]=ConvexHull(vv).equations
s=Usd.Stage.Open(str(asset/'robot/b2w/b2w_piper.usda'));bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
base_boxes=[]
for prim in s.Traverse():
    if str(prim.GetPath()) == '/b2w_description/base_link':
        b=bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
        base_boxes.append((str(prim.GetPath()),np.array(list(itertools.product(*zip(b.GetMin(),b.GetMax()))))))

cases=[]
for extra in [0.,.01,.02,.03]:
    qs={c:q0[c]+np.r_[np.array(delta[c])*(extra/.02),0.] for c in corners}
    R,b,residual=align(centers(qs),old_centers)
    G=R@Ga[:3,:3];g=b+R@Ga[:3,3];tip=g+.1358*G[:,2]
    relative=(worldv-g)@G;slab=relative[(relative[:,2]>=.0593)&(relative[:,2]<=.1358)]
    finger_width_slab=slab[(slab[:,0]>=-.028)&(slab[:,0]<=.028)]
    H=ConvexHull(relative).equations;open_intersection={}
    for link,sign in [('link7',-1),('link8',1)]:
        F=finger_hulls[link].copy();F[:,3]-=F[:,:3]@np.array([0.,sign*.035,0.])
        E=np.concatenate([H,F]);fit=linprog(np.zeros(3),A_ub=E[:,:3],b_ub=-E[:,3],bounds=[(None,None)]*3,method='highs')
        open_intersection[link]=bool(fit.success)
    bounds=np.array([[-.87,.87],[-.94,4.69],[-2.82,-.43]])
    qmargin=min(float(np.min(np.minimum(qs[c][:3]-bounds[:,0],bounds[:,1]-qs[c][:3]))) for c in corners)
    thighz=[]
    for c in corners:
        T=transforms(c,qs[c])['thigh'];lo,hi=boxes[c+'_thigh'];vv=np.array(list(itertools.product(*zip(lo,hi))))
        thighz.append(float(((vv@T[:3,:3].T+T[:3,3])@R.T+b)[:,2].min()))
    cases.append(dict(extra_reference_lowering_m=extra,total_reference_lowering_m=.02+extra,
        predicted_base_xyz=b.tolist(),predicted_base_descent_m=float(base[2]-b[2]),
        predicted_tilt_rad=float(np.arccos(np.clip(R[2,2],-1,1))),
        wheel_center_rigid_fit_max_error_m=float(residual.max()),minimum_leg_hard_limit_margin_rad=qmargin,
        minimum_thigh_visual_box_world_z_m=min(thighz),
        base_visual_box_min_z_m={name:float((vv@R.T+b)[:,2].min()) for name,vv in base_boxes},
        gripper_world=g.tolist(),finger_axis_tip_world=tip.tolist(),
        fixed_object_vertices_in_contact_depth=len(slab),
        fixed_object_contact_depth_xyz_bounds=None if not len(slab) else [slab.min(0).tolist(),slab.max(0).tolist()],
        object_y_bounds_within_finger_x_extent=None if not len(finger_width_slab) else [float(finger_width_slab[:,1].min()),float(finger_width_slab[:,1].max())],
        ideal_open_70mm_finger_hull_intersects_fixed_object=open_intersection,
        nominal_leg_q={c:qs[c].tolist() for c in corners}))

out=dict(run_id=run.name,baseline_step=int(t['step'][i]),baseline_base_xyz=base.tolist(),
    original_first_reach_lowering_parameter_range_m=[0,.03],
    original_fall_threshold_base_z_m=0.,original_illegal_contact_threshold_N=1.,
    cases=cases,limits=['Static wheel-center fit of public measured leg q plus fixed reference delta; no dynamic stability or collision-free guarantee.',
    'World/base/object poses are used only for this offline candidate review; they must not enter policy.',
    'Visual boxes are conservative geometric diagnostics, not cooked collision shapes.',
    'Contact-depth vertex overlap does not certify bilateral contact, grasp or friction.'])
output=Path(__file__).with_suffix('.json');output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
