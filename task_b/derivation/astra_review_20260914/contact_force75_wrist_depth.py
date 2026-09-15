"""Offline actual-quaternion geometry; never policy inputs."""
import contextlib, io, json
from pathlib import Path
import numpy as np
src=Path(__file__).with_name('contact_lowering_geometry.py').read_text().replace('plan_p4_grasp_feedback_seed42_01','plan_p4_grasp_force75_seed42_01')
src=src.replace('total_reference_lowering_m=.02+extra','total_reference_lowering_m=.03+extra')
ns={'__file__':str(Path(__file__).with_name('force75_lowering_baseline_loader.py'))}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src,'contact_geometry_loader','exec'),ns)
globals().update({k:v for k,v in ns.items() if k!='__file__'})

def geometry(deg,extra,width=.07,contact=False):
    qs={c:q0[c]+np.r_[np.asarray(delta[c])*(extra/.02),0.] for c in corners}
    R,b,residual=align(centers(qs),old_centers)
    q=qa.copy();q[5]+=np.deg2rad(deg);A=arm_fk(q)
    G=R@A[:3,:3];g=b+R@A[:3,3]
    rel=(worldv-g)@G;H=ConvexHull(rel).equations
    hits={}; details={}
    for link,sign in [('link7',-1),('link8',1)]:
        F=finger_hulls[link].copy();F[:,3]-=F[:,1]*sign*(width/2)
        E=np.concatenate([H,F]);p=linprog(np.zeros(3),A_ub=E[:,:3],b_ub=-E[:,3],bounds=[(None,None)]*3,method='highs')
        hits[link]=bool(p.success)
        if contact:
            F0=finger_hulls[link]
            AH=np.c_[H[:,:3],np.zeros(len(H))];AF=np.c_[F0[:,:3],-F0[:,1]*sign]
            p=linprog([0,0,0,-1],A_ub=np.r_[AH,AF],b_ub=-np.r_[H[:,3],F0[:,3]],bounds=[(None,None)]*3+[(0,.05)],method='highs')
            if p.success:
                slack=-(H[:,:3]@p.x[:3]+H[:,3]);active=H[slack<1e-6,:3];worldnormals=active@G.T
                details[link]={'contact_half_width_m':float(p.x[3]),'witness_gripper_xyz':p.x[:3].tolist(),'object_active_normals_world':worldnormals.tolist(),'normal_up_angles_deg':np.rad2deg(np.arcsin(np.clip(worldnormals[:,2],-1,1))).tolist()}
    return {'q6_relative_deg':float(deg),'extra_lower_m':float(extra),'open_width_m':float(width),'finger_hull_intersections':hits,'base_world_xyz':b.tolist(),'gripper_world_xyz':g.tolist(),'first_contact':details}

cases=[]
for extra in [0,.005,.01,.015,.02]:
    for deg in [-30,-20,-10,0,10,20,25,30,35,40,45,50,60,75,90]:
        cases.append(geometry(deg,extra,contact=True))
paths={}
for deg in [20,25,30,35,40]:
    rotate=[geometry(a,0) for a in np.linspace(0,deg,31)]
    lower=[geometry(deg,z) for z in np.linspace(0,.02,41)]
    paths[str(deg)]={'rotation_collisions':[x for x in rotate if any(x['finger_hull_intersections'].values())], 'descent_collisions':[x for x in lower if any(x['finger_hull_intersections'].values())]}
out={'run_id':run.name,'baseline_step':int(t['step'][i]),'actual_baseline_arm_q':qa.tolist(),'actual_baseline_base_xyz':base.tolist(),'actual_baseline_object_xyz':t['object_xyz'][i,9].tolist(),'actual_baseline_object_quat_wxyz':t['object_quat'][i,9].tolist(),'cases':cases,'paths':paths,'limitations':['Offline GT and exact mesh used for parameter feasibility diagnosis only; none enters the policy.','Convex hull intersection at zero contact offsets is conservative for original vertices but is not a cooked contact model; 10mm object contact_offset is not included.','Fixed wheel-center rigid fit predicts leg lowering geometry, not actual dynamic response.','First-contact outward normal is geometry only; no measured force or successful grasp is implied.']}
output=Path('/home/lybm/ATEC_Experiments_20260910/task_b_astra_review_20260914/contact_force75_wrist_depth.json');output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'run':run.name,'step':out['baseline_step'],'zero':geometry(0,0,contact=True),'candidates':[x for x in cases if x['q6_relative_deg'] in [20,25,30,35,40] and x['extra_lower_m'] in [.01,.02]],'path_counts':{k:{j:len(v) for j,v in p.items()} for k,p in paths.items()}},indent=2))
