import contextlib,io,runpy,json,numpy as np
with contextlib.redirect_stdout(io.StringIO()): d=runpy.run_path('/home/lybm/ATEC_Experiments_20260910/task_b_astra_review_20260914/contact_force75_wrist_depth.py')
f=d['geometry'];out=[]
for raise_m in [.002,.003,.005,.01]:
    for angle in [30,35,40]:
        p=[f(a,-raise_m,contact=True) for a in np.linspace(0,angle,41)]
        maxh=max(v['contact_half_width_m'] for x in p for v in x['first_contact'].values())
        lower=[f(angle,z,contact=True) for z in np.linspace(-raise_m,.01,41)]
        out.append({'raise_m':raise_m,'angle':angle,'rotate_hits':sum(any(x['finger_hull_intersections'].values()) for x in p),'rotate_worst_open_intrusion_m':maxh-.035,'lower_hits':sum(any(x['finger_hull_intersections'].values()) for x in lower),'rotation_gripper_motion_max_m':max(float(np.linalg.norm(np.array(x['gripper_world_xyz'])-np.array(p[0]['gripper_world_xyz']))) for x in p)})
# Only tight normal contact summary for each useful endpoint.
c=[]
for z in [.01,.015,.02]:
 for a in [30,35,40,45,50,60,75,90]:
  r=f(a,z,contact=True)
  c.append({'extra':z,'angle':a,'hits':r['finger_hull_intersections'],'half_widths':[v['contact_half_width_m'] for v in r['first_contact'].values()],'normal_angles':[v['normal_up_angles_deg'] for v in r['first_contact'].values()]})
output={'paths':out,'contacts':c};open('/home/lybm/ATEC_Experiments_20260910/task_b_astra_review_20260914/contact_rotation_clearance_followup.json','w').write(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
