import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lybm/ATEC_Robotics_Projects_20260910')
from task_b.arm_kinematics import fk, ee_camera_transform
out=[]
for q2 in [2.983,3.03,3.08,3.13]:
    f=minimize_scalar(lambda q3:(fk([0,q2,q3,0,-.088419,0])[0,3]-.5)**2,bounds=(-2.5,-.5),method='bounded')
    q=np.array([0,q2,f.x,0,-.088419,0]); p=fk(q)[:3,3]
    rows=[]
    for pitch in [0.,.03,.06,.09,.12]:
        R=Rotation.from_euler('y',pitch).as_matrix()
        relz=(R@p)[2]
        rows.append(dict(pitch_rad=pitch,max_base_z_for_20cm_vertical_gap=.1406505+.20-relz,max_base_z_for_18cm_vertical_gap=.1406505+.18-relz,lowering_from_519_for_18cm=.519-(.1406505+.18-relz)))
    out.append(dict(q=q.tolist(),p=p.tolist(),rows=rows))
path=[]
q=np.array([0,3.13,-1.43467,0,-.088419,0])
for a in np.linspace(0,1,11):
    p=fk(a*q); c=ee_camera_transform(a*q)
    path.append(dict(alpha=float(a),q=(a*q).tolist(),gripper_body=p[:3,3].tolist(),finger_axis_body=p[:3,2].tolist(),camera_position_body=c[:3,3].tolist(),camera_view_axis_body=c[:3,2].tolist()))
result=dict(note='Static position kinematics and projection only; no simulation/collision/dynamic validity claimed. No diagnostic root/object state enters policy.',limits_and_pitch=out,path=path)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
