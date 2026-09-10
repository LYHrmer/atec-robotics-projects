"""CPU audit of original-camera rays, optional head handoff and EE parity.

No simulation App or GPU. The optional baseline file is a pre-head version of
visual_approach.py and is imported only for frozen-input behavior comparison.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.arm_kinematics import ee_camera_transform, head_camera_transform
from task_b.audit_stability import fixture
from task_b.visual_approach import VisualApproachPolicy, detect_yellow_candidates


def load_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def camera_config_audit(source_root, lab_root):
    # Read actual installed B2 camera config expressions, without importing App.
    asset_path=source_root/'source/atec_rl_lab/atec_rl_lab/assets/robots/b2.py'
    tree=ast.parse(asset_path.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='UNITREE_B2_CFG')
    offset=next(k.value for k in assignment.value.keywords if k.arg=='head_camera_offset')
    position=ast.literal_eval(next(k.value for k in offset.keywords if k.arg=='pos'))
    quaternion_node=next(k.value for k in offset.keywords if k.arg=='rot')
    quaternion=eval(compile(ast.Expression(quaternion_node),str(asset_path),'eval'),{'np':np,'R':Rotation})
    assert ast.literal_eval(next(k.value for k in offset.keywords if k.arg=='convention'))=='world'
    math_path=lab_root/'source/isaaclab/isaaclab/utils/math.py'
    math_module=load_file('taskb_handoff_official_math',math_path)
    # This executes the camera-convention function used by the actual sensor.
    converted=math_module.convert_camera_frame_orientation_convention(
        torch.tensor([quaternion],dtype=torch.float32),origin='world',target='ros')
    official_rotation=math_module.matrix_from_quat(converted)[0].numpy()
    pose=head_camera_transform()
    rotation_error=float(np.max(np.abs(official_rotation-pose[:3,:3])))
    assert rotation_error<3e-7 and np.array_equal(pose[:3,3],position)
    np.testing.assert_allclose(pose[:3,2],[np.sqrt(3)/2,0,-.5],atol=1e-15)
    np.testing.assert_allclose(pose[:3,0],[0,-1,0],atol=1e-15)
    # Execute the real intrinsic-matrix builder against inert CPU metadata.
    camera_path=lab_root/'source/isaaclab/isaaclab/sensors/camera/camera.py'
    camera_tree=ast.parse(camera_path.read_text())
    cls=next(n for n in camera_tree.body if isinstance(n,ast.ClassDef) and n.name=='Camera')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_intrinsic_matrices')
    ast_module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    namespace={};exec(compile(ast.fix_missing_locations(ast_module),str(camera_path),'exec'),namespace)
    intrinsic={}
    for source,focal in [('ee',15.),('head',24.)]:
        prim=NS(GetFocalLengthAttr=lambda focal=focal:NS(Get=lambda:focal),GetHorizontalApertureAttr=lambda:NS(Get=lambda:20.955))
        camera=NS(_sensor_prims=[prim],image_shape=(480,640),_data=NS(intrinsic_matrices=np.zeros((1,3,3))))
        namespace['_update_intrinsic_matrices'](camera,[0])
        expected=np.array([[640*focal/20.955,0,320],[0,640*focal/20.955,240],[0,0,1.]])
        assert np.array_equal(camera._data.intrinsic_matrices[0],expected)
        intrinsic[source]=expected.tolist()
    return {'passed':True,'head_rotation_vs_actual_camera_convention_max_error':rotation_error,
            'head_position':list(position),'head_optical_forward_body':pose[:3,2].tolist(),
            'head_image_right_body':pose[:3,0].tolist(),'actual_sensor_intrinsic_builder':intrinsic,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [asset_path,math_path,camera_path]},
            'scope':'Actual local configuration and sensor math executed on CPU. Not a live-rendered calibration or camera-timestamp test.'}


def render_plane_object(point, pose, focal_length, *, visible=True):
    """Ray-cast a small vertical yellow rectangle at fixed body x, CPU only."""
    rgb=np.zeros((480,640,3),np.uint8);depth=np.zeros((480,640),np.float64)
    if not visible:
        return rgb,depth
    point=np.asarray(point);focal=640*focal_length/20.955
    corners=np.array([point+[0,y,z] for y,z in [(-.0275,-.09),(.0275,-.09),(.0275,.09),(-.0275,.09)]])
    camera=(corners-pose[:3,3])@pose[:3,:3]
    if np.any(camera[:,2]<=.1):
        return rgb,depth
    pixels=camera[:,:2]/camera[:,2,None]*focal+[320,240]
    mask=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(pixels).astype(np.int32),255)
    rows,cols=np.nonzero(mask)
    rays=np.column_stack(((cols-320)/focal,(rows-240)/focal,np.ones(len(rows))))@pose[:3,:3].T
    distance=(point[0]-pose[0,3])/rays[:,0]
    good=np.isfinite(distance)&(distance>.1)&(distance<8.)
    rows,cols,distance=rows[good],cols[good],distance[good]
    rgb[rows,cols]=[240,215,20];depth[rows,cols]=distance
    return rgb,depth


def scene(point=(2.2,0,-.4), *, ee=True, head=True):
    er,ed=render_plane_object(point,ee_camera_transform(np.zeros(6)),15.,visible=ee)
    hr,hd=render_plane_object(point,head_camera_transform(),24.,visible=head)
    return {'ee_rgb':er,'ee_depth':ed,'head_rgb':hr,'head_depth':hd}


def handoff_audit():
    s,n,d=fixture();defaults=dict(zip(n,d));observation=np.zeros(84);observation[11]=-1
    image=scene()
    detector=[]
    for source,focal,pose in [('ee',15.,ee_camera_transform(np.zeros(6))),('head',24.,head_camera_transform())]:
        candidates,details=detect_yellow_candidates(image[source+'_rgb'],image[source+'_depth'],pose,
                                                   projected_gravity=observation[9:12],focal_length=focal)
        assert candidates,(source,details)
        detector.append(np.array(candidates[0]['body_point']))
    disagreement=float(np.linalg.norm(detector[0]-detector[1]))
    assert disagreement<.12 # clipped silhouettes legitimately have different visible medians
    p=VisualApproachPolicy(s,n,defaults,settle_calls=0,ramp_calls=1,use_head=True)
    for _ in range(6):p.act(observation,scene(head=False))
    assert p.debug['source']=='ee' and p.debug['source_confirmed']
    for _ in range(4):p.act(observation,scene(head=False))
    first=p.act(observation,scene(ee=False))
    assert p.debug['source']=='head' and p.debug['source_changed'] and not p.debug['source_confirmed']
    assert np.array_equal(first,np.zeros(24,np.float32))
    for _ in range(4):p.act(observation,scene(ee=False))
    second=p.act(observation,scene(ee=False))
    assert p.debug['source']=='head' and p.debug['source_confirmed'] and np.any(second)
    states=[]
    for x in np.r_[np.linspace(2.2,.82,17),np.full(12,.82)]:
        for _ in range(5):out=p.act(observation,scene((x,0,-.35),ee=False))
        states.append({'x':float(x),'state':p.state,'source':p.debug.get('source'),'body':p.debug.get('target_body')})
    assert p.state=='AT_STANDOFF',states
    assert np.array_equal(out,np.zeros(24,np.float32))
    json.dumps(p.describe(),allow_nan=False)
    return {'passed':True,'camera_body_point_disagreement_m':disagreement,
            'handoff_first_new_camera_frame_stops':True,'second_new_camera_frame_confirms':True,
            'head_continues_to_082m_and_stops':True,'trace':states,
            'scope':'Synthetic observed rectangle and real fixed camera transforms; no physical approach claim.'}


def baseline_audit(path):
    old=load_file('taskb_handoff_baseline',path)
    s,n,d=fixture();defaults=dict(zip(n,d));a=old.VisualApproachPolicy(s,n,defaults);b=VisualApproachPolicy(s,n,defaults,use_head=False)
    observation=np.zeros(84);observation[11]=-1;max_error=0.;states_equal=True
    for i in range(500):
        point=(2.2+.1*np.sin(i*.01),.10*np.sin(i*.013),-.3)
        image=scene(point)
        if 250<=i<265:image={}
        if 350<=i<365:image=scene(point,ee=False)
        left=a.act(observation,image);right=b.act(observation,image)
        assert np.array_equal(left,right)
        max_error=max(max_error,float(np.max(np.abs(left-right))));states_equal &= a.state==b.state
        assert a.debug==b.debug
    assert states_equal
    return {'passed':True,'frames':500,'max_action_error':max_error,'actions_states_debug_identical':True,
            'baseline_file':str(path),'baseline_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path('/home/lybm/ATEC2026_Simulation_Challenge'))
    parser.add_argument('--isaaclab-root',type=Path,default=Path('/home/lybm/IsaacLab'))
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/task_b_visual_handoff_cpu_audit.json')
    args=parser.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    report={'status':'passed','camera_config_and_ray_mapping':camera_config_audit(args.source_root,args.isaaclab_root),
            'handoff':handoff_audit()}
    if args.baseline:report['ee_only_parity']=baseline_audit(args.baseline)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [ROOT/'task_b/arm_kinematics.py',ROOT/'task_b/visual_approach.py',Path(__file__)]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','handoff':{k:v for k,v in report['handoff'].items() if k!='trace'},
                      'ee_only_parity':report.get('ee_only_parity'),'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
