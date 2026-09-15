"""Static CPU diagnostic only. No simulator, actuator changes, or production goals from object truth."""
from pathlib import Path
import itertools
import json
import sys
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

REPO = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(REPO))
from task_b.arm_kinematics import fk, solve_ik
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

ASSETS = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model')
RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_p4_grasp_feedback_seed42_01')
OUT = Path(__file__).with_name('payload_raise_geometry.json')


def quat(value):
    if hasattr(value, 'GetReal'):
        return Rotation.from_quat([*value.GetImaginary(), value.GetReal()]).as_matrix()
    value = np.asarray(value)
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def transform(position, rotation):
    out = np.eye(4)
    out[:3,:3], out[:3,3] = rotation, position
    return out


def apply(pose, points):
    return np.asarray(points)@pose[:3,:3].T+pose[:3,3]


robot_file = ASSETS/'robot/b2w/b2w_piper.usda'
robot_stage = Usd.Stage.Open(str(robot_file))
joint_rows, masses = [], {}
for prim in robot_stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            continue
        name = prim.GetName()
        axis_attr = prim.GetAttribute('physics:axis')
        axis = axis_attr.Get() if axis_attr else None
        joint_rows.append(dict(name=name, parent=parents[0].name, child=children[0].name,
                               kind=prim.GetTypeName(), axis=axis,
                               parent_joint=transform(joint.GetLocalPos0Attr().Get(), quat(joint.GetLocalRot0Attr().Get())),
                               child_joint=transform(joint.GetLocalPos1Attr().Get(), quat(joint.GetLocalRot1Attr().Get()))))
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass = UsdPhysics.MassAPI(prim)
        value, center = mass.GetMassAttr().Get(), mass.GetCenterOfMassAttr().Get()
        if value is not None and center is not None and np.isfinite(center).all():
            masses[prim.GetName()] = (float(value), np.asarray(center,dtype=float))


def chain(qmap):
    poses, joints = {'base_link': np.eye(4)}, {}
    pending = list(joint_rows)
    while pending:
        previous = len(pending)
        for row in list(pending):
            if row['parent'] not in poses:
                continue
            origin = poses[row['parent']]@row['parent_joint']
            motion = np.eye(4)
            if row['axis']:
                axis = np.eye(3)['XYZ'.index(row['axis'])]
                value = qmap.get(row['name'], 0.)
                if row['kind'] == 'PhysicsRevoluteJoint':
                    motion[:3,:3] = Rotation.from_rotvec(axis*value).as_matrix()
                elif row['kind'] == 'PhysicsPrismaticJoint':
                    motion[:3,3] = axis*value
                joints[row['name']] = (origin[:3,3], origin[:3,:3]@axis)
            poses[row['child']] = origin@motion@np.linalg.inv(row['child_joint'])
            pending.remove(row)
        if len(pending) == previous:
            raise RuntimeError('Unresolved rigid-body tree')
    return poses, joints


telemetry = np.load(RUN/'telemetry.npz')
traces = [json.loads(s) for s in (RUN/'trace.jsonl').read_text().splitlines()]
index = next(i for i,r in enumerate(traces) if r['policy_state'] == 'PROBE_LIFT')
names = telemetry['joint_names'].tolist()
qmap = dict(zip(names, telemetry['q'][index].astype(float)))
q0 = np.array([qmap['arm_joint'+str(i)] for i in range(1,7)])
base_world = transform(telemetry['base_xyz'][index], quat(telemetry['base_quat'][index]))
gripper_world = transform(telemetry['gripper_xyz'][index], quat(telemetry['gripper_quat'][index]))
object_world = transform(telemetry['object_xyz'][index,9], quat(telemetry['object_quat'][index,9]))
object_file = ASSETS/'objects/task_b/006_mustard_bottle.usd'
object_stage = Usd.Stage.Open(str(object_file))
mesh = next(UsdGeom.Mesh(prim) for prim in object_stage.Traverse() if prim.IsA(UsdGeom.Mesh))
mesh_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
mesh_scale = np.asarray(UsdGeom.Xformable(mesh.GetPrim()).GetLocalTransformation()).T
mesh_points = apply(mesh_scale, mesh_points)
held_points = apply(np.linalg.inv(gripper_world)@object_world, mesh_points)
held_root = apply(np.linalg.inv(gripper_world), object_world[:3,3])
initial_fk = fk(q0)
tree_initial, _ = chain(qmap)
gravity_body = base_world[:3,:3].T@np.array([0.,0.,-9.81])


def evaluate(q):
    mapping = dict(qmap)
    mapping.update({'arm_joint'+str(i+1): value for i,value in enumerate(q)})
    poses, joints = chain(mapping)
    body_mesh = apply(fk(q), held_points)
    world_mesh = apply(base_world, body_mesh)
    torque = np.zeros(6)
    upper = np.zeros(6)
    cone_upper=np.zeros(6)
    for i in range(6):
        origin, axis = joints['arm_joint'+str(i+1)]
        arm_moment_vector=np.zeros(3)
        for name,(mass,com) in masses.items():
            if name.startswith('arm_link') and int(name[len('arm_link'):]) >= i+1 or name == 'gripper_base':
                center = apply(poses[name], com)
                torque[i] += np.dot(np.cross(center-origin, mass*gravity_body), axis)
                arm_moment_vector+=mass*np.cross(axis,center-origin)
        # Unknown physical COM lies inside the complete source mesh; extrema
        # bound every possible density-supported payload COM for 0.5 kg.
        moments = np.cross(body_mesh-origin, .5*gravity_body)@axis
        upper[i] = abs(torque[i])+float(np.max(np.abs(moments)))
        vectors=arm_moment_vector+.5*np.cross(axis,body_mesh-origin)
        horizontal=np.linalg.norm(vectors[:,:2],axis=1)
        vertical=np.abs(vectors[:,2])
        angle=np.minimum(np.arctan2(horizontal,vertical),.12)
        cone_upper[i]=float(np.max(9.81*(vertical*np.cos(angle)+horizontal*np.sin(angle))))
    return {'gripper_body': fk(q)[:3,3].tolist(), 'payload_lowest_world_z': float(world_mesh[:,2].min()),
            'payload_root_body': apply(fk(q),held_root).tolist(),
            'payload_highest_world_z': float(world_mesh[:,2].max()),
            'payload_body_aabb_min': body_mesh.min(axis=0).tolist(),
            'payload_body_aabb_max': body_mesh.max(axis=0).tolist(),
            'arm_only_gravity_torque_nm': torque.tolist(), 'gravity_plus_payload_absolute_upper_nm': upper.tolist(),
            'gravity_plus_payload_tilt_cone_012_upper_nm':cone_upper.tolist(),
            'static_position_error_required_at_K80_rad': (upper/80.).tolist()}


# Use the complete original collision geometry, including instance proxies.
cache = UsdGeom.XformCache()
collision_points = {}
for collision in robot_stage.Traverse():
    if not collision.HasAPI(UsdPhysics.CollisionAPI):
        continue
    body = collision.GetParent()
    pieces = []
    for prim in Usd.PrimRange(collision, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            points = np.array(list(itertools.product((-half,half),repeat=3)))
        else:
            extent = np.asarray(UsdGeom.Boundable(prim).GetExtentAttr().Get(),dtype=float)
            if extent.shape != (2,3) or not np.isfinite(extent).all():
                raise RuntimeError('Missing explicit shape extent: '+str(prim.GetPath()))
            points = np.array(list(itertools.product(*zip(extent[0],extent[1]))))
        relative = np.asarray(cache.ComputeRelativeTransform(prim,body)[0]).T
        pieces.append(apply(relative,points))
    if pieces:
        collision_points[body.GetName()] = np.concatenate(pieces)

body_collision = {name: apply(tree_initial[name],points) for name,points in collision_points.items()
                  if not name.startswith('arm_') and name!='gripper_base'}
body_union = np.concatenate(list(body_collision.values()))
heading=base_world[:2,0].copy();heading/=np.linalg.norm(heading)
world_left=np.r_[-heading[1],heading[0],0.]
world_nonarm_offsets=body_union@base_world[:3,:3].T
component_bounds = {name:{'min_body':p.min(axis=0).tolist(),'max_body':p.max(axis=0).tolist()}
                    for name,p in body_collision.items()}

direction = initial_fk[:2,3]-np.array([.2,0.])
direction /= np.linalg.norm(direction)
cases = []
for radius,z,pitch in itertools.product((.35,.40,.45,.50),(.35,.40,.45,.50),(0., -.20, .20)):
    position = np.r_[np.array([.2,0.])+radius*direction,z]
    axis = np.r_[-direction[1],direction[0],0.]
    rotation = Rotation.from_rotvec(axis*pitch).as_matrix()@initial_fk[:3,:3]
    fit = solve_ik(position,rotation=rotation,seed=q0,max_nfev=90)
    case = {'radius_from_shoulder_xy':radius,'desired_body_z':z,'tilt_change_rad':pitch,
            'success':bool(fit.success),'position_error_m':fit.position_error,'orientation_error_rad':fit.orientation_error,
            'q_rad':fit.joints.tolist()}
    if fit.success:
        case.update(evaluate(fit.joints))
    cases.append(case)

side_cases = []
for side,radius,z,pitch in itertools.product((-1.,1.),(.40,.45,.50,.52,.54,.56),(.40,.42),(0.,-.20,-.30,-.40)):
    angle = side*np.pi/2-q0[0]
    rz = Rotation.from_rotvec([0.,0.,angle]).as_matrix()
    rotation = Rotation.from_rotvec(np.array([-side,0.,0.])*pitch).as_matrix()@rz@initial_fk[:3,:3]
    position = np.array([.2,side*radius,z])
    seed=q0.copy();seed[0]=side*np.pi/2
    fit=solve_ik(position,rotation=rotation,seed=seed,max_nfev=90)
    case={'side':side,'desired_side_extension_m':radius,'desired_body_z':z,'tilt_change_rad':pitch,
          'success':bool(fit.success),'position_error_m':fit.position_error,'orientation_error_rad':fit.orientation_error,'q_rad':fit.joints.tolist()}
    if fit.success:
        case.update(evaluate(fit.joints))
        bodymesh=apply(fk(fit.joints),held_points)
        # A convex hull contains the original collision mesh. Its AABB near
        # side is a conservative radial-support bound, not contact simulation.
        envelope=float(np.max(side*(world_nonarm_offsets@world_left)))
        minimum_base_center_distance=1.+envelope+.03
        world_payload_offsets=bodymesh@base_world[:3,:3].T
        world_root_offset=np.array(case['payload_root_body'])@base_world[:3,:3].T
        payload_radius=np.linalg.norm(world_payload_offsets[:,:2]-side*minimum_base_center_distance*world_left[:2],axis=1)
        case.update(non_arm_near_side_support_m=envelope,
                    sufficient_axis_aligned_chassis_center_distance_m=minimum_base_center_distance,
                    payload_radial_range_at_that_center_m=[float(payload_radius.min()),float(payload_radius.max())],
                    payload_all_inside_inner_radius_098=bool(payload_radius.max()<.98))
        case['dock_center_distance_cases'] = []
        for center_distance in (1.35,1.4,1.42,1.45,1.48):
            center=side*center_distance*world_left[:2]
            root_radius=float(np.linalg.norm(world_root_offset[:2]-center))
            radial=np.linalg.norm(world_payload_offsets[:,:2]-center,axis=1)
            # The AABB vertices in this union overbound capsules/cubes, and the
            # vertex radius minimum alone cannot certify each connecting face.
            # An additional separating support plane gives a sufficient bound.
            plane_gap=center_distance-envelope-1.
            case['dock_center_distance_cases'].append({'center_distance_m':center_distance,
                'payload_root_radius_m':root_radius,'all_payload_vertices_radius_max_m':float(radial.max()),
                'sufficient_chassis_side_support_plane_gap_m':plane_gap})
    side_cases.append(case)

raise_case=next(c for c in cases if c['success'] and c['radius_from_shoulder_xy']==.35
                and c['desired_body_z']==.45 and c['tilt_change_rad']==-.20)
drop_case=next(c for c in side_cases if c['success'] and c['side']==1. and c['desired_side_extension_m']==.52
               and c['desired_body_z']==.40 and c['tilt_change_rad']==-.40)
raise_q=np.asarray(raise_case['q_rad'])
side_q=raise_q.copy();side_q[0]=np.pi/2
drop_q=np.asarray(drop_case['q_rad'])
waypoints=[('raise_before_swing',q0,raise_q),('swing_only_after_raising',raise_q,side_q),
           ('extend_at_height',side_q,drop_q)]
path_report=[]
for label,start_q,end_q in waypoints:
    duration=float(np.max(np.abs(end_q-start_q))/.10)
    summaries=[]
    for fraction in np.linspace(0.,1.,101):
        q=start_q+fraction*(end_q-start_q)
        metrics=evaluate(q)
        bodymesh=apply(fk(q),held_points)
        lower,upper=bodymesh.min(axis=0),bodymesh.max(axis=0)
        overlaps=[]
        for name,bounds in component_bounds.items():
            if np.all(upper>=bounds['min_body'])and np.all(lower<=bounds['max_body']):
                overlaps.append(name)
        summaries.append({'fraction':float(fraction),'q':q.tolist(),
                          'payload_lowest_z':metrics['payload_lowest_world_z'],
                          'gravity_upper_nm':metrics['gravity_plus_payload_absolute_upper_nm'],
                          'tilt_cone_gravity_upper_nm':metrics['gravity_plus_payload_tilt_cone_012_upper_nm'],
                          'payload_nonarm_aabb_overlaps':overlaps})
    path_report.append({'phase':label,'q_start':start_q.tolist(),'q_goal':end_q.tolist(),
                        'nominal_scalar_interpolated_duration_at_max_axis_01_rad_s':duration,
                        'sample_count':len(summaries),
                        'payload_minimum_world_z_over_path':min(s['payload_lowest_z']for s in summaries),
                        'payload_final_world_z':summaries[-1]['payload_lowest_z'],
                        'minimum_payload_z_change_per_sample':float(np.min(np.diff([s['payload_lowest_z']for s in summaries]))),
                        'gravity_upper_max_per_axis_nm':np.max([s['gravity_upper_nm']for s in summaries],axis=0).tolist(),
                        'tilt_cone_gravity_upper_max_per_axis_nm':np.max([s['tilt_cone_gravity_upper_nm']for s in summaries],axis=0).tolist(),
                        'sampled_payload_nonarm_aabb_overlaps':sorted({n for s in summaries for n in s['payload_nonarm_aabb_overlaps']}),
                        'every_tenth_sample':summaries[::10]})

finalmap=dict(qmap);finalmap.update({'arm_joint'+str(i+1):value for i,value in enumerate(drop_q)})
finalposes,_=chain(finalmap)
crossing_components=[]
for name,points in collision_points.items():
    if not name.startswith('arm_')and name!='gripper_base':
        continue
    bodypoints=apply(finalposes[name],points)
    worldpoints=apply(base_world,bodypoints)
    signed_side=(worldpoints-base_world[:3,3])@world_left
    if signed_side.max()>=1.45-1.:
        crossing_components.append({'name':name,'side_projection_max_m':float(signed_side.max()),
                                    'entire_component_minimum_world_z_m':float(worldpoints[:,2].min()),
                                    'entire_component_above_wall_055':bool(worldpoints[:,2].min()>.55)})

result={'scope':__doc__,'run':RUN.name,'reference_step':int(telemetry['step'][index]),
        'attachment_assumption':'The entire mustard mesh is hypothetically rigidly attached at the recorded closure-relative pose; this run did NOT prove a secure grasp.',
        'payload_mass_kg':.5,'mesh_vertices':len(mesh_points),'initial_q':q0.tolist(),
        'initial_geometry':evaluate(q0),'body_world_pose':base_world.tolist(),
        'fk_vs_usd_joint_tree_gripper_max_error':float(np.max(np.abs(initial_fk-tree_initial['gripper_base']))),
        'fk_world_vs_actual_gripper_position_error_m':float(np.linalg.norm(apply(base_world,initial_fk[:3,3])-gripper_world[:3,3])),
        'original_mesh_world_lowest_z':float(apply(object_world,mesh_points)[:,2].min()),
        'original_masses_kg':{name:mass for name,(mass,com) in masses.items()if name.startswith('arm_')or name=='gripper_base'},
        'non_arm_collision_union_body_aabb':[body_union.min(axis=0).tolist(),body_union.max(axis=0).tolist()],
        'non_arm_collision_components':component_bounds,'front_cases':cases,'side_cases':side_cases,
        'selected_side_dock':drop_case,'selected_raise_waypoint':raise_case,
        'selected_scalar_interpolation_path':path_report,
        'selected_side_dock_d145_arm_components_crossing_outer_support_plane':crossing_components,
        'bounded_command_contract_proposal':{
            'status':'planning recommendation only; not implemented or validated dynamically',
            'fixed_measured_goal_acceptance_rad':.04,'command_rate_each_axis_rad_s':.10,
            'q2':{'actual_tether_rad':.18,'position_feedback_gain':4.,'feedback_cap_rad':.14,'filter_tau_s':.10},
            'q3':{'actual_tether_rad':.10,'position_feedback_gain':2.,'feedback_cap_rad':.08,'filter_tau_s':.10},
            'other_axes_actual_tether_rad':.10,
            'command_bounds':'intersection of previous command +/- rate*dt, measured q +/- axis tether, and unchanged real hard joint limits; empty intersection stops without a jump',
            'trajectory':'one monotone shared interpolation parameter per joint-space segment; may pause for tracking, never jump to the next waypoint from command arrival alone',
            'stage_deadline':'nominal max-axis travel/.10 plus finite measured-settle allowance; retains real joint progress checks and hard posture budgets',
            'filter_behavior':'zero at a new measured waypoint reference; freeze both command and correction history during public-motion pauses',
            'effort_configuration':'original stiffness80/damping4/100Nm effort limit unchanged; the tether limits reference error, not physical torque capacity',
            'grasp_prerequisite':'real independent payload-following evidence before high raise or navigation; a width window alone is insufficient'},
        'limitations':['joint limits and static payload clearance only; no contact, balance, motion tracking or secure-grasp certificate',
                       'fixed body pose from one completed local seed; live production must use public measured q and bounded paths, never these GT coordinates',
                       'torque upper includes arm gravity and any 0.5 kg COM within the full source mesh; excludes acceleration, transient contact and unknown friction',
                       'non-arm collision envelope uses current measured leg geometry; leg motion and chassis rocking need added clearance'],
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in (robot_file,object_file,REPO/'task_b/arm_kinematics.py')}}
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(OUT),'initial':result['initial_geometry'],'collision_envelope':result['non_arm_collision_union_body_aabb'],
                  'front_feasible':sum(x['success']for x in cases),'side_feasible':sum(x['success']for x in side_cases),
                  'side_pass_mesh_inner_and_height':[x for x in side_cases if x['success']and x['payload_all_inside_inner_radius_098']and x['payload_lowest_world_z']>.58]},indent=2))
