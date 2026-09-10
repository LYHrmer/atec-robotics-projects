"""CPU boundary and static-geometry audit of the Task B stability filter.

No Isaac simulation application or GPU is started. Optional --before-parity
checks a frozen synthetic sequence from the pre-boundary-fix controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.control import ActionSchema, ActionTerm, BootstrapPolicy
from task_b.stability import StabilityController, StabilityError


def fixture(*, reorder=False, scale=.5, real_defaults=True):
    legs = tuple(f'{c}_{link}_joint' for c in ('FR','FL','RR','RL') for link in ('hip','thigh','calf'))
    wheels = tuple(f'{c}_foot_joint' for c in ('FR','FL','RR','RL'))
    arms = tuple(f'arm_joint{i}' for i in range(1,9))
    names = legs+wheels+arms
    if real_defaults:
        defaults = {n:(.1 if n[1]=='L' else -.1) if '_hip_' in n else ((.8 if n[0]=='F' else 1.) if '_thigh_' in n else (-1.5 if '_calf_' in n else 0.)) for n in names}
        limits = {n:[-.783,.783] if '_hip_' in n else ([-.6585,4.4085] if '_thigh_' in n else ([-2.7005,-.5495] if '_calf_' in n else [-4.,4.])) for n in names}
    else:
        defaults = dict.fromkeys(names,0.); limits = {n:[-4.,4.] for n in names}
    for n in wheels:
        limits[n] = [np.nan,np.nan]
    groups = [('joint_leg',legs,'position',scale),('joint_wheel',wheels,'velocity',5.),('joint_arm',arms,'position',.5)]
    if reorder:
        names = tuple(reversed(names)); groups = list(reversed(groups))
    terms=[]; start=0
    for key,joints,mode,s in groups:
        if reorder:
            joints=joints[::-1]
        terms.append(ActionTerm(key,start,len(joints),joints,mode,s,True,None,'resolved'));start+=len(joints)
    schema=ActionSchema.from_terms(terms,names,[defaults[n] for n in names],[limits[n] for n in names],24)
    # Observation order deliberately differs from articulation and action order.
    observed=tuple(np.roll(names,5))
    return schema,observed,np.array([defaults[n] for n in observed])


def obs(gravity=(0,0,-1),omega=(0,0,0)):
    value=np.zeros(84);value[3:6]=omega;value[9:12]=gravity;return value


def run_checks():
    checks={}; numerical={}
    schema,names,defaults=fixture(); wheels=schema.term('joint_wheel'); leg=schema.term('joint_leg'); arm=schema.term('joint_arm')
    c=StabilityController(schema,names,defaults)
    checks['public_layout_84_correct'] = c.layout.to_dict()['expected_size']==84 and c.layout.projected_gravity==slice(9,12) and c.layout.base_ang_vel==slice(3,6)
    # The wrapper preserves settling input; the bootstrap is responsible for zero.
    p=BootstrapPolicy(schema,'turn',wheel_cmd=.6)
    checks['first_100_bootstrap_zero_outputs']=all(np.array_equal(c.apply(p.act(obs()),obs()),np.zeros(24,np.float32)) for _ in range(100))
    c=StabilityController(schema,names,defaults,settle_calls=0)
    action=np.zeros(24,np.float32);action[wheels.start:wheels.stop]=.15
    for _ in range(120): out=c.apply(action,obs())
    checks['normal_drive_reaches_requested_value']=bool(np.allclose(out[wheels.start:wheels.stop],.15))
    checks['zero_request_stops_without_slew_tail']=bool(np.array_equal(c.apply(np.zeros(24),obs())[wheels.start:wheels.stop],np.zeros(4)))
    for gravity in ((0,0,1),(1,0,0)):
        c=StabilityController(schema,names,defaults,settle_calls=0)
        for _ in range(100):out=c.apply(action,obs(gravity))
        checks['inverted_or_horizontal_'+str(gravity)+'_aborts']=c.debug['state']=='abort' and np.array_equal(out[wheels.start:wheels.stop],np.zeros(4))
    for gravity in ((0,0,0),(0,0,-2),(0,0,-1e100),(np.nan,0,-1)):
        c=StabilityController(schema,names,defaults,settle_calls=0)
        try:c.apply(action,obs(gravity))
        except StabilityError:pass
        else:raise AssertionError('invalid gravity accepted')
    checks['invalid_gravity_norm_and_finiteness_rejected']=True
    for settle in (0,100):
        c=StabilityController(schema,names,defaults,settle_calls=settle)
        large=action.astype(np.float64);large[arm.start]=1e100
        try:c.apply(large,obs())
        except StabilityError:pass
        else:raise AssertionError('float32 overflow accepted')
    checks['float32_overflow_rejected_before_and_after_settle']=True
    c=StabilityController(schema,names,defaults,settle_calls=0,strict=False)
    for _ in range(100):c.apply(action,obs())
    for _ in range(30):out=c.apply(action,obs((np.nan,0,-1)))
    checks['non_strict_bad_proprio_aborts_and_retracts']=np.isfinite(out).all() and np.array_equal(out[wheels.start:wheels.stop],np.zeros(4)) and bool(c.debug['errors'])
    rng=np.random.default_rng(32);worst_arm=0.;worst_physical_bounds=0.
    for reorder,scale in ((False,.5),(True,.23)):
        s,n,d=fixture(reorder=reorder,scale=scale);l=s.term('joint_leg');w=s.term('joint_wheel');a=s.term('joint_arm')
        c=StabilityController(s,n,d,settle_calls=0)
        for _ in range(300):
            request=rng.uniform(-10,10,24).astype(np.float32)
            gx,gy=rng.uniform(-.2,.2,2);gravity=(gx,gy,-np.sqrt(1-gx*gx-gy*gy))
            out=c.apply(request,obs(gravity,rng.uniform(-.4,.4,3)))
            assert np.isfinite(out).all() and np.max(np.abs(out[w.start:w.stop]))<=.350001
            worst_arm=max(worst_arm,float(np.max(np.abs(out[a.start:a.stop]-request[a.start:a.stop]))))
            # Independent consumption of the actual position action affine map.
            for j,name in enumerate(l.joint_names):
                idx=s.joint_index(name);physical=out[l.start+j]*l.scale+s.default_joint_pos[idx];lo,hi=s.soft_joint_pos_limits[idx]
                worst_physical_bounds=max(worst_physical_bounds,lo-physical,physical-hi)
            json.dumps(c.last_debug(),allow_nan=False)
        c=StabilityController(s,n,d,settle_calls=0,leg_mode='off')
        request=np.zeros(24,np.float32);request[l.start:l.stop]=.02
        out=c.apply(request,obs((.15,0,-np.sqrt(1-.15**2))))
        assert np.array_equal(out[l.start:l.stop],request[l.start:l.stop])
        try:StabilityController(replace(s,terms=tuple(replace(t,clip={'.*':[-1,1]}) if t.name=='joint_leg' else t for t in s.terms)),n,d)
        except StabilityError:pass
        else:raise AssertionError('unsupported clipping accepted')
    assert worst_arm==0 and worst_physical_bounds<3e-7
    numerical.update(arm_preservation_error_max=worst_arm,physical_joint_bound_float32_error_max=float(worst_physical_bounds))
    checks['reordered_terms_observations_scales_arm_preservation']=True
    checks['finite_random_outputs_wheel_caps_leg_physical_bounds']=True
    checks['leg_off_preserves_in_range_incoming_legs']=True
    checks['unsupported_leg_clip_rejected']=True
    # Sign of correction selected from a geometric gravity observation.
    for axis,angle,expected_corners in [('y',.13,{'FR','FL'}),('y',-.13,{'RR','RL'}),('x',.13,{'FR','RR'}),('x',-.13,{'FL','RL'})]:
        rotation=Rotation.from_euler(axis,angle).as_matrix();g=rotation.T@np.array([0,0,-1.])
        c=StabilityController(schema,names,defaults,settle_calls=0)
        for _ in range(100):c.apply(np.zeros(24),obs(g))
        actual={key for key,value in c.debug['legs']['corner_units'].items() if value<-.001}
        assert actual==expected_corners,(axis,angle,actual)
    checks['roll_pitch_gravity_selects_geometrically_lower_corners']=True
    assert all(checks.values()),checks
    return checks,numerical


def usd_geometry(usd_path):
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(usd_path))
    joints={p.GetName():UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    def transform(position,quaternion):
        matrix=np.eye(4);matrix[:3,3]=position
        matrix[:3,:3]=Rotation.from_quat([*quaternion.GetImaginary(),quaternion.GetReal()]).as_matrix();return matrix
    def foot(corner,units):
        matrix=np.eye(4)
        for name in ('hip','thigh','calf','foot'):
            joint=joints[corner+'_'+name+'_joint']
            q={'hip':.1 if corner[1]=='L' else -.1,'thigh':.8 if corner[0]=='F' else 1.,'calf':-1.5,'foot':0.}[name]
            q+=units*{'hip':0.,'thigh':.20,'calf':-.50,'foot':0.}[name]
            rot=np.eye(4);rot[:3,:3]=Rotation.from_euler(joint.GetPrim().GetAttribute('physics:axis').Get().lower(),q).as_matrix()
            matrix=matrix@transform(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get())@rot@np.linalg.inv(transform(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get()))
        return matrix[:3,3]
    result={}
    for corner in ('FR','FL','RR','RL'):
        initial=foot(corner,0);delta=foot(corner,-.12)-initial
        assert (initial[0]>0)==(corner[0]=='F') and (initial[1]>0)==(corner[1]=='L') and delta[2]<0
        result[corner]={'default_foot_body_xyz':initial.tolist(),'negative_012_unit_foot_delta':delta.tolist()}
    return {'passed':True,'usd':str(usd_path),'sha256':hashlib.sha256(usd_path.read_bytes()).hexdigest(),'corners':result,
            'interpretation':'Static foot lowering relative to body supports the extension sign near default. Does not prove dynamic levelling or balance.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-parity',type=Path)
    parser.add_argument('--usd',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/task_b_stability_cpu_audit.json')
    args=parser.parse_args();checks,numerical=run_checks()
    report={'status':'passed','scope':'CPU only; no simulator or GPU','checks':checks,'numerical':numerical,
            'remaining_limitations':['H3 is validated only as a local static extension direction; dynamic benefit requires paired physical runs.',
              'First 100 calls preserve incoming float32 actions. They are zero only when the upstream bootstrap emits zero.',
              'Normal tilt/rate gains unchanged. A short single-seed survival run is not proof of safe steering or useful motion.',
              'Arm values are preserved, not certified safe; invalid actions always raise even with strict=False.',
              'Wheel default velocity offsets must remain zero; they are not encoded in the present schema.']}
    if args.before_parity:
        data=np.load(args.before_parity,allow_pickle=False);s,n,d=fixture(real_defaults=False)
        controller=StabilityController(s,n,d)
        after=np.array([controller.apply(a,o) for a,o in zip(data['action'],data['proprio'])])
        assert np.array_equal(after,data['output'])
        report['normal_sequence_parity']={'frames':len(after),'bit_exact_float32':True,'max_error':float(np.max(np.abs(after-data['output']))),'source':str(args.before_parity)}
    if args.usd:
        report['usd_geometry']=usd_geometry(args.usd)
    paths=[ROOT/'task_b/stability.py',ROOT/'task_b/control.py',Path(__file__)]
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','checks':len(checks),'numerical':numerical,'normal_sequence_parity':report.get('normal_sequence_parity'),'output':str(args.output)},indent=2))


if __name__=='__main__':
    main()
