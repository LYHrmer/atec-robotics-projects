"""CPU acceptance of the neutral-reference candidate; no simulation launched."""
from pathlib import Path
import hashlib
import json
import sys
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.audit_stability import fixture, obs
from task_b.stability import StabilityController, StabilityError
from task_b.stability_profiles import NeutralAwareStabilityController


def main():
    checks = {}
    schema, names, defaults = fixture()
    wheel = schema.term('joint_wheel'); arm = schema.term('joint_arm')
    gravity = np.array([.0633, 0., -np.sqrt(1-.0633**2)])
    request = np.zeros(24, np.float32)
    request[wheel.start:wheel.stop] = [.6, -.6, .6, -.6]
    request[arm.start:arm.stop] = np.linspace(-.05,.05,8)
    normal = obs(gravity)
    def calibrated(**kwargs):
        c = NeutralAwareStabilityController(schema, names, defaults, **kwargs)
        for _ in range(100):
            assert np.array_equal(c.apply(np.zeros(24,np.float32),normal),np.zeros(24,np.float32))
        return c
    c = calibrated(); base = StabilityController(schema,names,defaults)
    for _ in range(100): base.apply(np.zeros(24),normal)
    action_copy=request.copy(); observation_copy=normal.copy()
    for _ in range(400):
        new = c.apply(request,normal); old = base.apply(request,normal)
    checks['calibrates_nominal_tilt_and_preserves_authority'] = bool(c.calibrated and c.authority > .999 and base.authority < .31)
    checks['neutral_turn_request_is_not_permanently_cut_by_nominal_pitch'] = bool(np.min(np.abs(new[wheel.start:wheel.stop]))>.29 and np.max(np.abs(old[wheel.start:wheel.stop]))<.10)
    checks['caller_arrays_unchanged_and_arm_preserved'] = bool(np.array_equal(request,action_copy) and np.array_equal(normal,observation_copy) and np.array_equal(new[arm.start:arm.stop],request[arm.start:arm.stop]))
    stop=request.copy();stop[wheel.start:wheel.stop]=0
    checks['zero_wheel_request_immediate_stop'] = bool(np.array_equal(c.apply(stop,normal)[wheel.start:wheel.stop],np.zeros(4)))
    checks['proper_rotation'] = bool(np.allclose(c._rotation@gravity,[0,0,-1],atol=1e-12) and np.isclose(np.linalg.det(c._rotation),1))
    for label,g in [('inverted',(0,0,1)),('horizontal',(1,0,0)),('absolute_tilt',(.4,0,-np.sqrt(.84)))]:
        c=calibrated()
        out=c.apply(request,obs(g))
        checks[label+'_cannot_be_hidden_by_reference'] = bool(np.all(out[wheel.start:wheel.stop]==0) and c.debug['neutral_reference']['absolute']['danger'])
        if label=='inverted':
            checks['inverted_absolute_diagnostic_reports_180_degrees'] = bool(np.isclose(c.debug['neutral_reference']['absolute']['tilt_deg'],180.))
        json.dumps(c.debug,allow_nan=False)
    for label,g,rate in [('excessive_reference',(.2,0,-np.sqrt(.96)),(0,0,0)),('moving_reference',gravity,(.2,0,0))]:
        c=NeutralAwareStabilityController(schema,names,defaults)
        for _ in range(100):c.apply(np.zeros(24),obs(g,rate))
        checks[label+'_refused'] = not c.calibrated
    c=calibrated();frozen=c._reference.copy()
    for _ in range(30):c.apply(request,obs((.10,0,-np.sqrt(.99))))
    checks['reference_frozen_after_settle'] = bool(np.array_equal(frozen,c._reference))
    c.reset();checks['reset_clears_reference'] = not c.calibrated and c.calls==0
    for label,bad in [('zero_gravity',obs((0,0,0))),('nonfinite_gravity',obs((np.nan,0,-1))),('huge_gravity',obs((0,0,-1e100)))]:
        c=calibrated()
        try:c.apply(request,bad)
        except StabilityError:checks[label+'_rejected']=True
        else:checks[label+'_rejected']=False
    c=calibrated();huge=request.astype(float);huge[arm.start]=1e100
    try:c.apply(huge,normal)
    except StabilityError:checks['float32_overflow_rejected']=True
    else:checks['float32_overflow_rejected']=False
    c=calibrated();huge=request.astype(float);huge[wheel.start]=1e100
    try:c.apply(huge,obs((0,0,1)))
    except StabilityError:checks['absolute_gate_cannot_hide_wheel_input_overflow']=True
    else:checks['absolute_gate_cannot_hide_wheel_input_overflow']=False
    before=ROOT/'results/task_b_bootstrap/source_snapshots/stability_profiles_1bdadc29.py'
    if before.exists():
        spec=importlib.util.spec_from_file_location('task_b._profile_before_input_fix',before)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=module.NeutralAwareStabilityController(schema,names,defaults)
        current=NeutralAwareStabilityController(schema,names,defaults)
        rng=np.random.default_rng(92);same=True
        for i in range(500):
            a=np.zeros(24,np.float32) if i<100 else rng.uniform(-.6,.6,24).astype(np.float32)
            o=normal if i<100 else obs((.08,0,-np.sqrt(1-.08**2)),rng.uniform(-.2,.2,3))
            same &= np.array_equal(current.apply(a,o),old.apply(a,o))
        checks['bounded_action_outputs_unchanged_by_input_validation_fix_500_frames']=bool(same)
    c=NeutralAwareStabilityController(schema,names,defaults,enable_neutral_reference=False)
    base=StabilityController(schema,names,defaults);rng=np.random.default_rng(90);same=True
    for i in range(500):
        gxy=rng.uniform(-.1,.1,2);o=obs((*gxy,-np.sqrt(1-float(gxy@gxy))),rng.uniform(-.2,.2,3))
        a=np.zeros(24,np.float32) if i<100 else rng.uniform(-.3,.3,24).astype(np.float32)
        same &= np.array_equal(c.apply(a,o),base.apply(a,o))
    checks['disabled_profile_matches_baseline_500_frames'] = bool(same)
    replay = None
    if len(sys.argv)>1:
        source=Path(sys.argv[1]);data=np.load(source)
        c=NeutralAwareStabilityController(schema,names,defaults)
        for o in data['proprio'][:100]:c.apply(np.zeros(24),o)
        replay={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'calibrated':c.calibrated,'reason':c.calibration_reason,
                'interpretation':'CPU replay of recorded settle observations only; not a new physical episode'}
        checks['actual_recorded_settle_calibrates'] = bool(c.calibrated)
    report={'checks':checks,'passed':all(checks.values()),'count':len(checks),'recorded_settle_replay':replay,
            'profile_source_sha256':hashlib.sha256((ROOT/'task_b/stability_profiles.py').read_bytes()).hexdigest(),
            'scope':'CPU behavior and boundary tests; no claim of driving/grasping performance'}
    (ROOT/'results/task_b_neutral_profile_cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed']

if __name__=='__main__':main()
