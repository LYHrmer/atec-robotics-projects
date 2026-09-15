"""CPU boundary checks for the locomotion candidate; synthetic, not a physics test."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.audit_stability import fixture, obs
from task_b.control import ActionSchema
from task_b.locomotion import LocomotionController, LocomotionError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    schema, names, defaults = fixture()
    wheel = schema.term('joint_wheel')
    wi = np.arange(wheel.start, wheel.stop)
    nonwheel = np.ones(24, bool)
    nonwheel[wi] = False
    checks = {}

    def observation(velocities=None, gravity=(.06, 0., -np.sqrt(1-.06**2)), omega=(0.,0.,0.), order=names):
        p = obs(gravity, omega)
        if velocities is not None:
            for name, value in zip(wheel.joint_names, velocities):
                p[36 + order.index(name)] = value
        return p

    def action(common=.1):
        a = np.linspace(-.05, .05, 24).astype(np.float32)
        a[wi] = common
        return a

    def rejects(fn):
        try:
            fn()
        except (LocomotionError, ValueError):
            return True
        return False

    p = observation([.2]*4)
    a = action()
    before_a, before_p = a.copy(), p.copy()
    c = LocomotionController(schema, names)
    for _ in range(200):
        out = c.apply(a, p, bearing_error=.2)
    checks['caller_buffers_unchanged'] = np.array_equal(a,before_a) and np.array_equal(p,before_p)
    checks['nonwheel_entries_bit_exact'] = np.array_equal(out[nonwheel],a[nonwheel])
    checks['positive_bearing_positive_right_minus_left_hypothesis'] = out[wi[0]] > out[wi[1]] > 0
    checks['physical_scale_bound'] = np.max(np.abs(out[wi]*wheel.scale)) <= c.max_wheel_rad_s+1e-6
    zero = a.copy(); zero[wi] = 0
    checks['parking_immediate_and_feedback_cleared'] = np.all(c.apply(zero,p,bearing_error=.2)[wi]==0) and c._assist==0 and c._diff_i==0
    checks['disabled_immediate_stop_preserves_arm_and_leg'] = np.array_equal(c.apply(a,p,enabled=False)[nonwheel],a[nonwheel]) and np.all(c._applied==0)
    c.reset()
    checks['reset_clears_counters_and_filters'] = c.calls==0 and not c._filters_primed and np.all(c._applied==0)

    for label,bad in [('nan_action',np.nan),('infinite_action',np.inf),('float32_overflow',1e100)]:
        value=a.astype(float);value[wi[0]]=bad
        checks[label+'_rejected_even_disabled'] = rejects(lambda value=value: c.apply(value,p,enabled=False))
    for label,g in [('zero_gravity',(0,0,0)),('nan_gravity',(np.nan,0,-1)),('huge_gravity',(0,0,-1e100))]:
        checks[label+'_rejected'] = rejects(lambda g=g:c.apply(a,observation(gravity=g)))
    for label,g in [('inverted',(0,0,1)),('horizontal',(1,0,0)),('early_tilt',(.13,0,-np.sqrt(1-.13**2)))]:
        out=c.apply(a,observation(gravity=g),bearing_error=.2)
        checks[label+'_stops_wheels_preserves_nonwheel'] = np.all(out[wi]==0) and np.array_equal(out[nonwheel],a[nonwheel])
        json.dumps(c.debug,allow_nan=False)
    c.apply(a,observation(gravity=(0,0,1)))
    checks['inverted_tilt_reports_180_degrees'] = c.debug['posture']['tilt_deg']==180.
    checks['raw_roll_rate_stop'] = np.all(c.apply(a,observation(omega=(1.6,0,0)),bearing_error=.2)[wi]==0)
    checks['invalid_bearing_rejected'] = all(rejects(lambda v=v:c.apply(a,p,bearing_error=v)) for v in [np.nan,np.inf,np.pi+.01,[0.,1.]])
    checks['wrong_observation_names_rejected'] = rejects(lambda:LocomotionController(schema,names[:-1]))
    checks['nonzero_default_wheel_velocity_rejected'] = rejects(lambda:LocomotionController(schema,names,wheel_default_vel_rad_s=.1))
    c=LocomotionController(schema,names)
    c.apply(a,p,bearing_error=.2)
    rate_extremes_rejected=True
    for huge in (1e308,-1e308):
        extreme=observation([huge]*4,omega=(0,0,huge))
        rate_extremes_rejected &= rejects(lambda extreme=extreme:c.apply(a,extreme,bearing_error=.2))
    recovered=c.apply(a,p,bearing_error=.2)
    json.dumps(c.debug,allow_nan=False)
    checks['float64_extreme_feedback_rejected_without_filter_poisoning'] = rate_extremes_rejected and np.isfinite(recovered).all() and np.isfinite(c._wheel_f).all() and np.isfinite(c._yaw_f)
    permissive=LocomotionController(schema,names,strict=False)
    permissive.apply(a,p,bearing_error=.2)
    stopped=permissive.apply(a,observation([1e308]*4),bearing_error=.2)
    checks['permissive_extreme_feedback_stops_and_recovers'] = np.all(stopped[wi]==0) and np.array_equal(stopped[nonwheel],a[nonwheel]) and np.isfinite(permissive.apply(a,p,bearing_error=.2)).all()

    # A measured overspeed must remove assistance relative to the USER target,
    # not keep chasing a previously compensated, larger motor target.
    c=LocomotionController(schema,names)
    for _ in range(500): c.apply(a,observation([.2]*4),bearing_error=0.)
    assist_before=c._assist
    for _ in range(150): out=c.apply(a,observation([.7]*4),bearing_error=0.)
    checks['nominal_speed_overshoot_removes_assistance'] = assist_before>.05 and c._assist<1e-6 and np.max(out[wi])<=.100001

    c=LocomotionController(schema,names)
    maximum_assist=maximum_integral=0.
    for _ in range(1500):
        out=c.apply(a,observation([0.]*4),bearing_error=.4)
        maximum_assist=max(maximum_assist,c._assist);maximum_integral=max(maximum_integral,abs(c._diff_i))
    checks['blocked_wheels_bounded_and_tracking_failure_explicit'] = maximum_assist<=c.forward_assist_max and maximum_integral<=c.diff_i_max and c.common_tracking_failed

    # Same request, gyro response above desired yaw should reduce the correction.
    slow=LocomotionController(schema,names,yaw_ki=.2)
    fast=LocomotionController(schema,names,yaw_ki=.2)
    for _ in range(100):
        oslow=slow.apply(a,observation([.5]*4,omega=(0,0,0)),bearing_error=.15)
        ofast=fast.apply(a,observation([.5]*4,omega=(0,0,.35)),bearing_error=.15)
    checks['gyro_overspeed_reduces_yaw_correction'] = (ofast[wi[0]]-ofast[wi[1]]) < (oslow[wi[0]]-oslow[wi[1]])
    opposite=LocomotionController(schema,names,turn_sign=-1)
    for _ in range(100): reverse=opposite.apply(a,p,bearing_error=.2)
    checks['explicit_turn_sign_reverses_correction'] = reverse[wi[0]]<reverse[wi[1]]

    c=LocomotionController(schema,names)
    spin=a.copy();spin[wi]=[.2,-.2,.2,-.2]
    for _ in range(100):c.apply(spin,p)
    out=c.apply(a,p,bearing_error=.5)
    checks['old_spin_not_carried_into_arc'] = np.min(out[wi])>=0 and c.debug['mode']=='arc'
    checks['zero_common_arc_stops'] = np.all(c.apply(spin,p,bearing_error=.5)[wi]==0)

    c=LocomotionController(schema,names)
    rng=np.random.default_rng(471)
    envelope_ok=slew_ok=preserved_ok=True;previous=np.zeros(4)
    for i in range(600):
        request=action(float(rng.uniform(.02,.29)))
        out=c.apply(request,observation(rng.uniform(.1,.6,4)),bearing_error=float(rng.uniform(-.7,.7)))
        value=out[wi].astype(float);common=float(value.mean());diff=float((value[0]+value[2]-value[1]-value[3])/4)
        envelope_ok &= np.min(value)>=-1e-8 and abs(diff)<=c.curvature_ratio*abs(common)+1e-7 and np.isfinite(out).all()
        slew_ok &= np.max(np.abs(value-previous))<=c.slew_per_call+1e-7
        preserved_ok &= np.array_equal(out[nonwheel],request[nonwheel])
        previous=value
    checks['changing_arc_preserves_curvature_bound_600_calls'] = envelope_ok
    checks['arc_slew_bound_600_calls'] = slew_ok
    checks['arbitrary_nonwheel_values_preserved_600_calls'] = preserved_ok

    # Independently permute articulation order, observation order, action term
    # order and the four wheel action names. Compare physical named outputs.
    observed=tuple(reversed(names));joints=tuple(reversed(schema.joint_names));ids=[schema.joint_names.index(n) for n in joints]
    terms=[];offset=0
    for original in reversed(schema.terms):
        term=replace(original,start=offset,joint_names=tuple(reversed(original.joint_names)))
        terms.append(term);offset+=term.dim
    other=ActionSchema.from_terms(terms,joints,schema.default_joint_pos[ids],schema.soft_joint_pos_limits[ids],24)
    base=LocomotionController(schema,names);permuted=LocomotionController(other,observed)
    data_by_joint={joint:float(a[t.start+i]) for t in schema.terms for i,joint in enumerate(t.joint_names)}
    a2=np.zeros(24,np.float32)
    for t in other.terms:
        for i,joint in enumerate(t.joint_names):a2[t.start+i]=data_by_joint[joint]
    pair_equal=True
    for _ in range(120):
        o1=base.apply(a,observation([.21,.19,.24,.17]),bearing_error=.12)
        o2=permuted.apply(a2,observation([.21,.19,.24,.17],order=observed),bearing_error=.12)
        d1={joint:float(o1[t.start+i]) for t in schema.terms for i,joint in enumerate(t.joint_names)}
        d2={joint:float(o2[t.start+i]) for t in other.terms for i,joint in enumerate(t.joint_names)}
        pair_equal &= d1==d2
    checks['independent_joint_and_action_permutations_match'] = pair_equal

    scaled_terms=[replace(t,scale=2.5) if t.name=='joint_wheel' else t for t in schema.terms]
    scaled=ActionSchema.from_terms(scaled_terms,schema.joint_names,schema.default_joint_pos,schema.soft_joint_pos_limits,24)
    c=LocomotionController(scaled,names)
    large=a.copy();large[wi]=100
    for _ in range(400):out=c.apply(large,p)
    checks['physical_bound_uses_live_nondefault_scale'] = np.allclose(out[wi]*2.5,3.,atol=1e-6)
    json.dumps(c.describe(),allow_nan=False);json.dumps(c.debug,allow_nan=False)
    checks['json_safe_debug_and_description'] = True
    report={'scope':'Synthetic CPU behavior and boundary checks only; no simulation, performance or score claim.',
            'passed':bool(all(checks.values())),'count':len(checks),'checks':{k:bool(v) for k,v in checks.items()},
            'source_sha256':hashlib.sha256((ROOT/'task_b/locomotion.py').read_bytes()).hexdigest(),
            'assistance_overspeed_case':{'nominal_rad_s':.5,'observed_rad_s':.7,'before':assist_before},
            'contribution':'Claude Opus produced the original module; Codex completed this audit and corrected reviewed boundaries after an API spending-limit stop.'}
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
