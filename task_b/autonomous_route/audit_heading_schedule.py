"""Synthetic CPU checks of scalar heading scheduling; no simulator."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from task_b.heading_schedule import HeadingSchedule, HeadingScheduleError


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    checks={}
    def rejects(fn):
        try:fn()
        except (HeadingScheduleError,ValueError):return True
        return False
    c=HeadingSchedule()
    v=c.update(.02,.2,.1,distance_m=2)
    checks['aligned_keeps_requests']=np.isclose(v['common'],.2) and np.isclose(v['differential'],.1)
    c.reset();v=c.update(.5,.2,.5,distance_m=2)
    checks['turn_priority_4percent_forward_full_turn']=np.isclose(v['common'],.008) and np.isclose(v['differential'],.5)
    c.reset();v=c.update(.2,.1,.2,distance_m=.9)
    checks['close_unaligned_no_forward']=v['common']==0 and np.isclose(v['differential'],.2)
    c.reset();v=c.update(.02,.2,.1,distance_m=.5)
    checks['distance_taper_stops_forward_but_caller_owns_full_parking']=v['common']==0 and v['differential']==.1
    for label,kwargs in [('lost',{'target_valid':False}),('no_bearing',{})]:
        bearing=None if label=='no_bearing' else .5
        v=c.update(bearing,.2,.4,**kwargs)
        checks[label+'_immediate_full_stop']=v['common']==0 and v['differential']==0 and c._bearing_f is None and c._boost==1
    for _ in range(30):c.update(.4,.1,.2,observed_yaw_rate=1.)
    v=c.update(.4,0.,0.)
    checks['zero_pair_clears_yaw_and_schedule']=v['common']==0 and v['differential']==0 and c._yaw_f==0 and c._bearing_f is None and c._yaw_stall_calls==0
    for label,value in [('nan',np.nan),('inf',np.inf),('huge_finite',1e100)]:
        checks[label+'_requests_rejected_before_target_gate']=rejects(lambda value=value:c.update(.1,value,.2,target_valid=False))
    checks['invalid_bearing_and_distance_rejected']=rejects(lambda:c.update(np.pi+.01,.1,.2)) and rejects(lambda:c.update(.1,.1,.2,distance_m=-1))
    checks['invalid_gyro_rejected']=rejects(lambda:c.update(.1,.1,.2,observed_yaw_rate=51.))
    for label,g in [('zero',(0,0,0)),('nan',(np.nan,0,-1)),('huge',(0,0,-1e308)),('quaternion',(1,0,0,0))]:
        checks[label+'_gravity_rejected']=rejects(lambda g=g:c.update(.2,.1,.2,projected_gravity=g))
    for label,g in [('inverted',(0,0,1)),('tilted',(.13,0,-np.sqrt(1-.13**2))),('roll_rate',(0,0,-1,.46,0,0))]:
        v=c.update(.2,.1,.2,projected_gravity=g)
        checks[label+'_stops_both_requests']=v['common']==v['differential']==0
        json.dumps(v,allow_nan=False)
    c.reset();v=c.update(.01,.4,.5)
    checks['per_wheel_headroom_sacrifices_forward_first']=np.isclose(v['common'],.1) and v['differential']==.5
    c.reset();v=c.update(.01,.4,1.2)
    checks['large_turn_capped_without_forward']=v['common']==0 and v['differential']==.6
    c=HeadingSchedule();rng=np.random.default_rng(725);bounded=True
    for _ in range(700):
        forward,turn=rng.uniform(-2,2,2)
        v=c.update(float(rng.uniform(-1,1)),float(forward),float(turn),distance_m=float(rng.uniform(.4,3)))
        bounded &= abs(v['common'])<=abs(forward)+1e-12 and abs(v['differential'])<=abs(turn)+1e-12 and abs(v['common'])+abs(v['differential'])<=.6+1e-12
        bounded &= v['common']*forward>=0 and v['differential']*turn>=0
        json.dumps(v,allow_nan=False)
    checks['default_never_amplifies_or_changes_sign_700_calls']=bounded
    c.reset()
    for _ in range(300):v=c.update(.5,.1,.2,observed_yaw_rate=0.)
    checks['weak_yaw_is_only_diagnostic_by_default']=v['yaw']['response_weak'] and v['differential']==.2 and v['yaw']['boost']==1
    boosted=HeadingSchedule(turn_boost_max=1.5,turn_boost_per_s=.2)
    for _ in range(500):v=boosted.update(.5,.1,-.1,observed_yaw_rate=0.)
    checks['only_explicit_bounded_boost_can_amplify']=1<v['yaw']['boost']<=1.5 and -.1500001<=v['differential']<-.1
    c=HeadingSchedule();mode_ok=True
    for _ in range(30):c.update(.4,.1,.2)
    for k in range(40):mode_ok &= c.update(.26 if k%2 else .28,.1,.2)['mode']=='TURN'
    checks['turn_exit_hysteresis_avoids_threshold_chatter']=mode_ok
    # Actual bearing never becomes small in this side-switch counterexample.
    c=HeadingSchedule()
    for _ in range(30):c.update(.4,.1,.2,distance_m=.9)
    switched=[c.update(-.4,.1,-.2,distance_m=.9)['common'] for _ in range(20)]
    checks['signed_bearing_flip_cannot_fake_close_alignment']=all(x==0 for x in switched)
    c=HeadingSchedule()
    c.update(.09,.1,.2,distance_m=2.)
    v=c.update(-.4,.1,-.2,distance_m=2.)
    checks['mode_dwell_cannot_delay_large_bearing_forward_reduction']=abs(v['common'])<=.004+1e-12
    c.reset();fresh=HeadingSchedule()
    checks['reset_matches_fresh_state']=c.update(.1,.2,.1)==fresh.update(.1,.2,.1)
    json.dumps(c.describe(),allow_nan=False)
    checks['description_json_safe']=True
    report={'scope':'Synthetic CPU scalar scheduling checks only; no physical turn, alignment, score or safety claim.',
            'passed':bool(all(checks.values())),'count':len(checks),'checks':{k:bool(v) for k,v in checks.items()},
            'source_sha256':hashlib.sha256((ROOT/'task_b/heading_schedule.py').read_bytes()).hexdigest(),
            'counterexample_after_fix_common':switched,
            'contribution':'Claude Opus delivered the original helper/design in the existing real CLI session. Codex independent review fixed signed-bearing false alignment and stale yaw state, then executed these tests.'}
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
