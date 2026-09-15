"""CPU regression for the A-raise handoff boundary in task_b/delivery.py.

Two things are checked, and neither needs a GPU or an image:

1. SOURCE SCOPE. ``PayloadRaisePrefix`` must add exactly one behavioural branch.
   Its own ``__dict__`` is inspected so the override list is read from the class
   object rather than asserted in prose.
2. BRANCH BEHAVIOUR. With the handoff armed, ``_advance_segment('A')`` must hold
   the phase at the raise and must NOT begin the swing, must be idempotent, and
   ``take_carry_state()`` must hand over the SAME tracker instance exactly once.
   With the handoff disarmed the inherited path must still begin the swing, so
   the p13 behaviour is provably unchanged.

Honest limitation: this is a BRANCH-level test, not a whole-episode replay. A
faithful replay of p13 would need the RGB-D of every one of its 5864 control
ticks, and the run saved images only every 500 steps. What is established here is
that the only edited branch behaves as specified; the unedited prefix behaviour
is unchanged because the subclass delegates to ``super()`` on every other path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(REPO))

from task_b.audit_first_reach import fixture
from task_b.arm_kinematics import ARM_JOINT_NAMES
from task_b.delivery import PayloadRaisePrefix
from task_b.payload_motion import PayloadJointTracker, PayloadMotionPolicy, build_payload_goals

SCENE_LEG_DEFAULTS = {
    'FL_hip_joint': .1, 'FR_hip_joint': -.1, 'RL_hip_joint': .1, 'RR_hip_joint': -.1,
    'FL_thigh_joint': .8, 'FR_thigh_joint': .8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5,
}


def build(handoff_at_raise):
    """A real instance whose payload state is built from the public planner."""
    schema, names, defaults = fixture()
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .035, -.035])))
    defaults.update(SCENE_LEG_DEFAULTS)
    policy = PayloadRaisePrefix(schema, names, defaults, dt=.02, handoff_at_raise=handoff_at_raise,
                                settle_calls=100, ramp_calls=100, reach_only=False,
                                forward_cmd=.05, turn_cap=.6, turn_gain=2., standoff=.50,
                                lowering_m=.02, open_on='close_complete')
    reference = np.array([defaults[name] for name in ARM_JOINT_NAMES[:6]], dtype=float)
    plan = build_payload_goals(reference, reference)
    goals = {key: plan[key] for key in ('A', 'B', 'C')}
    policy.goals = goals
    policy.reference_q = reference.copy()
    policy.tracker = PayloadJointTracker(.02, reference.copy(), reference.copy())
    policy.tracker.begin(goals['A'])
    policy.arm_command = reference.copy()
    policy.finger_command = np.array([.035, -.035])
    policy.action_template = np.zeros(24, dtype=np.float32)
    policy.phase = policy.state = 'PAYLOAD_RAISE'
    policy.calls = 500
    return policy


def main():
    report = {'scope': 'CPU branch regression for the delivery handoff; NOT a grasp, carry or '
                        'delivery result', 'checks': []}

    def check(name, ok, detail):
        report['checks'].append({'check': name, 'passed': bool(ok), 'detail': detail})
        return bool(ok)

    own = sorted(k for k in PayloadRaisePrefix.__dict__ if not k.startswith('__'))
    check('the_subclass_defines_only_the_intended_members', own == ['_advance_segment', 'take_carry_state'],
          {'PayloadRaisePrefix_own_attributes': own,
           'expected': ['_advance_segment', 'take_carry_state']})
    check('the_subclass_is_a_payload_motion_policy',
          issubclass(PayloadRaisePrefix, PayloadMotionPolicy),
          {'mro': [c.__name__ for c in PayloadRaisePrefix.__mro__[:3]]})

    # -- armed: hold the raise, never begin the swing ------------------------
    armed = build(True)
    goal_before = armed.tracker.fixed_goal.copy()
    segments_before = armed.tracker.segments_begun
    armed._advance_segment('A')
    check('armed_handoff_holds_the_raise_phase', armed.phase == 'PAYLOAD_RAISE',
          {'phase': armed.phase})
    check('armed_handoff_publishes_ready', bool(armed.handoff_ready),
          {'handoff_ready': bool(armed.handoff_ready), 'handoff_call': armed.handoff_call})
    check('armed_handoff_does_not_begin_the_swing',
          armed.tracker.segments_begun == segments_before
          and bool(np.allclose(armed.tracker.fixed_goal, goal_before)),
          {'segments_begun_before': segments_before,
           'segments_begun_after': armed.tracker.segments_begun,
           'goal_changed': bool(not np.allclose(armed.tracker.fixed_goal, goal_before))})
    check('armed_handoff_reports_no_success_stop', armed.done_reason is None,
          {'done_reason': armed.done_reason})

    # idempotent: a second call must not start B either
    armed._advance_segment('A')
    check('armed_handoff_is_idempotent',
          armed.phase == 'PAYLOAD_RAISE'
          and armed.tracker.segments_begun == segments_before
          and bool(np.allclose(armed.tracker.fixed_goal, goal_before)),
          {'phase': armed.phase, 'segments_begun': armed.tracker.segments_begun})

    # -- the handoff record --------------------------------------------------
    state = armed.take_carry_state()
    check('carry_state_transfers_the_same_tracker_instance',
          state['tracker'] is armed.tracker,
          {'same_object': state['tracker'] is armed.tracker})
    check('carry_state_carries_the_public_provenance_and_goals',
          set(state['goals_rad']) == {'A', 'B', 'C'}
          and 'no object pose' in state['source'].lower(),
          {'goal_keys': sorted(state['goals_rad']), 'source': state['source']})
    check('carry_state_keeps_the_full_leg_template_untouched',
          state['action_template'].shape == (24,)
          and not state['action_template'].any(),
          {'template_shape': list(state['action_template'].shape)})
    try:
        armed.take_carry_state()
        check('carry_state_can_only_be_taken_once', False, {'second_call': 'returned a value'})
    except ValueError as error:
        check('carry_state_can_only_be_taken_once', True, {'second_call': str(error)})

    # -- disarmed: the inherited behaviour is unchanged ----------------------
    plain = build(False)
    plain._advance_segment('A')
    check('disarmed_handoff_still_begins_the_swing', plain.phase == 'PAYLOAD_SWING',
          {'phase': plain.phase})
    check('disarmed_handoff_never_publishes_ready', not plain.handoff_ready,
          {'handoff_ready': bool(plain.handoff_ready)})

    failed = [entry for entry in report['checks'] if not entry['passed']]
    report['failed_checks'] = [entry['check'] for entry in failed]
    report['passed'] = not failed
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
