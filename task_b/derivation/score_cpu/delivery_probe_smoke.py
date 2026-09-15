"""Synthetic-plant smoke test for the carry probe phase machine in task_b/delivery.py.

What this is: the probe's PHASE MACHINE and its wheel-anchor boundary exercised
against a trivial synthetic base that converts a wheel request into planar motion.
It answers questions a real GPU run is too expensive to answer first: does the
handoff actually hand over, is the anchor released before moving and re-engaged
only after a settle, are the movement and stationary budgets separate, and do the
stall/overspeed/NaN guards fire as normal stop reasons.

What this is NOT: it is not the robot, not physics, not a grasp, carry, clearance,
delivery or score result, and it does not validate the drive layer's real
response. The synthetic plant is deliberately simple and its constants are
invented; nothing measured here transfers to the real chassis.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(REPO))

from task_b.arm_kinematics import ARM_JOINT_NAMES
from task_b.audit_first_reach import fixture
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.delivery import PayloadDeliveryPolicy, PROBE_SPEED_CAP_M_S
from task_b.payload_motion import PayloadJointTracker, build_payload_goals

SCENE_LEG_DEFAULTS = {
    'FL_hip_joint': .1, 'FR_hip_joint': -.1, 'RL_hip_joint': .1, 'RR_hip_joint': -.1,
    'FL_thigh_joint': .8, 'FR_thigh_joint': .8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5,
}

DT = .02
#: INVENTED synthetic-plant constants. Not robot properties.
FAKE_TRACK_GAIN = .85         # achieved speed / requested speed
FAKE_TURN_GAIN = .50          # achieved yaw / requested yaw
FAKE_WEAK_TURN_GAIN = .10     # a deliberately weak-yaw plant, see main()
FAKE_HALF_TRACK_M = .25       # half the left-right wheel spacing
GRAVITY_LEVEL = np.array([0., 0., -1.])


class FakeDrive:
    """Maps a planar request to physical wheel rad/s; no dynamics, no slip model.

    Schema wheel order is FR, FL, RR, RL, so indices 0 and 2 are the RIGHT wheels
    and 1 and 3 are the LEFT ones. A positive yaw request therefore drives the
    right pair faster, which is the convention the real schema uses.
    """

    def __init__(self, radius_m, turn_gain=FAKE_TURN_GAIN):
        self.radius_m = float(radius_m)
        self.turn_gain = float(turn_gain)

    def update(self, proprio, speed_m_s, yaw_rate_rad_s):
        base = speed_m_s/self.radius_m
        # A yaw request becomes a left-right wheel SPEED difference across the
        # track, so it scales with half-track over radius, not with radius.
        split = yaw_rate_rad_s*FAKE_HALF_TRACK_M/self.radius_m
        return {'wheel_target_rad_s': np.array([base+split, base-split, base+split, base-split]),
                'state': {'speed_request_m_s': speed_m_s, 'yaw_request_rad_s': yaw_rate_rad_s}}


class FaultingDrive(FakeDrive):
    """Latches a fault the way the real drive does, instead of raising."""

    def __init__(self, radius_m, fault_at=40):
        super().__init__(radius_m)
        self.fault_at = int(fault_at)
        self.calls = 0
        self.done_reason = None

    def update(self, proprio, speed_m_s, yaw_rate_rad_s):
        self.calls += 1
        if self.calls >= self.fault_at and self.done_reason is None:
            self.done_reason = 'synthetic_drive_fault'
        return super().update(proprio, speed_m_s, yaw_rate_rad_s)


class FakeBase:
    """Integrates a forward speed and yaw rate from the requested wheel targets."""

    def __init__(self, radius_m, track_gain=FAKE_TRACK_GAIN, turn_gain=FAKE_TURN_GAIN):
        self.radius_m = float(radius_m)
        self.track_gain = float(track_gain)
        self.turn_gain = float(turn_gain)
        self.speed = 0.
        self.yaw_rate = 0.

    def step(self, wheel_rad_s):
        if wheel_rad_s is None or not np.any(wheel_rad_s):
            target_speed, target_yaw = 0., 0.
        else:
            right = float(np.mean(wheel_rad_s[[0, 2]]))
            left = float(np.mean(wheel_rad_s[[1, 3]]))
            target_speed = .5*(right+left)*self.radius_m*self.track_gain
            target_yaw = (right-left)*self.radius_m/(2.*FAKE_HALF_TRACK_M)*self.turn_gain
        # first-order approach so braking has a realistic lag
        self.speed += (target_speed-self.speed)*.25
        self.yaw_rate += (target_yaw-self.yaw_rate)*.25
        return self.speed, self.yaw_rate


def build(turn_gain=FAKE_TURN_GAIN, drive=None, mode='carry_probe', start_xy=None,
          start_yaw=None):
    schema, names, defaults = fixture()
    defaults = dict(defaults)
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .035, -.035])))
    defaults.update(SCENE_LEG_DEFAULTS)
    terms = tuple(replace(term, default_joint_pos=None) if hasattr(term, 'default_joint_pos') else term
                  for term in schema.terms)
    schema = replace(schema, terms=terms,
                     default_joint_pos=np.array([defaults[name] for name in names]))
    arm, leg, wheel = (schema.term(name) for name in (ARM_TERM, LEG_TERM, WHEEL_TERM))

    policy = PayloadDeliveryPolicy(schema, names, defaults, dt=DT, mode=mode,
                                   settle_calls=100, ramp_calls=100, forward_cmd=.05,
                                   turn_cap=.6, turn_gain=2., standoff=.50, lowering_m=.02,
                                   wheel_action_gain=8.,
                                   drive=FakeDrive(.1129, turn_gain) if drive is None else drive)
    reference = np.array([defaults[name] for name in ARM_JOINT_NAMES[:6]], dtype=float)
    plan = build_payload_goals(reference, reference)
    goals = {key: plan[key] for key in ('A', 'B', 'C')}
    # The handoff happens AFTER A has actually arrived, so the tracker is already
    # settled on the A goal and its command sits there. Seeding it at the pre-raise
    # pose instead would model a state the handoff can never occur in.
    tracker = PayloadJointTracker(DT, goals['A'].copy(), goals['A'].copy())
    tracker.begin(goals['A'])

    # Inject the handoff directly: the prefix's own completion needs RGB-D and is
    # covered by the separate branch regression, not by this smoke test.
    policy.carry = {
        'control_tick': 0, 'prefix_phase': 'PAYLOAD_RAISE',
        'action_template': np.zeros(24, dtype=np.float32),
        'finger_command': np.array([.035, -.035]),
        'goals_rad': goals, 'tracker': tracker, 'arm_command_rad': goals['A'].copy(),
        'prefix_state_reason': 'injected_for_the_smoke_test', 'reference_q_rad': reference.copy(),
        'source': 'injected synthetic handoff for a CPU smoke test',
    }
    policy.tracker = tracker
    policy.template = policy.carry['action_template']
    policy.finger_command = np.array([.035, -.035])
    policy.arm_command = goals['A'].copy()
    policy._begin('CARRY_READY', 'injected')
    if start_xy is not None:
        # The real run reaches the dock from its post-grasp pose, not from the
        # public spawn the odometry starts at; the smoke test starts it there.
        policy.odometry._xy = np.asarray(start_xy, dtype=float)
    if start_yaw is not None:
        policy.odometry._yaw_unwrapped = float(start_yaw)
    # The arm really sits ON the A goal once the raise has arrived, so the probe
    # fixture must hold it there; parking it at the seed pose would (correctly)
    # fail the goal-error element of the arrival gate.
    return policy, schema, names, defaults, arm, leg, wheel, goals['A'].copy()


def make_proprio(policy, names, defaults, wheel_joint_names, reference, speed, yaw_rate, arm_q=None,
                 wheel_q=None, jam=False):
    obs = np.zeros(84)
    up = np.array([0., 0., 1.])
    forward = np.array([1., 0., 0.])
    if jam:
        obs[:3] = 0.
    else:
        obs[:3] = forward*speed
    obs[3:6] = up*yaw_rate
    obs[9:12] = GRAVITY_LEVEL
    for index, name in enumerate(names):
        if name in defaults:
            obs[12+index] = (wheel_q.get(name, defaults[name]) if wheel_q else defaults[name])-defaults[name]
        obs[36+index] = 0.
    # The synthetic arm FOLLOWS the commanded position. Holding it fixed (as this
    # fixture first did) makes any phase that commands a new arm goal -- the
    # low-place IK before release -- time out, because the measured error never
    # closes. That is a property of the fixture, not of the phase machine.
    commanded = getattr(policy, 'arm_command', None)
    arm = reference if arm_q is not None else (
        np.asarray(commanded, dtype=float) if commanded is not None else reference)
    for offset, name in enumerate(ARM_JOINT_NAMES[:6]):
        index = names.index(name)
        obs[12+index] = arm[offset]-defaults[name]
    for offset, name in enumerate(ARM_JOINT_NAMES[6:]):
        index = names.index(name)
        obs[12+index] = policy.finger_command[offset]-defaults[name]
    # joint velocity of the arm: zero once settled, so quiet windows can close
    return obs


def wheels_of(action, wheel):
    return action[wheel.start:wheel.stop]*8.*wheel.scale


def run(policy, schema, names, defaults, wheel, reference, *, jam=False, steps=2600,
        turn_gain=FAKE_TURN_GAIN):
    plant = FakeBase(.1129, turn_gain=turn_gain)
    trace = []
    images = {}
    for _ in range(steps):
        proprio = make_proprio(policy, names, defaults, wheel.joint_names, reference,
                               plant.speed, plant.yaw_rate, jam=jam)
        action = policy.act(proprio, images)
        request = None if policy.wheel_hold_requested else wheels_of(action, wheel)
        plant.step(request)
        trace.append({'phase': policy.phase, 'anchor_released': policy.anchor_released,
                      'wheel_hold_requested': bool(policy.wheel_hold_requested),
                      'speed': plant.speed, 'yaw_rate': plant.yaw_rate,
                      'range_m': policy.range_to_centre_m,
                      'heading_error_rad': policy.heading_error_rad,
                      'gap_m': policy.width,
                      'request_rad_s': None if request is None else np.asarray(request, dtype=float).copy(),
                      'policy_request_rad_s': np.asarray(policy.wheel_request_rad_s, dtype=float).copy(),
                      'done_reason': policy.done_reason})
        if policy.done_reason is not None:
            break
    return trace


def main():
    report = {'scope': 'synthetic-plant smoke test of the carry probe phase machine; NOT the robot, '
                        'NOT physics, NOT a carry or delivery result',
              'synthetic_plant_constants_are_invented': True, 'checks': []}

    def check(name, ok, detail):
        report['checks'].append({'check': name, 'passed': bool(ok), 'detail': detail})
        return bool(ok)

    policy, schema, names, defaults, arm, leg, wheel, reference = build()
    trace = run(policy, schema, names, defaults, wheel, reference)
    phases = [entry['phase'] for entry in trace]
    seen = []
    for phase in phases:
        if not seen or seen[-1] != phase:
            seen.append(phase)
    report['phase_sequence'] = seen
    report['final_reason'] = policy.done_reason

    check('the_probe_walks_the_intended_phase_order',
          seen == ['CARRY_READY', 'PROBE_DRIVE', 'PROBE_BRAKE_DRIVE', 'PROBE_YAW_POS',
                   'PROBE_BRAKE_YAW_POS', 'PROBE_YAW_NEG', 'PROBE_BRAKE_YAW_NEG', 'PROBE_HOLD',
                   'STOPPED'],
          {'observed': seen})
    check('the_probe_ends_on_its_normal_reason',
          policy.done_reason == 'delivery_carry_probe_complete',
          {'done_reason': policy.done_reason})
    moved = max(entry['speed'] for entry in trace)
    check('the_base_actually_moved_in_the_synthetic_plant', moved > .005,
          {'peak_synthetic_speed_m_s': moved})
    moved_ticks = [entry for entry in trace if entry['phase'] in
                   ('PROBE_DRIVE', 'PROBE_YAW_POS', 'PROBE_YAW_NEG')]
    check('the_anchor_was_released_for_every_movement_tick',
          all(entry['anchor_released'] and not entry['wheel_hold_requested']
              for entry in moved_ticks),
          {'movement_ticks': len(moved_ticks),
           'any_held': any(entry['wheel_hold_requested'] for entry in moved_ticks)})
    brake_ticks = [entry for entry in trace if entry['phase'].startswith('PROBE_BRAKE')]
    check('no_new_anchor_is_taken_while_still_sliding',
          all(entry['speed'] < .01 or entry['wheel_hold_requested'] is False
              for entry in brake_ticks),
          {'brake_ticks': len(brake_ticks)})
    check('anchors_are_engaged_only_at_settles',
          len(policy.anchor_engage_calls) == 3,
          {'anchor_engage_calls': policy.anchor_engage_calls,
           'expected': 'one new anchor per brake settle: after the drive brake and after each yaw brake'})

    # The contract's unit chain, checked on real emitted actions: the action the
    # evaluator receives, multiplied by the gain it applies and the schema wheel
    # scale, must land back on the physical rad/s the drive layer asked for.
    # Getting this wrong by one factor of 8 is the concrete failure it guards.
    mismatch = 0
    compared = 0
    for entry in trace:
        if entry['request_rad_s'] is None:
            continue
        compared += 1
        if not np.allclose(entry['request_rad_s'], entry['policy_request_rad_s'], atol=1e-9):
            mismatch += 1
    check('the_wheel_unit_boundary_round_trips_exactly_once',
          compared > 0 and mismatch == 0,
          {'movement_ticks_compared': compared, 'mismatches': mismatch,
           'chain': 'physical_rad_s = normalized_action * wheel_action_gain * wheel_scale',
           'divisor_used': policy.wheel_divisor})

    # -- guards fire as NORMAL stop reasons, never as silent success ----------
    jammed, schema2, names2, defaults2, _, _, wheel2, reference2 = build()
    jam_trace = run(jammed, schema2, names2, defaults2, wheel2, reference2, jam=True)
    check('a_blocked_base_stops_on_no_progress_not_a_timeout',
          jammed.done_reason == 'delivery_probe_no_forward_progress',
          {'done_reason': jammed.done_reason})

    nan_policy, schema3, names3, defaults3, _, _, wheel3, reference3 = build()
    bad = np.full(84, np.nan)
    nan_policy.act(bad, {})
    check('a_non_finite_observation_stops_cleanly',
          nan_policy.done_reason == 'delivery_invalid_proprio',
          {'done_reason': nan_policy.done_reason})

    over_policy, schema4, names4, defaults4, _, _, wheel4, reference4 = build()
    over_policy.act(make_proprio(over_policy, names4, defaults4, wheel4.joint_names,
                                 reference4, PROBE_SPEED_CAP_M_S+1., 0.), {})
    check('an_overspeed_reading_stops_the_run',
          over_policy.done_reason == 'delivery_probe_overspeed',
          {'done_reason': over_policy.done_reason,
           'reading_m_s': PROBE_SPEED_CAP_M_S+1., 'cap_m_s': PROBE_SPEED_CAP_M_S})

    # A drive that latches a fault must surface as a normal stop, not leave the
    # controller commanding a drive that has already given up.
    fault_drive = FaultingDrive(.1129, fault_at=40)
    fault_policy, s6, n6, d6, _, _, w6, r6 = build(drive=fault_drive)
    run(fault_policy, s6, n6, d6, w6, r6)
    check('a_latched_drive_fault_surfaces_as_a_normal_stop',
          fault_policy.done_reason is not None
          and fault_policy.done_reason.startswith('delivery_drive_fault:'),
          {'done_reason': fault_policy.done_reason,
           'drive_latched': fault_drive.done_reason})

    # A chassis with weak yaw authority must be REPORTED, not smoothed into a
    # pass. This is the case the real robot may well turn out to be.
    weak, schema5, names5, defaults5, _, _, wheel5, reference5 = build(FAKE_WEAK_TURN_GAIN)
    run(weak, schema5, names5, defaults5, wheel5, reference5, turn_gain=FAKE_WEAK_TURN_GAIN)
    check('a_weak_yaw_plant_surfaces_as_an_honest_diagnostic',
          weak.done_reason in ('delivery_probe_yaw_timeout', 'delivery_probe_no_yaw_progress'),
          {'done_reason': weak.done_reason, 'plant_turn_gain': FAKE_WEAK_TURN_GAIN,
           'note': 'a weak chassis is a measurement this probe exists to produce'})

    # -- first_delivery: the frontal dock ------------------------------------
    dock_policy, s8, n8, d8, _, _, w8, r8 = build(mode='first_delivery',
                                                  start_xy=(-7.173, -8.990), start_yaw=0.2560)
    dock_trace = run(dock_policy, s8, n8, d8, w8, r8, steps=40000)
    dock_phases = []
    for entry in dock_trace:
        if not dock_phases or dock_phases[-1] != entry['phase']:
            dock_phases.append(entry['phase'])
    report['delivery_phase_sequence'] = dock_phases
    report['low_place_error'] = getattr(dock_policy, 'low_place_error', 'field missing')
    report['low_place_goal'] = getattr(dock_policy, 'low_place_goal', 'field missing')
    report['low_place_tried'] = getattr(dock_policy, 'low_place_tried', 'field missing')
    _lp = [e for e in dock_trace if e['phase'] == 'LOW_PLACE']
    report['low_place_ticks'] = len(_lp)
    report['low_place_final_gap'] = (_lp[-1]['gap_m'] if _lp else None)
    report['delivery_final_reason'] = dock_policy.done_reason
    check('the_delivery_walks_the_intended_phase_order',
          dock_phases == ['CARRY_READY', 'TURN_APPROACH', 'DOCK_BRAKE', 'LOW_PLACE',
                          'RELEASE_OPEN', 'RELEASE_OBSERVE', 'STOPPED'],
          {'observed': dock_phases})
    check('the_delivery_ends_on_its_release_reason',
          dock_policy.done_reason == 'delivery_release_sequence_complete',
          {'done_reason': dock_policy.done_reason})
    dock_rows = [e for e in dock_trace if e['phase'] in ('DOCK_BRAKE', 'RELEASE_OPEN',
                                                         'RELEASE_OBSERVE')
                 and e['range_m'] is not None]
    if dock_rows:
        ranges = [e['range_m'] for e in dock_rows]
        report['delivery_dock_range_m'] = [min(ranges), max(ranges)]
        check('the_dock_settles_inside_the_real_usable_window',
              all(1.511 <= r <= 1.66 for r in ranges),
              {'range_range_m': [min(ranges), max(ranges)],
               'window_m': [1.511, 1.66],
               'note': 'the window upper bound is 1.66 m WITH the arm swing; with the arm '
                       'straight ahead it is 1.578 m because the bottle leaves the mouth'})
    else:
        check('the_dock_settles_inside_the_real_usable_window', False,
              {'detail': 'the delivery never reached DOCK_BRAKE'})
    obs_rows = [e for e in dock_trace if e['phase'] == 'RELEASE_OBSERVE']
    check('the_observation_lasts_a_real_interval',
          len(obs_rows) * DT >= 3.0,
          {'observed_s': len(obs_rows)*DT, 'required_s': 3.0})
    check('the_release_gate_is_the_measured_gap_not_the_command',
          any(e['gap_m'] is not None and abs(e['gap_m']-0.07) <= 0.003 for e in obs_rows),
          {'gap_at_observe_m': [round(e['gap_m'], 5) for e in obs_rows[:3] if e['gap_m'] is not None]})

    failed = [entry for entry in report['checks'] if not entry['passed']]
    report['failed_checks'] = [entry['check'] for entry in failed]
    report['passed'] = not failed
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
