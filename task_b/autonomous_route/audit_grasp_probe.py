"""Independent CPU counterexamples for the bounded Task B grasp probe.

All images, detections, jaw obstructions and actuator responses here are
SYNTHETIC. The real controllers and FK execute, but there is no physics,
contact, object pose or reward. Passing does not demonstrate a grasp, score,
delivery or success rate. Real grasp evidence belongs to the original task.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.audit_first_reach import fixture
from task_b.arm_kinematics import ARM_JOINT_NAMES, fk, gripper_targets, pinch_position, solve_ik
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.first_reach import FirstReachPolicy
from task_b.grasp_probe import GraspProbePolicy
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

DT = .02
TOL = 2e-7  # Returned actions use float32.
# Independent specification for this combined candidate; do not inherit these
# expectations from the production module under test.
EXPECTED_PRELOAD_M = .075
EXPECTED_TETHER_RAD = np.array([.10, .18, .10, .10, .10, .10])
EXPECTED_FEEDBACK_GAIN = 3.
EXPECTED_FEEDBACK_CAP_RAD = .12
EXPECTED_FILTER_TAU_S = .10
EXPECTED_CLOSE_MIN_S = 2.4
EXPECTED_CLOSE_MAX_S = 4.
EXPECTED_TOTAL_S = 12.
EXPECTED_PAIRED_DELTA_RAD = .20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    schema, names, defaults = fixture()
    # Exercise offset and name resolution, not merely a zero-default identity
    # mapping: articulation/observation and arm action orders differ.
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .01, -.012])))
    terms = tuple(replace(term, joint_names=term.joint_names[::-1])
                  if term.name == ARM_TERM else term for term in schema.terms)
    schema = replace(schema, terms=terms,
                     default_joint_pos=np.array([defaults[name] for name in names]))
    arm, leg, wheel = (schema.term(name) for name in (ARM_TERM, LEG_TERM, WHEEL_TERM))
    checks, evidence = {}, {}
    point = np.array([.5, .179, -.28])
    fit = solve_ik(point+[0., 0., .12], seed=[0., 1.1, -.8, 0., .95, 0.], max_nfev=120)
    parameters = dict(settle_calls=0, ramp_calls=1, forward_cmd=.05,
                      turn_cap=.6, standoff=.50, turn_gain=2., lowering_m=.02)
    images = {key: value for source in ('ee', 'head') for key, value in (
        (source+'_rgb', np.zeros((24, 32, 3), np.uint8)),
        (source+'_depth', np.ones((24, 32))))}

    def detector(*unused, **kwargs):
        return [{'body_point': point.tolist(), 'height_span_m': .18,
                 'forward_planar_distance_m': float(np.linalg.norm(point[:2])),
                 'optical_depth_m': 1., 'depth_samples': 100, 'pixel_area': 100,
                 'pixel_uv': [16., 12.], 'bbox_xywh': [11, 2, 10, 20],
                 'aspect': 2., 'kind': 'synthetic_supported_yellow_component'}], {
                     'yellow_components': 1, 'rejected': {},
                     'association_mode': 'synthetic_detector_output'}

    def physical(action):
        by_name = {name: defaults[name]+float(value)*arm.scale
                   for name, value in zip(arm.joint_names, action[arm.start:arm.stop])}
        return np.array([by_name[name] for name in ARM_JOINT_NAMES])

    class PositionPlant:
        """One-step fake named-q actuator; jaw obstruction is not a contact model."""
        def __init__(self):
            self.q = dict(defaults)
            self.qdot = dict.fromkeys(names, 0.)
            self.last_action = np.zeros(schema.total_dim)

        def observe(self):
            obs = np.zeros(84)
            obs[11] = -1.
            for index, name in enumerate(names):
                obs[12+index] = self.q[name]-defaults[name]
                obs[36+index] = self.qdot[name]
            return obs

        def respond(self, action, *, jaw_half=None, freeze_arm=False, arm_bias=None, response_fraction=1.):
            old = dict(self.q)
            for term in (arm, leg):
                for name, value in zip(term.joint_names, action[term.start:term.stop]):
                    if not (freeze_arm and name in ARM_JOINT_NAMES[:6]):
                        target = defaults[name]+float(value)*term.scale
                        if arm_bias is not None and name in ARM_JOINT_NAMES[:6]:
                            index = ARM_JOINT_NAMES.index(name)
                            target = np.clip(target+arm_bias[index], JOINT_LOWER[index], JOINT_UPPER[index])
                            target = old[name]+response_fraction*(target-old[name])
                        self.q[name] = float(target)
            if jaw_half is not None:
                self.q['arm_joint7'] = float(np.clip(self.q['arm_joint7'], jaw_half, .035))
                self.q['arm_joint8'] = float(np.clip(self.q['arm_joint8'], -.035, -jaw_half))
            self.qdot = {name: (self.q[name]-old[name])/DT for name in names}
            self.last_action = np.asarray(action).copy()

        def step(self, policy, *, tweak=None, jaw_half=.02, freeze_arm=False, camera=None,
                 arm_bias=None, response_fraction=1.):
            obs = self.observe()
            if tweak is not None:
                tweak(obs)
            action = policy.act(obs, images if camera is None else camera)
            self.respond(action, jaw_half=jaw_half, freeze_arm=freeze_arm,
                         arm_bias=arm_bias, response_fraction=response_fraction)
            return np.asarray(action), obs

    def new(**override):
        return GraspProbePolicy(schema, names, defaults, **dict(parameters, **override)), PositionPlant()

    def run_until(policy, plant, wanted=None, limit=550, **kwargs):
        count = 0
        while not policy.done_reason and policy.state != wanted and count < limit:
            plant.step(policy, **kwargs)
            count += 1
        return count

    def check(name, value, detail=None):
        checks[name] = bool(value)
        if detail is not None:
            evidence[name] = detail

    def case(name, callback):
        try:
            callback()
        except Exception as error:
            check(name, False, {'error': type(error).__name__, 'message': str(error)})

    for override in ({'reach_only': True}, {'lowering_m': 0.}, {'lowering_m': .025},
                     {'lowering_m': .031}, {'lowering_m': np.nan}, {'lowering_m': np.inf}):
        try:
            new(**override)
        except ValueError:
            check('rejects_'+repr(override), True)
        else:
            check('rejects_'+repr(override), False)

    def parameter_contract():
        q, _ = new()
        constants = q.describe()['candidate_constants']
        lift = constants['lift']
        check('candidate_metadata_states_direct_preload_and_all_final_limits',
              constants['finger_close_target']['physical_q7_m'] == -EXPECTED_PRELOAD_M
              and constants['finger_close_target']['physical_q8_m'] == EXPECTED_PRELOAD_M
              and constants['close']['min_active_s'] == EXPECTED_CLOSE_MIN_S
              and constants['close']['max_s'] == EXPECTED_CLOSE_MAX_S
              and constants['total_probe_s'] == EXPECTED_TOTAL_S
              and lift['q2_position_feedback_gain'] == EXPECTED_FEEDBACK_GAIN
              and lift['q2_raw_correction_cap_rad'] == EXPECTED_FEEDBACK_CAP_RAD
              and lift['q2_correction_filter_tau_s'] == EXPECTED_FILTER_TAU_S
              and lift['q2_tether_rad'] == .18 and lift['tether_rad'] == .10
              and lift['arrival_qdot_cap_rad_per_s'] == .12
              and lift['arrival_position_window_s'] == .5
              and lift['arrival_position_window_span_rad'] == .002
              and lift['arrival_error_rad'] == .04 and lift['arrival_rise_m'] == .015
              and lift['arrival_window_s'] == .3 and lift['max_s'] == 5.
              and constants['observe']['window_s'] == 2.)
        try:
            gripper_targets(0., preload_m=EXPECTED_PRELOAD_M)
        except ValueError:
            check('shared_gripper_helper_still_rejects_new_private_preload', True)
        else:
            check('shared_gripper_helper_still_rejects_new_private_preload', False)
        # No need to run a different-frequency plant: both endpoint times must
        # cover at least the declared window even for a nonintegral .5/dt.
        q40, _ = new(dt=.04)
        required = q40.describe()['probe_state']['lift_position_window_required_samples'] if 'lift_position_window_required_samples' in q40.describe()['probe_state'] else q40.still_samples
        check('position_window_duration_never_rounds_below_half_second_at_other_dt',
              (required-1)*.04 >= .5 and required == 14,
              {'dt_s': .04, 'required_samples': required, 'endpoint_span_s': (required-1)*.04})
    case('candidate_parameter_contract_execution', parameter_contract)

    with patch('task_b.first_reach.detect_yellow_candidates', detector), \
            patch('task_b.visual_approach.detect_yellow_candidates', detector), \
            patch('task_b.first_reach.solve_ik', return_value=fit):
        p, plant = new()
        original = FirstReachPolicy(schema, names, defaults, **parameters)
        old_act, calls = FirstReachPolicy.act, []

        def count_child(self, *positional, **kw):
            calls.append(id(self))
            return old_act(self, *positional, **kw)

        max_error, once_per_step = 0., True
        with patch.object(FirstReachPolicy, 'act', count_child):
            for step in range(1400):
                obs = plant.observe()
                original.pause_for_stance = p.pause_for_stance = step < 4
                expected = original.act(obs.copy(), images)
                calls.clear()
                action = p.act(obs.copy(), images)
                once_per_step &= len(calls) == 1
                max_error = max(max_error, float(np.max(np.abs(action-expected))))
                plant.respond(action)
                if original.done_reason:
                    break
        check('entire_first_reach_action_prefix_exact',
              max_error == 0. and original.done_reason == 'reach_lowering_hold_complete',
              {'calls': step+1, 'max_action_difference': max_error, 'original_reason': original.done_reason})
        check('exactly_one_child_call_per_prefix_step', once_per_step)
        check('normal_completion_only_enters_close_without_action_jump',
              p.state == 'PROBE_CLOSE' and p.done_reason is None and p.wheel_hold_requested)
        checkpoint = copy.deepcopy((p, plant)) if checks['normal_completion_only_enters_close_without_action_jump'] else None

        for label, tweak in (
                ('nonfinite', lambda obs: obs.__setitem__(0, np.nan)),
                ('tilt', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.3), 0., -np.cos(.3)]))):
            q, world = new()
            ref = FirstReachPolicy(schema, names, defaults, **parameters)
            obs = world.observe(); tweak(obs)
            expected, actual = ref.act(obs.copy(), images), q.act(obs.copy(), images)
            check('prefix_error_'+label+'_propagates_unchanged',
                  ref.done_reason is not None and q.done_reason == ref.done_reason
                  and np.array_equal(actual, expected))

        def alternate_lowering():
            check('wrapper_default_lowering_remains_two_centimetres',
                  inspect.signature(GraspProbePolicy).parameters['lowering_m'].default == .02)
            q, world = new(lowering_m=.03)
            reference = FirstReachPolicy(schema, names, defaults, **dict(parameters, lowering_m=.03))
            exact, once = True, True
            with patch.object(FirstReachPolicy, 'act', count_child):
                for tick in range(1500):
                    obs = world.observe()
                    reference.pause_for_stance = q.pause_for_stance = tick < 4
                    expected = reference.act(obs.copy(), images)
                    calls.clear()
                    action = q.act(obs.copy(), images)
                    once &= len(calls) == 1
                    exact &= np.array_equal(action, expected)
                    world.respond(action)
                    if reference.done_reason:
                        break
            check('three_cm_full_first_reach_prefix_is_exact_and_called_once',
                  exact and once and reference.done_reason == 'reach_lowering_hold_complete',
                  {'calls': tick+1, 'original_reason': reference.done_reason, 'wrapper_reason': q.done_reason})
            check('three_cm_enters_probe_only_after_original_measured_lowering_completion',
                  q.done_reason is None and q.state == 'PROBE_CLOSE'
                  and q.debug['lower_authorized'] and q.debug['probe_lowering_alpha'] == 1.
                  and q.debug['lowering_m'] == .03 and q.wheel_hold_requested)
            description = q.describe()
            evidence['three_cm_metadata'] = {
                'required_first_reach': description['candidate_constants']['required_first_reach'],
                'probe_state': description.get('probe_state'),
                'child_lowering_m': description['first_reach']['lowering_m']}
            check('three_cm_actual_lowering_is_reported_by_child_and_live_diagnostics',
                  description['first_reach']['lowering_m'] == .03 and q.debug['lowering_m'] == .03
                  and description['first_reach_lowering_m'] == .03
                  and q.debug['probe_first_reach_lowering_m'] == .03)
            required = description['candidate_constants']['required_first_reach']
            check('metadata_separates_default_allowed_and_actual_lowering',
                  required['default_lowering_m'] == .02
                  and list(required['allowed_lowering_m']) == [.02, .03]
                  and description['first_reach_lowering_m'] == .03)
            assert checkpoint is not None
            held = action[leg.start:leg.stop].copy()
            check('three_cm_uses_original_scaled_leg_reference_instead_of_two_cm_pose',
                  np.allclose(held, 1.5*checkpoint[1].last_action[leg.start:leg.stop], atol=TOL, rtol=0.))
            three_checkpoint = copy.deepcopy((q, world))
            fixed = True
            for _ in range(430):
                action, _ = world.step(q)
                fixed &= (q.wheel_hold_requested and q.debug['probe_lowering_alpha'] == 1.
                          and np.array_equal(action[leg.start:leg.stop], held))
                if q.done_reason:
                    break
            check('three_cm_probe_keeps_achieved_low_pose_and_wheel_hold_through_observation',
                  fixed and q.done_reason == 'grasp_probe_observation_complete')
            q, world = copy.deepcopy(three_checkpoint)
            action, _ = world.step(q, tweak=lambda obs: obs.__setitem__(slice(9, 12),
                [np.sin(.13), 0., -np.cos(.13)]))
            check('three_cm_does_not_bypass_post_reach_instability_guard',
                  q.done_reason == 'reach_unstable_posture' and q.wheel_hold_requested
                  and np.array_equal(action[leg.start:leg.stop], held))
            for label, tweak in (
                    ('nonfinite', lambda obs: obs.__setitem__(0, np.nan)),
                    ('tilt', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.3), 0., -np.cos(.3)]))):
                q, world = new(lowering_m=.03)
                reference = FirstReachPolicy(schema, names, defaults, **dict(parameters, lowering_m=.03))
                obs = world.observe(); tweak(obs)
                expected, action = reference.act(obs.copy(), images), q.act(obs.copy(), images)
                check('three_cm_prefix_error_'+label+'_propagates_unchanged',
                      reference.done_reason is not None and q.done_reason == reference.done_reason
                      and np.array_equal(action, expected))
        case('alternate_lowering_execution', alternate_lowering)

        if checkpoint is not None:
            lowered = checkpoint[1].last_action[leg.start:leg.stop].copy()

            def hold_ok(q, action):
                return (q.wheel_hold_requested and np.array_equal(action[leg.start:leg.stop], lowered)
                        and np.max(np.abs(action[wheel.start:wheel.stop])) == 0.)

            def close_and_lift():
                q, world = copy.deepcopy(checkpoint)
                held = physical(world.last_action)
                close_deltas, lift_deltas, tethers = [], [], []
                fixed_hold, fixed_six, lift_start, observe_start = True, True, None, None
                lift_checkpoint = observe_checkpoint = None
                for tick in range(510):
                    before = physical(world.last_action)
                    state_before = q.state
                    action, obs = world.step(q)
                    command = physical(action)
                    actual = np.array([obs[12+names.index(n)]+defaults[n] for n in ARM_JOINT_NAMES])
                    fixed_hold &= hold_ok(q, action)
                    if state_before == 'PROBE_CLOSE':
                        close_deltas.append(float(np.max(np.abs(command[6:]-before[6:]))))
                        fixed_six &= np.allclose(command[:6], held[:6], atol=TOL, rtol=0.)
                    if q.state == 'PROBE_LIFT' and lift_start is None:
                        lift_start = tick+1
                        lift_checkpoint = copy.deepcopy((q, world))
                    if state_before == 'PROBE_LIFT':
                        lift_deltas.append(float(np.max(np.abs(command[:6]-before[:6]))))
                        tethers.append(float(np.max(np.abs(command[:6]-actual[:6])-EXPECTED_TETHER_RAD)))
                    if q.state == 'PROBE_OBSERVE' and observe_start is None:
                        observe_start = tick+1
                        observe_checkpoint = copy.deepcopy((q, world))
                    if q.done_reason:
                        break
                check('close_slew_and_six_joint_hold', fixed_six and max(close_deltas) <= .05*DT+TOL)
                check('preload_maps_to_physical_targets_with_nonzero_defaults_and_permuted_actions',
                      np.allclose(physical(world.last_action)[6:], [-EXPECTED_PRELOAD_M, EXPECTED_PRELOAD_M],
                                  atol=TOL, rtol=0.))
                check('stable_repeated_measured_width_authorizes_only_after_active_minimum',
                      lift_start is not None and lift_start*DT >= EXPECTED_CLOSE_MIN_S-1e-9,
                      {'lift_entry_s': None if lift_start is None else lift_start*DT})
                check('lift_rate_and_measured_tether', bool(lift_deltas) and max(lift_deltas) <= .10*DT+TOL
                      and max(tethers) <= TOL)
                check('lower_reference_and_wheel_anchor_request_continuous', fixed_hold)
                check('synthetic_response_ends_as_observation_only',
                      q.done_reason == 'grasp_probe_observation_complete' and observe_start is not None
                      and (tick+1-observe_start)*DT >= 2.-DT-1e-9,
                      {'reason': q.done_reason, 'probe_elapsed_s': (tick+1)*DT,
                       'notice': 'Synthetic actuator and artificial jaw obstruction, NOT a grasp.'})
                evidence['happy_path_measurements'] = {
                    key: value for key, value in q.debug.items()
                    if key in ('probe_lift_latched_q_rad', 'probe_lift_goal_rad',
                               'probe_lift_limit_violations', 'probe_pinch_rise_m',
                               'probe_lift_max_q_error_rad', 'probe_lift_max_qdot_rad_s')}
                checkpoints.update(lift=lift_checkpoint, observe=observe_checkpoint)

            checkpoints = {}
            case('synthetic_close_lift_execution', close_and_lift)

            def empty_close():
                q, world = copy.deepcopy(checkpoint)
                elapsed = run_until(q, world, jaw_half=0.)*DT
                check('actual_empty_jaws_stop_without_lift', q.done_reason == 'grasp_probe_empty_close'
                      and elapsed <= EXPECTED_CLOSE_MAX_S+DT, {'reason': q.done_reason, 'elapsed_s': elapsed})
            case('empty_close_execution', empty_close)

            def unstable_close():
                q, world = copy.deepcopy(checkpoint)
                for tick in range(215):
                    half = .014 if tick % 2 else .024
                    world.q.update(arm_joint7=half, arm_joint8=-half)
                    world.step(q, jaw_half=half)
                    if q.done_reason or q.state != 'PROBE_CLOSE':
                        break
                check('varying_measured_width_cannot_authorize_lift_and_close_deadline_is_finite',
                      q.done_reason == 'grasp_probe_close_not_confirmed' and (tick+1)*DT <= EXPECTED_CLOSE_MAX_S+DT,
                      {'reason': q.done_reason, 'elapsed_s': (tick+1)*DT})
            case('unstable_close_execution', unstable_close)

            def target_command_prerequisite():
                q, world = copy.deepcopy(checkpoint)
                previous = physical(world.last_action)
                slew_ok, premature_lift = True, False
                # Slow only this CPU instance's slew to .02 m/s. A stable fake
                # obstruction at 40mm must not authorize lift at 2.4s while the
                # full target has not been issued. The 4s phase deadline
                # must stop this deliberately slow
                # counterexample without jumping fingers to the final target.
                with patch('task_b.grasp_probe.FINGER_SLEW_M_S', .02):
                    for tick in range(220):
                        action, _ = world.step(q)
                        current = physical(action)
                        slew_ok &= bool(np.max(np.abs(current[6:]-previous[6:])) <= .02*DT+TOL)
                        premature_lift |= q.state in ('PROBE_LIFT', 'PROBE_OBSERVE')
                        previous = current
                        if tick == 119:
                            check('stable_width_at_close_minimum_cannot_skip_unissued_preload',
                                  q.state == 'PROBE_CLOSE' and q.done_reason is None
                                  and not q.debug['probe_finger_target_commanded'])
                        if q.done_reason:
                            break
                check('slow_finger_target_cannot_jump_at_lift_or_extend_close_deadline',
                      not premature_lift and slew_ok and q.done_reason == 'grasp_probe_close_not_confirmed'
                      and (tick+1)*DT <= EXPECTED_CLOSE_MAX_S+DT,
                      {'reason': q.done_reason, 'elapsed_s': (tick+1)*DT,
                       'final_physical_finger_targets_m': previous[6:].tolist()})
            case('target_command_prerequisite_execution', target_command_prerequisite)

            def measured_latch_and_limits():
                q, world = copy.deepcopy(checkpoint)
                # With a .20 paired delta, a hard-legal goal above q2's soft
                # upper bound cannot start below the hard upper bound. Exercise
                # the equally valid lower gap: .25-.20=.05 < soft lower .157.
                world.q['arm_joint2'] = .25
                run_until(q, world, wanted='PROBE_LIFT', freeze_arm=True, limit=151)
                goal = q.debug.get('probe_lift_goal_rad')
                check('hard_legal_q2_goal_outside_soft_window_is_not_rejected',
                      q.state == 'PROBE_LIFT' and q.done_reason is None and goal is not None
                      and abs(goal[1]-(.25-EXPECTED_PAIRED_DELTA_RAD)) < TOL,
                      {'state': q.state, 'reason': q.done_reason, 'goal': goal})
                q, world = copy.deepcopy(checkpoint)
                # Actual measured yaw differs from held command; q2/q3 form
                # the bounded paired goal, while measured yaw must be retained.
                world.q['arm_joint1'] += .03
                measured = np.array([world.q[n] for n in ARM_JOINT_NAMES[:6]])
                run_until(q, world, wanted='PROBE_LIFT', freeze_arm=True, limit=151)
                entered = q.state == 'PROBE_LIFT'
                run_until(q, world, limit=150)
                expected = measured+np.array([0., -EXPECTED_PAIRED_DELTA_RAD,
                                              EXPECTED_PAIRED_DELTA_RAD, 0., 0., 0.])
                check('lift_goal_uses_latched_measured_q_and_exact_paired_delta',
                      entered and np.allclose(q.debug['probe_lift_goal_rad'], expected, atol=TOL, rtol=0.))
                q, world = copy.deepcopy(checkpoint)
                world.q['arm_joint2'] = .05  # q2-.20 violates static hard lower bound 0.
                elapsed = run_until(q, world, limit=155, freeze_arm=True)*DT
                check('out_of_limits_paired_goal_stops_before_moving_or_clipping',
                      q.done_reason is not None and q.state != 'PROBE_LIFT'
                      and np.allclose(physical(world.last_action)[:6], physical(checkpoint[1].last_action)[:6], atol=TOL),
                      {'reason': q.done_reason, 'elapsed_s': elapsed})
            case('measured_latch_and_limits_execution', measured_latch_and_limits)

            def lift_counterexamples():
                assert checkpoints.get('lift') is not None, 'No real wrapper LIFT checkpoint'
                q, world = copy.deepcopy(checkpoints['lift'])
                elapsed = run_until(q, world, freeze_arm=True, limit=265)*DT
                check('frozen_actual_arm_cannot_arrive_from_command_and_stops_for_no_progress',
                      q.done_reason == 'no_joint_lift_progress' and elapsed <= 2.+3*DT,
                      {'reason': q.done_reason, 'elapsed_s': elapsed})
                q, world = copy.deepcopy(checkpoints['lift'])
                def moving(obs):
                    for name in ARM_JOINT_NAMES[:6]:
                        obs[36+names.index(name)] = .13
                elapsed = run_until(q, world, tweak=moving, limit=265)*DT
                check('actual_arm_velocity_blocks_arrival_until_original_lift_deadline',
                      q.done_reason == 'grasp_probe_lift_not_reached' and elapsed <= 5.+DT,
                      {'reason': q.done_reason, 'elapsed_s': elapsed})
                # The new .20 motion rises much farther at the main fixture;
                # merely perturbing its goal by <.04 no longer isolates the FK
                # gate. Drive a second complete FirstReach on a synthetic high
                # target whose real paired FK instead moves slightly downward.
                # No controller state, FK function or goal is patched.
                saved_point = point.copy()
                alternate_q = np.array([.54, 1.5, -2., 0., 0., 0.])
                alternate_fit = solve_ik(fk(alternate_q)[:3, 3], seed=alternate_q, multi_start=False)
                try:
                    point[:] = fk(alternate_q)[:3, 3]-[0., 0., .12]
                    q, world = new()
                    with patch('task_b.first_reach.solve_ik', return_value=alternate_fit):
                        run_until(q, world, wanted='PROBE_LIFT', limit=1600)
                    assert q.state == 'PROBE_LIFT', q.done_reason
                    start = np.array(q.debug['probe_lift_latched_q_rad'])
                    goal = np.array(q.debug['probe_lift_goal_rad'])
                    elapsed = run_until(q, world, limit=265)*DT
                    actual = np.array([world.q[n] for n in ARM_JOINT_NAMES[:6]])
                    rise = float((pinch_position(actual)-pinch_position(start))[2])
                finally:
                    point[:] = saved_point
                check('near_goal_quiet_actual_joints_without_fk_rise_cannot_authorize_observation',
                      q.done_reason == 'grasp_probe_lift_not_reached' and elapsed <= 5.+DT
                      and rise < .015 and np.max(np.abs(actual-goal)) < .04,
                      {'reason': q.done_reason, 'independent_measured_fk_rise_m': rise,
                       'actual_max_goal_error_rad': float(np.max(np.abs(actual-goal)))})
                for phase in ('lift', 'observe'):
                    assert checkpoints.get(phase) is not None
                    q, world = copy.deepcopy(checkpoints[phase])
                    elapsed = run_until(q, world, jaw_half=0., limit=25)*DT
                    expected = ('grasp_probe_empty_during_lift' if phase == 'lift'
                                else 'grasp_probe_empty_during_observation')
                    check('empty_width_stops_'+phase, q.done_reason == expected
                          and elapsed <= .2+3*DT, {'reason': q.done_reason, 'elapsed_s': elapsed})
                q, world = copy.deepcopy(checkpoints['lift'])
                before = world.last_action.copy()
                world.q['arm_joint1'] += .25
                action, _ = world.step(q, freeze_arm=True)
                jump = float(np.max(np.abs(physical(action)[:6]-physical(before)[:6])))
                check('incompatible_speed_and_measured_tether_stops_without_command_jump',
                      q.done_reason is not None and jump <= .10*DT+TOL,
                      {'reason': q.done_reason, 'maximum_command_change_rad': jump})
            case('lift_counterexamples_execution', lift_counterexamples)

            def biased_response():
                # A deliberately simple first-order fake actuator uses the
                # measured mean actual-minus-command of grasp run 01's final
                # second. This is a hypothesis, not replayed or simulated contact.
                bias = np.array([-.0066094446, .0606982279, -.0028718376,
                                 .0008871022, -.0037050359, .0006411922])
                outcomes = {}
                for gain in (0., EXPECTED_FEEDBACK_GAIN):
                    q, world = copy.deepcopy(checkpoint)
                    goal = None
                    limits_ok = goal_fixed = others_uncompensated = posture_fixed = True
                    correction_ok, corrections = True, []
                    peak_slew = peak_tether = 0.
                    tether_ok = True
                    previous_filtered = 0.
                    observe_filtered_samples = 0
                    # Explicit CPU-only gain=0 ablation. Never edits the module
                    # file, external state or a running simulator process.
                    with patch('task_b.grasp_probe.LIFT_Q2_FEEDBACK_GAIN', gain):
                        for tick in range(510):
                            previous = physical(world.last_action)
                            before_phase = q.state
                            action, obs = world.step(q, arm_bias=bias, response_fraction=.25)
                            command = physical(action)
                            actual = np.array([obs[12+names.index(n)]+defaults[n] for n in ARM_JOINT_NAMES])
                            posture_fixed &= hold_ok(q, action)
                            now_goal = q.debug.get('probe_lift_goal_rad')
                            if now_goal is not None:
                                if goal is None:
                                    goal = np.asarray(now_goal).copy()
                                goal_fixed &= np.array_equal(goal, now_goal)
                            if before_phase in ('PROBE_LIFT', 'PROBE_OBSERVE'):
                                peak_slew = max(peak_slew, float(np.max(np.abs(command[:6]-previous[:6]))))
                                peak_tether = max(peak_tether, float(np.max(np.abs(command[:6]-actual[:6]))))
                                tether_ok &= bool(np.all(np.abs(command[:6]-actual[:6]) <= EXPECTED_TETHER_RAD+TOL))
                                limits_ok &= bool(np.all(command[:6] >= JOINT_LOWER-TOL)
                                                  and np.all(command[:6] <= JOINT_UPPER+TOL))
                                other = [0, 2, 3, 4, 5]
                                # Non-q2 axes progress between their last command
                                # and the fixed goal; feedback must not drive an
                                # additional command past that goal on those axes.
                                others_uncompensated &= bool(np.all(command[other] >= np.minimum(previous[other], goal[other])-TOL)
                                    and np.all(command[other] <= np.maximum(previous[other], goal[other])+TOL))
                                if not q.done_reason:
                                    correction = q.debug.get('probe_q2_position_feedback_rad')
                                    raw = float(np.clip(gain*(goal[1]-actual[1]),
                                                        -EXPECTED_FEEDBACK_CAP_RAD, EXPECTED_FEEDBACK_CAP_RAD))
                                    expected_filtered = previous_filtered+(1.-np.exp(-DT/EXPECTED_FILTER_TAU_S))*(raw-previous_filtered)
                                    previous_filtered = expected_filtered
                                    correction_ok &= (correction is not None and np.isfinite(correction)
                                        and abs(correction) <= EXPECTED_FEEDBACK_CAP_RAD+TOL
                                        and abs(correction-expected_filtered) <= TOL
                                        and abs(q.debug['probe_q2_raw_correction_rad']-raw) <= TOL
                                        and q.debug['probe_q2_filtered_correction_rad'] == correction)
                                    if before_phase == 'PROBE_OBSERVE':
                                        observe_filtered_samples += 1
                                    if correction is not None:
                                        corrections.append(float(correction))
                            if q.done_reason:
                                break
                    outcomes[str(gain)] = {
                        'reason': q.done_reason, 'elapsed_s': (tick+1)*DT,
                        'actual_max_goal_error_rad': q.debug.get('probe_lift_max_q_error_rad'),
                        'actual_pinch_rise_m': q.debug.get('probe_pinch_rise_m'),
                        'goal_rad': None if goal is None else goal.tolist(),
                        'last_command_rad': physical(world.last_action)[:6].tolist(),
                        'peak_command_change_rad': peak_slew, 'peak_measured_tether_rad': peak_tether,
                        'final_actual_max_qdot_rad_s': q.debug.get('probe_lift_max_qdot_rad_s'),
                        'correction_range_rad': [min(corrections), max(corrections)] if corrections else None}
                    check('biased_response_bounds_and_fixed_goal_gain_'+str(gain),
                          limits_ok and goal_fixed and posture_fixed and others_uncompensated
                          and peak_slew <= .10*DT+TOL and tether_ok)
                    check('q2_feedback_diagnostic_uses_public_error_and_respects_cap_gain_'+str(gain),
                          correction_ok and bool(corrections))
                    if gain == EXPECTED_FEEDBACK_GAIN:
                        check('feedback_filter_is_not_reset_or_frozen_across_lift_to_observe',
                              correction_ok and observe_filtered_samples >= 50)
                check('uncompensated_static_bias_cannot_pass_actual_lift_requirements',
                      outcomes['0.0']['reason'] in ('no_joint_lift_progress', 'grasp_probe_lift_not_reached')
                      and outcomes['0.0']['actual_max_goal_error_rad'] > .04)
                check('q2_feedback_handles_synthetic_grasp01_bias_without_changing_actual_acceptance',
                      outcomes[str(EXPECTED_FEEDBACK_GAIN)]['reason'] == 'grasp_probe_observation_complete'
                      and outcomes[str(EXPECTED_FEEDBACK_GAIN)]['actual_max_goal_error_rad'] < .04
                      and outcomes[str(EXPECTED_FEEDBACK_GAIN)]['actual_pinch_rise_m'] >= .015)
                evidence['synthetic_bias_comparison'] = {
                    'basis': 'first-order CPU response fraction .25 per .02 s, constant bias from '
                             'plan_p4_grasp_seed42_01 final-second mean; no dynamics/contact/grasp claim',
                    'bias_rad': bias.tolist(), 'results': outcomes}
                # An excessive q2 bias reaches the real upper stop. Saturated
                # compensation must not move the measured goal or fake arrival.
                q, world = copy.deepcopy(checkpoint)
                large = np.zeros(6); large[1] = .20
                run_until(q, world, limit=450, arm_bias=large, response_fraction=.25)
                check('excessive_bias_cannot_pass_by_compensation_or_hard_limit_saturation',
                      q.done_reason in ('no_joint_lift_progress', 'grasp_probe_lift_not_reached')
                      and q.debug['probe_lift_max_q_error_rad'] > .04,
                      {'reason': q.done_reason, 'actual_error_rad': q.debug['probe_lift_max_q_error_rad']})
                q, world = copy.deepcopy(checkpoint)
                reverse = np.zeros(6); reverse[1] = -.06
                run_until(q, world, limit=450, arm_bias=reverse, response_fraction=.25)
                check('negative_q2_bias_produces_positive_bounded_feedback',
                      q.done_reason == 'grasp_probe_observation_complete'
                      and 0. < q.debug['probe_q2_position_feedback_rad'] <= EXPECTED_FEEDBACK_CAP_RAD
                      and q.debug['probe_lift_max_q_error_rad'] < .04,
                      {'reason': q.done_reason, 'correction_rad': q.debug['probe_q2_position_feedback_rad'],
                       'actual_error_rad': q.debug['probe_lift_max_q_error_rad']})
                assert checkpoints.get('lift') is not None
                q, world = copy.deepcopy(checkpoints['lift'])
                world.step(q)
                before = world.last_action.copy()
                for tick in range(10):
                    world.q['arm_joint2'] -= .001
                    action, _ = world.step(q, freeze_arm=True,
                                           tweak=lambda obs: obs.__setitem__(3, .13))
                check('paused_feedback_cannot_change_commands_when_public_actual_q_changes',
                      q.done_reason is None and np.array_equal(before, action)
                      and hold_ok(q, action))
            case('biased_response_execution', biased_response)

            def vector_tether_and_filter_pause():
                q, world = copy.deepcopy(checkpoints['lift'])
                initial_actual = np.array([world.q[n] for n in ARM_JOINT_NAMES[:6]])
                largest = np.zeros(6)
                bounded = True
                for _ in range(130):
                    before = physical(world.last_action)
                    action, _ = world.step(q, freeze_arm=True)
                    command = physical(action)
                    largest = np.maximum(largest, np.abs(command[:6]-initial_actual))
                    bounded &= (np.all(np.abs(command[:6]-initial_actual) <= EXPECTED_TETHER_RAD+TOL)
                                and np.max(np.abs(command[:6]-before[:6])) <= .10*DT+TOL)
                    if q.done_reason:
                        break
                check('only_q2_can_use_extended_tether_without_loosening_other_axes',
                      bounded and abs(largest[1]-.18) <= TOL and abs(largest[2]-.10) <= TOL
                      and q.done_reason == 'no_joint_lift_progress',
                      {'largest_actual_relative_command_rad': largest.tolist(), 'reason': q.done_reason})
                q, world = copy.deepcopy(checkpoints['lift'])
                previous = physical(world.last_action)
                world.q['arm_joint2'] += .20
                action, _ = world.step(q, freeze_arm=True)
                check('q2_empty_rate_tether_intersection_stops_with_unchanged_command',
                      q.done_reason is not None and np.allclose(physical(action), previous, atol=TOL, rtol=0.))
                q, world = copy.deepcopy(checkpoints['lift'])
                world.step(q)
                held = world.last_action.copy()
                filtered = q.debug['probe_q2_position_feedback_rad']
                frozen = filtered != 0.
                for _ in range(10):
                    world.q['arm_joint2'] -= .0005
                    action, _ = world.step(q, freeze_arm=True,
                                           tweak=lambda obs: obs.__setitem__(3, .13))
                    frozen &= (q.debug['probe_q2_position_feedback_rad'] == filtered
                               and np.array_equal(action, held))
                for _ in range(9):
                    action, _ = world.step(q, freeze_arm=True)
                    frozen &= (q.debug['probe_q2_position_feedback_rad'] == filtered
                               and np.array_equal(action, held))
                goal = q.debug['probe_lift_goal_rad'][1]
                actual = world.q['arm_joint2']
                raw = float(np.clip(EXPECTED_FEEDBACK_GAIN*(goal-actual),
                                    -EXPECTED_FEEDBACK_CAP_RAD, EXPECTED_FEEDBACK_CAP_RAD))
                expected = filtered+(1.-np.exp(-DT/EXPECTED_FILTER_TAU_S))*(raw-filtered)
                action, _ = world.step(q, freeze_arm=True)
                check('pause_freezes_filter_history_and_first_resumed_tick_updates_exactly_once',
                      frozen and q.done_reason is None
                      and abs(q.debug['probe_q2_position_feedback_rad']-expected) <= TOL,
                      {'before_pause_rad': filtered, 'expected_resumed_rad': expected,
                       'actual_resumed_rad': q.debug['probe_q2_position_feedback_rad']})
            case('vector_tether_and_filter_pause_execution', vector_tether_and_filter_pause)

            def loaded_public_position_response():
                # Static torque/K converted to a first-order CPU position bias.
                # These are actuator hypotheses, never force/contact simulation.
                results = {}
                for load, gain in ((.12704, 2.), (.12704, 3.), (.155, 3.)):
                    q, world = copy.deepcopy(checkpoint)
                    bias = np.zeros(6); bias[1] = load
                    history, peak_slew, bounded = [], 0., True
                    first_goal = None
                    with patch('task_b.grasp_probe.LIFT_Q2_FEEDBACK_GAIN', gain):
                        for tick in range(600):
                            before = physical(world.last_action)
                            before_phase = q.state
                            action, obs = world.step(q, arm_bias=bias, response_fraction=.25)
                            actual = np.array([obs[12+names.index(n)]+defaults[n] for n in ARM_JOINT_NAMES[:6]])
                            command = physical(action)[:6]
                            if before_phase in ('PROBE_LIFT', 'PROBE_OBSERVE'):
                                goal = np.asarray(q.debug['probe_lift_goal_rad'])
                                if first_goal is None:
                                    first_goal = goal.copy()
                                peak_slew = max(peak_slew, float(np.max(np.abs(command-before[:6]))))
                                bounded &= (np.array_equal(goal, first_goal)
                                            and np.all(np.abs(command-actual) <= EXPECTED_TETHER_RAD+TOL)
                                            and np.all(command >= JOINT_LOWER-TOL)
                                            and np.all(command <= JOINT_UPPER+TOL))
                                history.append(float(actual[1]-goal[1]))
                            if q.done_reason:
                                break
                    key = 'bias_'+str(load)+'_gain_'+str(gain)
                    results[key] = {'reason': q.done_reason, 'actual_error_rad': q.debug.get('probe_lift_max_q_error_rad'),
                                    'actual_pinch_rise_m': q.debug.get('probe_pinch_rise_m'),
                                    'last_second_q2_error_span_rad': float(np.ptp(history[-50:])) if history else None,
                                    'last_q2_error_rad': history[-1] if history else None,
                                    'static_unsaturated_error_prediction_rad': load/(1.+gain),
                                    'probe_elapsed_s': (tick+1)*DT, 'max_arm_command_step_rad': peak_slew}
                    check('loaded_response_keeps_rate_tether_hardlimits_fixed_goal_'+key,
                          bounded and history and peak_slew <= .10*DT+TOL)
                    check('loaded_response_matches_bounded_static_prediction_without_large_oscillation_'+key,
                          bool(history) and abs(history[-1]-load/(1.+gain)) < .003
                          and np.ptp(history[-50:]) < .003)
                check('gain_two_cannot_claim_arrival_for_point_12704_static_bias',
                      results['bias_0.12704_gain_2.0']['reason'] in ('no_joint_lift_progress', 'grasp_probe_lift_not_reached')
                      and results['bias_0.12704_gain_2.0']['actual_error_rad'] > .04)
                for load in (.12704, .155):
                    result = results['bias_'+str(load)+'_gain_3.0']
                    check('gain_three_loaded_synthetic_response_meets_unchanged_actual_gate_bias_'+str(load),
                          result['reason'] == 'grasp_probe_observation_complete'
                          and result['actual_error_rad'] < .04 and result['actual_pinch_rise_m'] >= .015)
                evidence['synthetic_loaded_response'] = {
                    'notice': 'CPU-only first-order response fraction .25; constant torque/K hypothesis; no contact or grasp proof',
                    'results': results}
            case('loaded_public_position_response_execution', loaded_public_position_response)

            def body_and_pause():
                for label, tweak in (
                        ('tangent_budget', lambda obs: obs.__setitem__(0, .055)),
                        ('yaw_budget', lambda obs: obs.__setitem__(5, .11)),
                        ('gravity_budget', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.06), 0., -np.cos(.06)])),
                        ('hard_tilt', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.3), 0., -np.cos(.3)])),
                        ('reach_tilt', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.13), 0., -np.cos(.13)])),
                        ('reach_angular_rate', lambda obs: obs.__setitem__(3, .46)),
                        ('invalid_public_state', lambda obs: obs.__setitem__(12, np.nan))):
                    q, world = copy.deepcopy(checkpoint)
                    elapsed = run_until(q, world, tweak=tweak, limit=45)*DT
                    check('probe_'+label+'_terminates', q.done_reason is not None,
                          {'reason': q.done_reason, 'elapsed_s': elapsed})
                q, world = copy.deepcopy(checkpoint)
                world.step(q)
                before = world.last_action.copy()
                for _ in range(9):
                    action, _ = world.step(q, tweak=lambda obs: obs.__setitem__(3, .13))
                frozen = np.array_equal(action, before) and hold_ok(q, action)
                for _ in range(9):
                    action, _ = world.step(q)
                check('soft_pause_freezes_commands_and_waits_point_two_seconds_quiet',
                      frozen and np.array_equal(action, before) and q.done_reason is None)
                for _ in range(2):
                    action, _ = world.step(q)
                check('quiet_resumption_restarts_bounded_close',
                      q.done_reason is None and not np.array_equal(action, before))
                q, world = copy.deepcopy(checkpoint)
                elapsed = run_until(q, world, tweak=lambda obs: obs.__setitem__(3, .13), limit=120)*DT
                check('pause_has_finite_two_second_budget', q.done_reason is not None and elapsed <= 2.+2*DT,
                      {'reason': q.done_reason, 'elapsed_s': elapsed})
                q, world = copy.deepcopy(checkpoint)
                for tick in range(110):
                    tweak = (lambda obs: obs.__setitem__(3, .13)) if tick % 35 < 25 else None
                    world.step(q, tweak=tweak)
                    if q.done_reason:
                        break
                check('separate_pause_episodes_share_two_second_cumulative_budget',
                      q.done_reason == 'grasp_probe_pause_budget_exceeded' and (tick+1)*DT <= 2.+3*DT,
                      {'reason': q.done_reason, 'elapsed_s': (tick+1)*DT,
                       'pause_episodes': q.debug.get('probe_pause_episodes')})
                q, world = copy.deepcopy(checkpoint)
                elapsed = run_until(q, world, tweak=lambda obs: obs.__setitem__(0, .065), limit=40)*DT
                check('body_drift_budget_continues_while_commands_are_paused',
                      q.done_reason == 'grasp_probe_base_displacement_budget_exceeded' and elapsed <= .5,
                      {'reason': q.done_reason, 'elapsed_s': elapsed})
                q, world = copy.deepcopy(checkpoint)
                for _ in range(115):
                    world.step(q)
                world.step(q, tweak=lambda obs: obs.__setitem__(3, .13))
                for _ in range(18):
                    world.step(q)
                reset_width = q.state == 'PROBE_CLOSE' and q.done_reason is None
                world.step(q)
                check('pause_discards_pre_pause_stable_width_window',
                      reset_width and q.state == 'PROBE_LIFT' and q.done_reason is None)
                q, world = copy.deepcopy(checkpoint)
                # No qualifying jaw-width window; 3.5s active followed by a
                # quiet-body pause must still hit the original CLOSE deadline.
                for tick in range(215):
                    half = .014 if tick % 2 else .024
                    world.q.update(arm_joint7=half, arm_joint8=-half)
                    tweak = None if tick < 175 else lambda obs: obs.__setitem__(3, .13)
                    world.step(q, jaw_half=half, tweak=tweak)
                    if q.done_reason:
                        break
                check('pause_cannot_reset_close_phase_deadline',
                      q.done_reason == 'grasp_probe_close_not_confirmed' and (tick+1)*DT <= EXPECTED_CLOSE_MAX_S+DT,
                      {'reason': q.done_reason, 'elapsed_s': (tick+1)*DT})
            case('body_and_pause_execution', body_and_pause)

            def total_deadline():
                q, world = copy.deepcopy(checkpoint)
                # Reach the actual CLOSE gate late but inside its 4 s limit.
                for tick in range(200):
                    half = (.014 if tick % 2 else .024) if tick < 185 else .02
                    world.q.update(arm_joint7=half, arm_joint8=-half)
                    world.step(q, jaw_half=half)
                    if q.done_reason or q.state != 'PROBE_CLOSE':
                        break
                assert q.state == 'PROBE_LIFT', q.done_reason
                def moving(obs):
                    for name in ARM_JOINT_NAMES[:6]:
                        obs[36+names.index(name)] = .13
                # 4 s ineligible, then .5 s position window plus .3 s arrival.
                for tick in range(249):
                    world.step(q, tweak=moving if tick < 200 else None)
                    if q.done_reason or q.state != 'PROBE_LIFT':
                        break
                assert q.state == 'PROBE_OBSERVE', q.done_reason
                for _ in range(75):
                    world.step(q)
                for _ in range(10):
                    world.step(q, tweak=lambda obs: obs.__setitem__(3, .13))
                run_until(q, world, limit=160)
                elapsed = q.debug.get('probe_elapsed_s')
                check('observation_pause_resets_its_quiet_window_but_not_total_twelve_second_deadline',
                      q.done_reason == 'grasp_probe_total_budget_exceeded'
                      and elapsed is not None and abs(elapsed-EXPECTED_TOTAL_S) <= DT,
                      {'reason': q.done_reason, 'elapsed_s': elapsed,
                       'pause_total_s': q.debug.get('probe_pause_total_s'),
                       'observation_active_s': q.debug.get('probe_observe_active_s')})
            case('total_deadline_execution', total_deadline)

            def measured_stability_window():
                # Obtain a real LIFT near-goal checkpoint using the public qdot
                # hard gate. No internal phase, goal, FK or clock is overwritten.
                q, world = copy.deepcopy(checkpoints['lift'])
                def velocity(value):
                    def inject(obs):
                        for name in ARM_JOINT_NAMES[:6]:
                            obs[36+names.index(name)] = value
                    return inject
                for _ in range(180):
                    world.step(q, tweak=velocity(.13))
                    if q.debug.get('probe_lift_max_q_error_rad', 1.) < .008:
                        break
                assert q.state == 'PROBE_LIFT' and q.done_reason is None
                ready_checkpoint = copy.deepcopy((q, world))
                for rate in (.06, .12):
                    q, world = copy.deepcopy(ready_checkpoint)
                    premature = False
                    for tick in range(45):
                        world.step(q, freeze_arm=True, tweak=velocity(rate))
                        premature |= tick < 39 and q.state == 'PROBE_OBSERVE'
                        if q.done_reason or q.state == 'PROBE_OBSERVE':
                            break
                    check('complete_half_second_q_window_then_arrival_accepts_stable_q_at_rate_'+str(rate),
                          not premature and q.state == 'PROBE_OBSERVE' and not q.done_reason
                          and tick+1 >= 40 and tick+1 <= 41,
                          {'samples_until_observe': tick+1, 'reported_qdot_rad_s': rate,
                           'elapsed_from_first_sample_s': tick*DT})
                q, world = copy.deepcopy(ready_checkpoint)
                elapsed = run_until(q, world, freeze_arm=True, tweak=velocity(.120001), limit=270)*DT
                check('stable_q_cannot_bypass_hard_measured_qdot_cap',
                      q.done_reason == 'grasp_probe_lift_not_reached',
                      {'reason': q.done_reason, 'remaining_elapsed_s': elapsed})
                q, world = copy.deepcopy(ready_checkpoint)
                fixed_yaw = world.q['arm_joint1']
                for tick in range(270):
                    world.q['arm_joint1'] = fixed_yaw+(.00151 if tick % 2 else -.00151)
                    world.step(q, freeze_arm=True, tweak=velocity(0.))
                    if q.done_reason or q.state == 'PROBE_OBSERVE':
                        break
                check('low_reported_qdot_cannot_bypass_measured_position_span',
                      q.done_reason == 'grasp_probe_lift_not_reached',
                      {'reason': q.done_reason, 'injected_position_span_rad': .00302})
                for interruption in ('rate', 'error', 'pause'):
                    q, world = copy.deepcopy(ready_checkpoint)
                    for _ in range(25):
                        world.step(q, freeze_arm=True, tweak=velocity(.06))
                    assert q.state == 'PROBE_LIFT' and q.done_reason is None
                    if interruption == 'rate':
                        world.step(q, freeze_arm=True, tweak=velocity(.120001))
                    elif interruption == 'error':
                        saved = world.q['arm_joint1']
                        world.q['arm_joint1'] = q.debug['probe_lift_goal_rad'][0]+.041
                        world.step(q, freeze_arm=True, tweak=velocity(.06))
                        world.q['arm_joint1'] = saved
                    else:
                        world.step(q, freeze_arm=True,
                                   tweak=lambda obs: obs.__setitem__(3, .13))
                        for _ in range(9):
                            world.step(q, freeze_arm=True, tweak=velocity(.06))
                    # Pause's tenth quiet tick is the first eligible sample.
                    premature = False
                    for tick in range(42):
                        world.step(q, freeze_arm=True, tweak=velocity(.06))
                        premature |= tick < 39 and q.state == 'PROBE_OBSERVE'
                        if q.done_reason or q.state == 'PROBE_OBSERVE':
                            break
                    check('position_window_and_arrival_restart_after_'+interruption,
                          not premature and q.state == 'PROBE_OBSERVE' and not q.done_reason
                          and tick+1 >= 40,
                          {'samples_after_interruption': tick+1, 'reason': q.done_reason})
            case('measured_stability_window_execution', measured_stability_window)

            def public_only():
                class ForbiddenTruth:
                    def __array__(self, *a, **kw):
                        raise AssertionError('Privileged truth converted')
                    def __getitem__(self, item):
                        raise AssertionError('Privileged truth indexed')
                    def __float__(self):
                        raise AssertionError('Score/reward converted')
                poisoned = dict(images, object_poses=ForbiddenTruth(), score=ForbiddenTruth(),
                                reward=ForbiddenTruth(), ground_truth=ForbiddenTruth())
                q, world = copy.deepcopy(checkpoint)
                run_until(q, world, camera=poisoned)
                check('poisoned_truth_and_score_extras_are_not_read', q.done_reason == 'grasp_probe_observation_complete')
                json.dumps(q.describe(), allow_nan=False)
                json.dumps(q.debug, allow_nan=False)
                check('probe_public_diagnostics_are_finite_json', True)
            case('public_input_boundary_execution', public_only)

    report = {'scope': __doc__, 'fixture_classification': 'SYNTHETIC_CPU_ONLY',
              'passed': bool(checks) and all(checks.values()), 'count': len(checks),
              'checks': checks, 'evidence': evidence,
              'source_sha256': {name: hashlib.sha256((ROOT/'task_b'/name).read_bytes()).hexdigest()
                                for name in ('first_reach.py', 'grasp_probe.py', 'arm_kinematics.py',
                                             'audit_grasp_probe.py')}}
    content = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
