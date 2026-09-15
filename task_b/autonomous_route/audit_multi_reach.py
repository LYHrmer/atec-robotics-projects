"""Independent CPU counterexamples for the bounded two-target wrapper.

RGB-D arrays and detected components below are explicitly synthetic. The real
FirstReachPolicy, wrapper, stationary gates, FK and public observation mapping
execute. A small position-actuator fixture follows returned commands; this is
not Isaac physics, collision validation, a score signal or a success-rate test.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.audit_first_reach import fixture
from task_b.arm_kinematics import solve_ik
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.first_reach import FirstReachPolicy
from task_b.multi_reach import MultiReachPolicy
from task_e_geometry import JOINT_LOWER, JOINT_UPPER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-run', type=Path, help='Optional completed multi run with public telemetry for the view quiet gate.')
    args = parser.parse_args()
    schema, names, defaults = fixture()
    arm, leg, wheel = (schema.term(name) for name in (ARM_TERM, LEG_TERM, WHEEL_TERM))
    checks, evidence = {}, {}
    images = {key: value for source in ('ee', 'head') for key, value in (
        (source+'_rgb', np.zeros((24, 32, 3), np.uint8)),
        (source+'_depth', np.ones((24, 32))))}
    old_point = np.array([.5, .179, -.28])
    components = [{'point': old_point.copy(), 'height': .18}]
    detector_calls = []
    fit = solve_ik(old_point+[0., 0., .12], seed=[0., 1.1, -.8, 0., .95, 0.], max_nfev=120)
    parameters = dict(settle_calls=0, ramp_calls=1, forward_cmd=.05,
                      turn_cap=.6, standoff=.50, turn_gain=2., lowering_m=.02)

    def public_field(policy, key, default=None):
        for record in (policy.debug, policy.describe()):
            if key in record:
                return record[key]
        return default

    def phase(policy):
        return public_field(policy, 'outer_phase', policy.state)

    def lower(policy):
        value = public_field(policy, 'lowering_alpha')
        if value is None:
            raise AssertionError('JSON diagnostics must expose current lowering_alpha')
        return float(value)

    def detector(*unused, **kwargs):
        detector_calls.append(kwargs.get('locked_target_body'))
        found = []
        for component in components:
            point = np.asarray(component['point'], dtype=float)
            found.append({'body_point': point.tolist(), 'height_span_m': component['height'],
                          'forward_planar_distance_m': float(np.linalg.norm(point[:2])),
                          'optical_depth_m': 1., 'depth_samples': 100, 'pixel_area': 100,
                          'pixel_uv': [16., 12.], 'bbox_xywh': [11, 2, 10, 20],
                          'aspect': 2., 'kind': 'synthetic_supported_yellow_component'})
        return found, {'yellow_components': len(found), 'rejected': {},
                       'association_mode': 'synthetic_detector_output'}

    class PositionPlant:
        """Named actual q and finite-difference qdot, one step behind action."""
        def __init__(self):
            self.q = dict(defaults)
            self.qdot = dict.fromkeys(names, 0.)
            self.last_action = np.zeros(schema.total_dim)
            self.arm_response_tau_s = 0.
            self.arm_bias_rad = {}

        def observe(self):
            obs = np.zeros(84)
            obs[11] = -1.
            for index, name in enumerate(names):
                obs[12+index] = self.q[name]-defaults[name]
                obs[36+index] = self.qdot[name]
            return obs

        def respond(self, action, *, freeze_arm=False, freeze_legs=False):
            old = dict(self.q)
            for term, frozen in ((arm, freeze_arm), (leg, freeze_legs)):
                if not frozen:
                    for name, value in zip(term.joint_names, action[term.start:term.stop]):
                        desired = defaults[name]+float(value)*term.scale
                        if term is arm and self.arm_response_tau_s > 0.:
                            desired += self.arm_bias_rad.get(name, 0.)
                            fraction = -np.expm1(-.02/self.arm_response_tau_s)
                            self.q[name] += fraction*(desired-self.q[name])
                            if name in arm.joint_names[:6]:
                                index = arm.joint_names.index(name)
                                self.q[name] = float(np.clip(self.q[name], JOINT_LOWER[index], JOINT_UPPER[index]))
                        else:
                            self.q[name] = desired
            self.qdot = {name: (self.q[name]-old[name])/.02 for name in names}
            self.last_action = np.asarray(action).copy()

        def step(self, policy, *, tweak=None, camera=None, freeze_arm=False, freeze_legs=False):
            obs = self.observe()
            if tweak is not None:
                tweak(obs)
            action = policy.act(obs, images if camera is None else camera)
            self.respond(action, freeze_arm=freeze_arm, freeze_legs=freeze_legs)
            return np.asarray(action), obs

    def new(max_attempts=2):
        components[:] = [{'point': old_point.copy(), 'height': .18}]
        return MultiReachPolicy(schema, names, defaults, max_attempts=max_attempts, **parameters), PositionPlant()

    def to_phase(policy, plant, wanted, limit=1600, **step_kw):
        for _ in range(limit):
            if phase(policy) == wanted or policy.done_reason:
                return phase(policy) == wanted
            plant.step(policy, **step_kw)
        return False

    def snapshot(pair):
        return copy.deepcopy(pair)

    def spin_until_stop(policy, plant, limit=800, **step_kw):
        for _ in range(limit):
            plant.step(policy, **step_kw)
            if policy.done_reason:
                break

    for value in (0, 4):
        rejected = False
        try:
            MultiReachPolicy(schema, names, defaults, max_attempts=value, **parameters)
        except (ValueError, TypeError):
            rejected = True
        checks['attempt_limit_rejects_'+str(value)] = rejected
    try:
        MultiReachPolicy(schema, names, defaults, reach_only=True, **parameters)
    except ValueError:
        checks['synthetic_reach_only_cannot_create_real_multi_attempt_experiment'] = True
    else:
        checks['synthetic_reach_only_cannot_create_real_multi_attempt_experiment'] = False

    with patch('task_b.first_reach.detect_yellow_candidates', detector), \
            patch('task_b.multi_reach.detect_yellow_candidates', detector, create=True), \
            patch('task_b.visual_approach.detect_yellow_candidates', detector), \
            patch('task_b.first_reach.solve_ik', return_value=fit):
        recovery_checkpoint = None
        for attempts in (1, 2):
            multi, plant = new(attempts)
            original = FirstReachPolicy(schema, names, defaults, **parameters)
            max_error, matching_detection_calls = 0., True
            terminal_action = None
            for index in range(1200):
                obs = plant.observe()
                # Include the public stance-transition control bit; the wrapper
                # must preserve it without independently resetting calibration.
                original.pause_for_stance = multi.pause_for_stance = index < 4
                detector_calls.clear()
                expected = original.act(obs.copy(), images)
                expected_calls = copy.deepcopy(detector_calls)
                detector_calls.clear()
                action = multi.act(obs.copy(), images)
                actual_calls = copy.deepcopy(detector_calls)
                matching_detection_calls = (matching_detection_calls
                    and len(expected_calls) == len(actual_calls)
                    and all((a is None and b is None) or (a is not None and b is not None
                            and np.array_equal(a, b)) for a, b in zip(expected_calls, actual_calls)))
                max_error = max(max_error, float(np.max(np.abs(action-expected))))
                plant.respond(action)
                if original.done_reason:
                    terminal_action = action.copy()
                    break
            checks['first_attempt_action_prefix_exact_for_limit_'+str(attempts)] = (
                original.done_reason == 'reach_lowering_hold_complete' and max_error == 0.)
            checks['first_attempt_detector_schedule_and_arguments_unchanged_for_limit_'+str(attempts)] = (
                matching_detection_calls)
            if attempts == 1:
                checks['one_attempt_stops_only_after_original_normal_completion'] = (
                    multi.done_reason == 'multi_reach_attempts_complete'
                    and public_field(multi, 'completed_attempts') == 1
                    and np.array_equal(terminal_action, expected))
            else:
                checks['two_attempt_wrapper_intercepts_only_normal_finish_without_action_jump'] = (
                    multi.done_reason is None and phase(multi) == 'RESTORE_HEIGHT'
                    and public_field(multi, 'completed_attempts') == 1
                    and np.array_equal(terminal_action, expected)
                    and multi.wheel_hold_requested)
                recovery_checkpoint = snapshot((multi, plant))
            evidence['first_attempt_limit_'+str(attempts)] = {
                'control_calls': index+1, 'maximum_action_difference': max_error,
                'original_reason': original.done_reason, 'wrapper_reason': multi.done_reason}
        assert recovery_checkpoint is not None

        for label, tweak, expected in (
                ('nonfinite_public_state', lambda obs: obs.__setitem__(0, np.nan), 'invalid_proprio'),
                ('hard_tilt', lambda obs: obs.__setitem__(slice(9, 12), [np.sin(.3), 0., -np.cos(.3)]),
                 'tilt_limit_exceeded')):
            p, plant = new()
            plant.step(p, tweak=tweak)
            checks['first_attempt_error_'+label+'_is_terminal_not_a_restart'] = (
                p.done_reason == expected and public_field(p, 'completed_attempts') == 0)

        p, plant = snapshot(recovery_checkpoint)
        initial_arm = plant.last_action[arm.start:arm.stop].copy()
        alphas, hold_flags, arms = [lower(p)], [], []
        for _ in range(400):
            action, _ = plant.step(p)
            alphas.append(lower(p)); hold_flags.append(p.wheel_hold_requested)
            arms.append(action[arm.start:arm.stop].copy())
            if phase(p) != 'RESTORE_HEIGHT' or p.done_reason:
                break
        checks['height_restore_is_monotone_at_most_dt_over_three_and_reaches_zero'] = (
            phase(p) == 'RETRACT_FOR_VIEW' and p.done_reason is None
            and min(alphas) == 0. and max(alphas) <= 1.
            and np.max(np.diff(alphas)) <= 1e-12
            and np.min(np.diff(alphas)) >= -.02/3.-1e-12)
        checks['height_restore_keeps_arm_fingers_and_wheel_anchor_request'] = (
            all(hold_flags) and all(np.array_equal(value, initial_arm) for value in arms))
        retraction_checkpoint = snapshot((p, plant))

        p, plant = snapshot(recovery_checkpoint)
        for name in leg.joint_names:
            plant.q[name] += .09
        for _ in range(15):
            action, _ = plant.step(p, freeze_legs=True)
            if p.done_reason:
                break
        checks['restoration_checks_actual_leg_response_before_allowing_retraction'] = (
            p.done_reason is not None and phase(p) != 'RETRACT_FOR_VIEW'
            and public_field(p, 'completed_attempts') == 1)

        p, plant = snapshot(recovery_checkpoint)
        before_arm, before_alpha = plant.last_action[arm.start:arm.stop].copy(), lower(p)
        for _ in range(9):
            action, _ = plant.step(p, tweak=lambda obs: obs.__setitem__(3, .13))
        paused = (p.done_reason is None and lower(p) == before_alpha
                  and np.array_equal(action[arm.start:arm.stop], before_arm)
                  and p.wheel_hold_requested)
        for _ in range(9):
            plant.step(p)
        not_early = lower(p) == before_alpha
        plant.step(p)
        checks['restoration_soft_pause_freezes_commands_and_requires_point_two_seconds_quiet'] = (
            paused and not_early and p.done_reason is None and lower(p) < before_alpha)

        for label, index, value in (('continuous_angular_rate', 3, .13),
                                    ('net_tangent_drift', 0, .055),
                                    ('net_yaw', 5, .06)):
            p, plant = snapshot(recovery_checkpoint)
            for _ in range(170):
                plant.step(p, tweak=lambda obs, i=index, v=value: obs.__setitem__(i, v))
                if p.done_reason:
                    break
            checks['stationary_recovery_has_finite_'+label+'_guard'] = (
                p.done_reason is not None and public_field(p, 'completed_attempts') == 1)

        p, plant = snapshot(recovery_checkpoint)
        # A deliberately inconsistent public arm velocity must not permit the
        # quiet exit even though the actual joint positions equal the goal.
        def moving_arm(obs):
            for name in arm.joint_names[:6]:
                obs[36+names.index(name)] = .06
        spin_until_stop(p, plant, limit=330, tweak=moving_arm)
        checks['restoration_six_second_budget_includes_wait_for_actual_quiet'] = (
            p.done_reason is not None and phase(p) != 'RETRACT_FOR_VIEW')

        p, plant = snapshot(retraction_checkpoint)
        initial_q1 = plant.q['arm_joint1']
        deltas, tethers, hold_flags = [], [], []
        for _ in range(800):
            before = plant.last_action[arm.start:arm.stop]*arm.scale
            action, obs = plant.step(p)
            after = action[arm.start:arm.stop]*arm.scale
            observed = np.array([obs[12+names.index(name)]+defaults[name] for name in arm.joint_names])
            deltas.append(float(np.max(np.abs(after[:6]-before[:6]))))
            tethers.append(float(np.max(np.abs(after[:6]-observed[:6]))))
            hold_flags.append(p.wheel_hold_requested)
            if phase(p) != 'RETRACT_FOR_VIEW' or p.done_reason:
                break
        checks['view_retraction_respects_joint_speed_and_actual_state_tether'] = (
            phase(p) == 'CONFIRM_NEXT_TARGET' and p.done_reason is None
            and max(deltas) <= .30*.02+1e-7 and max(tethers) <= .10+1e-7)
        expected_view = np.array([initial_q1, 1.1, -.8, 0., 0., 0.])
        actual = np.array([plant.q[name] for name in arm.joint_names[:6]])
        checks['view_retraction_retains_measured_shoulder_yaw_and_open_fingers'] = (
            np.max(np.abs(actual-expected_view)) < .04
            and plant.q['arm_joint7'] > .03 and plant.q['arm_joint8'] < -.03
            and lower(p) == 0. and all(hold_flags))
        confirmation_checkpoint = snapshot((p, plant))

        p, plant = snapshot(confirmation_checkpoint)
        components[:] = []
        remembered = np.asarray(public_field(p, 'visited_local_points'), dtype=float).copy()
        for _ in range(20):
            plant.step(p, tweak=lambda obs: obs.__setitem__(2, .055))
        checks['public_vertical_velocity_never_changes_the_completed_point_memory'] = (
            p.done_reason is None
            and np.array_equal(np.asarray(public_field(p, 'visited_local_points')), remembered))

        p, plant = snapshot(confirmation_checkpoint)
        components[:] = []
        remembered = np.asarray(public_field(p, 'visited_local_points'), dtype=float).copy()
        for _ in range(25):
            def tangent_turn(obs):
                obs[0], obs[5] = .02, .03
            plant.step(p, tweak=tangent_turn)
        # Compare with the continuous rigid-frame solution, not a copy of the
        # implementation's Euler update. Small Euler error is expected.
        theta = .03*.5
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        expected = remembered.copy()
        expected[:, :2] = remembered[:, :2] @ rotation.T + np.array([
            -.02*np.sin(theta)/.03, .02*(1.-np.cos(theta))/.03])
        checks['completed_point_memory_is_propagated_once_by_public_tangent_and_yaw'] = (
            p.done_reason is None
            and np.max(np.abs(np.asarray(public_field(p, 'visited_local_points'))-expected)) < 1e-5)

        p, plant = snapshot(retraction_checkpoint)
        before = plant.last_action[arm.start:arm.stop]*arm.scale
        for _ in range(40):
            plant.step(p, freeze_arm=True)
        commanded = plant.last_action[arm.start:arm.stop]*arm.scale
        checks['frozen_actual_arm_cannot_be_replaced_by_command_arrival'] = (
            phase(p) == 'RETRACT_FOR_VIEW' and p.done_reason is None
            and np.max(np.abs(commanded[:6]-before[:6])) <= .10+1e-7)
        spin_until_stop(p, plant, limit=180, freeze_arm=True)
        checks['retraction_stall_is_terminal_and_does_not_try_next_target'] = (
            p.done_reason is not None and public_field(p, 'completed_attempts') == 1)

        p, plant = snapshot(retraction_checkpoint)
        before = plant.last_action[arm.start:arm.stop].copy()
        # A measured-state jump makes the actual-state tether disjoint from
        # the permitted next command slew. An unsafe jump is not a valid fix.
        plant.q['arm_joint1'] += .4
        action, _ = plant.step(p)
        checks['disjoint_actual_tether_and_command_slew_stop_without_an_arm_jump'] = (
            p.done_reason == 'retract_command_bounds_infeasible'
            and np.array_equal(action[arm.start:arm.stop], before) and lower(p) == 0.)

        p, plant = snapshot(retraction_checkpoint)
        plant.arm_response_tau_s = .10
        plant.arm_bias_rad = {'arm_joint2': .03276, 'arm_joint3': .05606}
        fixed_goal = np.array([plant.q['arm_joint1'], 1.1, -.8, 0., 0., 0.])
        corrections, fixed_goals, command_steps, tracking_errors = [], [], [], []
        for _ in range(620):
            before = plant.last_action[arm.start:arm.stop]*arm.scale
            action, obs = plant.step(p)
            after = action[arm.start:arm.stop]*arm.scale
            actual = np.array([obs[12+names.index(name)]+defaults[name] for name in arm.joint_names])
            command_steps.append(float(np.max(np.abs(after[:6]-before[:6]))))
            tracking_errors.append(float(np.max(np.abs(after[:6]-actual[:6]))))
            if 'retract_feedback_correction_rad' in p.debug:
                corrections.append(p.debug['retract_feedback_correction_rad'])
                fixed_goals.append(p.debug['retract_fixed_goal_rad'])
            if phase(p) != 'RETRACT_FOR_VIEW' or p.done_reason:
                break
        actual = np.array([plant.q[name] for name in arm.joint_names[:6]])
        checks['first_order_loaded_arm_reaches_the_original_actual_goal_band'] = (
            phase(p) == 'CONFIRM_NEXT_TARGET' and p.done_reason is None
            and np.max(np.abs(actual-fixed_goal)) < .04
            and max(abs(plant.qdot[name]) for name in arm.joint_names[:6]) < .05)
        checks['retraction_feedback_never_moves_the_fixed_goal_or_compensates_other_axes'] = (
            bool(fixed_goals) and all(np.array_equal(g, fixed_goal) for g in fixed_goals)
            and np.max(np.abs(corrections)) <= .08+1e-12
            and np.all(np.asarray(corrections)[:, [0, 3, 4, 5]] == 0.))
        checks['loaded_retraction_feedback_still_obeys_slew_and_actual_tether'] = (
            max(command_steps) <= .30*.02+1e-7 and max(tracking_errors) <= .10+1e-7)
        evidence['loaded_first_order_retraction'] = {
            'basis': 'synthetic first-order position response, not Isaac dynamics',
            'response_tau_s': .10, 'injected_steady_bias_rad': dict(plant.arm_bias_rad),
            'fixed_goal_rad': fixed_goal.tolist(), 'actual_final_q_rad': actual.tolist(),
            'actual_goal_error_rad': (actual-fixed_goal).tolist(),
            'final_phase': phase(p), 'stop_reason': p.done_reason,
            'max_command_step_rad': max(command_steps), 'max_actual_tether_error_rad': max(tracking_errors)}

        p, plant = snapshot(retraction_checkpoint)
        fixed_goal = np.array([plant.q['arm_joint1'], 1.1, -.8, 0., 0., 0.])
        perturbed, tail = False, []
        for _ in range(620):
            error = max(abs(plant.q[name]-fixed_goal[j]) for j, name in enumerate(arm.joint_names[:6]))
            if not perturbed and error < .002:
                # A one-off 2.5 mrad disturbance is well inside the positional
                # tolerance. Undamped P=1 with a one-step position response can
                # turn it into an enduring two-tick velocity oscillation.
                plant.q['arm_joint3'] += .0025
                perturbed = True
            plant.step(p)
            if perturbed:
                tail.append([plant.q['arm_joint3'], plant.qdot['arm_joint3']])
            if phase(p) != 'RETRACT_FOR_VIEW' or p.done_reason:
                break
        checks['small_near_goal_position_disturbance_decays_enough_to_finish_retraction'] = (
            perturbed and p.done_reason is None and phase(p) == 'CONFIRM_NEXT_TARGET')
        tail = np.asarray(tail[-50:])
        evidence['near_goal_disturbance_response'] = {
            'basis': 'instant position actuator with one isolated 0.0025 rad observed-position disturbance',
            'perturbation_applied': perturbed, 'final_phase': phase(p), 'stop_reason': p.done_reason,
            'last_up_to_50_samples_q3_range_rad': float(np.ptp(tail[:, 0])) if len(tail) else None,
            'last_up_to_50_samples_qdot3_max_rad_s': float(np.max(np.abs(tail[:, 1]))) if len(tail) else None}

        p, plant = snapshot(retraction_checkpoint)
        plant.step(p)
        decay = np.exp(-.02/.10)
        raw = np.asarray(p.debug['retract_feedback_raw_correction_rad'])
        filtered = np.asarray(p.debug['retract_feedback_correction_rad'])
        checks['new_retraction_goal_initializes_feedback_history_to_zero'] = (
            np.allclose(filtered, (1.-decay)*raw, atol=1e-12, rtol=0.)
            and np.all(filtered[[0, 3, 4, 5]] == 0.))
        plant.step(p)
        previous = np.asarray(p.debug['retract_feedback_correction_rad']).copy()
        frozen = plant.last_action[arm.start:arm.stop].copy()
        for _ in range(7):
            plant.step(p, tweak=lambda obs: obs.__setitem__(3, .13))
        for _ in range(9):
            plant.step(p)
        held = np.array_equal(plant.last_action[arm.start:arm.stop], frozen)
        plant.step(p)  # The tenth quiet tick resumes exactly one filter update.
        raw = np.asarray(p.debug['retract_feedback_raw_correction_rad'])
        filtered = np.asarray(p.debug['retract_feedback_correction_rad'])
        checks['retraction_pause_preserves_filter_history_until_one_resumed_update'] = (
            held and p.done_reason is None
            and np.allclose(filtered, decay*previous+(1.-decay)*raw, atol=1e-12, rtol=0.))

        # Independent stationary-view observations deliberately separate the
        # measured actuator velocity from the small sampled position ripple.
        # This matches the observed failure mode without assuming dynamics.
        def view_sample(policy, position_plant, initial, offset=0., rate=0., tweak=None):
            for name in arm.joint_names[:6]:
                position_plant.q[name] = initial[name]
                position_plant.qdot[name] = 0.
            position_plant.q['arm_joint2'] += offset
            position_plant.qdot['arm_joint2'] = rate
            position_plant.step(policy, freeze_arm=True, tweak=tweak)
            return bool(policy.debug.get('recovery_arm_position_quiet'))

        components[:] = []
        p, plant = snapshot(confirmation_checkpoint)
        initial = dict(plant.q)
        accepted = [view_sample(p, plant, initial, (-1.)**i*.00012, (-1.)**i*.06) for i in range(26)]
        checks['view_phase_discards_previous_window_and_requires_full_half_second_position_span'] = (
            not any(accepted[:25]) and accepted[25] and p.done_reason is None
            and p.debug.get('recovery_arm_position_window_s') == .5)

        p, plant = snapshot(confirmation_checkpoint)
        initial = dict(plant.q)
        accepted = [view_sample(p, plant, initial, i*.00012, .006) for i in range(40)]
        checks['slow_actual_joint_drift_cannot_pass_a_position_window_despite_low_velocity'] = (
            not any(accepted) and p.done_reason is None
            and p.debug.get('recovery_arm_position_range_rad', 0.) > .002)

        p, plant = snapshot(confirmation_checkpoint)
        initial = dict(plant.q)
        accepted = [view_sample(p, plant, initial, (-1.)**i*.002, (-1.)**i*.10) for i in range(40)]
        checks['large_back_and_forth_motion_cannot_pass_by_cancelling_mean_velocity'] = (
            not any(accepted) and p.done_reason is None
            and p.debug.get('recovery_arm_position_range_rad', 0.) > .002)

        p, plant = snapshot(confirmation_checkpoint)
        initial = dict(plant.q)
        for _ in range(25):
            view_sample(p, plant, initial)
        emergency_accepted = view_sample(p, plant, initial, rate=.12001)
        after_fault = [view_sample(p, plant, initial) for _ in range(26)]
        checks['instantaneous_arm_rate_cap_rejects_static_positions_and_clears_prior_evidence'] = (
            not emergency_accepted and not any(after_fault[:25]) and after_fault[25]
            and p.done_reason is None)

        p, plant = snapshot(confirmation_checkpoint)
        initial = dict(plant.q)
        for _ in range(25):
            view_sample(p, plant, initial)
        view_sample(p, plant, initial, tweak=lambda obs: obs.__setitem__(3, .13))
        pause_return = [view_sample(p, plant, initial) for _ in range(9)]
        resumed = [view_sample(p, plant, initial) for _ in range(26)]
        checks['recovery_pause_clears_position_evidence_and_requires_new_complete_window'] = (
            not any(pause_return) and not any(resumed[:25]) and resumed[25]
            and p.done_reason is None and p.wheel_hold_requested)

        if args.replay_run:
            # Saved public joint observations are replayed against a synthetic
            # quiet body. The old run and this checkpoint have different tilt
            # anchors, so splicing their body frames would create a fake jump.
            # The q1 reference is translated to the checkpoint's fixed yaw;
            # all per-axis errors and observed velocity ripple stay intact.
            saved = np.load(args.replay_run/'telemetry.npz')
            meta = json.loads((args.replay_run/'environment_metadata.json').read_text())
            records = [json.loads(line) for line in (args.replay_run/'trace.jsonl').read_text().splitlines()]
            final_goal = np.asarray(next(record['policy_debug']['retract_fixed_goal_rad'] for record in reversed(records)
                                         if 'retract_fixed_goal_rad' in record['policy_debug']))
            public_names = meta['observations']['joint_names']
            articulation_names = meta['action_schema']['articulation_joint_names']
            original_defaults = dict(zip(articulation_names, meta['action_schema']['default_joint_pos']))
            p, plant = snapshot(confirmation_checkpoint)
            components[:] = []
            checkpoint_goal = np.array([initial_q1, 1.1, -.8, 0., 0., 0.])
            accepted = []
            for source_obs in saved['proprio'][-50:].astype(float):
                obs = plant.observe()
                for i, name in enumerate(names):
                    source_id = public_names.index(name)
                    actual = source_obs[12+source_id]+original_defaults[name]
                    if name in arm.joint_names[:6]:
                        joint = arm.joint_names.index(name)
                        actual += checkpoint_goal[joint]-final_goal[joint]
                    obs[12+i] = actual-defaults[name]
                    obs[36+i] = source_obs[36+source_id]
                p.act(obs, images)
                accepted.append(bool(p.debug.get('recovery_arm_position_quiet')))
            checks['saved_public_micro_ripple_passes_full_view_window_without_world_or_reward_input'] = (
                not any(accepted[:25]) and all(accepted[25:]) and p.done_reason is None)
            evidence['saved_public_view_waveform'] = {
                'run_name': args.replay_run.name, 'steps': saved['step'][-50:][[0, -1]].tolist(),
                'telemetry_sha256': hashlib.sha256((args.replay_run/'telemetry.npz').read_bytes()).hexdigest(),
                'first_accepted_sample_1_based': next((i+1 for i, value in enumerate(accepted) if value), None),
                'basis': 'recorded public joint q/qdot waveform with synthetic quiet body; fixed q1 reference translated to checkpoint; no environment step',
                'last_window_s': p.debug.get('recovery_arm_position_window_s'),
                'last_position_range_rad': p.debug.get('recovery_arm_position_range_rad')}

        # Invalid or old components cannot reuse observations gathered while
        # the arm was moving. Each case starts from the same real quiet entry.
        for label, candidate in (
                # Offset stays inside the .50 m exclusion disk but is beyond
                # x=.8, so this specifically exercises memory, not near range.
                ('previous_point_exclusion_disk', {'point': old_point+[.35, 0., 0.], 'height': .18}),
                ('low_sugar_like_component', {'point': np.array([1.2, -.6, -.3]), 'height': .09}),
                ('too_close_new_component', {'point': np.array([.7, -.6, -.3]), 'height': .18})):
            p, plant = snapshot(confirmation_checkpoint)
            components[:] = [candidate]
            spin_until_stop(p, plant, limit=430)
            checks['new_target_acquisition_rejects_'+label] = (
                p.done_reason == 'next_target_not_confirmed'
                and public_field(p, 'completed_attempts') == 1)

        p, plant = snapshot(confirmation_checkpoint)
        components[:] = [{'point': np.array([1.2, -.6, -.3]), 'height': .18}]
        detector_calls.clear()
        plant.step(p)
        checks['one_control_tick_cannot_confirm_a_second_target'] = (
            phase(p) == 'CONFIRM_NEXT_TARGET' and public_field(p, 'attempt_index') == 1)
        components[:] = []
        spin_until_stop(p, plant, limit=430)
        checks['stale_single_candidate_cannot_authorize_navigation'] = (
            p.done_reason == 'next_target_not_confirmed')

        # A clear route to a new far component permits child handoff, not a
        # reach: the next child must still use a new BRAKE/stationary gate.
        p, plant = snapshot(confirmation_checkpoint)
        components[:] = [{'point': np.array([1.2, -.6, -.3]), 'height': .18}]
        for _ in range(130):
            plant.step(p)
            if public_field(p, 'attempt_index') == 2 or p.done_reason:
                break
        checks['fresh_distinct_tall_target_hands_off_to_second_child_without_reach_authorization'] = (
            public_field(p, 'attempt_index') == 2 and p.done_reason is None
            and public_field(p, 'completed_attempts') == 1 and lower(p) == 0.
            and public_field(p, 'child_state') not in ('REACH_READY', 'REACH'))
        second_checkpoint = snapshot((p, plant))

        # Prescribe a simple public translation towards the second synthetic
        # point, then stop. This is an observation sequence, not a wheeled
        # locomotion model. The real second child must perform its own BRAKE,
        # stationary gate, actual arm arrival, lowering and normal hold.
        two, two_plant = snapshot(second_checkpoint)
        second_point = np.array([1.2, -.6, -.3])
        components[:] = [{'point': second_point.copy(), 'height': .07}]
        second_states = []
        for _ in range(1600):
            movement = np.clip(old_point[:2]-second_point[:2], -.003, .003)
            second_point[:2] += movement
            # The public vertical velocity remains zero. A fixed camera-point
            # height suffices; the original IK returns a finite nearby goal.
            components[0]['point'] = second_point.copy()
            def prescribed_translation(obs, delta=movement.copy()):
                obs[:2] = -delta/.02
            two_plant.step(two, tweak=prescribed_translation)
            second_states.append(public_field(two, 'child_state'))
            if two.done_reason:
                break
        checks['second_child_obtains_its_own_brake_and_reach_before_normal_completion'] = (
            'BRAKE' in second_states and 'REACH_READY' in second_states and 'REACH' in second_states)
        checks['two_real_child_normal_completions_end_the_experiment_without_a_score_input'] = (
            two.done_reason == 'multi_reach_attempts_complete'
            and public_field(two, 'completed_attempts') == 2
            and public_field(two, 'attempt_index') == 2)
        evidence['two_synthetic_public_attempts'] = {
            'terminal_reason': two.done_reason,
            'completed_attempts': public_field(two, 'completed_attempts'),
            'basis': 'real child controllers on synthetic public positions/rates and zero reward input; no dynamics claim'}

        p, plant = snapshot(second_checkpoint)
        components[:] = [{'point': np.array([1.2, -.6, -.3]), 'height': .07}]
        for _ in range(60):
            plant.step(p)
        checks['acquisition_height_threshold_does_not_reject_later_partial_views'] = (
            p.done_reason is None and public_field(p, 'attempt_index') == 2)

        p, plant = snapshot(second_checkpoint)
        components[:] = []
        spin_until_stop(p, plant, limit=350)
        checks['second_attempt_lost_target_is_terminal_not_an_arbitrary_third_selection'] = (
            p.done_reason is not None and public_field(p, 'completed_attempts') == 1
            and public_field(p, 'attempt_index') == 2)

        p, plant = snapshot(confirmation_checkpoint)
        components[:] = [{'point': np.array([2.5, 2., -.3]), 'height': .18}]
        reached_backoff = to_phase(p, plant, 'BACKOFF_CLEARANCE', limit=130)
        checks['remembered_old_target_blocking_new_ray_requires_bounded_backoff'] = (
            reached_backoff and p.done_reason is None)
        checks['first_backoff_hold_release_coincides_with_first_nonzero_reverse_command'] = (
            reached_backoff and not p.wheel_hold_requested
            and np.all(plant.last_action[wheel.start:wheel.stop] < 0.)
            and np.max(np.abs(plant.last_action[wheel.start:wheel.stop])) <= .5*.02+1e-7)
        backoff_checkpoint = snapshot((p, plant))
        actions, release_flags = [], []
        for _ in range(6):
            action, _ = plant.step(p)
            actions.append(action[wheel.start:wheel.stop].copy())
            release_flags.append(not p.wheel_hold_requested)
        checks['backoff_releases_hold_only_for_equal_slewed_reverse_wheel_commands'] = (
            all(release_flags) and all(np.allclose(a, a[0]) and -.05-1e-7 <= a[0] <= 0. for a in actions)
            and max(np.max(np.abs(b-a)) for a, b in zip(actions, actions[1:])) <= .5*.02+1e-7)
        spin_until_stop(p, plant, limit=180)
        checks['backoff_without_public_backward_progress_cannot_advance_to_navigation'] = (
            p.done_reason is not None and public_field(p, 'attempt_index') == 1)

        p, plant = snapshot(backoff_checkpoint)
        start_calls = public_field(p, 'global_control_calls')
        for _ in range(230):
            # Static-world synthetic target coordinates follow the same
            # prescribed public reverse translation, independently of action.
            for component in components:
                component['point'][0] += .10*.02
            plant.step(p, tweak=lambda obs: obs.__setitem__(0, -.10))
            if phase(p) != 'BACKOFF_CLEARANCE' or p.done_reason:
                break
        checks['finite_public_reverse_can_clear_blocker_and_engage_reconfirmation_hold'] = (
            phase(p) == 'RECONFIRM_NEXT_TARGET' and p.done_reason is None
            and p.wheel_hold_requested and public_field(p, 'attempt_index') == 1)
        reconfirmation_checkpoint = snapshot((p, plant))
        for _ in range(30):
            plant.step(p)
        checks['post_backoff_reconfirmation_does_not_reuse_pre_motion_visual_gate'] = (
            phase(p) == 'RECONFIRM_NEXT_TARGET' and public_field(p, 'attempt_index') == 1)
        components[:] = []
        spin_until_stop(p, plant, limit=450)
        checks['failed_post_backoff_reconfirmation_stops_without_repeating_backoff'] = (
            p.done_reason is not None and public_field(p, 'attempt_index') == 1)

        # Nearly collinear targets cannot be made safe by the allowed .45 m
        # reverse. Public progress must not be treated as clearance success.
        p, plant = snapshot(confirmation_checkpoint)
        components[:] = [{'point': np.array([2.5, .7, -.3]), 'height': .18}]
        entered = to_phase(p, plant, 'BACKOFF_CLEARANCE', limit=130)
        reverse_ticks = 0
        if entered:
            for _ in range(330):
                components[0]['point'][0] += .10*.02
                plant.step(p, tweak=lambda obs: obs.__setitem__(0, -.10))
                reverse_ticks += 1
                if p.done_reason or phase(p) != 'BACKOFF_CLEARANCE':
                    break
        checks['backoff_distance_limit_cannot_be_bypassed_when_ray_remains_blocked'] = (
            entered and p.done_reason == 'backoff_clearance_not_reached'
            and reverse_ticks*.10*.02 <= .45+.002+1e-9
            and public_field(p, 'attempt_index') == 1)

        for label, index, value in (('excessive_reverse_speed', 0, -.26),
                                    ('reverse_yaw_drift', 5, .10)):
            p, plant = snapshot(backoff_checkpoint)
            spin_until_stop(p, plant, limit=80,
                            tweak=lambda obs, i=index, v=value: obs.__setitem__(i, v))
            checks['backoff_stops_on_'+label] = p.done_reason is not None

        p, plant = snapshot(confirmation_checkpoint)
        components[:] = []
        class ForbiddenTruth:
            def __array__(self, *args, **kwargs):
                raise AssertionError('non-public simulator truth was converted')
            def __getitem__(self, item):
                raise AssertionError('non-public simulator truth was indexed')
            def __float__(self):
                raise AssertionError('a reward was used by the policy')
        poisoned = dict(images, object_poses=ForbiddenTruth(), score=ForbiddenTruth(),
                        reward=ForbiddenTruth(), ground_truth=ForbiddenTruth())
        for _ in range(20):
            plant.step(p, camera=poisoned)
        checks['reward_and_simulator_truth_extras_cannot_trigger_another_attempt'] = (
            phase(p) == 'CONFIRM_NEXT_TARGET' and public_field(p, 'attempt_index') == 1
            and p.done_reason is None)
        json.dumps(p.describe(), allow_nan=False)
        checks['multi_phase_public_diagnostics_are_finite_json'] = True
        evidence['fixtures'] = {'classification': 'synthetic CPU observations and detector components',
                                'actual_q_basis': 'position response to returned commands with finite-difference qdot',
                                'no_original_environment_or_score_claim': True,
                                'observation_order_differs_from_action_order': True}

    report = {'scope': __doc__, 'passed': bool(all(checks.values())), 'count': len(checks),
              'checks': {name: bool(value) for name, value in checks.items()}, 'evidence': evidence,
              'source_sha256': {name: hashlib.sha256((ROOT/'task_b'/name).read_bytes()).hexdigest()
                                for name in ('first_reach.py', 'multi_reach.py', 'stationary_target_gate.py',
                                             'audit_multi_reach.py')}}
    content = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
