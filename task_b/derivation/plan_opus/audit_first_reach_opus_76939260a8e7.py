"""CPU counterexamples for first-reach phase guards, not physical validation.

The policy-level cases stub the image detector with explicit raw detections. The
production camera parser, detection scheduling, stationary gate, joint geometry,
public observation mapping and state-machine code all execute. A separate block
runs the real detector on *generated* RGB-D pixels: those cases are synthetic CPU
tests of filter behavior and are explicitly not original-environment evidence.
Optional replay checks use real prior public telemetry and never feed object
truth into the policy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.arm_kinematics import head_camera_transform, solve_ik
from task_b.control import ActionSchema, ActionTerm, LEG_TERM, WHEEL_TERM, ARM_TERM
from task_b.first_reach import FirstReachPolicy
from task_b.stationary_target_gate import StationaryTargetGate
from task_b.visual_approach import detect_yellow_candidates


def fixture():
    legs = tuple(f'{side}_{link}_joint' for side in ('FR', 'FL', 'RR', 'RL')
                 for link in ('hip', 'thigh', 'calf'))
    wheels = tuple(f'{side}_foot_joint' for side in ('FR', 'FL', 'RR', 'RL'))
    arms = tuple(f'arm_joint{i}' for i in range(1, 9))
    names = (legs+wheels+arms)[::-1]  # Deliberately differs from action order.
    defaults = dict.fromkeys(names, 0.)
    limits = np.tile([-4., 4.], (24, 1))
    limits[names.index('arm_joint2')] = [.157, 2.983]  # Soft bound is NOT the IK hard bound.
    terms = (ActionTerm(LEG_TERM, 0, 12, legs, 'position', .5, True, None, 'resolved'),
             ActionTerm(WHEEL_TERM, 12, 4, wheels, 'velocity', 5., True, None, 'resolved'),
             ActionTerm(ARM_TERM, 16, 8, arms, 'position', .5, True, None, 'resolved'))
    schema = ActionSchema.from_terms(terms, names, np.zeros(24), limits, 24)
    return schema, names, defaults


def synthetic_head_frame(depth_m=1., side_px=36, shape=(240, 320)):
    """Generated head RGB-D holding one square, depth-supported yellow patch.

    SYNTHETIC CPU PIXELS ONLY, not original-environment evidence. A
    fronto-parallel square patch reproduces the near/overhead geometry in which
    the yellow component's bounding box is round (aspect about 1) and therefore
    fails the default vertical-aspect window. Returns the analytically expected
    body-frame point so locked association can be tested without trusting the
    detector's own output.
    """
    height, width = shape
    rgb = np.zeros((height, width, 3), np.uint8)
    depth = np.full(shape, 6.)
    top, left = height//2-side_px//2, width//2-side_px//2
    rgb[top:top+side_px, left:left+side_px] = [240, 215, 20]
    depth[top:top+side_px, left:left+side_px] = depth_m
    focal = width*24./20.955
    u, v = left+(side_px-1)/2., top+(side_px-1)/2.
    camera_point = np.array([(u-width/2.)*depth_m/focal, (v-height/2.)*depth_m/focal, depth_m])
    pose = head_camera_transform()
    return rgb, depth, pose[:3, :3] @ camera_point + pose[:3, 3]


def detector_shape_checks(checks, evidence):
    """Real detector on synthetic frames: default rejection vs locked acceptance."""
    gravity = np.array([0., 0., -1.])
    pose = head_camera_transform()
    rgb, depth, expected = synthetic_head_frame()
    plain, plain_info = detect_yellow_candidates(rgb, depth, pose, projected_gravity=gravity,
                                                focal_length=24.)
    locked, locked_info = detect_yellow_candidates(rgb, depth, pose, projected_gravity=gravity,
                                                   focal_length=24., locked_target_body=expected)
    distant, distant_info = detect_yellow_candidates(rgb, depth, pose, projected_gravity=gravity,
                                                     focal_length=24.,
                                                     locked_target_body=expected+[.3, 0., 0.])
    small_rgb, small_depth, small_expected = synthetic_head_frame(side_px=5)
    small, small_info = detect_yellow_candidates(small_rgb, small_depth, pose, projected_gravity=gravity,
                                                 focal_length=24., locked_target_body=small_expected)
    invalid = []
    for bad in ([1., 2.], [np.nan, 0., 0.], np.zeros((3, 1))):
        try:
            detect_yellow_candidates(rgb, depth, pose, projected_gravity=gravity, focal_length=24.,
                                     locked_target_body=bad)
            invalid.append(False)
        except ValueError:
            invalid.append(True)
    checks['synthetic_round_component_rejected_by_default_shape_filter'] = (
        not plain and plain_info['rejected'].get('image_shape') == 1
        and plain_info['association_mode'] == 'default_shape_filter'
        and plain_info['locked_target_body'] is None and plain_info['locked_shape_bypass'] == 0)
    item = locked[0] if locked else None
    checks['synthetic_round_component_accepted_only_by_locked_association'] = (
        len(locked) == 1 and locked_info['locked_shape_bypass'] == 1
        and item is not None and item['aspect'] < 1.15
        and np.allclose(item['body_point'], expected, atol=1e-6)
        and item['locked_association_distance_m'] < .01
        and locked_info['association_mode'] == 'locked_point_local_association')
    checks['locked_association_rejects_component_beyond_12cm'] = (
        not distant and distant_info['rejected'].get('locked_association_distance') == 1
        and distant_info['locked_association_radius_m'] == .12)
    checks['locked_association_keeps_area_height_and_metric_filters'] = (
        not small and small_info['rejected'].get('image_shape') == 1
        and small_info['locked_shape_bypass'] == 0
        and item is not None and item['height_span_m'] is not None
        and .045 < item['height_span_m'] < .35)
    checks['locked_target_must_be_a_finite_three_vector'] = all(invalid)
    evidence['synthetic_detector_case'] = {
        'basis': 'generated CPU pixels; NOT original-environment positive evidence',
        'expected_body_point': [float(value) for value in expected],
        'default_rejections': plain_info['rejected'],
        'locked_body_point': None if item is None else item['body_point'],
        'locked_aspect': None if item is None else item['aspect'],
        'locked_height_span_m': None if item is None else item['height_span_m'],
        'locked_pixel_area': None if item is None else item['pixel_area']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-dir', type=Path)
    args = parser.parse_args()
    schema, names, defaults = fixture()
    checks, evidence = {}, {}
    point = [.5, .1, -.28]
    gate = StationaryTargetGate()
    gate.update(0, True, point, 'ee')
    first = gate.update(15, True, point, 'ee', frame_token='frame_a')
    repeated = gate.update(20, True, point, 'ee', frame_token='frame_a')
    second = gate.update(25, True, point, 'ee', frame_token='frame_b')
    checks['gate_pre_settle_detection_and_repeated_frame_cannot_fake_two_samples'] = (
        not first['ready'] and not repeated['ready'] and second['ready'])
    checks['gate_explicit_frame_tokens_have_honest_distinct_basis'] = second['freshness_basis'] == 'distinct_frame_token'
    same_step = gate.update(25, True, [.52, .1, -.28], 'head', frame_token='frame_c')
    checks['gate_repeated_control_call_cannot_increment_confirmation'] = (
        same_step['confirmations'] == second['confirmations'] and same_step['source'] == 'ee')
    switched = gate.update(30, True, point, 'head', frame_token='head_a')
    checks['gate_camera_switch_requires_new_confirmation'] = not switched['ready'] and switched['confirmations'] == 1
    gate.update(35, True, point, 'head', frame_token='head_b')
    moving = gate.update(36, False)
    restarted = gate.update(37, True, point, 'head')
    checks['gate_motion_clears_samples_and_quiet_clock'] = (
        not moving['ready'] and moving['confirmations'] == 0
        and not restarted['ready'] and restarted['last_fresh_step'] is None)
    gate.reset(0)
    gate.update(0, True)
    gate.update(15, True, point, 'ee')
    fresh = gate.update(20, True, point, 'ee')
    stale = gate.update(46, True)
    checks['gate_stale_confirmation_expires_without_new_detection'] = fresh['ready'] and not stale['ready']
    checks['gate_no_token_does_not_claim_proven_frame_update'] = fresh['freshness_basis'] == 'sensor_period_elapsed_assumption'
    gate.reset(0)
    gate.update(0, True)
    gate.update(15, True, point, 'ee')
    too_soon = gate.update(16, True, point, 'ee')
    jump = gate.update(20, True, [.6, .1, -.28], 'ee')
    checks['gate_short_interval_and_spatial_jump_do_not_authorize_reach'] = (
        not too_soon['ready'] and too_soon['confirmations'] == 1
        and not jump['ready'] and jump['confirmations'] == 1)
    gate.reset(0)
    gate.update(0, True)
    gate.update(15, True, point, 'ee')
    after_gap = gate.update(60, True, point, 'ee')
    new_pair = gate.update(65, True, point, 'ee')
    checks['gate_sparse_call_gap_cannot_join_expired_sample_to_new_detection'] = (
        not after_gap['ready'] and after_gap['confirmations'] == 1
        and after_gap['sample_steps'] == [60] and new_pair['ready'])
    images = {key: value for source in ('head', 'ee')
              for key, value in ((source+'_rgb', np.zeros((24, 32, 3), np.uint8)),
                                 (source+'_depth', np.ones((24, 32))))}
    detector_shape_checks(checks, evidence)
    target = np.array([.5, .179, -.28])
    q_view = np.array([0., 1.1, -.8, 0., .95, 0.])
    fit = solve_ik(target+[0., 0., .12], seed=q_view, max_nfev=120)
    feed = {'point': target.copy()}
    detector_calls = []

    def detector(*unused, **kwargs):
        # Records the association keyword so lock plumbing is auditable, and
        # stays compatible with any further detector keywords.
        detector_calls.append(kwargs.get('locked_target_body'))
        if feed['point'] is None:
            return [], {}
        point = np.asarray(feed['point'])
        return [{'body_point': point.tolist(),
                 'forward_planar_distance_m': float(np.linalg.norm(point[:2]))}], {}

    def fresh(**kwargs):
        p = FirstReachPolicy(schema, names, defaults, settle_calls=0, ramp_calls=1,
                             standoff=.50, forward_cmd=.05, **kwargs)
        p.stance_verified = True
        p.arm_command[:6] = q_view
        detector_calls.clear()
        return p

    def observation(p, q=None):
        obs = np.zeros(84)
        obs[11] = -1.
        q = p.arm_command if q is None else np.r_[q[:6], .035, -.035]
        for i in range(1, 9):
            obs[12+names.index('arm_joint'+str(i))] = q[i-1]
        return obs

    def ready(**kwargs):
        p = fresh(**kwargs)
        feed['point'] = target.copy()
        for _ in range(130):
            p.act(observation(p), images)
            if p.reach_q is not None or p.done_reason is not None:
                break
        assert p.reach_q is not None, p.describe()
        return p

    with patch('task_b.first_reach.detect_yellow_candidates', detector), \
            patch('task_b.first_reach.solve_ik', return_value=fit):
        p = ready()
        checks['brake_requires_sensor_period_samples_after_one_second_stop'] = (
            p.calls-p.brake_start >= 65 and p.debug['stationary_target_gate']['confirmations'] >= 2)
        checks['stationary_ik_uses_unfiltered_raw_target'] = np.allclose(p.reach_target, target)
        checks['periodic_detection_not_control_tick_count'] = (
            p.vision_samples == p.calls//p.vision_stride and len(detector_calls) == 2*p.vision_samples)
        checks['hard_arm_joint2_limit_preserved_above_soft_limit'] = 2.983 < p.reach_q[1] <= 3.14
        checks['approach_and_brake_detection_is_never_locked'] = (
            len(detector_calls) > 0 and all(lock is None for lock in detector_calls))
        checks['ready_anchors_gravity_and_zeroes_base_drift_integrals'] = (
            p.reach_up_anchor is not None and np.allclose(p.reach_up_anchor, [0., 0., 1.])
            and np.isclose(np.linalg.norm(p.reach_up_anchor), 1.)
            and np.array_equal(p.reach_base_displacement, np.zeros(3))
            and p.reach_base_yaw == 0. and p.reach_timeout_s >= 15.
            and p.state == 'REACH_READY')
        evidence['first_reach_ready_step'] = p.calls

        p = ready()
        goal, fixed = p.reach_q.copy(), p.reach_target.copy()
        locked_before = len(detector_calls)
        while (p.calls+1) % p.vision_stride:
            p.act(observation(p), images)
        p.act(observation(p), images)
        reach_locks = detector_calls[locked_before:]
        checks['reach_detection_locks_association_to_the_fixed_target'] = (
            len(reach_locks) == 2 and all(lock is not None and np.allclose(lock, fixed)
                                          for lock in reach_locks)
            and set(p.debug['detector_rejections']) == {'ee', 'head'})
        feed['point'] = target + [0., .05, 0.]
        displacements = []
        for _ in range(40):
            p.act(observation(p), images)
            if 'reach_target_displacement_m' in p.debug:
                displacements.append(p.debug['reach_target_displacement_m'])
        checks['associated_points_update_diagnostics_but_never_retarget_or_resolve_ik'] = (
            bool(displacements) and np.allclose(displacements, .05) and p.done_reason is None
            and np.array_equal(p.reach_q, goal) and np.array_equal(p.reach_target, fixed))

        p = ready()
        feed['point'] = target + [.2, 0., 0.]
        unassociated = False
        for _ in range(160):
            p.act(observation(p), images)
            unassociated = unassociated or bool(p.debug.get('reach_unassociated_component'))
            if p.done_reason:
                break
        checks['distant_unmatched_component_is_not_called_target_movement_in_p1'] = (
            unassociated and p.done_reason is None and not p.reach_visual_valid
            and np.array_equal(p.reach_target, target))

        p = fresh()
        feed['point'] = target.copy()
        for _ in range(30):
            p.act(observation(p), images)
        feed['point'] = None
        for _ in range(500):
            p.act(observation(p), images)
            if p.done_reason:
                break
        checks['moving_period_target_cannot_authorize_reach_without_new_detections'] = (
            p.reach_q is None and p.done_reason == 'brake_or_stationary_vision_timeout')

        p = ready(lowering_m=.02)
        while (p.calls+1) % p.vision_stride:
            p.act(observation(p), images)
        feed['point'] = None
        before = p.arm_command.copy()
        p.act(observation(p), images)
        checks['lowering_request_still_holds_arm_on_visible_target_loss'] = (
            p.state == 'REACH_VISUAL_HOLD' and np.array_equal(before, p.arm_command))
        for _ in range(30):
            p.act(observation(p), images)
        checks['lowering_request_persistent_loss_stops_normally'] = (
            p.done_reason == 'reach_visual_confirmation_lost')

        p = ready(lowering_m=.02)
        feed['point'] = target + [.2, 0., 0.]
        for _ in range(60):
            p.act(observation(p), images)
            if p.done_reason:
                break
        checks['lowering_request_keeps_strict_target_movement_stop'] = (
            p.done_reason == 'reach_target_moved_beyond_12cm')

        # P1 (lowering_m == 0) executes the already authorized fixed joint goal
        # while shape detections are missing; it must not require an impossible
        # continued detection, and must not relax any bound to do so.
        p = ready()
        feed['point'] = None
        start = float(np.max(np.abs(p.arm_command[:6]-p.reach_q)))
        steps = []
        for _ in range(40):
            was = p.arm_command[:6].copy()
            p.act(observation(p), images)
            steps.append(float(np.max(np.abs(p.arm_command[:6]-was))))
        error = float(np.max(np.abs(p.arm_command[:6]-p.reach_q)))
        checks['missing_vision_beyond_half_second_still_makes_bounded_joint_progress'] = (
            p.done_reason is None and p.state == 'REACH' and start-error > .1
            and p.debug['reason'] == 'bounded_joint_reach_visual_association_optional_no_lowering'
            and not p.debug['reach_visual_valid'] and p.reach_lost_start is None)
        checks['bounded_reach_keeps_030_rad_s_slew_without_vision'] = max(steps) <= .30*p.dt+1e-12
        checks['missing_vision_reach_never_lowers_legs'] = p.lowering_alpha == 0.
        evidence['p1_missing_vision_progress'] = {
            'expected_out_of_view': bool(p.debug['expected_out_of_view']),
            'joint_error_rad': [start, error], 'state': p.state}

        p = ready()
        feed['point'] = None
        for _ in range(3000):
            p.act(observation(p), images)
            if p.done_reason:
                break
        checks['bounded_reach_without_vision_finishes_or_times_out'] = p.done_reason in (
            'reach_observation_complete', 'reach_joint_motion_timeout')
        evidence['p1_missing_vision_stop'] = {'reason': p.done_reason, 'steps': int(p.calls),
                                              'joint_error_rad': p.debug.get('reach_joint_error_rad')}

        p = ready()
        feed['point'] = None
        fixed_q = p.arm_command.copy()
        for _ in range(60):
            p.act(observation(p, fixed_q), images)
        checks['measured_state_tether_still_bounds_command_without_vision'] = (
            np.all(np.abs(p.arm_command[:6]-fixed_q[:6]) <= .10+1e-9) and p.done_reason is None)
        for _ in range(200):
            p.act(observation(p, fixed_q), images)
            if p.done_reason:
                break
        checks['missing_vision_does_not_disable_no_progress_stop'] = (
            p.done_reason == 'reach_joint_progress_below_002rad_in_3s')

        p = ready()
        feed['point'] = None
        p.reach_timeout_s = .05  # shortened deadline; the guard itself is production code
        for _ in range(5):
            p.act(observation(p), images)
        checks['missing_vision_does_not_disable_total_reach_deadline'] = (
            p.done_reason == 'reach_joint_motion_timeout')

        p = ready()
        feed['point'] = None
        begin = float(np.max(np.abs(p.arm_command[:6]-p.reach_q)))
        camera_errors = set()
        for _ in range(40):
            p.act(observation(p), {})
            camera_errors.update(p.debug.get('camera_errors', {}))
        checks['unavailable_camera_images_do_not_stop_bounded_p1_reach'] = (
            p.done_reason is None and p.state == 'REACH' and camera_errors == {'ee', 'head'}
            and float(np.max(np.abs(p.arm_command[:6]-p.reach_q))) < begin-.1)

        for label, index, value, expected in (
                ('displacement', 0, .05, 'reach_base_displacement_budget_exceeded'),
                ('yaw', 5, .05, 'reach_base_yaw_budget_exceeded')):
            p = ready()
            feed['point'] = None
            for _ in range(150):
                obs = observation(p)
                obs[index] = value  # below the instantaneous stop, above the integral budget
                p.act(obs, images)
                if p.done_reason:
                    break
            checks['reach_stops_on_integrated_base_'+label+'_budget'] = p.done_reason == expected

        p = ready()
        feed['point'] = None
        obs = observation(p)
        obs[9:12] = [np.sin(.07), 0., -np.cos(.07)]
        p.act(obs, images)
        checks['reach_stops_when_public_gravity_direction_leaves_anchor'] = (
            p.done_reason == 'reach_gravity_changed')

        p = ready()
        feed['point'] = None
        for _ in range(200):
            obs = observation(p)
            obs[2] = .055  # vertical only: never integrated into the budget
            p.act(obs, images)
            if p.done_reason:
                break
        checks['vertical_velocity_is_not_integrated_into_the_displacement_budget'] = (
            p.done_reason is None and p.debug['reach_base_displacement_m'] < 1e-9)

        p = ready()
        p.arm_command[:6] = p.reach_q
        feed['point'] = None
        for _ in range(20):
            p.act(observation(p, p.reach_q), images)
        checks['predicted_two_camera_loss_allows_bounded_p1_joint_reach'] = (
            p.done_reason is None and p.debug['expected_out_of_view'] and not p.debug['reach_visual_valid'])
        checks['p1_never_changes_leg_lowering_action'] = p.lowering_alpha == 0.

        p = ready(lowering_m=.02)
        p.arm_command[:6] = p.reach_q
        p.lowering_alpha = .2
        feed['point'] = None
        # Advance directly to the next scheduled camera sample.
        p.calls += (p.vision_stride-1-p.calls) % p.vision_stride
        before, lower = p.arm_command.copy(), p.lowering_alpha
        p.act(observation(p, p.reach_q), images)
        checks['out_of_view_never_authorizes_lowering'] = (
            p.state == 'REACH_VISUAL_HOLD' and p.lowering_alpha == lower
            and np.array_equal(before, p.arm_command)
            and p.debug['expected_out_of_view'] and not p.debug['out_of_view_reach_allowed'])

        p = ready(lowering_m=.02)
        p.lowering_alpha = .2
        obs = observation(p)
        obs[9:12] = [np.sin(.13), 0., -np.cos(.13)]
        before = p.arm_command.copy()
        action = p.act(obs, images)
        checks['reach_tilt_stops_arm_wheels_and_lowering_progress'] = (
            p.done_reason == 'reach_unstable_posture' and p.lowering_alpha == .2
            and np.array_equal(before, p.arm_command) and np.all(action[12:16] == 0.))

        p = ready()
        obs = observation(p)
        obs[0] = .07
        p.act(obs, images)
        checks['frozen_body_target_cannot_be_used_after_base_motion'] = p.done_reason == 'reach_base_not_stationary'

        p = ready()
        fixed_q = p.arm_command.copy()
        for _ in range(170):
            p.act(observation(p, fixed_q), images)
            if p.done_reason:
                break
        checks['joint_stall_has_normal_explicit_stop'] = p.done_reason == 'reach_joint_progress_below_002rad_in_3s'

        for label, point, expected in (
                ('far', [2., .6, -.28], None),
                ('near', [.7, .179, -.28], 'near_field_forward_progress_below_1cm_in_3s')):
            p = fresh()
            feed['point'] = np.array(point)
            for _ in range(190):
                p.act(observation(p), images)
                if p.done_reason:
                    break
            checks[label+'_stationary_visual_distance_has_correct_scope'] = p.done_reason == expected

        p = fresh(reach_only=True)
        feed['point'] = None
        for _ in range(1100):
            p.act(observation(p), {})
        checks['synthetic_reach_probe_does_not_require_images_or_visual_timeout'] = (
            p.reach_q is not None and p.done_reason is None and p.vision_samples == 0)
        json.dumps(p.describe(), allow_nan=False)
        checks['phase_records_are_json_safe'] = True

    if args.replay_dir:
        record = np.load(args.replay_dir/'telemetry.npz')
        record_names = record['joint_names'].tolist()
        leg_ids = [i for i, name in enumerate(record_names) if any(
            part in name for part in ('_hip', '_thigh', '_calf'))]
        obs = record['proprio'][350:375]
        gravity = obs[:, 9:12]
        tilt = np.arccos(np.clip(-gravity[:, 2]/np.linalg.norm(gravity, axis=1), -1., 1.))
        speed = np.max(np.abs(record['qdot'][350:375, leg_ids]), axis=1)
        stable = (tilt < .085) & (np.linalg.norm(obs[:, 3:6], axis=1) < .12) & (speed < .20)
        checks['recorded_post_compact_public_motion_satisfies_half_second_gate'] = bool(stable.all())
        evidence['public_replay'] = {'path': str(args.replay_dir), 'steps': [351, 375],
                                     'max_leg_speed_rad_s': float(speed.max()),
                                     'max_tilt_rad': float(tilt.max())}
    report = {'scope': 'CPU failure-branch and public-telemetry checks only; no dynamics, collision-free '
                       'path or score claim. Detector shape cases use generated synthetic images and are '
                       'not original-environment positive evidence.',
              'passed': bool(all(checks.values())), 'count': len(checks),
              'checks': {name: bool(value) for name, value in checks.items()}, 'evidence': evidence,
              'source_sha256': {name: hashlib.sha256((ROOT/'task_b'/name).read_bytes()).hexdigest()
                                for name in ('first_reach.py', 'visual_approach.py',
                                             'stationary_target_gate.py')}}
    content = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
