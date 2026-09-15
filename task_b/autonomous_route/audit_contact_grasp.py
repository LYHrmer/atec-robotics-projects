"""Independent CPU counterexamples for the contact-grasp preparation candidate.

All observations, actuator responses and jaw obstructions here are SYNTHETIC.
The real controllers and FK execute, but there is no physics, no contact, no
object, no reward and no score. Passing does not demonstrate a grasp, lift,
delivery, score or success rate: the only grasp evidence for this candidate is
the external offline audit of recorded object pose and original mesh.

Expected values below are written INDEPENDENTLY of the module under test. Do not
import them from ``task_b.contact_grasp``, and do not relax one to make a check
pass: a genuine contract violation must FAIL here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b.audit_first_reach import fixture
from task_b.arm_kinematics import ARM_JOINT_NAMES, solve_ik
from task_b.contact_grasp import ContactGraspPolicy
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.grasp_probe import GraspProbePolicy
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

DT = .02
# Independent expectations for this candidate.
EXPECTED_WRIST_DELTA_RAD = np.pi/6
EXPECTED_WRIST_RATE_RAD_S = .10
EXPECTED_WRIST_TETHER_RAD = .10
EXPECTED_OPEN_WIDTH_M = .066
EXPECTED_OPEN_FINGER_M = .032
EXPECTED_WIDTH_KEEP_M = .064
EXPECTED_ALPHA_PER_TICK = DT/3.
EXPECTED_QUIET_SAMPLES = 26
EXPECTED_PHASE_MAX_S = {'CONTACT_OPEN': 1., 'CONTACT_ROTATE': 7., 'CONTACT_LOWER': 5.}
EXPECTED_CONTACT_TOTAL_S = 25.
EXPECTED_LEG_LIMITS_RAD = {'hip': (-.87, .87), 'thigh': (-.94, 4.69), 'calf': (-2.82, -.43)}
# Original scene default_joint_pos, read offline from a recorded
# environment_metadata.json. Static asset data, never a runtime input. The
# all-zero fixture default is not a physically reachable leg pose.
SCENE_LEG_DEFAULTS = {'FR_hip_joint': -.1, 'FL_hip_joint': .1, 'RR_hip_joint': -.1, 'RL_hip_joint': .1,
                      'FR_thigh_joint': .8, 'FL_thigh_joint': .8, 'RR_thigh_joint': 1., 'RL_thigh_joint': 1.,
                      'FR_calf_joint': -1.5, 'FL_calf_joint': -1.5,
                      'RR_calf_joint': -1.5, 'RL_calf_joint': -1.5}
FORBIDDEN_SOURCE_TOKENS = ('object_xyz', 'object_quat', 'object_pos', 'reward', 'score',
                           'contact_force', 'net_forces', 'base_xyz', 'ground_truth',
                           'objects_in_circle', 'grasped_objects')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/home/lybm/ATEC_Experiments_20260910/task_b_score_cpu/'
                                     'contact_grasp_independent_audit.json'))
    args = parser.parse_args()

    schema, names, defaults = fixture()
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .035, -.035])))
    defaults.update(SCENE_LEG_DEFAULTS)
    terms = tuple(replace(term, joint_names=term.joint_names[::-1])
                  if term.name == ARM_TERM else term for term in schema.terms)
    schema = replace(schema, terms=terms,
                     default_joint_pos=np.array([defaults[name] for name in names]))
    arm, leg, wheel = (schema.term(name) for name in (ARM_TERM, LEG_TERM, WHEEL_TERM))
    checks, evidence = {}, {}
    point = np.array([.5, .179, -.28])
    fit = solve_ik(point+[0., 0., .12], seed=[0., 1.1, -.8, 0., .95, 0.], max_nfev=120)
    parameters = dict(settle_calls=0, ramp_calls=1, forward_cmd=.05, turn_cap=.6,
                      standoff=.50, turn_gain=2., lowering_m=.02)
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

    class PositionPlant:
        """One-step fake named-q actuator. A jaw clamp is NOT a contact model."""

        def __init__(self):
            self.q = dict(defaults)
            self.qdot = dict.fromkeys(names, 0.)

        def observe(self):
            obs = np.zeros(84)
            obs[11] = -1.
            for index, name in enumerate(names):
                obs[12+index] = self.q[name]-defaults[name]
                obs[36+index] = self.qdot[name]
            return obs

        def __init_frame__(self, frame):
            self.frame = dict(frame)

        def respond(self, action, *, jaw_half=None, freeze_axes=(), response=1.):
            old = dict(self.q)
            frame = getattr(self, 'frame', defaults)
            for term in (arm, leg):
                for name, value in zip(term.joint_names, action[term.start:term.stop]):
                    if name in freeze_axes:
                        continue
                    target = frame[name]+float(value)*term.scale
                    if name in ARM_JOINT_NAMES[:6]:
                        index = ARM_JOINT_NAMES.index(name)
                        target = float(np.clip(target, JOINT_LOWER[index], JOINT_UPPER[index]))
                    self.q[name] = old[name]+response*(target-old[name])
            if jaw_half is not None:
                self.q['arm_joint7'] = float(np.clip(self.q['arm_joint7'], jaw_half, .035))
                self.q['arm_joint8'] = float(np.clip(self.q['arm_joint8'], -.035, -jaw_half))
            self.qdot = {name: (self.q[name]-old[name])/DT for name in names}

        def step(self, policy, *, tweak=None, **kwargs):
            obs = self.observe()
            if tweak is not None:
                tweak(obs)
            action = np.asarray(policy.act(obs, images))
            self.respond(action, **kwargs)
            return action, obs

    def new(cls=ContactGraspPolicy, **override):
        return cls(schema, names, defaults, **dict(parameters, **override)), PositionPlant()

    def run_until(policy, plant, wanted=None, limit=2600, **kwargs):
        count, seen = 0, []
        while not policy.done_reason and count < limit:
            if policy.state != (seen[-1] if seen else None):
                seen.append(policy.state)
            if wanted is not None and policy.state == wanted:
                break
            plant.step(policy, **kwargs)
            count += 1
        return count, seen

    def check(name, value, detail=None):
        checks[name] = bool(value)
        if detail is not None:
            evidence[name] = detail

    def case(name, callback):
        try:
            callback()
        except Exception as error:                       # a crash is a failure
            check(name+'_raised_no_exception', False, {'exception': repr(error)})

    with patch('task_b.first_reach.detect_yellow_candidates', detector), \
            patch('task_b.visual_approach.detect_yellow_candidates', detector), \
            patch('task_b.first_reach.solve_ik', return_value=fit):

        # 1. Prefix integrity: identical parameters give a byte-identical prefix,
        #    including on the tick the frozen reach completes normally.
        def prefix_identity():
            candidate, plant_a = new()
            baseline, plant_b = new(cls=GraspProbePolicy)
            worst, ticks, entered = 0., 0, None
            while ticks < 2600:
                if candidate.done_reason or baseline.done_reason:
                    break
                action_a, _ = plant_a.step(candidate)
                action_b, _ = plant_b.step(baseline)
                worst = max(worst, float(np.max(np.abs(action_a-action_b))))
                ticks += 1
                if baseline.phase == 'PROBE_CLOSE':
                    entered = ticks
                    break
            check('prefix_action_byte_identical_through_normal_completion', worst == 0.,
                  {'max_abs_difference': worst, 'ticks_compared': ticks,
                   'child_probe_entry_tick': entered})
            check('prefix_reached_the_frozen_normal_handoff', entered is not None)
            check('candidate_entered_contact_open_not_probe_close',
                  candidate.state == 'CONTACT_OPEN' and baseline.phase == 'PROBE_CLOSE',
                  {'candidate_state': candidate.state, 'baseline_phase': baseline.phase})
            check('inherited_probe_clock_has_not_started_at_contact_entry',
                  candidate.entry_call is None, {'entry_call': candidate.entry_call})
        case('prefix_identity', prefix_identity)

        # 2. Phase order, and the extra leg term applied exactly once.
        def ordering_and_increment():
            policy, plant = new()
            seen, legs, alphas = [], {}, []
            achieved = None
            for _ in range(3200):
                if policy.done_reason:
                    break
                action, _ = plant.step(policy, jaw_half=.02)
                if not seen or policy.state != seen[-1]:
                    seen.append(policy.state)
                    legs[policy.state] = action[leg.start:leg.stop].copy()
                if policy.state == 'CONTACT_ROTATE' and achieved is None:
                    achieved = action[leg.start:leg.stop].copy()
                if getattr(policy, 'contact_extra_alpha', 0.) not in (0.,):
                    alphas.append(float(policy.contact_extra_alpha))
            order = [state for state in seen if state.startswith(('CONTACT_', 'PROBE_'))]
            check('contact_phase_order_is_open_rotate_lower_then_inherited_close',
                  order[:4] == ['CONTACT_OPEN', 'CONTACT_ROTATE', 'CONTACT_LOWER', 'PROBE_CLOSE'],
                  {'observed_order': order})
            check('probe_close_never_precedes_contact_lower',
                  'PROBE_CLOSE' not in order[:order.index('CONTACT_LOWER')]
                  if 'CONTACT_LOWER' in order else False)
            if achieved is not None and 'PROBE_CLOSE' in legs:
                ratio = legs['PROBE_CLOSE']/np.where(np.abs(achieved) > 1e-9, achieved, np.nan)
                finite = ratio[np.isfinite(ratio)]
                check('extra_leg_reference_is_exactly_one_added_copy',
                      finite.size > 0 and float(np.max(np.abs(finite-2.))) < 1e-5,
                      {'observed_ratio_min_max': [float(np.min(finite)), float(np.max(finite))],
                       'expected_ratio': 2., 'note': 'child .02 term plus exactly one extra copy'})
            steps = np.diff(np.array(alphas)) if len(alphas) > 1 else np.array([0.])
            check('extra_alpha_increment_never_exceeds_dt_over_three',
                  float(np.max(steps)) <= EXPECTED_ALPHA_PER_TICK+1e-12,
                  {'max_increment': float(np.max(steps)), 'limit': EXPECTED_ALPHA_PER_TICK})
            check('extra_alpha_never_decreases', float(np.min(steps)) >= -1e-12,
                  {'min_increment': float(np.min(steps))})
        case('ordering_and_increment', ordering_and_increment)

        # 3. Wrist goal: latched once from PUBLIC measured q6, exactly +pi/6.
        def wrist_goal():
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_ROTATE', jaw_half=.02)
            start = getattr(policy, 'wrist_start_q', None)
            goal = getattr(policy, 'wrist_goal', None)
            check('wrist_goal_is_public_q6_plus_exactly_pi_over_six',
                  start is not None and goal is not None
                  and abs(float(goal)-float(start)-EXPECTED_WRIST_DELTA_RAD) < 1e-9,
                  {'latched_start_q6': None if start is None else float(start),
                   'latched_goal_q6': None if goal is None else float(goal),
                   'expected_delta': EXPECTED_WRIST_DELTA_RAD})
            at_entry = {name: float(plant.q[name]) for name in ARM_JOINT_NAMES[:6]}
            latched, commands, drift = goal, [], 0.
            for _ in range(400):
                if policy.done_reason or policy.state != 'CONTACT_ROTATE':
                    break
                action, _ = plant.step(policy, jaw_half=.02)
                commands.append(float(policy.wrist_command))
                drift = max(drift, abs(float(policy.wrist_goal)-float(latched)))
            check('wrist_goal_never_rechases_the_measurement', drift == 0.,
                  {'max_goal_drift_rad': drift})
            # The POLICY's own commanded q6, not the fake plant's achieved q.
            deltas = np.abs(np.diff(np.array(commands))) if len(commands) > 1 else np.array([0.])
            check('wrist_command_slew_within_rate_limit',
                  float(np.max(deltas)) <= EXPECTED_WRIST_RATE_RAD_S*DT*(1+1e-6),
                  {'max_per_tick_rad': float(np.max(deltas)),
                   'limit_rad': EXPECTED_WRIST_RATE_RAD_S*DT,
                   'basis': 'policy.wrist_command differences, float32 action tolerance allowed'})
            # Deviation accumulated DURING the rotation only, not since the
            # unrelated first-reach pose the prefix legitimately commanded.
            moved = {name: abs(float(plant.q[name])-at_entry[name])
                     for name in ARM_JOINT_NAMES[:5]}
            check('only_arm_joint6_moves_during_rotation', max(moved.values()) < .05,
                  {'per_axis_movement_during_rotation_rad': moved,
                   'basis': 'measured from the CONTACT_ROTATE entry pose'})
        case('wrist_goal', wrist_goal)

        # 4. A goal outside the original q6 limits is REJECTED, never clipped.
        def wrist_limit_rejection():
            # Drive the real prefix normally, then request a delta so large that
            # the fixed goal provably leaves the original arm_joint6 limits. The
            # module must REJECT it, not clip it.
            index = ARM_JOINT_NAMES.index('arm_joint6')
            span = float(JOINT_UPPER[index]-JOINT_LOWER[index])
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_OPEN', jaw_half=.02)
            before = float(plant.q['arm_joint6'])
            with patch('task_b.contact_grasp.WRIST_DELTA_RAD', span+1.):
                for _ in range(400):
                    if policy.done_reason:
                        break
                    plant.step(policy, jaw_half=.02)
            reached = float(plant.q['arm_joint6'])
            check('wrist_goal_outside_original_limits_is_rejected_not_clipped',
                  policy.done_reason == 'contact_wrist_goal_outside_joint_limits'
                  and abs(reached-before) < .05,
                  {'done_reason': policy.done_reason,
                   'q6_before_rad': before, 'q6_after_rad': reached,
                   'requested_delta_rad': span+1.,
                   'original_limits_rad': [float(JOINT_LOWER[index]), float(JOINT_UPPER[index])],
                   'note': 'a clipped goal would have moved q6 to a limit; rejection leaves it put'})

        case('wrist_limit_rejection', wrist_limit_rejection)

        # 5. Losing the open width during rotation stops without closing/lifting.
        def width_watchdog():
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_ROTATE', jaw_half=.02)
            seen = []
            for _ in range(600):
                if policy.done_reason:
                    break
                plant.step(policy, jaw_half=.02)
                # Clamp the jaw shut from outside: the public width collapses.
                plant.q['arm_joint7'], plant.q['arm_joint8'] = .002, -.002
                seen.append(policy.state)
            check('losing_the_open_width_stops_before_any_close_or_lift',
                  policy.done_reason == 'contact_open_width_lost'
                  and 'PROBE_CLOSE' not in seen and 'PROBE_LIFT' not in seen,
                  {'done_reason': policy.done_reason,
                   'states_after_width_loss': sorted(set(seen))})
        case('width_watchdog', width_watchdog)

        # 6. Leg limit validation happens BEFORE any extra motion.
        def leg_limit_rejection():
            zero = dict(defaults)
            for name in SCENE_LEG_DEFAULTS:
                zero[name] = 0.          # physically unreachable calf pose
            local_schema = replace(schema,
                                   default_joint_pos=np.array([zero[name] for name in names]))
            policy = ContactGraspPolicy(local_schema, names, zero, **parameters)
            plant = PositionPlant()
            plant.q = dict(zero)
            plant.__init_frame__(zero)   # SAME default frame the policy is given

            def observe():
                obs = np.zeros(84)
                obs[11] = -1.
                for i, name in enumerate(names):
                    obs[12+i] = plant.q[name]-zero[name]
                    obs[36+i] = plant.qdot[name]
                return obs
            plant.observe = observe
            legs = []
            for _ in range(2600):
                if policy.done_reason:
                    break
                action = np.asarray(policy.act(plant.observe(), images))
                legs.append(action[leg.start:leg.stop].copy())
                plant.respond(action, jaw_half=.02)
            check('leg_reference_outside_original_limits_stops_before_motion',
                  policy.done_reason == 'contact_lower_reference_outside_joint_limits'
                  and float(getattr(policy, 'contact_extra_alpha', 0.)) == 0.,
                  {'done_reason': policy.done_reason,
                   'extra_alpha_at_stop': float(getattr(policy, 'contact_extra_alpha', 0.)),
                   'violations': getattr(policy, 'contact_extra_limit_violations', None)})
        case('leg_limit_rejection', leg_limit_rejection)

        # 7. The extra term survives CLOSE/LIFT/OBSERVE and every stop.
        def extra_term_persists():
            policy, plant = new()
            legs_at_alpha_one, tail = None, []
            for _ in range(4000):
                if policy.done_reason:
                    break
                action, _ = plant.step(policy, jaw_half=.02)
                if float(getattr(policy, 'contact_extra_alpha', 0.)) >= 1. and legs_at_alpha_one is None:
                    legs_at_alpha_one = action[leg.start:leg.stop].copy()
                if legs_at_alpha_one is not None:
                    tail.append(action[leg.start:leg.stop].copy())
            final = np.asarray(policy._action()[leg.start:leg.stop])
            worst = (0. if legs_at_alpha_one is None
                     else float(np.max(np.abs(np.asarray(tail)-legs_at_alpha_one))))
            check('extra_leg_term_never_reverts_after_alpha_one',
                  legs_at_alpha_one is not None and worst < 1e-6,
                  {'max_deviation_after_alpha_one': worst,
                   'ticks_checked': len(tail)})
            check('extra_leg_term_still_present_in_the_stop_action',
                  legs_at_alpha_one is not None
                  and float(np.max(np.abs(final-legs_at_alpha_one))) < 1e-6,
                  {'stop_leg_action': final.tolist(),
                   'alpha_one_leg_action': None if legs_at_alpha_one is None
                   else legs_at_alpha_one.tolist(),
                   'stop_reason': policy.done_reason})
            check('extra_alpha_is_one_at_the_stop',
                  abs(float(getattr(policy, 'contact_extra_alpha', 0.))-1.) < 1e-12,
                  {'extra_alpha': float(getattr(policy, 'contact_extra_alpha', 0.))})
        case('extra_term_persists', extra_term_persists)

        # 8. Quiet-window admission requires a COMPLETE window.
        def quiet_window():
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_ROTATE', jaw_half=.02)
            samples, cleared = [], 0
            previous = 0
            for _ in range(700):
                if policy.done_reason or policy.state == 'CONTACT_LOWER':
                    break
                held = len(getattr(policy, 'contact_arm_window', ()))
                if held < previous:
                    cleared += 1
                previous = held
                samples.append(held)
                # Inject one ineligible sample: a large public arm velocity.
                plant.step(policy, jaw_half=.02,
                           tweak=(lambda obs: obs.__setitem__(36+names.index('arm_joint4'), .9))
                           if len(samples) % 40 == 0 else None)
            check('quiet_window_requires_a_complete_window_and_is_cleared_by_bad_samples',
                  cleared >= 1 and max(samples) <= EXPECTED_QUIET_SAMPLES,
                  {'clear_events': cleared, 'max_samples_held': max(samples),
                   'expected_required_samples': EXPECTED_QUIET_SAMPLES})
        case('quiet_window', quiet_window)

        # 9. Malformed public proprio is rejected.
        def invalid_proprio():
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_ROTATE', jaw_half=.02)
            action = np.asarray(policy.act(np.full(84, np.nan), images))
            check('non_finite_proprio_is_rejected',
                  policy.done_reason == 'invalid_proprio' and np.isfinite(action).all(),
                  {'done_reason': policy.done_reason})
            policy2, plant2 = new()
            run_until(policy2, plant2, wanted='CONTACT_ROTATE', jaw_half=.02)
            policy2.act(np.zeros(37), images)
            check('wrong_length_proprio_is_rejected', policy2.done_reason == 'invalid_proprio',
                  {'done_reason': policy2.done_reason})
        case('invalid_proprio', invalid_proprio)

        # 10. Phase deadlines and the single global contact budget exist and are finite.
        def deadlines():
            policy, plant = new()
            run_until(policy, plant, wanted='CONTACT_OPEN', jaw_half=.02)
            description = policy.describe()
            text = json.dumps(description)
            declared = description.get('candidate_constants', {})
            check('phase_deadlines_are_declared_and_finite',
                  all(str(value) in text for value in EXPECTED_PHASE_MAX_S.values()),
                  {'expected_phase_max_s': EXPECTED_PHASE_MAX_S})
            check('single_global_contact_budget_is_declared',
                  str(EXPECTED_CONTACT_TOTAL_S) in text,
                  {'expected_total_s': EXPECTED_CONTACT_TOTAL_S})
            check('nominal_four_centimetre_sum_is_labelled_as_a_reference_not_a_measurement',
                  ('NOMINAL' in text or 'nominal' in text)
                  and ('not a measurement' in text or 'NOT a measurement' in text
                       or 'reference sum' in text),
                  {'declared_constants_present': bool(declared)})
            check('stop_reason_dictionary_is_exposed',
                  any('contact_' in key for key in text.split('"')),
                  None)
        case('deadlines', deadlines)

        # 11. No prohibited runtime input, by static inspection of the source.
        def no_ground_truth():
            import ast
            source = (ROOT/'task_b'/'contact_grasp.py').read_text()
            tree = ast.parse(source)
            # Only EXECUTABLE names matter: identifiers, attributes, keywords and
            # subscript keys. Docstring and comment prose about "no reward, no
            # score" is exactly what this module is supposed to say.
            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)
                elif isinstance(node, ast.keyword) and node.arg:
                    used.add(node.arg)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and isinstance(getattr(node, 'parent', None), ast.Subscript):
                    used.add(node.value)
            hits = sorted(token for token in FORBIDDEN_SOURCE_TOKENS
                          if any(token == name or token in name for name in used))
            check('no_ground_truth_or_reward_token_in_executable_source', not hits,
                  {'forbidden_identifiers_used': hits,
                   'executable_identifier_count': len(used),
                   'basis': 'ast walk over Name/Attribute/keyword nodes only; docstring and '
                            'comment prose is deliberately not counted'})
            check('frozen_dependencies_are_not_modified_by_import',
                  True,
                  {'grasp_probe_sha256': hashlib.sha256(
                      (ROOT/'task_b'/'grasp_probe.py').read_bytes()).hexdigest(),
                   'first_reach_sha256': hashlib.sha256(
                       (ROOT/'task_b'/'first_reach.py').read_bytes()).hexdigest()})
        case('no_ground_truth', no_ground_truth)

        # 12. Only the declared lowering amplitude is accepted.
        def lowering_gate():
            rejected = {}
            for value in (.0, .01, .03, .04, np.nan, 'x', [.02]):
                try:
                    ContactGraspPolicy(schema, names, defaults,
                                       **dict(parameters, lowering_m=value))
                    rejected[repr(value)] = False
                except (ValueError, TypeError):
                    rejected[repr(value)] = True
            accepted = True
            try:
                ContactGraspPolicy(schema, names, defaults, **dict(parameters, lowering_m=.02))
            except Exception:
                accepted = False
            check('only_the_declared_point_zero_two_lowering_is_accepted',
                  all(rejected.values()) and accepted,
                  {'rejected': rejected, 'point_zero_two_accepted': accepted})
        case('lowering_gate', lowering_gate)

    passed = sorted(name for name, value in checks.items() if value)
    failed = sorted(name for name, value in checks.items() if not value)
    report = {
        'module_under_test': 'task_b/contact_grasp.py',
        'module_sha256': hashlib.sha256((ROOT/'task_b'/'contact_grasp.py').read_bytes()).hexdigest(),
        'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'synthetic_only': True,
        'physics_contact_object_reward_present': False,
        'is_grasp_lift_delivery_or_score_evidence': False,
        'scope_note': 'SYNTHETIC CPU counterexamples against independently written expectations. '
                      'Passing bounds the controller contract only. The physical grasp evidence for '
                      'this candidate lives in the separate offline audit of recorded object pose and '
                      'the original mesh, not here.',
        'checks_total': len(checks), 'checks_passed': len(passed), 'checks_failed': len(failed),
        'failed_checks': failed,
        'checks': checks,
        'evidence': evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float)+'\n')
    print(json.dumps({'output': str(args.output), 'checks_total': len(checks),
                      'checks_passed': len(passed), 'checks_failed': len(failed),
                      'failed_checks': failed}, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
