#!/usr/bin/env python3
"""Independently verify original Task B positive-score evidence, without Isaac.

Usage: python audit_positive_taskb.py RUN_DIRECTORY [--output REPORT.json]
Exit 0: positive score independently supported; 2: zero score/not achieved;
exit 1: inconsistent or missing evidence. A zero-score run never passes.

This audits recorded evidence; it cannot establish simulator assets/driver
identity or rule out unrecorded runtime mutations. No grasp/pass is inferred
from proximity points. Policy source snapshots are retained for human review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


PIN = '4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc'
ORIGINAL = Path(os.environ.get('ATEC_TASK_ROOT', str(Path.home() / 'ATEC2026_Simulation_Challenge'))).expanduser().resolve()
OFFICIAL_PREFIX = 'source/atec_rl_lab/atec_rl_lab/'
KNOWN_INACTIVE_TASK_E_DIAGNOSTIC_SHA = '80121ef00a27cf335242eb0ead7925e31c111bab877bf094a6d630aca4eb8d74'
CRITICAL_OFFICIAL = {
    'tasks/task_b/env_cfg.py', 'tasks/task_b/mdp/rewards.py',
    'tasks/task_b/mdp/terminations.py', 'tasks/task_base/envs_base_cfg.py',
    'tasks/task_base/action_base.py', 'tasks/task_base/envs_base.py',
    'tasks/task_b/terrain.py', 'assets/robots/b2w.py', 'assets/robots/b2.py',
    'assets/objects/task_b/object.py',
}
POLICY_FILES = {'task_b/first_reach.py', 'task_b/arm_kinematics.py',
                'task_b/visual_approach.py', 'task_b/control.py',
                'task_b/stance_hold.py', 'task_b/stance_reference.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def geometry_sets(state):
    """Official float32 squared-distance comparisons; no grasp inference."""
    obj = np.asarray(state['object_xyz'], dtype=np.float32)
    ee = np.asarray(state['gripper_xyz'], dtype=np.float32)
    if obj.shape != (18, 3) or ee.shape != (3,) or not np.isfinite(obj).all() or not np.isfinite(ee).all():
        raise ValueError('Expected finite object_xyz[18,3] and gripper_xyz[3]')
    reached = set(np.flatnonzero(np.sum((obj-ee)**2, axis=1) <= np.float32(.20**2)).tolist())
    inside = set(np.flatnonzero(
        (np.sum((obj[:, :2]-np.array([-3., -10.], np.float32))**2, axis=1) <= 1.)
        & (obj[:, 2] >= 0.) & (obj[:, 2] <= .5)).tolist())
    return reached, inside


def audit(run):
    report = {'run_directory': str(run), 'official_commit': PIN, 'audit_schema_version': 2,
              'verified_positive_raw_score': False, 'checks': [], 'events': [],
              'scope_limits': [
                  'Recorded source/configuration/state audit, not a fresh simulation.',
                  'USD assets, Isaac packages, GPU drivers and unrecorded runtime mutations are not cryptographically attested.',
                  'AST rejects direct simulator access in policy modules; semantic absence of indirect GT input still requires human source review.',
                  'A proximity reward is not evidence of grasp, lift, delivery or full Task B completion.',
              ]}

    def check(name, condition, evidence=None):
        report['checks'].append({'name': name, 'passed': bool(condition), 'evidence': plain(evidence)})
        return bool(condition)

    for name in ['result.json', 'environment_metadata.json', 'source_manifest.json', 'telemetry.npz', 'trace.jsonl']:
        check('artifact_exists:' + name, (run / name).is_file())
    if not all(x['passed'] for x in report['checks']):
        report['status'] = 'missing_evidence'
        return report, 1
    result = load(run / 'result.json')
    meta = load(run / 'environment_metadata.json')
    manifest = load(run / 'source_manifest.json')
    rows = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines() if line.strip()]
    with np.load(run / 'telemetry.npz', allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    n = int(result['steps'])
    dt = float(meta['step_dt'])
    score = float(result['score_raw_total'])
    report.update(steps=n, score_raw_total=score, stop_reason=result.get('stop_reason'))
    check('completed_run_not_integration_failure', not (run / 'failure.txt').exists())
    check('original_task_identity', result['task'] == meta['task'] == 'ATEC-TaskB-B2wPiper')
    check('result_seed_matches_metadata', result['seed'] == meta['seed'])
    check('finite_positive_timestep', np.isfinite(dt) and dt > 0 and np.isclose(float(a['dt']), dt))
    check('original_timestep_and_episode_length', np.isclose(dt, .02)
          and np.isclose(meta['physics_dt'], .005) and meta['decimation'] == 4
          and np.isclose(dt, meta['physics_dt'] * meta['decimation'])
          and meta['episode_length_s'] == 1200. and meta['max_episode_length_steps'] == 60000)
    check('single_complete_step_sequence', n > 0 and len(rows) == n and np.array_equal(a['step'], np.arange(1, n+1)))
    check('trace_step_sequence', [r['step'] for r in rows] == list(range(1, n+1)))
    if not report['checks'][-2]['passed'] or not report['checks'][-1]['passed']:
        report['status'] = 'inconsistent_evidence'
        return report, 1

    rnames = a['reward_term_names'].tolist()
    tnames = a['termination_term_names'].tolist()
    expected_reward = {
        'objects_in_circle': {'center': [-3., -10.], 'radius': 1., 'reward_per_object': 1.},
        'grasped_objects': {'ee_body_name': 'gripper_base', 'grasp_dist_thresh': .20, 'reward_per_object': 1.},
    }
    check('exact_original_reward_terms', set(rnames) == set(expected_reward)
          and all(meta['reward_terms'][k]['weight'] == 1.
                  and meta['reward_terms'][k]['params'] == v for k, v in expected_reward.items()), meta['reward_terms'])
    termination = meta['termination_terms']
    check('original_termination_parameters', set(tnames) == {'time_out', 'illegal_contact', 'fall', 'objects_in_circle_done'}
          and termination['illegal_contact']['params']['threshold'] == 1.
          and termination['fall']['params']['minimum_height'] == 0.
          and termination['objects_in_circle_done']['params'] == {'center': [-3., -10.], 'radius': 1.}
          and termination['time_out']['time_out'] is True
          and all(termination[k]['time_out'] is False for k in tnames if k != 'time_out'), termination)
    check('original_illegal_body_set', set(meta['illegal_contact']['body_names']) ==
          {'base_link', *[f'{corner}_{link}' for corner in ('FR','FL','RR','RL') for link in ('hip','thigh')]})
    check('recorded_original_configuration', result['task_physics_modified'] is False
          and result['action_spec'] is None and result['apply_safe_action_spec'] == 'called with no participant spec')
    expected_actions = [
        ('joint_leg', [0,12], 12, 'position', .5,
         [f'{c}_{j}_joint' for c in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]),
        ('joint_wheel', [12,16], 4, 'velocity', 5., [f'{c}_foot_joint' for c in ('FR','FL','RR','RL')]),
        ('joint_arm', [16,24], 8, 'position', .5, [f'arm_joint{i}' for i in range(1,9)]),
    ]
    action_terms = meta['action_schema']['terms']
    check('original_resolved_action_schema', meta['action_schema']['total_action_dim'] == 24
          and len(action_terms) == 3 and all(
              (t['name'], t['action_slice'], t['dim'], t['mode'], t['scale'], t['joint_names']) == expected
              and t['use_default_offset'] is True and t['clip'] in (None, 'None')
              and t['joint_names_source'] == 'resolved'
              for t,expected in zip(action_terms, expected_actions)))

    project_sources, official_sources, inactive_differences = [], [], []
    official_seen = set()
    policy_forbidden = []
    for filename, entry in manifest['hashed_files'].items():
        rel = entry['relative_path']
        if not check('safe_manifest_path:' + rel,
                     not Path(rel).is_absolute() and '..' not in Path(rel).parts):
            continue
        if entry['root'] == 'project':
            p = run / 'source_snapshots' / rel
            valid = p.is_file() and p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
            project_sources.append({'path': rel, 'sha256': entry['sha256'], 'snapshot_matches': valid})
            if valid and (rel in POLICY_FILES or (rel.startswith('task_b/')
                    and rel.endswith('.py') and rel not in {'task_b/evaluate.py', 'task_b/diagnostics.py'})):
                tree = ast.parse(p.read_text(), filename=rel)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr in {'scene', 'root_pos_w', 'root_quat_w', 'body_pos_w', 'env_origins'}:
                        policy_forbidden.append({'path': rel, 'line': node.lineno, 'attribute': node.attr})
        elif entry['root'] == 'atec_rl_lab':
            official_seen.add(rel)
            p = Path(filename)
            git = subprocess.run(['git', 'show', PIN + ':' + OFFICIAL_PREFIX + rel], cwd=ORIGINAL,
                                 capture_output=True, check=False)
            upstream_sha = hashlib.sha256(git.stdout).hexdigest() if git.returncode == 0 else None
            current_matches = (p.resolve() == (ORIGINAL / OFFICIAL_PREFIX / rel).resolve()
                               and p.is_file() and sha(p) == entry['sha256'])
            same = upstream_sha == entry['sha256']
            known_inactive = (rel == 'tasks/task_e/mdp/rewards.py'
                              and entry['sha256'] == KNOWN_INACTIVE_TASK_E_DIAGNOSTIC_SHA)
            item = {'path': rel, 'recorded_sha256': entry['sha256'], 'upstream_sha256': upstream_sha,
                    'current_file_matches_record': current_matches, 'matches_upstream': same,
                    'known_inactive_task_e_diagnostic_only': known_inactive}
            official_sources.append(item)
            if not same:
                inactive_differences.append(item)
    report.update(project_source_verification=project_sources,
                  official_source_verification=official_sources,
                  upstream_differences=inactive_differences)
    check('project_source_snapshots_match_manifest', len(project_sources) > 0 and all(x['snapshot_matches'] for x in project_sources))
    required_project = {'task_b/evaluate.py', 'task_b/diagnostics.py', 'task_b/control.py',
                        'task_a/tools/d1g2_taska_camera_compat.py'}
    if result.get('mode') in {'first_reach', 'reach_probe'}:
        required_project |= {'task_b/first_reach.py', 'task_b/arm_kinematics.py', 'task_b/visual_approach.py'}
    if result.get('stance_hold'):
        required_project.add('task_b/stance_hold.py')
    if result.get('stance_reference'):
        required_project.add('task_b/stance_reference.py')
    check('evaluator_and_policy_snapshots_present', required_project <= {x['path'] for x in project_sources})
    check('critical_official_sources_loaded', CRITICAL_OFFICIAL <= official_seen, sorted(official_seen))
    check('loaded_official_sources_match_current_files', all(x['current_file_matches_record'] for x in official_sources))
    check('task_b_and_shared_sources_match_pinned_upstream', bool(official_sources) and all(
        x['matches_upstream'] or x['known_inactive_task_e_diagnostic_only'] for x in official_sources))
    check('policy_snapshots_no_direct_simulator_truth_access', not policy_forbidden, policy_forbidden)

    shapes = {'reward_raw_total': (n,), 'score': (n,), 'reward_terms': (n,2), 'termination': (n,4),
              'gripper_xyz': (n,3), 'object_xyz': (n,18,3), 'q': (n,24), 'qdot': (n,24),
              'proprio': (n,84), 'action': (n,24), 'base_xyz': (n,3), 'base_quat': (n,4),
              'illegal_force': (n,9)}
    good_arrays = all(k in a and a[k].shape == s and np.isfinite(a[k]).all() for k,s in shapes.items())
    check('finite_telemetry_shapes', good_arrays, {k: list(a[k].shape) for k in shapes if k in a})
    if not good_arrays:
        report['status'] = 'inconsistent_evidence'
        return report, 1
    check('raw_score_equals_reward_sum', np.isclose(score, a['reward_raw_total'].sum(), atol=1e-6, rtol=1e-6))
    check('score_cumulative_each_step', np.allclose(a['score'], np.cumsum(a['reward_raw_total']), atol=1e-6, rtol=1e-6))
    check('term_sum_equals_raw_reward_each_step', np.allclose(a['reward_terms'].sum(axis=1), a['reward_raw_total'], atol=1e-6, rtol=1e-6))
    check('trace_matches_raw_reward', np.allclose([r['reward_raw_total'] for r in rows], a['reward_raw_total'], atol=1e-8))
    check('trace_matches_score', np.allclose([r['score'] for r in rows], a['score'], atol=1e-8))
    check('trace_matches_term_rewards', all(np.allclose([r['reward_terms'][k] for k in rnames], a['reward_terms'][i], atol=1e-8) for i,r in enumerate(rows)))
    check('nonnegative_integer_original_rewards', np.all(a['reward_terms'] >= -1e-7)
          and np.allclose(a['reward_terms'], np.rint(a['reward_terms']), atol=1e-6, rtol=0)
          and np.all(a['reward_terms'].sum(axis=0) <= 18. + 1e-6)
          and 0 <= score <= 36. + 1e-6)
    has_env_reward = ('env_reward' in a and a['env_reward'].shape == (n,)
                      and np.isfinite(a['env_reward']).all() and all('env_reward' in r for r in rows))
    check('original_env_reward_present_for_positive_run', has_env_reward or score == 0.,
          {'present': has_env_reward, 'historical_zero_only_exception': score == 0. and not has_env_reward})
    if has_env_reward:
        check('raw_reward_equals_env_reward_divided_by_dt_once',
              np.allclose(a['env_reward'] / dt, a['reward_raw_total'], atol=1e-6, rtol=1e-6))
        check('trace_env_reward_matches_telemetry',
              np.allclose([r['env_reward'] for r in rows], a['env_reward'], atol=1e-8, rtol=0))
    check('boolean_termination_values', np.isin(a['termination'], [0, 1]).all())
    check('no_step_after_official_termination', not a['termination'][:-1].any())
    flags_by_step = [{k: bool(a['termination'][i,j]) for j,k in enumerate(tnames)} for i in range(n)]
    check('trace_termination_flags_match_telemetry', all(
        set(r['active_termination_terms']) == {k for k,v in flags_by_step[i].items() if v}
        and bool(r['truncated']) == flags_by_step[i]['time_out']
        and bool(r['terminated']) == any(v for k,v in flags_by_step[i].items() if k != 'time_out')
        for i,r in enumerate(rows)))
    check('result_terminal_flags_match_final_step',
          bool(result['terminated']) == bool(rows[-1]['terminated'])
          and bool(result['truncated']) == bool(rows[-1]['truncated'])
          and set(result['active_termination_terms_at_stop']) == set(rows[-1]['active_termination_terms']))
    schema = meta['action_schema']
    observed_names, articulation_names = meta['observations']['joint_names'], schema['articulation_joint_names']
    check('public_joint_mapping_is_complete_permutation', len(observed_names) == len(articulation_names) == 24
          and len(set(observed_names)) == 24 and set(observed_names) == set(articulation_names)
          and a['joint_names'].tolist() == articulation_names)
    order = [articulation_names.index(name) for name in observed_names]
    q_error = float(np.max(np.abs(a['proprio'][:,12:36]
                        + np.asarray(schema['default_joint_pos'])[order] - a['q'][:,order])))
    check('public_q_mapping_independently_recomputed', q_error < 1e-4, q_error)
    check('public_q_mapping_verified_each_step', all(np.isfinite(r['observation_joint_mapping_max_abs_error'])
          and 0 <= r['observation_joint_mapping_max_abs_error'] < 1e-4 for r in rows))

    events = result.get('scoring_events', [])
    event_file = run / 'scoring_events.json'
    positive_steps = (np.flatnonzero(a['reward_raw_total'] > 0) + 1).tolist()
    check('scoring_events_cover_exact_positive_steps', [e['step'] for e in events] == positive_steps)
    check('standalone_event_file_matches_result', (event_file.is_file() and load(event_file) == events) if events else not event_file.exists())
    check('first_positive_step_matches_telemetry', result.get('first_positive_step') == (positive_steps[0] if positive_steps else None))

    # Replay every available POST state. Array index i stores the PRE state of
    # step i+1, so step s post-state is array index s only if there was no reset.
    # The final step needs its own capture: same-index telemetry is never used.
    replay_reach, replay_circle = set(), set()
    replay_mismatches, termination_mismatches, missing_post_steps = [], [], []
    final_state = result.get('final_state_before_close')
    final_terminal = any(flags_by_step[-1].values())
    final_timing = 'terminal_pre_reset' if final_terminal else 'post_step_before_any_reset'
    final_complete = (isinstance(final_state, dict) and result.get('final_state_timing') == final_timing)
    check('final_post_state_available_for_positive_run', final_complete or score == 0.,
          {'present': final_complete, 'historical_zero_only_exception': score == 0. and not final_complete})
    if final_complete and final_terminal:
        pre = result.get('terminal_pre_reset') or {}
        check('final_state_matches_terminal_capture', pre.get('step_index') == n
              and pre.get('observer_error') is None
              and (pre.get('snapshot') or {}).get('state') == final_state)
    for s in range(1, n+1):
        idx = s-1
        if s < n and not any(flags_by_step[idx].values()):
            post = {k: a[k][s] for k in ('object_xyz', 'gripper_xyz', 'illegal_force', 'base_xyz')}
        elif s == n and final_complete:
            post = final_state
        elif s == n and final_terminal:
            post = ((result.get('terminal_pre_reset') or {}).get('snapshot') or {}).get('state')
        else:
            post = None
        if post is None:
            missing_post_steps.append(s)
            continue
        reached, inside = geometry_sets(post)
        expected_flags = {'illegal_contact': bool(np.max(post['illegal_force']) > 1.),
                          'fall': bool(np.asarray(post['base_xyz'])[2] < 0.),
                          'objects_in_circle_done': len(inside) == 18,
                          'time_out': s >= meta['max_episode_length_steps']}
        if expected_flags != flags_by_step[idx]:
            termination_mismatches.append({'step': s, 'expected': expected_flags, 'recorded': flags_by_step[idx]})
        expected = {'grasped_objects': len(reached-replay_reach), 'objects_in_circle': len(inside-replay_circle)}
        if not np.allclose([expected[k] for k in rnames], a['reward_terms'][idx], atol=1e-6, rtol=0):
            replay_mismatches.append({'step': s, 'expected': expected, 'recorded': a['reward_terms'][idx].tolist()})
        replay_reach |= reached
        replay_circle |= inside
    check('all_available_post_states_explain_unique_rewards', not replay_mismatches, replay_mismatches[:20])
    check('all_available_post_states_explain_termination_flags', not termination_mismatches, termination_mismatches[:20])
    check('complete_post_state_coverage_for_positive_run', not missing_post_steps or score == 0., missing_post_steps)
    report['reward_geometry_replay'] = {'replayed_steps': n-len(missing_post_steps),
        'missing_post_steps': missing_post_steps, 'mismatch_count': len(replay_mismatches),
        'note': 'next-step PRE samples only across nonterminal steps; dedicated final capture otherwise'}

    counted_reach, counted_circle = set(), set()
    for event in events:
        s = int(event['step']); idx = s-1
        echeck = {'step': s, 'state_timing': event.get('state_timing')}
        state = event.get('state')
        flags = {k: bool(a['termination'][idx,j]) for j,k in enumerate(tnames)}
        is_terminal = any(flags.values())
        check(f'event_{s}:term_flags_match_telemetry', event['termination_flags'] == flags)
        check(f'event_{s}:reward_terms_match_telemetry', np.allclose([event['reward_terms'][k] for k in rnames], a['reward_terms'][idx], atol=1e-8))
        check(f'event_{s}:env_and_raw_rewards_match_telemetry', has_env_reward
              and 'env_reward' in event and 'reward_raw_total' in event
              and np.isclose(event['env_reward'], a['env_reward'][idx], atol=1e-8, rtol=0)
              and np.isclose(event['reward_raw_total'], a['reward_raw_total'][idx], atol=1e-8, rtol=0))
        correct_timing = event.get('state_timing') == ('terminal_pre_reset' if is_terminal else 'post_step_before_any_reset')
        check(f'event_{s}:state_is_pre_reset_physics_result', isinstance(state,dict) and correct_timing and event.get('capture_error') is None)
        if not isinstance(state,dict):
            report['events'].append(echeck)
            continue
        obj, ee = np.asarray(state['object_xyz'],dtype=np.float64), np.asarray(state['gripper_xyz'],dtype=np.float64)
        check(f'event_{s}:geometry_finite', obj.shape == (18,3) and ee.shape == (3,) and np.isfinite(obj).all() and np.isfinite(ee).all())
        distances = np.linalg.norm(obj-ee,axis=1)
        # Match the official float32 squared-norm comparison. The actual reward
        # is still authoritative; a sub-ULP CPU/GPU boundary disagreement fails
        # closed and must be inspected, never rounded into a score.
        reached, inside = geometry_sets(state)
        newly_reached, newly_inside = reached-counted_reach, inside-counted_circle
        check(f'event_{s}:unique_object_geometry_explains_reward',
              np.isclose(event['reward_terms']['grasped_objects'], len(newly_reached), atol=1e-6)
              and np.isclose(event['reward_terms']['objects_in_circle'], len(newly_inside), atol=1e-6),
              {'new_proximity_objects': [f'object_{i+1}' for i in sorted(newly_reached)],
               'new_circle_objects': [f'object_{i+1}' for i in sorted(newly_inside)]})
        counted_reach |= reached; counted_circle |= inside
        if s < n and not is_terminal:
            matches = all(np.allclose(np.asarray(state[k]), a[k][s], atol=2e-6, rtol=0)
                          for k in ('gripper_xyz','object_xyz','q','qdot','base_xyz','base_quat'))
            check(f'event_{s}:poststate_matches_next_prestep', matches)
        if s == n and final_complete:
            check(f'event_{s}:matches_final_post_capture', state == final_state)
        if is_terminal:
            pre = result.get('terminal_pre_reset') or {}
            check(f'event_{s}:matches_terminal_capture', pre.get('step_index') == s and pre.get('observer_error') is None
                  and isinstance(pre.get('snapshot'),dict) and pre['snapshot'].get('state') == state)
            check(f'event_{s}:no_reset_image_presented_as_score_image', f'score_{s:05d}' not in result.get('rgb_frames',{}))
        echeck.update(nearest_object=f'object_{int(distances.argmin())+1}', minimum_gripper_root_distance_m=float(distances.min()),
                      distance_margin_below_20cm_m=float(.2-distances.min()), distances_m=distances.tolist(),
                      newly_reached_objects=[f'object_{i+1}' for i in sorted(newly_reached)],
                      newly_delivered_objects=[f'object_{i+1}' for i in sorted(newly_inside)],
                      termination_flags=flags,
                      no_illegal_or_fall_at_score=not (flags.get('illegal_contact') or flags.get('fall')))
        report['events'].append(echeck)

    no_errors = all(x['passed'] for x in report['checks'])
    positive = np.isfinite(score) and score > 0 and len(events) > 0
    report.update(verified_positive_raw_score=bool(no_errors and positive),
                  unique_proximity_object_count=len(counted_reach), unique_delivery_object_count=len(counted_circle),
                  no_illegal_or_fall_entire_run=not any(f['illegal_contact'] or f['fall'] for f in flags_by_step),
                  quality_limits=[f'{k} at step {i+1}' for i,f in enumerate(flags_by_step)
                                  for k in ('illegal_contact','fall') if f[k]],
                  full_task_pass_claimed=False,
                  failed_checks=[x['name'] for x in report['checks'] if not x['passed']])
    report['status'] = ('verified_positive_score' if positive else 'not_achieved_zero_score') if no_errors else 'inconsistent_evidence'
    return report, (0 if positive else 2) if no_errors else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();run=args.run_directory.resolve()
    try:
        report, code = audit(run)
    except Exception as exc:
        report, code = {'run_directory': str(run), 'status': 'audit_error',
                        'verified_positive_raw_score': False, 'error': f'{type(exc).__name__}: {exc}'}, 1
    target=args.output or run/'independent_positive_audit.json'
    target.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(target),'status':report['status'],
                      'verified_positive_raw_score':report['verified_positive_raw_score'],
                      'failed_checks':report.get('failed_checks',[]),'error':report.get('error')},ensure_ascii=False))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
