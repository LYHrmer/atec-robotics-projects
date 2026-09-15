#!/usr/bin/env python3
"""Synthetic CPU evidence regressions only; never run or publish as Task B results."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil

import numpy as np


BASE = Path('/home/lybm/ATEC_Experiments_20260910')
SOURCE = BASE / 'task_b_score_cpu/positive_audit_v2_synthetic/valid_post_step'
OUTPUT = BASE / 'task_b_score_cpu/multi_score_audit_fixtures_20260914_v2'
AUDITOR = Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_b/audit_positive.py')
FROZEN_SHA = 'f31f81582633b4fc53157b96e403a1fea4b1e571911a504c2923aeef555904ca'
assert hashlib.sha256(AUDITOR.read_bytes()).hexdigest() == FROZEN_SHA
spec = importlib.util.spec_from_file_location('frozen_positive_audit', AUDITOR)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)

OUTPUT.mkdir(exist_ok=False)
(OUTPUT / 'SYNTHETIC_ONLY.txt').write_text('AUDITOR REGRESSION FIXTURES. NOT SIMULATION RESULTS. NO REAL SCORE.\n')
original_result = json.loads((SOURCE / 'result.json').read_text())
original_rows = [json.loads(x) for x in (SOURCE / 'trace.jsonl').read_text().splitlines()]
with np.load(SOURCE / 'telemetry.npz', allow_pickle=False) as archive:
    original_arrays = {k: archive[k].copy() for k in archive.files}

state_keys = ['gripper_xyz', 'object_xyz', 'q', 'qdot', 'base_xyz', 'base_quat', 'illegal_force']
reward_names = original_arrays['reward_term_names'].tolist()
termination_names = original_arrays['termination_term_names'].tolist()


def create_case(name, post_objects, expected_exit, terminal=False, mutate=None):
    run = OUTPUT / name
    shutil.copytree(SOURCE, run)
    (run / 'audit.json').unlink(missing_ok=True)
    (run / 'SYNTHETIC_ONLY.txt').write_text('SYNTHETIC MULTI-SCORE AUDITOR FIXTURE. NOT A REAL RUN.\n')
    result, arrays, rows = copy.deepcopy((original_result, original_arrays, original_rows))
    arrays['gripper_xyz'][:] = [0., 0., .25]
    arrays['object_xyz'][:] = [[3.+i, 3., .1] for i in range(18)]
    arrays['illegal_force'][:] = 0.
    arrays['termination'][:] = False
    result['fixture_notice'] = 'SYNTHETIC CPU MULTI-SCORE AUDITOR FIXTURE; NEVER A REAL RUN'
    result['rgb_frames'] = {}
    result['video'] = None
    states = [{k: arrays[k][i].tolist() for k in state_keys} for i in range(3)]
    for i, changes in enumerate(post_objects):
        for object_index, xyz in changes.items():
            states[i]['object_xyz'][object_index] = xyz
    if name == 'same_step_proximity_and_delivery':
        arrays['gripper_xyz'][:] = [-3., -10., .25]
        for state in states:
            state['gripper_xyz'] = [-3., -10., .25]
    if terminal:
        arrays['termination'][2, termination_names.index('illegal_contact')] = True
        states[2]['illegal_force'][0] = 2.
    for i in range(2):
        # This is POST step i+1, therefore PRE step i+2. Never same-index.
        for k in state_keys:
            arrays[k][i+1] = states[i][k]
    metadata = json.loads((run / 'environment_metadata.json').read_text())
    schema = metadata['action_schema']
    order = [schema['articulation_joint_names'].index(name)
             for name in metadata['observations']['joint_names']]
    arrays['proprio'][:,12:36] = arrays['q'][:,order] - np.asarray(schema['default_joint_pos'])[order]
    arrays['proprio'][:,36:60] = arrays['qdot'][:,order]
    reached, inside = set(), set()
    for i,state in enumerate(states):
        now_reached, now_inside = auditor.geometry_sets(state)
        terms = {'grasped_objects': len(now_reached-reached), 'objects_in_circle': len(now_inside-inside)}
        arrays['reward_terms'][i] = [terms[k] for k in reward_names]
        reached |= now_reached
        inside |= now_inside
    # Match the actual float32 environment scalar / Python float dt pathway.
    arrays['env_reward'] = (arrays['reward_terms'].sum(axis=1)*np.float32(.02)).astype(np.float32).astype(float)
    arrays['reward_raw_total'] = arrays['env_reward'] / .02
    arrays['score'] = np.cumsum(arrays['reward_raw_total'])
    events = []
    for i,row in enumerate(rows):
        flags = {k: bool(arrays['termination'][i,j]) for j,k in enumerate(termination_names)}
        terms = dict(zip(reward_names, arrays['reward_terms'][i].tolist()))
        row.update(reward_terms=terms, env_reward=float(arrays['env_reward'][i]),
                   reward_raw_total=float(arrays['reward_raw_total'][i]), score=float(arrays['score'][i]),
                   terminated=any(v for k,v in flags.items() if k != 'time_out'), truncated=flags['time_out'],
                   active_termination_terms=[k for k,v in flags.items() if v])
        if arrays['reward_raw_total'][i] > 0:
            events.append({'step': i+1, 'reward_terms': terms, 'env_reward': float(arrays['env_reward'][i]),
                           'reward_raw_total': float(arrays['reward_raw_total'][i]),
                           'state': states[i], 'state_timing': 'terminal_pre_reset' if any(flags.values()) else 'post_step_before_any_reset',
                           'capture_error': None, 'termination_flags': flags})
    result.update(score_raw_total=float(arrays['score'][-1]), scoring_events=events,
                  first_positive_step=events[0]['step'] if events else None,
                  reward_term_totals_raw=dict(zip(reward_names, arrays['reward_terms'].sum(axis=0).tolist())),
                  final_reward_terms=rows[-1]['reward_terms'], final_state_before_close=states[-1],
                  final_state_timing='terminal_pre_reset' if terminal else 'post_step_before_any_reset',
                  terminated=terminal, truncated=False, stop_reason='terminated' if terminal else 'max_steps',
                  active_termination_terms_at_stop=['illegal_contact'] if terminal else [])
    result['terminal_pre_reset'] = ({'step_index': 3, 'observer_error': None,
        'snapshot': {'state': states[-1], 'reward_terms': rows[-1]['reward_terms'],
                     'termination_flags': events[-1]['termination_flags']}} if terminal else None)
    if mutate:
        mutate(result, arrays, rows)
    (run / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (run / 'scoring_events.json').write_text(json.dumps(result['scoring_events'], indent=2)+'\n')
    (run / 'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    np.savez_compressed(run / 'telemetry.npz', **arrays)
    report, code = auditor.audit(run)
    (run / 'audit.json').write_text(json.dumps(auditor.plain(report), indent=2, allow_nan=False)+'\n')
    return {'name': name, 'synthetic_only': True, 'expected_exit': expected_exit, 'actual_exit': code,
            'passed': code == expected_exit, 'events': len(result['scoring_events']),
            'raw_score': result['score_raw_total'], 'unique_proximity_objects': report.get('unique_proximity_object_count'),
            'unique_delivery_objects': report.get('unique_delivery_object_count'),
            'quality_limits': report.get('quality_limits'), 'failed_checks': report.get('failed_checks', [])}


near1, near2, near3 = [0., 0., .1], [.05, 0., .1], [-.05, 0., .1]
far1 = [3., 3., .1]
cases = [
    create_case('two_events_two_objects', [{0:near1}, {0:near1,1:near2}, {0:near1,1:near2}], 0),
    create_case('same_step_three_objects', [{0:near1,1:near2,2:near3}]*3, 0),
    create_case('same_step_proximity_and_delivery', [{0:[-3.,-10.,.1]}]*3, 0),
    create_case('same_object_leave_and_reenter_no_new_point', [{0:near1}, {0:far1}, {0:near1}], 0),
    create_case('second_event_on_terminal_step', [{0:near1}, {0:near1}, {0:near1,1:near2}], 0, terminal=True),
    create_case('omit_second_event_rejected', [{0:near1}, {0:near1,1:near2}, {0:near1,1:near2}], 1,
                mutate=lambda r,a,rows:r['scoring_events'].pop()),
    create_case('same_index_prestate_as_second_post_rejected', [{0:near1}, {0:near1,1:near2}, {0:near1,1:near2}], 1,
                mutate=lambda r,a,rows:r['scoring_events'][1].__setitem__('state', {k:a[k][1].tolist() for k in state_keys})),
]
summary = {'scope':'SYNTHETIC CPU AUDITOR TESTS ONLY; NOT REAL TASK B PERFORMANCE',
           'auditor_sha256': FROZEN_SHA, 'all_passed': all(x['passed'] for x in cases), 'cases': cases}
(OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary))
raise SystemExit(0 if summary['all_passed'] else 1)
