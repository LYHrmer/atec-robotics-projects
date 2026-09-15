"""Read-only completed-run command/state audit. No simulation or policy edits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_directories', type=Path, nargs='+')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
reports = []
for path in args.run_directories:
    result = json.loads((path/'result.json').read_text())
    metadata = json.loads((path/'environment_metadata.json').read_text())
    trace = [json.loads(line) for line in (path/'trace.jsonl').open()]
    data = np.load(path/'telemetry.npz')
    dt = float(data['dt'])
    assert len(trace) == result['steps'] == len(data['step'])
    assert np.array_equal(data['step'], [row['step'] for row in trace])
    assert np.array_equal(data['step'], np.arange(1, result['steps']+1))
    schema = metadata['action_schema']
    defaults = dict(zip(schema['articulation_joint_names'], schema['default_joint_pos']))
    names = data['joint_names'].tolist()
    arm = next(term for term in schema['terms'] if term['name'] == 'joint_arm')
    wheels = next(term for term in schema['terms'] if term['name'] == 'joint_wheel')
    canonical = ['arm_joint'+str(i) for i in range(1, 9)]
    arm_ids = [names.index(name) for name in canonical]
    wheel_ids = [names.index(name) for name in wheels['joint_names']]
    action_ids = [arm['action_slice'][0]+arm['joint_names'].index(name) for name in canonical]
    commands = data['action'][:, action_ids]*arm['scale']+np.array([defaults[name] for name in canonical])
    actual, rates = data['q'][:, arm_ids], data['qdot'][:, arm_ids]
    phases = np.array([row['policy_debug'].get('probe_phase') or '' for row in trace])
    probe = np.flatnonzero(phases != '')
    lift = np.flatnonzero(np.isin(phases, ['PROBE_LIFT', 'PROBE_OBSERVE']))
    report = {'run_id': path.name, 'steps': result['steps'], 'sim_seconds': result['sim_seconds'],
              'stop_reason': result['stop_reason'], 'score_raw_total': result['score_raw_total'],
              'official_terminated': result['terminated'], 'official_truncated': result['truncated'],
              'timing': 'telemetry actual q/qdot and trace diagnostics are pre_step; same row action is '
                        'computed from this observation. final_state and policy_stop_record are after the '
                        'last completed action. No same-index prestate is labelled post-action.',
              'phases': {}, 'checks': {'arm_commands_not_changed_by_outer_base_control': bool(np.array_equal(
                  data['action'][:, action_ids], data['requested_action'][:, action_ids]))}}
    for phase in ('PROBE_CLOSE', 'PROBE_LIFT', 'PROBE_OBSERVE'):
        indices = np.flatnonzero(phases == phase)
        if not len(indices):
            continue
        tail = indices[-min(len(indices), round(1./dt)):]
        report['phases'][phase] = {
            'first_step': int(data['step'][indices[0]]), 'last_step': int(data['step'][indices[-1]]),
            'tail_mean_actual_rad_m': actual[tail].mean(axis=0).tolist(),
            'tail_mean_command_minus_actual_rad_m': (commands[tail]-actual[tail]).mean(axis=0).tolist(),
            'tail_actual_range_rad_m': np.ptp(actual[tail], axis=0).tolist(),
            'tail_max_abs_qdot_rad_m_per_s': np.max(np.abs(rates[tail]), axis=0).tolist(),
            'width_min_max_m': [float(np.min(actual[indices, 6]-actual[indices, 7])),
                                float(np.max(actual[indices, 6]-actual[indices, 7]))]}
    if len(probe):
        debug = [trace[index]['policy_debug'] for index in probe]
        brake = [trace[index]['brake_wheel_hold_debug'] for index in probe]
        anchor = np.asarray([data['q'][index, wheel_ids]+np.array(brake[i]['wrapped_errors'])
                             for i, index in enumerate(probe)])
        anchor_change = np.arctan2(np.sin(anchor-anchor[0]), np.cos(anchor-anchor[0]))
        goal = np.array([trace[index]['policy_debug']['probe_lift_goal_rad'] for index in lift]) if len(lift) else None
        report['checks'].update(
            probe_lowering_alpha_held_at_one=all(row.get('probe_lowering_alpha') == 1. for row in debug),
            continuous_wheel_hold_request=all(row.get('wheel_hold_requested') for row in debug),
            continuous_active_wheel_hold=all(row.get('active') for row in brake),
            reconstructed_wheel_anchor_unchanged_modulo_2pi=bool(np.max(np.abs(anchor_change)) < 1e-5),
            goal_latched_once_and_fixed=bool(goal is not None and np.all(goal == goal[0])),
            complete_preload_command_before_lift=bool(len(lift) and trace[lift[0]]['policy_debug']['probe_finger_target_commanded']))
        report['wheel_anchor_max_change_rad'] = float(np.max(np.abs(anchor_change)))
        report['last_trace_actual'] = {key: debug[-1].get(key) for key in (
            'probe_lift_max_q_error_rad', 'probe_lift_max_qdot_rad_s', 'probe_pinch_rise_m',
            'probe_measured_width_m', 'probe_base_displacement_m', 'probe_base_yaw_rad',
            'probe_base_gravity_change_rad', 'probe_q2_position_feedback_rad')}
        valid = lift[lift > 0]
        if len(valid):
            peak_slew = float(np.max(np.abs(commands[valid, :6]-commands[valid-1, :6])))
            tether = float(np.max(np.abs(commands[valid, :6]-actual[valid, :6])))
            hard = np.array([metadata['articulation']['joint_pos_limits'][schema['articulation_joint_names'].index(name)]
                             for name in canonical[:6]])
            report['max_arm_command_step_rad'], report['max_measured_tether_rad'] = peak_slew, tether
            report['checks'].update(
                lift_arm_slew_at_most_point_one_rad_per_s=peak_slew <= .1*dt+2e-7,
                lift_arm_tether_at_most_point_one_rad=tether <= .1+2e-7,
                lift_commands_inside_original_hard_limits=bool(np.all(commands[valid, :6] >= hard[:, 0]-2e-7)
                    and np.all(commands[valid, :6] <= hard[:, 1]+2e-7)))
        close = np.flatnonzero(phases == 'PROBE_CLOSE')
        close = close[close > 0]
        if len(close):
            finger_slew = float(np.max(np.abs(commands[close, 6:]-commands[close-1, 6:])))
            report['max_close_finger_command_step_m'] = finger_slew
            report['checks']['close_finger_slew_at_most_point_zero_five_m_per_s'] = finger_slew <= .05*dt+2e-7
        report['final_policy_stop_probe_state'] = result.get('policy_stop_record', {}).get('policy', {}).get('probe_state')
    report['source_sha256'] = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                               for name in ('result.json', 'telemetry.npz', 'trace.jsonl', 'source_manifest.json')}
    report['all_command_checks_pass'] = all(report['checks'].values())
    reports.append(report)
content = json.dumps({'scope': __doc__, 'runs': reports}, indent=2)+'\n'
args.output.write_text(content)
print(content)
