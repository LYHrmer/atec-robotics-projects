"""CPU boundary audit: real Isaac action/reward methods, no simulation App.

Run with the Isaac Lab Python environment and PYTHONNOUSERSITE=1. Source AST
extraction executes only the named methods against CPU stand-ins; it does not
import Isaac, construct a scene, or exercise physics. Output is audit metadata.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_b import control
from task_b.diagnostics import PreResetRecorder


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    # Postponed annotations avoid importing simulation-only type dependencies.
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


def audit_recorder():
    class Env:
        def __init__(self):
            self.buffer = np.array([9., 8.]); self.calls = []; self.error = None; self.token = object()
        def _reset_idx(self, env_ids, **kwargs):
            self.calls.append((list(env_ids), kwargs.copy())); self.buffer[:] = 0
            if self.error:
                raise self.error
            return self.token

    checks = {}
    env = Env(); ids = [0]
    def callback(copied_ids):
        copied_ids[:] = [99]
        return {'array': env.buffer, 'nested': [env.buffer]}
    with PreResetRecorder(env, callback) as recorder:
        assert env._reset_idx(ids) is env.token and recorder.last_record is None
        checks['initial_reset_excluded'] = recorder.ignored_resets == 1
        env.buffer[:] = [4., 5.]
        with recorder.step():
            assert env._reset_idx(ids, extra='unchanged') is env.token
        saved = recorder.last_record
        checks['copy_precedes_mutable_reset'] = saved['snapshot']['array'].tolist() == [4., 5.] and saved['snapshot']['nested'][0].tolist() == [4., 5.]
        checks['callback_ids_cannot_mutate_reset_ids'] = ids == [0] and env.calls[-1] == ([0], {'extra': 'unchanged'})
        checks['one_original_call_and_return_identity'] = len(env.calls) == 2
        with recorder.step():
            pass
        checks['next_step_has_no_stale_snapshot'] = recorder.last_record is None
    checks['restores_class_method_lookup'] = '_reset_idx' not in vars(env)
    recorder.close()
    env = Env()
    def bad_callback(ids):
        raise ValueError('observer-only failure')
    with PreResetRecorder(env, bad_callback) as recorder:
        with recorder.step():
            assert env._reset_idx([0]) is env.token
        checks['observer_failure_still_resets'] = len(env.calls) == 1 and recorder.last_record['snapshot'] is None and recorder.last_record['observer_error']['type'] == 'ValueError'
        env.error = RuntimeError('official-reset-error')
        try:
            with recorder.step():
                env._reset_idx([0])
        except RuntimeError as error:
            checks['original_exception_identity_preserved'] = error is env.error and len(env.calls) == 2
        else:
            checks['original_exception_identity_preserved'] = False
    env = Env(); original = env._reset_idx; env._reset_idx = original
    with PreResetRecorder(env, lambda ids: {}) as recorder:
        with recorder.step():
            env._reset_idx([0]); env._reset_idx([1])
        checks['multiple_resets_preserved'] = len(recorder.step_records) == 2 and recorder.total_step_resets == 2
    checks['restores_prior_instance_override'] = env._reset_idx is original
    assert all(checks.values()), checks
    return checks


def audit_actions(source_root, lab_root):
    joint_path = lab_root / 'source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py'
    manager_path = lab_root / 'source/isaaclab/isaaclab/managers/action_manager.py'
    joint_process = extract(joint_path, 'process_actions', {'torch': torch}, 'JointAction')
    manager_process = extract(manager_path, 'process_action', {'torch': torch}, 'ActionManager')
    read_schema = extract(ROOT / 'task_b/evaluate.py', 'read_schema', {
        'control': control, 'ROBOT': 'robot',
        'ACTION_MODES': {'JointPositionAction': 'position', 'JointVelocityAction': 'velocity'},
    })
    asset_tree = ast.parse((source_root / 'source/atec_rl_lab/atec_rl_lab/assets/robots/b2w.py').read_text())
    names = {}
    for node in asset_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            target = node.targets[0]
            if isinstance(target.value, ast.Name) and target.value.id == 'UNITREE_B2W_PIPER_CFG' and target.attr.endswith('joint_names'):
                names[target.attr] = tuple(ast.literal_eval(node.value))
    groups = {'joint_leg': names['leg_joint_names'], 'joint_wheel': names['wheel_joint_names'], 'joint_arm': names['arm_joint_names']}
    # Deliberately scrambled articulation order, independently of manager order.
    articulation_names = tuple(reversed(names['joint_names']))
    defaults, limits = [], []
    for name in articulation_names:
        if '_hip_' in name:
            defaults.append(.1 if name[1] == 'L' else -.1); limits.append([-.783, .783])
        elif '_thigh_' in name:
            defaults.append(.8 if name[0] == 'F' else 1.); limits.append([-.6585, 4.4085])
        elif '_calf_' in name:
            defaults.append(-1.5); limits.append([-2.7005, -.5495])
        else:
            defaults.append(0.); limits.append([-10., 10.])
    defaults = np.asarray(defaults); limits = np.asarray(limits)
    errors = []; schemas = []; cases = 0
    for permutation in itertools.permutations(groups):
        for scale_set in ({'joint_leg': .5, 'joint_wheel': 5., 'joint_arm': .5},
                          {'joint_leg': .23, 'joint_wheel': 2.7, 'joint_arm': .7}):
            live_terms = {}
            for key in permutation:
                joint_names = groups[key][::-1]  # also scramble within each term
                kind = 'JointVelocityAction' if key == 'joint_wheel' else 'JointPositionAction'
                term = type(kind, (), {})()
                term._joint_names = list(joint_names); term.action_dim = len(joint_names)
                term.cfg = NS(joint_names=['.*'], scale=scale_set[key], use_default_offset=True, clip=None)
                term._raw_actions = torch.zeros((1, len(joint_names)), dtype=torch.float64)
                term._scale = scale_set[key]
                term._offset = torch.tensor([[0. if key == 'joint_wheel' else defaults[articulation_names.index(j)] for j in joint_names]])
                term.process_actions = MethodType(joint_process, term)
                live_terms[key] = term
            manager = NS(active_terms=list(live_terms), action_term_dim=[t.action_dim for t in live_terms.values()],
                         total_action_dim=24, _terms=live_terms, device='cpu', _action=torch.zeros((1, 24)), _prev_action=torch.zeros((1, 24)))
            manager.get_term = live_terms.__getitem__
            robot = NS(data=NS(joint_names=articulation_names, default_joint_pos=torch.tensor(defaults)[None], soft_joint_pos_limits=torch.tensor(limits)[None]))
            schema = read_schema(NS(action_manager=manager, scene={'robot': robot})); schemas.append(schema)
            for mode in control.MODES:
                policy = control.BootstrapPolicy(schema, mode)
                for call in range(1, 201):
                    action = policy.act(np.zeros(84))
                    if call <= 100:
                        assert np.array_equal(action, np.zeros(24)), (mode, call)
                    if call in (100, 101, 150, 200):
                        manager_process(manager, torch.from_numpy(action)[None])
                        alpha = {100: 0., 101: .01, 150: .5, 200: 1.}[call]
                        for key, term in live_terms.items():
                            for index, joint in enumerate(term._joint_names):
                                default = defaults[articulation_names.index(joint)]
                                if key == 'joint_wheel':
                                    sign = (-1. if joint[1] == 'L' else 1.) if mode == 'turn' else 1.
                                    expected = alpha * .1 * scale_set[key] * sign if mode in ('forward', 'turn') else 0.
                                elif key == 'joint_leg' and mode == 'crouch':
                                    physical_goal = {'hip': 0., 'thigh': 1., 'calf': -2.}[joint.split('_')[1]]
                                    expected = (1.-alpha)*default + alpha*physical_goal
                                else:
                                    expected = default
                                errors.append(abs(float(term._processed_actions[0, index])-expected))
                        cases += 1
    assert max(errors) < 6e-8, max(errors)
    schema = schemas[0]
    continuous_limits = schema.soft_joint_pos_limits.copy()
    wheel_ids = [schema.joint_index(name) for name in groups['joint_wheel']]
    continuous_limits[wheel_ids] = [[np.nan, np.nan], [-np.inf, np.inf], [np.nan, np.inf], [-np.inf, np.nan]]
    continuous = replace(schema, soft_joint_pos_limits=continuous_limits)
    continuous.validate()
    json.dumps(continuous.to_dict(), allow_nan=False)
    for mode in control.MODES:
        continuous_policy = control.BootstrapPolicy(continuous, mode)
        assert np.isfinite(continuous_policy.act(np.zeros(84))).all()
    malformed_limits = schema.soft_joint_pos_limits.copy()
    malformed_limits[schema.joint_index(groups['joint_leg'][0])] = [1., -1.]
    nonfinite_arm_limits = schema.soft_joint_pos_limits.copy()
    nonfinite_arm_limits[schema.joint_index(groups['joint_arm'][0])] = [np.nan, np.inf]
    invalid = [replace(schema, total_dim=23), replace(schema, terms=tuple(replace(t, scale=0.) if t.name == 'joint_leg' else t for t in schema.terms)),
               replace(schema, terms=tuple(replace(t, mode='position') if t.name == 'joint_wheel' else t for t in schema.terms)),
               replace(schema, terms=tuple(replace(t, use_default_offset=False) if t.name == 'joint_arm' else t for t in schema.terms)),
               replace(schema, soft_joint_pos_limits=malformed_limits),
               replace(schema, soft_joint_pos_limits=nonfinite_arm_limits),
               replace(schema, terms=tuple(replace(t, joint_names=(t.joint_names[0],)*t.dim) if t.name == 'joint_leg' else t for t in schema.terms))]
    for bad in invalid:
        try:
            bad.validate()
        except control.SchemaError:
            pass
        else:
            raise AssertionError('invalid schema accepted')
    return {'passed': True, 'schemas': len(schemas), 'physical_command_cases': cases,
            'manager_and_joint_order_permuted': True, 'different_scales_and_nonzero_default_offsets': True,
            'max_abs_physical_command_error': max(errors), 'settle_first_100_calls_zero': True,
            'invalid_schema_rejections': len(invalid),
            'continuous_wheel_nan_inf_unused_position_limits_accepted': True,
            'continuous_wheel_limits_json_serializes_as_null': True,
            'consumer_methods_executed': [str(manager_path)+':ActionManager.process_action', str(joint_path)+':JointAction.process_actions'],
            'scope': 'CPU affine/slicing boundary. Does not establish physical wheel sign, stability, collision freedom, or reachability.'}


def audit_reward(lab_root):
    path = lab_root / 'source/isaaclab/isaaclab/managers/reward_manager.py'
    compute = extract(path, 'compute', {'torch': torch}, 'RewardManager')
    current = [1., 0.]
    cfgs = [NS(weight=1., params={}, func=lambda env, i=i: torch.tensor([current[i]])) for i in range(2)]
    manager = NS(_reward_buf=torch.zeros(1), _step_reward=torch.zeros((1,2)), _term_names=['grasped_objects','objects_in_circle'],
                 _term_cfgs=cfgs, _env=None, _episode_sums={'grasped_objects':torch.zeros(1),'objects_in_circle':torch.zeros(1)})
    values = []
    for event in ([1., 0.], [0., 0.], [0., 1.]):
        current[:] = event
        reward = float(compute(manager, .02)[0]); raw_terms = manager._step_reward[0].tolist()
        assert abs(sum(raw_terms)-reward/.02) < 1e-6
        values.append({'reward_returned_dt_scaled': reward, 'terms_already_raw': raw_terms})
    assert abs(sum(v['reward_returned_dt_scaled']/.02 for v in values)-2.) < 1e-6
    return {'passed':True, 'actual_method':str(path)+':RewardManager.compute', 'cases':values,
            'correct_raw_score':2., 'division_of_active_terms_would_be_incorrect':True,
            'scope':'Tests dt normalization of supplied one-time term events, not physical grasp detection.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/home/lybm/ATEC2026_Simulation_Challenge'))
    parser.add_argument('--isaaclab-root', type=Path, default=Path('/home/lybm/IsaacLab'))
    parser.add_argument('--output', type=Path, default=ROOT/'results/task_b_bootstrap_cpu_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [ROOT/'task_b/control.py', ROOT/'task_b/diagnostics.py', ROOT/'task_b/evaluate.py', Path(__file__),
               args.source_root/'source/atec_rl_lab/atec_rl_lab/assets/robots/b2w.py',
               args.isaaclab_root/'source/isaaclab/isaaclab/managers/action_manager.py',
               args.isaaclab_root/'source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py',
               args.isaaclab_root/'source/isaaclab/isaaclab/managers/reward_manager.py']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {'status':'passed', 'execution':'CPU only; no simulation application, no GPU, no original source writes',
              'recorder_checks':audit_recorder(), 'action_boundary':audit_actions(args.source_root,args.isaaclab_root),
              'reward_dt_boundary':audit_reward(args.isaaclab_root), 'source_sha256':hashes,
              'remaining_nonblocking_schema_hardening':[
                  'Wheel default velocity offset is zero in the official configuration, but is not represented in ActionSchema.',
                  'Non-None action clips are described but not refused; current official terms have clip=None.',
                  'Policy checks proprio finite/nonempty but does not enforce the actual 84-element shape; it is open-loop.',
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'recorder_checks':len(result['recorder_checks']), 'action_boundary':result['action_boundary'], 'output':str(args.output)},indent=2))


if __name__ == '__main__':
    main()
