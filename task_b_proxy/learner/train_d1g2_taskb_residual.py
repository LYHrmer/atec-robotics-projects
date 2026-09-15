"""Train a D1+G2 residual delivery policy for the local Task B proxy.

This trains only the proxy environment in ``d1g2_taskb_train_env.py``: one
kinematically carried object, scripted release, flat ground with one bin.
Success counters here are proxy-environment deliveries, never official Task B
results.  The locomotion backbone (``flat_lab.onnx``) stays frozen; the policy
learns 16 residual joint deltas plus the backbone's forward/yaw command.

``--probe_only`` builds the scene, holds the policy at zero action for a short
while and dumps robot/bin/object geometry, so the carry offset and release
radius can be checked against the real model before spending GPU hours.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--assets_root', type=Path, required=True)
parser.add_argument('--policy', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=128)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--resume', type=Path)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--episode_seconds', type=float, default=20.0)
parser.add_argument('--spawn_min', type=float, default=4.0)
parser.add_argument('--spawn_max', type=float, default=6.5)
parser.add_argument('--heading_noise', type=float, default=1.5707963267948966)
parser.add_argument('--release_radius', type=float, default=0.40)
parser.add_argument('--bin_scale', type=float, default=2.0)
parser.add_argument('--probe_only', action='store_true')
parser.add_argument('--max_wall_seconds', type=float, default=3600)
parser.add_argument('--gpu_memory_limit_mib', type=int, default=7600)
parser.add_argument('--gpu_temperature_limit', type=int, default=85)
parser.add_argument('--minimum_available_ram_gib', type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.num_envs <= 1024 or args.iterations < 1 or args.max_wall_seconds <= 0:
    parser.error('Invalid training size or wall time')
args.output.mkdir(parents=True, exist_ok=False)
if not args.experience:
    args.experience = '/home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit'
args.enable_cameras = False
app = AppLauncher(args).app

import torch
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from tools.d1g2_taska_residual import ResidualState, ACTOR_OBS_DIM, RESIDUAL_SCALES, RESIDUAL_CLIP
from tools.d1g2_taskb_train_env import (
    BIN_RADIUS, CARRY_OFFSET, GOAL_OBS_DIM, NUM_ACTIONS, NUM_RESIDUAL_ACTIONS,
    PRIVILEGED_OBS_DIM, RELEASE_SPEED, build_d1g2_taskb_train_cfg, bin_center_w,
    command_from_action, delivery_success, taskb_state,
)

TASKB_ACTOR_OBS_DIM = ACTOR_OBS_DIM + GOAL_OBS_DIM
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def resources():
    data = {}
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
    used, total, temperature, utilization = [int(v.strip()) for v in rows[0].split(',')]
    data.update(gpu_memory_mib=used, gpu_total_mib=total, gpu_temperature_c=temperature, gpu_utilization=utilization)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    data['available_ram_gib'] = int(memory['MemAvailable'].split()[0]) / 1024**2
    data['disk_free_gib'] = shutil.disk_usage(args.output).free / 1024**3
    return data


class TrainingStop(Exception):
    pass


def check_resources(hardware):
    reason = None
    if hardware['gpu_memory_mib'] > args.gpu_memory_limit_mib:
        reason = 'gpu_memory_guard'
    elif hardware['gpu_temperature_c'] >= args.gpu_temperature_limit:
        reason = 'gpu_temperature_guard'
    elif hardware['available_ram_gib'] < args.minimum_available_ram_gib:
        reason = 'available_ram_guard'
    elif hardware['disk_free_gib'] < 5:
        reason = 'disk_space_guard'
    if reason:
        raise TrainingStop(reason)


class TaskBResidualVecEnv:
    """Residual + learned-command wrapper around the Task B proxy environment.

    The policy's command channels take effect on the following control step:
    the backbone action for step ``t`` was computed from the command produced
    at ``t-1``.  One 20 ms delay, and the proprioception the learner sees always
    reports the command the backbone will actually use next.
    """

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = NUM_ACTIONS
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length
        self.state = ResidualState(args.policy, self.num_envs, self.device)
        self.task = taskb_state(env)
        self.last_resource_check = time.monotonic()
        check_resources(resources())
        raw, _ = env.reset(seed=args.seed)
        self.obs = self._observe(raw)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @torch.no_grad()
    def _observe(self, raw):
        actor = torch.cat((self.state.observe(raw['proprio'], self.task['command']), raw['goal']), dim=-1)
        if actor.shape[-1] != TASKB_ACTOR_OBS_DIM:
            raise ValueError(f'Unexpected actor observation width: {actor.shape}')
        critic = torch.cat((actor, raw['proprio'][:, :3], raw['critic']), dim=-1)
        if not torch.isfinite(critic).all():
            raise FloatingPointError('Nonfinite training observation')
        return TensorDict({'policy': actor, 'critic': critic}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self.obs

    def step(self, action):
        if time.monotonic() - self.last_resource_check > 10:
            check_resources(resources())
            self.last_resource_check = time.monotonic()
        if STOP_REQUESTED:
            raise TrainingStop('signal_requested')
        if not torch.isfinite(action).all():
            raise FloatingPointError('Nonfinite policy action')
        residual = action[:, :NUM_RESIDUAL_ACTIONS]
        self.task['command'][:] = command_from_action(action[:, NUM_RESIDUAL_ACTIONS:])
        raw, rewards, terminated, truncated, extras = self.env.step(self.state.combine(residual))
        done = terminated | truncated
        self.state.reset(done.nonzero().flatten())
        self.obs = self._observe(raw)
        extras['time_outs'] = truncated
        return self.obs, rewards, done.long(), extras


class RecordedTaskBEnv(ManagerBasedRLEnv):
    """Records the outcome of every completed proxy episode."""

    def __init__(self, *pos, **kw):
        self.episode_outcomes = deque(maxlen=1024)
        self.failure_counts = {}
        self.delivery_count = 0
        self.episode_count = 0
        super().__init__(*pos, **kw)

    def _reset_idx(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(active):
            state = taskb_state(self)
            success = delivery_success(self)[active]
            distances = torch.norm(
                self.scene['object'].data.root_pos_w[active, :2] - bin_center_w(self)[active], dim=1)
            terms = {}
            for name in self.termination_manager.active_terms:
                flags = self.termination_manager.get_term(name)[active]
                self.failure_counts[name] = self.failure_counts.get(name, 0) + int(flags.sum())
                terms[name] = flags
            self.episode_count += len(active)
            self.delivery_count += int(success.sum())
            for index in range(len(active)):
                self.episode_outcomes.append({
                    'success': bool(success[index]),
                    'final_object_dist_m': float(distances[index]),
                    'min_object_dist_m': float(state['min_dist'][active[index]]),
                    'released': bool(state['released'][active[index]]),
                    'steps': int(self.episode_length_buf[active[index]]),
                    'terminations': [name for name, flags in terms.items() if bool(flags[index])],
                })
        super()._reset_idx(env_ids)

    def delivery_diagnostics(self):
        """Snapshot of recent proxy episodes; counters restart on resume."""
        recent = list(self.episode_outcomes)
        state = taskb_state(self)
        summary = {
            'scope': 'Completed episodes of the Task B PROXY environment in this process.',
            'official_task_b_success': False,
            'objects_per_env': 1,
            'grasping': 'skipped (kinematic carry, scripted release)',
            'episode_count_total': self.episode_count,
            'delivery_count_total': self.delivery_count,
            'delivery_rate_total': self.delivery_count / self.episode_count if self.episode_count else None,
            'termination_counts': dict(self.failure_counts),
            'spawn_min_m': state['spawn_min'],
            'spawn_max_m': state['spawn_max'],
            'heading_noise_rad': state['heading_noise'],
            'success_ema': state['success_ema'],
        }
        if recent:
            summary.update(
                recent_episodes=len(recent),
                recent_delivery_rate=sum(row['success'] for row in recent) / len(recent),
                recent_release_rate=sum(row['released'] for row in recent) / len(recent),
                recent_mean_final_object_dist_m=statistics.mean(row['final_object_dist_m'] for row in recent),
                recent_mean_min_object_dist_m=statistics.mean(row['min_object_dist_m'] for row in recent),
                recent_mean_steps=statistics.mean(row['steps'] for row in recent),
            )
        return summary


class GuardedRunner(OnPolicyRunner):
    def __init__(self, *pos, **kw):
        super().__init__(*pos, **kw)
        self.logger_type = 'tensorboard'
        self.started = time.monotonic()
        self.resource_peak = 0
        self.latest = {}

    def save(self, path, infos=None):
        infos = {'d1g2_taskb_residual': metadata,
                 'training_state': {'physical_episodes_restart_on_resume': True,
                                    'spawn_curriculum': {k: taskb_state(self.env.env)[k]
                                                         for k in ('spawn_min', 'spawn_max', 'heading_noise')}}}
        super().save(path, infos)

    def log(self, locs, *pos, **kw):
        import contextlib
        import io
        if locs['it'] % 10 == 0:
            super().log(locs, *pos, **kw)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                super().log(locs, *pos, **kw)
        inner = self.env.env
        diagnostics = inner.delivery_diagnostics()
        self.latest = {
            'status': 'training', 'iteration': locs['it'],
            'num_envs': self.env.num_envs, 'timesteps_this_run': self.tot_timesteps,
            'wall_seconds': time.monotonic() - self.started,
            'iteration_seconds': locs['collection_time'] + locs['learn_time'],
            'mean_episode_reward': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
            'mean_episode_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None,
            'delivery_rate_total': diagnostics['delivery_rate_total'],
            'recent_delivery_rate': diagnostics.get('recent_delivery_rate'),
            'recent_release_rate': diagnostics.get('recent_release_rate'),
            'recent_mean_min_object_dist_m': diagnostics.get('recent_mean_min_object_dist_m'),
            'heading_noise_rad': diagnostics['heading_noise_rad'],
            'spawn_max_m': diagnostics['spawn_max_m'],
            'termination_counts': diagnostics['termination_counts'],
            'mean_command_vx': float(taskb_state(inner)['command'][:, 0].mean()),
            'mean_command_wz_abs': float(taskb_state(inner)['command'][:, 2].abs().mean()),
            'residual_noise_std': self.alg.policy.action_std.mean().item(),
            'loss': {k: float(v) for k, v in locs['loss_dict'].items()},
            'official_task_b_success': False,
        }
        if locs['it'] % 10 == 0:
            diagnostics['iteration'] = locs['it']
            path = args.output / 'delivery_diagnostics.json'
            save_json(path, diagnostics)
            print('D1G2_TASKB_DIAGNOSTICS ' + json.dumps({
                'iteration': locs['it'], 'episodes': diagnostics['episode_count_total'],
                'deliveries': diagnostics['delivery_count_total'],
                'recent_delivery_rate': diagnostics.get('recent_delivery_rate'),
                'recent_mean_min_object_dist_m': diagnostics.get('recent_mean_min_object_dist_m'),
                'heading_noise_rad': diagnostics['heading_noise_rad'],
                'path': str(path), 'official_task_b_success': False,
            }), flush=True)
            hardware = resources()
            self.resource_peak = max(self.resource_peak, hardware['gpu_memory_mib'])
            self.latest['resources'] = hardware
            check_resources(hardware)
        self.latest['sampled_gpu_peak_mib'] = self.resource_peak
        save_json(args.output / 'progress.json', self.latest)
        with (args.output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(self.latest, ensure_ascii=False, allow_nan=False) + '\n')
        if STOP_REQUESTED:
            raise TrainingStop('signal_requested')
        if time.monotonic() - self.started > args.max_wall_seconds:
            raise TrainingStop('wall_time_budget')


runner_cfg = {
    'num_steps_per_env': 24, 'save_interval': 50, 'logger': 'tensorboard',
    'obs_groups': {'policy': ['policy'], 'critic': ['critic']},
    'policy': {'class_name': 'ActorCritic', 'init_noise_std': 0.25, 'noise_std_type': 'log',
               'actor_hidden_dims': [256, 128, 64], 'critic_hidden_dims': [256, 128, 64],
               'activation': 'elu', 'actor_obs_normalization': True, 'critic_obs_normalization': True},
    'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1.0, 'use_clipped_value_loss': True,
                  'clip_param': 0.2, 'entropy_coef': 0.005, 'num_learning_epochs': 5,
                  'num_mini_batches': 4, 'learning_rate': 0.0003, 'schedule': 'adaptive',
                  'gamma': 0.99, 'lam': 0.95, 'desired_kl': 0.01, 'max_grad_norm': 1.0},
}
metadata = {'base_policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            'base_normalization_backend': 'onnx', 'interface_version': 1,
            'task': 'task_b_delivery_proxy', 'official_task_b': False,
            'actor_obs_dim': TASKB_ACTOR_OBS_DIM, 'goal_obs_dim': GOAL_OBS_DIM,
            'privileged_obs_dim': PRIVILEGED_OBS_DIM,
            'num_actions': NUM_ACTIONS, 'residual_scales': RESIDUAL_SCALES,
            'residual_clip': RESIDUAL_CLIP, 'carry_offset_m': list(CARRY_OFFSET),
            'release_radius_m': args.release_radius, 'release_speed_mps': RELEASE_SPEED,
            'spawn_ring_m': (args.spawn_min, args.spawn_max), 'bin_scale': args.bin_scale,
            'official_bin_radius_m': BIN_RADIUS, 'training_bin_radius_m': BIN_RADIUS * args.bin_scale,
            'policy_cfg': deepcopy(runner_cfg['policy']), 'seed': args.seed,
            'actor_observation': '5x57 history,7 arm positions,7 arm velocities*.05,16 base actions,'
                                 '16 last residuals,9 goal values',
            'action_layout': '16 residual joint deltas + forward command + yaw-rate command',
            'training_model': 'Original D1+G2 combined USD with original actuators; arm held at default',
            'object_model': '006_mustard_bottle.usd, one per environment, kinematically carried',
            'evaluation_required': 'Proxy deliveries only; the official 18-object Task B is not attempted'}


def probe_geometry(env):
    """Report the real model geometry the carry offset depends on."""
    robot = env.scene['robot']
    obj = env.scene['object']
    state = taskb_state(env)
    for _ in range(5):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    root = robot.data.root_pos_w[0]
    names = robot.body_names
    local = robot.data.body_pos_w[0] - root
    bin_center = bin_center_w(env)[0]
    bin_radius = state['bin_radius']
    robot_dist = float(torch.norm(root[:2] - bin_center))
    report = {
        'bin_center_w': bin_center.tolist(),
        'bin_scale': state['bin_scale'],
        'bin_radius_m': bin_radius,
        'robot_root_w': root.tolist(),
        'robot_dist_to_bin_center_m': robot_dist,
        'robot_clearance_to_bin_rim_m': robot_dist - bin_radius - 0.27,
        'robot_root_height_above_tile_m': float(root[2] - state['ground_z']),
        'object_pos_w': obj.data.root_pos_w[0].tolist(),
        'object_height_above_tile_m': float(obj.data.root_pos_w[0, 2] - state['ground_z']),
        'object_dist_to_bin_m': float(torch.norm(obj.data.root_pos_w[0, :2] - bin_center)),
        'carry_offset_m': list(CARRY_OFFSET),
        'body_local_positions': {name: [round(v, 4) for v in local[i].tolist()]
                                 for i, name in enumerate(names)},
        'max_body_forward_x_m': float(local[:, 0].max()),
        'min_body_forward_x_m': float(local[:, 0].min()),
        'illegal_contact_bodies': env.termination_manager.get_term_cfg(
            'illegal_contact').params['sensor_cfg'].body_names,
    }
    forward = {name: local[i, 0].item() for i, name in enumerate(names)}
    report['front_bodies_sorted'] = sorted(forward.items(), key=lambda kv: -kv[1])[:8]
    return report


def main():
    env = None
    runner = None
    result = {'status': 'initializing', 'official_task_b_success': False,
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output / 'result.json', result)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = build_d1g2_taskb_train_cfg(
            device=args.device, seed=args.seed, ddt_root=args.assets_root, num_envs=args.num_envs,
            episode_length_s=args.episode_seconds, spawn_min=args.spawn_min, spawn_max=args.spawn_max,
            heading_noise=args.heading_noise, release_radius=args.release_radius, bin_scale=args.bin_scale)
        cfg.sim.use_fabric = True
        print('D1G2_TASKB_STAGE creating_environment', flush=True)
        env = RecordedTaskBEnv(cfg=cfg)
        state = taskb_state(env)
        state.update(cfg.d1g2_taskb_initial_curriculum)
        ground = cfg.d1g2_taskb_ground_plane_spawner(env)
        print('D1G2_TASKB_STAGE ground_plane ' + json.dumps({
            'prim': '/World/ground_plane', 'vertices': int(len(ground.vertices))}), flush=True)
        if args.probe_only:
            report = probe_geometry(env)
            save_json(args.output / 'geometry_probe.json', report)
            print('D1G2_TASKB_PROBE ' + json.dumps(report), flush=True)
            result.update(status='probe_finished')
            return
        wrapped = TaskBResidualVecEnv(env)
        metadata['critic_obs_dim'] = wrapped.obs['critic'].shape[-1]
        metadata['env_metadata'] = cfg.d1g2_taskb_metadata
        save_json(args.output / 'metadata.json', metadata)
        save_json(args.output / 'runner_cfg.json', runner_cfg)
        (args.output / 'sources').mkdir()
        for name in ('train_d1g2_taskb_residual.py', 'd1g2_taskb_train_env.py', 'd1g2_taska_train_env.py',
                     'd1g2_taska_env.py', 'd1g2_taska_residual.py', 'd1g2_taska_torch_policy.py'):
            shutil.copy2(ROOT / 'tools' / name, args.output / 'sources' / name)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(args.output / 'env.yaml'), cfg)
        runner = GuardedRunner(wrapped, deepcopy(runner_cfg), log_dir=str(args.output), device=args.device)
        if args.resume:
            resume_infos = torch.load(args.resume, map_location='cpu', weights_only=False)['infos']
            resumed = resume_infos['d1g2_taskb_residual']
            # Only the observation/action interface must match for a warm start.
            # The task geometry (release radius, success radius, reward shape) is
            # deliberately allowed to change: that is the point of resuming after
            # a reward fix.  The change is recorded rather than hidden.
            for key in ('base_policy_sha256', 'base_normalization_backend', 'interface_version',
                        'actor_obs_dim', 'critic_obs_dim', 'num_actions', 'residual_scales',
                        'residual_clip', 'policy_cfg'):
                if resumed.get(key) != metadata[key]:
                    raise ValueError(f'Resume interface mismatch for {key}: {resumed.get(key)} != {metadata[key]}')
            geometry_keys = ('carry_offset_m', 'release_radius_m', 'release_speed_mps',
                             'bin_scale', 'training_bin_radius_m', 'spawn_ring_m')
            changed = {key: (resumed.get(key), metadata.get(key)) for key in geometry_keys
                       if resumed.get(key) != metadata.get(key)}
            if changed:
                result['resumed_with_changed_task'] = {
                    key: {'from': before, 'to': after} for key, (before, after) in changed.items()}
                print('D1G2_TASKB_RESUME_TASK_CHANGED ' + json.dumps(changed, default=str), flush=True)
            runner.load(str(args.resume), map_location=args.device)
            runner.current_learning_iteration += 1
            curriculum = resume_infos['training_state'].get('spawn_curriculum', {})
            state.update({k: v for k, v in curriculum.items() if k in ('spawn_min', 'spawn_max', 'heading_noise')})
            result['resumed_curriculum'] = {k: state[k] for k in ('spawn_min', 'spawn_max', 'heading_noise')}
            wrapped.state.reset()
            raw, _ = env.reset(seed=args.seed)
            wrapped.obs = wrapped._observe(raw)
        else:
            # Start from the unmodified backbone: zero residual and the neutral
            # forward command, so iteration 0 is the frozen walking policy.
            with torch.no_grad():
                runner.alg.policy.actor[-1].weight.zero_()
                runner.alg.policy.actor[-1].bias.zero_()
        print('D1G2_TASKB_READY ' + json.dumps({
            'num_envs': env.num_envs, 'actor_dim': TASKB_ACTOR_OBS_DIM,
            'critic_dim': metadata['critic_obs_dim'], 'num_actions': NUM_ACTIONS,
            'spawn': [state['spawn_min'], state['spawn_max']], 'heading_noise': state['heading_noise'],
            'resources': resources()}), flush=True)
        result.update(status='training')
        save_json(args.output / 'result.json', result)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        result.update(status='training_batch_finished', reason='iterations_completed')
    except TrainingStop as exc:
        result.update(status='stopped', reason=str(exc))
    except BaseException as exc:
        result.update(status='error', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if runner is not None:
            checkpoint = args.output / f'model_{runner.current_learning_iteration}_final.pt'
            try:
                runner.save(str(checkpoint))
                result.update(checkpoint=str(checkpoint.resolve()), latest=runner.latest)
            except Exception as exc:
                result['checkpoint_error'] = repr(exc)
        if env is not None:
            try:
                result['delivery_diagnostics'] = env.delivery_diagnostics()
            except Exception as exc:
                result['diagnostics_error'] = repr(exc)
        save_json(args.output / 'result.json', result)
        print('D1G2_TASKB_RESULT ' + json.dumps(result, ensure_ascii=False), flush=True)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
