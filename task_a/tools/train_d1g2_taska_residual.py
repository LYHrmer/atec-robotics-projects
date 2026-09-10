"""Train a D1+G2 residual locomotion policy; validate success on the original course separately."""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import deque
import hashlib
import json
import math
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
parser.add_argument('--profile', choices=('mixed', 'upstairs', 'rough'), default='mixed')
parser.add_argument('--seed', type=int, default=42)
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
from tools.d1g2_taska_train_env import build_d1g2_taska_train_cfg
from tools.d1g2_taska_residual import ResidualState, ACTOR_OBS_DIM, RESIDUAL_SCALES, RESIDUAL_CLIP

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


class ResidualVecEnv:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = 16
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length
        self.state = ResidualState(args.policy, self.num_envs, self.device)
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
        actor = self.state.observe(raw['proprio'], self.env.command_manager.get_command('base_velocity'))
        critic = torch.cat((actor, raw['proprio'][:, :3], raw['critic']), dim=-1)
        if not torch.isfinite(critic).all():
            raise FloatingPointError('Nonfinite training observation')
        return TensorDict({'policy': actor, 'critic': critic}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self.obs

    def step(self, residual):
        if time.monotonic() - self.last_resource_check > 10:
            check_resources(resources())
            self.last_resource_check = time.monotonic()
        if STOP_REQUESTED:
            raise TrainingStop('signal_requested')
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite residual action')
        raw, rewards, terminated, truncated, extras = self.env.step(self.state.combine(residual))
        done = terminated | truncated
        self.state.reset(done.nonzero().flatten())
        self.obs = self._observe(raw)
        extras['time_outs'] = truncated
        return self.obs, rewards, done.long(), extras


class RecordedTrainingEnv(ManagerBasedRLEnv):
    def __init__(self, *pos, **kw):
        self.completed_distances = deque(maxlen=512)
        self.failure_counts = {}
        self.terrain_episode_bins = {}
        super().__init__(*pos, **kw)

    def _reset_idx(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(active):
            distances = self.scene['robot'].data.root_pos_w[active, 0] - self.scene.env_origins[active, 0]
            distance_values = distances.detach().cpu().tolist()
            self.completed_distances.extend(distance_values)
            # ManagerBasedRLEnv._reset_idx begins with curriculum_manager.compute,
            # which can replace levels and origins. Capture the completed episode's
            # terrain and displacement before delegating to that implementation.
            terrain = self.scene.terrain
            terrain_types = terrain.terrain_types[active].detach().cpu().tolist()
            terrain_levels = terrain.terrain_levels[active].detach().cpu().tolist()
            termination_flags = {}
            for name in self.termination_manager.active_terms:
                flags = self.termination_manager.get_term(name)[active]
                count = int(flags.sum())
                self.failure_counts[name] = self.failure_counts.get(name, 0) + count
                if name in ('time_out', 'illegal_contact', 'bad_orientation'):
                    termination_flags[name] = flags.detach().cpu().tolist()
            for index, (column, level, distance) in enumerate(zip(terrain_types, terrain_levels, distance_values)):
                counts = self.terrain_episode_bins.setdefault((int(column), int(level)), {
                    'episode_count': 0, 'forward4m_count': 0,
                    'timeout_count': 0, 'illegal_contact_count': 0,
                    'bad_orientation_count': 0, 'distance_sum_m': 0.0,
                    'invalid_distance_count': 0,
                })
                counts['episode_count'] += 1
                if math.isfinite(distance):
                    counts['distance_sum_m'] += distance
                    counts['forward4m_count'] += int(distance > 4.0)
                else:
                    counts['invalid_distance_count'] += 1
                for term, key in (('time_out', 'timeout_count'),
                                  ('illegal_contact', 'illegal_contact_count'),
                                  ('bad_orientation', 'bad_orientation_count')):
                    if term in termination_flags:
                        counts[key] += int(termination_flags[term][index])
        super()._reset_idx(env_ids)

    def terrain_diagnostics(self):
        """Snapshot completed training episodes by original column and level."""
        generator = self.scene.terrain.cfg.terrain_generator
        terrains = list(generator.sub_terrains.items())
        total_proportion = sum(cfg.proportion for _, cfg in terrains)
        cumulative = []
        running = 0.0
        for _, cfg in terrains:
            running += cfg.proportion / total_proportion
            cumulative.append(running)
        # Same deterministic column assignment as TerrainGenerator's curriculum.
        columns = []
        for column in range(generator.num_cols):
            terrain_index = next(i for i, upper in enumerate(cumulative)
                                 if column / generator.num_cols + 0.001 < upper)
            columns.append({'terrain_type_column': column, 'terrain_name': terrains[terrain_index][0]})
        bins = []
        for (column, level), counts in sorted(self.terrain_episode_bins.items()):
            valid_distances = counts['episode_count'] - counts['invalid_distance_count']
            bins.append({
                'terrain_type_column': column, 'terrain_name': columns[column]['terrain_name'],
                'terrain_level': level, **counts,
                'forward4m_rate': counts['forward4m_count'] / counts['episode_count'],
                'mean_distance_m': counts['distance_sum_m'] / valid_distances if valid_distances else None,
            })
        episode_count = sum(row['episode_count'] for row in bins)
        forward4m_count = sum(row['forward4m_count'] for row in bins)
        return {
            'scope': 'Completed episodes in this training process; counters restart on resume.',
            'distance_definition': 'Terminal root x minus the episode terrain origin x, before curriculum update.',
            'forward4m_definition': 'Terminal forward distance strictly greater than 4 m, including failed episodes.',
            'termination_counts_may_overlap': True,
            'full_course_reached_goal': False,
            'evaluation_required': 'These training diagnostics do not establish completion of the original Task A course.',
            'terrain_columns': columns, 'bins': bins,
            'episode_count': episode_count, 'forward4m_count': forward4m_count,
            'forward4m_rate': forward4m_count / episode_count if episode_count else None,
        }


class TrainingStop(Exception):
    pass


class GuardedRunner(OnPolicyRunner):
    def __init__(self, *pos, **kw):
        super().__init__(*pos, **kw)
        self.logger_type = 'tensorboard'
        self.started = time.monotonic()
        self.resource_peak = 0
        self.latest = {}

    def save(self, path, infos=None):
        terrain = self.env.env.scene.terrain
        infos = {'d1g2_residual': metadata,
                 'training_state': {'terrain_levels': terrain.terrain_levels.cpu(),
                                    'terrain_types': terrain.terrain_types.cpu(),
                                    'physical_episodes_restart_on_resume': True}}
        super().save(path, infos)

    def log(self, locs, *pos, **kw):
        # Keep normal TensorBoard statistics and readable diagnostics every 10 iterations.
        import contextlib
        import io
        if locs['it'] % 10 == 0:
            super().log(locs, *pos, **kw)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                super().log(locs, *pos, **kw)
        self.latest = {
            'status': 'training', 'iteration': locs['it'],
            'num_envs': self.env.num_envs, 'timesteps_this_run': self.tot_timesteps,
            'wall_seconds': time.monotonic()-self.started,
            'iteration_seconds': locs['collection_time']+locs['learn_time'],
            'mean_episode_reward': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
            'mean_episode_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None,
            'mean_terrain_level': self.env.env.scene.terrain.terrain_levels.float().mean().item(),
            'mean_forward_speed_mps': self.env.env.scene['robot'].data.root_lin_vel_b[:, 0].mean().item(),
            'mean_completed_forward_distance_m': statistics.mean(self.env.env.completed_distances) if self.env.env.completed_distances else None,
            'termination_counts': dict(self.env.env.failure_counts),
            'residual_noise_std': self.alg.policy.action_std.mean().item(),
            'loss': {k: float(v) for k, v in locs['loss_dict'].items()},
            'full_course_reached_goal': False,
        }
        if locs['it'] % 10 == 0:
            diagnostics = self.env.env.terrain_diagnostics()
            diagnostics['iteration'] = locs['it']
            diagnostics_path = args.output/'terrain_diagnostics.json'
            save_json(diagnostics_path, diagnostics)
            print('D1G2_TERRAIN_DIAGNOSTICS ' + json.dumps({
                'iteration': locs['it'], 'episode_count': diagnostics['episode_count'],
                'forward4m_rate': diagnostics['forward4m_rate'],
                'populated_bins': len(diagnostics['bins']), 'path': str(diagnostics_path),
                'full_course_reached_goal': False,
            }), flush=True)
            hardware = resources()
            self.resource_peak = max(self.resource_peak, hardware['gpu_memory_mib'])
            self.latest['resources'] = hardware
            check_resources(hardware)
        self.latest['sampled_gpu_peak_mib'] = self.resource_peak
        save_json(args.output/'progress.json', self.latest)
        with (args.output/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(self.latest, ensure_ascii=False, allow_nan=False)+'\n')
        if STOP_REQUESTED:
            raise TrainingStop('signal_requested')
        if time.monotonic()-self.started > args.max_wall_seconds:
            raise TrainingStop('wall_time_budget')


runner_cfg = {
    'num_steps_per_env': 24, 'save_interval': 50, 'logger': 'tensorboard',
    'obs_groups': {'policy': ['policy'], 'critic': ['critic']},
    'policy': {'class_name': 'ActorCritic', 'init_noise_std': 0.25, 'noise_std_type': 'log',
               'actor_hidden_dims': [256, 128, 64], 'critic_hidden_dims': [256, 128, 64],
               'activation': 'elu', 'actor_obs_normalization': True, 'critic_obs_normalization': True},
    'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1.0, 'use_clipped_value_loss': True,
                  'clip_param': 0.2, 'entropy_coef': 0.003, 'num_learning_epochs': 5,
                  'num_mini_batches': 4, 'learning_rate': 0.0003, 'schedule': 'adaptive',
                  'gamma': 0.99, 'lam': 0.95, 'desired_kl': 0.01, 'max_grad_norm': 1.0},
}
if args.profile == 'upstairs':
    runner_cfg['algorithm']['entropy_coef'] = .006
metadata = {'base_policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            'base_normalization_backend': 'onnx', 'interface_version': 1,
            'actor_obs_dim': ACTOR_OBS_DIM, 'residual_scales': RESIDUAL_SCALES,
            'residual_clip': RESIDUAL_CLIP,
            'policy_cfg': deepcopy(runner_cfg['policy']), 'seed': args.seed,
            'actor_observation': '5x57 history,7 arm positions,7 arm velocities*.05,16 base actions,16 last residuals',
            'training_model': 'Original D1+G2 combined USD with original actuators; arm held at default',
            'training_profile': args.profile,
            'evaluation_required': 'Original Task A x=-141 to x>145 without failed termination'}
if args.profile == 'rough':
    metadata['rough_training_distribution'] = {
        'kind': 'original_full_rough_all_rows',
        'noise_range_m': [.02, .10], 'noise_step_m': .02,
        'training_tile_size_m': [8., 8.], 'original_task_tile_size_m': [20., 20.],
        'rough_curriculum': False, 'other_terrain_curriculum': True,
    }


def main():
    global metadata
    env = None
    runner = None
    result = {'status': 'initializing', 'full_course_reached_goal': False, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output/'result.json', result)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = build_d1g2_taska_train_cfg(device=args.device, seed=args.seed, ddt_root=args.assets_root,
                                       num_envs=args.num_envs, profile=args.profile)
        cfg.sim.use_fabric = True
        print('D1G2_TRAIN_STAGE creating_environment', flush=True)
        env = RecordedTrainingEnv(cfg=cfg)
        wrapped = ResidualVecEnv(env)
        metadata['critic_obs_dim'] = wrapped.obs['critic'].shape[-1]
        save_json(args.output/'metadata.json', metadata)
        save_json(args.output/'runner_cfg.json', runner_cfg)
        (args.output/'sources').mkdir()
        for name in ('train_d1g2_taska_residual.py', 'd1g2_taska_train_env.py', 'd1g2_taska_env.py',
                     'd1g2_taska_residual.py', 'd1g2_taska_torch_policy.py'):
            shutil.copy2(ROOT/'tools'/name, args.output/'sources'/name)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(args.output/'env.yaml'), cfg)
        runner = GuardedRunner(wrapped, deepcopy(runner_cfg), log_dir=str(args.output), device=args.device)
        if args.resume:
            resume_infos = torch.load(args.resume, map_location='cpu', weights_only=False)['infos']
            resumed = resume_infos['d1g2_residual']
            for key in ('base_policy_sha256', 'base_normalization_backend', 'interface_version',
                        'actor_obs_dim', 'critic_obs_dim', 'residual_scales', 'residual_clip', 'policy_cfg'):
                if resumed.get(key) != metadata[key]:
                    raise ValueError(f'Resume interface mismatch for {key}: {resumed.get(key)} != {metadata[key]}')
            runner.load(str(args.resume), map_location=args.device)
            runner.current_learning_iteration += 1
            terrain = env.scene.terrain
            state = resume_infos['training_state']
            if len(state['terrain_levels']) != env.num_envs:
                raise ValueError('Resume requires the same number of parallel environments')
            changed_profile = resumed.get('training_profile', 'mixed') != args.profile
            if changed_profile:
                # Column meanings change with terrain proportions. Retain the
                # newly generated column assignments and start levels 0--1.
                terrain.terrain_levels[:] = torch.randint(0, 2, (env.num_envs,), device=env.device)
                result['curriculum_restart'] = {'from': resumed.get('training_profile', 'mixed'),
                                                'to': args.profile, 'initial_levels': [0, 1]}
            else:
                terrain.terrain_levels[:] = state['terrain_levels'].to(env.device)
                terrain.terrain_types[:] = state['terrain_types'].to(env.device)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            if changed_profile and args.profile in ('upstairs', 'rough'):
                # Increase thigh/calf exploration only when entering the new
                # phase; resuming that phase preserves its learned noise.
                ids = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], device=env.device)
                with torch.no_grad():
                    log_std = runner.alg.policy.log_std
                    log_std[ids] = torch.maximum(log_std[ids], torch.full_like(log_std[ids], math.log(.45)))
                result['thigh_calf_initial_noise_floor'] = .45
            wrapped.state.reset()
            raw, _ = env.reset(seed=args.seed)
            wrapped.obs = wrapped._observe(raw)
        else:
            # Exactly preserve the base actor mean before the first PPO update.
            with torch.no_grad():
                runner.alg.policy.actor[-1].weight.zero_()
                runner.alg.policy.actor[-1].bias.zero_()
        print('D1G2_TRAIN_READY '+json.dumps({'num_envs': env.num_envs, 'actor_dim': ACTOR_OBS_DIM,
                                            'critic_dim': metadata['critic_obs_dim'], 'resources': resources()}), flush=True)
        result.update(status='training')
        save_json(args.output/'result.json', result)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        result.update(status='training_batch_finished', reason='iterations_completed')
    except TrainingStop as exc:
        result.update(status='stopped', reason=str(exc))
    except BaseException as exc:
        result.update(status='error', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if runner is not None:
            checkpoint = args.output/f'model_{runner.current_learning_iteration}_final.pt'
            try:
                runner.save(str(checkpoint))
                result.update(checkpoint=str(checkpoint.resolve()), latest=runner.latest)
            except Exception as exc:
                result['checkpoint_error'] = repr(exc)
        save_json(args.output/'result.json', result)
        print('D1G2_TRAIN_RESULT '+json.dumps(result, ensure_ascii=False), flush=True)
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
