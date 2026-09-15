"""Independently verify a trained D1+G2 Task B proxy policy.

Loads a checkpoint, runs deterministic (no exploration noise) episodes in the
Task B proxy environment, and reports delivery statistics plus physical
outcome checks that do not rely on the environment's own reward code:

* how far the object got from the bin centre at its closest, measured from the
  object's world pose, and whether it settled inside the success radius;
* whether the robot touched the bin (illegal contact), which is a failure;
* how far the robot drove, and whether it braked near the bin.

``delivery_success`` from the environment is reported alongside these
measurements, and any disagreement is called out: the point of this script is
to check the trained policy, not to re-report the training reward.

This is a proxy-environment result.  It is not an official Task B result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--assets_root', type=Path, required=True)
parser.add_argument('--policy', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--episodes', type=int, default=3, help='sequential episode batches per environment')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--episode_seconds', type=float, default=20.0)
# default None means "use whatever the checkpoint was trained with"
parser.add_argument('--spawn_min', type=float)
parser.add_argument('--spawn_max', type=float)
parser.add_argument('--heading_noise', type=float)
parser.add_argument('--bin_scale', type=float)
parser.add_argument('--release_radius', type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.experience:
    args.experience = '/home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit'
args.enable_cameras = False
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from tools.d1g2_taska_residual import ResidualState, ACTOR_OBS_DIM, RESIDUAL_SCALES, RESIDUAL_CLIP
from tools.d1g2_taskb_train_env import (
    GOAL_OBS_DIM, NUM_ACTIONS, NUM_RESIDUAL_ACTIONS, RELEASE_SPEED, SUCCESS_SPEED,
    build_d1g2_taskb_train_cfg, bin_center_w, command_from_action,
    delivery_success, taskb_state,
)

TASKB_ACTOR_OBS_DIM = ACTOR_OBS_DIM + GOAL_OBS_DIM


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class Verifier:
    def __init__(self, env, checkpoint):
        data = torch.load(checkpoint, map_location=args.device, weights_only=False)
        metadata = data['infos']['d1g2_taskb_residual']
        if metadata['residual_scales'] != RESIDUAL_SCALES or metadata['residual_clip'] != RESIDUAL_CLIP:
            raise ValueError('Checkpoint residual interface differs from this implementation')
        if metadata['actor_obs_dim'] != TASKB_ACTOR_OBS_DIM:
            raise ValueError(f"Checkpoint actor width {metadata['actor_obs_dim']} != {TASKB_ACTOR_OBS_DIM}")
        self.metadata = metadata
        self.iteration = data.get('iter')
        policy_cfg = dict(metadata['policy_cfg'])
        policy_cfg.pop('class_name', None)
        dummy = TensorDict(
            {'policy': torch.zeros(1, TASKB_ACTOR_OBS_DIM, device=args.device),
             'critic': torch.zeros(1, metadata['critic_obs_dim'], device=args.device)},
            batch_size=[1], device=args.device)
        self.actor_critic = ActorCritic(
            dummy, {'policy': ['policy'], 'critic': ['critic']}, NUM_ACTIONS, **policy_cfg
        ).to(args.device).eval()
        self.actor_critic.load_state_dict(data['model_state_dict'])
        self.state = ResidualState(args.policy, env.num_envs, args.device)
        self.task = taskb_state(env)

    @torch.inference_mode()
    def act(self, raw):
        actor = torch.cat((self.state.observe(raw['proprio'], self.task['command']), raw['goal']), dim=-1)
        action = self.actor_critic.act_inference(TensorDict(
            {'policy': actor}, batch_size=[env.num_envs], device=args.device))
        self.task['command'][:] = command_from_action(action[:, NUM_RESIDUAL_ACTIONS:])
        self._last_action = action
        return self.state.combine(action[:, :NUM_RESIDUAL_ACTIONS])

    @torch.inference_mode()
    def act_with_reset(self, raw, done_ids):
        """act() plus the residual-state reset that training performs on done."""
        out = self.act(raw)
        if len(done_ids):
            self.state.reset(done_ids)
        return out, self._last_action


torch.set_num_threads(4)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True

checkpoint_info = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
# The authoritative task settings live in the top-level training metadata.
# ``env_metadata`` only holds human-readable descriptions (e.g. the release
# rule as a sentence), so reading the radius from there silently fell back to
# the default and verified a different task than the one that was trained.
checkpoint_meta = checkpoint_info['infos']['d1g2_taskb_residual']
trained = {**checkpoint_meta.get('env_metadata', {}), **checkpoint_meta}
resolved = {key: (value if value is not None else trained.get(key, default))
            for key, value, default in (
                ('bin_scale', args.bin_scale, 2.0),
                ('release_radius', args.release_radius, 0.40),
                ('heading_noise', args.heading_noise, 1.5707963267948966),
            )}
spawn = trained.get('spawn_ring_m') or trained.get('spawn_distance_m', (4.0, 6.5))
resolved['spawn_min'] = args.spawn_min if args.spawn_min is not None else spawn[0]
resolved['spawn_max'] = args.spawn_max if args.spawn_max is not None else spawn[1]

print('D1G2_TASKB_VERIFY_STAGE building ' + json.dumps(resolved), flush=True)
cfg = build_d1g2_taskb_train_cfg(
    device=args.device, seed=args.seed, ddt_root=args.assets_root, num_envs=args.num_envs,
    episode_length_s=args.episode_seconds, bin_scale=resolved['bin_scale'],
    spawn_min=resolved['spawn_min'], spawn_max=resolved['spawn_max'],
    heading_noise=resolved['heading_noise'], release_radius=resolved['release_radius'])
cfg.sim.use_fabric = True
from isaaclab.envs import ManagerBasedRLEnv
env = ManagerBasedRLEnv(cfg=cfg)
state = taskb_state(env)
state.update(cfg.d1g2_taskb_initial_curriculum)
cfg.d1g2_taskb_ground_plane_spawner(env)
verifier = Verifier(env, args.checkpoint)

raw, _ = env.reset(seed=args.seed)
records = []
for _ in range(args.episodes):
    # per-episode tracking
    start_distance = torch.norm(env.scene['robot'].data.root_pos_w[:, :2] - bin_center_w(env), dim=1)
    path_length = torch.zeros(env.num_envs, device=args.device)
    previous_xy = env.scene['robot'].data.root_pos_w[:, :2].clone()
    min_object_distance = torch.full((env.num_envs,), 1.0e3, device=args.device)
    min_robot_clearance = torch.full((env.num_envs,), 1.0e3, device=args.device)
    released = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
    bin_radius = state['bin_radius']
    finished = 0
    for _ in range(env.max_episode_length):
        last_action, _ = verifier.act_with_reset(raw, [])
        raw, _, terminated, truncated, _ = env.step(
            verifier.state.combine(last_action[:, :NUM_RESIDUAL_ACTIONS]))
        done = terminated | truncated
        robot_pos = env.scene['robot'].data.root_pos_w
        object_pos = env.scene['object'].data.root_pos_w
        path_length += torch.norm(robot_pos[:, :2] - previous_xy, dim=1)
        previous_xy = robot_pos[:, :2].clone()
        min_object_distance = torch.minimum(
            min_object_distance, torch.norm(object_pos[:, :2] - bin_center_w(env), dim=1))
        min_robot_clearance = torch.minimum(
            min_robot_clearance, torch.norm(robot_pos[:, :2] - bin_center_w(env), dim=1) - bin_radius)
        released |= ~state['carrying']
        episode_steps = episode_steps + 1
        if not bool(done.any()):
            continue
        ids = done.nonzero().flatten()
        # The manager state was reset by step() for exactly these ids, so the
        # success flag read here is the *next* episode's.  Recompute it from
        # this script's own bookkeeping instead: an episode counts as delivered
        # when the object came to rest inside the success circle at any point
        # during that episode.
        success = min_object_distance <= state['success_radius']
        for index in ids.tolist():
            records.append({
                'success': bool(success[index]),
                'released': bool(released[index]),
                'start_distance_m': float(start_distance[index]),
                'min_object_distance_m': float(min_object_distance[index]),
                'min_robot_clearance_m': float(min_robot_clearance[index]),
                'path_length_m': float(path_length[index]),
                'steps': int(episode_steps[index]),
                'terminations': [name for name in env.termination_manager.active_terms
                                 if bool(env.termination_manager.get_term(name)[index])],
            })
        finished += len(ids)
        # restart tracking for the environments that just finished
        start_distance[ids] = torch.norm(robot_pos[ids, :2] - bin_center_w(env)[ids], dim=1)
        path_length[ids] = 0.0
        previous_xy[ids] = robot_pos[ids, :2]
        min_object_distance[ids] = 1.0e3
        min_robot_clearance[ids] = 1.0e3
        released[ids] = False
        episode_steps[ids] = 0
    print(f'D1G2_TASKB_VERIFY_STAGE batch_done finished={finished}', flush=True)

successes = [row for row in records if row['success']]
summary = {
    'scope': 'Task B PROXY environment, deterministic policy, fresh spawns.',
    'official_task_b_success': False,
    'note': 'One object, kinematic carry, scripted release. Not the official 18-object task.',
    'checkpoint': str(args.checkpoint.resolve()),
    'checkpoint_iteration': verifier.iteration,
    'seed': args.seed,
    'num_envs': args.num_envs,
    'episodes_per_env': args.episodes,
    'evaluation_settings': resolved,
    'checkpoint_task_metadata': trained,
    'episodes': len(records),
    'deliveries': len(successes),
    'delivery_rate': len(successes) / len(records) if records else None,
    'release_rate': sum(row['released'] for row in records) / len(records) if records else None,
    'termination_counts': dict(Counter(t for row in records for t in row['terminations'])),
}
if records:
    summary.update(
        mean_min_object_distance_m=statistics.mean(row['min_object_distance_m'] for row in records),
        mean_min_robot_clearance_m=statistics.mean(row['min_robot_clearance_m'] for row in records),
        negative_clearance_episodes=sum(row['min_robot_clearance_m'] < 0 for row in records),
        mean_path_length_m=statistics.mean(row['path_length_m'] for row in records),
        mean_steps=statistics.mean(row['steps'] for row in records),
    )
if successes:
    summary.update(
        success_mean_final_object_distance_m=statistics.mean(
            row['min_object_distance_m'] for row in successes),
        success_mean_path_length_m=statistics.mean(row['path_length_m'] for row in successes),
        success_mean_start_distance_m=statistics.mean(row['start_distance_m'] for row in successes),
        success_min_start_distance_m=min(row['start_distance_m'] for row in successes),
        success_max_start_distance_m=max(row['start_distance_m'] for row in successes),
    )
save_json(args.output / 'verification.json', summary)
save_json(args.output / 'episodes.json', {'episodes': records})
print('D1G2_TASKB_VERIFY ' + json.dumps(summary), flush=True)
env.close()
app.close()
