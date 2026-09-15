"""Check whether one episode yields several deliveries.

Loads a trained checkpoint, runs the max-length episode in chunks so per-chunk
delivery counts are observable, and reports how many deliveries the policy
chains inside a single episode -- the multi-delivery behaviour added to the
Task B proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
parser.add_argument('--episode_seconds', type=float, default=40.0)
parser.add_argument('--chunk', type=int, default=250)
parser.add_argument('--release_radius_override', type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.experience:
    args.experience = '/home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit'
args.enable_cameras = False
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from isaaclab.envs import ManagerBasedRLEnv
from tools.d1g2_taska_residual import ResidualState, ACTOR_OBS_DIM, RESIDUAL_SCALES, RESIDUAL_CLIP
from tools.d1g2_taskb_train_env import (
    GOAL_OBS_DIM, NUM_ACTIONS, NUM_RESIDUAL_ACTIONS,
    build_d1g2_taskb_train_cfg, command_from_action, delivery_success, taskb_state,
)

ACTOR_DIM = ACTOR_OBS_DIM + GOAL_OBS_DIM

torch.set_num_threads(4)
torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = True

data = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
meta = data['infos']['d1g2_taskb_residual']
cfg = build_d1g2_taskb_train_cfg(
    device=args.device, seed=1234, ddt_root=args.assets_root, num_envs=args.num_envs,
    episode_length_s=args.episode_seconds, bin_scale=meta.get('bin_scale', 2.0),
    release_radius=(args.release_radius_override if args.release_radius_override
                    else meta['release_radius_m']),
    spawn_min=meta['spawn_ring_m'][0], spawn_max=meta['spawn_ring_m'][1],
    heading_noise=1.5707963267948966)
cfg.sim.use_fabric = True
env = ManagerBasedRLEnv(cfg=cfg)
state = taskb_state(env)
state.update(cfg.d1g2_taskb_initial_curriculum)
cfg.d1g2_taskb_ground_plane_spawner(env)

policy_cfg = dict(meta['policy_cfg']); policy_cfg.pop('class_name', None)
dummy = TensorDict({'policy': torch.zeros(1, ACTOR_DIM, device=args.device),
                    'critic': torch.zeros(1, meta['critic_obs_dim'], device=args.device)},
                   batch_size=[1], device=args.device)
actor_critic = ActorCritic(dummy, {'policy': ['policy'], 'critic': ['critic']},
                           NUM_ACTIONS, **policy_cfg).to(args.device).eval()
actor_critic.load_state_dict(data['model_state_dict'])
residual = ResidualState(args.policy, env.num_envs, args.device)

raw, _ = env.reset(seed=1234)
chunks = []
total_steps = 0
total_deliveries = 0
total_drops = 0
for chunk in range(int(env.max_episode_length // args.chunk) + 1):
    for _ in range(args.chunk):
        actor = torch.cat((residual.observe(raw['proprio'], state['command']), raw['goal']), dim=-1)
        action = actor_critic.act_inference(TensorDict(
            {'policy': actor}, batch_size=[env.num_envs], device=args.device))
        state['command'][:] = command_from_action(action[:, NUM_RESIDUAL_ACTIONS:])
        raw, _, terminated, truncated, _ = env.step(
            residual.combine(action[:, :NUM_RESIDUAL_ACTIONS]))
        residual.reset((terminated | truncated).nonzero().flatten())
        total_steps += 1
    total_deliveries += int(state['deliveries_this_episode'].sum())
    total_drops += int(state['drops_this_episode'].sum())
    chunks.append({
        'steps': total_steps,
        'cum_deliveries': total_deliveries,
        'cum_drops': total_drops,
        'deliveries_per_env_mean': float(state['deliveries_this_episode'].float().mean()),
        'deliveries_per_env_max': int(state['deliveries_this_episode'].max()),
        'drops_per_env_mean': float(state['drops_this_episode'].float().mean()),
        'carrying_now': int(state['carrying'].sum()),
        'episodes_ended': int((terminated | truncated).sum()),
    })

summary = {
    'scope': 'Task B PROXY, deterministic policy, multi-delivery episodes.',
    'official_task_b_success': False,
    'checkpoint': str(args.checkpoint.resolve()),
    'num_envs': args.num_envs,
    'episode_length_s': args.episode_seconds,
    'chunks': chunks,
    'total_deliveries_in_window': total_deliveries,
    'total_drops_in_window': total_drops,
    'deliveries_per_drop': total_deliveries / total_drops if total_drops else None,
    'deliveries_per_env_per_episode_window': total_deliveries / (args.num_envs * max(1, len(chunks))),
}
(args.output / 'multidelivery.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print('D1G2_MULTIDELIVERY ' + json.dumps(summary), flush=True)
env.close()
app.close()
