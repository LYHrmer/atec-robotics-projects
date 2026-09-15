"""Check whether one episode yields several deliveries.

Loads a trained checkpoint, runs the max-length episode in chunks so per-chunk
delivery counts are observable, and reports how many deliveries the policy
chains inside a single episode -- the multi-delivery behaviour added to the
Task B proxy.

Totals come from an episode-ended tally, not from the per-chunk sums.  The
counters this script reads are cumulative *within* an episode and are cleared
only when the episode ends, so summing them at every chunk boundary re-counted
every delivery at every later boundary, and the denominator was env-chunks
rather than episodes.  That is the same reading the trainer's own diagnostics
does in ``_reset_idx`` -- which is what an independent check has to match.
"""
from __future__ import annotations

import argparse
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
parser.add_argument('--episode_seconds', type=float, default=40.0)
parser.add_argument('--chunk', type=int, default=250)
parser.add_argument('--target_episodes', type=int, default=320,
                    help='Stop once this many episodes have ended.')
parser.add_argument('--max_steps', type=int, default=8000,
                    help='Hard step cap, in case episodes never end.')
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
    DROPS_PER_EPISODE, GOAL_OBS_DIM, NUM_ACTIONS, NUM_RESIDUAL_ACTIONS,
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
    heading_noise=1.5707963267948966,
    # Without this the environment builds in SINGLE-delivery mode, next_delivery
    # returns immediately, and every measurement of this script is of the wrong
    # mode -- which is exactly what made four runs of it read zero.
    multi_delivery=True)
class TallyEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that records each episode's tally as the episode ends.

    The counters have to be read in ``_reset_idx`` and *before* the parent
    clears them, which is exactly where the trainer reads them.
    """

    def __init__(self, *pos, **kw):
        self.episode_tally = []
        super().__init__(*pos, **kw)

    def _reset_idx(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(active):
            state = taskb_state(self)
            deliveries = state['deliveries_this_episode'][active]
            drops = state['drops_this_episode'][active]
            released = state['released'][active]
            for index in range(len(active)):
                self.episode_tally.append({
                    'deliveries': int(deliveries[index]),
                    'drops': int(drops[index]),
                    'released': bool(released[index]),
                })
        super()._reset_idx(env_ids)


cfg.sim.use_fabric = True
env = TallyEnv(cfg=cfg)
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
while len(env.episode_tally) < args.target_episodes and total_steps < args.max_steps:
    for _ in range(args.chunk):
        actor = torch.cat((residual.observe(raw['proprio'], state['command']), raw['goal']), dim=-1)
        action = actor_critic.act_inference(TensorDict(
            {'policy': actor}, batch_size=[env.num_envs], device=args.device))
        state['command'][:] = command_from_action(action[:, NUM_RESIDUAL_ACTIONS:])
        raw, _, terminated, truncated, _ = env.step(
            residual.combine(action[:, :NUM_RESIDUAL_ACTIONS]))
        residual.reset((terminated | truncated).nonzero().flatten())
        total_steps += 1
    # Live snapshot only: these counters are cumulative within an episode, so
    # they show how far the current episodes have chained, and rise until the
    # episodes end.  They are not episode totals -- the tally above is.
    print(f'CHUNK_SNAPSHOT steps={total_steps} episodes_done={len(env.episode_tally)} '
          f'carrying={int(state["carrying"].sum())}', flush=True)
    chunks.append({
        'steps': total_steps,
        'episodes_done': len(env.episode_tally),
        'deliveries_per_env_mean': float(state['deliveries_this_episode'].float().mean()),
        'deliveries_per_env_max': int(state['deliveries_this_episode'].max()),
        'drops_per_env_mean': float(state['drops_this_episode'].float().mean()),
        'carrying_now': int(state['carrying'].sum()),
    })

tally = env.episode_tally
episodes = len(tally)
deliveries = sum(row['deliveries'] for row in tally)
drops = sum(row['drops'] for row in tally)
summary = {
    'scope': 'Task B PROXY, deterministic policy, multi-delivery episodes.',
    'official_task_b_success': False,
    'checkpoint': str(args.checkpoint.resolve()),
    'num_envs': args.num_envs,
    'episode_length_s': args.episode_seconds,
    'steps_run': total_steps,
    'chunks': chunks,
    'completed_episodes': episodes,
    'total_deliveries': deliveries,
    'total_drops': drops,
    'deliveries_per_episode': deliveries / episodes if episodes else None,
    'deliveries_per_drop': deliveries / drops if drops else None,
    'max_deliveries_in_episode': max((row['deliveries'] for row in tally), default=None),
    'mean_drops_per_episode': drops / episodes if episodes else None,
    'episodes_with_delivery': (sum(1 for row in tally if row['deliveries'] > 0) / episodes
                               if episodes else None),
    'episodes_that_released': (sum(1 for row in tally if row['released']) / episodes
                               if episodes else None),
    # ``deliveries_per_episode`` is right-censored by the per-episode drop cap: an
    # episode that hits the cap would have delivered more given more attempts, so
    # for a strong policy the mean is a lower bound.  Report the censoring rate so
    # the number can be read with that in mind instead of as an unbiased mean.
    'drop_cap': DROPS_PER_EPISODE,
    'episodes_at_drop_cap': sum(1 for row in tally if row['drops'] >= DROPS_PER_EPISODE),
    'episodes_at_drop_cap_fraction': (sum(1 for row in tally if row['drops'] >= DROPS_PER_EPISODE)
                                      / episodes if episodes else None),
    'deliveries_per_episode_stdev': (statistics.pstdev([row['deliveries'] for row in tally])
                                     if episodes else None),
}

(args.output / 'episodes.json').write_text(
    json.dumps({'episodes': tally}, ensure_ascii=False, indent=2) + '\n')
(args.output / 'multidelivery.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print('D1G2_MULTIDELIVERY ' + json.dumps(summary), flush=True)
env.close()
app.close()
