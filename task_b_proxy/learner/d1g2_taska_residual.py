"""Shared training/deployment state for a residual on the supplied D1 policy.

The residual sees proprioception and its own action history. Height samples and
simulator position are never included in its actor input. G2 stays a physical
load under its original joint controller.
"""
from __future__ import annotations

from pathlib import Path
import torch
from tools.d1g2_taska_env import ACTION_JOINT_NAMES
from tools.d1g2_taska_torch_policy import D1FlatTorchController

ACTOR_OBS_DIM = 331
RESIDUAL_SCALES = [1.0] * 12 + [0.5] * 4
RESIDUAL_CLIP = 3.0


class ResidualState:
    def __init__(self, policy_path, num_envs, device):
        self.base = D1FlatTorchController(
            policy_path, num_envs=num_envs, device=device,
            action_joint_names=ACTION_JOINT_NAMES,
            normalization_backend='onnx',
        )
        self.last_residual = torch.zeros(num_envs, 16, device=device)
        self.scales = torch.tensor(RESIDUAL_SCALES, device=device)
        self.base_action = torch.zeros(num_envs, 23, device=device)

    def reset(self, env_ids=None):
        self.base.reset(env_ids)
        if env_ids is None:
            self.last_residual.zero_()
        else:
            self.last_residual[env_ids] = 0

    @torch.no_grad()
    def observe(self, proprio, command):
        self.base_action = self.base.action_from_state(
            proprio[:, 3:6], proprio[:, 9:12], proprio[:, 12:35],
            proprio[:, 35:58], command,
        )
        # The controller appends once per call: these are 4 prior frames and
        # the current frame, precisely the history consumed by the base net.
        actor = torch.cat((self.base.history[:, -5:].flatten(1),
                           proprio[:, 28:35], proprio[:, 51:58] * 0.05,
                           self.base_action[:, :16], self.last_residual), dim=-1)
        if actor.shape[-1] != ACTOR_OBS_DIM:
            raise ValueError(f'Unexpected residual input shape: {actor.shape}')
        return actor.clamp(-100, 100)

    @torch.no_grad()
    def combine(self, residual):
        bounded = residual.clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
        self.last_residual.copy_(bounded)
        action = self.base_action.clone()
        action[:, :16] += bounded * self.scales
        return action


class D1G2ResidualPolicy:
    """Single robot adapter with the same action_from_state API as the baseline."""
    def __init__(self, checkpoint, policy_path, device='cpu'):
        import hashlib
        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict
        self.device = device
        checkpoint = Path(checkpoint).resolve(strict=True)
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        metadata = data['infos']['d1g2_residual']
        base_sha = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
        if metadata['base_policy_sha256'] != base_sha:
            raise ValueError('Residual checkpoint was trained on a different base policy')
        if metadata['residual_scales'] != RESIDUAL_SCALES or metadata['actor_obs_dim'] != ACTOR_OBS_DIM:
            raise ValueError('Residual checkpoint interface differs from this implementation')
        if (metadata.get('base_normalization_backend') != 'onnx'
                or metadata.get('residual_clip') != RESIDUAL_CLIP
                or metadata.get('interface_version') != 1):
            raise ValueError('Residual checkpoint normalization or action semantics differ')
        dummy = TensorDict({'policy': torch.zeros(1, ACTOR_OBS_DIM, device=device),
                            'critic': torch.zeros(1, metadata['critic_obs_dim'], device=device)},
                           batch_size=[1], device=device)
        policy_cfg = dict(metadata['policy_cfg'])
        policy_cfg.pop('class_name', None)
        self.actor_critic = ActorCritic(dummy, {'policy': ['policy'], 'critic': ['critic']},
                                      16, **policy_cfg).to(device).eval()
        self.actor_critic.load_state_dict(data['model_state_dict'])
        self.state = ResidualState(policy_path, 1, device)
        self.metadata = {'kind': 'D1_flat_backbone_plus_D1G2_trained_residual',
                         'trained_with_G2': True, 'checkpoint': str(checkpoint),
                         'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'base_policy_sha256': base_sha, 'iteration': data['iter'],
                         'actor_uses_terrain_truth': False,
                         'training_is_not_full_course_success': True}

    def reset(self):
        self.state.reset()

    @torch.inference_mode()
    def action_from_state(self, ang_vel, projected_gravity, joint_pos_rel, joint_vel, command, dt=0.02):
        from tensordict import TensorDict
        if abs(dt - 0.02) > 1e-6:
            raise ValueError('Residual policy requires 50 Hz control')
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, -1)
        proprio = torch.zeros(1, 81, device=self.device)
        proprio[:, 3:6] = tensor(ang_vel)
        proprio[:, 9:12] = tensor(projected_gravity)
        proprio[:, 12:35] = tensor(joint_pos_rel)
        proprio[:, 35:58] = tensor(joint_vel)
        actor_obs = self.state.observe(proprio, tensor(command))
        obs = TensorDict({'policy': actor_obs}, batch_size=[1], device=self.device)
        residual = self.actor_critic.act_inference(obs)
        return self.state.combine(residual)[0].cpu().numpy()
