"""Can the official locomotion policy drive B2wPiper? Offline first, on recorded proprio.

The contract's fallback ladder says: before proposing any training, do a read-only
check for a matching checkpoint. The only locomotion checkpoint provided is
``atec_robot_model/baseline/unitree_b2_flat/policy.pt`` - an RSL-RL export with
``actor: MLP(45 -> 512 -> 256 -> 128 -> 12)`` and ``normalizer = Identity``. It
drives 12 LEG joints and nothing else; the official ``demo/solution.py`` maps its
output into action indices 0..11 and leaves the rest zero, which on this robot
means the WHEELS AND ARM ARE COMMANDED ZERO.

This answers the cheap question before spending a GPU run: does that network
produce structured, bounded leg commands when fed this robot's own recorded
public proprio, and does it RESPOND to a velocity command? A policy fed the wrong
observation layout outputs noise that does not respond to anything.

Nothing here is a policy input to the real robot; it reads recorded telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

POLICY = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/baseline/unitree_b2_flat/policy.pt')
RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/plan_d4_first_delivery_seed42_01')

#: The 12 leg joints, in the order the action schema lists them (action slice 0:12).
LEG_JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
              "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")


def build_obs(proprio, leg_obs_ids, previous_action, command, *, joint_velocity_scale=1.0,
              order="ang_grav_cmd"):
    """One 45-dim rsl_rl locomotion observation from the public 84-dim proprio."""
    ang_vel = proprio[3:6]*0.25   # the training config's base_ang_vel.scale
    gravity = proprio[9:12]
    joint_pos = proprio[12+leg_obs_ids]
    joint_vel = proprio[36+leg_obs_ids]*joint_velocity_scale
    if order == "ang_grav_cmd":
        parts = [ang_vel, gravity, command, joint_pos, joint_vel, previous_action]
    else:  # cmd first, then gravity: the other common rsl_rl ordering
        parts = [ang_vel, command, gravity, joint_pos, joint_vel, previous_action]
    return np.concatenate(parts).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample', type=int, default=5000,
                        help='a settled sample in the recorded run')
    args = parser.parse_args()

    policy = torch.jit.load(str(POLICY), map_location='cpu')
    policy.eval()

    telemetry = np.load(RUN/'telemetry.npz')
    names = telemetry['joint_names'].tolist()
    # The proprio joint block is in SCHEMA order (leg, wheel, arm), not scene order.
    # The proprio joint block is in the ARTICULATION order, which interleaves the
    # arm among the legs - NOT the schema's leg-then-wheel-then-arm order.
    articulation = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                    "arm_joint1", "FL_calf_joint", "FR_calf_joint", "RL_calf_joint",
                    "RR_calf_joint", "arm_joint2", "FR_foot_joint", "FL_foot_joint",
                    "RL_foot_joint", "RR_foot_joint", "arm_joint3", "arm_joint4",
                    "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8"]
    position = {name: index for index, name in enumerate(articulation)}
    leg_obs_ids = np.array([position[name] for name in LEG_JOINTS])
    previous_action = np.zeros(12)
    print('joint order check: LEG_JOINTS ==', 'FR,FL,RR,RL grouped, matching the training config')

    report = {
        'scope': 'offline probe of the official locomotion checkpoint on recorded public proprio; '
                 'NOT a robot run, NOT a locomotion result',
        'policy': str(POLICY), 'run': RUN.name, 'sample': args.sample,
        'architecture': 'MLP(45 -> 512 -> 256 -> 128 -> 12), normalizer Identity',
        'drives': '12 leg joints only; wheels and arm zeroed, as demo/solution.py does',
        'hypotheses': {},
    }
    index = args.sample
    proprio = telemetry['proprio'][index].astype(np.float64)
    action_env = telemetry['action'][index].astype(np.float64)

    for order in ("ang_grav_cmd", "ang_cmd_grav"):
        for jvel_scale in (1.0, 0.05):
            entry = {}
            for label, command in (("zero_command", np.zeros(3)),
                                   ("forward_1ms", np.array([1.0, 0.0, 0.0])),
                                   ("turn_1rads", np.array([0.0, 0.0, 1.0]))):
                obs = build_obs(proprio, leg_obs_ids, previous_action, command,
                                joint_velocity_scale=jvel_scale, order=order)
                with torch.no_grad():
                    out = policy(torch.as_tensor(obs).unsqueeze(0)).numpy().ravel()
                entry[label] = {
                    'absmax': float(np.abs(out).max()), 'mean': float(out.mean()),
                    'std': float(out.std()),
                    'first6': [round(float(v), 4) for v in out[:6]],
                }
            entry['responds_to_command'] = bool(
                np.abs(np.array(entry['forward_1ms']['first6'])
                       - np.array(entry['zero_command']['first6'])).max() > 1e-3)
            report['hypotheses']['%s_jvel%.2f' % (order, jvel_scale)] = entry

    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    for name, entry in report['hypotheses'].items():
        print('%-22s zero|a|max=%.4f  fwd|a|max=%.4f  turn|a|max=%.4f  responds=%s'
              % (name, entry['zero_command']['absmax'], entry['forward_1ms']['absmax'],
                 entry['turn_1rads']['absmax'], entry['responds_to_command']))
    print()
    best = max(report['hypotheses'].items(), key=lambda kv: kv[1]['responds_to_command'])
    print('most responsive hypothesis:', best[0])
    print('  zero-command output absmax %.4f (a policy fed correctly should be SMALL here:'
          % best[1]['zero_command']['absmax'])
    print('   a settled robot with a zero command should hold roughly its default stance)')
    print('  forward-command first6:', best[1]['forward_1ms']['first6'])


if __name__ == '__main__':
    main()
