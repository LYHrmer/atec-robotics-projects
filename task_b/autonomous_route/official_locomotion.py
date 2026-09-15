"""Drive B2wPiper with the OFFICIAL locomotion checkpoint. Diagnostic only.

Why this exists. Task B cannot be cleared while the base turns at ~0.0007 rad/s:
one delivery costs ~1050 s and the episode limit is 1200 s, so at most one object
per episode and never eighteen. The official baseline does not roll this robot on
its wheels at all - ``demo/solution.py`` maps its policy into action indices 0..11
and leaves the rest zero, i.e. it drives the 12 LEG joints and commands the WHEELS
AND ARM TO ZERO. That is the intended locomotion mechanism, and this project has
so far used wheel differential instead.

Before proposing any training, the contract requires a read-only check for a
matching checkpoint. The only one provided is
``atec_robot_model/baseline/unitree_b2_flat/policy.pt``: an RSL-RL export with
``actor: MLP(45 -> 512 -> 256 -> 128 -> 12)``, ``normalizer = Identity``. This
wrapper asks the one question that decides whether training is needed at all:
DOES IT MOVE THIS ROBOT?

Reading the policy. An offline probe against recorded proprio showed the standard
rsl_rl ordering (ang_vel, projected_gravity, command, joint_pos, joint_vel,
last_action) with the joint velocities scaled by 0.05 gives by far the smallest
output on a zero command - 0.95 against 4.4-5.1 for the alternative ordering - which
is what a correctly fed policy holding its stance should do. That layout is what
this wrapper uses.

What this is NOT: not a verified capability, not a delivery, not a Task B result.
A checkpoint trained for a LEGGED Unitree B2 is being applied to a WHEEL-LEGGED
B2wPiper, and whether it transfers is exactly what the run is for. Nothing here
touches physics, assets, rewards, terminations or the provided checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path('/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/'
                  'baseline/unitree_b2_flat/policy.pt')

#: The 12 leg joints in the order the action schema lists them (action slice 0:12).
LEG_JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
              "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")

#: Observation scales, read from the training config that ships in this repo at
#: ``task_a/source/atec_rl_lab/.../unitree_b2/rough_env_cfg.py``:
#: ``base_ang_vel.scale = 0.25``, ``joint_pos.scale = 1.0`` (and it is
#: ``mdp.joint_pos_rel``, i.e. relative to the default), ``joint_vel.scale = 0.05``,
#: ``base_lin_vel = None``, ``height_scan = None``. The first attempt omitted the
#: 0.25 on the angular velocity and fed it four times too large, which is why the
#: policy output ran to 19 for a zero command.
BASE_ANG_VEL_SCALE = .25
JOINT_VELOCITY_SCALE = .05
#: Actions: the training config sets hip .125 and everything else .25. Our leg
#: term applies a uniform .5, so reproducing the same joint offsets here needs
#: train -> env of .125/.5 = .25 for hips and .25/.5 = .5 for thigh and calf.
TRAIN_TO_ENV_SCALE = np.array([.25, .5, .5]*4, dtype=np.float64)
ENV_LEG_SCALE = .5


class OfficialLocomotionPolicy:
    """The provided locomotion checkpoint, driving the 12 leg joints only."""

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 command=(0., 0., 0.), checkpoint=CHECKPOINT, settle_calls=100):
        self.dt = float(dt)
        self.schema = schema
        self.names = tuple(observation_joint_names)
        self.defaults = dict(defaults)
        self.command = np.asarray(command, dtype=np.float64)
        self.leg = schema.term('joint_leg')
        self.wheel = schema.term('joint_wheel')
        self.arm = schema.term('joint_arm')
        # The proprio joint block is in the ARTICULATION order the evaluator
        # verified against the simulator, which is NOT the schema order: the
        # metadata lists FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ... with
        # arm_joint1/2 interleaved among the legs. Building the indices from the
        # schema's leg-then-wheel-then-arm order read the wrong proprio entries
        # and fed the policy nonsense - its output ran to 19 for a ZERO command,
        # where a correctly fed locomotion policy should sit near zero.
        self.leg_obs_ids = np.array([self.names.index(name) for name in LEG_JOINTS])
        self.calls = 0
        #: The official episode spawns the robot at 0.788 m and it FALLS about
        #: 0.29 m before it is standing. A locomotion policy trained on a standing
        #: robot has no reason to cope with free fall, so the first attempt fed it
        #: from step 1 and it terminated on illegal_contact at step 31 - which
        #: tests the spawn, not the checkpoint. The leg command is held at the
        #: default pose for this many calls first, exactly as the other modes do.
        self.settle_calls = int(settle_calls)
        #: The evaluator's recorder reads this; measured modes ramp it, this one is
        #: always fully engaged because the checkpoint drives the legs directly.
        self.alpha = 1.0
        self.previous_action = np.zeros(12)
        self.state = 'OFFICIAL_LOCOMOTION'
        self.state_reason = 'the provided Unitree B2 flat checkpoint drives the 12 leg joints'
        self.done_reason = None
        self.debug = {}
        self.policy = torch.jit.load(str(checkpoint), map_location='cpu')
        self.policy.eval()
        self.last_action = np.zeros(12)

    @property
    def wheel_hold_requested(self):
        """The official baseline commands the wheels to zero, not to a held angle."""
        return False

    @property
    def pause_for_stance(self):
        return False

    @pause_for_stance.setter
    def pause_for_stance(self, value):
        pass

    def _observation(self, proprio):
        ang_vel = proprio[3:6]*BASE_ANG_VEL_SCALE
        gravity = proprio[9:12]
        joint_pos = proprio[12+self.leg_obs_ids]-np.array(
            [self.defaults[name] for name in LEG_JOINTS])
        joint_vel = proprio[36+self.leg_obs_ids]*JOINT_VELOCITY_SCALE
        return np.concatenate([ang_vel, gravity, self.command, joint_pos, joint_vel,
                               self.previous_action]).astype(np.float32)

    def act(self, proprio, images=None):
        self.calls += 1
        observation = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if observation.size != 84 or not np.isfinite(observation).all():
            self.done_reason = self.state_reason = 'official_locomotion_invalid_proprio'
            self.state = 'STOPPED'
            return np.zeros(int(self.schema.total_dim), dtype=np.float32)
        obs = self._observation(observation)
        if self.calls <= self.settle_calls:
            self.debug = {'module': 'task_b.official_locomotion.OfficialLocomotionPolicy',
                          'calls': self.calls, 'phase': 'SETTLE',
                          'reason': 'holding the default leg pose through the spawn drop before '
                                    'the checkpoint is engaged'}
            return np.zeros(int(self.schema.total_dim), dtype=np.float32)
        with torch.no_grad():
            output = self.policy(torch.as_tensor(obs).unsqueeze(0)).numpy().ravel()
        if not np.isfinite(output).all():
            self.done_reason = self.state_reason = 'official_locomotion_non_finite_output'
            self.state = 'STOPPED'
            return np.zeros(int(self.schema.total_dim), dtype=np.float32)
        self.last_action = output.copy()
        # The observation's last-action entry is in TRAIN space, and the policy
        # output IS the train-space action, so it is fed straight back. The old
        # expression multiplied by the scale and divided by the env scale, which
        # is not a train-space value at all.
        self.previous_action = output.copy()
        action = np.zeros(int(self.schema.total_dim), dtype=np.float32)
        action[self.leg.start:self.leg.stop] = (output*TRAIN_TO_ENV_SCALE).astype(np.float32)
        # wheels and arm stay at exactly zero, as the official solution does
        self.debug = {
            'module': 'task_b.official_locomotion.OfficialLocomotionPolicy',
            'calls': self.calls, 'command': self.command.tolist(),
            'policy_output_absmax': float(np.abs(output).max()),
            'policy_output_mean': float(output.mean()),
            'leg_action_env_absmax': float(np.abs(action[self.leg.start:self.leg.stop]).max()),
            'wheel_command': 'zero, as demo/solution.py does',
            'arm_command': 'zero, as demo/solution.py does',
            'evidence_note': 'diagnostic bookkeeping only; whether this moves the robot is a '
                             'GPU-run question, and nothing here is a delivery or Task B claim',
        }
        return action

    def describe(self):
        return {
            'module': 'task_b.official_locomotion.OfficialLocomotionPolicy',
            'candidate': 'the PROVIDED Unitree B2 flat locomotion checkpoint, driving the 12 leg '
                         'joints of B2wPiper with wheels and arm commanded to zero',
            'why': 'the official baseline moves this robot with legged locomotion; the wheel '
                   'differential used so far turns at only ~0.0007 rad/s, which cannot clear '
                   'Task B within a 1200 s episode',
            'not_claimed': 'a checkpoint trained for a LEGGED B2 is being applied to a '
                           'WHEEL-LEGGED B2wPiper; transfer is exactly what the run tests, and '
                           'this is not a verified capability',
            'observation_layout': 'ang_vel(3), projected_gravity(3), command(3), joint_pos(12), '
                                  'joint_vel*0.05(12), last_action(12)',
        }
