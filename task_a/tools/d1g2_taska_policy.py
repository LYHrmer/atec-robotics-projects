"""CPU adapter for a D1 chassis policy with G2 held at its default pose.

This baseline uses the 16-action ``flat_lab.onnx`` policy trained for D1
without an arm. It is not a policy trained for the 23-action D1+G2 robot.
Inputs and returned actions follow ``action_joint_names`` when provided,
otherwise ATEC's grouped joint order documented below. The caller applies
scales 0.25 (legs), 5.0 (wheels), and 0.1 (arm).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


ATEC_LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
ATEC_WHEEL_JOINT_NAMES = tuple(
    f"{leg}_foot_joint" for leg in ("FR", "FL", "RR", "RL")
)
G2_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7)) + ("g2_joint",)
ATEC_JOINT_NAMES = ATEC_LEG_JOINT_NAMES + ATEC_WHEEL_JOINT_NAMES + G2_JOINT_NAMES
POLICY_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf", "foot")
)
POLICY_TO_ATEC = np.asarray(
    [ATEC_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES], dtype=np.int64
)
POLICY_WHEEL_INDICES = np.asarray([3, 7, 11, 15], dtype=np.int64)


class D1FlatLoadedPolicy:
    """Run a D1 flat-ground ONNX policy, adding seven zero G2 actions.

    ``joint_pos_rel`` must be relative to the policy's leg defaults:
    hip=0.0, thigh=0.8, calf=-1.5 radians for all four legs. Wheel positions
    are discarded. G2 state is accepted in the 23-element vectors but is
    unavailable to this 16-action policy.

    The history contains preceding observations in oldest-to-newest order.
    On reset, the first observation fills all ten history slots. Inference
    consumes the current observation and existing history, then appends the
    current observation, matching FSMState_RL.cpp's deployment convention.
    """

    atec_joint_names = ATEC_JOINT_NAMES
    policy_joint_names = POLICY_JOINT_NAMES

    def __init__(self, policy_path=None, action_joint_names=None):
        self.action_joint_names = tuple(
            ATEC_JOINT_NAMES if action_joint_names is None else action_joint_names
        )
        if (len(self.action_joint_names) != 23
                or set(self.action_joint_names) != set(ATEC_JOINT_NAMES)):
            raise ValueError("action_joint_names must contain each of the 23 D1+G2 joint names once")
        self.atec_joint_names = self.action_joint_names
        self._policy_to_action = np.asarray(
            [self.action_joint_names.index(name) for name in POLICY_JOINT_NAMES],
            dtype=np.int64,
        )
        if policy_path is None:
            policy_path = (
                Path(__file__).resolve().parents[1]
                / "third_party/ddt_ros2_control/controller/rl_controller"
                / "config/d1/flat_lab.onnx"
            )
        self.policy_path = Path(policy_path).expanduser().resolve(strict=True)
        self._session = None
        self._reference = None
        try:
            import onnxruntime as ort
        except ImportError:
            import onnx
            from onnx.reference import ReferenceEvaluator

            model = onnx.load(str(self.policy_path))
            self._reference = ReferenceEvaluator(model)
            inputs = {
                value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
                for value in model.graph.input
            }
            outputs = {
                value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
                for value in model.graph.output
            }
            backend = "onnx.reference.ReferenceEvaluator"
        else:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.policy_path), options, providers=["CPUExecutionProvider"]
            )
            inputs = {value.name: value.shape for value in self._session.get_inputs()}
            outputs = {value.name: value.shape for value in self._session.get_outputs()}
            backend = "onnxruntime.CPUExecutionProvider"
        expected_inputs = {"nn_input0": [1, 57], "nn_input1": [1, 10, 57]}
        expected_outputs = {"nn_output": [1, 16]}
        if inputs != expected_inputs or outputs != expected_outputs:
            raise ValueError(
                "Expected a D1 flat policy with inputs "
                f"{expected_inputs} and output {expected_outputs}; "
                f"found inputs {inputs}, outputs {outputs}."
            )
        self.metadata = {
            "kind": "D1_16_action_flat_policy_with_G2_held_at_default_pose",
            "trained_with_G2": False,
            "policy_path": str(self.policy_path),
            "sha256": hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
            "backend": backend,
            "inputs": inputs,
            "outputs": outputs,
            "atec_joint_names": list(self.action_joint_names),
            "policy_joint_names": list(POLICY_JOINT_NAMES),
            "policy_to_atec_indices": self._policy_to_action.tolist(),
            "observation_order": [
                "base_ang_vel * 0.25", "projected_gravity",
                "clipped_command * [2, 2, 0.25]",
                "joint_pos_rel_16_wheels_zero", "joint_vel_16 * 0.05",
                "previous_raw_policy_action_16",
            ],
            "policy_leg_defaults_radians": [0.0, 0.8, -1.5],
            "command_limits": [-1.0, 1.0],
            "action_scales_atec": [
                5.0 if name in ATEC_WHEEL_JOINT_NAMES
                else 0.25 if name in ATEC_LEG_JOINT_NAMES else 0.1
                for name in self.action_joint_names
            ],
            "g2_raw_actions": [0.0] * 7,
            "history": "10 preceding frames oldest-to-newest; append after inference",
            "reset_history": "repeat first observation ten times",
            "action_filter": "none; flat_lab ROS FSM applies raw actions directly",
            "nominal_control_dt_s": 0.02,
            "source": "third_party/ddt_ros2_control/controller/rl_controller/config/d1/controllers.yaml:130",
        }
        self.reset()

    def reset(self):
        """Clear the previous policy action and defer history fill to first use."""
        self._last_action = np.zeros(16, dtype=np.float32)
        self._history = None

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains non-finite values")
        return vector

    def action_from_state(
        self, ang_vel, projected_gravity, joint_pos_rel, joint_vel, command, dt=0.02
    ):
        """Return 23 raw actions in ATEC grouped order for one 50 Hz step."""
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("dt must be positive and finite")
        angular = self._vector(ang_vel, 3, "ang_vel")
        gravity = self._vector(projected_gravity, 3, "projected_gravity")
        position = self._vector(joint_pos_rel, 23, "joint_pos_rel")[self._policy_to_action].copy()
        velocity = self._vector(joint_vel, 23, "joint_vel")[self._policy_to_action]
        commands = np.clip(self._vector(command, 3, "command"), -1.0, 1.0)
        position[POLICY_WHEEL_INDICES] = 0.0
        observation = np.concatenate(
            [angular * 0.25, gravity, commands * np.asarray([2.0, 2.0, 0.25], np.float32),
             position, velocity * 0.05, self._last_action]
        ).astype(np.float32, copy=False)
        if self._history is None:
            self._history = np.repeat(observation[None, None, :], 10, axis=1)
        inputs = {"nn_input0": observation[None, :], "nn_input1": self._history}
        runtime = self._session if self._session is not None else self._reference
        result = np.asarray(runtime.run(["nn_output"], inputs)[0], dtype=np.float32)
        if result.shape != (1, 16) or not np.isfinite(result).all():
            raise ValueError(f"Policy returned invalid action with shape {result.shape}")
        self._last_action = result[0].copy()
        self._history[:, :-1, :] = self._history[:, 1:, :].copy()
        self._history[:, -1, :] = observation
        action = np.zeros(23, dtype=np.float32)
        action[self._policy_to_action] = self._last_action
        return action
