"""Batched, frozen PyTorch conversion of the supplied D1 ``flat_lab.onnx``.

This is numerically equivalent to the D1-only 16-action base policy. Adding
seven zero G2 actions does not turn it into a policy trained with the arm.
No Isaac Sim import or ONNX Runtime dependency is needed. ONNX is used only
at construction to read the supplied weights.

``D1FlatTorchNetwork(path).to(device)(current, history)`` accepts float32
``(N, 57)`` and ``(N, 10, 57)`` tensors and returns ``(N, 16)`` raw actions.
``D1FlatBatchedPolicy`` provides the same state preprocessing and action-name
mapping as ``D1FlatLoadedPolicy``, with independent history/reset per robot.
The caller applies joint action scales; this module does not apply them.

The default ``normalization_backend="onnx"`` uses fixed running statistics
as in ONNX Runtime inference. ``"reference"`` reproduces the existing CPU
adapter's ReferenceEvaluator fallback, including that backend's unusual
BatchNormalization_9 momentum handling. These modes are not numerically
interchangeable; training and deployment must select the same one.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .d1g2_taska_policy import ATEC_JOINT_NAMES, POLICY_JOINT_NAMES
except ImportError:
    from d1g2_taska_policy import ATEC_JOINT_NAMES, POLICY_JOINT_NAMES


FLAT_LAB_SHA256 = "f8a721d4f7f05777db91a70608992ad2a86fc373982758937ba2fb3d9919d21b"


class _FrozenBatchNorm(nn.Module):
    """ONNX inference BatchNormalization, independent of Module.train()."""

    def __init__(self, tensors, prefix, epsilon, momentum, backend):
        super().__init__()
        for name in ("weight", "bias", "running_mean", "running_var"):
            self.register_buffer(name, tensors[f"{prefix}.{name}"].clone())
        self.epsilon = epsilon
        self.momentum = momentum
        self.backend = backend

    def forward(self, x):
        if self.backend == "reference":
            # ReferenceEvaluator BatchNormalization_9 mixes batch statistics
            # whenever momentum is present, even for a single-output export.
            # Reproduce N independent batch=1 evaluations, not an N-wide batch.
            mean = self.running_mean * self.momentum + x * (1.0 - self.momentum)
            variance = self.running_var * self.momentum
            return self.weight * (x - mean) / torch.sqrt(variance + self.epsilon) + self.bias
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            training=False, momentum=0.0, eps=self.epsilon,
        )


class D1FlatTorchNetwork(nn.Module):
    """Exact architecture/weights of the supported ONNX, generalized to N.

    The SHA check deliberately rejects other exports: graph rewrites, input
    normalization and history slicing must be audited before using their
    weights with this fixed architecture. The base parameters are frozen.
    """

    def __init__(self, policy_path=None, normalization_backend="onnx"):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        if normalization_backend not in ("reference", "onnx"):
            raise ValueError("normalization_backend must be 'reference' or 'onnx'")
        self.normalization_backend = normalization_backend
        if policy_path is None:
            policy_path = (
                Path(__file__).resolve().parents[1]
                / "third_party/ddt_ros2_control/controller/rl_controller"
                / "config/d1/flat_lab.onnx"
            )
        self.policy_path = Path(policy_path).expanduser().resolve(strict=True)
        self.sha256 = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        if self.sha256 != FLAT_LAB_SHA256:
            raise ValueError(
                "This fixed conversion supports only the audited D1 flat_lab.onnx "
                f"SHA256 {FLAT_LAB_SHA256}; received {self.sha256}"
            )
        model = onnx.load(str(self.policy_path))
        tensors = {
            item.name: torch.from_numpy(numpy_helper.to_array(item).copy())
            for item in model.graph.initializer
        }
        bn_attributes = {}
        for node in model.graph.node:
            if node.op_type == "BatchNormalization":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                bn_attributes[node.input[1].removesuffix(".weight")] = attrs

        def linear(prefix):
            weight = tensors[f"{prefix}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(tensors[f"{prefix}.bias"])
            return layer

        def norm(prefix):
            attrs = bn_attributes[prefix]
            return _FrozenBatchNorm(tensors, prefix, attrs.get("epsilon", 1e-5),
                                    attrs.get("momentum", 0.9), normalization_backend)

        self.register_buffer("obs_mean", tensors["backbone.obs_normalizer._mean"])
        self.register_buffer("obs_divisor", tensors["onnx::Div_122"])
        self.encoder = nn.Sequential(
            linear("backbone.mlp_encoder.0"), norm("backbone.mlp_encoder.1"), nn.ELU(),
            linear("backbone.mlp_encoder.3"), norm("backbone.mlp_encoder.4"), nn.ELU(),
        )
        self.latent = nn.Sequential(
            linear("backbone.latent_layer.0"), norm("backbone.latent_layer.1"),
            nn.ELU(), linear("backbone.latent_layer.3"),
        )
        self.velocity = linear("backbone.vel_layer")
        self.actor = nn.Sequential(
            linear("backbone.actor.0"), nn.ELU(),
            linear("backbone.actor.2"), nn.ELU(),
            linear("backbone.actor.4"), nn.ELU(), linear("backbone.actor.6"),
        )
        self.requires_grad_(False)
        self.eval()

    def forward(self, current, history):
        if current.ndim != 2 or current.shape[1] != 57:
            raise ValueError(f"current must have shape (N, 57), got {tuple(current.shape)}")
        if history.shape != (current.shape[0], 10, 57):
            raise ValueError(f"history must have shape (N, 10, 57), got {tuple(history.shape)}")
        normalized = (current - self.obs_mean) / self.obs_divisor
        # ONNX shifts history, appends current, then keeps its last five frames.
        previous = (history[:, -4:, :] - self.obs_mean) / self.obs_divisor
        encoded = self.encoder(torch.cat((previous, normalized[:, None, :]), dim=1).flatten(1))
        return self.actor(torch.cat((self.velocity(encoded), self.latent(encoded), normalized), dim=-1))


class D1FlatBatchedPolicy:
    """Stateful base controller for N D1+G2 robots on CPU or CUDA.

    Inputs are ``(N, 3)`` vectors and ``(N, 23)`` joint states in the supplied
    ``action_joint_names`` order. Leg positions are relative to hip=0,
    thigh=0.8, calf=-1.5 rad. ``reset(env_ids)`` clears only those robots.
    Output is raw ``(N, 23)`` actions with G2 entries zero.

    A residual controller can call ``record_applied_action(combined_action)``
    after modifying the returned action to feed the actual previous D1
    action into the next observation. This does not alter history already
    recorded for the current control step. The base-only default records its
    own output, exactly matching the existing CPU adapter.
    """

    def __init__(self, num_envs, policy_path=None, action_joint_names=None, device="cpu", normalization_backend="onnx"):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.action_joint_names = tuple(ATEC_JOINT_NAMES if action_joint_names is None else action_joint_names)
        if len(self.action_joint_names) != 23 or set(self.action_joint_names) != set(ATEC_JOINT_NAMES):
            raise ValueError("action_joint_names must contain each of the 23 D1+G2 joint names once")
        self.network = D1FlatTorchNetwork(policy_path, normalization_backend).to(self.device)
        self._policy_to_action = torch.tensor(
            [self.action_joint_names.index(name) for name in POLICY_JOINT_NAMES],
            dtype=torch.long, device=self.device,
        )
        self._wheel_indices = torch.tensor([3, 7, 11, 15], dtype=torch.long, device=self.device)
        self._command_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.last_action = torch.zeros((num_envs, 16), device=self.device)
        self.current_obs = torch.zeros((num_envs, 57), device=self.device)
        self.history = torch.zeros((num_envs, 10, 57), device=self.device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.metadata = {
            "kind": "D1_16_action_flat_policy_with_G2_held_at_default_pose",
            "backend": "torch_fixed_onnx_conversion",
            "policy_path": str(self.network.policy_path),
            "sha256": self.network.sha256,
            "trained_with_G2": False,
            "action_joint_names": list(self.action_joint_names),
            "policy_joint_names": list(POLICY_JOINT_NAMES),
            "history": "10 preceding frames oldest-to-newest; append after inference",
            "effective_encoder_history": "last 4 preceding frames and current frame",
            "reset_history": "repeat first observation ten times",
            "action_filter": "none",
            "parameters_frozen": True,
            "normalization_backend": normalization_backend,
        }

    @torch.no_grad()
    def reset(self, env_ids=None):
        """Reset all environments, or integer indices / a boolean mask."""
        if env_ids is None:
            self.last_action.zero_()
            self._initialized.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
            if ids.dtype != torch.bool:
                ids = ids.to(torch.long)
            self.last_action[ids] = 0.0
            self._initialized[ids] = False

    def _batch(self, value, width, name):
        result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if result.shape != (self.num_envs, width):
            raise ValueError(f"{name} must have shape ({self.num_envs}, {width}), got {tuple(result.shape)}")
        return result

    @torch.no_grad()
    def record_applied_action(self, action):
        """Record combined raw actions after an optional residual adjustment."""
        self.last_action.copy_(self._batch(action, 23, "action").index_select(1, self._policy_to_action))

    @torch.no_grad()
    def action_from_state(self, ang_vel, projected_gravity, joint_pos_rel, joint_vel, command, dt=0.02):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        angular = self._batch(ang_vel, 3, "ang_vel")
        gravity = self._batch(projected_gravity, 3, "projected_gravity")
        position = self._batch(joint_pos_rel, 23, "joint_pos_rel").index_select(1, self._policy_to_action)
        velocity = self._batch(joint_vel, 23, "joint_vel").index_select(1, self._policy_to_action)
        commands = self._batch(command, 3, "command").clamp(-1.0, 1.0)
        position[:, self._wheel_indices] = 0.0
        observation = torch.cat(
            (angular * 0.25, gravity, commands * self._command_scale,
             position, velocity * 0.05, self.last_action), dim=-1,
        )
        self.current_obs.copy_(observation)
        # No CUDA-to-host scalar checks: an elementwise mask fills new rows.
        self.history.copy_(torch.where(self._initialized[:, None, None], self.history, observation[:, None, :]))
        policy_action = self.network(observation, self.history)
        self.last_action.copy_(policy_action)
        self.history[:, :-1, :].copy_(self.history[:, 1:, :].clone())
        self.history[:, -1, :].copy_(observation)
        self._initialized.fill_(True)
        action = torch.zeros((self.num_envs, 23), device=self.device)
        action[:, self._policy_to_action] = policy_action
        return action


D1FlatTorchPolicy = D1FlatTorchNetwork


class D1FlatTorchController(D1FlatBatchedPolicy):
    """Training-facing API; history includes current after action_from_state.

    ``history[:, -5:].flatten(1)`` is the unnormalized 285-vector consumed by
    the base encoder on this step. ``current_obs`` is the current 57-vector.
    Reading either does not advance the controller. By default the previous
    action field records the base's own 16 raw outputs, even when the caller
    applies a separate residual to the robot.
    """

    def __init__(self, policy_path, num_envs, device="cpu", action_joint_names=None, normalization_backend="onnx"):
        super().__init__(num_envs, policy_path, action_joint_names, device, normalization_backend)


class D1FlatTorchSinglePolicy:
    """NumPy single-robot adapter with the existing loaded-policy interface."""

    def __init__(self, policy_path=None, action_joint_names=None, device="cpu", normalization_backend="onnx"):
        self.controller = D1FlatTorchController(
            policy_path, 1, device=device, action_joint_names=action_joint_names,
            normalization_backend=normalization_backend,
        )
        self.action_joint_names = self.controller.action_joint_names
        self.atec_joint_names = self.action_joint_names
        self.policy_joint_names = POLICY_JOINT_NAMES
        self.policy_path = self.controller.network.policy_path
        self.metadata = dict(self.controller.metadata)
        self.metadata["single_robot_numpy_adapter"] = True

    def reset(self):
        self.controller.reset()

    def action_from_state(self, ang_vel, projected_gravity, joint_pos_rel, joint_vel, command, dt=0.02):
        inputs = []
        for name, value, width in zip(
            ("ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel", "command"),
            (ang_vel, projected_gravity, joint_pos_rel, joint_vel, command), (3, 3, 23, 23, 3),
        ):
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must be a finite vector with shape ({width},)")
            inputs.append(vector[None, :])
        result = self.controller.action_from_state(*inputs, dt=dt)[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Policy returned non-finite actions")
        return result


def _validate(policy_path, cases):
    """CPU reference parity checks for conversion, batches and controller state."""
    import json
    import time
    import onnx
    from onnx.reference import ReferenceEvaluator
    try:
        from .d1g2_taska_policy import D1FlatLoadedPolicy
    except ImportError:
        from d1g2_taska_policy import D1FlatLoadedPolicy

    torch.set_num_threads(1)
    rng = np.random.default_rng(20260909)
    net = D1FlatTorchNetwork(policy_path, normalization_backend="reference")
    reference = ReferenceEvaluator(onnx.load(str(net.policy_path)))
    records = []
    for size in (1, 7, 32):
        for scale in (0.0, 0.1, 1.0, 3.0):
            for _ in range(cases):
                current = (rng.normal(size=(size, 57)) * scale).astype(np.float32)
                history = (rng.normal(size=(size, 10, 57)) * scale).astype(np.float32)
                # np.where in reference ELU evaluates exp on positive values
                # that its output discards; extreme stress inputs overflow it.
                with np.errstate(over="ignore"):
                    expected = np.concatenate([
                        reference.run(None, {"nn_input0": current[i:i+1], "nn_input1": history[i:i+1]})[0]
                        for i in range(size)
                    ])
                with torch.no_grad():
                    actual = net(torch.from_numpy(current), torch.from_numpy(history)).numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=5e-4)
                records.append(float(np.max(np.abs(actual - expected))))
    # Permuting action order must preserve state preprocessing and scatter.
    names = tuple(np.asarray(ATEC_JOINT_NAMES)[rng.permutation(23)].tolist())
    controller = D1FlatBatchedPolicy(4, policy_path, names, normalization_backend="reference")
    singles = [D1FlatLoadedPolicy(policy_path, names) for _ in range(4)]
    for single in singles:
        # Keep this reference-specific check deterministic even when ORT is
        # installed in the Python environment running the validation command.
        single._session = None
        single._reference = reference
    state_errors = []
    for step in range(35):
        if step in (3, 12, 23):
            ids = [0, 2] if step != 12 else [1]
            controller.reset(ids)
            for i in ids:
                singles[i].reset()
        inputs = [rng.normal(size=(4, w)).astype(np.float32) * .3 for w in (3, 3, 23, 23, 3)]
        expected = np.stack([singles[i].action_from_state(*(x[i] for x in inputs)) for i in range(4)])
        actual = controller.action_from_state(*inputs).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        assert np.count_nonzero(actual[:, [names.index(n) for n in ATEC_JOINT_NAMES[-7:]]]) == 0
        state_errors.append(float(np.max(np.abs(actual - expected))))
    sample = torch.randn(256, 57)
    history = torch.randn(256, 10, 57)
    with torch.no_grad():
        for _ in range(10):
            net(sample, history)
        start = time.perf_counter()
        for _ in range(100):
            net(sample, history)
        milliseconds = (time.perf_counter() - start) * 10.0
    print(json.dumps({
        "sha256": net.sha256, "device": "cpu", "torch": torch.__version__,
        "reference_backend": "onnx.reference.ReferenceEvaluator",
        "network_batches_checked": len(records), "batch_sizes": [1, 7, 32],
        "input_standard_deviations": [0.0, 0.1, 1.0, 3.0],
        "normalization_backend": "reference",
        "network_max_abs_error": max(records), "network_tolerance": {"rtol": 3e-5, "atol": 5e-4},
        "stateful_controller_steps": 35, "controller_envs": 4,
        "controller_max_abs_error": max(state_errors), "controller_tolerance": {"rtol": 1e-4, "atol": 1e-4},
        "controller_checks": ["permuted_joint_names", "selective_resets", "zero_arm_actions"],
        "batch_256_mean_forward_ms_one_cpu_thread": milliseconds,
    }, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--validate", action="store_true", help="Run CPU ONNX reference parity checks")
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    if args.validate:
        if args.cases < 1:
            parser.error("--cases must be positive")
        _validate(args.policy, args.cases)
    else:
        parser.error("Specify --validate to run parity checks, or import the module")
