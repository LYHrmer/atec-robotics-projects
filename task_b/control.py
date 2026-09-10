"""Open-loop Task B bootstrap policies for the original ATEC-TaskB-B2wPiper.

These policies measure mobility and posture only. They receive the public
proprioception vector plus a static joint/action schema that the evaluator reads
from the live action manager at startup; they never receive the robot root pose,
object poses or any other simulator state.

Every mode spends `settle_calls` calls on the all-zero normalized action, then
ramps to its target over `ramp_calls` calls. Zero is the official rest command:
the leg and arm terms are position terms with `use_default_offset=True`, so zero
requests the default joint positions, and the wheel term is a velocity term with
a zero default velocity offset. Scales are read per term from the real
configuration and are never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

MODES = ("hold", "forward", "turn", "crouch")

SETTLE_CALLS = 100
RAMP_CALLS = 100
#: Normalized forward command per wheel. The official wheel scale is 5.0, so
#: 0.10 asks for 0.5 rad/s of wheel velocity; the resulting linear speed and the
#: sign convention are hypotheses that the evaluator measures, not knowledge.
FORWARD_WHEEL_CMD = 0.10

#: Experimental crouch posture, applied to the *actual* defaults and clipped to
#: the robot's real soft joint limits. Reaching the ground is unproven.
CROUCH_LEG_TARGETS = {"hip": 0.0, "thigh": 1.0, "calf": -2.0}

LEG_TERM, WHEEL_TERM, ARM_TERM = "joint_leg", "joint_wheel", "joint_arm"
EXPECTED_TERM_DIMS = {LEG_TERM: 12, WHEEL_TERM: 4, ARM_TERM: 8}
EXPECTED_TOTAL_ACTIONS = 24
EXPECTED_WHEEL_JOINTS = ("FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint")
LEG_JOINT_PATTERN = re.compile(r"^(?P<corner>FR|FL|RR|RL)_(?P<link>hip|thigh|calf)_joint$")


class SchemaError(ValueError):
    """The live task does not match the Task B schema this bootstrap supports."""


@dataclass(frozen=True)
class ActionTerm:
    """One action-manager term, as resolved by the running environment."""

    name: str
    start: int
    dim: int
    joint_names: tuple[str, ...]
    mode: str  # "position", "velocity" or "effort"
    scale: float
    use_default_offset: bool | None
    clip: object
    joint_names_source: str  # "resolved" (action term) or "cfg" (term config)

    @property
    def stop(self) -> int:
        return self.start + self.dim

    def to_dict(self) -> dict:
        return {
            "name": self.name, "action_slice": [self.start, self.stop], "dim": self.dim,
            "joint_names": list(self.joint_names), "joint_names_source": self.joint_names_source,
            "mode": self.mode, "scale": self.scale,
            "use_default_offset": self.use_default_offset, "clip": repr(self.clip),
        }


@dataclass(frozen=True)
class ActionSchema:
    """Static joint/action schema: term order, joint order, defaults and limits."""

    terms: tuple[ActionTerm, ...]
    total_dim: int
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray  # (num_joints,)
    soft_joint_pos_limits: np.ndarray  # (num_joints, 2)

    @classmethod
    def from_terms(cls, terms, joint_names, default_joint_pos, soft_joint_pos_limits, total_dim):
        """Validate the live schema. Raises SchemaError instead of guessing."""
        schema = cls(
            terms=tuple(terms), total_dim=int(total_dim), joint_names=tuple(joint_names),
            default_joint_pos=np.asarray(default_joint_pos, dtype=np.float64).reshape(-1),
            soft_joint_pos_limits=np.asarray(soft_joint_pos_limits, dtype=np.float64).reshape(-1, 2),
        )
        schema.validate()
        return schema

    def validate(self) -> None:
        num_joints = len(self.joint_names)
        if self.default_joint_pos.shape != (num_joints,):
            raise SchemaError(f"default_joint_pos {self.default_joint_pos.shape} != ({num_joints},)")
        if self.soft_joint_pos_limits.shape != (num_joints, 2):
            raise SchemaError(f"soft_joint_pos_limits {self.soft_joint_pos_limits.shape} != ({num_joints}, 2)")
        if not np.isfinite(self.default_joint_pos).all():
            raise SchemaError("Non-finite default joint positions")
        if len(set(self.joint_names)) != num_joints:
            raise SchemaError("Duplicate articulation joint names")

        summed = sum(term.dim for term in self.terms)
        if summed != self.total_dim or self.total_dim != EXPECTED_TOTAL_ACTIONS:
            raise SchemaError(
                f"Expected {EXPECTED_TOTAL_ACTIONS} B2w+Piper actions; manager reports total "
                f"{self.total_dim} and terms {[(t.name, t.dim) for t in self.terms]}"
            )
        offset = 0
        for term in self.terms:
            if term.start != offset:
                raise SchemaError(f"Term '{term.name}' starts at {term.start}, expected {offset}")
            offset += term.dim
            if len(term.joint_names) != term.dim:
                raise SchemaError(
                    f"Term '{term.name}' has {len(term.joint_names)} joint names for {term.dim} actions"
                )
            if len(set(term.joint_names)) != term.dim:
                raise SchemaError(f"Duplicate joint names in '{term.name}'")
            unknown = [name for name in term.joint_names if name not in self.joint_names]
            if unknown:
                raise SchemaError(f"Term '{term.name}' uses joints absent from the articulation: {unknown}")
            if not np.isfinite(term.scale) or term.scale <= 0.0:
                raise SchemaError(f"Term '{term.name}' has unusable scale {term.scale!r}")

        names = {term.name: term for term in self.terms}
        missing = sorted(set(EXPECTED_TERM_DIMS) - set(names))
        if missing:
            raise SchemaError(f"Missing expected action terms {missing}; found {sorted(names)}")
        for name, dim in EXPECTED_TERM_DIMS.items():
            if names[name].dim != dim:
                raise SchemaError(f"Term '{name}' has {names[name].dim} actions, expected {dim}")
        for name in (LEG_TERM, ARM_TERM):
            term = names[name]
            if term.mode != "position" or not term.use_default_offset:
                raise SchemaError(
                    f"Term '{name}' is {term.mode} with use_default_offset={term.use_default_offset}; "
                    "this bootstrap only sends default-offset position actions to legs and arm"
                )
            limits = self.soft_joint_pos_limits[[self.joint_index(joint) for joint in term.joint_names]]
            if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
                raise SchemaError(f"Invalid position limits in '{name}'")
        # Continuous wheels are velocity controlled. Isaac's soft position
        # limit calculation can yield NaN from +/-inf; those limits are unused.
        if names[WHEEL_TERM].mode != "velocity":
            raise SchemaError(f"Term '{WHEEL_TERM}' is {names[WHEEL_TERM].mode}, expected velocity")

        wheels = names[WHEEL_TERM].joint_names
        if set(wheels) != set(EXPECTED_WHEEL_JOINTS):
            raise SchemaError(f"Wheel joints {list(wheels)} != expected {list(EXPECTED_WHEEL_JOINTS)}")
        for joint in names[LEG_TERM].joint_names:
            if not LEG_JOINT_PATTERN.match(joint):
                raise SchemaError(f"Unrecognized leg joint '{joint}'; cannot derive a crouch target")

    def term(self, name: str) -> ActionTerm:
        for term in self.terms:
            if term.name == name:
                return term
        raise SchemaError(f"Action term '{name}' not present")

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def to_dict(self) -> dict:
        return {
            "total_action_dim": self.total_dim,
            "terms": [term.to_dict() for term in self.terms],
            "articulation_joint_names": list(self.joint_names),
            "default_joint_pos": self.default_joint_pos.tolist(),
            "soft_joint_pos_limits": [[float(x) if np.isfinite(x) else None for x in pair]
                                      for pair in self.soft_joint_pos_limits],
            "null_limit_definition": "unbounded/undefined soft position limit on a continuous velocity-controlled wheel",
            "note": "Action indices come from the action manager term order, which is independent "
                    "of the articulation (USD) joint order listed here.",
        }


def wheel_side(joint_name: str) -> str:
    """Return 'left' or 'right' for a B2w wheel joint from its FR/FL/RR/RL corner."""
    corner = joint_name.split("_")[0]
    if len(corner) != 2 or corner[0] not in "FR" or corner[1] not in "LR":
        raise SchemaError(f"Cannot read a side from wheel joint '{joint_name}'")
    return "left" if corner[1] == "L" else "right"


class BootstrapPolicy:
    """Open-loop settle/ramp policy for one of the four bootstrap modes."""

    def __init__(self, schema: ActionSchema, mode: str, settle_calls: int = SETTLE_CALLS,
                 ramp_calls: int = RAMP_CALLS, wheel_cmd: float = FORWARD_WHEEL_CMD,
                 crouch_fraction: float = 1.0):
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'; expected one of {list(MODES)}")
        if settle_calls < 0 or ramp_calls < 1:
            raise ValueError("settle_calls must be >= 0 and ramp_calls >= 1")
        if not np.isfinite(wheel_cmd):
            raise ValueError("wheel_cmd must be finite")
        if not np.isfinite(crouch_fraction) or not 0 <= crouch_fraction <= 1:
            raise ValueError("crouch_fraction must be in [0, 1]")
        self.crouch_fraction = float(crouch_fraction)
        self.schema, self.mode = schema, mode
        self.settle_calls, self.ramp_calls, self.wheel_cmd = int(settle_calls), int(ramp_calls), float(wheel_cmd)
        self.calls, self.alpha = 0, 0.0

        wheel = schema.term(WHEEL_TERM)
        self._wheel_indices = {name: wheel.start + offset for offset, name in enumerate(wheel.joint_names)}
        self._wheel_sides = {name: wheel_side(name) for name in wheel.joint_names}
        self._crouch = self._build_crouch_targets()

    def _build_crouch_targets(self) -> dict:
        """Full-crouch leg targets from the real defaults, clipped to real limits."""
        leg = self.schema.term(LEG_TERM)
        requested, clipped, defaults, lower, upper = [], [], [], [], []
        for joint in leg.joint_names:
            index = self.schema.joint_index(joint)
            link = LEG_JOINT_PATTERN.match(joint).group("link")
            low, high = self.schema.soft_joint_pos_limits[index]
            target = float(CROUCH_LEG_TARGETS[link])
            requested.append(target)
            clipped.append(float(np.clip(target, low, high)))
            defaults.append(float(self.schema.default_joint_pos[index]))
            lower.append(float(low))
            upper.append(float(high))
        requested_np, clipped_np, defaults_np = (np.asarray(values, dtype=np.float64)
                                                for values in (requested, clipped, defaults))
        # Position term: processed target = default + scale * action.
        normalized = (clipped_np - defaults_np) / leg.scale
        if not np.isfinite(normalized).all():
            raise SchemaError("Non-finite crouch action; check the leg scale and joint limits")
        return {
            "joint_names": list(leg.joint_names), "requested_target_rad": requested_np.tolist(),
            "clipped_target_rad": clipped_np.tolist(), "default_rad": defaults_np.tolist(),
            "soft_limit_lower_rad": lower, "soft_limit_upper_rad": upper,
            "clipped_by_limits": [bool(abs(a - b) > 1e-9) for a, b in zip(requested, clipped)],
            "full_crouch_normalized_action": normalized.tolist(), "leg_scale": leg.scale,
            "_normalized": normalized,
        }

    def _ramp(self) -> float:
        if self.calls <= self.settle_calls:
            return 0.0
        if self.calls >= self.settle_calls + self.ramp_calls:
            return 1.0
        return (self.calls - self.settle_calls) / self.ramp_calls

    def act(self, proprio) -> np.ndarray:
        """Return one normalized action from the public proprio vector only."""
        proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio.size == 0 or not np.isfinite(proprio).all():
            raise ValueError(f"Non-finite or empty proprio observation of size {proprio.size}")
        self.calls += 1
        self.alpha = alpha = self._ramp()
        action = np.zeros(self.schema.total_dim, dtype=np.float64)

        if self.mode == "forward":
            for index in self._wheel_indices.values():
                action[index] = alpha * self.wheel_cmd
        elif self.mode == "turn":
            # Sign convention under test: right wheels positive, left wheels
            # negative. Which way the base actually yaws is measured, not known.
            for joint, index in self._wheel_indices.items():
                sign = 1.0 if self._wheel_sides[joint] == "right" else -1.0
                action[index] = sign * alpha * self.wheel_cmd
        elif self.mode == "crouch":
            leg = self.schema.term(LEG_TERM)
            action[leg.start:leg.stop] = alpha * self.crouch_fraction * self._crouch["_normalized"]

        if not np.isfinite(action).all():
            raise ValueError(f"Non-finite action produced in mode '{self.mode}'")
        return action.astype(np.float32)

    def describe(self) -> dict:
        crouch = {key: value for key, value in self._crouch.items() if not key.startswith("_")}
        return {
            "mode": self.mode, "settle_calls": self.settle_calls, "ramp_calls": self.ramp_calls,
            "crouch_fraction": self.crouch_fraction,
            "wheel_normalized_cmd": self.wheel_cmd,
            "wheel_action_index": dict(sorted(self._wheel_indices.items())),
            "wheel_side": dict(sorted(self._wheel_sides.items())),
            "wheel_scale": self.schema.term(WHEEL_TERM).scale,
            "wheel_target_rad_per_s_at_full_ramp": self.wheel_cmd * self.schema.term(WHEEL_TERM).scale,
            "crouch_target": crouch,
            "inputs": "public proprio observation and the static joint/action schema only",
            "sign_hypotheses": {
                "forward": "all four wheels positive; the direction of travel is measured from the "
                           "recorded base position, not assumed",
                "turn": "right wheels positive and left wheels negative; the yaw direction is measured",
            },
        }
