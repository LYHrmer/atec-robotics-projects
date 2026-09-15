"""Parallel Task B *delivery* training environment for the local D1 + G2 model.

This is a training proxy for ATEC Task B, not the official Task B environment.
Differences that matter, stated once and repeated in the training metadata:

* One object per environment, not the official 18.
* Grasping is skipped.  The object is *kinematically carried* at a fixed
  offset in front of the robot base (a virtual tray) and is released
  automatically once it is above the bin interior and the robot is still
  driving forward.  A kinematically carried object applies no load to the
  robot, so this proves navigation + delivery geometry, never manipulation.
* Flat ground with one bin, not the official Task B scene layout.
* The bin is the official Task B mesh (annulus wall + disc bottom from
  ``flat_terrain_with_trash_bin``) scaled uniformly by ``bin_scale`` (default
  2x, i.e. outer radius 2.00 m and rim 1.10 m).  It is scaled because the official 1.0 m
  radius is sized for robots that drive *into* the circle, while D1's
  ``base_link``/hips/thighs are illegal-contact bodies and must stop short of
  the wall.  The success test keeps the official form -- object centre within
  the circle and below the rim -- with radius and height limit scaled by the
  same factor.
* The release is scripted, not learned: the object is dropped when the tray
  crosses the release radius *while the robot is still driving forward*, which
  is a state the learned forward command makes easy to reach and hold.
* Rewards here are training signals, not competition scores.  Nothing in
  this file establishes an official Task B result.

The robot, actuators, 23-action order/scales, physics step, contact rule and
81-value proprioception all come from the local Task A evaluation adapter, so
the frozen ``flat_lab.onnx`` locomotion backbone stays valid.

Action interface used by ``train_d1g2_taskb_residual.py`` (18 values):
``[0:16]`` residual joint deltas on the frozen base action (12 leg positions,
4 wheel velocities), ``[16]`` forward velocity command and ``[17]`` yaw-rate
command for the frozen backbone.  The learned command is what lets the policy
steer with the backbone's own trained turning behaviour instead of fighting it.

Importing this module does not start or import Isaac Sim; call the builder
after ``AppLauncher``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

# --- Task B geometry (kept in one place; the bin sits on the tile origin) ---
BIN_XY_IN_TILE = (0.0, 0.0)
BIN_RADIUS = 1.0
BIN_RIM_Z = 0.55
SUCCESS_Z_MAX = 0.5
SUCCESS_SPEED = 0.4
# The official circle is 1.0 m in radius, but merely landing anywhere inside it
# is reachable by walking past, which is not a delivery.  Delivery additionally
# requires the object to settle near the middle.  Scaled with the bin.
SUCCESS_DISTANCE_FRACTION = 0.5

# --- geometry ---------------------------------------------------------------
# Dimensions below are the ones the terrain function actually builds at the
# default 2x scale, NOT the official 1x ones:
#
#   outer wall radius  = 1.00 * bin_scale            = 2.00 m
#   wall thickness     = 0.02 * bin_scale            = 0.04 m
#   rim height         = 0.05 * bin_scale (bottom)
#                      + 0.50 * bin_scale (wall)     = 1.10 m
#
# base_link, [FR]_base_link, .*_hip and .*_thigh are illegal-contact bodies and
# the wall is solid, so the base has to stop clear of the wall.  Measured from
# the composed D1+G2 model those bodies reach 0.27 m ahead of the root, so the
# base must stay outside 2.27 m.  The object therefore has to be carried far
# enough ahead that it is already over the middle while the base is still well
# outside -- the tray reach is what buys that separation, not the release
# radius.
#
# At the defaults (verified arithmetically in tools/verify_d1g2_taskb_policy.py
# and by the --probe_only geometry dump):
#   release at r <= 0.40 m -> base at 2.80 + 0.40 = 3.20 m, 0.93 m clear
#   object held 1.00 m above the root (~0.43 m) -> 1.43 m, 0.33 m above the rim
#   object dropped with no horizontal velocity -> lands at its release radius,
#     0.40 m from the centre, 0.60 m inside the 1.00 m success circle
# The trained checkpoint in weights/ used 0.40 m; the maximum the wall clearance
# permits is 0.53 m (CARRY_OFFSET[0] - 0.27 - wall radius), and the builder
# refuses anything larger.
#
# Heading matters: the release radius is measured along the tray, so a robot
# that is 45 deg off still clears the wall (2.48 m), while 60 deg off does not.
# The policy therefore has to face the bin to deliver, which is the point.
CARRY_OFFSET = (2.80, 0.0, 1.00)
RELEASE_RADIUS = 0.40
RELEASE_SPEED = 0.55
# Spawn ring and heading spread.
#
# The first version of this environment spawned the robot already facing the
# bin and at a fixed radius, and a control run delivered 101/103 episodes with
# an *untrained* policy: walking straight forward is enough to fall into the
# circle, so the task trained no navigation.  Spawning anywhere on the circle
# facing anywhere is what makes the yaw command load-bearing -- the policy has
# to turn toward the bin before it can close the distance.
SPAWN_MIN_DEFAULT = 4.0
SPAWN_MAX_DEFAULT = 6.5
# pi/2, not pi: the release radius is measured along a 2.8 m tray, so a robot
# facing away cannot bring the tray into the release band at all and the episode
# is unsolvable rather than merely hard.
HEADING_NOISE_DEFAULT = 1.5707963267948966
# Where the training bin sits relative to the official one.
BIN_SCALE_DEFAULT = 2.0

GOAL_OBS_DIM = 9
PRIVILEGED_OBS_DIM = 10
NUM_RESIDUAL_ACTIONS = 16
NUM_COMMAND_ACTIONS = 2
NUM_ACTIONS = NUM_RESIDUAL_ACTIONS + NUM_COMMAND_ACTIONS

# Learned command mapping, kept inside the frozen backbone's training range
# (Task A trained it on forward 0.35--0.85 m/s and yaw rate +/-0.5 rad/s).
COMMAND_VX_BIAS = 0.25
COMMAND_VX_SCALE = 0.60
# Negative is allowed so braking and backing out of an overshoot are reachable.
# Task A's own velocity range is (-1.0, 1.0), so reversing stays in-distribution
# for the frozen backbone; the forward end is capped at the 0.85 m/s the
# backbone was actually trained on.
COMMAND_VX_RANGE = (-0.30, 0.85)
COMMAND_WZ_SCALE = 0.50
COMMAND_WZ_RANGE = (-0.5, 0.5)

FAILURE_TERMS = ("illegal_contact", "bad_orientation", "too_far")


def command_from_action(action):
    """Map policy channels 16/17 to a (vx, vy=0, wz) backbone command."""
    import torch

    vx = (COMMAND_VX_BIAS + COMMAND_VX_SCALE * action[:, 0]).clamp(*COMMAND_VX_RANGE)
    wz = (COMMAND_WZ_SCALE * action[:, 1]).clamp(*COMMAND_WZ_RANGE)
    return torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)


def scaled_flat_terrain_with_bin(difficulty, cfg):
    """Official Task B bin geometry, scaled about the bin center by ``cfg.scale``.

    Same construction as ``atec_rl_lab.tasks.task_b.terrain``: a ground slab
    plus a solid disc bottom and an annular wall, in the terrain's local frame
    with the origin at the tile center.  ``scale`` is a training-only change;
    at ``scale=1`` this reproduces the upstream mesh exactly.
    """
    import numpy as np
    import trimesh

    scale = float(getattr(cfg, "scale", 1.0))
    mesh_list = []

    ground = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], 0.10 * scale),
        transform=trimesh.transformations.translation_matrix((0.0, 0.0, -0.005 * scale)),
    )
    mesh_list.append(ground)

    bin_x = cfg.trash_bin_x
    bin_y = float(getattr(cfg, "trash_bin_y", 0.0))
    bin_radius = 1.0 * scale
    bin_height = 0.5 * scale
    wall_thickness = 0.02 * scale
    bottom_thickness = 0.05 * scale
    bin_color = np.asarray([255, 128, 0, 255], dtype=np.uint8)

    bottom = trimesh.creation.cylinder(
        radius=bin_radius,
        height=bottom_thickness,
        transform=trimesh.transformations.translation_matrix((bin_x, bin_y, bottom_thickness / 2)),
    )
    bottom.visual.vertex_colors = np.tile(bin_color, (bottom.vertices.shape[0], 1))
    mesh_list.append(bottom)

    wall = trimesh.creation.annulus(
        r_min=max(0.0, bin_radius - wall_thickness),
        r_max=bin_radius,
        height=bin_height,
        transform=trimesh.transformations.translation_matrix((bin_x, bin_y, bottom_thickness + bin_height / 2)),
    )
    wall.visual.vertex_colors = np.tile(bin_color, (wall.vertices.shape[0], 1))
    mesh_list.append(wall)

    return mesh_list, np.array([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# shared per-environment task state
# ---------------------------------------------------------------------------
def taskb_state(env):
    """Return (creating on first use) the delivery state attached to ``env``.

    Managers are independent objects, so the carry/release bookkeeping shared
    by events, rewards, terminations and observations lives on the env itself.
    """
    import torch

    state = getattr(env, "_d1g2_taskb_state", None)
    if state is not None and state["carrying"].shape[0] == env.num_envs:
        return state
    device = env.device
    n = env.num_envs
    state = {
        "carrying": torch.ones(n, dtype=torch.bool, device=device),
        "released": torch.zeros(n, dtype=torch.bool, device=device),
        "reach_counted": torch.zeros(n, dtype=torch.bool, device=device),
        "success_counted": torch.zeros(n, dtype=torch.bool, device=device),
        "steps_since_release": torch.zeros(n, dtype=torch.long, device=device),
        "release_height": torch.zeros(n, device=device),
        "prev_dist": torch.zeros(n, device=device),
        "min_dist": torch.full((n,), 1.0e3, device=device),
        "command": torch.zeros(n, 3, device=device),
        # spawn curriculum, widened/narrowed by ``delivery_curriculum``
        "spawn_min": SPAWN_MIN_DEFAULT,
        "spawn_max": SPAWN_MAX_DEFAULT,
        "heading_noise": HEADING_NOISE_DEFAULT,
        "bin_scale": 1.0,
        "bin_radius": BIN_RADIUS,
        "success_radius": BIN_RADIUS * SUCCESS_DISTANCE_FRACTION,
        "release_radius": RELEASE_RADIUS,
        "ground_z": 0.0,
        "success_ema": 0.0,
        "episodes": 0,
        "curriculum_successes": 0.0,
        "curriculum_samples": 0,
    }
    state["command"][:, 0] = COMMAND_VX_BIAS
    env._d1g2_taskb_state = state
    return state


def bin_center_w(env):
    """World xy of each environment's bin (the bin sits on the tile origin)."""
    import torch

    offset = torch.tensor(BIN_XY_IN_TILE, device=env.device)
    return env.scene.env_origins[:, :2] + offset


def _carry_pose_w(env):
    """World pose the carried object is held at: base yaw frame + offset."""
    import torch
    from isaaclab.utils.math import quat_apply_yaw, yaw_quat

    robot = env.scene["robot"]
    quat = robot.data.root_quat_w
    offset = torch.tensor(CARRY_OFFSET, device=env.device).expand(env.num_envs, 3)
    return robot.data.root_pos_w + quat_apply_yaw(quat, offset), yaw_quat(quat)


def _object_dist_to_bin(env):
    import torch

    obj = env.scene["object"]
    return torch.norm(obj.data.root_pos_w[:, :2] - bin_center_w(env), dim=1)


def _robot_dist_to_bin(env):
    import torch

    robot = env.scene["robot"]
    return torch.norm(robot.data.root_pos_w[:, :2] - bin_center_w(env), dim=1)


def _heading_error(env):
    """Signed yaw error between the base forward axis and the bin direction."""
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    robot = env.scene["robot"]
    delta = bin_center_w(env) - robot.data.root_pos_w[:, :2]
    desired = torch.atan2(delta[:, 1], delta[:, 0])
    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]
    return wrap_to_pi(desired - yaw)


def delivery_success(env):
    """Official Task B test, applied to the released object once it settles.

    ``BIN_RADIUS`` / ``SUCCESS_Z_MAX`` are the official 1.0 m / 0.5 m values
    scaled by the same ``bin_scale`` the terrain used, so the test keeps its
    shape and only its size changes.
    """
    import torch

    state = taskb_state(env)
    obj = env.scene["object"]
    pos = obj.data.root_pos_w
    inside = _object_dist_to_bin(env) <= state["bin_radius"] * SUCCESS_DISTANCE_FRACTION
    height_ok = (pos[:, 2] - _ground_z(env) >= 0.0) & (pos[:, 2] - _ground_z(env) <= SUCCESS_Z_MAX * state["bin_scale"])
    settled = torch.norm(obj.data.root_lin_vel_w, dim=1) < SUCCESS_SPEED
    return (~state["carrying"]) & inside & height_ok & settled


def _ground_z(env):
    """Tile surface height for each environment (the bin stands on it)."""
    return env.scene.env_origins[:, 2]


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def reset_robot_on_ring(env, env_ids, height: float = 0.6):
    """Spawn the robot on a ring around the bin, roughly facing it.

    Distance range and heading noise come from the task state so the spawn
    curriculum can widen them; both are absolute world placements, never
    terrain-origin offsets added on top of an absolute pose.
    """
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi

    state = taskb_state(env)
    robot = env.scene["robot"]
    n = len(env_ids)
    device = env.device
    center = bin_center_w(env)[env_ids]

    angle = torch.rand(n, device=device) * (2.0 * torch.pi)
    radius = state["spawn_min"] + torch.rand(n, device=device) * (state["spawn_max"] - state["spawn_min"])
    pos_xy = center + radius.unsqueeze(-1) * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    # Facing the bin is angle + pi; the noise is what forces the policy to turn.
    noise = (torch.rand(n, device=device) * 2.0 - 1.0) * state["heading_noise"]
    yaw = wrap_to_pi(angle + torch.pi + noise)
    zero = torch.zeros(n, device=device)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] = pos_xy
    root_state[:, 2] = state["ground_z"] + height
    root_state[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def reset_carried_object(env, env_ids):
    """Put the object back on the virtual tray and clear delivery bookkeeping.

    Must run after the robot root reset: the carry pose is read from the robot.
    """
    import torch

    state = taskb_state(env)
    obj = env.scene["object"]
    pos, quat = _carry_pose_w(env)
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device)
    obj.write_root_pose_to_sim(torch.cat((pos[ids], quat[ids]), dim=-1), env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)

    state["carrying"][ids] = True
    state["released"][ids] = False
    state["reach_counted"][ids] = False
    state["success_counted"][ids] = False
    state["steps_since_release"][ids] = 0
    state["release_height"][ids] = 0.0
    dist = torch.norm(pos[ids, :2] - bin_center_w(env)[ids], dim=1)
    state["prev_dist"][ids] = dist
    state["min_dist"][ids] = dist
    state["command"][ids] = 0.0
    state["command"][ids, 0] = COMMAND_VX_BIAS


def update_carried_object(env, env_ids, release_radius: float = RELEASE_RADIUS,
                          release_speed: float = RELEASE_SPEED):
    """Track the tray pose while carrying, and release above the bin.

    Runs once per control step (``interval`` mode).  Writing pose and the
    robot's linear velocity each step is what makes the attachment rigid; the
    object stays a normal dynamic body, so once released it simply falls in.

    The release condition is "inside the release radius *and* still moving
    forward", and on release the object's horizontal velocity is cleared: the
    tray sets the object down rather than throwing it.

    Carrying the robot's forward velocity into the drop was the reason the
    first training run plateaud.  At the old 1.2 m release radius the object
    left the tray at ~0.45 m/s and travelled a further ~0.2 m before landing,
    so it came to rest near the rim and the policy could not learn the extra
    precision needed to centre it -- the delivery rate flatlined at ~10-22%.
    Setting it down means the landing point *is* the release point, so the
    precision the policy must learn is expressed directly in its own position.

    The object's height is recorded so ``--probe_only`` can confirm it cleared
    the rim before falling.
    """
    import torch

    state = taskb_state(env)
    obj = env.scene["object"]
    robot = env.scene["robot"]
    carrying = state["carrying"]
    if bool(carrying.any()):
        ids = carrying.nonzero().flatten()
        pos, quat = _carry_pose_w(env)
        obj.write_root_pose_to_sim(torch.cat((pos[ids], quat[ids]), dim=-1), env_ids=ids)
        # No horizontal carry-over: settling straight down is what makes the
        # landing point predictable.  Spin is left alone so the object tumbles
        # naturally while it falls.
        velocity = torch.zeros(len(ids), 6, device=env.device)
        obj.write_root_velocity_to_sim(velocity, env_ids=ids)
        dist = torch.norm(pos[ids, :2] - bin_center_w(env)[ids], dim=1)
        forward_speed = robot.data.root_lin_vel_b[ids, 0]
        drop = ids[(dist <= release_radius) & (forward_speed > release_speed)]
        state["carrying"][drop] = False
        state["released"][drop] = True
        state["release_height"][drop] = pos[drop, 2]
    state["steps_since_release"] += (~state["carrying"]).long()
    state["min_dist"] = torch.minimum(state["min_dist"], _object_dist_to_bin(env))


def delivery_curriculum(env, env_ids, window: int = 512,
                        success_rate_up: float = 0.55, success_rate_down: float = 0.15):
    """Widen the spawn ring and heading noise once deliveries start succeeding.

    This term does **not** use ``env_ids`` as "the episodes that just finished".
    ``CurriculumManager.compute`` is called from ``_reset_idx`` on every step in
    which *any* environment resets, and ``env_ids`` is the whole batch of
    resetters -- with hundreds of parallel environments that is every step, and
    on the very first reset it is the full batch of brand-new episodes that
    cannot have succeeded.  An earlier version nudged the difficulty once per
    such call, which walked ``heading_noise`` from pi down to its floor in about
    two hundred *steps* -- a few iterations -- and silently deleted the random
    spawn heading that makes the yaw command worth learning.

    So instead: accumulate the outcome of the episodes that are actually ending
    (``delivery_success`` is only ever true for an episode that has been
    through a full drop), and adjust at most once per ``window`` completed
    episodes.  The step size is per window, not per call.

    Returns the current heading noise for the training log.
    """
    state = taskb_state(env)
    finished = env.episode_length_buf[env_ids] > 0
    if bool(finished.any()):
        ids = env_ids[finished]
        state["curriculum_successes"] += float(delivery_success(env)[ids].float().sum())
        state["curriculum_samples"] += len(ids)
    if state["curriculum_samples"] >= window:
        rate = state["curriculum_successes"] / state["curriculum_samples"]
        state["success_ema"] = rate
        state["episodes"] = state["curriculum_samples"]
        if rate > success_rate_up:
            state["heading_noise"] = min(HEADING_NOISE_DEFAULT, state["heading_noise"] + 0.15)
            state["spawn_max"] = min(7.0, state["spawn_max"] + 0.25)
        elif rate < success_rate_down:
            state["heading_noise"] = max(1.2, state["heading_noise"] - 0.10)
            state["spawn_max"] = max(4.0, state["spawn_max"] - 0.25)
        state["curriculum_successes"] = 0.0
        state["curriculum_samples"] = 0
    return state["heading_noise"]


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def learned_velocity_command(env):
    """The backbone command the policy asked for on the previous step."""
    return taskb_state(env)["command"]


def goal_observation(env):
    """9 goal-conditioning values, all robot-relative (no world coordinates)."""
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    state = taskb_state(env)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    center = bin_center_w(env)
    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    def to_base(delta):
        x = delta[:, 0] * cos_yaw + delta[:, 1] * sin_yaw
        y = -delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
        return x, y

    rx, ry = to_base(center - robot.data.root_pos_w[:, :2])
    ox, oy = to_base(center - obj.data.root_pos_w[:, :2])
    dist = _robot_dist_to_bin(env)
    error = _heading_error(env)
    return torch.stack(
        (
            (rx * 0.2).clamp(-2.0, 2.0),
            (ry * 0.2).clamp(-2.0, 2.0),
            (dist * 0.2).clamp(0.0, 2.0),
            torch.cos(error),
            torch.sin(error),
            (ox * 0.2).clamp(-2.0, 2.0),
            (oy * 0.2).clamp(-2.0, 2.0),
            ((obj.data.root_pos_w[:, 2] - _ground_z(env) - BIN_RIM_Z * state["bin_scale"])
             .clamp(-1.0, 1.0)),
            state["carrying"].float(),
        ),
        dim=-1,
    )


def privileged_observation(env):
    """10 critic-only values: true object state relative to the robot."""
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    state = taskb_state(env)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    delta = obj.data.root_pos_w - robot.data.root_pos_w
    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    bx = delta[:, 0] * cos_yaw + delta[:, 1] * sin_yaw
    by = -delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
    center_delta = bin_center_w(env) - robot.data.root_pos_w[:, :2]
    return torch.stack(
        (
            bx.clamp(-5.0, 5.0),
            by.clamp(-5.0, 5.0),
            (delta[:, 2]).clamp(-2.0, 2.0),
            *[obj.data.root_lin_vel_w[:, i].clamp(-5.0, 5.0) for i in range(3)],
            (center_delta[:, 0] * 0.2).clamp(-2.0, 2.0),
            (center_delta[:, 1] * 0.2).clamp(-2.0, 2.0),
            state["carrying"].float(),
            (state["steps_since_release"].float() / 100.0).clamp(0.0, 5.0),
        ),
        dim=-1,
    )


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def approach_progress(env, max_rate: float = 2.0):
    """Potential-based closing speed of the *object* on the bin, while carried.

    Expressed as a rate because Isaac Lab multiplies every reward by the step
    time, so the integral over an episode is ``weight * metres closed``.
    """
    import torch

    state = taskb_state(env)
    dist = _object_dist_to_bin(env)
    rate = ((state["prev_dist"] - dist) / env.step_dt).clamp(-max_rate, max_rate)
    state["prev_dist"] = dist
    return rate * state["carrying"].float()


def heading_alignment(env):
    """Dense reward for pointing the base at the bin: cos(yaw error).

    1 when the robot faces the bin, -1 when it faces directly away, 0 when
    broadside.  This is the term that makes the first part of the task
    learnable: with a fully random spawn heading, "is the bin in front of me"
    is the only signal that distinguishes a good yaw command from a bad one
    before any episode has ever been delivered.
    """
    import torch

    return torch.cos(_heading_error(env))


def heading_error_penalty(env, min_dist: float = 1.2):
    """Absolute yaw error toward the bin while carrying and still far away."""
    import torch

    state = taskb_state(env)
    active = state["carrying"] & (_robot_dist_to_bin(env) > min_dist)
    return _heading_error(env).abs() * active.float()


def objective_proximity(env, sigma: float = 1.0):
    """Dense exponential reward for the object sitting close to the bin centre.

    ``approach_progress`` only pays while the distance is shrinking, so it
    gives no gradient once the robot has stopped or is circling.  This term
    keeps paying for being close, which is what shapes the final approach and
    makes "stop near the middle" the resting optimum rather than "stop
    anywhere".
    """
    import torch

    state = taskb_state(env)
    distance = _object_dist_to_bin(env)
    return torch.exp(-((distance / sigma) ** 2)) * state["carrying"].float()


def approach_speed_penalty(env, zone_extra: float = 0.5):
    """Penalise forward speed inside the delivery zone, smoothly.

    This is the term that teaches the robot to *stop* rather than to drive on
    into the bin wall.  Whatever the release threshold, a policy that keeps its
    forward command after the object is away walks into the rim, which is an
    illegal-contact body and ends the episode.  The penalty fades in linearly
    from zero at ``release_radius + zone_extra`` to full weight at the release
    radius, so there is no discontinuity to induce chatter at the boundary.

    Only the forward component is penalised: reversing out is allowed.
    """
    import torch

    state = taskb_state(env)
    robot = env.scene["robot"]
    dist = _object_dist_to_bin(env)
    outer = state["release_radius"] + zone_extra
    weight = ((outer - dist) / zone_extra).clamp(0.0, 1.0)
    return robot.data.root_lin_vel_b[:, 0].clamp(min=0.0) * weight


def reach_bonus(env):
    """One-shot bonus at release: the tray reached the bin interior."""
    import torch

    state = taskb_state(env)
    newly = state["released"] & (~state["reach_counted"])
    state["reach_counted"] |= state["released"]
    return newly.float() / env.step_dt


def success_bonus(env):
    """One-shot bonus when the released object satisfies the official test."""
    import torch

    state = taskb_state(env)
    success = delivery_success(env)
    newly = success & (~state["success_counted"])
    state["success_counted"] |= success
    return newly.float() / env.step_dt


def failure_penalty(env, term_keys=FAILURE_TERMS):
    """One-shot penalty for falling, illegal contact or leaving the tile.

    Deliberately not ``is_terminated``: the success termination must not be
    punished, and a time-out is a neutral outcome here.
    """
    import torch

    total = torch.zeros(env.num_envs, device=env.device)
    active = env.termination_manager.active_terms
    for key in term_keys:
        if key in active:
            total = total + env.termination_manager.get_term(key).float()
    return total.clamp(max=1.0) / env.step_dt


def elapsed_step(env):
    """Constant 1.0 per step; a small negative weight discourages dithering."""
    import torch

    return torch.ones(env.num_envs, device=env.device)


def delivery_done(env):
    """Success: the object is settled inside the bin circle."""
    return delivery_success(env)


def delivery_resolved(env, settle_steps: int = 250):
    """Give up shortly after a release that did not land in the circle."""
    state = taskb_state(env)
    return state["released"] & (state["steps_since_release"] >= settle_steps)


def too_far_from_bin(env, max_dist: float = 8.0):
    """The training tile is finite; wandering off it is a failed episode."""
    return _robot_dist_to_bin(env) > max_dist


def global_ground_plane(env, thickness: float = 0.20, margin: float = 100.0):
    """Add one collision surface underneath every environment tile.

    The bin terrain function emits only its own 20 m tile, and PhysX filters
    collisions between environment prims, so a few hundred tiles side by side
    have no floor between them.  This plane is created at ``/World``, outside
    ``/World/envs``, so the replicator treats it as a global prim and
    deliberately keeps it collidable.

    It is built as a trimesh box and imported with ``create_prim_from_mesh`` --
    the same path the bin terrain itself uses -- rather than through Isaac Lab's
    ``GroundPlaneCfg``.  That spawner loads a USD from Nucleus and then binds a
    physics material by searching for a "Plane"-typed child prim; in this
    environment the asset did not always resolve and the search raised, which
    left the world with no floor at all.  A mesh has no such dependency.

    Top surface sits 5 mm below the tiles' own top so the robot's wheels always
    contact the tile, not this plane.
    """
    import numpy as np
    import trimesh
    from isaaclab.terrains.utils import create_prim_from_mesh

    origins = env.scene.env_origins
    center = origins.mean(dim=0)
    span = float(max(origins[:, 0].max() - origins[:, 0].min(),
                     origins[:, 1].max() - origins[:, 1].min()).abs())
    size = span + 2.0 * margin
    top = float(origins[:, 2].min()) - 0.005
    ground = trimesh.creation.box(
        extents=(size, size, thickness),
        transform=trimesh.transformations.translation_matrix(
            (float(center[0]), float(center[1]), top - thickness / 2.0)),
    )
    ground.visual.vertex_colors = np.tile(
        np.asarray([90, 90, 90, 255], dtype=np.uint8), (ground.vertices.shape[0], 1))
    create_prim_from_mesh(
        "/World/ground_plane", ground, physics_material=env.cfg.scene.terrain.physics_material)
    return ground


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
def build_d1g2_taskb_train_cfg(
    device: str = "cuda:0",
    seed: int = 42,
    ddt_root: str | Path | None = None,
    num_envs: int = 128,
    episode_length_s: float = 20.0,
    spawn_min: float = SPAWN_MIN_DEFAULT,
    spawn_max: float = SPAWN_MAX_DEFAULT,
    heading_noise: float = HEADING_NOISE_DEFAULT,
    release_radius: float = RELEASE_RADIUS,
    tile_size: float = 20.0,
    bin_scale: float = BIN_SCALE_DEFAULT,
):
    """Build the camera-free Task B delivery training configuration.

    The bin is the official Task B geometry scaled ``bin_scale``-fold about
    its centre: the official 1.0 m radius is sized for competing robots that
    drive *into* the circle, while this proxy must stop outside it because
    ``base_link``/``.*_hip``/``.*_thigh`` are illegal-contact bodies.  Scaling
    to a 2.0 m radius keeps the official shape and the official test form,
    with the constants moved into the training metadata.

    Observation groups produced for the learner:

    * ``proprio`` [N, 81] -- the evaluation adapter's exact order, with the
      command slot carrying the policy's own learned backbone command.
    * ``goal`` [N, 9] -- robot-relative bin/object geometry (actor input).
    * ``critic`` [N, 10] -- true object state, critic only.

    All environments share one 20 m tile, so every bin sits on its
    environment's origin and inter-environment collisions are filtered by the
    scene, exactly as in the Task A training proxy.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if not 0.5 <= spawn_min <= spawn_max:
        raise ValueError("spawn distances must satisfy 0.5 <= spawn_min <= spawn_max")
    if not bin_scale >= 1.0:
        raise ValueError("bin_scale must be at least 1.0 (the official size)")
    if not 0.0 <= release_radius <= BIN_RADIUS * bin_scale:
        raise ValueError("release_radius must lie inside the bin circle")
    if release_radius + 0.28 > spawn_min:
        raise ValueError("spawn_min must clear the release radius by the robot's half-length")
    if release_radius > BIN_RADIUS * bin_scale * SUCCESS_DISTANCE_FRACTION:
        raise ValueError("release_radius must be inside the success radius, or a released object "
                         "starts already outside the delivery zone")
    _reach = CARRY_OFFSET[0] - 0.27 - BIN_RADIUS * bin_scale
    if release_radius > _reach:
        raise ValueError(
            f"release_radius {release_radius} exceeds the wall clearance {_reach:.3f} m: at that "
            f"release distance the base would be inside the bin wall.  The tray must reach "
            f"further ahead (CARRY_OFFSET[0]={CARRY_OFFSET[0]}) or the release radius must shrink.")

    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp
    from isaaclab.managers import CurriculumTermCfg, EventTermCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
    from isaaclab.utils import configclass

    from atec_rl_lab.assets.objects import Mustard_cfg
    from atec_rl_lab.tasks.task_b.terrain import (
        FlatTerrainWithTrashBinCfg as _OfficialBinTerrainCfg,
    )
    from tools.d1g2_taska_env import ARM_JOINT_NAMES, LEG_JOINT_NAMES, build_d1g2_taska_cfg

    @configclass
    class ScaledBinTerrainCfg(_OfficialBinTerrainCfg):
        """Official Task B bin geometry at a configurable overall scale.

        Upstream declares only ``trash_bin_x``.  ``scale`` multiplies the
        radius and every height, so the official shape is preserved exactly
        and the offset from the tile origin stays at ``trash_bin_x``.
        """

        scale: float = 2.0
        function = scaled_flat_terrain_with_bin

    cfg = build_d1g2_taska_cfg(device=device, cameras=False, seed=seed, ddt_root=ddt_root)
    original_terrain = cfg.scene.terrain
    cfg.scene.num_envs = num_envs
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    cfg.episode_length_s = episode_length_s

    # -- scene: one flat tile carrying the official Task B bin at its origin --
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            curriculum=False,
            size=(tile_size, tile_size),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat_with_bin": ScaledBinTerrainCfg(
                    proportion=1.0,
                    trash_bin_x=BIN_XY_IN_TILE[0],
                    scale=bin_scale,
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=original_terrain.collision_group,
        physics_material=deepcopy(original_terrain.physics_material),
        visual_material=deepcopy(original_terrain.visual_material),
        debug_vis=False,
    )
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.scene.lidar_sensor = None
    cfg.observations.extero = None
    cfg.scene.contact_sensor.update_period = cfg.sim.dt
    ground_plane_spawner = global_ground_plane
    # The delivered object: the official Task B mustard bottle asset.  Its
    # spawn pose is overwritten by ``reset_carried_object`` on every reset.
    cfg.scene.object = Mustard_cfg([2.0, 0.0, 0.9], [0.0, 0.0, -0.707, 0.707], "object")
    # The terrain physics material deep-copied from Task A sets restitution 1.0
    # with multiply combine mode, so a dropped object bounces off the bin floor
    # and wanders.  Success requires it to *settle*, so the object gets its own
    # non-bouncing, higher-friction material.
    cfg.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )

    # -- commands: the backbone command is an action, not a sampled command --
    @configclass
    class NoCommandsCfg:
        pass

    cfg.commands = NoCommandsCfg()
    command_term = cfg.observations.proprio.velocity_commands
    command_term.func = learned_velocity_command
    command_term.params = {}

    @configclass
    class GoalObservationsCfg(ObservationGroupCfg):
        goal = ObservationTermCfg(func=goal_observation, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticObservationsCfg(ObservationGroupCfg):
        privileged = ObservationTermCfg(func=privileged_observation, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    cfg.observations.goal = GoalObservationsCfg()
    cfg.observations.critic = CriticObservationsCfg()

    # -- events -------------------------------------------------------------
    step_dt = cfg.sim.dt * cfg.decimation

    @configclass
    class TaskBEventsCfg:
        reset_robot_root = EventTermCfg(func=reset_robot_on_ring, mode="reset")
        reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        )
        reset_object = EventTermCfg(func=reset_carried_object, mode="reset")
        carry_object = EventTermCfg(
            func=update_carried_object,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            is_global_time=False,
            params={"release_radius": release_radius},
        )

    cfg.events = TaskBEventsCfg()

    # -- rewards ------------------------------------------------------------
    @configclass
    class TaskBRewardsCfg:
        # task
        approach_progress = RewardTermCfg(func=approach_progress, weight=3.0)
        objective_proximity = RewardTermCfg(func=objective_proximity, weight=1.2)
        heading_alignment = RewardTermCfg(func=heading_alignment, weight=1.0)
        heading_error = RewardTermCfg(func=heading_error_penalty, weight=-0.3)
        approach_speed = RewardTermCfg(func=approach_speed_penalty, weight=-4.0)
        reach_bonus = RewardTermCfg(func=reach_bonus, weight=25.0)
        success_bonus = RewardTermCfg(func=success_bonus, weight=60.0)
        failure_penalty = RewardTermCfg(func=failure_penalty, weight=-30.0)
        time_penalty = RewardTermCfg(func=elapsed_step, weight=-0.05)
        # locomotion regularisation, identical in spirit to the Task A proxy
        lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3)
        ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.3)
        action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
        joint_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
        joint_pos_limits = RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
        )
        arm_default_pose = RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
        )

    illegal_contact = deepcopy(cfg.terminations.illegal_contact)

    @configclass
    class TaskBTerminationsCfg:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
        bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.0})
        too_far = TerminationTermCfg(func=too_far_from_bin, params={"max_dist": 8.0})
        delivery_success = TerminationTermCfg(func=delivery_done)
        # Truncation, not failure: the drop is resolved, the episode is over.
        delivery_resolved = TerminationTermCfg(func=delivery_resolved, time_out=True)

    @configclass
    class TaskBCurriculumCfg:
        spawn_difficulty = CurriculumTermCfg(func=delivery_curriculum)

    cfg.rewards = TaskBRewardsCfg()
    cfg.terminations = TaskBTerminationsCfg()
    cfg.terminations.illegal_contact = illegal_contact
    cfg.curriculum = TaskBCurriculumCfg()

    cfg.d1g2_taskb_metadata = {
        "proxy": True,
        "official_task_b": False,
        "objects": 1,
        "official_objects": 18,
        "grasping": "skipped: object kinematically carried on a virtual tray",
        "carried_object_loads_robot": False,
        "release": (f"scripted: dropped when the tray is within {release_radius} m of the bin "
                    f"centre and the robot is still moving forward (>{RELEASE_SPEED} m/s)"),
        "success_test": (f"object settled within {SUCCESS_DISTANCE_FRACTION} x bin radius of the bin "
                         f"centre and below the rim; the official test is the same form at the "
                         f"full 1.0 m radius"),
        "bin_geometry_source": "atec_rl_lab.tasks.task_b.terrain.flat_terrain_with_trash_bin",
        "carry_offset_m": CARRY_OFFSET,
        "spawn_distance_m": (spawn_min, spawn_max),
        "spawn_heading_noise_rad": heading_noise,
        "episode_length_s": episode_length_s,
    }

    cfg.viewer.eye = (4.0, -4.0, 3.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.4)

    # Initial curriculum values are read from the task state at first reset;
    # stash the requested starting point on the cfg for the runner to apply.
    cfg.d1g2_taskb_initial_curriculum = {
        "spawn_min": spawn_min,
        "spawn_max": spawn_max,
        "heading_noise": heading_noise,
        "bin_scale": bin_scale,
        "bin_radius": BIN_RADIUS * bin_scale,
        "success_radius": BIN_RADIUS * bin_scale * SUCCESS_DISTANCE_FRACTION,
        "release_radius": release_radius,
        "ground_z": 0.0,
    }
    # Spawned by the runner once the scene exists: the tile mesh has no floor
    # outside its own 20 m cell and PhysX filters collisions between
    # environments, so the world needs one global ground plane underneath.
    cfg.d1g2_taskb_ground_plane_spawner = ground_plane_spawner
    return cfg
