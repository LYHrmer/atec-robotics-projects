"""Parallel locomotion training for the local D1 + G2 Task A adapter.

This is a training proxy, not the original 300 m Task A environment.  Its
velocity rewards and curriculum levels are not competition scores or proof
of completion.  Final policies must still pass the unmodified full course
created by :func:`d1g2_taska_env.build_d1g2_taska_cfg`.

Importing this module does not start or import Isaac Sim.  Call the builder
after AppLauncher.  The robot, actuator parameters, 23-action order/scales,
physics step, contact rule and 81-value proprioception come from the local
evaluation adapter.  Height measurements are provided only in a separate
privileged critic observation group.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _full_range_difficulty(difficulty):
    """Make the eighth curriculum row reach the full specified difficulty.

    Isaac Lab samples row ``r`` in [r/8, (r+1)/8), so its last row otherwise
    never reaches 1.  This scaling keeps the first seven rows progressive
    and fixes every eighth-row tile at the full Task A obstacle magnitude.
    """
    return min(1.0, max(0.0, float(difficulty)) * 8.0 / 7.0)


def curriculum_slope_terrain(difficulty, cfg):
    """Generate a normal or inverted slope with full difficulty in row 8."""
    from isaaclab.terrains.height_field.hf_terrains import pyramid_sloped_terrain

    return pyramid_sloped_terrain(_full_range_difficulty(difficulty), cfg)


def curriculum_stairs_terrain(difficulty, cfg):
    """Generate normal stairs with full difficulty in row 8."""
    from isaaclab.terrains.trimesh.mesh_terrains import pyramid_stairs_terrain

    return pyramid_stairs_terrain(_full_range_difficulty(difficulty), cfg)


def curriculum_inverted_stairs_terrain(difficulty, cfg):
    """Generate inverted stairs with full difficulty in row 8."""
    from isaaclab.terrains.trimesh.mesh_terrains import inverted_pyramid_stairs_terrain

    return inverted_pyramid_stairs_terrain(_full_range_difficulty(difficulty), cfg)


def curriculum_rough_terrain(difficulty, cfg):
    """Increase roughness; upstream random-uniform terrain ignores difficulty.

    The height range grows from (0.005, 0.020) to (0.020, 0.100) m.
    A 0.005 m step keeps every level valid at the terrain's vertical scale.
    The final training range matches Task A, with finer height sampling than
    the official course's 0.020 m step.  Only this training copy is changed.
    """
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    difficulty = _full_range_difficulty(difficulty)
    rough_cfg = deepcopy(cfg)
    rough_cfg.noise_range = (0.005 + 0.015 * difficulty, 0.020 + 0.080 * difficulty)
    rough_cfg.noise_step = max(0.005, rough_cfg.vertical_scale)
    return random_uniform_terrain(difficulty, rough_cfg)


def original_full_rough_terrain(difficulty, cfg):
    """Use original Task A rough height sampling at EVERY training row.

    This deliberately does not scale roughness with curriculum difficulty:
    noise_range is (0.02, 0.10) m and noise_step is 0.02 m at all levels.
    It matches the original rough height distribution on 8 m training tiles,
    not the original 20 m tile extent or the full-course traversal duration.
    Other terrain types retain their existing progressive curriculum.
    """
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    rough_cfg = deepcopy(cfg)
    rough_cfg.noise_range = (0.02, 0.10)
    rough_cfg.noise_step = 0.02
    return random_uniform_terrain(difficulty, rough_cfg)


def bounded_height_scan(env, sensor_cfg):
    """Return 187 finite base-relative ground heights, clipped to [-1, 1]."""
    import torch
    # The ray origins are 20 m above the base. Remove that artificial sensor
    # offset, rather than saturating every privileged sample at +1.
    heights = env.scene['robot'].data.root_pos_w[:, 2:3] - env.scene[sensor_cfg.name].data.ray_hits_w[..., 2] - 0.5
    return torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def terrain_levels_forward(env, env_ids):
    """Promote forward traversal and demote short episodes at each reset.

    Use the per-environment terrain origin, never Task A's absolute start.
    Sideways displacement cannot advance this forward-only curriculum.
    Initial resets do not update levels before any control steps occurred.
    """
    import torch

    terrain = env.scene.terrain
    robot = env.scene["robot"]
    forward = robot.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    command = env.command_manager.get_command("base_velocity")[env_ids, 0]
    had_episode = env.episode_length_buf[env_ids] > 0
    move_up = (forward > terrain.cfg.terrain_generator.size[0] / 2.0) & had_episode
    move_down = (forward < command * env.max_episode_length_s * 0.5) & ~move_up & had_episode
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def upstairs_feet_air_time(env, sensor_cfg, command_name="base_velocity"):
    """Reward short foot swings at touchdown on the upstairs profile only.

    The profile's ten terrain columns allocate 0--1 to rough ground, 2--3
    to slopes, 4 to descending stairs, and 5--9 to ascending stairs.  The
    selected bodies must be the four feet.  Require forward motion commands
    and at least two feet currently touching the ground; contact times use
    the existing contact sensor's force threshold.  No world height enters
    this reward, and each touchdown contributes at most 0.25 seconds.
    """
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    enabled = (
        (env.scene.terrain.terrain_types >= 5)
        & (env.command_manager.get_command(command_name)[:, 0] > 0.25)
        & (torch.sum(in_contact, dim=1) >= 2)
    )
    touchdown_reward = torch.sum((last_air_time - 0.08).clamp(0.0, 0.25) * first_contact, dim=1)
    return touchdown_reward * enabled


def rough_feet_air_time(env, sensor_cfg, command_name="base_velocity"):
    """Reward bounded foot touchdown air time only on rough columns 0--4.

    Uses the same short-swing mechanism as the upstairs profile, gated by a
    forward command above 0.25 m/s and at least two currently contacting feet.
    Selected bodies must be the four feet. No world height enters this term.
    """
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    terrain_types = env.scene.terrain.terrain_types
    enabled = (
        (terrain_types >= 0)
        & (terrain_types < 5)
        & (env.command_manager.get_command(command_name)[:, 0] > 0.25)
        & (torch.sum(in_contact, dim=1) >= 2)
    )
    touchdown_reward = torch.sum((last_air_time - 0.08).clamp(0.0, 0.25) * first_contact, dim=1)
    return touchdown_reward * enabled


def build_d1g2_taska_train_cfg(
    device: str = "cuda:0",
    seed: int = 42,
    ddt_root: str | Path | None = None,
    num_envs: int = 128,
    profile: str = "mixed",
):
    """Build a camera-free ``BaseRLEnv`` / ``ManagerBasedRLEnv`` config.

    ``obs['proprio']`` is [N, 81], in the evaluation adapter's exact order:
    base linear velocity (3), angular velocity (3), velocity command (3),
    projected gravity (3), relative joint positions (23), joint velocities
    (23), previous actions (23).  ``obs['critic']`` is [N, 187], containing
    only the 17 x 11 height grid.  Concatenate the desired proprioception
    and critic heights in the learning wrapper; do not give heights to a
    proprioception-only actor.  Commands are available through
    ``env.command_manager.get_command('base_velocity')`` as [N, 3].

    The command is forward 0.35--0.85 m/s, lateral zero, and world heading
    zero.  Heading feedback generates angular commands within +/-0.5 rad/s.
    Resets place the robot at its assigned terrain origin plus (0, 0, 0.6)
    and small x/y/yaw perturbations.  Joint reset uses the evaluation default
    pose, avoiding the DDT training configuration's near-zero scale reset.

    ``mixed`` preserves the initial training configuration.  ``upstairs``
    assigns half the terrain columns to stairs ascending from the spawn
    platform and adds a small, contact-gated touchdown reward.  ``rough``
    retains the mixed proportions but uses original_full_rough_all_rows:
    every rough row has Task A's (0.02, 0.10) m noise range and 0.02 m noise
    step. Only slopes/stairs retain progressive difficulty in that profile.
    Switching
    profiles requires the runner to reset the terrain curriculum because
    column meanings change; the builder does not restore checkpoint state.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if profile not in ("mixed", "upstairs", "rough"):
        raise ValueError("profile must be 'mixed', 'upstairs' or 'rough'")

    from isaaclab.envs import mdp
    from isaaclab.managers import CurriculumTermCfg, EventTermCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab.sensors import RayCasterCfg, patterns
    import isaaclab.terrains as terrain_gen
    from isaaclab.utils import configclass

    from tools.d1g2_taska_env import ARM_JOINT_NAMES, LEG_JOINT_NAMES, build_d1g2_taska_cfg

    cfg = build_d1g2_taska_cfg(device=device, cameras=False, seed=seed, ddt_root=ddt_root)
    original_terrain = cfg.scene.terrain
    cfg.scene.num_envs = num_envs
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    cfg.episode_length_s = 20.0
    cfg.scene.terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=seed,
            curriculum=True,
            size=(8.0, 8.0),
            border_width=10.0,
            num_rows=8,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    function=curriculum_rough_terrain,
                    proportion=0.50,
                    noise_range=(0.02, 0.10),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    function=curriculum_slope_terrain,
                    proportion=0.10, slope_range=(0.0, 0.40), platform_width=3.0, border_width=0.25,
                ),
                "slope_inverted": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    function=curriculum_slope_terrain,
                    proportion=0.10, slope_range=(0.0, 0.40), platform_width=3.0, border_width=0.25,
                ),
                "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    function=curriculum_stairs_terrain,
                    proportion=0.15,
                    step_height_range=(0.02, 0.20),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "stairs_inverted": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    function=curriculum_inverted_stairs_terrain,
                    proportion=0.15,
                    step_height_range=(0.02, 0.20),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
            },
        ),
        max_init_terrain_level=1,
        collision_group=original_terrain.collision_group,
        physics_material=deepcopy(original_terrain.physics_material),
        visual_material=deepcopy(original_terrain.visual_material),
        debug_vis=False,
    )
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.scene.lidar_sensor = None
    cfg.observations.extero = None
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        update_period=cfg.sim.dt * cfg.decimation,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        max_distance=30.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    cfg.scene.contact_sensor.update_period = cfg.sim.dt

    @configclass
    class CriticObservationsCfg(ObservationGroupCfg):
        height_scan = ObservationTermCfg(
            func=bounded_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    cfg.observations.critic = CriticObservationsCfg()
    cfg.commands.base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.35, 0.85),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(0.0, 0.0),
        ),
    )
    cfg.events.reset_robot_root = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )
    cfg.events.reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    @configclass
    class TrainingRewardsCfg:
        track_lin_vel_xy_exp = RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        track_ang_vel_z_exp = RewardTermCfg(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3)
        ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.3)
        termination = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
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
    class TrainingTerminationsCfg:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
        bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.0})

    cfg.rewards = TrainingRewardsCfg()
    if profile == "upstairs":
        # With the existing dict order and ten columns, Isaac Lab's
        # col / num_cols + 0.001 cumulative-proportion rule gives:
        # rough 0--1, slope 2, slope_inverted 3, stairs 4,
        # stairs_inverted 5--9.  Inverted stairs ascend from the origin.
        for name, proportion in (
            ("rough", 0.20),
            ("slope", 0.10),
            ("slope_inverted", 0.10),
            ("stairs", 0.10),
            ("stairs_inverted", 0.50),
        ):
            cfg.scene.terrain.terrain_generator.sub_terrains[name].proportion = proportion
        # The original Task A sensor already enables this; keep it explicit
        # for the new reward without changing any contact-force threshold.
        cfg.scene.contact_sensor.track_air_time = True
        cfg.rewards.track_lin_vel_xy_exp.weight = 3.0
        cfg.rewards.action_rate_l2.weight = -0.005
        cfg.rewards.upstairs_feet_air_time = RewardTermCfg(
            func=upstairs_feet_air_time,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(FL|FR|RL|RR)_foot"),
                "command_name": "base_velocity",
            },
        )
    if profile == "rough":
        # Keep mixed proportions: rough columns 0--4, slopes 5--6,
        # descending stairs 7--8 and ascending stairs 9 from tile origins.
        rough_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["rough"]
        rough_cfg.function = original_full_rough_terrain
        rough_cfg.noise_range = (0.02, 0.10)
        rough_cfg.noise_step = 0.02
        cfg.scene.contact_sensor.track_air_time = True
        cfg.rewards.track_lin_vel_xy_exp.weight = 3.0
        cfg.rewards.ang_vel_xy_l2.weight = -0.1
        cfg.rewards.flat_orientation_l2.weight = -0.5
        cfg.rewards.action_rate_l2.weight = -0.005
        cfg.rewards.rough_feet_air_time = RewardTermCfg(
            func=rough_feet_air_time,
            weight=0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(FL|FR|RL|RR)_foot"),
                "command_name": "base_velocity",
            },
        )
        cfg.d1g2_training_profile_metadata = {
            "profile": "rough",
            "rough_terrain_mode": "original_full_rough_all_rows",
            "rough_noise_range_m": (0.02, 0.10),
            "rough_noise_step_m": 0.02,
            "rough_columns": (0, 1, 2, 3, 4),
            "training_tile_size_m": (8.0, 8.0),
            "progressive_terrain_types": ("slope", "slope_inverted", "stairs", "stairs_inverted"),
            "full_course_success": False,
        }
    cfg.terminations = TrainingTerminationsCfg()
    cfg.terminations.illegal_contact = illegal_contact

    @configclass
    class TrainingCurriculumCfg:
        terrain_levels = CurriculumTermCfg(func=terrain_levels_forward)

    cfg.curriculum = TrainingCurriculumCfg()
    cfg.viewer.eye = (7.0, -7.0, 5.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.5)
    return cfg
