"""Local D1 + G2 adapter for the original ATEC Task A course.

Import this module freely, but call ``build_d1g2_taska_cfg`` only after the
Isaac Lab AppLauncher has started.  It does not register a competition robot
or modify the official task configurations.  D1 + G2 is a local evaluation
model; its presence here does not imply eligibility for official submission.

The action interface is 12 leg positions, 4 wheel velocities, and 7 G2
positions.  DDT checkpoints use an interleaved leg/wheel layout and may also
require their training-time low-pass filters in the policy adapter.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import importlib.util
from pathlib import Path
import re
import tempfile


DDT_ROOT = Path("/home/lybm/DDT_Lab")
DDT_ASSET_CONFIG_PATH = DDT_ROOT / "source/ddt_lab/ddt_lab/assets/ddt_robot.py"
D1G2_USD_PATH = DDT_ROOT / "ddt_ros2_control/urdfs/d1_description/usd/d1_with_g2_combined.usda"

LEG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
WHEEL_JOINT_NAMES = [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["g2_joint"]
ACTION_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES + ARM_JOINT_NAMES
ACTION_SCALES = [0.25] * 12 + [5.0] * 4 + [0.1] * 7
START_POSITION = (-141.0, 0.0, 0.60)


def reset_d1g2_root(env, env_ids):
    """Restore the absolute Task A start pose and zero root velocity.

Task A specifies world-coordinate checkpoints and robot start coordinates.
The terrain importer also supplies per-tile origins; adding those origins to
the already absolute start position would place the robot outside the course.
Joint reset remains the original Task A reset event.
"""
    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[env_ids].clone()
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


def _resolve_combined_usd(ddt_root: Path) -> Path:
    """Relocate exported absolute references without changing source assets.

The workspace2 export embeds ``/workspace2/DDT_Lab`` in its arm reference.
A temporary composition layer anchors both the D1 sublayer and the G2
reference in the explicitly selected package.  Its physics opinions are
identical to those in the source combined layer.
"""
    source = ddt_root / "ddt_ros2_control/urdfs/d1_description/usd/d1_with_g2_combined.usda"
    if not source.is_file():
        raise FileNotFoundError(f"Required local D1 + G2 asset is missing: {source}")
    source_text = source.read_text()
    replacements = {}
    relocate = False
    for asset_path in re.findall(r"@([^@]+)@", source_text):
        path = Path(asset_path)
        if path.is_absolute() and "/DDT_Lab/" in asset_path:
            path = ddt_root / asset_path.split("/DDT_Lab/", 1)[1]
        elif not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"USD reference {asset_path!r} cannot be resolved in {ddt_root}: {path}")
        replacements[asset_path] = str(path)
        relocate |= Path(asset_path).is_absolute() and str(path) != asset_path
    if not relocate:
        return source
    composed = re.sub(r"@([^@]+)@", lambda match: f"@{replacements[match.group(1)]}@", source_text)
    output_dir = Path(tempfile.mkdtemp(prefix="d1g2_taska_assets_"))
    output = output_dir / source.name
    output.write_text(composed)
    return output


def _load_ddt_robot_cfg(ddt_root: str | Path | None = None):
    """Read the maintained DDT actuator settings without importing its tasks."""
    root = DDT_ROOT if ddt_root is None else Path(ddt_root).expanduser().resolve()
    asset_config_path = root / "source/ddt_lab/ddt_lab/assets/ddt_robot.py"
    if not asset_config_path.is_file():
        raise FileNotFoundError(f"Required DDT configuration is missing: {asset_config_path}")
    usd_path = _resolve_combined_usd(root)
    spec = importlib.util.spec_from_file_location("_local_d1g2_ddt_asset", asset_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DDT robot configuration: {asset_config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot_cfg = deepcopy(module.DDT_D1_CFG)
    # The DDT source computes its asset directory relative to source/.  Use the
    # existing combined USD at the repository root, retaining all other values.
    robot_cfg.spawn.usd_path = str(usd_path)
    return robot_cfg


def build_d1g2_taska_cfg(
    device: str = "cuda:0", cameras: bool = False, seed: int = 42, ddt_root: str | Path | None = None
):
    """Build a single-robot configuration accepted by ``BaseRLEnv``.

The original terrain (including its fixed seed), checkpoint rewards, illegal
contact threshold, fall threshold, 1,200-second time limit, and x > 145 goal
are retained.  ``seed`` controls environment randomness, not the course seed.
Camera-free evaluation retains lidar and proprioceptive observations.
``ddt_root`` selects the robot configuration and assets from one DDT package;
the default remains ``/home/lybm/DDT_Lab``.
"""
    from isaaclab.managers import EventTermCfg
    from isaaclab.sensors import CameraCfg

    from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg
    from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvCfg
    from atec_rl_lab.tasks.task_base.envs_base_cfg import BaseSceneCfg

    ddt_cfg = _load_ddt_robot_cfg(ddt_root)
    robot_cfg = ATECArticulationCfg(
        **{field.name: deepcopy(getattr(ddt_cfg, field.name)) for field in fields(ddt_cfg) if field.init}
    )
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.init_state = robot_cfg.init_state.replace(
        pos=START_POSITION,
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )
    robot_cfg.base_link_name = "base_link"
    robot_cfg.lidar_sensor_link_name = "F_base_link"
    robot_cfg.head_camera_link_name = "F_base_link"
    robot_cfg.ee_camera_link_name = "play_g2/link6"
    # Virtual observation cameras mounted to existing rigid links.  They add no
    # physical camera mass or collision geometry to the DDT model.
    robot_cfg.head_camera_offset = CameraCfg.OffsetCfg(
        # Match the official B2/Tron1 front-camera downward pitch (30 deg).
        # A horizontal optical axis loses the ground while climbing a slope.
        pos=(0.22, 0.0, 0.10), rot=(0.9659258262890683, 0.0, 0.25881904510252074, 0.0), convention="world"
    )
    robot_cfg.ee_camera_offset = CameraCfg.OffsetCfg(
        pos=(0.06, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
    )
    robot_cfg.joint_names = list(ACTION_JOINT_NAMES)
    robot_cfg.leg_joint_names = list(LEG_JOINT_NAMES)
    robot_cfg.wheel_joint_names = list(WHEEL_JOINT_NAMES)
    robot_cfg.arm_joint_names = list(ARM_JOINT_NAMES)

    cfg = TaskAEnvCfg(scene=BaseSceneCfg(num_envs=1, env_spacing=2.5, robot=robot_cfg))
    # TaskAEnvCfg assigns a module-global terrain object.  Keep modifications
    # made by this local runner isolated from future official environments.
    cfg.scene.terrain = deepcopy(cfg.scene.terrain)
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.sim.device = device
    cfg.seed = seed

    for term in (cfg.observations.proprio.joint_pos, cfg.observations.proprio.joint_vel):
        term.params["asset_cfg"].joint_names = list(ACTION_JOINT_NAMES)
        term.params["asset_cfg"].preserve_order = True

    cfg.actions.joint_leg.joint_names = list(LEG_JOINT_NAMES)
    cfg.actions.joint_leg.scale = 0.25
    cfg.actions.joint_wheel.joint_names = list(WHEEL_JOINT_NAMES)
    cfg.actions.joint_wheel.scale = 5.0
    cfg.actions.joint_arm.joint_names = list(ARM_JOINT_NAMES)
    cfg.actions.joint_arm.scale = 0.1

    # These names are verified in the composed D1 + G2 USD.  Both D1 torso
    # halves, hips, and thighs retain the same illegal-contact rule as the
    # official quadruped Task A configurations (threshold 1 N).
    cfg.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        "base_link", "[FR]_base_link", ".*_hip", ".*_thigh"
    ]
    cfg.events.reset_robot_root = EventTermCfg(func=reset_d1g2_root, mode="reset")

    if not cameras:
        cfg.scene.head_camera = None
        cfg.scene.ee_camera = None
        cfg.scene.ee_dual_camera = None
        cfg.observations.image = None

    cfg.viewer.eye = (-138.0, -6.0, 3.5)
    cfg.viewer.lookat = (-141.0, 0.0, 0.5)
    return cfg
