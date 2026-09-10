"""Run one local D1+G2 Task A episode and preserve terminal-state evidence.

The bundled baseline is a D1 flat-ground policy with G2 held at its default
pose, not a policy trained on the combined robot. No online submission is made.
``--diagnostic_last_stairs`` resets to the original 109→131 m segment and
can never report full-course success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_LAST_STAIRS_START_X = 109.0
DIAGNOSTIC_LAST_STAIRS_EXIT_X = 131.0
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--assets_root', type=Path, default=Path('/home/lybm/DDT_Lab'))
parser.add_argument('--policy', type=Path)
parser.add_argument('--policy_backend', choices=('reference', 'torch'), default='torch')
parser.add_argument('--residual_checkpoint', type=Path)
parser.add_argument('--navigation', choices=('heading', 'odometry_50hz', 'odometry_200hz', 'rgbd'), default='heading')
parser.add_argument('--capture_sensors', action='store_true')
parser.add_argument('--axis_heading', action='store_true',
                    help='Correct visual yaw using observed known Task A grid directions, RGB-D and gravity.')
parser.add_argument('--visual_recovery', action='store_true',
                    help='Allow explicitly bounded proprioceptive bridging to new RGB-D segments after tracking loss.')
parser.add_argument('--uncompressed_sensors', action='store_true',
                    help='Save the same NPZ sensor arrays without CPU compression; uses more disk space.')
parser.add_argument('--camera_pitch_deg', type=float, default=30.0,
                    help='Fixed front camera downward pitch, matching the official 30-degree mount by default.')
parser.add_argument('--mode', choices=('stand', 'policy'), default='policy')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--speed', type=float, default=0.5)
parser.add_argument('--rough_speed', type=float,
                    help='Optional reduced speed over known rough-course bounds, located by RGB-D relative x.')
parser.add_argument('--diagnostic_last_stairs', action='store_true',
                    help='Diagnostic only: reset on the original last stairs at x=109, y=0 and stop at x>=131; never a full-course pass.')
parser.add_argument('--max_steps', type=int, default=60000)
parser.add_argument('--max_wall_seconds', type=float, default=1800)
parser.add_argument('--stall_seconds', type=float, default=30)
parser.add_argument('--settle_steps', type=int, default=50)
parser.add_argument('--video', action='store_true')
parser.add_argument('--trace_interval', type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_steps <= 0 or args.settle_steps < 0 or args.trace_interval <= 0:
    parser.error('Step counts must be positive (settle_steps may be zero).')
if not 0 <= args.speed <= 1 or args.max_wall_seconds <= 0 or args.stall_seconds <= 0:
    parser.error('speed must be in [0,1]; time limits must be positive.')
if not 0 <= args.camera_pitch_deg <= 45:
    parser.error('camera_pitch_deg must be in [0,45].')
if args.rough_speed is not None and (args.navigation != 'rgbd' or not 0 < args.rough_speed <= args.speed):
    parser.error('rough_speed requires RGB-D navigation and must be in (0,speed].')
if args.axis_heading and args.navigation != 'rgbd':
    parser.error('axis_heading requires RGB-D navigation.')
if args.visual_recovery and (args.navigation != 'rgbd' or not args.axis_heading):
    parser.error('visual_recovery requires RGB-D navigation and axis_heading.')
if args.diagnostic_last_stairs and args.rough_speed is not None:
    parser.error('diagnostic_last_stairs cannot use rough_speed: visual relative x starts at the diagnostic entry.')
args.output.mkdir(parents=True, exist_ok=False)
args.enable_cameras = args.video or args.navigation == 'rgbd' or args.capture_sensors
if not args.experience:
    suffix = 'headless.rendering' if args.enable_cameras and args.headless else 'headless' if args.headless else 'rendering'
    args.experience = f'/home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.{suffix}.kit'
started_boot = time.monotonic()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from atec_rl_lab.tasks.task_base.envs_base import BaseRLEnv
from tools.d1g2_taska_env import build_d1g2_taska_cfg, ACTION_JOINT_NAMES, ACTION_SCALES
from tools.d1g2_taska_policy import D1FlatLoadedPolicy


def state_record(env):
    robot = env.scene['robot']
    sensor = env.scene['contact_sensor']
    forces = sensor.data.net_forces_w[0].norm(dim=-1)
    ids = torch.nonzero(forces > 1.0).flatten().tolist()
    history_peak = sensor.data.net_forces_w_history[0].norm(dim=-1).amax(dim=0)
    history_ids = torch.nonzero(history_peak > 1.0).flatten().tolist()
    gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
    quaternion = robot.data.root_quat_w[0].tolist()
    w, x, y, z = quaternion
    return {
        'xyz': robot.data.root_pos_w[0].tolist(),
        'velocity_world': robot.data.root_lin_vel_w[0].tolist(),
        'projected_gravity': gravity.tolist(),
        'quaternion_wxyz': quaternion,
        'diagnostic_world_yaw_radians': math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
        'tilt_degrees': math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1)))),
        'contacts_over_1n': {sensor.body_names[i]: float(forces[i]) for i in ids},
        'contact_history_peak_over_1n': {sensor.body_names[i]: float(history_peak[i]) for i in history_ids},
    }


class RecordedTaskAEnv(BaseRLEnv):
    """Capture terminal physics before Isaac Lab's normal automatic reset."""
    capture_terminal = False
    terminal_record = None

    def attach_inertial_navigator(self, navigator, initial_proprio):
        """Sample the existing body velocity/gyro/gravity fields at physics rate.

        These are the same proprioceptive quantities exposed at control rate;
        no world position or orientation is used by the navigation controller.
        Physics, actions and original task terminations remain unchanged.
        """
        self.navigation_speed = 0.0
        navigator.command(initial_proprio, self.navigation_speed, self.physics_dt)
        original_update = self.scene.update

        def sample_after_physics(dt, *pos, **kw):
            original_update(dt, *pos, **kw)
            robot = self.scene['robot']
            sensor_values = torch.cat((robot.data.root_lin_vel_b[0], robot.data.root_ang_vel_b[0],
                                       robot.data.projected_gravity_b[0])).detach().cpu().numpy()
            proprio = np.zeros(81, dtype=np.float32)
            proprio[:6] = sensor_values[:6]
            proprio[9:12] = sensor_values[6:9]
            navigator.command(proprio, self.navigation_speed, dt)

        self.scene.update = sample_after_physics

    def _reset_idx(self, env_ids):
        if self.capture_terminal and len(env_ids):
            self.terminal_record = state_record(self)
            self.terminal_record['termination_terms'] = [
                name for name in self.termination_manager.active_terms
                if bool(self.termination_manager.get_term(name)[0])
            ]
            self.terminal_record['joint_positions'] = self.scene['robot'].data.joint_pos[0].tolist()
            self.terminal_record['joint_velocities'] = self.scene['robot'].data.joint_vel[0].tolist()
        super()._reset_idx(env_ids)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def hardware_record():
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()[0]
    memory, temperature = [int(v.strip()) for v in values.split(',')]
    available = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    ram = int(available.split()[1]) / 1024**2
    return {'gpu_memory_mib': memory, 'gpu_temperature_c': temperature, 'available_ram_gib': ram}


def main():
    env = None
    writer = None
    save_sensor = np.savez if args.uncompressed_sensors else np.savez_compressed
    restore = lambda: None
    report = {
        'status': 'initializing', 'robot': 'D1+G2', 'mode': args.mode,
        'assets_root': str(args.assets_root.resolve()), 'official_submission': False,
        'seed': args.seed, 'speed_command_mps': args.speed, 'video': args.video,
        'action_joint_names': list(ACTION_JOINT_NAMES), 'action_scales': list(ACTION_SCALES),
        'scope': 'Original Task A terrain, reward, goal and time limit; custom D1+G2 robot; front RGB-D enabled only for visual navigation/capture.',
        'terminal_state_captured_before_auto_reset': True,
        'heading_estimator': 'ZYX yaw rate from body gyro and projected gravity; no world pose input',
        'navigation_mode': args.navigation,
        'axis_heading_enabled': args.axis_heading,
        'visual_recovery_enabled': args.visual_recovery,
        'sensor_archive_compression': 'stored' if args.uncompressed_sensors else 'deflated',
        'speed_profile': {'cruise_mps': args.speed, 'rough_mps': args.rough_speed,
                          'rough_relative_x_bounds_m': [26., 112.],
                          'position_source': 'accepted RGB-D relative x only',
                          'speed_slew_mps2': 0.1},
    }
    if args.diagnostic_last_stairs:
        report.update(
            diagnostic_only=True, full_course_reached_goal=False,
            diagnostic_segment_exit_reached=False,
            scope='Diagnostic traversal of the original Task A last stairs only; original robot physics, camera, controls and terminations retained; not a full-course attempt.',
            diagnostic_case={
                'name': 'original_last_stairs_109_to_131',
                'start_x': DIAGNOSTIC_LAST_STAIRS_START_X, 'start_y': 0.0,
                'exit_x': DIAGNOSTIC_LAST_STAIRS_EXIT_X,
                'original_tile_x_bounds': [110.0, 130.0],
                'reset_height_method': 'Highest of nine footprint samples from the already imported USD terrain plus original 0.6 m root clearance.',
                'world_pose_usage': 'Diagnostic reset placement and local endpoint detection; original scoring/terminations, recorded diagnostics and watchdogs remain in place.',
                'navigation_origin': 'Relative xy starts at [0,0] at the diagnostic entry; world reset coordinates are not passed to navigation or the actor.',
                'axis_prior': 'The original fixed course axes, when --axis_heading is enabled.',
                'settle_steps': args.settle_steps, 'ramp_seconds': 2.0,
            },
        )
    save_json(args.output / 'result.json', report)
    try:
        use_front_camera = args.navigation == 'rgbd' or args.capture_sensors
        cfg = build_d1g2_taska_cfg(device=args.device, cameras=use_front_camera, seed=args.seed, ddt_root=args.assets_root)
        if use_front_camera:
            cfg.scene.ee_camera = None
            cfg.scene.ee_dual_camera = None
            cfg.observations.image = None
            camera_half_pitch = math.radians(args.camera_pitch_deg) / 2
            cfg.scene.head_camera.offset.rot = (math.cos(camera_half_pitch), 0., math.sin(camera_half_pitch), 0.)
            report['front_camera_mount_pitch_degrees'] = args.camera_pitch_deg
        cfg.sim.use_fabric = True
        cfg.sim.render.antialiasing_mode = 'FXAA'
        cfg.sim.render.enable_dl_denoiser = False
        cfg.sim.render.enable_dlssg = False
        cfg.sim.render.samples_per_pixel = 1
        if args.enable_cameras:
            # Both original front RGB-D and optional recorder are sampled at
            # 10 Hz; rendering unused intermediate frames wastes GPU time.
            cfg.sim.render_interval = 20
        if args.enable_cameras:
            from tools.d1g2_taska_camera_compat import prepare_camera_views
            restore = prepare_camera_views()
        if args.video:
            yaw = math.atan2(2.5, 2.65)
            pitch = math.atan2(1.4, math.hypot(2.5, 2.65))
            cy, sy = math.cos(yaw/2), math.sin(yaw/2)
            cp, sp = math.cos(pitch/2), math.sin(pitch/2)
            cfg.scene.overview_camera = CameraCfg(
                prim_path='{ENV_REGEX_NS}/Robot/base_link/OverviewCamera', update_period=0.1,
                height=480, width=640, data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18, clipping_range=(0.05, 1000)),
                offset=CameraCfg.OffsetCfg(pos=(-2.4, -2.5, 1.6),
                    rot=(cy*cp, -sy*sp, cy*sp, sy*cp), convention='world'),
            )
        if args.residual_checkpoint:
            from tools.d1g2_taska_residual import D1G2ResidualPolicy
            policy = D1G2ResidualPolicy(args.residual_checkpoint, args.policy)
        elif args.policy_backend == 'torch':
            from tools.d1g2_taska_torch_policy import D1FlatTorchSinglePolicy
            policy = D1FlatTorchSinglePolicy(args.policy, action_joint_names=ACTION_JOINT_NAMES)
        else:
            policy = D1FlatLoadedPolicy(args.policy, action_joint_names=ACTION_JOINT_NAMES)
        report['policy'] = policy.metadata
        source_paths = [Path(__file__), ROOT / 'tools/d1g2_taska_env.py', ROOT / 'tools/d1g2_taska_policy.py',
                        args.assets_root / 'source/ddt_lab/ddt_lab/assets/ddt_robot.py',
                        Path(cfg.scene.robot.spawn.usd_path)]
        if args.policy_backend == 'torch' or args.residual_checkpoint:
            source_paths.append(ROOT / 'tools/d1g2_taska_torch_policy.py')
        if args.residual_checkpoint:
            source_paths.append(ROOT / 'tools/d1g2_taska_residual.py')
        if args.navigation in ('odometry_50hz', 'odometry_200hz'):
            source_paths.append(ROOT / 'tools/d1g2_taska_navigation.py')
        if args.navigation == 'rgbd':
            source_paths.append(ROOT / 'tools/d1g2_taska_rgbd.py')
            source_paths.append(ROOT / 'tools/d1g2_taska_visual_navigation.py')
            source_paths.append(ROOT / 'tools/d1g2_taska_navigation.py')
        if args.axis_heading:
            source_paths.append(ROOT / 'tools/d1g2_taska_axis_heading.py')
        if args.visual_recovery:
            source_paths.append(ROOT / 'tools/d1g2_taska_visual_recovery.py')
        if args.enable_cameras:
            source_paths.append(ROOT / 'tools/d1g2_taska_camera_compat.py')
        if args.diagnostic_last_stairs:
            source_paths.append(ROOT / 'tools/probe_d1g2_taska_segments.py')
        (args.output / 'sources').mkdir()
        report['source_sha256'] = {}
        for i, path in enumerate(source_paths):
            report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copy2(path, args.output / 'sources' / f'{i}_{path.name}')
        print('D1G2_STAGE create_environment', flush=True)
        env = RecordedTaskAEnv(cfg=cfg)
        if args.diagnostic_last_stairs:
            from tools.probe_d1g2_taska_segments import read_terrain_mesh_from_usd, sample_spawn_height
            terrain_mesh = read_terrain_mesh_from_usd(env.scene.stage, cfg.scene.terrain.prim_path + '/terrain')
            spawn_z, footprint_samples = sample_spawn_height(terrain_mesh, DIAGNOSTIC_LAST_STAIRS_START_X, 0.0)
            diagnostic_spawn_xyz = [DIAGNOSTIC_LAST_STAIRS_START_X, 0.0, spawn_z]
            # Only the reset translation changes. Default yaw, joint state and
            # zero root velocity remain those of the original D1+G2 config.
            env.scene['robot'].data.default_root_state[0, :3] = torch.tensor(diagnostic_spawn_xyz, device=env.device)
            report['diagnostic_case'].update(
                root_spawn_xyz=diagnostic_spawn_xyz, reset_footprint_samples=footprint_samples,
                reset_height_mesh_source=dict(terrain_mesh.metadata),
            )
            del terrain_mesh
        obs, _ = env.reset(seed=args.seed)
        initial = state_record(env)
        expected_start_x = DIAGNOSTIC_LAST_STAIRS_START_X if args.diagnostic_last_stairs else -141.0
        if abs(initial['xyz'][0] - expected_start_x) > 0.1:
            raise RuntimeError(f'Incorrect Task A world start: {initial["xyz"]}')
        if args.diagnostic_last_stairs and not np.allclose(initial['xyz'], diagnostic_spawn_xyz, atol=.01):
            raise RuntimeError(f'Incorrect diagnostic reset placement: {initial["xyz"]}')
        if tuple(env.action_space.shape) != (1, 23):
            raise RuntimeError(f'Unexpected action shape: {env.action_space.shape}')
        if tuple(obs['proprio'].shape) != (1, 81):
            raise RuntimeError(f'Unexpected proprio shape: {obs["proprio"].shape}')
        report.update(initial=initial, startup_seconds=time.monotonic()-started_boot,
                      actual_joint_names=env.scene['robot'].joint_names,
                      actual_body_names=env.scene['robot'].body_names,
                      simulation_dt=env.step_dt, episode_limit_seconds=env.max_episode_length_s,
                      num_envs=env.num_envs, status='running')
        navigator = None
        if args.navigation in ('odometry_50hz', 'odometry_200hz'):
            from tools.d1g2_taska_navigation import D1G2ProprioNavigator
            navigator = D1G2ProprioNavigator()
            report['navigation'] = dict(navigator.metadata, sample_hz=200 if args.navigation == 'odometry_200hz' else 50)
            if args.navigation == 'odometry_200hz':
                env.attach_inertial_navigator(navigator, obs['proprio'][0].detach().cpu().numpy())
        rgbd = None
        rgbd_result = None
        visual_navigator = None
        startup_navigator = None
        visual_started = False
        axis_result = None
        estimate_axis_heading = None
        recovery_status = {}
        if args.axis_heading:
            from tools.d1g2_taska_axis_heading import estimate as estimate_axis_heading
            report['visual_yaw_source'] = 'RGB-D PnP plus gated observed known-course grid directions'
        if use_front_camera:
            camera = env.scene['head_camera']
            # Warm up after env.reset at the actual start pose; no physics steps.
            for _ in range(3):
                env.sim.render()
                camera._is_outdated[:] = True
                camera.update(0.0, force_recompute=True)
            camera_k_raw = camera.data.intrinsic_matrices[0].cpu().numpy().copy()
            camera_k = camera_k_raw.copy()
            # Isaac's raster principal point addresses pixel corners; OpenCV
            # feature coordinates address pixel centres (first centre = 0).
            report['front_camera_intrinsics_raster'] = camera_k.tolist()
            camera_k[:2, 2] -= 0.5
            report['front_camera_intrinsics'] = camera_k.tolist()
            report['intrinsics_pixel_convention'] = 'opencv_pixel_centres'
            report['visual_tilt_source'] = 'projected_gravity from original proprioception'
            if args.capture_sensors:
                (args.output/'sensors').mkdir()
            if args.navigation == 'rgbd':
                import cv2
                cv2.setNumThreads(1)
                report['opencv_threads'] = cv2.getNumThreads()
                from tools.d1g2_taska_rgbd import RGBDOdometry
                from tools.d1g2_taska_visual_navigation import VisualNavigator
                from tools.d1g2_taska_navigation import D1G2ProprioNavigator
                if args.visual_recovery:
                    from tools.d1g2_taska_visual_recovery import RecoveringRGBDOdometry
                    rgbd = RecoveringRGBDOdometry(mount_pitch_deg=args.camera_pitch_deg)
                    recovery_status = rgbd.observe_proprio(obs['proprio'][0].cpu().numpy(), 0.)
                    report['visual_recovery'] = dict(
                        budgets=recovery_status['recovery_budgets'],
                        source='Original body velocity, gyro and gravity bridge disconnected RGB-D segments.',
                        candidate_required_consecutive_visual_accepts=2,
                        uses_world_pose=False,
                        inherited_uncertainty='Unquantified short inertial drift; explicitly recorded per reanchor.')
                    report['speed_profile']['position_source'] = 'RGB-D relative x, including explicitly recorded bounded inertial reanchors'
                else:
                    rgbd = RGBDOdometry(mount_pitch_deg=args.camera_pitch_deg)
                visual_navigator = VisualNavigator()
                startup_navigator = D1G2ProprioNavigator()
                startup_navigator.command(obs['proprio'][0].cpu().numpy(), 0., env.step_dt)
                report['navigation'] = dict(visual_navigator.metadata,
                    startup='At least 2 seconds of proprioceptive relative-pose estimation during settling; then RGB-D anchors.',
                    startup_world_pose_input=False)
            def read_front_camera():
                # env.step has just rendered this control frame. SensorBase's
                # float32 period accumulator can otherwise miss a 0.1 s refresh
                # and pair a cached image with newer proprioception.
                camera._is_outdated[:] = True
                camera.update(0.0, force_recompute=True)
                return (camera.data.output['rgb'][0, :, :, :3].cpu().numpy(),
                        camera.data.output['depth'][0].cpu().numpy().reshape(480, 640))
            rgb, depth = read_front_camera()
            if args.capture_sensors:
                save_sensor(args.output/'sensors'/'frame_000000.npz', rgb=rgb, depth=depth, K=camera_k,
                                    K_raw=camera_k_raw, principal_point_correction_px=-0.5,
                                    camera_pitch_deg=args.camera_pitch_deg,
                                    intrinsics_pixel_convention='opencv_pixel_centres',
                                    proprio=obs['proprio'][0].cpu().numpy(),
                                    diagnostic_true_xyz=initial['xyz'], diagnostic_true_quat=initial['quaternion_wxyz'])
        save_json(args.output/'result.json', report)
        print('D1G2_READY ' + json.dumps({'xyz': initial['xyz'], 'actions': 23,
              'proprio': 81, 'policy_kind': policy.metadata['kind']}), flush=True)
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output/'run.mp4'), fps=10, codec='libx264', quality=7)
        env.capture_terminal = True
        policy.reset()
        start = time.monotonic()
        max_x = initial['xyz'][0]
        checkpoint_x = max_x
        last_progress_step = args.settle_steps
        score = 0.0
        aggregate_reward = 0.0
        heading_estimate = 0.0
        previous_yaw_rate = None
        reason = 'max_steps'
        last_state = initial
        steps = 0
        max_tilt = initial['tilt_degrees']
        sampled_gpu_peak = 0
        route_speed = args.speed
        trace = (args.output/'trace.jsonl').open('w')
        try:
            with torch.inference_mode():
                for step in range(args.max_steps):
                    if time.monotonic()-start > args.max_wall_seconds:
                        reason = 'wall_time_budget'
                        break
                    proprio = obs['proprio'][0].detach().cpu().numpy()
                    if not np.isfinite(proprio).all():
                        reason = 'nonfinite_observation'
                        break
                    # World-z angular velocity includes roll/pitch coupling.
                    # ZYX yaw rate removes it, avoiding heading drift on rough ground.
                    gy, gz = proprio[10:12]
                    yaw_rate = float((-gy*proprio[4] - gz*proprio[5]) / max(float(gy*gy+gz*gz), 0.05))
                    if previous_yaw_rate is not None:
                        heading_estimate += 0.5 * (previous_yaw_rate + yaw_rate) * env.step_dt
                    previous_yaw_rate = yaw_rate
                    heading_estimate = math.atan2(math.sin(heading_estimate), math.cos(heading_estimate))
                    ramp = np.clip((step-args.settle_steps)*env.step_dt/2.0, 0, 1)
                    target_speed = args.speed
                    if args.rough_speed is not None and visual_started:
                        # Known Task A rough section is 31--111 m from the
                        # start. Slow before entry and resume just beyond it;
                        # locate this interval using visual estimates only.
                        if 26. <= visual_navigator.estimated_xy[0] <= 112.:
                            target_speed = args.rough_speed
                    route_speed += float(np.clip(target_speed-route_speed, -.1*env.step_dt, .1*env.step_dt))
                    requested_speed = route_speed * ramp
                    command = np.asarray([requested_speed, 0.0, np.clip(-heading_estimate, -.5, .5)], np.float32)
                    if args.navigation == 'odometry_200hz':
                        env.navigation_speed = requested_speed
                        command = navigator.command_from_estimate(env.navigation_speed)
                    elif args.navigation == 'odometry_50hz':
                        command = navigator.command(proprio, requested_speed, env.step_dt)
                    elif args.navigation == 'rgbd':
                        if visual_started:
                            command = visual_navigator.command(proprio, requested_speed, env.step_dt)
                        elif step * env.step_dt > 5.0:
                            command = np.zeros(3, np.float32)
                    if args.visual_recovery and recovery_status.get('recovery_stop_required'):
                        command = np.zeros(3, np.float32)
                    if args.mode == 'stand' or step < args.settle_steps:
                        action_np = np.zeros(23, np.float32)
                    else:
                        action_np = policy.action_from_state(proprio[3:6], proprio[9:12],
                                   proprio[12:35], proprio[35:58], command, dt=env.step_dt)
                    obs, reward, terminated, truncated, _ = env.step(torch.as_tensor(action_np[None], device=env.device))
                    steps = step+1
                    aggregate_reward += float(reward[0])
                    score += float(reward[0]) / env.step_dt
                    done = bool(terminated[0]) or bool(truncated[0])
                    last_state = env.terminal_record if done else state_record(env)
                    diagnostic_exit = args.diagnostic_last_stairs and last_state['xyz'][0] >= DIAGNOSTIC_LAST_STAIRS_EXIT_X
                    if args.visual_recovery and not done:
                        recovery_status = rgbd.observe_proprio(obs['proprio'][0].cpu().numpy(), steps*env.step_dt)
                    if startup_navigator is not None and not visual_started and not done:
                        startup_navigator.command(obs['proprio'][0].cpu().numpy(), 0., env.step_dt)
                    if use_front_camera and steps % 5 == 0 and not done:
                        rgb, depth = read_front_camera()
                        if rgbd is not None:
                            camera_proprio = obs['proprio'][0].cpu().numpy()
                            if estimate_axis_heading is not None:
                                prior_yaw = (visual_navigator.estimated_yaw if visual_started
                                             else startup_navigator.estimated_yaw)
                                axis_result = estimate_axis_heading(
                                    rgb, depth, camera_k, camera_proprio[9:12],
                                    mount_pitch_deg=args.camera_pitch_deg, predicted_yaw=prior_yaw)
                            stable_start = (steps * env.step_dt >= 2.0 and
                                            np.linalg.norm(camera_proprio[9:11]) < .15 and
                                            np.linalg.norm(camera_proprio[3:6]) < .5)
                            if not visual_started and stable_start:
                                rgbd_result = rgbd.initialize_from_proprio(
                                    rgb, depth, camera_k, camera_proprio,
                                    startup_navigator.estimated_xy,
                                    axis_result['heading_rad'] if axis_result and axis_result['accepted']
                                    else startup_navigator.estimated_yaw,
                                    **({'timestamp': steps*env.step_dt} if args.visual_recovery else {}))
                                if rgbd_result['reason'] == 'initialized':
                                    visual_started = True
                                    visual_navigator.observe_visual(dict(rgbd_result, accepted=True,
                                                                        reason='proprio_startup_anchor'))
                                    report['visual_startup'] = dict(step=steps,
                                        estimated_xy=startup_navigator.estimated_xy.tolist(),
                                        estimated_yaw=startup_navigator.estimated_yaw,
                                        initial_rotation=rgbd_result['initial_rotation'], uses_world_pose=False)
                                    report['visual_startup']['axis_heading'] = axis_result
                            elif visual_started:
                                rgbd_result = rgbd.update(rgb, depth, camera_k, 0.1,
                                                         projected_gravity=camera_proprio[9:12],
                                                         axis_heading_result=axis_result,
                                                         **({'timestamp': steps*env.step_dt} if args.visual_recovery else {}))
                                visual_navigator.observe_visual(rgbd_result)
                                if args.visual_recovery and not rgbd_result['accepted'] and rgbd_result.get('heading_axis_applied'):
                                    visual_navigator.observe_heading(dict(accepted=True, heading_rad=rgbd_result['heading_yaw']))
                            if args.visual_recovery:
                                recovery_status = rgbd.state_dict()
                        if args.capture_sensors:
                            save_sensor(args.output/'sensors'/f'frame_{steps:06d}.npz',
                                                rgb=rgb, depth=depth, K=camera_k,
                                                K_raw=camera_k_raw, principal_point_correction_px=-0.5,
                                                camera_pitch_deg=args.camera_pitch_deg,
                                                intrinsics_pixel_convention='opencv_pixel_centres',
                                                proprio=obs['proprio'][0].cpu().numpy(),
                                                diagnostic_true_xyz=last_state['xyz'], diagnostic_true_quat=last_state['quaternion_wxyz'])
                    max_x = max(max_x, last_state['xyz'][0])
                    max_tilt = max(max_tilt, last_state['tilt_degrees'])
                    if max_x >= checkpoint_x + .1:
                        checkpoint_x = max_x
                        last_progress_step = step
                    row = dict(step=steps, sim_seconds=steps*env.step_dt,
                               wall_seconds=time.monotonic()-start, max_x=max_x,
                               raw_progress_score=score, command=command.tolist(),
                               route_speed_mps=route_speed,
                               axis_heading=axis_result,
                               command_heading_estimate_rad=heading_estimate,
                               yaw_rate_estimate=yaw_rate,
                               action_max_abs=float(np.abs(action_np).max()), **last_state)
                    if navigator is not None:
                        row['navigation'] = navigator.state_dict()
                    if rgbd is not None:
                        row['navigation'] = visual_navigator.state_dict()
                        row['visual_odometry'] = rgbd.state_dict()
                        row['visual_tracking'] = rgbd_result
                    if step % args.trace_interval == 0 or done or diagnostic_exit:
                        trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                    if steps % 250 == 0 or done:
                        hardware = hardware_record()
                        sampled_gpu_peak = max(sampled_gpu_peak, hardware['gpu_memory_mib'])
                        row['hardware'] = hardware
                        print('D1G2_PROGRESS '+json.dumps(row, allow_nan=False), flush=True)
                        save_json(args.output/'progress.json', row)
                        if ((hardware['gpu_memory_mib'] > 7600 or hardware['gpu_temperature_c'] >= 85
                             or hardware['available_ram_gib'] < 2)
                                and not (args.diagnostic_last_stairs and done)):
                            reason = 'hardware_guard'
                            break
                    if args.video and step % 5 == 0 and not done:
                        from PIL import Image, ImageDraw
                        frame = env.scene['overview_camera'].data.output['rgb'][0, :, :, :3].cpu().numpy()
                        image = Image.fromarray(frame.astype(np.uint8))
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, 640, 40), fill='black')
                        policy_label = 'trained residual' if args.residual_checkpoint else 'flat-policy baseline'
                        run_label = 'D1+G2 diagnostic last stairs' if args.diagnostic_last_stairs else 'D1+G2'
                        target_label = 'diagnostic exit x>=131m' if args.diagnostic_last_stairs else 'target x>145m'
                        draw.text((8, 5), f'{run_label} | {policy_label} | t={steps*env.step_dt:.1f}s', fill='white')
                        draw.text((8, 22), f'x={last_state["xyz"][0]:.2f}m | progress score={score:.3f} | {target_label}', fill='white')
                        writer.append_data(np.asarray(image))
                        if step % 250 == 0:
                            image.save(args.output/f'frame_{steps:06d}.png')
                    if done:
                        terms = last_state['termination_terms']
                        reason = '+'.join(terms) or 'environment_done'
                        break
                    if diagnostic_exit:
                        reason = 'diagnostic_segment_exit'
                        break
                    if args.mode == 'policy' and (step-last_progress_step)*env.step_dt >= args.stall_seconds:
                        reason = 'stalled_no_0.1m_forward_progress'
                        break
        finally:
            trace.close()
        report.update(status='finished', reason=reason, steps=steps,
                      simulation_seconds=steps*env.step_dt, wall_seconds=time.monotonic()-start,
                      final=last_state, max_x=max_x, maximum_forward_progress_m=max_x-initial['xyz'][0],
                      net_forward_progress_m=last_state['xyz'][0]-initial['xyz'][0],
                      raw_progress_score=score, aggregate_reward=aggregate_reward,
                      max_tilt_degrees=max_tilt,
                      sampled_gpu_peak_mib=sampled_gpu_peak,
                      goal_triggered='reach_goal_x' in last_state.get('termination_terms', []),
                      reached_goal=not args.diagnostic_last_stairs and reason == 'reach_goal_x' and set(last_state.get('termination_terms', [])) == {'reach_goal_x'},
                      max_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if args.diagnostic_last_stairs:
            report.update(diagnostic_segment_exit_reached=reason == 'diagnostic_segment_exit',
                          full_course_reached_goal=False)
        print('D1G2_RESULT '+json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='error', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save_json(args.output/'result.json', report)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        restore()
    return 0 if report['status'] == 'finished' else 1


if __name__ == '__main__':
    exit_code = main()
    app.close()
    sys.exit(exit_code)
