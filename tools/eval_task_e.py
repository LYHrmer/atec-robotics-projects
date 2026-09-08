"""Evaluate an observation-only Task E solution without changing task physics or scores.

Run with PYTHONNOUSERSITE=1 in the local IsaacLab environment. Diagnostic scene
state is saved by this evaluator only; it is never passed into the policy.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solution", required=True, type=Path)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=6000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--snapshot_interval", type=int, default=200)
parser.add_argument("--post_terminal_release_steps", type=int, default=0,
                    help="Optional separate release audit after terminal 18/18: hold the arm, "
                         "open the jaws and record this many extra diagnostic steps (150 = 3 s).")
parser.add_argument("--video", type=Path, help="Save a continuous two-camera MP4 with score and state HUD.")
parser.add_argument("--render_preset", choices=("default", "crisp"), default="crisp")
parser.add_argument("--use_fabric", action="store_true", default=True,
                    help="Use native Fabric pose publication (enabled by default).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.post_terminal_release_steps < 0:
    parser.error("--post_terminal_release_steps must be nonnegative")
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import carb
from PIL import Image
import atec_rl_lab.tasks
from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec
from tools.task_e.fabric_compat import prepare_legacy_fabric_camera_views


def load_solution():
    path = args.solution.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("task_e_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AlgSolution()


def snapshot_sources():
    """Save the evaluated source before stepping, including local dependencies."""
    paths = {Path(__file__).resolve(), args.solution.resolve()}
    solution_root = args.solution.resolve().parent
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (path.is_relative_to(ROOT) or path.is_relative_to(solution_root)):
                paths.add(path)
    manifest = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        data = path.read_bytes()
        relative = (path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("external_solution") / path.relative_to(solution_root))
        target = args.output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(),
                               "snapshot": str(target.relative_to(args.output))}
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def post_terminal_release_audit(env, output, requested_steps, official_result, is_running):
    """Run bounded diagnostics after the official files/recording are finalized.

    No policy is called and no returned reward is accumulated. Task E has no
    pose-reset events, so its post-terminal manager resets leave physical poses
    intact. Only action targets change; config, physics and scoring do not.
    """
    task = env.unwrapped
    robot = task.scene["robot"]
    names = ("object_1", "object_2", "object_3")
    params = task.cfg.terminations.basket_success.params
    center = np.asarray(params.get("center", (1.08, -.30, .74)), dtype=float)[:2]
    half_x, half_y = float(params.get("half_x", .20)), float(params.get("half_y", .11))
    min_z = float(params.get("table_top_z", .8266))
    # ObjectsInBasketDone uses the configured tabletop plus a fixed 0.15 m.
    max_z = min_z + .15
    dt = float(task.step_dt)
    action_cfg = task.cfg.actions.joint_arm
    scale = float(action_cfg.scale)
    if (not action_cfg.use_default_offset or not np.isfinite(scale) or scale == 0
            or robot.data.joint_pos.shape != (1, 8)):
        raise ValueError("Release audit requires Task E's 8-joint default-offset position action")
    target = robot.data.joint_pos[0].detach().clone()
    frozen_arm = target[:6].clone()
    default = robot.data.default_joint_pos[0].detach().clone()
    open_jaws = torch.as_tensor([.035, -.035], device=target.device, dtype=target.dtype)
    origin = task.scene.env_origins[0].detach().cpu().numpy().copy()
    records = {key: [] for key in ("qpos", "target_qpos", "objects_xyz_w", "objects_xyz_local",
                                    "jaw_width", "inside_basket")}

    def sample():
        q = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        positions = np.asarray([task.scene[name].data.root_pos_w[0].detach().cpu().numpy().copy()
                                for name in names])
        local = positions - origin
        inside = ((np.abs(local[:, 0] - center[0]) <= half_x)
                  & (np.abs(local[:, 1] - center[1]) <= half_y)
                  & (local[:, 2] >= min_z) & (local[:, 2] <= max_z))
        records["qpos"].append(q)
        records["target_qpos"].append(target.detach().cpu().numpy().copy())
        records["objects_xyz_w"].append(positions)
        records["objects_xyz_local"].append(local)
        records["jaw_width"].append(float(q[6] - q[7]))
        records["inside_basket"].append(inside)

    sample()  # index zero is the official terminal physical state
    executed, stop_reason, last_obs = 0, "requested_steps_complete", None
    for _ in range(requested_steps):
        if not is_running():
            stop_reason = "app_stopped"
            break
        target[:6] = frozen_arm
        target[6:] += torch.clamp(open_jaws - target[6:], -.002, .002)
        action = ((target - default) / scale).unsqueeze(0)
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Non-finite post-terminal diagnostic action")
        with torch.inference_mode():
            last_obs, _, _, _, _ = env.step(action)
        executed += 1
        sample()

    arrays = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(output / "post_terminal_release.npz", dt=dt,
                        audit_step=np.arange(executed + 1),
                        audit_seconds=np.arange(executed + 1) * dt,
                        object_names=np.asarray(names), **arrays)
    window_steps = int(np.ceil(1. / dt))
    enough = executed >= window_steps
    continuous = (np.all(arrays["inside_basket"][-(window_steps + 1):], axis=0)
                  if enough else None)
    report = {
        "diagnostic": "POST-TERMINAL RELEASE AUDIT",
        "not_part_of_official_episode": True,
        "official_result_file": "result.json",
        "official_score": official_result["score"],
        "official_steps": official_result["steps"],
        "official_sim_seconds": official_result["sim_seconds"],
        "requested_steps": requested_steps, "executed_steps": executed,
        "diagnostic_sim_seconds": executed * dt, "stop_reason": stop_reason,
        "fixed_arm_target_rad": frozen_arm.detach().cpu().numpy().tolist(),
        "finger_target_delta_limit_m_per_step": .002,
        "finger_open_target_m": [.035, -.035], "original_action_scale": scale,
        "basket_thresholds_local": {"center_xy": center.tolist(), "half_x": half_x,
                                    "half_y": half_y, "min_z": min_z, "max_z": max_z},
        "initial_objects_xyz_w": dict(zip(names, arrays["objects_xyz_w"][0].tolist())),
        "final_objects_xyz_w": dict(zip(names, arrays["objects_xyz_w"][-1].tolist())),
        "initial_jaw_width_m": float(arrays["jaw_width"][0]),
        "final_jaw_width_m": float(arrays["jaw_width"][-1]),
        "final_jaws_open": bool(arrays["jaw_width"][-1] >= .069),
        "final_membership": dict(zip(names, arrays["inside_basket"][-1].tolist())),
        "final_all_inside": bool(np.all(arrays["inside_basket"][-1])),
        "continuous_last_1s_membership": None if continuous is None else dict(zip(names, continuous.tolist())),
        "continuous_last_1s_all_inside": None if continuous is None else bool(np.all(continuous)),
        "continuous_window_seconds": window_steps * dt if enough else None,
        "maximum_arm_drift_rad": float(np.max(np.abs(arrays["qpos"][:, :6]
                                                       - frozen_arm.detach().cpu().numpy()))),
        "telemetry": "post_terminal_release.npz",
        "interpretation": "Membership uses official root-position bounds. Open jaws and continuous "
                          "membership do not independently prove zero velocity or collision-free rest.",
    }
    (output / "post_terminal_release.json").write_text(json.dumps(report, indent=2) + "\n")
    if last_obs is not None:
        try:
            rgb = last_obs["image"]["video_rgb"][0].detach().cpu().numpy()
            Image.fromarray(rgb[..., :3].astype(np.uint8)).save(output / "post_terminal_release_final.png")
        except (KeyError, TypeError, ValueError) as error:
            print("TASK_E_RELEASE_SNAPSHOT_SKIPPED " + str(error), flush=True)
    print("TASK_E_POST_TERMINAL_RELEASE " + json.dumps(report), flush=True)
    return report


def main():
    cfg = TaskEEnvPiperCfg(seed=args.seed)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.use_fabric = args.use_fabric
    if args.render_preset == "crisp":
        cfg.sim.render.antialiasing_mode = "FXAA"
        cfg.sim.render.enable_dl_denoiser = False
        cfg.sim.render.enable_dlssg = False
        cfg.sim.render.samples_per_pixel = 8
    solution = load_solution()
    source_manifest = snapshot_sources()
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    cfg = apply_safe_action_spec(cfg, json.dumps(action_spec) if action_spec else None)
    restore_camera_views = prepare_legacy_fabric_camera_views()
    env = gym.make("ATEC-TaskE-Piper", cfg=cfg)
    score, steps, reason = 0.0, 0, "max_steps"
    telemetry = {key: [] for key in ("qpos", "qvel", "action", "score", "state", "object_id")}
    recorder = None
    if args.video:
        from tools.task_e.video_recorder import TaskEVideoRecorder
        recorder = TaskEVideoRecorder(args.video, step_dt=env.unwrapped.step_dt, seed=args.seed)
    started = time.monotonic()
    trace = (args.output / "trace.jsonl").open("w")
    try:
        obs, _ = env.reset(seed=args.seed)
        initial = {name: env.unwrapped.scene[name].data.root_pos_w[0].tolist()
                   for name in ("object_1", "object_2", "object_3")}
        print("TASK_E_INITIAL " + json.dumps(initial), flush=True)
        robot = env.unwrapped.scene["robot"]
        ee_id = robot.find_bodies("gripper_base")[0][0]
        camera = env.unwrapped.scene["video_cam"]
        meta = {"joint_names": robot.data.joint_names,
                "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
                "camera_intrinsics": camera.data.intrinsic_matrices[0].tolist(),
                "camera_pos": camera.data.pos_w[0].tolist(),
                "camera_quat_ros": camera.data.quat_w_ros[0].tolist()}
        (args.output / "environment_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        for step in range(args.max_steps):
            if not app.is_running():
                reason = "app_stopped"
                break
            if recorder and step % recorder.stride == 0:
                recorder.append(obs, step=step, score=score, state=getattr(solution, "state", ""),
                                object_id=getattr(solution, "current_object", None),
                                completed=getattr(solution, "completed", ()))
            if step % args.snapshot_interval == 0:
                rgb = obs["image"]["video_rgb"][0].detach().cpu().numpy()
                Image.fromarray(rgb[..., :3].astype(np.uint8)).save(args.output / f"rgb_{step:05d}.png")
                np.savez_compressed(args.output / f"observation_{step:05d}.npz", **{
                    key: value[0].detach().cpu().numpy() for key, value in obs["image"].items()},
                    proprio=obs["proprio"][0].detach().cpu().numpy())
            with torch.inference_mode():
                response = solution.predicts(obs, score)
                if response["giveup"]:
                    reason = "policy_giveup"
                    break
                action = torch.as_tensor(response["action"], device=env.unwrapped.device, dtype=torch.float32)
                if action.shape == (8,):
                    action = action.unsqueeze(0)
                if action.shape != (1, 8) or not bool(torch.isfinite(action).all()):
                    raise ValueError(f"Expected finite action of shape (1, 8), got {action.shape}")
                obs, reward, terminated, truncated, info = env.step(action)
            steps = step + 1
            increment = float(reward.item()) / float(env.unwrapped.step_dt)
            score += increment
            row = {"step": steps, "score": score, "reward": increment,
                   "qpos": env.unwrapped.scene["robot"].data.joint_pos[0].tolist(),
                   "ee_pos": robot.data.body_pos_w[0, ee_id].tolist(),
                   "ee_quat": robot.data.body_quat_w[0, ee_id].tolist(),
                   "objects": {name: env.unwrapped.scene[name].data.root_pos_w[0].tolist()
                               for name in initial},
            "state": getattr(solution, "state", None)}
            telemetry["qpos"].append(row["qpos"])
            telemetry["qvel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
            telemetry["action"].append(action[0].detach().cpu().numpy().copy())
            telemetry["score"].append(score)
            telemetry["state"].append(row["state"] or "")
            telemetry["object_id"].append(getattr(solution, "current_object", None) or 0)
            if step % 20 == 0 or increment or bool(terminated.item()) or bool(truncated.item()):
                trace.write(json.dumps(row) + "\n")
                trace.flush()
            if step % 100 == 0 or increment:
                print("TASK_E_PROGRESS " + json.dumps(row), flush=True)
            if bool(terminated.item()) or bool(truncated.item()):
                reason = "terminated" if bool(terminated.item()) else "time_limit"
                break
        if recorder:
            recorder.append(obs, step=steps, score=score, state=getattr(solution, "state", ""),
                            object_id=getattr(solution, "current_object", None),
                            completed=getattr(solution, "completed", ()), final=True)
            recorder.close()
        np.savez_compressed(args.output / "telemetry.npz", dt=env.unwrapped.step_dt,
                            **{key: np.asarray(value) for key, value in telemetry.items()})
        result = {"solution": str(args.solution.resolve()), "seed": args.seed,
                  "initial_objects": initial, "score": round(score, 5), "full_score": score >= 17.999,
                  "steps": steps, "sim_seconds": steps * env.unwrapped.step_dt,
                  "wall_seconds": time.monotonic() - started, "stop_reason": reason,
                  "action_spec": action_spec, "task_physics_modified": False}
        result["render_preset"] = args.render_preset
        result["usd_pose_updates"] = bool(carb.settings.get_settings().get("/physics/updateToUsd"))
        result["use_fabric"] = args.use_fabric
        result["solution_sha256"] = source_manifest[str(args.solution.resolve())]["sha256"]
        result["source_manifest"] = "source_manifest.json"
        result["telemetry"] = "telemetry.npz"
        if recorder:
            result["video"] = {"path": str(recorder.path), "fps": recorder.fps,
                               "frames": recorder.frames, "simulation_speed": "1x",
                               "ending_hold_seconds": 2,
                               "sha256": hashlib.sha256(recorder.path.read_bytes()).hexdigest()}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("TASK_E_RESULT " + json.dumps(result), flush=True)
        if args.post_terminal_release_steps and reason == "terminated" and result["full_score"]:
            # Official score, timing, telemetry and video are already immutable
            # files here. A diagnostic error must not overwrite official success.
            try:
                post_terminal_release_audit(env, args.output, args.post_terminal_release_steps,
                                            result, app.is_running)
            except Exception:
                diagnostic_error = traceback.format_exc()
                print(diagnostic_error, file=sys.stderr, flush=True)
                (args.output / "post_terminal_release_failure.txt").write_text(diagnostic_error)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        (args.output / "failure.txt").write_text(error)
        raise
    finally:
        trace.close()
        if recorder:
            recorder.close()
        env.close()
        restore_camera_views()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        (args.output / "failure.txt").write_text(error)
        raise
    finally:
        app.close()
