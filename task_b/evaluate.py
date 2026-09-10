"""Run one fresh-process episode of the original ATEC-TaskB-B2wPiper task.

This is a mobility/reachability bootstrap. It imports the installed original
environment (set ATEC_TASK_ROOT / PYTHONPATH), keeps the official physics,
assets, actions, rewards and terminations exactly as configured, and only sets
`num_envs`, the simulation device and Fabric. It stops at the first real
terminated/truncated signal, including during settling.

The policy receives public proprio and a static joint/action schema; the
visual approach mode also receives the original public RGB-D observations.
Root pose, object poses and contact forces are diagnostic records only and are
never passed to the policy.

Run headless with PYTHONNOUSERSITE=1 in the Isaac Lab environment; see
task_b/README.md for the exact invocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402  (only reads CLI args)
from tools.task_e.check_environment import compatible_experience  # noqa: E402  (read-only preflight)

TASK_ID = "ATEC-TaskB-B2wPiper"
ROBOT = "robot"
OBJECT_NAMES = tuple(f"object_{index}" for index in range(1, 19))
#: Action term class name -> command mode, used to describe the real terms.
ACTION_MODES = {"JointPositionAction": "position", "JointVelocityAction": "velocity",
                "JointEffortAction": "effort"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output", type=Path, required=True, help="New artifact directory; never overwritten.")
parser.add_argument("--mode", choices=("hold", "forward", "turn", "crouch", "visual_approach"), default="hold")
parser.add_argument("--stabilize", action="store_true", help="Apply the experimental Claude Opus stability controller.")
parser.add_argument("--stability_profile", choices=("baseline", "neutral"), default="baseline")
parser.add_argument("--stability_legs", choices=("raise_low_corners", "off"), default="raise_low_corners")
parser.add_argument("--vision_head", action="store_true", help="Allow the original head RGB-D camera to continue near-field visual tracking.")
parser.add_argument("--seed", type=int, default=42, help="Config seed (object layout) and reset seed.")
parser.add_argument("--max_steps", type=int, default=1500, help="Step cap; the official episode is far longer.")
parser.add_argument("--settle_calls", type=int, default=100)
parser.add_argument("--ramp_calls", type=int, default=100)
parser.add_argument("--wheel_cmd", type=float, default=0.10, help="Normalized wheel command after the ramp.")
parser.add_argument("--crouch_fraction", type=float, default=1.0, help="Fraction of the experimental crouch target, in [0,1].")
parser.add_argument("--video", action="store_true", help="Record the two public RGB cameras at their native 10 Hz.")
parser.add_argument("--rgb_interval", type=int, default=100, help="Save public RGB frames every N steps (0: only step 0 and final).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_steps < 1:
    parser.error("--max_steps must be >= 1")
if args.rgb_interval < 0:
    parser.error("--rgb_interval must be >= 0")
# This bootstrap is offscreen only: it shares a single 8 GB GPU and must not take
# a display. The official image observation group always needs Kit cameras.
args.headless = True
args.enable_cameras = True
if not getattr(args, "experience", ""):
    experience = compatible_experience(headless=True)
    if experience is not None:
        args.experience = str(experience)
args.output.mkdir(parents=True, exist_ok=False)
selected_experience = getattr(args, "experience", "")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import atec_rl_lab  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402  (registers ATEC-TaskB-*)
from atec_rl_lab.tasks.task_b.env_cfg import TaskBEnvB2WCfg  # noqa: E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from task_a.tools.d1g2_taska_camera_compat import prepare_camera_views  # noqa: E402
from task_b import control  # noqa: E402
from task_b.diagnostics import PreResetRecorder  # noqa: E402


def jsonable(value):
    """Best-effort JSON view of config objects; unknown objects become repr()."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    return repr(value)


def read_schema(task) -> control.ActionSchema:
    """Build the action schema from the live action manager, not from assumptions."""
    manager = task.action_manager
    robot = task.scene[ROBOT]
    terms, start = [], 0
    for name, dim in zip(manager.active_terms, manager.action_term_dim):
        term = manager.get_term(name)
        mode = ACTION_MODES.get(type(term).__name__)
        if mode is None:
            raise control.SchemaError(f"Action term '{name}' is {type(term).__name__}, which this bootstrap "
                                      "cannot describe; refusing to guess its command mode")
        resolved = getattr(term, "_joint_names", None)
        joint_names, source = ((tuple(resolved), "resolved") if resolved
                               else (tuple(term.cfg.joint_names), "cfg"))
        if not isinstance(term.cfg.scale, (int, float)):
            raise control.SchemaError(f"Action term '{name}' uses a non-scalar scale {term.cfg.scale!r}; "
                                      "this bootstrap only handles one scale per term")
        terms.append(control.ActionTerm(
            name=name, start=start, dim=int(dim), joint_names=joint_names, mode=mode,
            scale=float(term.cfg.scale), use_default_offset=getattr(term.cfg, "use_default_offset", None),
            clip=term.cfg.clip, joint_names_source=source))
        start += int(dim)
    return control.ActionSchema.from_terms(
        terms, tuple(robot.data.joint_names), robot.data.default_joint_pos[0].detach().cpu().numpy(),
        robot.data.soft_joint_pos_limits[0].detach().cpu().numpy(), manager.total_action_dim)


def read_illegal_contact(task) -> dict | None:
    """Resolve the official illegal_contact bodies so their forces can be logged."""
    manager = task.termination_manager
    if "illegal_contact" not in manager.active_terms:
        return None
    term_cfg = manager.get_term_cfg("illegal_contact")
    sensor_cfg = term_cfg.params["sensor_cfg"]
    sensor = task.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    body_ids = (list(range(sensor.num_bodies)) if isinstance(body_ids, slice)
                else [int(index) for index in body_ids])
    return {"sensor_name": sensor_cfg.name, "body_ids": body_ids,
            "body_names": [sensor.body_names[index] for index in body_ids],
            "threshold": float(term_cfg.params["threshold"]),
            "definition": "max over the contact-sensor history of |net force| per body, exactly as the "
                          "official illegal_contact term computes it"}


def illegal_force_norms(task, illegal: dict | None):
    """Per-body force norms mirroring the official term. Empty when unavailable."""
    if illegal is None:
        return np.zeros(0, dtype=np.float64)
    sensor = task.scene.sensors[illegal["sensor_name"]]
    history = sensor.data.net_forces_w_history
    if history is None:
        return np.zeros(len(illegal["body_ids"]), dtype=np.float64)
    norms = torch.norm(history[0][:, illegal["body_ids"]], dim=-1)
    return torch.max(norms, dim=0)[0].detach().cpu().numpy().astype(np.float64)


def sample_state(task, illegal: dict | None) -> dict:
    """Diagnostic simulator state. Not visible to the policy."""
    robot = task.scene[ROBOT]
    return {
        "q": robot.data.joint_pos[0].detach().cpu().numpy().copy(),
        "qdot": robot.data.joint_vel[0].detach().cpu().numpy().copy(),
        "base_xyz": robot.data.root_pos_w[0].detach().cpu().numpy().copy(),
        "base_quat": robot.data.root_quat_w[0].detach().cpu().numpy().copy(),
        "illegal_force": illegal_force_norms(task, illegal),
    }


def save_rgb(obs, output: Path, tag: str) -> dict:
    """Save the public RGB observations only; raw camera arrays are not kept."""
    saved = {}
    images = obs.get("image") if isinstance(obs, dict) else None
    if not isinstance(images, dict):
        return saved
    for key in ("head_rgb", "ee_rgb"):
        frame = images.get(key)
        if frame is None:
            continue
        array = frame[0].detach().cpu().numpy()
        name = f"{key}_{tag}.png"
        Image.fromarray(array[..., :3].astype(np.uint8)).save(output / name)
        saved[key] = name
    return saved


def write_source_manifest(output: Path) -> dict:
    """Hash the loaded bootstrap and original task sources. Hashes only, no copies."""
    roots = {"project": ROOT, "atec_rl_lab": Path(atec_rl_lab.__file__).resolve().parent}
    files = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not filename.endswith(".py"):
            continue
        path = Path(filename).resolve()
        for label, root in roots.items():
            if path.is_relative_to(root) and path.is_file():
                data = path.read_bytes()
                files[str(path)] = {"root": label, "relative_path": str(path.relative_to(root)),
                                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                break
    manifest = {
        "generated": "after environment creation, before the first step",
        "hashed_files": files,
        "no_physics_change_assertion": {
            "claim": "This run did not modify the official physics, assets, action terms, reward terms or "
                     "termination terms.",
            "basis": "configured, not measured: the evaluator constructs the official TaskBEnvB2WCfg and only "
                     "sets scene.num_envs=1, sim.device and sim.use_fabric, then calls the official "
                     "apply_safe_action_spec with no participant action spec.",
            "covered_by_hashes": "the exact bootstrap and atec_rl_lab .py sources loaded in this process",
            "not_covered_by_hashes": [
                "USD/USDA robot, object and scene assets, textures and MDL materials",
                "Isaac Lab and Isaac Sim packages, Kit experience files and GPU drivers",
                "runtime monkeypatching, including the Isaac Sim 4.5 camera Fabric pose reader that this "
                "evaluator installs on purpose (see result.json:camera_fabric_compatibility)",
                "whether the installed original repository itself matches upstream",
            ],
        },
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    output = args.output
    started = time.monotonic()
    steps, score, reason = 0, 0.0, "max_steps"
    env, trace, recorder, video, restore_camera_views = None, None, None, None, lambda: None
    telemetry = {key: [] for key in ("step", "sim_seconds", "alpha", "action", "requested_action", "proprio", "q", "qdot", "base_xyz",
                                     "base_quat", "reward_raw_total", "score", "illegal_force",
                                     "termination", "reward_terms")}
    try:
        cfg = TaskBEnvB2WCfg(seed=args.seed)  # seed drives the official object layout
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.sim.use_fabric = True
        # Mirror the official runner, which always applies the action spec helper.
        cfg = apply_safe_action_spec(cfg, None)

        restore_camera_views = prepare_camera_views()
        env = gym.make(TASK_ID, cfg=cfg)
        trace = (output / "trace.jsonl").open("w")
        obs, _ = env.reset(seed=args.seed)
        task = env.unwrapped
        robot = task.scene[ROBOT]
        dt = float(task.step_dt)
        schema = read_schema(task)
        illegal = read_illegal_contact(task)
        # Managers deep-copy then resolve their own SceneEntityCfg. task.cfg
        # still has an unresolved slice(None), which is NOT the observed order.
        observed_cfg = task.observation_manager.cfg.proprio.joint_pos.params["asset_cfg"]
        observed_ids = observed_cfg.joint_ids
        if isinstance(observed_ids, slice):
            observed_names = list(robot.data.joint_names)[observed_ids]
        else:
            observed_names = [robot.data.joint_names[int(i)] for i in observed_ids]
        defaults = dict(zip(schema.joint_names, schema.default_joint_pos.tolist()))
        if args.mode == "visual_approach":
            from task_b.visual_approach import VisualApproachPolicy
            policy = VisualApproachPolicy(schema, observed_names, defaults, dt=dt,
                                          settle_calls=args.settle_calls, ramp_calls=args.ramp_calls,
                                          **({"use_head": True} if args.vision_head else {}))
        else:
            policy = control.BootstrapPolicy(schema, args.mode, settle_calls=args.settle_calls,
                                             ramp_calls=args.ramp_calls, wheel_cmd=args.wheel_cmd,
                                             crouch_fraction=args.crouch_fraction)
        stabilizer = None
        if args.stabilize:
            if args.stability_profile == "neutral":
                from task_b.stability_profiles import NeutralAwareStabilityController as StabilityController
            else:
                from task_b.stability import StabilityController
            stabilizer = StabilityController(schema, observed_names, [defaults[name] for name in observed_names],
                                             dt=dt, leg_mode=args.stability_legs,
                                             settle_calls=args.settle_calls)
        reward_manager, termination_manager = task.reward_manager, task.termination_manager
        reward_names = list(reward_manager.active_terms)
        termination_names = list(termination_manager.active_terms)
        proprio_dim = int(obs["proprio"].shape[-1])
        observed_articulation_ids = [schema.joint_index(name) for name in observed_names]
        observation_q_error_max = 0.0

        def capture_terminal(env_ids):
            # Read cached physics buffers only; no render, scene update or reset.
            return jsonable({
                "state": sample_state(task, illegal),
                "termination_flags": {name: bool(termination_manager.get_term(name)[0].item())
                                      for name in termination_names},
                "reward_terms": {name: float(value[0]) for name, value in
                                 reward_manager.get_active_iterable_terms(0)},
                "object_xyz": {name: task.scene[name].data.root_pos_w[0]
                               for name in OBJECT_NAMES if name in task.scene.keys()},
            })

        recorder = PreResetRecorder(task, capture_terminal)
        recorder.__enter__()
        terminal_pre_reset = None

        metadata = {
            "task": TASK_ID, "seed": args.seed, "device": args.device,
            "step_dt": dt, "physics_dt": float(task.cfg.sim.dt), "decimation": int(task.cfg.decimation),
            "episode_length_s": float(task.max_episode_length_s),
            "max_episode_length_steps": int(task.max_episode_length),
            "action_schema": schema.to_dict(),
            "articulation": {"body_names": list(robot.data.body_names),
                             "joint_pos_limits": jsonable(robot.data.joint_pos_limits[0]),
                             "soft_joint_pos_limits": jsonable(robot.data.soft_joint_pos_limits[0])},
            "observations": {"proprio_dim": proprio_dim,
                             "joint_names": observed_names,
                             "proprio_expected_dim_for_this_schema": 12 + 3 * schema.total_dim,
                             "image_keys": sorted(obs["image"].keys()) if isinstance(obs.get("image"), dict) else [],
                             "policy_inputs": ("proprio + head/ee RGB-D" if args.vision_head else "proprio + ee RGB-D")
                             if args.mode == "visual_approach" else "proprio only"},
            "reward_terms": {name: {"weight": jsonable(reward_manager.get_term_cfg(name).weight),
                                    "func": repr(reward_manager.get_term_cfg(name).func),
                                    "params": jsonable(reward_manager.get_term_cfg(name).params)}
                             for name in reward_names},
            "termination_terms": {name: {"time_out": bool(termination_manager.get_term_cfg(name).time_out),
                                         "func": repr(termination_manager.get_term_cfg(name).func),
                                         "params": jsonable(termination_manager.get_term_cfg(name).params)}
                                  for name in termination_names},
            "illegal_contact": illegal,
            "contact_sensor_body_names": list(task.scene.sensors["contact_sensor"].body_names)
            if "contact_sensor" in task.scene.sensors else [],
            "policy": policy.describe(),
            "stability_controller": stabilizer.describe() if stabilizer else None,
            "diagnostics_not_visible_to_policy": {
                "note": "recorded for auditing only; never passed to the policy",
                "env_origin": jsonable(task.scene.env_origins[0]),
                "initial_base_xyz": jsonable(robot.data.root_pos_w[0]),
                "initial_base_quat_wxyz": jsonable(robot.data.root_quat_w[0]),
                "initial_joint_pos": jsonable(robot.data.joint_pos[0]),
                "initial_object_xyz": {name: jsonable(task.scene[name].data.root_pos_w[0])
                                       for name in OBJECT_NAMES if name in task.scene.keys()},
            },
        }
        (output / "environment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest = write_source_manifest(output)
        frames = {"step_00000": save_rgb(obs, output, "step_00000")}
        if args.video:
            import imageio.v2 as imageio
            import cv2
            video = imageio.get_writer(output / "public_cameras.mp4", fps=10,
                                       codec="libx264", quality=7, macro_block_size=1,
                                       ffmpeg_params=["-threads", "2", "-movflags", "+faststart"])

        terminated_flag = truncated_flag = False
        active_terms, reward_step, final_state = [], {}, None
        for step in range(args.max_steps):
            if not app.is_running():
                reason = "app_stopped"
                break
            state = sample_state(task, illegal)
            proprio = obs["proprio"][0].detach().cpu().numpy().copy()
            # Audit the declared public ordering against the simulator buffer.
            # This diagnostic is never supplied to the controller.
            observed_q = proprio[12:36] + np.asarray([defaults[name] for name in observed_names])
            observation_q_error = float(np.max(np.abs(observed_q - state["q"][observed_articulation_ids])))
            observation_q_error_max = max(observation_q_error_max, observation_q_error)
            if observation_q_error > 1e-4:
                raise RuntimeError(f"Observation joint mapping mismatch: {observation_q_error}")
            if args.mode == "visual_approach":
                images = {key: value[0].detach().cpu().numpy()
                          for key, value in obs["image"].items() if args.vision_head or key.startswith("ee_")}
                requested_action = policy.act(proprio, images)
            else:
                requested_action = policy.act(proprio)
            action = stabilizer.apply(requested_action, proprio) if stabilizer else requested_action.copy()
            tensor = torch.as_tensor(action, device=task.device, dtype=torch.float32).unsqueeze(0)
            if tensor.shape != (1, schema.total_dim) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Expected a finite action of shape (1, {schema.total_dim}), got {tensor.shape}")
            with torch.inference_mode(), recorder.step():
                obs, reward, terminated, truncated, _ = env.step(tensor)
            steps = step + 1

            # Official reward is dt-scaled; dividing by step_dt gives the raw
            # weighted term sum per step, which is what the official runner sums.
            reward_raw_total = float(reward.item()) / dt
            score += reward_raw_total
            reward_step = {name: float(value[0]) for name, value in
                           reward_manager.get_active_iterable_terms(0)}
            term_flags = {name: bool(termination_manager.get_term(name)[0].item())
                          for name in termination_names}
            if recorder.last_record is not None:
                terminal_pre_reset = jsonable(recorder.last_record)
                snapshot = terminal_pre_reset.get("snapshot")
                if snapshot is not None:
                    reward_step = snapshot["reward_terms"]
                    term_flags = snapshot["termination_flags"]
            terminated_flag, truncated_flag = bool(terminated[0].item()), bool(truncated[0].item())
            active_terms = [name for name, fired in term_flags.items() if fired]
            if video is not None and steps % 5 == 0 and not (terminated_flag or truncated_flag):
                rgb = [obs["image"][key][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
                       for key in ("head_rgb", "ee_rgb")]
                frame = np.concatenate(rgb, axis=1)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (15, 20, 26), -1)
                cv2.putText(frame, f"Task B | {args.mode} | {steps * dt:.1f}s | raw score {score:.0f} | head RGB / wrist RGB",
                            (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
                video.append_data(frame)

            telemetry["step"].append(steps)
            telemetry["sim_seconds"].append(steps * dt)
            telemetry["alpha"].append(policy.alpha)
            telemetry["action"].append(action)
            telemetry["requested_action"].append(requested_action)
            telemetry["proprio"].append(proprio)
            telemetry["q"].append(state["q"])
            telemetry["qdot"].append(state["qdot"])
            telemetry["base_xyz"].append(state["base_xyz"])
            telemetry["base_quat"].append(state["base_quat"])
            telemetry["reward_raw_total"].append(reward_raw_total)
            telemetry["score"].append(score)
            telemetry["illegal_force"].append(state["illegal_force"])
            telemetry["termination"].append([term_flags[name] for name in termination_names])
            telemetry["reward_terms"].append([reward_step.get(name, 0.0) for name in reward_names])

            row = {
                "step": steps, "sim_seconds": round(steps * dt, 4), "alpha": policy.alpha,
                "reward_raw_total": reward_raw_total, "score": score,
                "reward_terms": reward_step,
                "pre_step_base_xyz": [round(float(value), 5) for value in state["base_xyz"]],
                "pre_step_base_quat_wxyz": [round(float(value), 5) for value in state["base_quat"]],
                "pre_step_max_illegal_force": (float(state["illegal_force"].max())
                                               if state["illegal_force"].size else None),
                "terminated": terminated_flag, "truncated": truncated_flag,
                "active_termination_terms": active_terms,
                "policy_state": getattr(policy, "state", args.mode),
                "observation_joint_mapping_max_abs_error": observation_q_error,
                "policy_debug": getattr(policy, "debug", None),
                "stability_debug": stabilizer.debug if stabilizer else None,
            }
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            if steps % 100 == 0 or reward_raw_total or terminated_flag or truncated_flag:
                print("TASK_B_PROGRESS " + json.dumps(row), flush=True)
            if args.rgb_interval and steps % args.rgb_interval == 0:
                frames[f"step_{steps:05d}"] = save_rgb(obs, output, f"step_{steps:05d}")
            if terminated_flag or truncated_flag:
                reason = "terminated" if terminated_flag else "truncated"
                # The official env already reset joints for this env, so this is
                # the post-reset state; the base pose is untouched because Task B
                # has no root-state reset event.
                final_state = sample_state(task, illegal)
                break

        frames["final"] = save_rgb(obs, output, "final")
        arrays = {key: np.asarray(value) for key, value in telemetry.items()}
        np.savez_compressed(output / "telemetry.npz", dt=dt, joint_names=np.asarray(schema.joint_names),
                            termination_term_names=np.asarray(termination_names),
                            reward_term_names=np.asarray(reward_names),
                            illegal_contact_body_names=np.asarray(illegal["body_names"] if illegal else []),
                            **arrays)

        displacement = None
        if steps:
            start_xyz, last_xyz = arrays["base_xyz"][0], arrays["base_xyz"][-1]
            displacement = {
                "first_recorded_base_xyz": start_xyz.tolist(), "last_recorded_base_xyz": last_xyz.tolist(),
                "delta_xyz": (last_xyz - start_xyz).tolist(),
                "planar_distance_m": float(np.linalg.norm((last_xyz - start_xyz)[:2])),
                "max_planar_distance_m": float(np.max(np.linalg.norm(
                    arrays["base_xyz"][:, :2] - start_xyz[:2], axis=1))),
                "min_base_z_m": float(np.min(arrays["base_xyz"][:, 2])),
                "max_base_z_m": float(np.max(arrays["base_xyz"][:, 2])),
                "note": "world frame, pre-step samples; the sign tells which way the wheel command drove the base",
            }
        term_totals = (dict(zip(reward_names, arrays["reward_terms"].sum(axis=0).tolist()))
                       if steps else {})
        checksum = (float(np.max(np.abs(arrays["reward_terms"].sum(axis=1) - arrays["reward_raw_total"])))
                    if steps else None)
        result = {
            "task": TASK_ID, "mode": args.mode, "seed": args.seed, "device": args.device,
            "headless": True, "enable_cameras": True, "use_fabric": True,
            "experience": selected_experience or None,
            "camera_fabric_compatibility": "task_a.tools.d1g2_taska_camera_compat.prepare_camera_views "
                                           "(read-only recursive world-pose reader; preserves moving camera inheritance)",
            "action_spec": None, "apply_safe_action_spec": "called with no participant spec",
            "steps": steps, "max_steps": args.max_steps, "sim_seconds": steps * float(task.step_dt),
            "wall_seconds": time.monotonic() - started, "stop_reason": reason,
            "terminated": terminated_flag, "truncated": truncated_flag,
            "active_termination_terms_at_stop": active_terms,
            "terminated_during_settle": bool((terminated_flag or truncated_flag)
                                             and steps <= args.settle_calls),
            "score_definition": "sum over steps of env reward / step_dt, matching the official "
                                "scripts/play_atec_task.py accumulation",
            "score_raw_total": score,
            "reward_term_totals_raw": term_totals,
            "reward_term_total_definition": "per step, weighted term value with dt removed "
                                            "(RewardManager.get_active_iterable_terms), summed over steps",
            "reward_term_sum_vs_env_reward_max_abs_error": checksum,
            "final_reward_terms": reward_step,
            "base_motion": displacement,
            "observation_joint_mapping_max_abs_error": observation_q_error_max,
            "terminal_pre_reset": terminal_pre_reset,
            "terminal_post_step_state_after_official_reset": None if final_state is None else {
                "base_xyz": final_state["base_xyz"].tolist(),
                "base_quat_wxyz": final_state["base_quat"].tolist(),
                "max_illegal_force": (float(final_state["illegal_force"].max())
                                      if final_state["illegal_force"].size else None),
                "illegal_force_per_body": dict(zip(illegal["body_names"] if illegal else [],
                                                   final_state["illegal_force"].tolist())),
                "note": "read after env.step(); the official reset already restored default joint "
                        "positions for this env, while the base pose is untouched",
            },
            "policy": policy.describe(),
            "stability_controller": stabilizer.describe() if stabilizer else None,
            "rgb_frames": frames,
            "video": "public_cameras.mp4" if args.video else None,
            "files": {"trace": "trace.jsonl", "telemetry": "telemetry.npz",
                      "metadata": "environment_metadata.json", "manifest": "source_manifest.json"},
            "task_physics_modified": False,
            "manifest_coverage": manifest["no_physics_change_assertion"]["not_covered_by_hashes"],
            "interpretation": "Mobility/reachability bootstrap. Survival, driving or crouching is NOT a "
                              "successful grasp and NOT a Task B pass. Any grasped_objects reward only "
                              "means the gripper body came within the official distance threshold of an "
                              "object; objects_in_circle is the term that scores delivery.",
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("TASK_B_RESULT " + json.dumps(result), flush=True)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        (output / "failure.txt").write_text(error)
        raise
    finally:
        if video is not None:
            video.close()
        if recorder is not None:
            recorder.close()
        if trace is not None:
            trace.close()
        if env is not None:
            env.close()
        restore_camera_views()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
