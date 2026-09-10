"""Diagnostic short traversals on the ORIGINAL Task A terrain; never a pass.

One environment evaluates multiple starts sequentially. Defaults cross the
first slope and three original stair tiles, including their full 20 m length:
  --starts -31 49 69 109  (default exits are each start + 22 m)
Use --exits to specify matching individual endpoints. World positions are
used only for reset placement, diagnostics and deciding when a case ends.
The controller receives only proprioception and integrates gyro heading.
This script is import-safe and does not import the full-course main runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_terrain_mesh_from_usd(stage, terrain_path="/World/ground/terrain"):
    """Cache the imported static terrain in world coordinates, for RESET only.

    TerrainImporter.import_mesh / create_prim_from_mesh author a triangle
    Mesh at ``{terrain_path}/mesh`` (lowercase). Read that composed geometry
    once, including all ancestor transforms, without generating terrain again
    or depending on importer-specific mesh/generator attributes.
    """
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(terrain_path)
    if not root.IsValid():
        raise RuntimeError(f"Imported terrain prim is missing: {terrain_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices, faces, paths, offset = [], [], [], 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise RuntimeError(f"Invalid terrain points: {prim.GetPath()}")
        if not len(counts) or np.any(counts != 3) or indices.size != 3 * len(counts):
            raise RuntimeError(f"Expected original triangle terrain topology: {prim.GetPath()}")
        if indices.min() < 0 or indices.max() >= len(points):
            raise RuntimeError(f"Invalid terrain face indices: {prim.GetPath()}")
        # Gf.Matrix4d uses row vectors: translation occupies the last ROW.
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ transform
        if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 3]) < 1e-12):
            raise RuntimeError(f"Invalid terrain world transform: {prim.GetPath()}")
        vertices.append(homogeneous[:, :3] / homogeneous[:, 3:4])
        faces.append(indices.reshape(-1, 3) + offset)
        paths.append(str(prim.GetPath()))
        offset += len(points)
    if not vertices:
        raise RuntimeError(f"No imported terrain Mesh beneath {terrain_path}")
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False,
        metadata={"source": "existing USD geometry", "source_prim_paths": paths,
                  "coordinate_frame": "world", "usage": "reset height sampling only"},
    )


def sample_spawn_height(mesh, x, y=0.0):
    """Intersect vertical lines with terrain triangles on CPU, for RESET only.

    Take the highest surface at each of nine footprint samples, then their
    maximum plus the original 0.6 m root clearance. No ray/height observation
    is added to the policy. Barycentric interpolation needs no rtree package.
    Missing geometry is an error, never silently assumed to be flat ground.
    """
    import numpy as np

    triangles = np.asarray(mesh.triangles)
    lo, hi = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
    nearby = (hi[:, 0] >= x - .45) & (lo[:, 0] <= x + .45)
    nearby &= (hi[:, 1] >= y - .30) & (lo[:, 1] <= y + .30)
    triangles = triangles[nearby]
    a = triangles[:, 0]
    ab, ac = triangles[:, 1] - a, triangles[:, 2] - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    valid = np.abs(det) > 1e-12  # vertical faces have no XY area
    a, ab, ac, det = a[valid], ab[valid], ac[valid], det[valid]
    samples = []
    for dx in (-.45, 0., .45):
        for dy in (-.30, 0., .30):
            qx, qy = x + dx - a[:, 0], y + dy - a[:, 1]
            u = (qx * ac[:, 1] - qy * ac[:, 0]) / det
            v = (ab[:, 0] * qy - ab[:, 1] * qx) / det
            inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
            heights = (a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[inside]
            heights = heights[np.isfinite(heights)]
            if not len(heights):
                raise RuntimeError(f"No terrain at reset footprint ({x + dx}, {y + dy})")
            samples.append({"x": x + dx, "y": y + dy, "ground_z": float(heights.max())})
    return max(p["ground_z"] for p in samples) + .6, samples


class ProprioHeadingController:
    """ZYX yaw integration from the same 81-value proprioceptive interface."""

    def __init__(self):
        self.heading = 0.0
        self.previous_rate = None

    def command(self, proprio, speed, dt):
        import numpy as np

        proprio = np.asarray(proprio)
        if proprio.shape != (81,) or not np.isfinite(proprio).all():
            raise ValueError("Expected finite 81-value proprioception")
        gy, gz = proprio[10:12]
        rate = float((-gy * proprio[4] - gz * proprio[5]) / max(float(gy * gy + gz * gz), .05))
        if self.previous_rate is not None:
            self.heading += .5 * (self.previous_rate + rate) * dt
        self.previous_rate = rate
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        return np.asarray([speed, 0., np.clip(-self.heading, -.5, .5)], dtype=np.float32)


def diagnostic_state(env, include_contacts=True):
    """World pose here is recording only; never passed to the controller."""
    import torch

    robot, sensor = env.scene["robot"], env.scene["contact_sensor"]
    gravity = robot.data.projected_gravity_b[0].tolist()
    quaternion = robot.data.root_quat_w[0].tolist()
    w, x, y, z = quaternion
    contacts = None
    if include_contacts:
        peak = sensor.data.net_forces_w_history[0].norm(dim=-1).amax(dim=0)
        ids = torch.nonzero(peak > 1.0).flatten().tolist()
        contacts = {sensor.body_names[i]: float(peak[i]) for i in ids}
    return {
        "xyz": robot.data.root_pos_w[0].tolist(),
        "quaternion_wxyz": quaternion,
        "diagnostic_world_yaw_rad": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        "projected_gravity": gravity,
        "tilt_degrees": math.degrees(math.acos(max(-1., min(1., -gravity[2])))),
        "contact_history_peak_over_1n": contacts,
    }


def segment_env_class(base_env_class):
    """Keep the first terminal state in place until the next explicit reset."""

    class SegmentEnv(base_env_class):
        capture_terminal = False
        terminal_record = None

        def _reset_idx(self, env_ids):
            if self.capture_terminal and len(env_ids):
                self.terminal_record = diagnostic_state(self)
                self.terminal_record["termination_terms"] = [
                    name for name in self.termination_manager.active_terms
                    if bool(self.termination_manager.get_term(name)[0])
                ]
                # env.step will return its original terminated/truncated flags.
                # Do not let automatic reset replace the failed robot state.
                # The case loop stops immediately; the next case explicitly
                # disables capture before calling env.reset().
                return
            super()._reset_idx(env_ids)

    return SegmentEnv


def course_tiles(generator_cfg):
    """Derive world bounds from the original generator's centering rule."""
    if tuple(generator_cfg.size) != (20., 20.) or generator_cfg.num_rows != 15 or generator_cfg.num_cols != 1:
        raise ValueError("Expected the original 15 x 1 Task A terrain of 20 m tiles")
    names = generator_cfg.terrain_sequence
    if len(names) != generator_cfg.num_rows:
        raise ValueError("Task A terrain sequence does not match its row count")
    return [{"index": i, "type": name, "x_min": -150. + 20 * i, "x_max": -130. + 20 * i}
            for i, name in enumerate(names)]


def run_case(env, policy, args, start_x, exit_x, index, tiles, terrain_mesh):
    import numpy as np
    import torch

    directory = args.output / f"case_{index:02d}"
    directory.mkdir()
    z, samples = sample_spawn_height(terrain_mesh, start_x, args.start_y)
    # The original reset_d1g2_root event writes these absolute default states.
    # It deliberately does not add a training terrain origin.
    default = env.scene["robot"].data.default_root_state
    default[0, :3] = torch.tensor([start_x, args.start_y, z], device=env.device)
    default[0, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
    default[0, 7:13] = 0.
    env.capture_terminal = False
    env.terminal_record = None
    obs, _ = env.reset(seed=args.seed)
    policy.reset()
    controller = ProprioHeadingController()
    # Before any new physics step, PhysX's last contact report may still be
    # from the previous case. Do not repopulate cleared sensor history with it.
    state = diagnostic_state(env, include_contacts=False)
    if not np.allclose(state["xyz"], [start_x, args.start_y, z], atol=.01):
        raise RuntimeError(f"Incorrect diagnostic reset: {state['xyz']}")
    if tuple(obs["proprio"].shape) != (1, 81) or tuple(env.action_space.shape) != (1, 23):
        raise RuntimeError("Unexpected D1+G2 observation or action interface")
    result = {
        "diagnostic_only": True, "full_course_reached_goal": False,
        "case_index": index, "start_x": start_x, "exit_x": exit_x,
        "start_y": args.start_y, "root_spawn_z": z, "reset_footprint_samples": samples,
        "fully_crossed_tile_targets": [t for t in tiles if start_x <= t["x_min"] and exit_x >= t["x_max"]],
        "initial_state": state, "trace": str((directory / "trace.jsonl").resolve()),
        "status": "running", "segment_exit_reached": False,
    }
    save_json(directory / "result.json", result)
    env.capture_terminal = True
    max_x, max_abs_y, max_tilt = state["xyz"][0], abs(state["xyz"][1]), state["tilt_degrees"]
    started = time.monotonic()
    steps, done, terminated_flag, truncated_flag = 0, False, False, False
    reason = "segment_time_limit"
    with (directory / "trace.jsonl").open("w") as trace, torch.inference_mode():
        trace.write(json.dumps({"step": 0, "sim_seconds": 0., **state}, allow_nan=False) + "\n")
        for step in range(math.floor(args.case_seconds / env.step_dt)):
            if time.monotonic() - started >= args.max_wall_seconds:
                reason = "wall_time_limit"
                break
            if not args.sim_app.is_running():
                reason = "application_closed"
                break
            proprio = obs["proprio"][0].detach().cpu().numpy()
            if not np.isfinite(proprio).all():
                reason = "nonfinite_observation"
                break
            ramp = float(np.clip((step - args.settle_steps) * env.step_dt / 2., 0., 1.))
            command = controller.command(proprio, args.speed * ramp, env.step_dt)
            action = np.zeros(23, np.float32) if step < args.settle_steps else policy.action_from_state(
                proprio[3:6], proprio[9:12], proprio[12:35], proprio[35:58], command, dt=env.step_dt)
            if not np.isfinite(action).all():
                reason = "nonfinite_action"
                break
            obs, _, terminated, truncated, _ = env.step(torch.as_tensor(action[None], device=env.device))
            steps = step + 1
            terminated_flag, truncated_flag = bool(terminated[0]), bool(truncated[0])
            done = terminated_flag or truncated_flag
            state = env.terminal_record if done else diagnostic_state(env)
            if state is None:
                raise RuntimeError("Termination occurred without a captured terminal state")
            max_x = max(max_x, state["xyz"][0])
            max_abs_y = max(max_abs_y, abs(state["xyz"][1]))
            max_tilt = max(max_tilt, state["tilt_degrees"])
            at_exit = state["xyz"][0] >= exit_x
            row = {"step": steps, "sim_seconds": steps * env.step_dt,
                   "heading_estimate_rad": controller.heading, "command": command.tolist(),
                   "terminated": terminated_flag, "truncated": truncated_flag, **state}
            if steps % args.trace_interval == 0 or done or at_exit:
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                trace.flush()
            if done:  # Original contact/fall failure wins over simultaneous exit crossing.
                reason = "original_termination"
                break
            if at_exit:
                reason = "segment_exit"
                result["segment_exit_reached"] = True
                break
        trace.write(json.dumps({"final": True, "step": steps, "sim_seconds": steps * env.step_dt, **state},
                               allow_nan=False) + "\n")
    env.capture_terminal = False
    result.update(status="finished", stop_reason=reason, steps=steps, sim_seconds=steps * env.step_dt,
                  wall_seconds=time.monotonic() - started, final_state=state,
                  max_x=max_x, maximum_forward_m=max_x - start_x, max_abs_y=max_abs_y,
                  max_tilt_degrees=max_tilt, terminated=terminated_flag, truncated=truncated_flag,
                  full_course_reached_goal=False)
    save_json(directory / "result.json", result)
    print("D1G2_SEGMENT_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    sys.path.insert(0, str(ROOT))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual_checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--assets_root", type=Path, default=Path("/home/lybm/DDT_Lab"))
    parser.add_argument("--starts", type=float, nargs="+", default=[-31., 49., 69., 109.])
    parser.add_argument("--exits", type=float, nargs="+")
    parser.add_argument("--start_y", type=float, default=0., help="Lateral reset coordinate; never a controller input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--case_seconds", type=float, default=120.)
    parser.add_argument("--max_wall_seconds", type=float, default=600., help="Per case, excludes application startup")
    parser.add_argument("--settle_steps", type=int, default=0)
    parser.add_argument("--trace_interval", type=int, default=25)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.exits is None:
        args.exits = [x + 22. for x in args.starts]
    if len(args.starts) != len(args.exits):
        parser.error("--starts and --exits must have the same length")
    if not math.isfinite(args.start_y) or abs(args.start_y) > 9.7:
        parser.error("start_y must be finite and in [-9.7,9.7] so the reset footprint stays on the course")
    if not all(math.isfinite(s) and math.isfinite(e) and -149.5 <= s < e <= 145.
               for s, e in zip(args.starts, args.exits)):
        parser.error("Each case must satisfy -149.5 <= start < exit <= 145")
    if not (0. < args.case_seconds <= 1200. and math.isfinite(args.max_wall_seconds) and args.max_wall_seconds > 0):
        parser.error("case_seconds must be in (0,1200]; max_wall_seconds must be finite and positive")
    if not 0 < args.speed <= 1 or args.settle_steps < 0 or args.trace_interval < 1:
        parser.error("speed must be in (0,1]; settle_steps >= 0; trace_interval >= 1")
    args.policy = args.policy.resolve(strict=True)
    args.residual_checkpoint = args.residual_checkpoint.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    args.enable_cameras = False
    if not args.experience:
        suffix = "headless" if args.headless else "rendering"
        args.experience = f"/home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.{suffix}.kit"
    app, env = None, None
    report = {
        "status": "initializing", "diagnostic_only": True, "full_course_reached_goal": False,
        "official_submission": False, "seed": args.seed, "speed_command_mps": args.speed,
        "start_y": args.start_y,
        "case_seconds_limit": args.case_seconds,
        "scope": "Original Task A terrain and D1+G2 physics; diagnostic teleports to segment entries",
        "controller_inputs": ["proprio gyro", "proprio projected gravity", "proprio joints", "action history"],
        "heading_limitation": "Gyro integration can drift; segment failure may include navigation error",
        "world_pose_usage": "Reset placement, recorded diagnostics and segment endpoint detection only",
        "terminal_state_policy": "Original terminations preserved; automatic reset suppressed until case ends",
        "cases": [],
    }
    save_json(args.output / "result.json", report)
    try:
        app = AppLauncher(args).app
        args.sim_app = app
        from atec_rl_lab.tasks.task_base.envs_base import BaseRLEnv
        from tools.d1g2_taska_env import build_d1g2_taska_cfg
        from tools.d1g2_taska_residual import D1G2ResidualPolicy
        import torch

        cfg = build_d1g2_taska_cfg(device=args.device, cameras=False, seed=args.seed, ddt_root=args.assets_root)
        cfg.sim.use_fabric = True
        tiles = course_tiles(cfg.scene.terrain.terrain_generator)
        policy = D1G2ResidualPolicy(args.residual_checkpoint, args.policy)
        report.update(policy=policy.metadata, terrain_tiles=tiles,
                      course_seed=cfg.scene.terrain.terrain_generator.seed,
                      original_time_limit_seconds=cfg.episode_length_s,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        print("D1G2_SEGMENT_STAGE create_original_environment", flush=True)
        env = segment_env_class(BaseRLEnv)(cfg=cfg)
        terrain_mesh = read_terrain_mesh_from_usd(env.scene.stage, cfg.scene.terrain.prim_path + "/terrain")
        report.update(status="running", simulation_dt=env.step_dt, num_envs=env.num_envs,
                      reset_height_mesh={**terrain_mesh.metadata, "vertices": len(terrain_mesh.vertices),
                                         "triangles": len(terrain_mesh.faces), "bounds": terrain_mesh.bounds.tolist()},
                      original_termination_terms=list(env.termination_manager.active_terms))
        save_json(args.output / "result.json", report)
        # Physics/policy buffers first created while stepping in inference mode
        # must also be reset in that mode between diagnostic cases.
        with torch.inference_mode():
            for i, (start, end) in enumerate(zip(args.starts, args.exits)):
                report["cases"].append(run_case(env, policy, args, start, end, i, tiles, terrain_mesh))
                save_json(args.output / "result.json", report)
                if report["cases"][-1]["stop_reason"] == "application_closed":
                    break
        report["status"] = "finished"
    except BaseException as exc:
        report.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save_json(args.output / "result.json", report)
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
