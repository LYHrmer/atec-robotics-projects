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
parser.add_argument("--mode", choices=("hold", "forward", "turn", "crouch", "visual_approach", "first_reach", "reach_probe", "stance_descend", "grasp_probe"), default="hold")
parser.add_argument("--camera_free", action="store_true", help="Build the official task with the three scene cameras and the lidar nulled and the image/extero observation groups removed, so Kit never creates a render product. Required for modes that do not consume images.")
parser.add_argument("--stance_drop_max", type=float, default=.24, help="stance_descend: maximum commanded wheels-planted body drop, in metres.")
parser.add_argument("--stance_drop_rate", type=float, default=.03, help="stance_descend: commanded descent rate, m/s.")
parser.add_argument("--stance_hold_s", type=float, default=2., help="stance_descend: seconds to hold the final commanded drop.")
parser.add_argument("--stance_grid_step", type=float, default=.005, help="stance_descend: leg-solution grid spacing, m.")
parser.add_argument("--probe_object", type=int, default=0, help="grasp_probe: object index 1-18; 0 picks the one nearest the robot after settling.")
parser.add_argument("--probe_standoff", type=float, default=.50, help="grasp_probe: body-frame forward distance to park the object at, m.")
parser.add_argument("--probe_drop", type=float, default=.05, help="grasp_probe: wheels-planted body descent after the reach, m.")
parser.add_argument("--probe_clearance", type=float, default=.03, help="grasp_probe: how far below the object's top the jaw midpoint aims, m. Negative aims above the object, which isolates the gripper from the object.")
parser.add_argument("--probe_park_settle", type=int, default=100, help="grasp_probe: steps to settle after parking, before measuring the object.")
parser.add_argument("--probe_preload", type=float, default=.025, help="grasp_probe: finger travel commanded past the closed stop, m. The position servo only presses as hard as its error, so closing exactly at the stop leaves a weak grip.")
parser.add_argument("--reach_forward", type=float, default=.20)
parser.add_argument("--reach_turn_cap", type=float, default=.20)
parser.add_argument("--reach_turn_gain", type=float, default=.4)
parser.add_argument("--reach_standoff", type=float, default=.56)
parser.add_argument("--reach_lowering", type=float, default=0., help="Fixed nominal reference lowering in [0,.25] m after stationary reach; actual descent is measured independently. The wheels-planted envelope was measured at 0.253 m with no official contact.")
parser.add_argument("--reach_grasp", action="store_true", help="After the fixed lowering hold, close the jaws and lift back along the descent reference. Opt-in: the default path ends at the lowering hold.")
parser.add_argument("--wheel_action_gain", type=float, default=1., help="Explicit actuator-drive probe: multiply normalized wheel requests before optional stabilization.")
parser.add_argument("--stance_hold", action="store_true")
parser.add_argument("--brake_wheel_hold", action="store_true", help="Hold public wheel-angle anchors during stationary first_reach phases; experimental physical wheel-speed feedback.")
parser.add_argument("--stance_profile", choices=("off", "compact"), default="off")
parser.add_argument("--score_hold_steps", type=int, default=0, help="If positive, stop this evaluation N steps after first positive reward; no score is fed to policy.")
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
if not 0 < args.wheel_action_gain <= 8:
    parser.error("--wheel_action_gain must be in (0,8]")
if args.stance_profile != "off" and not args.stance_hold:
    parser.error("--stance_profile requires --stance_hold")
if args.mode in ("first_reach", "reach_probe") and not 0 <= args.reach_lowering <= .25:
    parser.error("--reach_lowering must be in [0,.25] m")
if args.reach_grasp:
    if args.mode != "first_reach":
        parser.error("--reach_grasp requires --mode first_reach")
    if args.reach_lowering <= 0:
        parser.error("--reach_grasp requires --reach_lowering > 0: from the unlowered stance the "
                     "fingertips stop above every Task B object")
if args.brake_wheel_hold and args.mode != "first_reach":
    parser.error("--brake_wheel_hold requires --mode first_reach")
if args.mode == "first_reach" and args.reach_lowering > 0 and not (
        args.brake_wheel_hold and args.stance_hold and args.stance_profile == "compact"):
    parser.error("first_reach lowering requires --brake_wheel_hold --stance_hold --stance_profile compact")
if args.mode == "stance_descend":
    if not args.stance_hold:
        parser.error("stance_descend requires --stance_hold so the commanded leg geometry is tracked")
    if not args.camera_free:
        parser.error("stance_descend consumes no images; pass --camera_free")
    for name in ("stance_drop_max", "stance_drop_rate", "stance_grid_step"):
        if not getattr(args, name) > 0:
            parser.error(f"--{name} must be positive")
    if args.stance_drop_rate <= 0 or args.stance_hold_s < 0:
        parser.error("--stance_drop_rate must be positive and --stance_hold_s non-negative")
    if args.stance_grid_step > args.stance_drop_max:
        parser.error("--stance_grid_step must not exceed --stance_drop_max")
if args.camera_free and (args.video or args.vision_head):
    parser.error("--camera_free cannot be combined with --video or --vision_head")
if args.mode == "grasp_probe":
    if not args.camera_free:
        parser.error("grasp_probe is an oracle probe and consumes no images; pass --camera_free")
    if not 1 <= args.probe_object <= 18 and args.probe_object != 0:
        parser.error("--probe_object must be 0 (nearest) or 1-18")
    if not .1 <= args.probe_standoff <= .9:
        parser.error("--probe_standoff must be in [0.1, 0.9] m")
    if not 0. < args.probe_drop <= .25:
        parser.error("--probe_drop must be in (0, 0.25] m")
    if not -.12 <= args.probe_clearance <= .08:
        parser.error("--probe_clearance must be in [-0.12, 0.08] m")
    if args.probe_park_settle < 1:
        parser.error("--probe_park_settle must be >= 1")
    if not 0. <= args.probe_preload <= .08:
        parser.error("--probe_preload must be in [0, 0.08] m")
# This bootstrap is offscreen only: it shares a single 8 GB GPU and must not take
# a display. The official image observation group always needs Kit cameras, so
# image-free modes must remove that group rather than rely on headless alone.
args.headless = True
args.enable_cameras = not args.camera_free
if not args.camera_free and not getattr(args, "experience", ""):
    # The rendering experience is only correct when the image observation group
    # is present. An image-free run must take AppLauncher's plain headless
    # experience so Kit builds no render product at all.
    experience = compatible_experience(headless=True)
    if experience is not None:
        args.experience = str(experience)
args.output.mkdir(parents=True, exist_ok=False)
selected_experience = getattr(args, "experience", "")
if args.camera_free:
    # Isaac Lab 2.3.2 drops the `create_new_stage=False` it computes for headless
    # runs: AppLauncher._sim_app_config keeps only the keys listed in
    # _SIM_APP_CFG_TYPES, and `create_new_stage` is absent from that dict. The app
    # then falls back to its own default of True, so SimulationApp._wait_for_viewport
    # never takes its "no new stage" exit and spins forever -- a renderer-free
    # experience never produces a viewport handle. Declaring the key lets the value
    # Isaac Lab already computed reach the app. It changes nothing else: with a
    # viewport present the loop exits on its own, which is why camera runs are fine.
    AppLauncher._SIM_APP_CFG_TYPES["create_new_stage"] = [bool]
app = AppLauncher(args).app
if args.camera_free:
    # With create_new_stage=False, SimulationApp skips its own new_stage() call, so
    # the stage that SimulationContext requires has to be created here -- the same
    # single call SimulationApp would have made. Nothing else about the app changes.
    import omni.usd  # noqa: E402  (importable only after the app exists)
    omni.usd.get_context().new_stage()

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
        "gripper_xyz": robot.data.body_pos_w[0, robot.data.body_names.index("gripper_base")].detach().cpu().numpy().copy(),
        "object_xyz": np.stack([task.scene[name].data.root_pos_w[0].detach().cpu().numpy().copy() for name in OBJECT_NAMES]),
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
    """Hash loaded sources and preserve exact project Python for reproducibility."""
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
                if label == "project":
                    snapshot = output / "source_snapshots" / path.relative_to(root)
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(data)
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


#: Vertical half-extent of each Task B object type, from its USD bounds and the
#: spawn rotation in task_b/env_cfg.py. The origin is the bounding-box centre, so
#: top-of-object = resting root z + this.
OBJECT_VERTICAL_HALF_EXTENT = {"sugar": 0.04635, "mustard": 0.09565, "banana": 0.01930}
#: World-frame direction of each type's narrowest horizontal extent, from the spawn
#: quaternions in task_b/env_cfg.py. The fingers must close across this axis.
NARROW_HORIZONTAL_AXIS = {"sugar": (1., 0.), "mustard": (0., 1.), "banana": (0., 1.)}


def object_kind(index: int) -> str:
    return "sugar" if index <= 6 else "mustard" if index <= 12 else "banana"


def park_and_measure(env, robot, schema, args, dt) -> dict:
    """Probe setup: settle, park the base beside one object, measure the plan.

    This is ORACLE setup, not policy behaviour. It reads true object poses and
    writes the root pose so the chosen object ends up at the requested standoff,
    then re-measures the object in the settled body frame and picks the grasp
    point. Everything after this -- reach, descent, finger closure, lift -- is
    commanded through the ordinary action terms and executed by the environment.
    """
    from scipy.spatial.transform import Rotation
    from task_b.control import LEG_TERM

    zero = torch.zeros(1, schema.total_dim, device=env.unwrapped.device)

    def object_xyz(index):
        return task_object(index).data.root_pos_w[0].detach().cpu().numpy().copy()

    def task_object(index):
        return env.unwrapped.scene[f"object_{index}"]

    def base_pose():
        position = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        return position, quaternion

    with torch.inference_mode():
        for _ in range(args.settle_calls):
            obs = env.step(zero)[0]
    base, _ = base_pose()

    if args.probe_object:
        target = int(args.probe_object)
    else:
        target = int(np.argmin([np.linalg.norm(object_xyz(i)[:2] - base[:2])
                                for i in range(1, 19)])) + 1
    obj = object_xyz(target)
    # Aim the JAWS, not just the base. The fingers slide along the gripper's local
    # +Y, so that axis must line up with the object's narrow horizontal side or the
    # jaws close across the wide side and cannot straddle it. Each type's narrow
    # side is fixed in the WORLD by its spawn quaternion (env_cfg.py): sugar has its
    # 45 mm axis along world x, mustard and banana their 58 mm / 74 mm axis along
    # world y. Park so body +Y is that axis, and place the base behind the object
    # along the perpendicular.
    narrow = np.array(NARROW_HORIZONTAL_AXIS[object_kind(target)])
    approach = np.array([-narrow[1], narrow[0]])          # body +X, perpendicular to the jaws
    parked = np.array([obj[0] - args.probe_standoff * approach[0],
                       obj[1] - args.probe_standoff * approach[1], base[2]])
    yaw = float(np.arctan2(approach[1], approach[0]))
    quaternion = np.array([np.cos(yaw / 2.), 0., 0., np.sin(yaw / 2.)])
    pose = torch.tensor(np.r_[parked, quaternion], device=env.unwrapped.device,
                        dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.unwrapped.device))
        for _ in range(args.probe_park_settle):
            obs = env.step(zero)[0]

    # Descend BEFORE measuring. The descent slides the base several centimetres, so
    # aiming the arm on the standing stance and lowering afterwards carries the jaws
    # off the object -- measured 4.4 cm off on a mustard bottle, wider than the
    # fingers' 2.2 cm half-span.
    parked_leg_q = leg_pose(robot, schema)
    delta = solve_descent(parked_leg_q, schema, args.probe_drop)
    leg_term = schema.term(LEG_TERM)
    with torch.inference_mode():
        for fraction in np.linspace(0., 1., max(2, int(2. / dt))):
            action = torch.zeros(1, schema.total_dim, device=env.unwrapped.device)
            action[0, leg_term.start:leg_term.stop] = torch.as_tensor(
                fraction * delta / leg_term.scale, device=env.unwrapped.device)
            obs = env.step(action)[0]
        for _ in range(args.probe_park_settle):
            action = torch.zeros(1, schema.total_dim, device=env.unwrapped.device)
            action[0, leg_term.start:leg_term.stop] = torch.as_tensor(
                delta / leg_term.scale, device=env.unwrapped.device)
            obs = env.step(action)[0]

    base, quaternion = base_pose()
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    obj_body = rotation.T @ (object_xyz(target) - base)

    # A wheel standing on the object ruins the experiment: it tips it over during the
    # reach and the object can never be gripped. Measured on a mustard bottle, the
    # front wheel edge reaches body x 0.563 in the descended stance.
    from task_b import leg_kinematics as legs
    leg_q = leg_pose(robot, schema)
    leg_names = list(schema.term(LEG_TERM).joint_names)
    wheel_front = max(
        legs.foot_body_xyz(corner, *[leg_q[leg_names.index(n)] for n in legs.leg_joint_names(corner)])[0]
        + legs.WHEEL_RADIUS_M for corner in legs.CORNERS)
    if obj_body[0] <= wheel_front + 0.01:
        raise RuntimeError(
            f"target object sits at body x {obj_body[0]:.4f} but the furthest-forward wheel edge reaches "
            f"{wheel_front:.4f}; the wheel would stand on it. Increase --probe_standoff to at least "
            f"{wheel_front + 0.02:.2f} m (the top-down IK stays solvable to about 0.62 m).")

    kind = object_kind(target)
    top = float(obj_body[2] + OBJECT_VERTICAL_HALF_EXTENT[kind])
    grasp_point = np.array([obj_body[0], obj_body[1], top - args.probe_clearance])
    return {
        "target_object": target, "target_kind": kind,
        "standoff_m": float(args.probe_standoff), "park_settle_steps": int(args.probe_park_settle),
        "settled_base_xyz": base.tolist(),
        "descended_base_z_m": float(base[2]),
        "object_body_xyz": obj_body.tolist(), "object_top_body_z": top,
        "front_wheel_edge_body_x_m": wheel_front,
        "grasp_point_body": grasp_point.tolist(),
        "drop_m": float(args.probe_drop), "clearance_m": float(args.probe_clearance),
        "descent_leg_delta_rad": delta.tolist(),
        "oracle": "the object's true pose is read, the base pose is written and the body is "
                  "lowered before the measurement; this is probe setup, not a policy result. The "
                  "descent is commanded through the same leg action term the policy uses.",
    }, obs, delta


def leg_pose(robot, schema) -> np.ndarray:
    """Current leg joint angles in action-term order."""
    from task_b.control import LEG_TERM
    q = robot.data.joint_pos[0].detach().cpu().numpy()
    names = list(schema.joint_names)
    return np.array([q[names.index(name)] for name in schema.term(LEG_TERM).joint_names])


def solve_descent(leg_q, schema, drop_m) -> np.ndarray:
    """Joint delta that lowers the body by drop_m with the wheels planted."""
    from task_b import leg_kinematics as legs
    from task_b.control import LEG_TERM
    by_corner = {corner: np.array([leg_q[schema.term(LEG_TERM).joint_names.index(name)]
                                   for name in legs.leg_joint_names(corner)])
                 for corner in legs.CORNERS}
    solved, residual = legs.descend_all(by_corner, drop_m)
    if residual > 1e-6:
        raise RuntimeError(f"Probe descent IK residual {residual:.2e} m at {drop_m:.4f} m")
    return np.array([solved[name.split('_')[0]][legs.LEG_LINKS.index(name.split('_')[1])]
                     for name in schema.term(LEG_TERM).joint_names]) - leg_q


def main() -> None:
    output = args.output
    started = time.monotonic()
    steps, score, reason = 0, 0.0, "max_steps"
    env, trace, recorder, video, restore_camera_views = None, None, None, None, lambda: None
    telemetry = {key: [] for key in ("step", "sim_seconds", "alpha", "action", "requested_action", "proprio", "q", "qdot", "base_xyz",
                                     "base_quat", "env_reward", "reward_raw_total", "score", "illegal_force",
                                     "termination", "reward_terms", "gripper_xyz", "object_xyz")}
    try:
        cfg = TaskBEnvB2WCfg(seed=args.seed)  # seed drives the official object layout
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.sim.use_fabric = True
        if args.camera_free:
            # Null the sensors and drop the observation groups that read them, so
            # Kit creates no render product at all. Task B's object layout, terrain,
            # robot, actions, rewards and terminations are untouched.
            for name in ("head_camera", "ee_camera", "ee_dual_camera", "lidar_sensor"):
                if getattr(cfg.scene, name, None) is not None:
                    setattr(cfg.scene, name, None)
            cfg.observations.image = None
            cfg.observations.extero = None
            # The terrain's MDL material exists only for appearance. Leaving it set
            # makes a renderer-free app drive MDL compilation, which stalls startup.
            if getattr(cfg.scene, "terrain", None) is not None:
                cfg.scene.terrain.visual_material = None
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
        probe_plan, probe_descent_delta = None, None
        if args.mode == "grasp_probe":
            probe_plan, obs, probe_descent_delta = park_and_measure(env, robot, schema, args, dt)
        visual_mode = args.mode in ("visual_approach", "first_reach", "reach_probe")
        if args.mode in ("first_reach", "reach_probe"):
            from task_b.first_reach import FirstReachPolicy
            policy = FirstReachPolicy(schema, observed_names, defaults, dt=dt,
                                      settle_calls=args.settle_calls, ramp_calls=args.ramp_calls,
                                      reach_only=args.mode == "reach_probe", forward_cmd=args.reach_forward,
                                      turn_cap=args.reach_turn_cap, standoff=args.reach_standoff,
                                      turn_gain=args.reach_turn_gain, lowering_m=args.reach_lowering,
                                      grasp=args.reach_grasp)
        elif args.mode == "stance_descend":
            from task_b.stance_descend import StanceDescendPolicy
            policy = StanceDescendPolicy(schema, observed_names, defaults, dt=dt,
                                         settle_calls=args.settle_calls, drop_max=args.stance_drop_max,
                                         drop_rate=args.stance_drop_rate, hold_s=args.stance_hold_s,
                                         grid_step=args.stance_grid_step)
        elif args.mode == "grasp_probe":
            from task_b.grasp_probe import GraspProbePolicy
            policy = GraspProbePolicy(schema, observed_names, defaults, dt=dt,
                                      grasp_point_body=probe_plan["grasp_point_body"],
                                      descent_delta_rad=probe_descent_delta,
                                      preload_m=args.probe_preload)
        elif args.mode == "visual_approach":
            from task_b.visual_approach import VisualApproachPolicy
            policy = VisualApproachPolicy(schema, observed_names, defaults, dt=dt,
                                          settle_calls=args.settle_calls, ramp_calls=args.ramp_calls,
                                          **({"use_head": True} if args.vision_head else {}))
        else:
            policy = control.BootstrapPolicy(schema, args.mode, settle_calls=args.settle_calls,
                                             ramp_calls=args.ramp_calls, wheel_cmd=args.wheel_cmd,
                                             crouch_fraction=args.crouch_fraction)
        stance_reference = None
        if args.stance_profile != "off":
            from task_b.stance_reference import StanceReference
            hard_limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
            stance_reference = StanceReference(schema, observed_names, dt=dt, profile=args.stance_profile,
                                               settle_calls=args.settle_calls,
                                               hard_joint_pos_limits=dict(zip(schema.joint_names, hard_limits)))
        stance_holder = None
        if args.stance_hold:
            from task_b.stance_hold import StanceHold
            stance_holder = StanceHold(schema, observed_names, dt=dt, settle_calls=args.settle_calls)
        brake_holder = None
        brake_states = ("BRAKE", "REACH_READY", "REACH", "REACH_VISUAL_HOLD")
        if args.brake_wheel_hold:
            from task_b.brake_wheel_hold import BrakeWheelHold
            brake_holder = BrakeWheelHold(dt=dt)
            wheel_names = schema.term(control.WHEEL_TERM).joint_names
            brake_observation_ids = np.array([observed_names.index(name) for name in wheel_names])
            brake_joint_defaults = np.array([defaults[name] for name in wheel_names])
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
            "telemetry_timing": {
                "proprio_q_qdot_base_gripper_object_and_contact": "pre_step",
                "env_reward_reward_terms_and_termination": "post_step; terminal term values captured before official reset",
                "env_reward": "unmodified scalar returned by env.step; reward_raw_total = env_reward / step_dt",
                "scoring_event_state": "post_step_before_any_reset or terminal_pre_reset; see each event",
            },
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
                             "policy_inputs": ("proprio + head/ee RGB-D" if args.vision_head or args.mode in ("first_reach", "reach_probe") else "proprio + ee RGB-D")
                             if visual_mode else "proprio only"},
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
            "wheel_action_gain": args.wheel_action_gain,
            "stance_hold": stance_holder.describe() if stance_holder else None,
            "brake_wheel_hold": brake_holder.describe() if brake_holder else None,
            "brake_wheel_hold_integration": {
                "active_states": brake_states,
                "inputs": "public proprio wheel q/qdot and policy phase only",
                "composition": "after ordinary wheel gain and leg controllers, replace wheel slice by physical speed / schema scale; then apply optional stability controller",
                "no_second_wheel_gain": True,
            } if brake_holder else None,
            "stance_reference": stance_reference.describe() if stance_reference else None,
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
        first_positive_step, scoring_events = None, []
        policy_stop_record = None
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
            if args.mode in ("first_reach", "reach_probe"):
                policy.pause_for_stance = bool(stance_reference and stance_reference.requires_wheel_stop)
            if visual_mode:
                images = {key: value[0].detach().cpu().numpy()
                          for key, value in obs["image"].items()
                          if args.vision_head or args.mode != "visual_approach" or key.startswith("ee_")}
                requested_action = policy.act(proprio, images)
            else:
                requested_action = policy.act(proprio)
            if getattr(policy, "done_reason", None) is not None:
                if brake_holder:
                    brake_holder.release()
                reason = "policy_stop:" + str(policy.done_reason)
                policy_stop_record = jsonable({
                    "before_step": step + 1, "last_completed_step": steps,
                    "reason": policy.done_reason, "policy": policy.describe(),
                    "proprio": proprio, "proprio_timing": "same_public_observation_used_by_the_stopping_policy_call",
                    "state": state, "state_timing": "before_unexecuted_step_after_last_completed_step",
                })
                break
            drive_action = requested_action.copy()
            wheel_term = schema.term(control.WHEEL_TERM)
            drive_action[wheel_term.start:wheel_term.stop] *= args.wheel_action_gain
            if stance_reference:
                drive_action = stance_reference.apply(drive_action, proprio)
                if stance_reference.requires_wheel_stop:
                    drive_action[wheel_term.start:wheel_term.stop] = 0.
            if stance_holder:
                drive_action = stance_holder.apply(drive_action, proprio)
            if brake_holder:
                if policy.state in brake_states:
                    wheel_q = proprio[12+brake_observation_ids] + brake_joint_defaults
                    wheel_qdot = proprio[36+brake_observation_ids]
                    brake_holder.engage(wheel_q)  # repeated engage preserves the initial anchor
                    common_speed = brake_holder.update(wheel_q, wheel_qdot)
                    # The normal gain has already been applied above. The
                    # holder outputs physical rad/s, so divide only by scale.
                    drive_action[wheel_term.start:wheel_term.stop] = common_speed / wheel_term.scale
                else:
                    brake_holder.release()
            action = stabilizer.apply(drive_action, proprio) if stabilizer else drive_action
            tensor = torch.as_tensor(action, device=task.device, dtype=torch.float32).unsqueeze(0)
            if tensor.shape != (1, schema.total_dim) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Expected a finite action of shape (1, {schema.total_dim}), got {tensor.shape}")
            with torch.inference_mode(), recorder.step():
                obs, reward, terminated, truncated, _ = env.step(tensor)
            steps = step + 1

            # Official reward is dt-scaled; dividing by step_dt gives the raw
            # weighted term sum per step, which is what the official runner sums.
            env_reward = float(reward.item())
            reward_raw_total = env_reward / dt
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
            if reward_raw_total > 0:
                first_positive_step = first_positive_step or steps
                if terminated_flag or truncated_flag:
                    snapshot = (terminal_pre_reset or {}).get("snapshot")
                    evidence = snapshot.get("state") if isinstance(snapshot, dict) else None
                    evidence_timing = "terminal_pre_reset" if evidence is not None else "missing_terminal_capture"
                else:
                    evidence = sample_state(task, illegal)
                    evidence_timing = "post_step_before_any_reset"
                scoring_events.append(jsonable({"step": steps, "reward_terms": reward_step,
                                                "env_reward": env_reward, "reward_raw_total": reward_raw_total,
                                                "state": evidence, "state_timing": evidence_timing,
                                                "capture_error": (terminal_pre_reset or {}).get("observer_error"),
                                                "termination_flags": term_flags}))
                (output / "scoring_events.json").write_text(json.dumps(scoring_events, indent=2)+"\n")
                if not (terminated_flag or truncated_flag):
                    frames[f"score_{steps:05d}"] = save_rgb(obs, output, f"score_{steps:05d}")
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
            telemetry["gripper_xyz"].append(state["gripper_xyz"])
            telemetry["object_xyz"].append(state["object_xyz"])
            telemetry["env_reward"].append(env_reward)
            telemetry["reward_raw_total"].append(reward_raw_total)
            telemetry["score"].append(score)
            telemetry["illegal_force"].append(state["illegal_force"])
            telemetry["termination"].append([term_flags[name] for name in termination_names])
            telemetry["reward_terms"].append([reward_step.get(name, 0.0) for name in reward_names])

            row = {
                "step": steps, "sim_seconds": round(steps * dt, 4), "alpha": policy.alpha,
                "env_reward": env_reward, "reward_raw_total": reward_raw_total, "score": score,
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
                "diagnostic_min_gripper_object_distance_m": float(np.min(np.linalg.norm(state["object_xyz"]-state["gripper_xyz"], axis=1))),
                "stability_debug": stabilizer.debug if stabilizer else None,
                "stance_debug": stance_holder.debug if stance_holder else None,
                "brake_wheel_hold_debug": brake_holder.debug if brake_holder else None,
                "stance_reference_debug": stance_reference.debug if stance_reference else None,
            }
            row = jsonable(row)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            if steps % 100 == 0 or reward_raw_total or terminated_flag or truncated_flag:
                print("TASK_B_PROGRESS " + json.dumps(row), flush=True)
            if args.rgb_interval and steps % args.rgb_interval == 0 and not (terminated_flag or truncated_flag):
                frames[f"step_{steps:05d}"] = save_rgb(obs, output, f"step_{steps:05d}")
            if terminated_flag or truncated_flag:
                reason = "terminated" if terminated_flag else "truncated"
                # The official env already reset joints for this env, so this is
                # the post-reset state; the base pose is untouched because Task B
                # has no root-state reset event.
                final_state = sample_state(task, illegal)
                break
            if first_positive_step and args.score_hold_steps > 0 and steps >= first_positive_step + args.score_hold_steps:
                reason = "positive_score_observation_window_complete"
                break

        if terminated_flag or truncated_flag:
            terminal_snapshot = (terminal_pre_reset or {}).get("snapshot")
            final_before_close = terminal_snapshot.get("state") if isinstance(terminal_snapshot, dict) else None
            final_timing = "terminal_pre_reset" if final_before_close is not None else "missing_terminal_capture"
        else:
            final_before_close = sample_state(task, illegal)
            final_timing = "post_step_before_any_reset"
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
            "headless": True, "enable_cameras": not args.camera_free,
            "camera_free": args.camera_free,
            "camera_free_scope": ("scene head_camera/ee_camera/ee_dual_camera and lidar_sensor set to None, "
                                  "observations.image and observations.extero group set to None, and the "
                                  "terrain visual_material set to None; object layout, terrain geometry, "
                                  "robot, actions, rewards and terminations untouched"
                                  if args.camera_free else None),
            "camera_free_launcher_patch": ("AppLauncher._SIM_APP_CFG_TYPES['create_new_stage']=[bool] added "
                                           "in-process so the create_new_stage=False Isaac Lab already "
                                           "computes survives AppLauncher's config filter; without it a "
                                           "renderer-free app spins in SimulationApp._wait_for_viewport. "
                                           "omni.usd.get_context().new_stage() is then called once to create "
                                           "the stage SimulationApp would otherwise have created"
                                           if args.camera_free else None),
            "use_fabric": True,
            "experience": selected_experience or None,
            "camera_fabric_compatibility": "task_a.tools.d1g2_taska_camera_compat.prepare_camera_views "
                                           "(read-only recursive world-pose reader; preserves moving camera inheritance)",
            "action_spec": None, "apply_safe_action_spec": "called with no participant spec",
            "steps": steps, "max_steps": args.max_steps, "sim_seconds": steps * float(task.step_dt),
            "wall_seconds": time.monotonic() - started, "stop_reason": reason,
            "policy_stop_record": policy_stop_record,
            "terminated": terminated_flag, "truncated": truncated_flag,
            "active_termination_terms_at_stop": active_terms,
            "terminated_during_settle": bool((terminated_flag or truncated_flag)
                                             and steps <= args.settle_calls),
            "score_definition": "sum over steps of env reward / step_dt, matching the official "
                                "scripts/play_atec_task.py accumulation",
            "score_raw_total": score,
            "first_positive_step": first_positive_step, "scoring_events": scoring_events,
            "reward_term_totals_raw": term_totals,
            "reward_term_total_definition": "per step, weighted term value with dt removed "
                                            "(RewardManager.get_active_iterable_terms), summed over steps",
            "reward_term_sum_vs_env_reward_max_abs_error": checksum,
            "final_reward_terms": reward_step,
            "base_motion": displacement,
            "observation_joint_mapping_max_abs_error": observation_q_error_max,
            "terminal_pre_reset": terminal_pre_reset,
            "final_state_before_close": jsonable(final_before_close),
            "final_state_timing": final_timing,
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
            "wheel_action_gain": args.wheel_action_gain,
            "stance_hold": stance_holder.describe() if stance_holder else None,
            "brake_wheel_hold": brake_holder.describe() if brake_holder else None,
            "stance_reference": stance_reference.describe() if stance_reference else None,
            "stability_controller": stabilizer.describe() if stabilizer else None,
            "grasp_probe": probe_plan,
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
