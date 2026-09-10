"""Replay captured Task A RGB-D on CPU, with sensor-only settling bootstrap.

Example::

    PYTHONNOUSERSITE=1 python tools/replay_d1g2_taska_rgbd.py \
        --capture_dir logs/d1g2_taska_20260909/rgbd_observation_probe_02 \
        --bootstrap_step 100 --output logs/rgbd_replay_01

The capture must contain sensors/frame_XXXXXX.npz starting at step zero.
Only rgb/depth/K/proprio enter estimation. Optional diagnostic poses are read
in a separate evaluation pass, after all estimates are finalized. Captured
proprio is usually only 10 Hz; it does not reproduce a live 50 Hz bootstrap.
The fixed camera pitch is read from camera_pitch_deg in the first NPZ and
checked against every later frame. Legacy captures without this field use 0.
K is used as saved by default. For older RTX raster intrinsics, explicitly
pass --raster_intrinsics to subtract 0.5 from the principal point. Captures
marked as already using OpenCV pixel centres reject that option. Same-frame
projected gravity constrains roll/pitch after each accepted visual update.
The output directory must not already exist. No simulator or GPU is started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.d1g2_taska_navigation import D1G2ProprioNavigator
from tools.d1g2_taska_rgbd import RGBDOdometry


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _rotation_from_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
        raise ValueError("Diagnostic quaternion must be a finite nonzero w,x,y,z vector")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _capture_files(capture_dir, bootstrap_step):
    sensor_dir = capture_dir / "sensors"
    files = {}
    for path in sensor_dir.glob("frame_*.npz"):
        match = re.fullmatch(r"frame_(\d+)\.npz", path.name)
        if match:
            step = int(match.group(1))
            if step in files:
                raise ValueError(f"Duplicate frame step {step}")
            files[step] = path
    if 0 not in files:
        raise ValueError(f"{sensor_dir} must include frame_000000.npz to bootstrap from the original start")
    if bootstrap_step not in files:
        raise ValueError(f"Requested bootstrap step {bootstrap_step} is absent from {sensor_dir}")
    if max(files) <= bootstrap_step:
        raise ValueError("Capture must contain at least one frame after the bootstrap step")
    return dict(sorted(files.items()))


def _mount_pitch_from_sample(sample):
    if "camera_pitch_deg" not in sample:
        return 0.
    value = np.asarray(sample["camera_pitch_deg"])
    if value.size != 1 or not np.issubdtype(value.dtype, np.number):
        raise ValueError("camera_pitch_deg must be a numeric scalar")
    pitch = float(value.reshape(-1)[0])
    if not math.isfinite(pitch):
        raise ValueError("camera_pitch_deg must be finite")
    return pitch


def _intrinsics_from_sample(sample, raster_intrinsics=False):
    """Make an explicit raster-to-OpenCV conversion, never apply it twice."""
    K = np.asarray(sample["K"], dtype=np.float64).copy()
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be a finite 3x3 matrix")
    convention = "legacy_unlabelled"
    if "intrinsics_pixel_convention" in sample:
        value = np.asarray(sample["intrinsics_pixel_convention"])
        if value.size != 1:
            raise ValueError("intrinsics_pixel_convention must be scalar")
        convention = value.reshape(-1)[0]
        if isinstance(convention, bytes):
            convention = convention.decode("utf-8")
        convention = str(convention)
        if convention not in ("opencv_pixel_centres", "raster"):
            raise ValueError(f"Unsupported intrinsics_pixel_convention: {convention}")
    saved_correction = None
    if "principal_point_correction_px" in sample:
        value = np.asarray(sample["principal_point_correction_px"])
        if value.size != 1:
            raise ValueError("principal_point_correction_px must be scalar")
        saved_correction = float(value.reshape(-1)[0])
        if not math.isfinite(saved_correction) or saved_correction not in (0., -.5):
            raise ValueError("Unsupported principal_point_correction_px")
    raw_difference = None
    if "K_raw" in sample:
        raw_K = np.asarray(sample["K_raw"], dtype=np.float64)
        if raw_K.shape != (3, 3) or not np.isfinite(raw_K).all():
            raise ValueError("K_raw must be a finite 3x3 matrix")
        delta = K - raw_K
        half_pixel_delta = np.zeros((3, 3))
        half_pixel_delta[:2, 2] = -.5
        if np.allclose(delta, half_pixel_delta, rtol=0., atol=1e-6):
            raw_difference = -.5
        elif np.allclose(delta, 0., rtol=0., atol=1e-6):
            raw_difference = 0.
        else:
            raise ValueError("K differs from K_raw by an unsupported transform")
        if saved_correction is not None and saved_correction != raw_difference:
            raise ValueError("K/K_raw contradict principal_point_correction_px")
    evidence = [value for value in (saved_correction, raw_difference) if value is not None]
    if convention == "opencv_pixel_centres" and 0. in evidence:
        raise ValueError("OpenCV intrinsics marker contradicts uncorrected raster metadata")
    if convention == "raster" and -.5 in evidence:
        raise ValueError("Raster intrinsics marker contradicts corrected K metadata")
    already_corrected = convention == "opencv_pixel_centres" or -.5 in evidence
    if raster_intrinsics and already_corrected:
        raise ValueError("--raster_intrinsics would repeat a half-pixel correction already present in this capture")
    if raster_intrinsics:
        K[:2, 2] -= .5
    return K, {
        "capture_intrinsics_pixel_convention": convention,
        "capture_principal_point_correction_px": saved_correction,
        "capture_K_minus_K_raw_principal_point_px": raw_difference,
        "capture_intrinsics_already_corrected": already_corrected,
        "replay_raster_intrinsics": bool(raster_intrinsics),
        "replay_principal_point_correction_px": -.5 if raster_intrinsics else 0.,
    }


def _estimate(files, bootstrap_step, physics_dt, mount_pitch_deg, raster_intrinsics=False):
    """This pass deliberately never accesses any diagnostic array."""
    navigator = D1G2ProprioNavigator()
    odometry = RGBDOdometry(mount_pitch_deg=mount_pitch_deg)
    records = []
    previous_step = None
    initial_state = None
    bootstrap_intervals = []
    initial_intrinsics_metadata = None
    start = time.perf_counter()
    for step, path in files.items():
        interval = physics_dt if previous_step is None else (step - previous_step) * physics_dt
        with np.load(path, allow_pickle=False) as sample:
            sample_pitch = _mount_pitch_from_sample(sample)
            if not math.isclose(sample_pitch, mount_pitch_deg, rel_tol=0., abs_tol=1e-6):
                raise ValueError(f"Camera mount pitch changed at step {step}: {sample_pitch} != {mount_pitch_deg}")
            K, intrinsics_metadata = _intrinsics_from_sample(sample, raster_intrinsics)
            if initial_intrinsics_metadata is None:
                initial_intrinsics_metadata = intrinsics_metadata
            elif intrinsics_metadata != initial_intrinsics_metadata:
                raise ValueError(f"Camera intrinsics convention/correction metadata changed at step {step}")
            if step <= bootstrap_step:
                navigator.command(sample["proprio"], 0., interval)
                if previous_step is not None:
                    bootstrap_intervals.append(interval)
            if step == bootstrap_step:
                result = odometry.initialize_from_proprio(
                    sample["rgb"], sample["depth"], K, sample["proprio"],
                    navigator.estimated_xy, navigator.estimated_yaw,
                )
                if not result["initialized"]:
                    raise RuntimeError(f"Bootstrap image at step {step} is unusable: {result['reason']}")
                initial_state = odometry.state_dict()
            elif step > bootstrap_step:
                result = odometry.update(sample["rgb"], sample["depth"], K, interval,
                                         projected_gravity=sample["proprio"][9:12])
            else:
                previous_step = step
                continue
        records.append({"step": step, "simulation_time_s": step * physics_dt,
                        "source_frame": path.name, **result})
        previous_step = step
    accepted = sum(record["accepted"] for record in records[1:])
    summary = {
        **initial_intrinsics_metadata,
        "projected_gravity_correction": True,
        "estimation_wall_seconds": time.perf_counter() - start,
        "capture_frames": len(files), "replay_frames": len(records),
        "initialization_step": bootstrap_step,
        "initialization_time_s": bootstrap_step * physics_dt,
        "initialization_proprio_samples": sum(step <= bootstrap_step for step in files),
        "initialization_proprio_sample_hz_mean": (
            float(1. / np.mean(bootstrap_intervals)) if bootstrap_intervals else None
        ),
        "initialization_proprio_interval_s_min": min(bootstrap_intervals, default=None),
        "initialization_proprio_interval_s_max": max(bootstrap_intervals, default=None),
        "live_runner_recommended_proprio_sample_hz": 50,
        "initialization_limit": (
            "Saved proprio commonly samples at 10 Hz and undersamples settling; "
            "bootstrap XY/yaw are estimates and may differ from live 50 Hz integration. "
            "Z is a relative translation datum. No later pose reset occurs."
        ),
        "accepted_frames": accepted,
        "rejected_frames": len(records) - 1 - accepted,
        "reason_counts": dict(Counter(record["reason"] for record in records)),
        "first_rejected_after_initialization": next(
            ({"step": record["step"], "reason": record["reason"]}
             for record in records[1:] if not record["accepted"]), None
        ),
        "initial_estimated_xy": initial_state["estimated_xy"],
        "initial_estimated_yaw": initial_state["estimated_yaw"],
        "initial_estimated_rotation": initial_state["estimated_rotation"],
        "initial_estimated_position": initial_state["estimated_position"],
        "final_estimated_xy": records[-1]["estimated_xy"],
        "final_estimated_yaw": records[-1]["estimated_yaw"],
        "final_state": odometry.state_dict(),
    }
    return records, initial_state, summary


def _read_diagnostic_pose(path):
    """Optional evaluation data; called only after _estimate has completed."""
    with np.load(path, allow_pickle=False) as sample:
        for xyz_key, quat_key in (("diagnostic_true_xyz", "diagnostic_true_quat"),
                                  ("diag_true_xyz", "diag_true_quat")):
            if xyz_key in sample and quat_key in sample:
                xyz = np.asarray(sample[xyz_key], dtype=np.float64)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError(f"Invalid diagnostic position in {path}")
                return xyz, _rotation_from_quaternion(sample[quat_key])
    return None


def _evaluate(files, records, initial_state):
    """Annotate fixed estimates with errors without feeding them back."""
    diagnostics = {step: _read_diagnostic_pose(files[step]) for step in
                   {0, *(record["step"] for record in records)}}
    if diagnostics[0] is None or any(diagnostics[record["step"]] is None for record in records):
        return {"diagnostics_available": False, "evaluation_status": "unavailable",
                "evaluation_note": "Diagnostic poses are optional; complete poses were not present for evaluation."}
    start_position, start_rotation = diagnostics[0]
    for record in records:
        position, rotation = diagnostics[record["step"]]
        relative_position = start_rotation.T @ (position - start_position)
        relative_rotation = start_rotation.T @ rotation
        true_yaw = math.atan2(relative_rotation[1, 0], relative_rotation[0, 0])
        record.update(
            diagnostic_true_relative_xy=relative_position[:2].tolist(),
            diagnostic_true_relative_yaw=true_yaw,
            diagnostic_xy_error=float(np.linalg.norm(np.asarray(record["estimated_xy"]) - relative_position[:2])),
            diagnostic_yaw_error=_wrap_angle(record["estimated_yaw"] - true_yaw),
        )
    first, last = records[0], records[-1]
    bootstrap_position, bootstrap_rotation = diagnostics[first["step"]]
    final_position, final_rotation = diagnostics[last["step"]]
    estimate_initial_rotation = np.asarray(initial_state["estimated_rotation"])
    estimate_relative_translation = estimate_initial_rotation.T @ (
        np.asarray(last["estimated_position"]) - np.asarray(initial_state["estimated_position"])
    )
    true_relative_translation = bootstrap_rotation.T @ (final_position - bootstrap_position)
    estimate_relative_rotation = estimate_initial_rotation.T @ np.asarray(last["estimated_rotation"])
    true_relative_rotation = bootstrap_rotation.T @ final_rotation
    return {
        "diagnostics_available": True, "evaluation_status": "completed",
        "initial_diagnostic_true_xy": first["diagnostic_true_relative_xy"],
        "initial_diagnostic_true_yaw": first["diagnostic_true_relative_yaw"],
        "initial_xy_error": first["diagnostic_xy_error"],
        "initial_yaw_error": first["diagnostic_yaw_error"],
        "final_diagnostic_true_xy": last["diagnostic_true_relative_xy"],
        "final_diagnostic_true_yaw": last["diagnostic_true_relative_yaw"],
        "final_xy_error": last["diagnostic_xy_error"],
        "final_yaw_error": last["diagnostic_yaw_error"],
        "final_y_error_signed": last["estimated_xy"][1] - last["diagnostic_true_relative_xy"][1],
        "post_initialization_visual_translation_in_bootstrap_body": estimate_relative_translation.tolist(),
        "post_initialization_diagnostic_translation_in_bootstrap_body": true_relative_translation.tolist(),
        "post_initialization_translation_error_independent_of_initial_pose_estimate": float(
            np.linalg.norm(estimate_relative_translation - true_relative_translation)
        ),
        "post_initialization_rotation_error_independent_of_initial_pose_estimate": float(
            np.linalg.norm(cv2.Rodrigues(estimate_relative_rotation.T @ true_relative_rotation)[0])
        ),
        "post_initialization_yaw_drift": _wrap_angle(
            (last["estimated_yaw"] - first["estimated_yaw"])
            - (last["diagnostic_true_relative_yaw"] - first["diagnostic_true_relative_yaw"])
        ),
        "xy_error_rmse_all_replayed_frames": float(np.sqrt(np.mean([
            record["diagnostic_xy_error"] ** 2 for record in records
        ]))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--bootstrap_step", type=int, default=100)
    parser.add_argument("--control_dt", type=float, default=.02, help="Seconds per frame filename step (Task A control step: .02)")
    parser.add_argument("--opencv_threads", type=int, default=1)
    parser.add_argument("--raster_intrinsics", action="store_true",
                        help="Explicitly convert legacy RTX raster K to OpenCV by principal point -0.5; rejects already corrected captures")
    args = parser.parse_args(argv)
    if args.bootstrap_step < 0 or not math.isfinite(args.control_dt) or args.control_dt <= 0 or args.opencv_threads < 1:
        parser.error("bootstrap_step must be nonnegative; control_dt and opencv_threads must be positive")
    capture_dir = args.capture_dir.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    files = _capture_files(capture_dir, args.bootstrap_step)
    with np.load(files[0], allow_pickle=False) as first_frame:
        mount_pitch_deg = _mount_pitch_from_sample(first_frame)
        mount_source = "first_npz_camera_pitch_deg" if "camera_pitch_deg" in first_frame else "legacy_default_zero"
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(args.opencv_threads)
    cv2.setRNGSeed(20260909)
    sources = [Path(__file__).resolve(), ROOT / "tools/d1g2_taska_rgbd.py", ROOT / "tools/d1g2_taska_navigation.py"]
    summary = {
        "schema_version": 1, "status": "running", "capture_dir": str(capture_dir), "output": str(output),
        "device": "cpu", "simulation_started": False, "full_course_tested": False,
        "scope": "Sensor-only replay and optional diagnostic comparison; not proof of Task A completion",
        "estimator_inputs": ["rgb", "depth", "K", "proprio", "fixed_camera_pitch_configuration", "frame_index_and_control_dt"],
        "camera_mount_pitch_deg": mount_pitch_deg, "camera_mount_source": mount_source,
        "replay_raster_intrinsics": args.raster_intrinsics,
        "replay_principal_point_correction_px": -.5 if args.raster_intrinsics else 0.,
        "projected_gravity_correction": True,
        "truth_used_for_estimation": False,
        "diagnostic_access_order": "Read only in a separate pass after all estimates are finalized",
        "control_step_dt_s": args.control_dt,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        "opencv_threads": args.opencv_threads, "opencv_rng_seed": 20260909,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    try:
        records, initial_state, estimation = _estimate(
            files, args.bootstrap_step, args.control_dt, mount_pitch_deg, args.raster_intrinsics)
        summary.update(estimation)
        summary.update(_evaluate(files, records, initial_state))
        summary["status"] = "finished"
        with (output / "trace.jsonl").open("x") as trace:
            for record in records:
                trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        _write_json(output / "replay_summary.json", summary)
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
        _write_json(output / "replay_summary.json", summary)
        raise
    print(json.dumps({key: summary.get(key) for key in (
        "status", "output", "accepted_frames", "rejected_frames", "reason_counts",
        "initial_xy_error", "initial_yaw_error", "final_xy_error", "final_y_error_signed", "final_yaw_error",
        "post_initialization_translation_error_independent_of_initial_pose_estimate", "post_initialization_yaw_drift",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
