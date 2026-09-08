#!/usr/bin/env python3
"""Compare Task E result.json and per-step telemetry.npz without simulation.

Example:
    python tools/task_e/compare_motion.py BASELINE_RUN CANDIDATE_RUN \
        --output-dir logs/task_e_motion_comparison

Only the first six (revolute arm) joints enter motion statistics. The two
prismatic finger joints are excluded. Derivatives are computed on adjacent
samples BEFORE grouping, and assigned to the destination sample's labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_run(directory: Path, terminal_sample: str) -> dict[str, Any]:
    result = json.loads((directory / "result.json").read_text())
    telemetry_path = directory / result.get("telemetry", "telemetry.npz")
    with np.load(telemetry_path, allow_pickle=False) as saved:
        required = {"qpos", "qvel", "action", "score", "state", "object_id", "dt"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"{telemetry_path}: missing fields {sorted(missing)}")
        arrays = {key: saved[key].copy() for key in required}
    raw_dt = np.asarray(arrays.pop("dt"))
    if raw_dt.size != 1:
        raise ValueError(f"{telemetry_path}: dt must be one constant sample interval")
    dt = float(raw_dt.reshape(-1)[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{telemetry_path}: dt must be finite and positive")
    qpos, qvel = np.asarray(arrays["qpos"], float), np.asarray(arrays["qvel"], float)
    if qpos.ndim != 2 or qpos.shape[1] < 6 or qvel.shape != qpos.shape:
        raise ValueError(f"{telemetry_path}: qpos/qvel must share shape (N, >=6)")
    count = len(qpos)
    if count == 0:
        raise ValueError(f"{telemetry_path}: no executed-step samples")
    for key, values in arrays.items():
        if values.ndim == 0 or len(values) != count:
            raise ValueError(f"{telemetry_path}: {key} does not have {count} rows")
    for key in ("qpos", "qvel", "action", "score"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"{telemetry_path}: non-finite {key} values")
    for key in ("score", "state", "object_id"):
        if arrays[key].shape != (count,):
            raise ValueError(f"{telemetry_path}: {key} must have shape (N,)")
    action = arrays["action"]
    if action.ndim != 2 or action.shape[1] < 6:
        raise ValueError(f"{telemetry_path}: action must have shape (N, >=6)")
    objects = np.asarray(arrays["object_id"], dtype=int)
    if not np.all(objects == arrays["object_id"]):
        raise ValueError(f"{telemetry_path}: object_id must contain integers")
    states = arrays["state"].astype(str)
    states[states == ""] = "UNLABELLED"
    metadata_path = directory / "environment_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    joint_names = metadata.get("joint_names", [f"joint{i}" for i in range(1, 9)])[:6]
    if len(joint_names) != 6:
        raise ValueError(f"{metadata_path}: need six arm joint names")

    # Post-step physical buffers may expose an auto-reset or retain terminal
    # physics, depending on the environment/readback. A done flag alone cannot
    # distinguish them. Detect a return to the declared default with zero
    # velocity and a position jump incompatible with adjacent measured speeds.
    reset_signature = False
    default_q = np.asarray(metadata.get("default_joint_pos", []), dtype=float).reshape(-1)
    if count >= 2 and len(default_q) >= 6 and result.get("stop_reason") in ("terminated", "time_limit"):
        jump = np.abs(qpos[-1, :6] - qpos[-2, :6])
        adjacent_speed = np.maximum(np.abs(qvel[-1, :6]), np.abs(qvel[-2, :6]))
        reset_signature = bool(
            np.max(np.abs(qpos[-1, :6] - default_q[:6])) <= .005
            and np.max(np.abs(qvel[-1, :6])) <= .005
            and np.any((jump >= .05) & (jump / dt > 5. * adjacent_speed + .5))
        )
    drop_terminal = terminal_sample == "exclude" or (terminal_sample == "auto" and reset_signature)
    valid = np.ones(count, dtype=bool)
    if drop_terminal:
        valid[-1] = False
    notes = []
    if drop_terminal:
        cause = "detected default-pose reset signature" if reset_signature else "explicit --terminal-sample exclude"
        notes.append(f"Final physical sample excluded ({cause}); its duration is retained.")
    elif reset_signature:
        notes.append("Final sample matches an auto-reset signature but is included by explicit override.")
    if count != result.get("steps"):
        notes.append(f"Telemetry has {count} samples; result.json reports {result.get('steps')} steps.")
    recorded_duration = result.get("sim_seconds")
    if recorded_duration is not None and not np.isclose(count * dt, recorded_duration, rtol=1e-7, atol=1e-7):
        notes.append("Telemetry duration differs from result.json sim_seconds; both values are reported.")
    if not np.isclose(dt, .02, rtol=1e-7, atol=1e-9):
        notes.append(f"Sample rate is {1 / dt:g} Hz, not the usual local 50 Hz.")
    return dict(directory=str(directory.resolve()), result=result, dt=dt,
                qpos=qpos[:, :6], qvel=qvel[:, :6], state=states,
                object_id=objects, valid=valid, joint_names=joint_names,
                terminal_physical_sample_excluded=drop_terminal,
                terminal_reset_signature_detected=reset_signature, notes=notes)


def _metrics(run: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    """Aggregate globally adjacent intervals; never difference filtered rows."""
    physical = selected & run["valid"]
    edge_mask = selected[1:] & run["valid"][1:] & run["valid"][:-1]
    displacement = np.diff(run["qpos"], axis=0)[edge_mask]
    acceleration = np.diff(run["qvel"], axis=0)[edge_mask] / run["dt"]
    absolute_speed = np.abs(run["qvel"][physical])
    path = np.sum(np.abs(displacement), axis=0)
    speed_count, edge_count = len(absolute_speed), len(displacement)
    peak = np.max(absolute_speed, axis=0) if speed_count else np.full(6, np.nan)
    p95 = np.percentile(absolute_speed, 95, axis=0) if speed_count else np.full(6, np.nan)
    rms = np.sqrt(np.mean(acceleration ** 2, axis=0)) if edge_count else np.full(6, np.nan)

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "samples": int(np.sum(selected)),
        "duration_s": float(np.sum(selected) * run["dt"]),
        "physical_samples": int(np.sum(physical)),
        "adjacent_intervals": edge_count,
        "arm_path_rad": float(np.sum(path)),
        "wrist_joint6_path_rad": float(path[5]),
        "arm_speed_peak_abs_rad_s": float(np.max(absolute_speed)) if speed_count else None,
        "arm_speed_p95_max_joint_rad_s": float(np.percentile(np.max(absolute_speed, axis=1), 95)) if speed_count else None,
        "wrist_joint6_speed_peak_abs_rad_s": finite(peak[5]),
        "wrist_joint6_speed_p95_abs_rad_s": finite(p95[5]),
        "arm_acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))) if edge_count else None,
        "wrist_joint6_acceleration_rms_rad_s2": finite(rms[5]),
        "per_joint": {
            name: {"path_rad": float(path[i]), "speed_peak_abs_rad_s": finite(peak[i]),
                   "speed_p95_abs_rad_s": finite(p95[i]), "acceleration_rms_rad_s2": finite(rms[i])}
            for i, name in enumerate(run["joint_names"])
        },
    }


def _summarize(run: dict[str, Any]) -> dict[str, Any]:
    states, objects = run["state"], run["object_id"]
    count = len(states)
    by_state = {state: _metrics(run, states == state) for state in dict.fromkeys(states)}
    by_object = {str(int(obj)): _metrics(run, objects == obj) for obj in dict.fromkeys(objects)}
    by_object_state = {
        f"object_{int(obj)}:{state}": _metrics(run, (objects == obj) & (states == state))
        for obj, state in dict.fromkeys(zip(objects, states))
    }
    boundaries = np.r_[0, np.flatnonzero((states[1:] != states[:-1]) | (objects[1:] != objects[:-1])) + 1, count]
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = np.zeros(count, dtype=bool)
        selected[start:end] = True
        segments.append({"state": str(states[start]), "object_id": int(objects[start]),
                         "first_step": int(start + 1), "last_step": int(end),
                         "start_time_s": float(start * run["dt"]), "end_time_s": float(end * run["dt"]),
                         **_metrics(run, selected)})
    result = run["result"]
    return {
        "directory": run["directory"],
        "result": {key: result.get(key) for key in (
            "solution", "solution_sha256", "implementation", "seed", "score", "full_score", "steps", "sim_seconds",
            "wall_seconds", "stop_reason", "action_spec", "task_physics_modified", "render_preset", "use_fabric")},
        "sample_dt_s": run["dt"], "sample_rate_hz": 1. / run["dt"],
        "arm_joint_names": run["joint_names"],
        "terminal_physical_sample_excluded": run["terminal_physical_sample_excluded"],
        "terminal_reset_signature_detected": run["terminal_reset_signature_detected"],
        "notes": run["notes"], "overall": _metrics(run, np.ones(count, dtype=bool)),
        "by_state": by_state, "by_object": by_object, "by_object_state": by_object_state,
        "segments": segments,
    }


def _difference(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    difference = {}
    for key in ("duration_s", "arm_path_rad", "wrist_joint6_path_rad", "arm_speed_peak_abs_rad_s",
                "arm_speed_p95_max_joint_rad_s", "wrist_joint6_speed_peak_abs_rad_s",
                "wrist_joint6_speed_p95_abs_rad_s", "arm_acceleration_rms_rad_s2",
                "wrist_joint6_acceleration_rms_rad_s2"):
        a, b = before.get(key), after.get(key)
        delta = None if a is None or b is None else b - a
        difference[key] = {"baseline": a, "candidate": b, "delta": delta,
                           "change_percent": None if a is None or b is None or a == 0 else 100. * (b - a) / a}
    return difference


def compare_runs(baseline: Path, candidate: Path, terminal_sample: str = "auto") -> dict[str, Any]:
    """Load two completed local runs and return a JSON-serializable report."""
    if terminal_sample not in ("auto", "exclude", "include"):
        raise ValueError("terminal_sample must be auto, exclude or include")
    before_raw, after_raw = _read_run(Path(baseline), terminal_sample), _read_run(Path(candidate), terminal_sample)
    before, after = _summarize(before_raw), _summarize(after_raw)
    notes = []
    for key in ("seed", "score", "full_score", "action_spec", "task_physics_modified", "render_preset", "use_fabric"):
        if before["result"][key] != after["result"][key]:
            notes.append(f"Different {key}: baseline={before['result'][key]!r}, candidate={after['result'][key]!r}.")
    if before_raw["result"].get("initial_objects") != after_raw["result"].get("initial_objects"):
        notes.append("Initial object placements differ; this is not a matched-initial-state comparison.")
    if before["sample_dt_s"] != after["sample_dt_s"]:
        notes.append("Sampling intervals differ; discrete derivative statistics have different bandwidths.")
    if before["arm_joint_names"] != after["arm_joint_names"]:
        raise ValueError("Arm joint ordering differs between runs")
    if before["result"]["score"] != after["result"]["score"]:
        notes.append("Completion scores differ: a shorter failed run does not establish faster task completion.")
    grouped = {}
    for group in ("by_state", "by_object", "by_object_state"):
        grouped[group] = {
            label: _difference(before[group].get(label, {}), after[group].get(label, {}))
            for label in dict.fromkeys([*before[group], *after[group]])
        }
    return {
        "schema_version": 1,
        "measurement_contract": {
            "scope": "Local sampled telemetry, normally 50 Hz; no continuous-time jerk, contact-force or smoothness guarantee.",
            "duration": "Every executed step contributes dt, including a terminal step whose reset physics is excluded.",
            "path": "Sum abs(diff(qpos)) for first six arm joints; radians, no angle unwrapping. First reset-to-step displacement is unavailable.",
            "speed": "Measured absolute qvel. Per-joint P95 uses time samples; aggregate P95 uses max(abs(qvel[:6])) per sample.",
            "acceleration": "diff(measured qvel)/dt on valid adjacent rows. Arm RMS pools all six joints and valid intervals.",
            "grouping": "Intervals belong to destination sample state/object. Labels are recorded after predicts; transitions can be offset by one control step.",
            "terminal_sample_policy": terminal_sample,
            "auto_reset_heuristic": "At done/time_limit: last arm pose within .005 rad of declared default, abs qvel <=.005 rad/s, and a >=.05 rad jump whose rate exceeds 5x adjacent measured speed +.5 rad/s. Only then auto excludes terminal physics; include/exclude override it.",
            "wall_time": "Reported result wall_seconds, not normalized for recording, rendering or process startup differences.",
        },
        "comparison_notes": notes, "baseline": before, "candidate": after,
        "overall_difference": _difference(before["overall"], after["overall"]),
        "grouped_differences": grouped,
    }


def relativize_paths(report: dict[str, Any], root: Path) -> None:
    """Make known path fields portable without retaining a machine-local root."""
    root = root.resolve()
    for label in ("baseline", "candidate"):
        run = report[label]
        try:
            run["directory"] = Path(run["directory"]).resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("Both run directories must be inside --relative-to") from error
        solution = run["result"].get("solution")
        if solution and Path(solution).is_absolute():
            try:
                run["result"]["solution"] = Path(solution).relative_to(root).as_posix()
            except ValueError:
                run["result"]["solution"] = Path(solution).name
                run["notes"].append("Solution was outside the report root; only its filename is retained. Use source hash to identify the implementation.")
    report["measurement_contract"]["path_reference"] = "Paths are relative to the root passed as --relative-to; that machine-local root is not stored."


def render_markdown(report: dict[str, Any]) -> str:
    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.3f}"

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    before, after = report["baseline"], report["candidate"]
    lines = ["# Task E 运动对比", "", f"基线：`{cell(before['directory'])}`", "",
             f"候选：`{cell(after['directory'])}`", "",
             "这是本地离散采样统计，通常为 50 Hz。有限差分加速度不构成连续时间 jerk 或平顺性保证。", "",
             "| 指标 | 基线 | 候选 | 候选−基线 |", "| --- | ---: | ---: | ---: |"]
    for label, key in (("随机种子", "seed"), ("实际得分", "score"), ("result 仿真时间 / s", "sim_seconds"),
                       ("result 墙钟时间 / s", "wall_seconds")):
        a, b = before["result"][key], after["result"][key]
        lines.append(f"| {label} | {number(a)} | {number(b)} | {number(None if a is None or b is None else b-a)} |")
    for label, key in (("遥测计时 / s", "duration_s"), ("六臂关节总路程 / rad", "arm_path_rad"),
                       ("腕 joint6 路程 / rad", "wrist_joint6_path_rad"),
                       ("臂关节绝对速度峰值 / rad/s", "arm_speed_peak_abs_rad_s"),
                       ("逐帧最大臂关节速度 P95 / rad/s", "arm_speed_p95_max_joint_rad_s"),
                       ("腕 joint6 绝对速度 P95 / rad/s", "wrist_joint6_speed_p95_abs_rad_s"),
                       ("臂关节差分加速度 RMS / rad/s²", "arm_acceleration_rms_rad_s2"),
                       ("腕 joint6 差分加速度 RMS / rad/s²", "wrist_joint6_acceleration_rms_rad_s2")):
        values = report["overall_difference"][key]
        lines.append(f"| {label} | {number(values['baseline'])} | {number(values['candidate'])} | {number(values['delta'])} |")
    lines += ["", f"结束方式：基线 `{cell(before['result']['stop_reason'])}`；候选 `{cell(after['result']['stop_reason'])}`。", "",
              f"采样：基线 {number(before['sample_rate_hz'])} Hz；候选 {number(after['sample_rate_hz'])} Hz。"]
    all_notes = [*report["comparison_notes"],
                 *["Baseline: " + note for note in before["notes"]],
                 *["Candidate: " + note for note in after["notes"]]]
    if all_notes:
        lines += ["", "## 比较边界", "", *["- " + note for note in all_notes]]
    for title, group in (("按状态", "by_state"), ("按物体", "by_object"), ("按物体与状态", "by_object_state")):
        lines += ["", f"## {title}", "",
                  "| 分组 | 耗时 B→C / s | 六臂关节路程 B→C / rad | 腕6路程 B→C / rad | 速度 P95 B→C / rad/s | 加速度 RMS B→C / rad/s² |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, metrics in report["grouped_differences"][group].items():
            pairs = [f"{number(metrics[key]['baseline'])} → {number(metrics[key]['candidate'])}"
                     for key in ("duration_s", "arm_path_rad", "wrist_joint6_path_rad",
                                 "arm_speed_p95_max_joint_rad_s", "arm_acceleration_rms_rad_s2")]
            lines.append(f"| {cell(label)} | " + " | ".join(pairs) + " |")
    lines += ["", "## 统计定义", "",
              "- B 为基线，C 为候选；缺失阶段显示 —，不当成零。object_0 表示尚未选择物体。",
              "- 仅统计前六个旋转臂关节，排除两根直线夹爪关节；JSON 同时保留每关节峰值、P95 和 RMS。",
              "- 先对完整序列相邻行差分，再按后一个样本的阶段归组；跨阶段边计入后一个阶段，非相邻片段不会被连成假跳变。",
              "- 阶段标签来自 predicts 后的状态，切换时可能偏移一个控制步。JSON segments 保留连续片段和步号，可区分重试。",
              "- 默认仅在终止帧回到已知默认姿态、速度近零且出现与实际速度不符的大跳变时判作 reset 并排除物理量；此启发式可用 include/exclude 覆盖，始终保留该步耗时。首次动作前没有关节样本，其位移未计入。",
              "- 速度来自实际 qvel；加速度是相邻 qvel 除以 dt。没有计算或承诺连续时间 jerk、接触力峰值或全路径无碰撞。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Completed baseline run directory")
    parser.add_argument("candidate", type=Path, help="Completed candidate run directory")
    parser.add_argument("--output-dir", type=Path, help="Write comparison.json and comparison.md here")
    parser.add_argument("--output-json", type=Path, help="Explicit JSON path; overrides --output-dir JSON path")
    parser.add_argument("--output-md", type=Path, help="Explicit Markdown path; overrides --output-dir Markdown path")
    parser.add_argument("--relative-to", type=Path, help="Make report paths relative to this root for portable/public reports")
    parser.add_argument("--terminal-sample", choices=("auto", "exclude", "include"), default="auto",
                        help="auto excludes only a detected default-pose reset jump; include/exclude override detection")
    args = parser.parse_args()
    try:
        report = compare_runs(args.baseline, args.candidate, args.terminal_sample)
        if args.relative_to is not None:
            relativize_paths(report, args.relative_to)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"compare_motion: {error}\n")
    json_path = args.output_json or (args.output_dir / "comparison.json" if args.output_dir else None)
    md_path = args.output_md or (args.output_dir / "comparison.md" if args.output_dir else None)
    if json_path is not None and md_path is not None and json_path.resolve() == md_path.resolve():
        parser.error("JSON and Markdown output paths must differ")
    markdown = render_markdown(report)
    for path, contents in ((json_path, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                           (md_path, markdown)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    if json_path is None and md_path is None:
        print(markdown)
    else:
        print(json.dumps({"json": str(json_path) if json_path else None,
                          "markdown": str(md_path) if md_path else None,
                          "baseline_score": report["baseline"]["result"]["score"],
                          "candidate_score": report["candidate"]["result"]["score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
