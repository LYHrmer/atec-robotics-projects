"""CPU audit of the third-person overview video layout.

No simulator or GPU. Checks the mount rotation and the frame composition that
``--video_view overview`` uses, on synthetic frames, so a wrong convention or an
inset in the wrong place shows up here instead of in a 47-second recording.

Prints a JSON report; exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task_b import overview_video as overview  # noqa: E402

checks, notes = {}, []


def record(name, passed, note=""):
    checks[name] = bool(passed)
    if note:
        notes.append(f"{name}: {note}")


def recorded_point_margin(run_directory, stride=10):
    """Worst frame margin over a recorded run's own gripper and held-object points.

    Uses only what the run recorded -- the gripper body position and every
    object's root position, converted to the body frame with the recorded base
    pose -- and the mount THAT RUN actually used, read back from its own
    result.json. Nothing is assumed: not a chassis box, which the run does not
    record, and not the current default mount, which would make every run score
    the same and the comparison meaningless.
    """
    import json
    from task_b import camera_calibration as cal

    result = json.loads((Path(run_directory) / "result.json").read_text())
    mounted = result.get("overview_camera") or {}
    if not mounted:
        raise ValueError(f"{run_directory} recorded no overview camera; it is not an overview run")
    position = np.asarray(mounted["base_link_from_camera_pos"], dtype=float)
    target = np.asarray(mounted["aimed_at_base_link_point"], dtype=float)
    run = np.load(Path(run_directory) / "telemetry.npz")
    base, quat = run["base_xyz"], run["base_quat"]
    gripper, objects = run["gripper_xyz"], run["object_xyz"]
    worst, worst_step = 1e9, None
    for index in range(0, len(base), stride):
        rotation = cal.body_from_world(base[index], quat[index])[:3, :3]
        body = [(gripper[index] - base[index]) @ rotation.T]
        roots = (objects[index] - base[index]) @ rotation.T
        nearest = int(np.argmin(np.linalg.norm(roots - body[0], axis=1)))
        body.append(roots[nearest])
        margin, _ = overview.project_points(position, target, np.asarray(body))
        if margin < worst:
            worst, worst_step = margin, index
    return margin is not None and worst, worst_step, int(len(base)), position, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="a recorded run directory whose gripper and held-object points are "
                             "projected into the mount; repeatable")
    args = parser.parse_args()

    position = np.asarray(overview.OVERVIEW_CAMERA_POS, dtype=float)
    target = np.asarray(overview.OVERVIEW_CAMERA_TARGET, dtype=float)
    rotation = overview.world_camera_from_look_at(position, target)

    record("look_at_rotation_is_proper",
           bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                and np.isclose(np.linalg.det(rotation), 1., atol=1e-12)),
           f"det={np.linalg.det(rotation):.12f}")

    forward = target - position
    distance = float(np.linalg.norm(forward))
    record("camera_forward_axis_points_at_the_target",
           float(np.abs(rotation[:, 0] - forward / distance).max()) < 1e-12,
           f"mount is {distance:.2f} m from the aim point")

    # The strongest check on the convention: in the camera's own frame the aim
    # point must sit exactly on +X, with no sideways or vertical offset.
    aimed = rotation.T @ forward
    record("aim_point_lands_on_the_camera_forward_axis",
           float(np.abs(aimed[[1, 2]]).max()) < 1e-12 and aimed[0] > 0.,
           f"camera-frame aim point {np.round(aimed, 12).tolist()}")

    # A look-at built from world up has no roll about the view axis, so the
    # camera's left axis is exactly horizontal while its up axis tilts back
    # toward world up. That is what keeps the horizon level in the recording.
    record("camera_has_no_roll_about_the_view_axis",
           abs(float(rotation[2, 1])) < 1e-12 and float(rotation[2, 2]) > 0.5,
           f"camera left sits at world z {rotation[2, 1]:+.1e} and camera up reaches "
           f"{rotation[2, 2]:+.3f} of world up")

    record("camera_is_behind_and_above_the_robot",
           position[0] < -1.0 and position[2] > 1.0 and 2. < distance < 6.,
           f"pos {position.tolist()}, distance {distance:.2f} m -- a chase view, not an "
           "onboard one")

    margin, span = overview.project_envelope(position, target)
    record("robot_envelope_stays_inside_the_frame",
           margin >= overview.FRAMING_MARGIN_PX,
           f"nearest edge {margin:.1f} px away, against a {overview.FRAMING_MARGIN_PX} px floor; "
           "the envelope is a superset of every chassis, arm and held-object configuration the "
           "probe reaches")
    record("robot_fills_enough_of_the_frame_to_read",
           span >= 0.30,
           f"the envelope spans {span * 100:.1f} percent of the frame width")
    for name, worse in (("too_far_and_high", ((-2.9, -2.9, 2.1), target)),
                        ("aimed_over_the_robot", (position, (0.15, 0.0, 1.30)))):
        try:
            other_margin, other_span = overview.project_envelope(*worse)
            record(f"framing_check_rejects_{name}", other_margin < margin or other_span < span,
                   f"that mount gives margin {other_margin:.1f} px and span {other_span * 100:.1f} "
                   f"percent, against {margin:.1f} and {span * 100:.1f} for the chosen one")
        except ValueError as error:
            record(f"framing_check_rejects_{name}", True, str(error))

    # NB: do not name this loop variable `args`; it would shadow the argparse
    # namespace the --run loop below still needs.
    for name, bad in (("a_coincident_aim_point", (target, target)),
                      ("a_vertical_view_direction", ((0., 0., 0.), (0., 0., 1.)))):
        try:
            overview.world_camera_from_look_at(*bad)
            record(f"look_at_rejects_{name}", False, "no error raised")
        except ValueError as error:
            record(f"look_at_rejects_{name}", True, str(error))

    rng = np.random.default_rng(11)
    width, height = overview.OVERVIEW_FRAME
    overview_rgb = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    head_rgb = np.zeros((480, 640, 4), np.uint8); head_rgb[..., 0] = 200
    ee_rgb = np.zeros((480, 640, 4), np.uint8); ee_rgb[..., 1] = 200
    originals = [array.copy() for array in (overview_rgb, head_rgb, ee_rgb)]
    frame = overview.compose_overview_frame(overview_rgb, head_rgb, ee_rgb, "audit caption")

    record("composed_frame_has_the_declared_size",
           frame.shape == (height, width, 3), f"shape {frame.shape}")
    record("composition_does_not_modify_its_inputs",
           all(np.array_equal(array, original)
               for array, original in zip((overview_rgb, head_rgb, ee_rgb), originals)))

    inset_width, inset_height = overview.OVERVIEW_INSET
    inset_y = height - inset_height - 16
    placed = {}
    for index, (label, colour) in enumerate((("head", 0), ("wrist", 1))):
        x = 16 + index * (inset_width + 12)
        patch = frame[inset_y + 60:inset_y + inset_height - 20, x + 40:x + inset_width - 40]
        # The label text sits in the top strip of the inset, so sample below it.
        placed[label] = bool(np.all(patch[..., colour] > 150)
                             and np.all(patch[..., 1 - colour] < 60))
    record("both_onboard_insets_are_visible_where_expected", all(placed.values()),
           f"head and wrist insets found at y={inset_y}; {placed}")
    record("title_bar_is_drawn_over_the_overview",
           bool(np.all(frame[4:12, 4:12] == np.array([15, 20, 26], np.uint8))))

    try:
        overview.compose_overview_frame(np.zeros((720, 1280, 3), np.uint8), head_rgb, ee_rgb, "x")
        record("composition_rejects_a_wrong_overview_size", False, "no error raised")
    except ValueError as error:
        record("composition_rejects_a_wrong_overview_size", True, str(error))

    for directory in args.run:
        margin, step, steps, used_pos, _ = recorded_point_margin(directory)
        record(f"recorded_points_stay_inside_the_frame:{Path(directory).name}",
               margin >= overview.FRAMING_MARGIN_PX,
               f"{Path(directory).name}: with its own mount {np.round(used_pos, 2).tolist()} the "
               f"gripper and held-object points stay {margin:.1f} px inside the frame over {steps} "
               f"steps (worst at step {step})")

    report = {"checks": checks, "count": len(checks), "notes": notes,
              "passed": all(checks.values()),
              "scope": "CPU geometry and layout only. It does not show that the overview camera "
                       "renders, that the GPU can carry a third 1080p render product, or what the "
                       "scene looks like from the mount."}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
