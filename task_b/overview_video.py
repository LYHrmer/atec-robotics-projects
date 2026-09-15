"""Third-person overview frame for Task B recordings.

The onboard cameras are the only views the official task has, and at 640x480
they are both low resolution and pointed at the work, not at the robot. The Task
A runs solved this with an extra overview camera on the chassis that looks back
at the robot; this module holds the same construction for B2wPiper -- the mount
rotation and the frame composition -- with no simulator import, so both can be
checked on the CPU before a GPU run is spent.

The camera itself is created by ``task_b/evaluate.py`` from these constants. It
is a render product added for the recording: absent from the official scene, not
read by any policy, and declared in the run's result and source manifest.
"""
from __future__ import annotations

import numpy as np

#: The mount, in the base_link frame: behind-left and above the chassis, aimed
#: at the body's mid-height so the chassis sits centred rather than low. The
#: distance and aim were chosen on CPU against a recorded run's body-frame
#: trajectory (see audit_overview_video.py): every chassis corner, the gripper
#: and the held object stay at least 115 px inside a 1920x1080 frame, where the
#: original 3.7 m mount clipped the chassis bottom by 11 px, while the chassis
#: still spans about 39 percent of the frame width.
OVERVIEW_CAMERA_POS = (-2.00, -2.00, 1.25)
OVERVIEW_CAMERA_TARGET = (0.15, 0.0, 0.20)
#: Size of each onboard inset in the composed frame.
OVERVIEW_INSET = (448, 336)
#: Composed frame size, matching the overview camera's own 1920x1080 render.
OVERVIEW_FRAME = (1920, 1080)
_TITLE_BAR = 34
#: Conservative body-frame envelope the camera must keep in view: the chassis
#: box, the arm's working volume and the held object at full reach. A superset
#: of every configuration the probe reaches, so a pass bounds the whole run.
ROBOT_ENVELOPE = ((0.45, 0.32, 0.35), (-0.45, -0.32, -0.15))
#: Narrowest acceptable gap between the envelope and any frame edge.
FRAMING_MARGIN_PX = 30


def project_envelope(position, target, envelope=ROBOT_ENVELOPE, image=OVERVIEW_FRAME,
                     focal_length=24.0, aperture=20.955):
    """Pixel extents of a body-frame box seen from a candidate mount.

    Returns the margin to the nearest frame edge and the fraction of the frame
    width the box spans. Both are what a readable chase view needs: the box has
    to stay inside the frame throughout, and it has to be big enough to read.
    """
    position = np.asarray(position, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    (x_high, y_high, z_high), (x_low, y_low, z_low) = envelope
    corners = np.array([[x, y, z]
                        for x in (x_low, x_high) for y in (y_low, y_high) for z in (z_low, z_high)])
    rotation = world_camera_from_look_at(position, target)
    camera = (corners - position) @ rotation
    if (camera[:, 0] <= 0.2).any():
        raise ValueError("part of the envelope is behind the camera")
    width, height = image
    focal = width * focal_length / aperture
    # World-camera (+X forward, +Y left, +Z up) to image: right is -Y, down is -Z.
    u = focal * (-camera[:, 1] / camera[:, 0]) + width / 2.
    v = focal * (-camera[:, 2] / camera[:, 0]) + height / 2.
    margin = float(min(u.min(), v.min(), width - u.max(), height - v.max()))
    return margin, float(u.max() - u.min()) / width


def world_camera_from_look_at(position, target):
    """Rotation of a camera at ``position`` aimed at ``target``, world up.

    Returned in Isaac's "world camera" convention -- +X forward, +Y left,
    +Z up -- which is the convention the official head camera's ``OffsetCfg``
    uses, so the same offset axes apply.
    """
    position = np.asarray(position, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if position.size != 3 or target.size != 3:
        raise ValueError("position and target must each be three metres")
    if not (np.isfinite(position).all() and np.isfinite(target).all()):
        raise ValueError("position and target must be finite")
    forward = target - position
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("position and target coincide; there is no view direction")
    forward /= norm
    left = np.cross(np.array([0., 0., 1.]), forward)
    norm = float(np.linalg.norm(left))
    if norm < 1e-9:
        raise ValueError("the view direction is vertical; world up cannot fix the roll")
    left /= norm
    return np.column_stack([forward, left, np.cross(forward, left)])


def compose_overview_frame(overview_rgb, head_rgb, ee_rgb, caption):
    """1920x1080 third-person frame with the two onboard cameras as insets.

    The overview frame is the whole canvas; the onboard views are scaled into
    the bottom-left corner, bordered and labelled, where they do not cover the
    robot. Inputs are not modified.
    """
    import cv2

    frame = np.ascontiguousarray(np.asarray(overview_rgb)[..., :3].astype(np.uint8))
    width, height = OVERVIEW_FRAME
    if frame.shape[:2] != (height, width):
        raise ValueError(f"overview frame is {frame.shape[1]}x{frame.shape[0]}, "
                         f"expected {width}x{height}")
    cv2.rectangle(frame, (0, 0), (width, _TITLE_BAR), (15, 20, 26), -1)
    cv2.putText(frame, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .62,
                (240, 240, 240), 1, cv2.LINE_AA)
    inset_width, inset_height = OVERVIEW_INSET
    for index, (label, image) in enumerate((("head RGB", head_rgb), ("wrist RGB", ee_rgb))):
        small = cv2.resize(np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8)),
                           (inset_width, inset_height), interpolation=cv2.INTER_AREA)
        x = 16 + index * (inset_width + 12)
        y = height - inset_height - 16
        cv2.rectangle(frame, (x - 2, y - 2), (x + inset_width + 2, y + inset_height + 2),
                      (15, 20, 26), -1)
        frame[y:y + inset_height, x:x + inset_width] = small
        cv2.putText(frame, label, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return frame
