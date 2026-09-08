"""Observation-only Task E RGB-D object localization.

The output coordinates are finger contact points, not simulator object roots
and not gripper_base targets. Camera calibration and nominal object dimensions
come from the supplied assets. Missing/occluded objects are omitted; spawn
coordinates and simulator state are never used to invent a detection.

Runtime dependencies: NumPy and SciPy. The optional command-line diagnostic
also uses Matplotlib, but importing this module does not import it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

try:
    from .task_e_geometry import TABLE_TOP_Z, project_world, unproject_depth, video_camera_parameters
except ImportError:  # Standalone submission directory / offline diagnostic CLI.
    from task_e_geometry import TABLE_TOP_Z, project_world, unproject_depth, video_camera_parameters


# Robust observed (long XY extent, short XY extent, vertical extent), metres.
# Sugar is lying on its broad side; mustard is upright; banana is low and long.
# Tolerances allow oblique views and partial occlusion, not a known spawn pose.
_SHAPE_MEAN = np.array([[.17, .045, .090], [.065, .050, .155], [.19, .055, .030]])
_SHAPE_SCALE = np.array([[.050, .025, .035], [.035, .025, .060], [.055, .035, .025]])
_OBJECT_NAMES = {1: "sugar_box", 2: "mustard_bottle", 3: "banana"}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _images(obs):
    images = obs.get("image", obs)
    depth = _numpy(images["video_depth"]).squeeze()
    if depth.ndim != 2:
        raise ValueError("Task E perception requires one video_depth image")
    rgb = images.get("video_rgb")
    if rgb is None:
        return None, depth
    rgb = _numpy(rgb)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.ndim != 3 or rgb.shape[:2] != depth.shape or rgb.shape[-1] not in (3, 4):
        raise ValueError("video_rgb and video_depth must have matching image dimensions")
    rgb = rgb[..., :3].astype(float)
    if rgb.size and np.nanmax(rgb) <= 1.01:
        rgb *= 255.
    return np.clip(np.nan_to_num(rgb), 0., 255.).astype(np.uint8), depth


def _point_components(points, voxel_size=.004, connection_radius=.009):
    """3-D connectivity separates objects whose projected silhouettes overlap."""
    _, representative, inverse = np.unique(np.floor(points / voxel_size).astype(np.int32),
                                           axis=0, return_index=True, return_inverse=True)
    cloud = points[representative]
    pairs = cKDTree(cloud).query_pairs(connection_radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(cloud), len(cloud)))
    count, labels = connected_components(graph, directed=False)
    pixel_labels = labels[inverse]
    return [np.flatnonzero(pixel_labels == i) for i in range(count)]


def _features(points, rgb):
    lower, upper = np.percentile(points, [1, 99], axis=0)
    center = (lower + upper) / 2
    xy = points[:, :2] - np.median(points[:, :2], axis=0)
    _, axes = np.linalg.eigh(xy.T @ xy / len(xy))
    major_axis = axes[:, -1]
    minor_axis = axes[:, 0]
    projected = xy @ np.column_stack((major_axis, minor_axis))
    span = np.diff(np.percentile(projected, [1, 99], axis=0), axis=0)[0]
    # PCA orientation is ambiguous for nearly circular bottle cross-sections.
    shape = np.array([max(span), min(span), upper[2] - lower[2]])
    yellow_fraction, saturation = 0., 0.
    if rgb is not None and len(rgb):
        colors = rgb.astype(float) / 255.
        maximum = colors.max(axis=1)
        chroma = maximum - colors.min(axis=1)
        saturation = float(np.median(chroma / np.maximum(maximum, .01)))
        yellow_fraction = float(np.mean((colors[:, 0] > 1.3 * colors[:, 2] + .04)
                                        & (colors[:, 1] > 1.3 * colors[:, 2] + .04)
                                        & (maximum > .18)))
    return dict(lower=lower, upper=upper, center=center, shape=shape,
                major_axis=major_axis, yellow_fraction=yellow_fraction,
                saturation=saturation)


def _contact_position(points, features, object_id):
    lower, upper = features["lower"], features["upper"]
    contact = features["center"].copy()
    if object_id == 3:
        # The bounding-box center may fall in the banana's empty inner curve.
        # Select the visible middle segment along its long axis, away from tips.
        axis = features["major_axis"]
        projection = points[:, :2] @ axis
        lo, hi = np.percentile(projection, [2, 98])
        middle = points[np.abs(projection - (lo + hi) / 2) <= max(.012, .12 * (hi - lo))]
        if len(middle) >= 10:
            contact[:2] = np.median(middle[:, :2], axis=0)
            contact[2] = np.percentile(middle[:, 2], 80) - .009
        else:
            contact[2] = upper[2] - .010
        contact[2] = max(TABLE_TOP_Z + .027, contact[2])
    else:
        # Grip the body, below the mustard cap and below the sugar-box top.
        fraction = .43 if object_id == 2 else .48
        contact[2] = max(TABLE_TOP_Z + .033, lower[2] + fraction * (upper[2] - lower[2]))
    return contact


def estimate_objects_detailed(obs):
    """Return (contact positions, diagnostics) using only the supplied RGB-D.

    Designed for a scan while the arm is clear of the objects. Call again after
    settling or moving the arm if an object is missing. Points already in the
    basket are excluded from the remaining-object list.
    """
    rgb, depth = _images(obs)
    height, width = depth.shape
    k, pose = video_camera_parameters(width, height)
    points, pixels = unproject_depth(depth, intrinsics=k, world_from_camera=pose, return_pixels=True)
    # Public task geometry, with margin for randomization and small disturbances.
    # This excludes the basket (negative Y), table plane and the robot pedestal.
    region = ((points[:, 0] > .77) & (points[:, 0] < 1.23)
              & (points[:, 1] > -.10) & (points[:, 1] < .44)
              & (points[:, 2] > TABLE_TOP_Z + .009)
              & (points[:, 2] < TABLE_TOP_Z + .245))
    points, pixels = points[region], pixels[region]
    diagnostics = dict(roi_points=int(len(points)), components=[], detections={})
    if len(points) < 50:
        return {}, diagnostics
    smooth_rgb = None if rgb is None else median_filter(rgb, size=(3, 3, 1))
    components = []
    for indices in _point_components(points):
        if len(indices) < 50:
            continue
        cloud = points[indices]
        colors = None if smooth_rgb is None else smooth_rgb[tuple(pixels[indices].T)]
        feature = _features(cloud, colors)
        long_side, short_side, vertical = feature["shape"]
        if not (.027 <= long_side <= .275 and .012 <= short_side <= .155 and .010 <= vertical <= .23):
            continue
        shape_cost = np.sum(((feature["shape"] - _SHAPE_MEAN) / _SHAPE_SCALE) ** 2, axis=1)
        # Yellow supports bottle/banana identification when texture rendering is
        # available. Geometry remains usable during initial noisy RGB frames.
        if feature["yellow_fraction"] > .35:
            shape_cost[1:] -= .15
        components.append((cloud, feature, shape_cost))
        diagnostics["components"].append(dict(points=int(len(cloud)),
            center=feature["center"].tolist(), shape=feature["shape"].tolist(),
            shape_cost=shape_cost.tolist(), yellow_fraction=feature["yellow_fraction"]))
    if not components:
        return {}, diagnostics

    costs = np.stack([entry[2] for entry in components])
    rows, columns = linear_sum_assignment(costs)
    detections = {}
    for row, column in zip(rows, columns):
        if costs[row, column] > 7.0:
            continue
        cloud, feature, _ = components[row]
        object_id = int(column + 1)
        contact = _contact_position(cloud, feature, object_id)
        detections[object_id] = contact
        diagnostics["detections"][str(object_id)] = dict(name=_OBJECT_NAMES[object_id],
            contact=contact.tolist(), component=int(row), shape_cost=float(costs[row, column]),
            pixels=int(len(cloud)), visible_bounds=[feature["lower"].tolist(), feature["upper"].tolist()])
    return detections, diagnostics


def estimate_objects(obs) -> dict[int, np.ndarray]:
    """Estimate visible object finger-contact XYZ points from Task E observations."""
    return estimate_objects_detailed(obs)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True, help="Saved observation NPZ; no trace/state file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path, help="Optional diagnostic figure showing projections on RGB")
    args = parser.parse_args()
    with np.load(args.observation, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files}
    detections, report = estimate_objects_detailed(obs)
    report["input"] = str(args.observation)
    report["observation_only"] = True
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rgb, depth = _images(obs)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(depth if rgb is None else rgb)
        colors = {1: "cyan", 2: "magenta", 3: "lime"}
        for object_id, contact in detections.items():
            uv, _ = project_world(contact[None], width=depth.shape[1], height=depth.shape[0])
            x, y = uv[0]
            ax.plot(x, y, "+", color=colors[object_id], markersize=15, markeredgewidth=2)
            ax.annotate(f"{object_id}: {_OBJECT_NAMES[object_id]}", (x, y), xytext=(8, -10),
                        textcoords="offset points", color=colors[object_id], fontsize=9,
                        bbox=dict(facecolor="black", alpha=.7, edgecolor="none"))
        ax.set_title("Observation-only contact estimates (initial frame may be unsettled)")
        ax.set_axis_off()
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
