"""Public RGB-D observation of the Task B bin wall, with honest quality data.

Why this file exists
--------------------
The first real delivery attempt (p13, ``payload_motion``) ended with the robot
at ``(-7.188, -9.007)`` holding the bottle, i.e. about 4.30 m from the public
static bin centre ``(-3, -10)``, and the delivery route now has to approach,
turn tangential at 1.45 m and stop.  The contract asks for one thing before
that stop: a *public* estimate of where the bin wall actually is in the body
frame, so the last metre is not driven blind.  This module produces exactly
that estimate and nothing else - it never emits an action, never reads the
object root, never projects the known bin centre, and never touches a seed map
or a global planner.

What the saved p13 cameras actually showed (checked, not assumed)
----------------------------------------------------------------
``task_b_score/plan_p13_payload_denyquist_seed42_01/`` keeps 14 head and 14
wrist RGB frames.  Projecting the public static bin (centre ``(-3,-10)``,
outer radius 1 m, wall top z = .55 m) through the static camera transforms and
the recorded public proprio:

* the head camera never contains the bin.  At the p13 end pose the wall is
  2.95-4.97 m from that camera and projects *above the top edge* (v from -181
  to +1.5 px).  This is geometry, not a detector failure: the head camera is
  pitched ~30 deg down and its whole useful ground band is roughly 0.5-2.7 m
  ahead, so a bin further than ~3.2 m from the body centre is out of frame by
  construction.  Measured directly on the saved truth poses: the head camera
  only starts to see this bin at about 3.3-3.5 m of centre standoff, and only
  sees the full wall height from about 1.6 m outwards.
* therefore the head frames are all ground texture, which is what they look
  like, and the wrist frames are dominated by the carried bottle (mean RGB of
  the lower-centre patch at the final wrist frame is (209, 195, 67) - strongly
  yellow).  The bin itself renders as a *neutral* surface: pixels sampled on
  the projected bin wall are (171, 174, 176), i.e. no chroma at all, and the
  ground it stands on is (165, 171, 178).  A colour detector cannot separate
  them, and a yellow detector would lock onto the payload instead.  The
  designed cue here is therefore depth *surface orientation*, not appearance.
* the wrist camera does see the bin early (steps ~500-1500, at 5-8 m, the
  projected wall ring lands on the rendered wall), so a wrist observation is
  possible in principle - but the payload posture points that camera down at
  the ground and the held bottle, and at the contract's final tangential dock
  heading the bin sits at body +Y, about 106 deg off the head camera's optical
  axis versus a 23.6 deg half field of view.  Sustained visibility is
  explicitly *not* assumed anywhere in this module.

What that means for the design, stated plainly
---------------------------------------------
The only regime where this module has a chance of working is short range: bin
roughly ahead (|bearing| <= ~45 deg), head camera, and - from the synthetic
ray-cast check at the bottom of this docstring - a bin centre standoff of about
2.1-3.5 m.  Closer than that the visible arc of the wall falls under 45 deg,
further than that the wall drops below the head camera's 30-deg-down field of
view entirely; at 4.0 m and beyond there is no vertical surface in the image at
all.  The contract already asks for the same thing - anchor with a public
observation *before* turning tangential, then run the last leg on short range
odometry and propagate uncertainty.  This module is the anchor, not the final
position source, and it will refuse rather than guess outside that window.

Known risks, all unresolved
---------------------------
* **Depth is not in the saved artefacts.**  p13 stored RGB only, so the
  estimator below has been exercised against synthetic depth built from the
  public static bin constants, never against a real recorded depth frame.  Its
  residual, inlier and arc thresholds are therefore *candidates*, and the
  reported residual says nothing about the absolute position error.
* **A flat vertical surface is locally a cylinder.**  Over a short arc a
  straight wall fits a .975281 m circle almost as well as the real wall does
  (sagitta of a flat chord of length L against radius R is L^2/8R).  The
  reported ``flat_wall_sagitta_m`` is the number that makes the margin visible;
  a low residual alone is not evidence of curvature, and a low residual is
  never reported as a low position error.
* **Robot self-occlusion.**  With the arm extended over the side the head
  camera can see the robot's own arm or the carried bottle as a vertical
  surface.  Height band, cluster size, arc span and residual gates are meant
  to reject that; none of them has been measured against a real frame.
* **The wrist camera can be above the rim.**  In the p13 payload posture the
  wrist camera sits at 0.878 m, well above the 0.55 m rim, so it can see the
  *inner* wall as well as the outer one; the two surfaces are 20 mm apart and
  a single known-radius fit then has two candidate circles.  The gate handles
  this by refusing (the synthetic check reports a 40 mm residual there), but
  the refusal is conservative, not a diagnosis.
* **Which polygon radius is the right one to fit.**  The contract is explicit
  that .98 m is not a safe circular bound for the 32-gon inner wall and that
  .975281 m must be used instead.  The camera here is *outside* the bin, so the
  surface it actually sees is the outer wall, whose own inscribed radius is
  1.0*cos(pi/32) = .995185 m.  Fitting the inner radius to the outer surface
  moved the estimated centre 22.5 mm towards the camera in the synthetic check,
  against a predicted 19.9-24.7 mm - a systematic error far larger than the
  facet depth it was meant to bound.  The module therefore applies the same
  inscribed-polygon rule to whichever surface is being observed
  (``observed_surface``), keeps .975281 m exported as
  ``inner_wall_inscribed_radius_m`` for anything that goes *inside* the bin, and
  reports the residual facet bias as ``radius_model_bias_bound_m`` (<= 4.815 mm
  for the outer surface) rather than pretending the fit residual covers it.
* **Camera staleness.**  The official cameras run at 0.1 s update period while
  control runs at 0.02 s, so four of five ticks legitimately repeat a frame.
  That is reported, not treated as failure; a frame that has not changed for a
  full second is treated as a frozen pipeline.

Status: candidate and hypothesis.  Nothing here is a verified result, and
``trustworthy`` means "the checks in this file passed on this frame", not "the
estimate is correct".

What has actually been run through it
-------------------------------------
A ray-cast renderer built from the same public constants (32-gon annulus, the
static head/wrist transforms, optical-axis depth) was used as the only real
exercise of this code so far.  On that synthetic depth the estimator recovers
the centre to 1.6-1.7 mm over 2.1-3.5 m of standoff with a 1.4 mm fit
residual, stays inside 1.7 mm with 0.5 mm rms depth noise, and refuses at
1.8/2.0 m (arc under 45 deg), 3.7 m and beyond (no vertical surface in view),
at the contract's tangential dock heading, when facing away, under a near-field
occluder, and against a genuinely flat vertical wall of the same height (a flat
wall fits the known radius only over a short arc, which is what
``flat_wall_sagitta_m`` quantifies).  Those numbers come from a noise-free
synthetic render with no robot body in it.  They are a plumbing proof, not an
accuracy claim.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from task_b.arm_kinematics import arm_joints_from_proprio, ee_camera_transform, head_camera_transform

# ---------------------------------------------------------------- static config
#: Public static task configuration (the bin is part of the terrain, not an
#: object).  The bin *centre* (-3, -10) is deliberately absent from this module:
#: seeding or projecting it would make every check below circular.
BARREL_OUTER_RADIUS_M = 1.0
BARREL_WALL_THICKNESS_M = 0.02
BARREL_WALL_HEIGHT_M = 0.5
BARREL_RIM_Z_M = 0.55
BARREL_FLOOR_TOP_Z_M = 0.05
BARREL_WALL_SEGMENTS = 32
#: The wall is an annulus polygon: .98 m and 1.0 m are the *vertex* radii of the
#: inner and outer surfaces, so neither is a safe circular bound.  The contract
#: is explicit about the inner one because the bottle goes inside; the same
#: inscribed rule applies to the outer surface, which is the one a camera
#: outside the bin actually sees.
BARREL_INNER_WALL_INSCRIBED_RADIUS_M = (BARREL_OUTER_RADIUS_M - BARREL_WALL_THICKNESS_M) * math.cos(
    math.pi / BARREL_WALL_SEGMENTS)
BARREL_OUTER_INSCRIBED_RADIUS_M = BARREL_OUTER_RADIUS_M * math.cos(math.pi / BARREL_WALL_SEGMENTS)
#: Deliberately not part of the estimator, kept only so trials can be compared.
BARREL_INNER_WALL_VERTEX_RADIUS_M = BARREL_OUTER_RADIUS_M - BARREL_WALL_THICKNESS_M
#: Fitting the outer surface with the inner radius moves the centre 20 mm
#: towards the camera - measured at 22.5 mm on the synthetic check below.
BARREL_SURFACE_CHOICE_BIAS_M = BARREL_OUTER_INSCRIBED_RADIUS_M - BARREL_INNER_WALL_INSCRIBED_RADIUS_M

HEAD_CAMERA_FOCAL_LENGTH_MM = 24.0
EE_CAMERA_FOCAL_LENGTH_MM = 15.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
CONFIGURED_RASTER = (480, 640)
#: Official CameraCfg.update_period.  The public frame is up to this old.
CAMERA_UPDATE_PERIOD_S = 0.1

# ------------------------------------------------------------ initial candidate
#: Every threshold below is an initial candidate, not a measured capability.
DEFAULT_STRIDE_PX = 4
MIN_DEPTH_M = 0.30           #: below this the wrist camera sees its own payload
MAX_DEPTH_M = 4.50
MAX_WALL_PLANAR_RANGE_M = 3.20   #: keep only the near wall; the far side is out of band anyway
MIN_WALL_HEIGHT_M = -0.85    #: body-frame height band containing a bin wall
MAX_WALL_HEIGHT_M = 0.30
MAX_FACE_TILT_COS = 0.50     #: |normal . up| below this; the ground scores ~1
MAX_NORMAL_BASELINE_M = 0.08 #: refuse normals computed across a depth step
MIN_WALL_POINTS = 150        #: for the cluster and again for the robust inliers at the default stride
MIN_WALL_ARC_DEG = 45.0      #: below this a flat wall can impersonate the cylinder
MAX_RESIDUAL_RMS_M = 0.020
MAX_RESIDUAL_ABS_M = 0.050
MIN_CENTRE_STANDOFF_M = 1.20
MAX_CENTRE_STANDOFF_M = 4.00
MAX_WALL_DISTANCE_M = 3.00
MAX_CENTRE_SIGMA_M = 0.050
MAX_FROZEN_FRAME_CALLS = 50  #: 1.0 s at dt = 0.02
MAX_SOURCE_DISAGREEMENT_M = 0.15
BLOCKING_DEPTH_M = 0.30
MAX_BLOCKED_FRACTION = 0.35

_EPS = 1e-9


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive_float(value: Any, name: str) -> float:
    out = _finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _camera_intrinsics(width: int, height: int, focal_length_mm: float) -> Tuple[float, float, float]:
    """Raster pinhole K used by the frozen visual module: f = w * mm / aperture.

    The official ``distance_to_image_plane`` depth is the optical-axis Z, so the
    back-projection is ``((u-cx)Z/f, (v-cy)Z/f, Z)`` with the principal point at
    (width/2, height/2).  Matching that convention exactly matters more than
    being right about a half pixel, because a silent change here would move
    every estimate by centimetres at these ranges.
    """
    focal = width * focal_length_mm / CAMERA_HORIZONTAL_APERTURE_MM
    return focal, width * 0.5, height * 0.5


def _rgbd_arrays(image: Any, source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one camera's public RGB-D, rejecting anything ambiguous.

    Mirrors the shape rules of the frozen visual module without importing it:
    a leading batch axis of 1 is dropped and a channels-first CHW frame is
    moved back.  Depth is used as the optical-axis Z in metres.
    """
    if not isinstance(image, dict) or source + "_rgb" not in image or source + "_depth" not in image:
        raise ValueError("missing_" + source + "_rgbd")
    rgb = np.asarray(image[source + "_rgb"])
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    depth = np.asarray(image[source + "_depth"]).squeeze()
    if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or depth.ndim != 2
            or rgb.shape[:2] != depth.shape or min(depth.shape) < 16):
        raise ValueError("invalid_" + source + "_rgbd_shape")
    return np.ascontiguousarray(rgb[..., :3]), np.ascontiguousarray(depth.astype(np.float64, copy=False))


def _frame_digest(depth: np.ndarray) -> str:
    """Cheap content hash of a frame, used only to detect a frozen camera."""
    sample = np.ascontiguousarray(depth[::8, ::8], dtype=np.float32)
    return hashlib.sha256(sample.tobytes()).hexdigest()[:16]


def _backproject(us: np.ndarray, vs: np.ndarray, depth: np.ndarray,
                 body_from_camera: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """Pixel grid plus optical-axis depth -> body-frame points, shape (..., 3)."""
    camera = np.stack(((us - cx) * depth / focal, (vs - cy) * depth / focal, depth), axis=-1)
    rotation, translation = body_from_camera[:3, :3], body_from_camera[:3, 3]
    return camera @ rotation.T + translation


def _wall_point_mask(depth: np.ndarray, body_from_camera: np.ndarray, up: np.ndarray, *,
                     focal: float, cx: float, cy: float, stride: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Boolean mask of depth samples that look like the bin's vertical wall.

    The bin wall and the ground are both neutral grey in the saved renders, so
    the discriminator has to be surface *orientation*: the ground's normal is
    the up axis, the wall's is horizontal.  Normals come from neighbouring
    grid samples, and any neighbour pair whose depths disagree by more than
    ``MAX_NORMAL_BASELINE_M`` is discarded, because a normal computed across a
    depth discontinuity describes neither surface.
    """
    height, width = depth.shape
    grid_y, grid_x = np.mgrid[0:height:stride, 0:width:stride]
    sampled = depth[grid_y, grid_x]
    finite = np.isfinite(sampled)
    usable = finite & (sampled > MIN_DEPTH_M) & (sampled < MAX_DEPTH_M)
    filled = np.where(usable, sampled, 1.0)
    points = _backproject(grid_x.astype(float), grid_y.astype(float), filled, body_from_camera, focal, cx, cy)

    step_u = points[:, 1:] - points[:, :-1]
    step_v = points[1:, :] - points[:-1, :]
    depth_u = np.abs(sampled[:, 1:] - sampled[:, :-1])
    depth_v = np.abs(sampled[1:, :] - sampled[:-1, :])
    normals = np.cross(step_u[:-1], step_v[:, :-1])
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(lengths, _EPS)
    consistent = (depth_u[:-1] < MAX_NORMAL_BASELINE_M) & (depth_v[:, :-1] < MAX_NORMAL_BASELINE_M)
    neighbour_usable = (usable[:, 1:][:-1] & usable[:, :-1][:-1] & usable[1:, :][:, :-1] & usable[:-1, :][:, :-1])

    tilt = np.abs(normals @ up)
    vertical = consistent & neighbour_usable & np.isfinite(tilt) & (tilt <= MAX_FACE_TILT_COS)
    # Point i of the normal grid belongs to grid cell (i, j).
    cell_points = points[:-1, :-1]
    height_body = cell_points @ up
    in_band = (height_body > MIN_WALL_HEIGHT_M) & (height_body < MAX_WALL_HEIGHT_M) & usable[:-1, :-1]
    planar = np.linalg.norm(cell_points[..., :2], axis=-1) < MAX_WALL_PLANAR_RANGE_M
    mask = vertical & in_band & planar

    diagnostics = {
        "sampled_pixels": int(sampled.size),
        "valid_depth_pixels": int(usable.sum()),
        "vertical_surface_pixels": int(vertical.sum()),
        "wall_candidate_pixels": int(mask.sum()),
        "blocked_fraction": float((finite & (sampled < BLOCKING_DEPTH_M)).sum() / max(int(finite.sum()), 1)),
        "depth_range_m": ([float(sampled[usable].min()), float(sampled[usable].max())]
                          if usable.any() else None),
    }
    return points[:-1, :-1], mask, diagnostics


def _fit_known_radius_circle(points_xy: np.ndarray, radius: float, body_origin_xy: np.ndarray,
                             *, trim_rounds: int = 2) -> Optional[Dict[str, Any]]:
    """Fit the *known* wall radius and return only the centre.

    Known radius turns a 3-parameter circle fit into a well-conditioned
    2-parameter one, and a short arc is then still identifiable - but only
    along the arc's bisector, which is exactly what the returned covariance
    measures.  The seed places the centre ``radius`` behind the mean of the
    observed points, away from the robot, which is where the far side of a bin
    wall has to be.
    """
    if len(points_xy) < 4:
        return None
    origin = np.asarray(body_origin_xy, dtype=float)
    mean_point = points_xy.mean(axis=0)
    direction = mean_point - origin
    norm = float(np.linalg.norm(direction))
    if norm < _EPS:
        direction, norm = np.array([1.0, 0.0]), 1.0
    box = MAX_CENTRE_STANDOFF_M
    lower = origin - box
    upper = origin + box

    active = np.arange(len(points_xy))
    centre = mean_point + radius * direction / norm
    for _ in range(int(trim_rounds)):
        current = points_xy[active]

        def residual(candidate):
            return np.linalg.norm(current - candidate, axis=1) - radius

        fit = least_squares(residual, np.clip(centre, lower, upper), bounds=(lower, upper),
                            method="trf", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
        centre = np.asarray(fit.x, dtype=float)
        errors = residual(centre)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        limit = max(3.0 * 1.4826 * mad, 0.004)
        keep = np.abs(errors) <= limit
        if int(keep.sum()) < 4:
            break
        active = active[keep]

    inliers = points_xy[active]
    if len(inliers) < 4:
        return None
    offsets = inliers - centre
    ranges = np.linalg.norm(offsets, axis=1)
    if np.any(ranges < _EPS):
        return None
    residuals = ranges - radius
    rss = float(residuals @ residuals)
    dof = max(len(inliers) - 2, 1)
    sigma2 = rss / dof
    # d(residual)/d(centre) is -unit(centre -> point); JtJ is that outer product.
    jacobian = -offsets / ranges[:, None]
    jtj = jacobian.T @ jacobian
    try:
        covariance = sigma2 * np.linalg.inv(jtj)
        eigenvalues = np.linalg.eigvalsh(covariance)
        centre_sigma = float(math.sqrt(max(eigenvalues[-1], 0.0)))
    except np.linalg.LinAlgError:
        centre_sigma = float("inf")

    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    ordered = np.sort(angles)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * math.pi])
    arc_span = float(2.0 * math.pi - gaps.max()) if len(ordered) > 1 else 0.0
    return {
        "centre_xy": centre,
        "inlier_index": active,
        "inlier_count": int(len(inliers)),
        "residual_rms_m": float(math.sqrt(rss / len(inliers))),
        "residual_max_m": float(np.abs(residuals).max()),
        "residual_median_m": float(np.median(residuals)),
        "arc_span_rad": arc_span,
        "centre_sigma_m": centre_sigma,
        "centre_distance_m": float(np.linalg.norm(centre - origin)),
    }


def _cluster_mask(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label the wall-candidate mask; the bin wall is one large component."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, int(count)


class BarrelObservation:
    """Estimate the bin wall and centre in the body frame from public RGB-D.

    The object under observation is the Task B goal bin: the ring wall of the
    terrain at the public static centre, outer radius 1 m, wall thickness .02 m,
    wall top z = .55 m, built as a 32-segment annulus.  The file keeps the task's
    own word ("bucket") in its name; the constants keep saying "barrel".

    Lifecycle: construct once per episode, call :meth:`observe` with the public
    image dict and the public 84-element proprio where an estimate is wanted.
    Nothing here is an action, and nothing here is written back anywhere.
    """

    def __init__(self, observation_joint_names: Sequence[str], default_joint_positions: Dict[str, float], *,
                 dt: float = 0.02, sources: Sequence[str] = ("head", "ee"),
                 observed_surface: str = "outer", radius_m: Optional[float] = None,
                 stride: int = DEFAULT_STRIDE_PX) -> None:
        self.dt = _positive_float(dt, "dt")
        self.sources = tuple(sources)
        if not self.sources or any(source not in ("head", "ee") for source in self.sources):
            raise ValueError("sources must be a non-empty subset of ('head', 'ee')")
        if observed_surface not in ("outer", "inner"):
            raise ValueError("observed_surface must be 'outer' or 'inner'")
        self.observed_surface = observed_surface
        default_radius = (BARREL_OUTER_INSCRIBED_RADIUS_M if observed_surface == "outer"
                          else BARREL_INNER_WALL_INSCRIBED_RADIUS_M)
        self.radius_m = _positive_float(default_radius if radius_m is None else radius_m, "radius_m")
        if not 0.5 < self.radius_m < 1.5:
            raise ValueError("radius_m must be a plausible bin wall radius in metres")
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be a positive pixel step")
        self.observation_joint_names = tuple(observation_joint_names)
        self.defaults = dict(default_joint_positions)
        self.calls = 0
        self._digests: Dict[str, str] = {}
        self._repeat_calls: Dict[str, int] = {}
        self._previous: Dict[str, Dict[str, Any]] = {}
        # Verify the public joint mapping now, exactly as the frozen visual
        # module does, so a schema change fails at construction and not mid-run.
        arm_joints_from_proprio(np.zeros(84), self.observation_joint_names, self.defaults)
        cv2.setNumThreads(1)

    # ------------------------------------------------------------------ helpers
    def _camera(self, source: str, joints: np.ndarray) -> Tuple[np.ndarray, float]:
        if source == "head":
            return head_camera_transform(), HEAD_CAMERA_FOCAL_LENGTH_MM
        return ee_camera_transform(joints), EE_CAMERA_FOCAL_LENGTH_MM

    def _rejected(self, reason: str, *, reasons: Optional[List[str]] = None,
                  quality: Optional[Dict[str, Any]] = None, timestamp: Optional[Dict[str, Any]] = None,
                  sources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One shape for every non-trustworthy answer, so callers never guess.

        The per-source diagnostics are kept on the refusal path as well: a
        caller that has to brake needs to see *why* nothing was trustworthy.
        """
        return {
            "candidate": True,
            "trustworthy": False,
            "reason": reason,
            "reasons": list(reasons or [reason]),
            "source": None,
            "body_centre_xy": None,
            "body_centre_distance_m": None,
            "body_wall_distance_m": None,
            "body_outer_wall_clearance_m": None,
            "bearing_rad": None,
            "quality": quality or {},
            "frame_to_frame": None,
            "timestamp": timestamp or {},
            "sources": self._public_sources(sources or {}),
            "diagnostics": {},
        }

    @staticmethod
    def _public_sources(sources: Dict[str, Any]) -> Dict[str, Any]:
        """Strip arrays out of the per-source records to keep the report flat."""
        public: Dict[str, Any] = {}
        for name, record in sources.items():
            fit = record.get("fit")
            clean = None
            if isinstance(fit, dict):
                clean = {key: value for key, value in fit.items() if key != "inlier_index"}
                clean = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                         for key, value in clean.items()}
            public[name] = {"available": record.get("available", False),
                            "failures": record.get("failures", []),
                            "estimated": record.get("estimated", False),
                            "diagnostics": record.get("diagnostics", {}),
                            "error": record.get("error"),
                            "fit": clean}
        return public

    def _timestamp(self, source: str, digest: str, control_time_s: float) -> Dict[str, Any]:
        if self._digests.get(source) == digest:
            self._repeat_calls[source] = self._repeat_calls.get(source, 0) + 1
        else:
            self._repeat_calls[source] = 0
        self._digests[source] = digest
        repeats = int(self._repeat_calls.get(source, 0))
        return {
            "call": int(self.calls),
            "control_time_s": float(control_time_s),
            "frame_digest": digest,
            "repeated_frame": repeats > 0,
            "frame_age_calls": repeats,
            "camera_update_period_s": CAMERA_UPDATE_PERIOD_S,
            "observation_age_upper_bound_s": min(CAMERA_UPDATE_PERIOD_S, (repeats + 1) * self.dt),
            "frozen_frame": repeats >= MAX_FROZEN_FRAME_CALLS,
        }

    def _observe_source(self, source: str, depth: np.ndarray, joints: np.ndarray, up: np.ndarray
                        ) -> Dict[str, Any]:
        shape = depth.shape
        body_from_camera, focal_mm = self._camera(source, joints)
        focal, cx, cy = _camera_intrinsics(shape[1], shape[0], focal_mm)
        cells, mask, diagnostics = _wall_point_mask(
            depth, body_from_camera, up, focal=focal, cx=cx, cy=cy, stride=self.stride)
        diagnostics["raster"] = list(shape)
        diagnostics["intrinsics_match_configured"] = tuple(shape) == CONFIGURED_RASTER
        labels, component_count = _cluster_mask(mask)
        diagnostics["wall_components"] = component_count - 1
        diagnostics["blocked"] = diagnostics["blocked_fraction"] > MAX_BLOCKED_FRACTION

        best: Optional[Dict[str, Any]] = None
        for label in range(1, component_count):
            selected = labels == label
            count = int(selected.sum())
            if count < MIN_WALL_POINTS:
                continue
            fit = _fit_known_radius_circle(cells[selected][:, :2], self.radius_m, np.zeros(2))
            if fit is None:
                continue
            if best is None or fit["inlier_count"] > best["inlier_count"]:
                # Height band of the *robust inliers*, not of the whole cluster:
                # a bin wall is a vertical strip, a mislabelled blob is not.
                lifted = cells[selected][fit["inlier_index"]] @ up
                fit["inlier_height_min_m"] = float(lifted.min())
                fit["inlier_height_max_m"] = float(lifted.max())
                fit["inlier_height_span_m"] = float(lifted.max() - lifted.min())
                fit["component_label"] = label
                best = fit
        if best is None:
            return {"source": source, "estimated": False, "diagnostics": diagnostics}
        return {"source": source, "estimated": True, "fit": best, "diagnostics": diagnostics}

    def _gate(self, source: str, result: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Apply the candidate thresholds; return the estimate and every failure."""
        diagnostics = result["diagnostics"]
        failures: List[str] = []
        fit = result.get("fit")
        if fit is None:
            if diagnostics["valid_depth_pixels"] == 0:
                failures.append("insufficient_valid_depth")
            elif diagnostics["vertical_surface_pixels"] == 0:
                failures.append("no_vertical_wall_surface_in_view")
            else:
                failures.append("no_known_radius_wall_cluster_found")
            # A bin wall that is merely close fills the near field legitimately,
            # so near-field depth is only evidence of an occluder when nothing
            # wall-shaped was found at all.
            if diagnostics["blocked"]:
                failures.append("forward_window_likely_occluded")
            return None, failures
        if fit["inlier_count"] < MIN_WALL_POINTS:
            failures.append("too_few_wall_points")
        if fit["residual_rms_m"] > MAX_RESIDUAL_RMS_M:
            failures.append("wall_fit_residual_too_large")
        if fit["residual_max_m"] > MAX_RESIDUAL_ABS_M:
            failures.append("wall_fit_residual_outlier")
        if fit["arc_span_rad"] < math.radians(MIN_WALL_ARC_DEG):
            failures.append("visible_arc_too_short")
        if fit["centre_sigma_m"] > MAX_CENTRE_SIGMA_M:
            failures.append("centre_estimate_too_uncertain")
        centre = fit["centre_xy"]
        standoff = float(np.linalg.norm(centre))
        if not MIN_CENTRE_STANDOFF_M <= standoff <= MAX_CENTRE_STANDOFF_M:
            failures.append("centre_standoff_outside_plausible_band")
        # The surface actually observed is at radius_m; the outermost material of
        # the wall is one facet deeper (the vertex radius), so the clearance the
        # caller may rely on is measured from there.
        wall_distance = standoff - self.radius_m
        outer_clearance = standoff - BARREL_OUTER_RADIUS_M
        if not 0.0 <= wall_distance <= MAX_WALL_DISTANCE_M:
            failures.append("wall_distance_outside_plausible_band")
        if failures:
            return None, failures
        estimate = {
            "source": source,
            "centre_xy": [float(centre[0]), float(centre[1])],
            "centre_distance_m": standoff,
            "wall_distance_m": wall_distance,
            "outer_wall_clearance_m": outer_clearance,
            "bearing_rad": float(math.atan2(centre[1], centre[0])),
            "fit": fit,
        }
        return estimate, []

    # --------------------------------------------------------------------- main
    def observe(self, image: Any, proprio: Sequence[float], *,
                timestamp_s: Optional[float] = None) -> Dict[str, Any]:
        """One public observation.  Returns data only; this never commands.

        Every failure mode the caller has to brake for - no depth, no vertical
        wall in view, too short an arc, an implausible centre, a frozen camera,
        two cameras disagreeing - is a named ``reason`` in the returned record.
        It is never a silent empty result and never a good-looking number.
        """
        self.calls += 1
        control_time = float(timestamp_s) if timestamp_s is not None else self.calls * self.dt
        observation = np.asarray(proprio, dtype=float).reshape(-1)
        if observation.size != 84 or not np.isfinite(observation).all():
            return self._rejected("invalid_public_proprio")
        gravity = observation[9:12]
        gravity_norm = float(np.linalg.norm(gravity))
        if not math.isfinite(gravity_norm) or gravity_norm < 0.5:
            return self._rejected("unusable_projected_gravity")
        up = -gravity / gravity_norm
        joints = arm_joints_from_proprio(observation, self.observation_joint_names, self.defaults)

        estimates: Dict[str, Any] = {}
        sources: Dict[str, Any] = {}
        timestamps: Dict[str, Any] = {}
        for source in self.sources:
            try:
                _, depth = _rgbd_arrays(image, source)
            except (ValueError, TypeError) as error:
                sources[source] = {"source": source, "available": False, "error": str(error)}
                continue
            digest = _frame_digest(depth)
            timestamps[source] = self._timestamp(source, digest, control_time)
            result = self._observe_source(source, depth, joints, up)
            estimate, failures = self._gate(source, result)
            failures += (["frozen_camera_frame"] if timestamps[source]["frozen_frame"] else [])
            sources[source] = {"available": True, "failures": failures,
                               "diagnostics": result["diagnostics"],
                               "fit": result.get("fit"),
                               "estimated": estimate is not None}
            if estimate is not None and not failures:
                estimates[source] = estimate

        if not estimates:
            reasons = sorted({reason for record in sources.values() for reason in record.get("failures", [])})
            timestamp = timestamps.get(self.sources[0], {})
            return self._rejected("no_trustworthy_barrel_observation", reasons=reasons or ["no_source_available"],
                                  timestamp=timestamp, sources=sources)

        chosen = min(estimates, key=lambda name: estimates[name]["fit"]["centre_sigma_m"])
        reasons: List[str] = []
        if len(estimates) > 1:
            names = sorted(estimates)
            first, second = estimates[names[0]], estimates[names[1]]
            disagreement = float(np.linalg.norm(np.asarray(first["centre_xy"]) - np.asarray(second["centre_xy"])))
            if disagreement > MAX_SOURCE_DISAGREEMENT_M:
                reasons.append("sources_disagree_on_centre")
        estimate = estimates[chosen]
        fit = estimate["fit"]
        if reasons:
            return self._rejected("sources_disagree_on_centre", reasons=reasons,
                                  quality={"source": chosen, "centre_disagreement_m": disagreement},
                                  timestamp=timestamps[chosen], sources=sources)

        previous = self._previous.get(chosen)
        centre = np.asarray(estimate["centre_xy"], dtype=float)
        frame_to_frame = None
        if previous is not None:
            change = centre - np.asarray(previous["centre_xy"], dtype=float)
            elapsed = max(control_time - previous["control_time_s"], self.dt)
            frame_to_frame = {
                "centre_change_m": float(np.linalg.norm(change)),
                "wall_distance_change_m": float(estimate["wall_distance_m"] - previous["wall_distance_m"]),
                "elapsed_s": float(elapsed),
                "centre_rate_m_per_s": float(np.linalg.norm(change) / elapsed),
                "bearing_change_rad": float(np.arctan2(math.sin(estimate["bearing_rad"] - previous["bearing_rad"]),
                                                        math.cos(estimate["bearing_rad"] - previous["bearing_rad"]))),
            }
        self._previous[chosen] = {"centre_xy": centre.copy(),
                                  "wall_distance_m": estimate["wall_distance_m"],
                                  "bearing_rad": estimate["bearing_rad"],
                                  "control_time_s": control_time}

        arc = float(fit["arc_span_rad"])
        # Sagitta of a straight chord over the same arc against the fitted
        # radius: this is the margin by which the fit distinguishes the real
        # wall from a flat surface that merely happens to be nearby.
        sagitta = (self.radius_m * arc) ** 2 / (8.0 * self.radius_m)
        surface_vertex = (BARREL_OUTER_RADIUS_M if self.observed_surface == "outer"
                          else BARREL_INNER_WALL_VERTEX_RADIUS_M)
        timestamp = timestamps[chosen]
        quality = {
            "source": chosen,
            "valid_depth_points": sources[chosen]["diagnostics"]["valid_depth_pixels"],
            "wall_points": fit["inlier_count"],
            "inlier_count": fit["inlier_count"],
            "fit_residual_rms_m": fit["residual_rms_m"],
            "fit_residual_max_m": fit["residual_max_m"],
            "visible_arc_span_deg": float(math.degrees(arc)),
            "visible_arc_span_rad": arc,
            "centre_sigma_m": fit["centre_sigma_m"],
            "flat_wall_sagitta_m": float(sagitta),
            "inlier_height_min_m": fit.get("inlier_height_min_m"),
            "inlier_height_max_m": fit.get("inlier_height_max_m"),
            "inlier_height_span_m": fit.get("inlier_height_span_m"),
            "observed_surface": self.observed_surface,
            "radius_used_m": self.radius_m,
            "inner_wall_inscribed_radius_m": BARREL_INNER_WALL_INSCRIBED_RADIUS_M,
            "radius_model_bias_bound_m": float(surface_vertex - self.radius_m),
            "radius_model_bias_direction": "the fitted centre can sit up to that far from the observed "
                                           "wall, i.e. the reported standoff can be over-stated by it",
            "bearing_rad": estimate["bearing_rad"],
            "frozen_frame": timestamp["frozen_frame"],
            "repeated_frame": timestamp["repeated_frame"],
        }
        return {
            "candidate": True,
            "trustworthy": True,
            "reason": "wall_observation_quality_gates_passed",
            "reasons": [],
            "source": chosen,
            "body_centre_xy": estimate["centre_xy"],
            "body_centre_distance_m": estimate["centre_distance_m"],
            "body_wall_distance_m": estimate["wall_distance_m"],
            "body_outer_wall_clearance_m": estimate["outer_wall_clearance_m"],
            "bearing_rad": estimate["bearing_rad"],
            "quality": quality,
            "frame_to_frame": frame_to_frame,
            "timestamp": timestamp,
            "sources": self._public_sources(sources),
            "diagnostics": {
                "claim": "body-frame bin wall estimate from public depth geometry only; "
                         "residual is a fit residual, not an absolute position error",
                "centre_uncertainty_excludes": ["depth sensor systematic error", "camera extrinsic error",
                                                "the 32-gon radius model bias bound"],
                "seed_used": "the fitted centre is seeded from the observed points only; the public "
                             "static bin centre is deliberately never read, so this cannot echo it",
            },
        }

    def reset(self) -> None:
        """Drop frame-to-frame history; does not touch pose or any estimate."""
        self.calls = 0
        self._digests.clear()
        self._repeat_calls.clear()
        self._previous.clear()

    def describe(self) -> Dict[str, Any]:
        """Plain statement of what this is, what it needs, and what it is not."""
        return {
            "name": "barrel_observation",
            "role": "public RGB-D bin wall/rim observation and body-frame centre estimate with quality data",
            "status": "candidate and hypothesis, not a verified capability",
            "object": {"constants_are_public_static_task_configuration": True,
                       "outer_radius_m": BARREL_OUTER_RADIUS_M,
                       "wall_thickness_m": BARREL_WALL_THICKNESS_M,
                       "wall_height_m": BARREL_WALL_HEIGHT_M,
                       "rim_world_z_m": BARREL_RIM_Z_M,
                       "wall_segments": BARREL_WALL_SEGMENTS,
                       "inner_wall_vertex_radius_m": BARREL_INNER_WALL_VERTEX_RADIUS_M,
                       "inner_wall_inscribed_radius_m": BARREL_INNER_WALL_INSCRIBED_RADIUS_M,
                       "outer_wall_inscribed_radius_m": BARREL_OUTER_INSCRIBED_RADIUS_M,
                       "why_inscribed": ".98 m is the inner vertex radius of the 32-gon; only "
                                        ".98*cos(pi/32) bounds every azimuth, and the same rule is "
                                        "applied to the outer surface this camera observes",
                       "surface_choice_bias_m": BARREL_SURFACE_CHOICE_BIAS_M,
                       "surface_choice_note": "fitting the inner inscribed radius to the outer wall "
                                              "surface moves the centre 22.5 mm towards the camera "
                                              "(synthetic measurement)"},
            "inputs": ["public image dict with head_rgb/head_depth and/or ee_rgb/ee_depth",
                       "public 84-element proprio for the absolute arm joints and projected gravity",
                       "static joint/action/camera configuration"],
            "never_read": ["object root or any ground-truth pose", "the public static bin centre as a fit seed",
                           "seed maps", "world base pose", "reward, score or termination",
                           "contact forces", "any planner state"],
            "outputs": ["body-frame bin centre xy, its distance and bearing",
                        "body-origin-to-wall distance for the observed surface, plus the conservative "
                        "outer wall clearance and the inner wall inscribed radius",
                        "valid depth point count, inliers, fit residual, visible arc span, "
                        "centre covariance, frame-to-frame change, frame timestamp and freshness",
                        "one named reason whenever the observation is not trustworthy"],
            "what_was_actually_observed_in_p13": {
                "head_camera": "never contains the bin; at the final pose the wall is 2.95-4.97 m "
                               "away and projects above the top image edge, and the saved head frames "
                               "are ground texture only",
                "wrist_camera": "sees the bin at ~5-8 m during steps ~500-1500, then points at the "
                                "ground and the carried bottle",
                "bin_colour": "neutral (171,174,176) in the saved frames, the same neutrality as the "
                              "ground, so appearance cannot identify it",
                "consequence": "usable envelope is short range and roughly ahead; sustained "
                               "long-range visibility is not assumed",
            },
            "parameters": {"dt": self.dt, "sources": list(self.sources), "observed_surface": self.observed_surface,
                           "radius_m": self.radius_m, "stride_px": self.stride,
                           "min_wall_points": MIN_WALL_POINTS, "min_wall_arc_deg": MIN_WALL_ARC_DEG,
                           "max_residual_rms_m": MAX_RESIDUAL_RMS_M,
                           "max_centre_sigma_m": MAX_CENTRE_SIGMA_M,
                           "all_thresholds_are_initial_candidates": True},
            "known_risks": [
                "no real recorded depth frame has ever been run through this estimator; the saved "
                "p13 artefacts are RGB only, so the thresholds are candidates",
                "a flat vertical surface fits the known radius over a short arc about as well as the "
                "real wall does; the reported flat_wall_sagitta_m is the visible margin",
                "the robot's own arm or the carried bottle can appear as a vertical surface, and the "
                "near-field occlusion test is deliberately weaker whenever a wall-shaped cluster fits",
                "observed_surface='outer' is a declared hypothesis about the viewpoint; an estimate "
                "taken from inside the bin would need the other radius",
                "camera extrinsics and depth sensor bias are excluded from the reported centre sigma",
                "the head camera cannot see this bin beyond about 3.2 m of body-centre standoff and "
                "the wrist camera cannot see it past the carried bottle in the payload posture",
                "a low fit residual is never a low absolute position error and is never reported as one",
            ],
            "not_verified": ["no score, delivery, clearance or docking accuracy is claimed",
                             "no threshold here has been validated against a real run"],
            "claim": "if the reported gates pass, the depth in front of one camera is consistent with a "
                     "cylinder of the known bin radius over a sufficient arc. Nothing more.",
            "calls": self.calls,
        }
