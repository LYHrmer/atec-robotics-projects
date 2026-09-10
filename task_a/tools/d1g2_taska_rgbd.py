"""Small RGB-D visual odometer using only camera measurements (NumPy/OpenCV).

Depth is camera-Z depth in metres (Isaac ``distance_to_image_plane``), not
Euclidean distance to the camera. RGB and depth must be registered and K must
describe the unrectified-distortion-free pinhole image provided here.

Relative transforms map CURRENT body coordinates into the LAST ACCEPTED body
frame. Rejected images never replace that reference frame. The accumulated
pose starts at identity, or at an explicitly supplied relative sensor estimate
when bootstrapping after settling; it never reads simulator pose or position.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


R_BODY_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
T_BODY_CAMERA = np.array([0.4075, 0., 0.10])


def body_camera_rotation(mount_pitch_deg=0.):
    """Fixed camera-to-body rotation; positive pitch points the camera down.

    The zero-pitch optical +Z axis is body +X. Left-multiplying by body
    R_y(+pitch) changes it to [cos(pitch), 0, -sin(pitch)]. This is the mount
    setting only; it does not measure or correct the robot's attitude.
    """
    pitch = float(mount_pitch_deg)
    if not math.isfinite(pitch):
        raise ValueError("mount_pitch_deg must be finite")
    angle = math.radians(pitch)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]) @ R_BODY_CAMERA


class RGBDOdometry:
    """LK tracks + depth backprojection + robust PnP, with ORB/SIFT fallback.

    ``update(rgb, depth, K, dt, inertial_relative_guess=None,
    projected_gravity=None, axis_heading_result=None)`` returns a dict.
    The optional guess is a dict with ``relative_body_rotation`` (3x3) and
    ``relative_body_translation`` (3,), or a 4x4 homogeneous transform. It
    must cover the interval since the last accepted reference, not merely
    the latest frame. It initializes PnP and never bypasses visual checks.

    Optional same-frame projected gravity corrects accumulated roll/pitch
    after an accepted visual measurement, preserving visual yaw. Corrected
    rotations are also saved in recovery keyframes. Rejections hold pose.
    Optional ``axis_heading_result`` is an externally gated sensor estimate
    with ``accepted``, ``heading_rad`` (in this odometer's local frame), and
    optional ``source``. After visual acceptance, a finite heading within
    10 degrees of visual yaw replaces yaw while retaining sensed roll/pitch.
    Invalid or rejected headings leave the accepted visual estimate intact.
    This never changes position retrospectively or bypasses visual checks.
    K must already use OpenCV pixel coordinates; no half-pixel conversion
    is performed here because ordinary calibrated/synthetic K needs none.

    ``accepted=False`` holds the estimated pose. ``initialized=True`` means
    a usable reference exists. ``lost`` marks a rejected post-initialization
    update. After ``max_reference_age_s`` direct tracking expires, but a small
    bank of earlier accepted viewpoints can still recover the original local
    coordinate frame. Recovery requires two independent keyframes to agree
    and a bounded change from the last accepted pose; it never resets pose.
    All statistics and returned transforms are JSON-safe.
    """

    def __init__(self, *, max_features=1000, min_inliers=18, min_inlier_ratio=.4,
                 max_reprojection_px=2.5, min_depth=.15, max_depth=35.,
                 max_speed_m_s=3., max_angular_speed_rad_s=5.,
                 max_reference_age_s=5., orb_fallback=True,
                 keyframe_interval_s=.5, keyframe_history_s=12., max_keyframes=24,
                 recovery_search_interval_s=.2, recovery_max_jump_m=1.,
                 recovery_min_inliers=35, recovery_max_candidates=8,
                 mount_pitch_deg=0., sift_fallback=True):
        self.max_features = int(max_features)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reprojection_px = float(max_reprojection_px)
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_angular_speed_rad_s = float(max_angular_speed_rad_s)
        self.max_reference_age_s = float(max_reference_age_s)
        self.orb_fallback = bool(orb_fallback)
        self.sift_fallback = bool(sift_fallback)
        self.mount_pitch_deg = float(mount_pitch_deg)
        self.R_body_camera = body_camera_rotation(self.mount_pitch_deg)
        self.t_body_camera = T_BODY_CAMERA.copy()
        self.keyframe_interval_s = float(keyframe_interval_s)
        self.keyframe_history_s = float(keyframe_history_s)
        self.max_keyframes = int(max_keyframes)
        self.recovery_search_interval_s = float(recovery_search_interval_s)
        self.recovery_max_jump_m = float(recovery_max_jump_m)
        self.recovery_min_inliers = int(recovery_min_inliers)
        self.recovery_max_candidates = int(recovery_max_candidates)
        if (self.keyframe_interval_s <= 0 or self.keyframe_history_s <= 0
                or self.max_keyframes < 0 or self.recovery_search_interval_s < 0
                or self.recovery_max_jump_m <= 0 or self.recovery_min_inliers < self.min_inliers
                or self.recovery_max_candidates < 2):
            raise ValueError("Invalid keyframe recovery configuration")
        self._orb = cv2.ORB_create(nfeatures=self.max_features, fastThreshold=12)
        self._sift = (cv2.SIFT_create(nfeatures=2*self.max_features, contrastThreshold=.01)
                      if self.sift_fallback else None)
        self.reset()

    def reset(self, initial_rotation=None, initial_position=None):
        """Clear tracking, optionally retaining a caller's relative pose estimate.

        Supplied values must be estimates in the original local body frame,
        obtained from permitted sensors, not simulator diagnostic truth. A
        reset creates a new reference; it does not recover motion during a
        previous tracking gap. Omitted values retain the identity-origin API.
        """
        rotation = np.eye(3) if initial_rotation is None else np.asarray(initial_rotation, dtype=np.float64)
        position = np.zeros(3) if initial_position is None else np.asarray(initial_position, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("initial_rotation must be a finite 3x3 rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0):
            raise ValueError("initial_rotation must be orthonormal with determinant +1")
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("initial_position must be a finite 3-vector")
        self.rotation = rotation.copy()
        self.position = position.copy()
        self._initialization_mode = "identity" if initial_rotation is None and initial_position is None else "supplied_relative_estimate"
        self._initial_rotation = self.rotation.copy()
        self._initial_position = self.position.copy()
        self._gray = self._depth = self._K = self._points = None
        self._reference_age = 0.
        self._elapsed_time = 0.
        self._reference_frame = None
        self._keyframes = []
        self._last_recovery_search_time = -float("inf")
        self._recovery_count = 0
        self._gravity_corrected_frames = 0
        self._axis_corrected_frames = 0
        self._last_axis_heading_source = None
        self._frames = self._accepted = self._rejected = 0
        self._last = None

    @staticmethod
    def rotation_from_gravity_yaw(projected_gravity, estimated_yaw):
        """Body-to-local rotation using sensed roll/pitch and estimated yaw."""
        gravity = np.asarray(projected_gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-8:
            raise ValueError("projected_gravity must be a finite nonzero 3-vector")
        if not math.isfinite(float(estimated_yaw)):
            raise ValueError("estimated_yaw must be finite")
        gx, gy, gz = gravity / np.linalg.norm(gravity)
        roll = math.atan2(-gy, -gz)
        pitch = math.atan2(gx, math.hypot(gy, gz))
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(estimated_yaw), math.cos(estimated_yaw)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])

    def initialize_from_proprio(self, rgb, depth, K, proprio, estimated_xy, estimated_yaw):
        """Establish the first reference after settling, without assuming yaw=0.

        ``proprio`` is the normal 81-vector; only projected gravity [9:12]
        supplies roll/pitch. The caller provides its *short-term sensor-only*
        XY/yaw estimates accumulated from the actual original start. Z is
        assigned zero as a translation datum, not claimed as measured height.
        This cannot be called after a visual reference is established. A
        featureless image leaves initialization pending and can be retried
        with the caller's updated estimates. It never resets a tracked pose.
        """
        if self._gray is not None:
            raise RuntimeError("A visual reference already exists; initialization cannot silently reset it")
        proprio = np.asarray(proprio, dtype=np.float64)
        xy = np.asarray(estimated_xy, dtype=np.float64)
        if proprio.shape != (81,) or not np.isfinite(proprio[9:12]).all():
            raise ValueError("proprio must have shape (81,) with finite projected gravity")
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("estimated_xy must be a finite 2-vector")
        rotation = self.rotation_from_gravity_yaw(proprio[9:12], estimated_yaw)
        self.reset(rotation, np.array([xy[0], xy[1], 0.]))
        self._initialization_mode = "proprio_bootstrap"
        return self.update(rgb, depth, K, .1)

    def state_dict(self):
        return {
            "initialized": self._gray is not None,
            "estimated_xy": self.position[:2].tolist(),
            "estimated_yaw": float(math.atan2(self.rotation[1, 0], self.rotation[0, 0])),
            "estimated_position": self.position.tolist(),
            "estimated_rotation": self.rotation.tolist(),
            "reference_age_s": self._reference_age,
            "frames": self._frames, "accepted_frames": self._accepted,
            "rejected_frames": self._rejected,
            "pose_frame": "initial_body_relative_frame",
            "initialization_mode": self._initialization_mode,
            "initial_rotation": self._initial_rotation.tolist(),
            "initial_position": self._initial_position.tolist(),
            "keyframe_count": len(self._keyframes),
            "recovery_count": self._recovery_count,
            "gravity_corrected_frames": self._gravity_corrected_frames,
            "axis_corrected_frames": self._axis_corrected_frames,
            "last_axis_heading_source": self._last_axis_heading_source,
            "sift_fallback_enabled": self.sift_fallback,
            "camera_mount_pitch_deg": self.mount_pitch_deg,
            "R_body_camera": self.R_body_camera.tolist(),
            "t_body_camera": self.t_body_camera.tolist(),
        }

    def _result(self, accepted, reason, **statistics):
        if not accepted and self._gray is not None and reason != "initialized":
            self._rejected += 1
        result = {
            "accepted": accepted, "reason": reason,
            "confidence": 0., "inliers": 0, "matches": 0, "reproj": None,
            "relative_body_rotation": np.eye(3).tolist(),
            "relative_body_translation": [0., 0., 0.],
            "lost": not accepted and self._gray is not None and reason != "initialized",
            **self.state_dict(), **statistics,
        }
        self._last = result
        return result

    @staticmethod
    def _image(rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or min(rgb.shape[:2]) < 32:
            raise ValueError("RGB must have shape (H,W,3|4), with H,W >= 32")
        if not np.isfinite(rgb).all():
            raise ValueError("RGB contains non-finite pixels")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255. if np.max(rgb) <= 1. else 1.), 0, 255).astype(np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2GRAY)

    def _features(self, gray, depth):
        mask = ((depth >= self.min_depth) & (depth <= self.max_depth) & np.isfinite(depth)).astype(np.uint8) * 255
        # Avoid feature points exactly on a discontinuous/invalid depth edge.
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        points = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features,
                                        qualityLevel=.01, minDistance=7,
                                        mask=mask, blockSize=7)
        return np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)

    def _set_reference(self, gray, depth, K):
        self._gray, self._depth, self._K = gray.copy(), depth.copy(), K.copy()
        self._points = self._features(gray, depth)
        self._reference_age = 0.
        self._reference_frame = self._frames

    def _archive_reference(self, statistics=None):
        if not self.max_keyframes:
            return
        if self._keyframes and self._elapsed_time - self._keyframes[-1]["time"] < self.keyframe_interval_s - 1e-9:
            return
        if statistics is not None and (statistics.get("inliers", 0) < self.recovery_min_inliers
                                       or statistics.get("reproj", float("inf")) > 1.5):
            return
        # Frame arrays are immutable after _set_reference replaces them, so
        # retaining references avoids an extra image/depth copy per keyframe.
        self._keyframes.append({
            "time": self._elapsed_time, "frame": self._reference_frame,
            "gray": self._gray, "depth": self._depth, "K": self._K, "points": self._points,
            "rotation": self.rotation.copy(), "position": self.position.copy(),
        })
        # Prune while tracking advances, not while the robot waits after loss.
        # Otherwise an expired current view would also erase all recovery views.
        self._keyframes = [keyframe for keyframe in self._keyframes
                           if self._elapsed_time - keyframe["time"] <= self.keyframe_history_s][-self.max_keyframes:]

    def _track_lk(self, gray, wide=False):
        if self._points is None or len(self._points) < self.min_inliers:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        old = self._points.reshape(-1, 1, 2)
        window, levels, eigenvalue = ((61, 61), 4, 1e-7) if wide else ((25, 25), 3, 1e-5)
        new, status, error = cv2.calcOpticalFlowPyrLK(
            self._gray, gray, old, None, winSize=window, maxLevel=levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
            minEigThreshold=eigenvalue,
        )
        if new is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._gray, new, None, winSize=window, maxLevel=levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
            minEigThreshold=eigenvalue,
        )
        good = status.ravel().astype(bool) & backward_status.ravel().astype(bool)
        good &= np.linalg.norm(old.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 1.
        good &= error.ravel() < 40.
        return old.reshape(-1, 2)[good], new.reshape(-1, 2)[good]

    def _track_orb(self, gray):
        kp0, des0 = self._orb.detectAndCompute(self._gray, None)
        kp1, des1 = self._orb.detectAndCompute(gray, None)
        if des0 is None or des1 is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        candidates = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
        good = [pair[0] for pair in candidates if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
        # Enforce unique current-image features to prevent duplicate votes.
        unique = {}
        for match in sorted(good, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        return (np.asarray([kp0[m.queryIdx].pt for m in unique.values()], np.float32).reshape(-1, 2),
                np.asarray([kp1[m.trainIdx].pt for m in unique.values()], np.float32).reshape(-1, 2))

    def _body_to_camera_relative(self, rotation, translation):
        camera_translation = self.R_body_camera.T @ (rotation @ self.t_body_camera + translation - self.t_body_camera)
        camera_rotation = self.R_body_camera.T @ rotation @ self.R_body_camera
        return camera_rotation, camera_translation

    def _track_sift(self, gray, depth):
        """Scale-invariant last fallback for rapid near-ground viewpoint changes."""
        reference_mask = ((self._depth > self.min_depth) & (self._depth < self.max_depth)
                          & np.isfinite(self._depth)).astype(np.uint8) * 255
        current_mask = ((depth > self.min_depth) & (depth < self.max_depth)
                        & np.isfinite(depth)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        keypoints0, descriptors0 = self._sift.detectAndCompute(self._gray, cv2.erode(reference_mask, kernel))
        keypoints1, descriptors1 = self._sift.detectAndCompute(gray, cv2.erode(current_mask, kernel))
        if (descriptors0 is None or descriptors1 is None
                or len(descriptors0) < 2 or len(descriptors1) < 2):
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors0, descriptors1, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
        unique = {}
        for match in sorted(good, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        return (np.asarray([keypoints0[m.queryIdx].pt for m in unique.values()], np.float32).reshape(-1, 2),
                np.asarray([keypoints1[m.trainIdx].pt for m in unique.values()], np.float32).reshape(-1, 2))

    def _estimate(self, previous, current, K, depth, guess):
        height, width = self._depth.shape
        valid = np.isfinite(previous).all(1) & np.isfinite(current).all(1)
        valid &= (previous[:, 0] >= 1) & (previous[:, 0] < width - 1)
        valid &= (previous[:, 1] >= 1) & (previous[:, 1] < height - 1)
        valid &= (current[:, 0] >= 1) & (current[:, 0] < width - 1)
        valid &= (current[:, 1] >= 1) & (current[:, 1] < height - 1)
        previous, current = previous[valid], current[valid]
        pixels = np.rint(previous).astype(int)
        z = self._depth[pixels[:, 1], pixels[:, 0]]
        valid = np.isfinite(z) & (z >= self.min_depth) & (z <= self.max_depth)
        previous, current, z = previous[valid], current[valid], z[valid]
        stats = {"matches": int(len(z))}
        if len(z) < self.min_inliers:
            return None, "insufficient_depth_matches", stats
        homogeneous = np.column_stack((previous, np.ones(len(previous))))
        objects = (homogeneous @ np.linalg.inv(self._K).T) * z[:, None]
        objects = np.ascontiguousarray(objects, dtype=np.float64)
        current = np.ascontiguousarray(current, dtype=np.float64)
        kwargs = {}
        if guess is not None:
            if isinstance(guess, dict):
                r = np.asarray(guess["relative_body_rotation"], np.float64)
                t = np.asarray(guess["relative_body_translation"], np.float64)
            else:
                guess = np.asarray(guess, np.float64)
                r, t = guess[:3, :3], guess[:3, 3]
            if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
                return None, "invalid_inertial_guess", stats
            camera_r, camera_t = self._body_to_camera_relative(r, t)
            kwargs = {"rvec": cv2.Rodrigues(camera_r.T)[0],
                      "tvec": (-camera_r.T @ camera_t).reshape(3, 1), "useExtrinsicGuess": True}
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects, current, K, None, iterationsCount=120,
            reprojectionError=self.max_reprojection_px, confidence=.999,
            # ITERATIVE also handles the near-planar floor. EPNP's final
            # all-inlier solution can select a wrong planar pose that LM
            # cannot recover, despite hundreds of consistent RANSAC inliers.
            flags=cv2.SOLVEPNP_ITERATIVE,
            **kwargs,
        )
        if not success or inliers is None:
            return None, "pnp_failed", stats
        ids = inliers.ravel()
        stats.update(inliers=int(len(ids)), inlier_ratio=float(len(ids) / len(z)))
        if len(ids) < self.min_inliers or len(ids) / len(z) < self.min_inlier_ratio:
            return None, "insufficient_pnp_inliers", stats
        rvec, tvec = cv2.solvePnPRefineLM(objects[ids], current[ids], K, None, rvec, tvec)
        pnp_rotation = cv2.Rodrigues(rvec)[0]
        transformed = objects[ids] @ pnp_rotation.T + tvec.reshape(1, 3)
        if np.count_nonzero(transformed[:, 2] > self.min_depth) < .95 * len(ids):
            return None, "points_behind_camera", stats
        projected = cv2.projectPoints(objects[ids], rvec, tvec, K, None)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - current[ids], axis=1)
        stats.update(reproj=float(np.median(errors)), reproj_rms=float(np.sqrt(np.mean(errors ** 2))))
        if stats["reproj"] > self.max_reprojection_px or stats["reproj_rms"] > self.max_reprojection_px * 1.2:
            return None, "reprojection_error", stats
        # Current depth is an independent consistency check where available.
        uv = np.rint(current[ids]).astype(int)
        measured = depth[uv[:, 1], uv[:, 0]]
        measurable = np.isfinite(measured) & (measured >= self.min_depth) & (measured <= self.max_depth)
        if np.count_nonzero(measurable) >= self.min_inliers:
            dz = np.abs(measured[measurable] - transformed[measurable, 2])
            consistent = dz < np.maximum(.10, .06 * transformed[measurable, 2])
            stats["current_depth_consistency"] = float(np.mean(consistent))
            if np.mean(consistent) < .6:
                return None, "depth_inconsistent", stats
        spread = np.ptp(current[ids], axis=0)
        stats["image_coverage"] = float(np.prod(spread) / (height * width))
        if stats["image_coverage"] < .015:
            return None, "features_too_concentrated", stats
        # PnP maps previous camera -> current camera. Invert before composing.
        camera_rotation = pnp_rotation.T
        camera_translation = -camera_rotation @ tvec.reshape(3)
        body_rotation = self.R_body_camera @ camera_rotation @ self.R_body_camera.T
        body_translation = self.R_body_camera @ camera_translation + self.t_body_camera - body_rotation @ self.t_body_camera
        if not np.isfinite(body_rotation).all() or not np.isfinite(body_translation).all():
            return None, "nonfinite_motion", stats
        angle = float(np.linalg.norm(cv2.Rodrigues(body_rotation)[0]))
        distance = float(np.linalg.norm(body_translation))
        stats.update(translation_norm=distance, rotation_angle=angle)
        if distance > .025 + self.max_speed_m_s * self._reference_age:
            return None, "translation_rate_exceeded", stats
        if angle > .02 + self.max_angular_speed_rad_s * self._reference_age:
            return None, "rotation_rate_exceeded", stats
        stats["confidence"] = float(min(1., len(ids) / 100.) * stats["inlier_ratio"] * math.exp(-stats["reproj"] / self.max_reprojection_px))
        return (body_rotation, body_translation), "visual_motion", stats

    def _track_reference(self, gray, depth, K, guess):
        previous, current = self._track_lk(gray)
        motion, reason, statistics = self._estimate(previous, current, K, depth, guess)
        statistics["tracker"] = "lk"
        if motion is None:
            # Large windows help low-contrast floor tiles, but can average
            # incompatible flows during rotation; retain geometric checks.
            previous, current = self._track_lk(gray, wide=True)
            alternate, alternate_reason, alternate_stats = self._estimate(previous, current, K, depth, guess)
            if alternate is not None or alternate_stats.get("inliers", 0) > statistics.get("inliers", 0):
                motion, reason, statistics = alternate, alternate_reason, alternate_stats
                statistics["tracker"] = "lk_wide"
        if motion is None and self.orb_fallback:
            previous, current = self._track_orb(gray)
            alternate, alternate_reason, alternate_stats = self._estimate(previous, current, K, depth, guess)
            if alternate is not None or alternate_stats.get("inliers", 0) > statistics.get("inliers", 0):
                motion, reason, statistics = alternate, alternate_reason, alternate_stats
                statistics["tracker"] = "orb"
        if motion is None and self.sift_fallback:
            previous, current = self._track_sift(gray, depth)
            alternate, alternate_reason, alternate_stats = self._estimate(previous, current, K, depth, guess)
            if alternate is not None or alternate_stats.get("inliers", 0) > statistics.get("inliers", 0):
                motion, reason, statistics = alternate, alternate_reason, alternate_stats
                statistics["tracker"] = "sift"
        return motion, reason, statistics

    def _recover_from_keyframes(self, gray, depth, K):
        """Recover a global estimate via stored visual poses, never a reset."""
        info = {"recovery_attempted": False, "recovery_candidates_tested": 0}
        if len(self._keyframes) < 2:
            return None, {**info, "recovery_reason": "insufficient_keyframe_history"}
        if self._elapsed_time - self._last_recovery_search_time < self.recovery_search_interval_s - 1e-9:
            return None, {**info, "recovery_reason": "search_interval"}
        self._last_recovery_search_time = self._elapsed_time
        info["recovery_attempted"] = True
        saved_reference = (self._gray, self._depth, self._K, self._points, self._reference_age)
        last_rotation, last_position = self.rotation.copy(), self.position.copy()
        last_age = self._reference_age
        candidates = [keyframe for keyframe in reversed(self._keyframes)
                      if keyframe["frame"] != self._reference_frame][:self.recovery_max_candidates]
        usable = []
        chosen = support = None
        try:
            for keyframe in candidates:
                self._gray, self._depth, self._K, self._points = (
                    keyframe["gray"], keyframe["depth"], keyframe["K"], keyframe["points"]
                )
                self._reference_age = self._elapsed_time - keyframe["time"]
                motion, _, statistics = self._track_reference(gray, depth, K, None)
                info["recovery_candidates_tested"] += 1
                if motion is None:
                    continue
                if (statistics["inliers"] < self.recovery_min_inliers
                        or statistics["inlier_ratio"] < .5 or statistics["reproj"] > 1.5
                        or statistics["confidence"] < .25):
                    continue
                relative_rotation, relative_translation = motion
                rotation = keyframe["rotation"] @ relative_rotation
                position = keyframe["position"] + keyframe["rotation"] @ relative_translation
                jump = float(np.linalg.norm(position - last_position))
                angle = float(np.linalg.norm(cv2.Rodrigues(last_rotation.T @ rotation)[0]))
                if (jump > min(self.recovery_max_jump_m, .05 + self.max_speed_m_s * last_age)
                        or angle > min(1.75, .05 + self.max_angular_speed_rad_s * last_age)):
                    continue
                candidate = {"rotation": rotation, "position": position, "keyframe": keyframe,
                             "statistics": statistics, "jump": jump, "angle": angle}
                # Require two distinct old views to reconstruct the SAME current
                # pose. This rejects many repeated-tile/locally plausible aliases.
                for earlier in usable:
                    position_disagreement = float(np.linalg.norm(position - earlier["position"]))
                    rotation_disagreement = float(np.linalg.norm(cv2.Rodrigues(earlier["rotation"].T @ rotation)[0]))
                    if position_disagreement <= .12 and rotation_disagreement <= .08:
                        chosen = max((candidate, earlier), key=lambda item: item["statistics"]["confidence"])
                        support = (candidate, earlier, position_disagreement, rotation_disagreement)
                        break
                if chosen is not None:
                    break
                usable.append(candidate)
        finally:
            self._gray, self._depth, self._K, self._points, self._reference_age = saved_reference
        if chosen is None:
            return None, {**info, "recovery_reason": "no_consistent_keyframe_match", "recovery_usable_candidates": len(usable)}
        info.update(
            recovery_reason="two_keyframes_agree",
            recovery_reference_frame=chosen["keyframe"]["frame"],
            recovery_reference_age_s=self._elapsed_time - chosen["keyframe"]["time"],
            recovery_supporting_keyframes=[support[0]["keyframe"]["frame"], support[1]["keyframe"]["frame"]],
            recovery_position_disagreement_m=support[2],
            recovery_rotation_disagreement_rad=support[3],
            recovery_pose_jump_m=chosen["jump"],
            recovery_primary_reference_expired=last_age > self.max_reference_age_s,
        )
        # Public relative motion remains current -> LAST accepted body, although
        # the measurement was obtained through an older keyframe's local frame.
        relative = (last_rotation.T @ chosen["rotation"], last_rotation.T @ (chosen["position"] - last_position))
        return (relative, chosen["rotation"], chosen["position"]), {**chosen["statistics"], **info}

    @staticmethod
    def _validated_axis_heading(result, visual_yaw):
        """Defensive checks only; image evidence gating belongs to the caller."""
        statistics = {"axis_heading_applied": False}
        if not isinstance(result, dict):
            return None, {**statistics, "axis_heading_reason": "invalid_result"}
        source = result.get("source", "external_axis_heading")
        statistics["axis_heading_source"] = source[:120] if isinstance(source, str) else "external_axis_heading"
        if not isinstance(result.get("accepted"), (bool, np.bool_)) or not result["accepted"]:
            return None, {**statistics, "axis_heading_reason": "external_rejection"}
        try:
            value = result["heading_rad"]
            if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
                raise ValueError("heading must be a scalar angle")
            heading = float(value)
            if not math.isfinite(heading):
                raise ValueError("heading must be finite")
            innovation = math.atan2(math.sin(heading - visual_yaw), math.cos(heading - visual_yaw))
        except (ValueError, TypeError, KeyError, OverflowError):
            return None, {**statistics, "axis_heading_reason": "invalid_heading"}
        statistics["axis_heading_innovation_rad"] = innovation
        if abs(innovation) > math.radians(10.):
            return None, {**statistics, "axis_heading_reason": "innovation_exceeded"}
        statistics.update(axis_heading_applied=True, axis_heading_reason="accepted",
                          axis_yaw_correction_rad=innovation)
        return visual_yaw + innovation, statistics

    def update(self, rgb, depth, K, dt, inertial_relative_guess=None, projected_gravity=None,
               axis_heading_result=None):
        self._frames += 1
        if not math.isfinite(float(dt)) or dt <= 0:
            return self._result(False, "invalid_dt")
        self._elapsed_time += float(dt)
        if self._gray is not None:
            self._reference_age += float(dt)
        try:
            if projected_gravity is not None:
                projected_gravity = np.asarray(projected_gravity, dtype=np.float64)
                if (projected_gravity.shape != (3,) or not np.isfinite(projected_gravity).all()
                        or np.linalg.norm(projected_gravity) < 1e-8):
                    return self._result(False, "invalid_projected_gravity")
            gray = self._image(rgb)
            depth = np.asarray(depth, dtype=np.float64)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            K = np.asarray(K, dtype=np.float64)
            if depth.shape != gray.shape or K.shape != (3, 3):
                return self._result(False, "image_depth_intrinsic_shape_mismatch")
            if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
                return self._result(False, "invalid_intrinsics")
            valid_depth = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
            if np.count_nonzero(valid_depth) < self.min_inliers:
                return self._result(False, "missing_depth")
            if self._gray is None:
                points = self._features(gray, depth)
                if len(points) < self.min_inliers:
                    return self._result(False, "insufficient_initial_features", initial_features=len(points))
                self._set_reference(gray, depth, K)
                self._archive_reference()
                return self._result(False, "initialized", initial_features=len(points))
            if gray.shape != self._gray.shape or not np.allclose(K, self._K, atol=1e-7):
                return self._result(False, "camera_calibration_changed")
            if self._reference_age > self.max_reference_age_s:
                motion, reason, statistics = None, "reference_expired", {}
            else:
                motion, reason, statistics = self._track_reference(gray, depth, K, inertial_relative_guess)
            recovered_pose = None
            if motion is None:
                recovered, recovery_statistics = self._recover_from_keyframes(gray, depth, K)
                if recovered is None:
                    return self._result(False, reason, **statistics, **recovery_statistics)
                motion, recovered_rotation, recovered_position = recovered
                recovered_pose = (recovered_rotation, recovered_position)
                reason, statistics = "recovered_keyframe", recovery_statistics
            relative_rotation, relative_translation = motion
            interval = self._reference_age
            previous_rotation = self.rotation.copy()
            if recovered_pose is None:
                self.position += self.rotation @ relative_translation
                self.rotation = self.rotation @ relative_rotation
            else:
                self.rotation, self.position = recovered_pose
                self._recovery_count += 1
            visual_rotation = self.rotation
            visual_yaw = math.atan2(visual_rotation[1, 0], visual_rotation[0, 0])
            axis_heading = None
            if axis_heading_result is not None:
                axis_heading, axis_statistics = self._validated_axis_heading(axis_heading_result, visual_yaw)
                statistics.update(axis_statistics)
            if projected_gravity is not None:
                self.rotation = self.rotation_from_gravity_yaw(projected_gravity, visual_yaw)
                statistics.update(
                    gravity_tilt_correction_rad=float(np.linalg.norm(cv2.Rodrigues(visual_rotation.T @ self.rotation)[0])),
                )
                self._gravity_corrected_frames += 1
            if axis_heading is not None:
                delta = axis_heading - visual_yaw
                cosine, sine = math.cos(delta), math.sin(delta)
                self.rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]) @ self.rotation
                self._axis_corrected_frames += 1
                self._last_axis_heading_source = statistics["axis_heading_source"]
            if projected_gravity is not None or axis_heading is not None:
                statistics["visual_relative_body_rotation"] = relative_rotation.tolist()
                # Public relative motion agrees with the committed pose;
                # retain the uncorrected visual rotation separately above.
                relative_rotation = previous_rotation.T @ self.rotation
            self._accepted += 1
            self._set_reference(gray, depth, K)
            self._archive_reference(statistics)
            return self._result(True, reason, relative_body_rotation=relative_rotation.tolist(),
                                relative_body_translation=relative_translation.tolist(),
                                relative_interval_s=interval, **statistics)
        except (ValueError, TypeError, IndexError, KeyError, np.linalg.LinAlgError, cv2.error) as error:
            return self._result(False, "invalid_frame_or_cv_error", error=str(error)[:240])
