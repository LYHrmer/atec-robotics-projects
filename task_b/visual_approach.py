"""Conservative RGB-D approach to a visible yellow, vertically long object.

This is an image heuristic, not a semantic mustard classifier or a grasp
policy. Runtime inputs are only public proprio and EE RGB-D, optionally the
original head RGB-D for near-field handoff. No scene/root/
object pose, object index, seed-derived position, or score is used.
"""
from __future__ import annotations

import cv2
import numpy as np

from task_b.arm_kinematics import arm_joints_from_proprio, ee_camera_transform, head_camera_transform
from task_b.control import WHEEL_TERM, wheel_side


def _camera_arrays(image, source):
    if not isinstance(image, dict) or source+'_rgb' not in image or source+'_depth' not in image:
        raise ValueError('missing_'+source+'_rgbd')
    rgb = np.asarray(image[source+'_rgb'])
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    depth = np.asarray(image[source+'_depth']).squeeze()
    if (rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or depth.ndim != 2
            or rgb.shape[:2] != depth.shape or min(depth.shape) < 16):
        raise ValueError('invalid_'+source+'_rgbd_shape')
    if not np.isfinite(rgb).all():
        raise ValueError('nonfinite_'+source+'_rgb')
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size and float(rgb.max()) <= 1.01:
        rgb = rgb * 255.
    return np.clip(rgb, 0, 255).astype(np.uint8), depth.astype(np.float64, copy=False)


def _ee_arrays(image):
    """Backward-compatible EE-only image extraction."""
    return _camera_arrays(image, 'ee')


def detect_yellow_candidates(rgb, depth, body_from_camera, *, projected_gravity=None,
                             focal_length=15.):
    """Return depth-supported yellow image components in current body frame.

    Uses the supplied official focal length (EE 15, head 24), common aperture
    and requested raster K principal
    point (width/2,height/2). Geometry is explicit in debug metadata so a future
    pixel-centre convention change cannot silently happen twice.
    """
    height, width = depth.shape
    if not np.isfinite(focal_length) or focal_length <= 0.:
        raise ValueError('invalid_focal_length')
    focal = width * focal_length / 20.955
    cx, cy = width / 2., height / 2.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([18, 90, 60], np.uint8), np.array([42, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pose = np.asarray(body_from_camera, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('invalid_camera_transform')
    up = None
    if projected_gravity is not None:
        gravity = np.asarray(projected_gravity, dtype=float).reshape(-1)
        if gravity.size != 3 or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < .5:
            raise ValueError('invalid_projected_gravity')
        up = -gravity / np.linalg.norm(gravity)
    candidates, rejected = [], {}
    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1
    for label in range(1, count):
        left, top, box_width, box_height, area = map(int, statistics[label])
        aspect = box_height / max(box_width, 1)
        if area < 35 or not 1.15 <= aspect <= 6.0 or box_height < 10:
            reject('image_shape'); continue
        rows, cols = np.nonzero(labels == label)
        depths = depth[rows, cols]
        valid = np.isfinite(depths) & (depths > .10) & (depths < 8.)
        if int(valid.sum()) < 25 or float(valid.mean()) < .50:
            reject('missing_depth'); continue
        rows, cols, depths = rows[valid], cols[valid], depths[valid]
        median_depth = float(np.median(depths))
        # Reject background leakage rather than averaging it into the object.
        supported = np.abs(depths - median_depth) < max(.06, .03 * median_depth)
        if int(supported.sum()) < 25:
            reject('inconsistent_depth'); continue
        rows, cols, depths = rows[supported], cols[supported], depths[supported]
        u, v, optical_depth = float(np.median(cols)), float(np.median(rows)), float(np.median(depths))
        camera_point = np.array([(u-cx)*optical_depth/focal, (v-cy)*optical_depth/focal, optical_depth])
        body_point = pose[:3, :3] @ camera_point + pose[:3, 3]
        if not .25 < body_point[0] < 7.5 or abs(body_point[1]) > 4.:
            reject('outside_forward_region'); continue
        camera_cloud = np.column_stack(((cols-cx)*depths/focal, (rows-cy)*depths/focal, depths))
        body_cloud = camera_cloud @ pose[:3, :3].T + pose[:3, 3]
        height_span = None if up is None else float(np.diff(np.percentile(body_cloud @ up, [2, 98]))[0])
        if height_span is not None and not .045 < height_span < .35:
            reject('metric_height'); continue
        candidates.append({
            'kind': 'yellow_vertical_component_not_semantic_classification',
            'body_point': body_point.tolist(), 'camera_point': camera_point.tolist(),
            'pixel_uv': [u, v], 'bbox_xywh': [left, top, box_width, box_height],
            'pixel_area': area, 'depth_samples': int(len(depths)),
            'optical_depth_m': optical_depth, 'height_span_m': height_span,
            'forward_planar_distance_m': float(np.linalg.norm(body_point[:2])),
        })
    candidates.sort(key=lambda item: item['forward_planar_distance_m'])
    return candidates, {'yellow_components': int(count-1), 'rejected': rejected,
                        'K': [[focal, 0., cx], [0., focal, cy], [0., 0., 1.]],
                        'intrinsics_pixel_convention': 'configured_raster_K_width_over_2',
                        'classification_claim': 'none; color and shape heuristic only'}


class VisualApproachPolicy:
    """Low-speed differential approach; invalid/lost observations stop wheels.

    Only the wheel action slice changes. Arm and legs retain official zero
    normalized actions throughout. ``turn_sign`` must be confirmed in a real
    turn probe; positive assumes right+ / left- produces positive body yaw.
    """

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 turn_sign=1., turn_gain=1.2, wheel_cap=.6, forward_cmd=.10,
                 target_x=.8, settle_calls=100, ramp_calls=100, use_head=False):
        schema.validate()
        if not np.isfinite(dt) or dt <= 0. or turn_sign not in (-1., 1.):
            raise ValueError('dt must be positive and turn_sign must be +/-1')
        if not (0. < forward_cmd <= wheel_cap <= .6 and 0. < turn_gain <= 5. and .5 <= target_x <= 2.):
            raise ValueError('Invalid approach speed, turn gain, wheel cap or standoff')
        if settle_calls < 0 or ramp_calls < 1:
            raise ValueError('Invalid settling/ramp length')
        self.schema = schema
        self.observation_joint_names = tuple(observation_joint_names)
        self.defaults = dict(defaults)
        self.dt = float(dt); self.turn_sign = float(turn_sign); self.turn_gain = float(turn_gain)
        self.wheel_cap = float(wheel_cap); self.forward_cmd = float(forward_cmd)
        self.target_x = float(target_x)
        self.use_head = bool(use_head)
        self.settle_calls = int(settle_calls); self.ramp_calls = int(ramp_calls)
        self.vision_stride = max(1, int(round(.1/self.dt)))
        self._wheel = schema.term(WHEEL_TERM)
        self._sides = np.array([1. if wheel_side(name) == 'right' else -1. for name in self._wheel.joint_names])
        self._target = None; self._streak = 0; self._last_vision_call = -self.vision_stride
        self._target_source = None
        self._valid_vision = False; self._previous_wheels = np.zeros(4)
        self.calls = 0; self.alpha = 0.; self.state = 'SETTLE'; self.debug = {}
        cv2.setNumThreads(1)
        # Verify mapping at construction, including the real observation order.
        arm_joints_from_proprio(np.zeros(84), self.observation_joint_names, self.defaults)

    def _stop(self, reason, *, clear_target=False):
        self._previous_wheels[:] = 0.
        self.state = reason
        if clear_target:
            self._valid_vision = False; self._streak = 0; self._target = None
            self._target_source = None
        self.debug['reason'] = reason
        self.debug['wheel_action'] = [0., 0., 0., 0.]
        return np.zeros(self.schema.total_dim, dtype=np.float32)

    def act(self, proprio, image):
        self.calls += 1
        self.alpha = float(np.clip((self.calls-self.settle_calls)/self.ramp_calls, 0., 1.))
        observation = np.asarray(proprio, dtype=float).reshape(-1)
        if observation.size != 84 or not np.isfinite(observation).all():
            return self._stop('INVALID_PROPRIO', clear_target=True)
        gravity = observation[9:12]
        if np.linalg.norm(gravity) < .5 or -gravity[2]/np.linalg.norm(gravity) < np.cos(np.deg2rad(35.)):
            return self._stop('UNSTABLE_BODY', clear_target=True)
        if self.calls <= self.settle_calls:
            return self._stop('SETTLE', clear_target=True)
        frames, image_errors = {}, {}
        if not self.use_head:
            # Preserve the original EE-only behavior, including invalid-frame
            # stopping, when the optional near-field camera is disabled.
            try:
                frames['ee'] = _ee_arrays(image)
            except (ValueError, TypeError) as error:
                self.debug = {'image_error': str(error)}
                return self._stop('INVALID_RGBD', clear_target=True)
        else:
            for source in ('ee', 'head'):
                try:
                    frames[source] = _camera_arrays(image, source)
                except (ValueError, TypeError) as error:
                    image_errors[source] = str(error)
            if not frames:
                self.debug = {'image_errors': image_errors}
                return self._stop('INVALID_RGBD', clear_target=True)
        if self.calls-self._last_vision_call >= self.vision_stride:
            self._last_vision_call = self.calls
            q = arm_joints_from_proprio(observation, self.observation_joint_names, self.defaults)
            if not self.use_head:
                rgb, depth = frames['ee']
                candidates, detection = detect_yellow_candidates(
                    rgb, depth, ee_camera_transform(q), projected_gravity=gravity)
            else:
                candidates = []
                detection = {'yellow_components': 0, 'rejected': {}, 'cameras': {},
                             'image_errors': image_errors,
                             'classification_claim': 'none; color and shape heuristic only'}
                for source, (rgb, depth) in frames.items():
                    camera_pose = ee_camera_transform(q) if source == 'ee' else head_camera_transform()
                    items, info = detect_yellow_candidates(
                        rgb, depth, camera_pose, projected_gravity=gravity,
                        focal_length=15. if source == 'ee' else 24.)
                    for item in items:
                        item['source'] = source
                    candidates.extend(items)
                    detection['cameras'][source] = info
                    detection['yellow_components'] += info['yellow_components']
                    for reason, count in info['rejected'].items():
                        detection['rejected'][reason] = detection['rejected'].get(reason, 0)+count
                candidates.sort(key=lambda item: item['forward_planar_distance_m'])
            self.debug = {'vision_call': self.calls, 'detector': detection,
                          'candidates': candidates, 'target_body': None}
            if not candidates:
                return self._stop('NO_VISIBLE_TARGET', clear_target=True)
            selected = candidates[0]
            if self._target is not None:
                nearest = min(candidates, key=lambda item: np.linalg.norm(np.asarray(item['body_point'])-self._target))
                if np.linalg.norm(np.asarray(nearest['body_point'])-self._target) < .30:
                    selected = nearest
                    self._streak += 1
                else:
                    self._streak = 1
                    self._target = None
            else:
                self._streak = 1
            source_changed = self.use_head and selected['source'] != self._target_source
            if source_changed:
                # A close spatial match permits handoff, but movement waits for
                # the new camera to support this target on two detection calls.
                self._streak = 1
                self._target_source = selected['source']
            point = np.asarray(selected['body_point'])
            self._target = point if self._target is None else .55*point + .45*self._target
            self._valid_vision = self._streak >= 2
            self.debug.update(selected=selected, target_body=self._target.tolist(), confirmations=self._streak)
            if self.use_head:
                self.debug.update(source=self._target_source, source_changed=source_changed,
                                  source_confirmed=self._valid_vision)
        if not self._valid_vision or self._target is None:
            return self._stop('NO_VISIBLE_TARGET' if self._target is None else 'CONFIRM_TARGET')
        x, y, _ = self._target
        bearing = float(np.arctan2(y, x))
        self.debug.update(target_body=self._target.tolist(), bearing_rad=bearing,
                          forward_error_m=float(x-self.target_x), confirmations=self._streak)
        if x <= self.target_x+.04 and abs(y) <= .08:
            return self._stop('AT_STANDOFF')
        if x < self.target_x-.10:
            return self._stop('TARGET_TOO_CLOSE')
        turn = 0. if abs(bearing) < .025 else self.turn_sign*np.clip(self.turn_gain*bearing, -self.wheel_cap, self.wheel_cap)
        # First face the target; avoid sweeping forward through a lateral object.
        forward = self.forward_cmd*np.clip((x-self.target_x)/.50, 0., 1.) if abs(bearing) < .25 else 0.
        requested = np.clip(forward + self._sides*turn, -self.wheel_cap, self.wheel_cap)*self.alpha
        # Valid visual commands slew at 1 normalized unit per second. Lost target
        # stops immediately instead of carrying a blind command through the ramp.
        wheels = self._previous_wheels + np.clip(requested-self._previous_wheels, -self.dt, self.dt)
        self._previous_wheels = wheels.copy()
        action = np.zeros(self.schema.total_dim, dtype=np.float32)
        action[self._wheel.start:self._wheel.stop] = wheels
        self.state = 'ALIGN' if abs(bearing) >= .25 else 'APPROACH'
        self.debug.update(reason=self.state, forward_normalized=float(forward),
                          turn_normalized=float(turn), wheel_action=wheels.tolist())
        return action

    def describe(self):
        return {'mode':'visual_approach', 'inputs':('public proprio84 and ee_rgb/ee_depth'
                + (' plus original head_rgb/head_depth' if self.use_head else '')
                + '; static joint/action/camera geometry'),
                'use_head':self.use_head,
                'camera_handoff':'nearest current body-frame target within .30m; two detection calls on each new source',
                'target':'nearest forward yellow vertical component; not a verified mustard class',
                'grasping':False, 'target_x_m':self.target_x, 'forward_normalized_cap':self.forward_cmd,
                'wheel_normalized_cap':self.wheel_cap, 'turn_sign':self.turn_sign,
                'turn_gain':self.turn_gain, 'turn_sign_requires_physical_probe':True,
                'settle_calls':self.settle_calls, 'ramp_calls':self.ramp_calls,
                'vision_stride_calls':self.vision_stride, 'missing_target_behavior':'stop; no blind scan',
                'wheel_joint_names':list(self._wheel.joint_names),
                'wheel_action_slice':[self._wheel.start,self._wheel.stop],
                'legs_and_arm':'official zero normalized action', 'calls':self.calls,
                'alpha':self.alpha, 'state':self.state, 'debug':self.debug}
