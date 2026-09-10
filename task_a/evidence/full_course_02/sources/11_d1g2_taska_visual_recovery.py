"""Explicit, bounded inertial reanchoring between disconnected RGB-D segments.

The active odometer always gets the first opportunity to recover its old map.
After a failure, a separate candidate starts at a short proprioceptive estimate
in the SAME original coordinate frame. Two consecutive accepted candidate
motions can promote it. This is labelled ``inertial_reanchor``, never visual
relocalization across the missing interval. No policy or action history lives
here, and no simulator pose, terrain geometry, or diagnostic input is read.

Call observe_proprio once per 50 Hz sample using absolute simulation seconds.
Camera operations consume an already observed, identical timestamp; they never
integrate another dt. Counters/budgets survive candidate restarts/promotions.
"""
from __future__ import annotations

import math
import numpy as np

from tools.d1g2_taska_rgbd import RGBDOdometry


class RecoveringRGBDOdometry:
    def __init__(self, *, mount_pitch_deg=30., candidate_accepts=2,
                 max_gap_seconds=2., max_gap_path_m=1., max_reanchors=4,
                 max_episode_blind_seconds=4., max_episode_blind_path_m=2.,
                 **odometry_kwargs):
        self._odometry_kwargs = dict(odometry_kwargs, mount_pitch_deg=mount_pitch_deg)
        self.candidate_accepts = int(candidate_accepts)
        self.max_reanchors = int(max_reanchors)
        self.max_gap_seconds = float(max_gap_seconds)
        self.max_gap_path_m = float(max_gap_path_m)
        self.max_episode_blind_seconds = float(max_episode_blind_seconds)
        self.max_episode_blind_path_m = float(max_episode_blind_path_m)
        if (self.candidate_accepts < 2 or self.max_reanchors < 0 or any(
                not math.isfinite(value) or value <= 0 for value in (
                    self.max_gap_seconds, self.max_gap_path_m,
                    self.max_episode_blind_seconds, self.max_episode_blind_path_m))):
            raise ValueError("Invalid visual recovery budget")
        self.reset()

    def reset(self):
        """New episode only. Candidate promotion never calls this method."""
        self.active = RGBDOdometry(**self._odometry_kwargs)
        self.candidate = None
        self._candidate_time = None
        self._candidate_seed = None
        self._candidate_consecutive = 0
        self._candidate_seeds = self._candidate_reseeds = 0
        self._sensor_time = self._camera_time = None
        self._proprio = None
        self._previous_rate = self._previous_speed = None
        self._previous_world_velocity = None
        self._shadow_position = self._shadow_rotation = None
        self._anchor_time = None
        self._pending_path = 0.
        self._gap_open = False
        self._gap_seconds = self._gap_path = 0.
        self._gap_reason = self._blocked = self._fault = None
        self._episode_blind_seconds = self._episode_blind_path = 0.
        self._segment_id = 0
        self._frames = self._accepted = self._rejected = 0
        self._gravity_corrected = self._axis_corrected = self._map_recoveries = 0
        self._proprio_samples = self._axis_updates = 0
        self._heading_source = "not_initialized"
        self._heading_time = None
        self._last_axis_time = None
        self._origin = None
        self._last_reanchor = None
        self._last = None

    @staticmethod
    def _yaw(rotation):
        return math.atan2(rotation[1, 0], rotation[0, 0])

    @staticmethod
    def _sensor_values(proprio):
        observation = np.asarray(proprio, dtype=np.float64)
        if observation.shape != (81,):
            raise ValueError("proprio must have shape (81,)")
        if not np.isfinite(np.r_[observation[:6], observation[9:12]]).all():
            return observation, None, None, "nonfinite_proprioception"
        gravity = observation[9:12]
        norm = float(np.linalg.norm(gravity))
        if norm < 1e-8:
            return observation, None, None, "invalid_projected_gravity"
        gravity = gravity / norm
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1., 1.))))
        if tilt > 60.:
            return observation, None, None, "extreme_tilt"
        denominator = float(gravity[1]**2 + gravity[2]**2)
        if denominator < 1e-4:
            return observation, None, None, "singular_yaw_geometry"
        rate = float((-gravity[1]*observation[4] - gravity[2]*observation[5]) / denominator)
        return observation, rate, float(np.linalg.norm(observation[:3])), None

    def observe_proprio(self, proprio, timestamp_seconds):
        """Advance one sensor interval; repeated/out-of-order timestamps fail.

        A sensor fault latches until an explicit episode reset. Gap budgets
        include the complete 3-D speed integral and the proposed current dt;
        exceeding a bound prevents propagation/promotion on that same sample.
        Actual observed blind durations remain reported even after a stop.
        """
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self._sensor_time is not None and timestamp <= self._sensor_time:
            raise ValueError("proprio timestamps must strictly increase; do not integrate twice")
        observation, rate, speed, fault = self._sensor_values(proprio)
        dt = 0. if self._sensor_time is None else timestamp-self._sensor_time
        self._sensor_time = timestamp
        self._proprio = observation.copy()
        self._proprio_samples += 1
        if fault is not None:
            self._fault = fault
        if self._fault is not None:
            return self.state_dict()
        if self._shadow_rotation is not None and dt:
            path_increment = .5*(self._previous_speed+speed)*dt
            self._pending_path += path_increment
            if self._gap_open:
                self._gap_seconds += dt
                self._gap_path += path_increment
                self._episode_blind_seconds += dt
                self._episode_blind_path += path_increment
                self._check_budget()
            yaw = self._yaw(self._shadow_rotation) + .5*(self._previous_rate+rate)*dt
            rotation = RGBDOdometry.rotation_from_gravity_yaw(observation[9:12], yaw)
            velocity = rotation @ observation[:3]
            # Budget gating precedes committing this proposed position step.
            # Heading can remain observable while position propagation stops.
            if self._blocked is None:
                self._shadow_position += .5*(self._previous_world_velocity+velocity)*dt
            self._shadow_rotation = rotation
            self._previous_world_velocity = velocity
            self._heading_source = "short_term_proprio_prediction"
            self._heading_time = timestamp
        self._previous_rate, self._previous_speed = rate, speed
        return self.state_dict()

    def _camera_timestamp(self, timestamp, dt=None):
        if self._sensor_time is None:
            raise ValueError("observe_proprio is required before a camera frame")
        timestamp = self._sensor_time if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp) or abs(timestamp-self._sensor_time) > 1e-7:
            raise ValueError("Camera and latest proprio timestamps must match")
        interval = .1 if self._camera_time is None else timestamp-self._camera_time
        if interval <= 0:
            raise ValueError("Camera timestamp must strictly increase")
        if dt is not None and (not math.isfinite(float(dt)) or abs(float(dt)-interval) > 1e-6):
            raise ValueError("Camera dt disagrees with absolute timestamps")
        self._camera_time = timestamp
        self._frames += 1
        return timestamp, interval

    def _anchor_prediction(self, timestamp):
        self._shadow_position = self.active.position.copy()
        self._shadow_rotation = self.active.rotation.copy()
        _, rate, speed, fault = self._sensor_values(self._proprio)
        if fault is not None:
            raise ValueError("Cannot anchor invalid proprioception")
        self._previous_rate, self._previous_speed = rate, speed
        self._previous_world_velocity = self._shadow_rotation @ self._proprio[:3]
        self._anchor_time = timestamp
        self._pending_path = 0.
        self._heading_source = "accepted_visual_pose"
        self._heading_time = timestamp

    def initialize_from_proprio(self, rgb, depth, K, proprio, estimated_xy, estimated_yaw,
                                *, timestamp=None):
        """Original episode bootstrap only; its zero-z datum is retained later."""
        if self.active._gray is not None:
            raise RuntimeError("A reference already exists; cannot silently reset it")
        timestamp, _ = self._camera_timestamp(timestamp)
        if not np.array_equal(np.asarray(proprio), self._proprio):
            raise ValueError("Bootstrap proprio must match the observed timestamp")
        if self._fault is not None:
            return self._result(False, "recovery_sensor_fault")
        result = self.active.initialize_from_proprio(rgb, depth, K, proprio,
                                                     estimated_xy, estimated_yaw)
        if result["reason"] == "initialized":
            self._origin = {key: result[key] for key in
                            ("initial_rotation", "initial_position", "initialization_mode", "pose_frame")}
            self._anchor_prediction(timestamp)
        return self._result(False, result["reason"], result)

    def initialize_from_pose(self, rgb, depth, K, initial_rotation, initial_position,
                             *, timestamp=None):
        """Replay/bootstrap of an existing sensor estimate, never diagnostic truth."""
        if self.active._gray is not None:
            raise RuntimeError("A reference already exists")
        timestamp, interval = self._camera_timestamp(timestamp)
        if self._fault is not None:
            return self._result(False, "recovery_sensor_fault")
        self.active.reset(initial_rotation, initial_position)
        result = self.active.update(rgb, depth, K, interval)
        if result["reason"] == "initialized":
            self._origin = {key: result[key] for key in
                            ("initial_rotation", "initial_position", "initialization_mode", "pose_frame")}
            self._anchor_prediction(timestamp)
        return self._result(False, result["reason"], result)

    def _check_budget(self):
        checks = [(self._gap_seconds > self.max_gap_seconds+1e-9, "gap_time_budget"),
                  (self._gap_path > self.max_gap_path_m+1e-9, "gap_path_budget"),
                  (self._segment_id >= self.max_reanchors, "episode_reanchor_budget"),
                  (self._episode_blind_seconds > self.max_episode_blind_seconds+1e-9, "episode_time_budget"),
                  (self._episode_blind_path > self.max_episode_blind_path_m+1e-9, "episode_path_budget")]
        if self._blocked is None:
            self._blocked = next((reason for exceeded, reason in checks if exceeded), None)
        return self._blocked

    def _open_gap(self, reason, timestamp):
        if not self._gap_open:
            self._gap_open = True
            self._gap_reason = reason
            self._gap_seconds = timestamp-self._anchor_time
            self._gap_path = self._pending_path
            self._episode_blind_seconds += self._gap_seconds
            self._episode_blind_path += self._gap_path
        self._check_budget()

    def _close_gap(self):
        self.candidate = None
        self._candidate_seed = self._candidate_time = None
        self._candidate_consecutive = 0
        self._gap_open = False
        self._gap_seconds = self._gap_path = 0.
        self._gap_reason = self._blocked = None

    def _observe_axis(self, result):
        if result is None or self._shadow_rotation is None:
            return False
        heading, statistics = RGBDOdometry._validated_axis_heading(result, self._yaw(self._shadow_rotation))
        if heading is None:
            return False
        self._shadow_rotation = RGBDOdometry.rotation_from_gravity_yaw(self._proprio[9:12], heading)
        self._previous_world_velocity = self._shadow_rotation @ self._proprio[:3]
        self._heading_source = statistics["axis_heading_source"]
        self._heading_time = self._sensor_time
        self._last_axis_time = self._sensor_time
        self._axis_updates += 1
        return True

    def _seed_candidate(self, rgb, depth, K, interval, timestamp, *, reseed=False):
        """Use the current provisional XYZ/R; never renew a gap's budget."""
        self.candidate = RGBDOdometry(**self._odometry_kwargs)
        self.candidate.reset(self._shadow_rotation, self._shadow_position)
        result = self.candidate.update(rgb, depth, K, interval)
        self._candidate_time = timestamp
        self._candidate_consecutive = 0
        self._candidate_seeds += 1
        self._candidate_reseeds += int(reseed)
        self._candidate_seed = {"timestamp": timestamp,
                                "estimated_position": self._shadow_position.tolist(),
                                "estimated_rotation": self._shadow_rotation.tolist(),
                                "unobserved_interval_seconds": self._gap_seconds}
        return result

    def update(self, rgb, depth, K, dt=None, inertial_relative_guess=None,
               projected_gravity=None, axis_heading_result=None, *, timestamp=None):
        timestamp, interval = self._camera_timestamp(timestamp, dt)
        if self._fault is not None:
            return self._result(False, "recovery_sensor_fault")
        if self._anchor_time is None:
            raise RuntimeError("Initialize an original sensor pose before update")
        gravity = self._proprio[9:12]
        if projected_gravity is not None and not np.allclose(projected_gravity, gravity, atol=1e-7, rtol=0):
            raise ValueError("Camera gravity must be from the same proprio timestamp")
        previous_position, previous_rotation = self.active.position.copy(), self.active.rotation.copy()
        previous_timestamp = self._anchor_time
        result = self.active.update(rgb, depth, K, interval,
                                    inertial_relative_guess=inertial_relative_guess,
                                    projected_gravity=gravity, axis_heading_result=axis_heading_result)
        if result["accepted"]:
            self._close_gap()
            self._anchor_prediction(timestamp)
            return self._publish(result, previous_position, previous_rotation, previous_timestamp)

        axis_applied = self._observe_axis(axis_heading_result)
        self._open_gap(result["reason"], timestamp)
        extra = {"active_rejection_reason": result["reason"], "heading_axis_applied": axis_applied}
        if self._blocked is not None:
            return self._result(False, result["reason"], result, **extra)
        if self.candidate is None or self.candidate._gray is None:
            # Full XYZ and R at this failed camera timestamp, including z.
            candidate_result = self._seed_candidate(rgb, depth, K, interval, timestamp,
                                                    reseed=self.candidate is not None)
        else:
            candidate_result = self.candidate.update(
                rgb, depth, K, timestamp-self._candidate_time,
                projected_gravity=gravity, axis_heading_result=axis_heading_result)
            self._candidate_time = timestamp
            self._candidate_consecutive = self._candidate_consecutive+1 if candidate_result["accepted"] else 0
            if not candidate_result["accepted"]:
                # A second view change can disconnect the candidate as well.
                # Start evidence again at today's image in the ORIGINAL gap;
                # retaining its obsolete image would duplicate active's loss.
                reseeded = self._seed_candidate(rgb, depth, K, interval, timestamp, reseed=True)
                extra.update(candidate_reseeded=True, candidate_reseed_reason=reseeded["reason"])
        extra.update(candidate_reason=candidate_result["reason"],
                     candidate_accepted=bool(candidate_result["accepted"]),
                     candidate_inliers=int(candidate_result.get("inliers", 0)),
                     candidate_reproj=candidate_result.get("reproj"))
        if self._candidate_consecutive < self.candidate_accepts:
            return self._result(False, result["reason"], result, **extra)
        displacement = float(np.linalg.norm(self.candidate.position-previous_position))
        if displacement > self.max_gap_path_m:
            self._blocked = "candidate_displacement_budget"
            return self._result(False, result["reason"], result, **extra)
        self._segment_id += 1
        self._last_reanchor = {
            "timestamp": timestamp, "segment_id": self._segment_id,
            "source": "proprio_inertial_gap_then_candidate_visual_motion",
            "visually_connected_to_previous_segment": False,
            "gap_reason": self._gap_reason, "gap_seconds": self._gap_seconds,
            "gap_path_m": self._gap_path, "committed_displacement_m": displacement,
            "candidate_consecutive_accepts": self._candidate_consecutive,
            "candidate_seed": self._candidate_seed,
            "position_uncertainty": "unquantified inertial drift; inherited by this segment",
        }
        self.active = self.candidate
        self._close_gap()
        self._anchor_prediction(timestamp)
        candidate_result = dict(candidate_result, reason="inertial_reanchor",
                                visually_connected_to_previous_segment=False)
        return self._publish(candidate_result, previous_position, previous_rotation, previous_timestamp, **extra)

    def _publish(self, result, previous_position, previous_rotation, previous_timestamp, **extra):
        self._accepted += 1
        self._gravity_corrected += int("gravity_tilt_correction_rad" in result)
        self._axis_corrected += int(bool(result.get("axis_heading_applied")))
        self._map_recoveries += int(result["reason"] == "recovered_keyframe")
        if result.get("axis_heading_applied"):
            self._heading_source = result.get("axis_heading_source", "axis_heading")
            if self._last_axis_time != self._sensor_time:
                self._axis_updates += 1
            self._last_axis_time = self._sensor_time
        result = dict(result, relative_body_rotation=(previous_rotation.T @ self.active.rotation).tolist(),
                      relative_body_translation=(previous_rotation.T @ (self.active.position-previous_position)).tolist(),
                      relative_interval_s=self._sensor_time-previous_timestamp)
        return self._result(True, result["reason"], result, **extra)

    def state_dict(self):
        state = self.active.state_dict()
        if self._origin is not None:
            state.update(self._origin)
        state.update(frames=self._frames, accepted_frames=self._accepted,
                     rejected_frames=self._rejected, gravity_corrected_frames=self._gravity_corrected,
                     axis_corrected_frames=self._axis_corrected, recovery_count=self._map_recoveries,
                     segment_id=self._segment_id, inertial_reanchor_count=self._segment_id,
                     pose_provenance=("visual_from_original_sensor_bootstrap" if self._segment_id == 0
                                      else "inertial_reanchor_then_visual"),
                     last_inertial_reanchor=self._last_reanchor,
                     position_timestamp=self._anchor_time,
                     position_age_s=(0. if self._anchor_time is None else self._sensor_time-self._anchor_time),
                     heading_yaw=(state["estimated_yaw"] if self._shadow_rotation is None else self._yaw(self._shadow_rotation)),
                     heading_timestamp=self._heading_time, heading_source=self._heading_source,
                     independent_axis_updates=self._axis_updates, last_axis_timestamp=self._last_axis_time,
                     proprio_samples=self._proprio_samples, gap_open=self._gap_open,
                     gap_seconds=self._gap_seconds, gap_path_m=self._gap_path,
                     episode_blind_seconds=self._episode_blind_seconds,
                     episode_blind_path_m=self._episode_blind_path,
                     candidate_initialized=self.candidate is not None and self.candidate._gray is not None,
                     candidate_consecutive_accepts=self._candidate_consecutive,
                     candidate_seed_count=self._candidate_seeds, candidate_reseed_count=self._candidate_reseeds,
                     recovery_block_reason=self._fault or self._blocked,
                     recovery_stop_required=self._fault is not None or self._blocked is not None,
                     recovery_budgets={"gap_seconds": self.max_gap_seconds, "gap_path_m": self.max_gap_path_m,
                                       "episode_seconds": self.max_episode_blind_seconds,
                                       "episode_path_m": self.max_episode_blind_path_m,
                                       "reanchors": self.max_reanchors})
        return state

    def _result(self, accepted, reason, visual_result=None, **extra):
        if not accepted and self.active._gray is not None and reason != "initialized":
            self._rejected += 1
        result = dict(visual_result or {})
        result.update(self.state_dict())
        result.update(accepted=bool(accepted), reason=reason,
                      lost=not accepted and self.active._gray is not None and reason != "initialized")
        result.update(extra)
        result.setdefault("relative_body_rotation", np.eye(3).tolist())
        result.setdefault("relative_body_translation", [0., 0., 0.])
        self._last = result
        return result
