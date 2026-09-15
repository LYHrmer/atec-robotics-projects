"""Bounded repeated proximity attempts using public observations only.

Native fallback implementation after the external implementation call failed.
A completed attempt is a controller outcome, never a statement about reward.
The original FirstReachPolicy owns the entire first attempt without changes.
"""
from __future__ import annotations

import numpy as np

from task_b.arm_kinematics import (arm_joints_from_proprio, arm_targets_to_action,
                                  ee_camera_transform, head_camera_transform)
from task_b.first_reach import FirstReachPolicy
from task_b.stationary_target_gate import StationaryTargetGate
from task_b.visual_approach import _camera_arrays, detect_yellow_candidates
from task_e_geometry import JOINT_LOWER, JOINT_UPPER


_BRAKE_STATES = {'BRAKE', 'REACH_READY', 'REACH', 'REACH_VISUAL_HOLD'}
_HOLD_PHASES = {'RESTORE_HEIGHT', 'RETRACT_FOR_VIEW', 'CONFIRM_NEXT_TARGET',
                'RECONFIRM_NEXT_TARGET'}
_VIEW_PHASES = {'RETRACT_FOR_VIEW', 'CONFIRM_NEXT_TARGET', 'RECONFIRM_NEXT_TARGET'}


def _project(point, up):
    return point - np.dot(point, up) * up


def _collect(obs, images, q):
    """Use the unchanged detector, with no locked-shape bypass on acquisition."""
    candidates, errors, diagnostics = [], {}, {}
    for source in ('ee', 'head'):
        try:
            rgb, depth = _camera_arrays(images, source)
            pose = ee_camera_transform(q) if source == 'ee' else head_camera_transform()
            items, info = detect_yellow_candidates(
                rgb, depth, pose, projected_gravity=obs[9:12],
                focal_length=15. if source == 'ee' else 24.)
            diagnostics[source] = {'accepted': len(items), 'rejected': info.get('rejected', {})}
            for item in items:
                point = np.asarray(item.get('body_point'), dtype=float)
                if point.shape == (3,) and np.isfinite(point).all():
                    candidates.append(dict(item, source=source))
        except (ValueError, TypeError, KeyError) as error:
            errors[source] = str(error)
    return candidates, {'candidates': candidates, 'camera_errors': errors,
                        'detector_rejections': diagnostics}


class _AssociatedReach(FirstReachPolicy):
    """Only later acquisition is restricted to the already selected candidate."""
    def __init__(self, *args, owner, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner

    def _detect(self, obs, images, q):
        if self.reach_q is not None:
            return super()._detect(obs, images, q)
        candidates, detail = _collect(obs, images, q)
        up = -obs[9:12] / np.linalg.norm(obs[9:12])
        eligible = [c for c in candidates
                    if not self.owner._visited_near(np.asarray(c['body_point']), up)
                    and self.target is not None
                    and np.linalg.norm(np.asarray(c['body_point']) - self.target) <= .35]
        self.debug.update(detail, later_target_association='same propagated visual target; no arbitrary fallback')
        if not eligible:
            return None
        selected = min(eligible, key=lambda c: np.linalg.norm(np.asarray(c['body_point']) - self.target))
        point = np.asarray(selected['body_point'])
        self.target = .7 * point + .3 * self.target
        self.confirmations += 1
        self.last_seen = self.calls
        self.debug.update(selected=selected, target_body=self.target.tolist())
        return selected


class MultiReachPolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, ramp_calls=100, reach_only=False,
                 forward_cmd=.05, turn_cap=.6, standoff=.50, turn_gain=2.,
                 lowering_m=.02, max_attempts=2):
        if isinstance(max_attempts, bool) or int(max_attempts) != max_attempts or not 1 <= max_attempts <= 3:
            raise ValueError('max_attempts must be an integer in [1,3]')
        if reach_only or not np.isfinite(lowering_m) or lowering_m <= 0:
            raise ValueError('multi_reach requires visual input and positive fixed lowering')
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.dt = float(dt)
        self.parameters = dict(settle_calls=settle_calls, ramp_calls=ramp_calls, reach_only=False,
                               forward_cmd=forward_cmd, turn_cap=turn_cap, standoff=standoff,
                               turn_gain=turn_gain, lowering_m=lowering_m)
        self.child = FirstReachPolicy(schema, self.names, self.defaults, dt=dt, **self.parameters)
        self.arm, self.leg, self.wheel = self.child.arm, self.child.leg, self.child.wheel
        self.max_attempts, self.attempt_index, self.completed_attempts = int(max_attempts), 1, 0
        self.calls, self.alpha, self.pause_for_stance = 0, 0., False
        self.outer_phase, self.state, self.done_reason = 'ATTEMPT', self.child.state, None
        self.debug, self.transitions = {}, []
        self.lowering_alpha = 0.
        self.arm_command = self.child.arm_command.copy()
        self.lower_delta = self.child.lower_delta.copy()
        self._wheels = np.zeros(4)
        self.visited = []
        self._attempt_point, self.new_target = None, None
        self._selection_source = None
        self._phase_start = 0
        self._handoff_hold = False
        self._gate = StationaryTargetGate(dt=dt, sensor_period=.1)
        self._quiet_calls, self._leg_window = 0, []
        self._view_arm_window = []
        self._lower_q0, self._view_goal = None, None
        self._increment_fault = None
        self._progress = None
        self._up_anchor = None
        self._displacement, self._yaw = np.zeros(3), 0.
        self._pause_start, self._pause_calls, self._pause_quiet = None, 0, 0
        self._back_forward, self._back_distance, self._back_progress = None, 0., None
        self._arm_ids, self._leg_ids = self.child.arm_obs_ids.copy(), self.child.leg_obs_ids.copy()
        self._leg_defaults = self.child.leg_defaults.copy()
        self.retract_feedback_gain = 1.
        self.retract_feedback_cap_rad = .08
        self.retract_feedback_filter_tau_s = .10
        self._retract_feedback_filtered = np.zeros(6)

    @property
    def wheel_hold_requested(self):
        if self.done_reason is not None:
            return False
        if self.outer_phase in _HOLD_PHASES:
            return True
        return self.outer_phase == 'ATTEMPT' and (self.child.state in _BRAKE_STATES or self._handoff_hold)

    def _record(self, action):
        self.debug.update(outer_phase=self.outer_phase, state=self.state,
                          attempt_index=self.attempt_index, completed_attempts=self.completed_attempts,
                          max_attempts=self.max_attempts, global_control_calls=self.calls,
                          child_local_calls=self.child.calls, child_state=self.child.state,
                          done_reason=self.done_reason, lowering_alpha=float(self.lowering_alpha),
                          arm_target=self.arm_command.tolist(), wheel_hold_requested=self.wheel_hold_requested,
                          visited_local_points=[p.tolist() for p in self.visited],
                          visited_basis='short local tangent/yaw dead reckoning of public visual points; not world pose',
                          new_target_body=None if self.new_target is None else self.new_target.tolist(),
                          phase_elapsed_s=(self.calls-self._phase_start)*self.dt)
        return action

    def _action(self, wheels=None):
        out = np.zeros(self.schema.total_dim, dtype=np.float32)
        out[self.arm.start:self.arm.stop] = arm_targets_to_action(
            self.arm_command, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        out[self.leg.start:self.leg.stop] = self.lowering_alpha*self.lower_delta/self.leg.scale
        if wheels is None:
            self._wheels[:] = 0.
        else:
            self._wheels += np.clip(np.asarray(wheels)-self._wheels, -.5*self.dt, .5*self.dt)
            out[self.wheel.start:self.wheel.stop] = self._wheels
        return self._record(out)

    def _stop(self, reason):
        self.done_reason, self.state, self.outer_phase = reason, 'STOPPED', 'STOPPED'
        return self._action()

    def _phase(self, phase):
        self.transitions.append({'step': self.calls, 'from': self.outer_phase, 'to': phase})
        self.outer_phase, self.state, self._phase_start = phase, phase, self.calls
        self._quiet_calls, self._leg_window = 0, []
        self._view_arm_window.clear()
        self._gate.reset(self.calls)
        self._selection_source = None
        if phase == 'RETRACT_FOR_VIEW':
            self._retract_feedback_filtered[:] = 0.

    def _motion(self, obs):
        up = -obs[9:12]/np.linalg.norm(obs[9:12])
        velocity = _project(obs[:3], up)
        yaw_rate = float(np.dot(obs[3:6], up))
        tilt = float(np.arccos(np.clip(up[2], -1., 1.)))
        return up, velocity, yaw_rate, tilt

    def _propagate(self, obs):
        up, velocity, yaw_rate, _ = self._motion(obs)
        points = self.visited + [p for p in (self._attempt_point, self.new_target) if p is not None]
        for point in points:
            point -= self.dt*(velocity + np.cross(yaw_rate*up, point))

    def _visited_near(self, point, up):
        return any(np.linalg.norm(_project(point-old, up)) < .50 for old in self.visited)

    def _reset_budget(self, up):
        self._up_anchor = up.copy()
        self._displacement, self._yaw = np.zeros(3), 0.

    def _budget(self, obs, *, moving=False):
        up, velocity, yaw_rate, tilt = self._motion(obs)
        if tilt > .12 or np.linalg.norm(obs[3:5]) > .45:
            return 'multi_unstable_posture'
        self._displacement += velocity*self.dt
        self._yaw += yaw_rate*self.dt
        gravity_change = float(np.arccos(np.clip(np.dot(up, self._up_anchor), -1., 1.)))
        self.debug.update(public_tangent_speed_m_s=float(np.linalg.norm(velocity)),
                          public_displacement_m=float(np.linalg.norm(self._displacement)),
                          public_yaw_rad=float(self._yaw), gravity_change_rad=gravity_change)
        if abs(self._yaw) > .05 or gravity_change > .05:
            return 'multi_stationary_orientation_budget_exceeded'
        if not moving and np.linalg.norm(self._displacement) > .03:
            return 'multi_stationary_displacement_budget_exceeded'
        if moving and np.linalg.norm(velocity) > .25:
            return 'backoff_speed_limit_exceeded'
        return None

    def _pause(self, obs, q):
        violation = np.linalg.norm(obs[:3]) >= .06 or np.linalg.norm(obs[3:6]) >= .12
        if self._pause_start is None and not violation:
            return False
        if self._pause_start is None:
            self._pause_start, self._pause_quiet = self.calls, 0
        self._pause_calls += 1
        self._pause_quiet = 0 if violation else self._pause_quiet+1
        self._quiet_calls, self._leg_window = 0, []
        self._view_arm_window.clear()
        self._gate.reset(self.calls)
        if self._view_goal is not None:
            self._progress = (self.calls, float(np.max(np.abs(q[:6]-self._view_goal))))
        episode = (self.calls-self._pause_start+1)*self.dt
        total = self._pause_calls*self.dt
        self.debug.update(recovery_pause_active=True, recovery_pause_episode_s=episode,
                          recovery_pause_total_s=total, recovery_pause_quiet_s=self._pause_quiet*self.dt)
        if episode >= 2. or total >= 3.:
            self._stop('multi_recovery_pause_timeout')
            return True
        if self._pause_quiet*self.dt >= .2:
            self._pause_start, self._pause_quiet = None, 0
            self.debug['recovery_pause_active'] = False
            return False
        return True

    def _quiet(self, obs, q, goal):
        _, velocity, _, tilt = self._motion(obs)
        error = float(np.max(np.abs(q[:6]-goal)))
        speed = float(np.max(np.abs(obs[36+self._arm_ids])))
        view = self.outer_phase in _VIEW_PHASES
        rate_ok = speed <= .12 if view else speed < .05
        quiet = (error < .04 and rate_ok and np.linalg.norm(velocity) < .01
                 and np.linalg.norm(obs[:3]) < .06 and np.linalg.norm(obs[3:6]) < .12 and tilt <= .10)
        self._quiet_calls = self._quiet_calls+1 if quiet else 0
        if quiet:
            self._leg_window.append((obs[12+self._leg_ids]+self._leg_defaults).copy())
            del self._leg_window[:-max(1, round(.5/self.dt))]
        else:
            self._leg_window.clear()
        if view and quiet:
            self._view_arm_window.append(q[:6].copy())
            # Include both endpoints of a full .5 s measured-position span.
            samples = int(np.ceil(.5/self.dt))+1
            del self._view_arm_window[:-samples]
        else:
            self._view_arm_window.clear()
        arm_span = float(np.max(np.ptp(self._view_arm_window, axis=0))) if self._view_arm_window else None
        arm_duration = max(0, len(self._view_arm_window)-1)*self.dt
        position_quiet = (arm_duration >= .5-1e-9 and arm_span is not None and arm_span <= .002)
        span = float(np.max(np.ptp(self._leg_window, axis=0))) if self._leg_window else None
        self.debug.update(recovery_actual_arm_error_rad=error, recovery_arm_qdot_rad_s=speed,
                          recovery_quiet_calls=self._quiet_calls, recovery_leg_range_rad=span,
                          recovery_quiet_basis='measured_position_window_and_instantaneous_rate_cap' if view else 'original_instantaneous_qdot',
                          recovery_arm_position_range_rad=arm_span,
                          recovery_arm_position_window_s=arm_duration,
                          recovery_arm_position_quiet=position_quiet if view else None,
                          recovery_arm_hard_qdot_limit_rad_s=.12 if view else .05)
        return (self._quiet_calls*self.dt >= .5-1e-9 and span is not None and span < .02
                and (not view or position_quiet))

    def _clearance(self, up):
        target = _project(self.new_target, up)
        length = float(np.linalg.norm(target))
        if length < .01:
            return False
        direction = target/length
        rows, blocked = [], False
        for point in self.visited:
            p = _project(point, up)
            along = float(np.dot(p, direction))
            clearance = float(np.linalg.norm(p-along*direction))
            blocker = 0. < along < length and clearance < .30
            rows.append({'along_m': along, 'clearance_m': clearance, 'blocks': bool(blocker)})
            blocked = blocked or blocker
        self.debug.update(path_clearance=rows, path_blocked=bool(blocked))
        return not blocked

    def _candidate(self, obs, images, q, *, initial):
        candidates, detail = _collect(obs, images, q)
        self.debug.update(detail)
        up = -obs[9:12]/np.linalg.norm(obs[9:12])
        eligible, rejected = [], {'visited': 0, 'height_or_range': 0, 'association': 0}
        for c in candidates:
            point = np.asarray(c['body_point'])
            if self._visited_near(point, up):
                rejected['visited'] += 1
                continue
            height = c.get('height_span_m')
            if initial and (height is None or not np.isfinite(height) or height < .12 or point[0] <= .8):
                rejected['height_or_range'] += 1
                continue
            if not initial and (self.new_target is None or np.linalg.norm(point-self.new_target) > .35):
                rejected['association'] += 1
                continue
            eligible.append(c)
        self.debug['new_target_rejections'] = rejected
        if not eligible:
            return None
        same_source = [c for c in eligible if c['source'] == self._selection_source]
        choices = same_source or eligible
        selected = min(choices, key=lambda c: abs(np.arctan2(c['body_point'][1], c['body_point'][0]))
                       + .035*np.linalg.norm(np.asarray(c['body_point'])[:2]))
        self._selection_source = selected['source']
        self.debug['selected'] = selected
        return selected

    def _handoff(self, obs, q):
        parameters = dict(self.parameters, settle_calls=0)
        child = _AssociatedReach(self.schema, self.names, self.defaults, dt=self.dt,
                                 owner=self, **parameters)
        child.arm_command = q.copy()
        child.stance_verified, child.near_view = True, True
        child.target = self.new_target.copy()
        child.confirmations, child.last_seen = 2, 0
        child.state = 'WAIT_FOR_TARGET'
        self.child = child
        self.attempt_index += 1
        self._attempt_point = None
        self._handoff_hold = True
        self._phase('ATTEMPT')
        self.state = child.state
        self.arm_command = q.copy()
        return self._action()

    def act(self, proprio, images):
        self.calls += 1
        if self.done_reason is not None:
            return self._action()
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        valid = obs.size == 84 and np.isfinite(obs).all()
        gravity_valid = valid and np.linalg.norm(obs[9:12]) >= .5
        if gravity_valid:
            self._propagate(obs)
        if self.outer_phase == 'ATTEMPT':
            self.child.pause_for_stance = self.pause_for_stance
            action = self.child.act(proprio, images)
            self.alpha, self.state = self.child.alpha, self.child.state
            self.lowering_alpha = self.child.lowering_alpha
            self.arm_command = self.child.arm_command.copy()
            self.debug = dict(self.child.debug)
            if np.any(action[self.wheel.start:self.wheel.stop]):
                self._handoff_hold = False
            if self._attempt_point is None and self.child.reach_target is not None:
                self._attempt_point = self.child.reach_target.copy()
            reason = self.child.done_reason
            if reason is None:
                return self._record(action)
            if reason != 'reach_lowering_hold_complete':
                self.done_reason, self.outer_phase = reason, 'STOPPED'
                return self._record(action)
            self.completed_attempts += 1
            if self._attempt_point is not None:
                self.visited.append(self._attempt_point.copy())
            self._attempt_point = None
            if self.completed_attempts >= self.max_attempts:
                self.done_reason, self.outer_phase = 'multi_reach_attempts_complete', 'STOPPED'
                return self._record(action)
            if not gravity_valid or self.child.lower_reference_q0 is None:
                return self._stop('missing_completed_reach_reference')
            self._lower_q0 = self.child.lower_reference_q0.copy()
            self._view_goal, self.new_target = None, None
            self._increment_fault = None
            self._pause_start, self._pause_calls, self._pause_quiet = None, 0, 0
            self._reset_budget(-obs[9:12]/np.linalg.norm(obs[9:12]))
            self._phase('RESTORE_HEIGHT')
            return self._record(action)
        self.debug = {}
        if not valid or not gravity_valid:
            return self._stop('invalid_proprio')
        q = arm_joints_from_proprio(obs, self.names, self.defaults)
        up, velocity, yaw_rate, tilt = self._motion(obs)
        if tilt >= .25:
            return self._stop('tilt_limit_exceeded')
        if self.pause_for_stance:
            return self._stop('unexpected_stance_recalibration')
        moving = self.outer_phase == 'BACKOFF_CLEARANCE'
        problem = self._budget(obs, moving=moving)
        if problem:
            return self._stop(problem)
        elapsed = (self.calls-self._phase_start)*self.dt
        limits = {'RESTORE_HEIGHT': 6., 'RETRACT_FOR_VIEW': 12., 'CONFIRM_NEXT_TARGET': 8.,
                  'BACKOFF_CLEARANCE': 6., 'RECONFIRM_NEXT_TARGET': 8.}
        self.debug['phase_remaining_s'] = max(0., limits[self.outer_phase]-elapsed)
        if not moving:
            if elapsed >= limits[self.outer_phase]:
                reasons = {'RESTORE_HEIGHT': 'height_restore_timeout',
                           'RETRACT_FOR_VIEW': 'retract_view_timeout',
                           'CONFIRM_NEXT_TARGET': 'next_target_not_confirmed',
                           'RECONFIRM_NEXT_TARGET': 'next_target_reconfirmation_timeout'}
                return self._stop(reasons[self.outer_phase])
            if self._pause(obs, q):
                return self._action()
        if self.outer_phase == 'RESTORE_HEIGHT':
            leg_q = obs[12+self._leg_ids]+self._leg_defaults
            error = float(np.max(np.abs(leg_q-self._lower_q0-self.lowering_alpha*self.lower_delta)))
            self.debug['restore_increment_error_rad'] = error
            self._increment_fault = (self.calls if self._increment_fault is None else self._increment_fault) if error > .06 else None
            if self._increment_fault is not None and (self.calls-self._increment_fault+1)*self.dt >= .2:
                return self._stop('restore_joint_increment_error_exceeded')
            if self.lowering_alpha > 0.:
                self.lowering_alpha = max(0., self.lowering_alpha-self.dt/3.)
                self._quiet_calls, self._leg_window = 0, []
            elif self._quiet(obs, q, self.arm_command[:6]):
                self._view_goal = np.array([q[0], 1.1, -.8, 0., 0., 0.])
                if np.any(self._view_goal < JOINT_LOWER) or np.any(self._view_goal > JOINT_UPPER):
                    return self._stop('retract_goal_outside_joint_limits')
                self._progress = (self.calls, float(np.max(np.abs(q[:6]-self._view_goal))))
                self._phase('RETRACT_FOR_VIEW')
            return self._action()
        if self.outer_phase == 'RETRACT_FOR_VIEW':
            error = float(np.max(np.abs(q[:6]-self._view_goal)))
            start, old_error = self._progress
            if (self.calls-start)*self.dt >= 3.:
                if error > .04 and old_error-error < .01:
                    return self._stop('retract_no_joint_progress')
                self._progress = (self.calls, error)
            previous = self.arm_command[:6]
            lower = np.maximum.reduce((previous-.30*self.dt, q[:6]-.10, JOINT_LOWER))
            upper = np.minimum.reduce((previous+.30*self.dt, q[:6]+.10, JOINT_UPPER))
            if np.any(lower > upper):
                self.debug['retract_infeasible_command_axes'] = np.flatnonzero(lower > upper).tolist()
                return self._stop('retract_command_bounds_infeasible')
            # These two axes showed a persistent load-dependent position error
            # in the first physical retraction. Keep the measured pose goal
            # fixed: compensation changes only the actuator reference.
            correction = np.zeros(6)
            correction[1:3] = np.clip(
                self.retract_feedback_gain*(self._view_goal[1:3]-q[1:3]),
                -self.retract_feedback_cap_rad, self.retract_feedback_cap_rad)
            # Exact first-order filter update remains stable for every dt.
            # This block is reached only on active retract ticks, so a motion
            # pause freezes the correction history as well as the command.
            weight = -np.expm1(-self.dt/self.retract_feedback_filter_tau_s)
            self._retract_feedback_filtered += weight*(correction-self._retract_feedback_filtered)
            self.arm_command[:6] = np.clip(self._view_goal+self._retract_feedback_filtered, lower, upper)
            self.debug.update(retract_fixed_goal_rad=self._view_goal.tolist(),
                              retract_feedback_raw_correction_rad=correction.tolist(),
                              retract_feedback_correction_rad=self._retract_feedback_filtered.tolist(),
                              retract_actual_goal_error_rad=(q[:6]-self._view_goal).tolist())
            if self._quiet(obs, q, self._view_goal):
                self._phase('CONFIRM_NEXT_TARGET')
            return self._action()
        if self.outer_phase == 'BACKOFF_CLEARANCE':
            self._back_forward -= self.dt*np.cross(yaw_rate*up, self._back_forward)
            self._back_forward /= np.linalg.norm(self._back_forward)
            self._back_distance -= float(np.dot(velocity, self._back_forward))*self.dt
            self.debug.update(backward_distance_m=self._back_distance, backoff_elapsed_s=elapsed)
            if self._clearance(up) and self._back_distance >= .01:
                self._phase('RECONFIRM_NEXT_TARGET')
                self._reset_budget(up)
                return self._action()
            if abs(self._back_distance) >= .45 or elapsed >= 6.:
                return self._stop('backoff_clearance_not_reached')
            start, distance = self._back_progress
            if (self.calls-start)*self.dt >= 3.:
                if self._back_distance-distance < .01:
                    return self._stop('backoff_no_public_progress')
                self._back_progress = (self.calls, self._back_distance)
            return self._action(np.full(4, -self.parameters['forward_cmd']))
        # Stationary new selection/reconfirmation. Its samples cannot arm REACH.
        initial = self.outer_phase == 'CONFIRM_NEXT_TARGET'
        quiet = self._quiet(obs, q, self._view_goal)
        if not initial:
            quiet = quiet and elapsed >= 1.
        selected = None
        if self.calls % self.child.vision_stride == 0:
            selected = self._candidate(obs, images, q, initial=initial)
        status = self._gate.update(self.calls, quiet,
                                   point_body=None if selected is None else selected['body_point'],
                                   source=None if selected is None else selected['source'])
        self.debug['next_stationary_gate'] = status
        if status['ready']:
            point = np.asarray(status['target_body'])
            if self._visited_near(point, up):
                return self._stop('confirmed_target_matches_completed_point')
            self.new_target = point.copy()
            if self._clearance(up):
                return self._handoff(obs, q)
            if not initial:
                return self._stop('reconfirmed_path_still_blocked')
            self._phase('BACKOFF_CLEARANCE')
            self._reset_budget(up)
            self._back_forward = _project(np.array([1., 0., 0.]), up)
            self._back_forward /= np.linalg.norm(self._back_forward)
            self._back_distance = 0.
            self._back_progress = (self.calls, 0.)
            # Release the stationary wheel anchor on the first actual reverse
            # request, rather than one zero-command control tick earlier.
            return self._action(np.full(4, -self.parameters['forward_cmd']))
        return self._action()

    def describe(self):
        return {'mode': 'multi_reach', 'implementation': 'native fallback after external implementation timeout',
                'inputs': 'public proprio84, original RGB-D, static action/robot geometry only',
                'max_attempts': self.max_attempts, 'attempt_index': self.attempt_index,
                'completed_attempts': self.completed_attempts, 'outer_phase': self.outer_phase,
                'global_control_calls': self.calls, 'child_local_calls': self.child.calls,
                'child_state': self.child.state, 'state': self.state, 'done_reason': self.done_reason,
                'alpha': self.alpha, 'lowering_alpha': self.lowering_alpha,
                'wheel_hold_requested': self.wheel_hold_requested,
                'completed_attempts_claim': 'normal controller completions, no score input or score certification',
                'target_memory': 'public visual points with bounded local tangent/yaw propagation, no world map or object IDs',
                'claim': 'bounded proximity attempts only; no grasp or delivery implementation',
                'retract_position_feedback': {
                    'axes': ['arm_joint2', 'arm_joint3'],
                    'gain': self.retract_feedback_gain,
                    'correction_cap_rad': self.retract_feedback_cap_rad,
                    'filter_tau_s': self.retract_feedback_filter_tau_s,
                    'filter_update': 'exact first-order dt update; zero at retract entry; frozen during pause',
                    'acceptance': 'fixed measured goal error < .04 rad plus the explicit view_quiet position/rate conditions',
                    'command_bounds': 'intersection of .30 rad/s slew, measured q +/- .10 rad, original joint limits'},
                'view_quiet': {'phases': sorted(_VIEW_PHASES),
                               'measured_position_window_s': .5, 'maximum_arm_position_span_rad': .002,
                               'instantaneous_arm_qdot_hard_cap_rad_s': .12,
                               'actual_goal_error_rad': .04,
                               'unchanged': 'body/leg checks and deadlines; restore and FirstReach use original quiet thresholds'},
                'parameters': dict(self.parameters), 'transitions': list(self.transitions),
                'child': self.child.describe(), 'debug': dict(self.debug)}
