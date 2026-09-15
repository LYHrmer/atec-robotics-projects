"""Visual approach followed by a stationary arm reach in the original Task B.

Only public RGB-D and proprioception enter this policy. The yellow detector is
a shape/color heuristic. Reaching is not a claim of physical grasp or delivery.
"""
from __future__ import annotations

import numpy as np

from task_b import leg_kinematics as legs
from task_b.arm_kinematics import (arm_joints_from_proprio, arm_targets_to_action,
                                  ee_camera_transform, head_camera_transform,
                                  fk, solve_ik)
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM, wheel_side
from task_b.visual_approach import _camera_arrays, detect_yellow_candidates
from task_b.stationary_target_gate import StationaryTargetGate


class FirstReachPolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, ramp_calls=100, reach_only=False,
                 forward_cmd=.20, turn_cap=.20, standoff=.56, turn_gain=.4, lowering_m=0.,
                 grasp=False):
        schema.validate()
        if dt <= 0 or not np.isfinite(dt) or not 0 < forward_cmd <= .6:
            raise ValueError('Invalid timestep or forward command')
        if not 0 <= turn_cap <= .6 or not .45 <= standoff <= .8:
            raise ValueError('Invalid turn cap or standoff')
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.dt, self.settle_calls, self.ramp_calls = dt, settle_calls, max(ramp_calls, 1)
        self.reach_only, self.forward_cmd, self.turn_cap, self.standoff = reach_only, forward_cmd, turn_cap, standoff
        self.turn_gain = float(turn_gain)
        if not np.isfinite(turn_gain) or not 0 < turn_gain <= 3:
            raise ValueError('Invalid turn gain')
        self.recover_until = 0
        self.quiet_calls = 0
        self.pause_for_stance = False
        self.near_view = False
        self.view_locked = False
        if not np.isfinite(lowering_m) or not 0 <= lowering_m <= .25:
            raise ValueError('Reach lowering must lie in [0,.25] m')
        if grasp and lowering_m <= 0.:
            raise ValueError('grasp requires a lowering amplitude: the jaws cannot reach an object '
                             'from the unlowered stance')
        self.grasp = bool(grasp)
        self.lowering_m, self.lowering_alpha = lowering_m, 0.
        self.leg = schema.term(LEG_TERM)
        # The descent reference is solved from the leg geometry at the moment it is
        # authorized, against the pose actually measured then -- see
        # _solve_lower_delta. It is zero until then, and lowering_alpha is zero with
        # it, so no descent is commanded before authorization.
        self.lower_delta = np.zeros(len(self.leg.joint_names))
        self.arm, self.wheel = schema.term(ARM_TERM), schema.term(WHEEL_TERM)
        self.sides = np.array([1. if wheel_side(n) == 'right' else -1. for n in self.wheel.joint_names])
        self.calls, self.alpha, self.state, self.debug = 0, 0., 'SETTLE', {}
        self.target, self.confirmations, self.last_seen = None, 0, -10000
        self.previous_wheels = np.zeros(4)
        self.arm_command = np.array([defaults[n] for n in ('arm_joint1', 'arm_joint2', 'arm_joint3',
                                                          'arm_joint4', 'arm_joint5', 'arm_joint6',
                                                          'arm_joint7', 'arm_joint8')])
        self.reach_q, self.brake_start, self.reach_start = None, None, None
        self.vision_stride = max(1, round(.1 / dt))
        self.stationary_gate = StationaryTargetGate(dt=self.dt, sensor_period=.1)
        self.done_reason = None
        self.state_reason = 'initial_settle'
        self.state_entry_call, self._recorded_state = 0, self.state
        self.state_calls, self.state_transitions = {}, []
        self.phase, self.phase_start = 'SETTLE', 0
        self.wait_start = self.recovery_start = None
        self.creep_anchor = None
        self.brake_retries = 0
        self.vision_samples, self.last_vision_sample = 0, None
        self.camera_shapes = {}
        self.camera_available = set()
        self.reach_target = None
        self.reach_visual_valid, self.reach_valid_samples = False, 0
        self.reach_lost_start = self.reach_progress_anchor = self.reach_at_goal_start = None
        self.reach_timeout_s = None
        self.reach_settle_start, self.reach_settle_calls = None, 0
        self.reach_settle_quiet_calls, self.reach_settle_episodes = 0, 0
        self.reach_settle_cause = None
        self.reach_up_anchor = None
        self.reach_base_displacement, self.reach_base_yaw = np.zeros(3), 0.
        self.arm_obs_ids = np.array([self.names.index('arm_joint'+str(i)) for i in range(1, 7)])
        self.leg_obs_ids = np.array([self.names.index(n) for n in self.leg.joint_names])
        self.leg_defaults = np.array([self.defaults[n] for n in self.leg.joint_names])
        self.stance_verified, self.stance_quiet_calls, self.stance_check_start = False, 0, None
        # Fixed-amplitude lowering gate. Authorization is latched once a single
        # continuous public quiet window forms; no image, detection or later
        # observation can re-arm, refresh or repeat it.
        self.lower_window_calls = max(1, round(.5 / dt))
        self.lower_authorized, self.lower_authorized_call = False, None
        self.lower_reference_q0 = None
        self.lower_wait_start, self.lower_quiet_calls = None, 0
        self.lower_quiet_window_s, self.lower_leg_window = 0., []
        self.lower_increment_error_rad, self.lower_increment_fault_start = 0., None
        self.lower_beta, self.lower_alpha_half_call = None, None
        self.lower_hold_start = None
        # The descent window is bounded on its own; when a grasp follows, the same
        # budget has to cover the close-and-lift sequence too, or the window would
        # expire mid-grasp.
        self.grasp_close_s = 2.5      # long enough for the 0.02 m/s finger slew
        self.grasp_hold_s = 1.5
        self.grasp_lift_s = 3.
        self.grasp_budget_s = 12.
        self.lower_window_s = 6. + (self.grasp_budget_s if grasp else 0.)
        # Jaw command, in metres of finger travel. The grip closes by driving both
        # finger joints to their zero stops; see task_b/arm_kinematics.gripper_targets.
        self.jaw_target = np.array([.035, -.035])
        self.grasp_stage, self.grasp_stage_start = None, None
        self.grasp_start = None
        arm_joints_from_proprio(np.zeros(84), self.names, self.defaults)

    def _phase(self, phase):
        if self.phase != phase:
            self.phase, self.phase_start = phase, self.calls
            self.creep_anchor = None

    def _finish(self, reason):
        """Request a normal evaluator stop while holding the last command.

        This is an experimental stopping reason, never an official task
        termination or a claim of score/success.
        """
        self.done_reason = self.state_reason = reason
        self.state = 'STOPPED'
        return self._action()

    def _action(self, wheels=None):
        out = np.zeros(self.schema.total_dim, dtype=np.float32)
        out[self.arm.start:self.arm.stop] = arm_targets_to_action(
            self.arm_command, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        out[self.leg.start:self.leg.stop] = self.lowering_alpha*self.lower_delta/self.leg.scale
        if wheels is None:
            self.previous_wheels[:] = 0.
        else:
            self.previous_wheels += np.clip(wheels - self.previous_wheels, -self.dt*.5, self.dt*.5)
            out[self.wheel.start:self.wheel.stop] = self.previous_wheels
        if self.state != self._recorded_state:
            self.state_transitions.append({'from': self._recorded_state, 'to': self.state,
                                           'step': self.calls, 'reason': self.state_reason})
            self._recorded_state, self.state_entry_call = self.state, self.calls
        self.state_calls[self.state] = self.state_calls.get(self.state, 0)+1
        self.debug.update(state=self.state, reason=self.state_reason, done_reason=self.done_reason,
                          state_age_s=(self.calls-self.state_entry_call)*self.dt,
                          phase=self.phase, phase_age_s=(self.calls-self.phase_start)*self.dt,
                          vision_samples=self.vision_samples, last_vision_sample=self.last_vision_sample,
                          wheel_action=out[self.wheel.start:self.wheel.stop].tolist(),
                          arm_target=self.arm_command.tolist(), lowering_alpha=self.lowering_alpha,
                          lowering_m=self.lowering_m,
                          lower_authorized=bool(self.lower_authorized),
                          lower_authorized_step=self.lower_authorized_call,
                          lower_reference_q0=(None if self.lower_reference_q0 is None
                                              else self.lower_reference_q0.tolist()),
                          lower_increment_error_rad=float(self.lower_increment_error_rad),
                          lower_beta=self.lower_beta,
                          lower_quiet_window_s=float(self.lower_quiet_window_s),
                          lower_remaining_s=self._lower_remaining_s(),
                          lower_remaining_basis='seconds left in the 3 s admission wait before '
                                                'authorization, then in the 6 s descent-plus-hold window')
        self.debug.setdefault('reach_subphase', None)
        return out

    def _lower_remaining_s(self):
        """Seconds left in whichever bounded lowering window is running."""
        if self.lower_authorized:
            return max(0., 6.-(self.calls-self.lower_authorized_call)*self.dt)
        if self.lower_wait_start is not None:
            return max(0., 3.-(self.calls-self.lower_wait_start)*self.dt)
        return None

    def _reach_deadline_s(self):
        """Total per-target deadline; only authorization adds the descent window.

        The original arm-reach deadline is never reset by a detection, a pause or
        the admission wait, all of which keep counting inside it. The grasp
        sequence is itself bounded, so its budget extends the deadline rather than
        being cut off by the reach's own.
        """
        return (self.reach_timeout_s + (6. if self.lower_authorized else 0.)
                + (self.grasp_budget_s if self.grasp else 0.))

    def _detect(self, obs, images, q):
        candidates, errors, detectors = [], {}, {}
        # Local shape-assisted association is only offered after the stationary
        # gate has already authorized a fixed goal. Approach and BRAKE keep the
        # unchanged default filtering, so nothing about initial localization is
        # relaxed. A locked association can never retarget the reach.
        locked = None
        if self.reach_q is not None and self.reach_target is not None and not self.reach_only:
            locked = self.reach_target.copy()
        self.camera_available = set()
        for source in ('ee', 'head'):
            try:
                rgb, depth = _camera_arrays(images, source)
                self.camera_shapes[source] = tuple(depth.shape)
                self.camera_available.add(source)
                pose = ee_camera_transform(q) if source == 'ee' else head_camera_transform()
                items, info = detect_yellow_candidates(rgb, depth, pose, projected_gravity=obs[9:12],
                                                   focal_length=15. if source == 'ee' else 24.,
                                                   locked_target_body=locked)
                detectors[source] = {'yellow_components': info.get('yellow_components'),
                                     'rejected': info.get('rejected'),
                                     'association_mode': info.get('association_mode'),
                                     'locked_shape_bypass': info.get('locked_shape_bypass'),
                                     'accepted': len(items)}
                for item in items:
                    item['source'] = source
                candidates.extend(items)
            except (ValueError, TypeError) as e:
                errors[source] = str(e)
        self.debug.update(candidates=candidates, camera_errors=errors, detector_rejections=detectors,
                          locked_target_body=None if locked is None else locked.tolist(),
                          association_claim='locked mode is local geometry near the already localized '
                                            'point; not object identity or continuous tracking')
        if not candidates:
            return
        # Steering demand is more important than raw distance for this chassis.
        selected = min(candidates, key=lambda c: abs(np.arctan2(c['body_point'][1], c['body_point'][0]))
                       + .035*c['forward_planar_distance_m'])
        if self.target is not None:
            nearest = min(candidates, key=lambda c: np.linalg.norm(np.asarray(c['body_point'])-self.target))
            if np.linalg.norm(np.asarray(nearest['body_point'])-self.target) < .35:
                selected = nearest
                self.confirmations += 1
            elif self.calls-self.last_seen < round(1./self.dt):
                return
            else:
                self.confirmations = 1
        else:
            self.confirmations = 1
        point = np.array(selected['body_point'])
        self.target = point if self.target is None or self.confirmations == 1 else .7*point + .3*self.target
        self.last_seen = self.calls
        self.debug.update(selected=selected, target_body=self.target.tolist())
        # BRAKE confirmation must consume this observation's raw point/source,
        # never the moving-camera EMA stored in self.target.
        return selected

    def _target_expected_visible(self, q):
        """Project the frozen, visually localized target using static cameras.

        The arm reach rotates its camera away from the floor target. This
        geometric prediction allows a bounded P1 joint-space movement without
        claiming continuous visual tracking; it never enables lowering.
        """
        projections = {}
        for source, (height, width) in self.camera_shapes.items():
            camera = ee_camera_transform(q) if source == 'ee' else head_camera_transform()
            point = camera[:3, :3].T @ (self.reach_target-camera[:3, 3])
            focal = width*(15. if source == 'ee' else 24.)/20.955
            visible = False
            uv = None
            if point[2] > .1:
                uv = (focal*point[:2]/point[2] + [width/2., height/2.]).tolist()
                visible = 0 <= uv[0] < width and 0 <= uv[1] < height
            projections[source] = {'expected_center_in_frame': bool(visible), 'pixel_uv': uv,
                                   'optical_depth_m': float(point[2]),
                                   'reason': ('inside_detector_near_limit' if point[2] <= .1 else
                                              'center_inside_frame' if visible else 'center_outside_frame')}
        self.debug['reach_camera_projection'] = projections
        # Missing camera metadata cannot establish that the target is out of
        # view. Both original cameras must be covered by this exemption.
        if set(projections) != {'head', 'ee'} or self.camera_available != {'head', 'ee'}:
            return True
        return any(item['expected_center_in_frame'] for item in projections.values())

    def _base_budget_exit(self, obs):
        """Integrate public body twist inside REACH against small local budgets.

        Small-angle integration of the public linear/angular velocity in the
        current body frame. It is a local drift estimate, not world truth, and
        the vertical component is deliberately removed so no vertical motion can
        ever be integrated in pursuit of proximity. The existing instantaneous
        velocity stop stays in force independently of these budgets.
        """
        gravity = obs[9:12]
        up = -gravity/np.linalg.norm(gravity)
        velocity = obs[:3] - np.dot(obs[:3], up)*up
        self.reach_base_displacement = self.reach_base_displacement + self.dt*velocity
        self.reach_base_yaw += self.dt*float(np.dot(obs[3:6], up))
        displacement = float(np.linalg.norm(self.reach_base_displacement))
        yaw = abs(float(self.reach_base_yaw))
        anchor = up if self.reach_up_anchor is None else self.reach_up_anchor
        gravity_change = float(np.arccos(np.clip(np.dot(up, anchor), -1., 1.)))
        self.debug.update(reach_base_displacement_m=displacement, reach_base_yaw_rad=yaw,
                          reach_gravity_change_rad=gravity_change,
                          reach_base_budgets={'displacement_m': .03, 'yaw_rad': .05,
                                              'gravity_rad': .05},
                          reach_base_integration_claim='approximate small-angle integration of public '
                                                       'twist in the local frame; not world odometry')
        if displacement > .03:
            return 'reach_base_displacement_budget_exceeded'
        if yaw > .05:
            return 'reach_base_yaw_budget_exceeded'
        if gravity_change > .05:
            return 'reach_gravity_changed'
        return None

    def _public_motion(self, obs):
        """Record the public body twist the stationarity guards actually read.

        The tangential speed removes the vertical component in the same way as
        the displacement budget, so the reported number is the quantity that is
        integrated; none of it is world odometry.
        """
        gravity = obs[9:12]
        up = -gravity/np.linalg.norm(gravity)
        linear, angular = obs[:3], obs[3:6]
        motion = {'linear': linear.tolist(), 'linear_norm': float(np.linalg.norm(linear)),
                  'angular': angular.tolist(), 'angular_norm': float(np.linalg.norm(angular)),
                  'tangent_speed_m_s': float(np.linalg.norm(linear-np.dot(linear, up)*up))}
        self.debug.update(reach_public_linear_velocity=motion['linear'],
                          reach_public_linear_norm=motion['linear_norm'],
                          reach_public_angular_velocity=motion['angular'],
                          reach_public_angular_norm=motion['angular_norm'],
                          reach_base_tangent_speed_m_s=motion['tangent_speed_m_s'],
                          reach_stationarity_thresholds={'linear_norm_m_s': .06, 'angular_norm_rad_s': .12})
        return motion

    def _reach_settle(self, q, motion, violation):
        """Bounded pause and recovery for a zero-lowering stationarity violation.

        The thresholds are exactly the previous instantaneous ones: nothing is
        relaxed. A violating public twist now freezes the already fixed joint
        command for a bounded time and demands a continuously quiet window
        before the original goal resumes, instead of ending the run on a single
        sample. A settle is never a claim that the base held still, and the
        drift budgets and the total reach deadline keep running throughout.
        Returns an action while settling, or None when the caller may continue.
        """
        if self.reach_settle_start is None:
            if not violation:
                return None
            self.reach_settle_start, self.reach_settle_quiet_calls = self.calls, 0
            self.reach_settle_episodes += 1
            self.reach_settle_cause = (
                'public_linear_and_angular_norm' if motion['linear_norm'] >= .06 and motion['angular_norm'] >= .12
                else 'public_linear_norm' if motion['linear_norm'] >= .06 else 'public_angular_norm')
        # Both violating and recovery-quiet ticks count towards the durations.
        # A further violating call restarts only the quiet window, never the
        # episode clock, and the accumulated total never resets in this reach.
        self.reach_settle_calls += 1
        self.reach_settle_quiet_calls = 0 if violation else self.reach_settle_quiet_calls+1
        # A rate settle is never part of the lowering admission window, and the
        # current alpha and arm command stay frozen while it runs.
        if not self.lower_authorized:
            self.lower_quiet_calls, self.lower_leg_window = 0, []
            self.lower_quiet_window_s = 0.
        episode_s = (self.calls-self.reach_settle_start+1)*self.dt
        total_s = self.reach_settle_calls*self.dt
        quiet_s = self.reach_settle_quiet_calls*self.dt
        error = float(np.max(np.abs(q[:6]-self.reach_q)))
        # A frozen command is not commanded motion: hold the progress anchor at
        # the current error so a settle cannot be read as a joint stall, and
        # restart the final observation window after any pause.
        self.reach_progress_anchor = (self.calls, error)
        self.reach_at_goal_start = None
        self.state_reason = 'bounded_settle_after_public_base_motion_before_resuming_fixed_goal'
        self.debug.update(reach_subphase='SETTLE_AFTER_MOTION', reach_joint_error_rad=error,
                          reach_settle_cause=self.reach_settle_cause,
                          reach_settle_episodes=self.reach_settle_episodes,
                          reach_settle_episode_s=episode_s, reach_settle_total_s=total_s,
                          reach_settle_quiet_s=quiet_s,
                          reach_settle_budgets={'episode_s': 2., 'total_s': 3., 'quiet_s': .2},
                          reach_settle_claim='bounded pause holding the last command with unchanged '
                                             'thresholds; not a claim that the base stayed stationary')
        age = (self.calls-self.reach_start)*self.dt
        self.debug.update(reach_age_s=age, reach_timeout_s=self.reach_timeout_s,
                          reach_deadline_s=self._reach_deadline_s())
        if age >= self._reach_deadline_s():
            return self._finish('reach_joint_motion_timeout')
        if episode_s >= 2.:
            return self._finish('reach_settle_timeout')
        if total_s >= 3.:
            return self._finish('reach_settle_budget_exceeded')
        if quiet_s >= .2:
            self.reach_settle_start, self.reach_settle_quiet_calls = None, 0
            self.reach_progress_anchor = None  # re-anchor progress on resuming
            self.debug.update(reach_subphase='RESUME_FIXED_GOAL', reach_settle_resumed_step=self.calls)
            return None
        return self._action()

    def _reach(self, obs, q, sampled, selected):
        self.state, self.state_reason = 'REACH', 'bounded_joint_reach_after_stationary_visual_localization'
        self.debug.update(reach_goal_q=self.reach_q.tolist(),
                          predicted_gripper_body=fk(self.reach_q)[:3, 3].tolist(),
                          reach_target_body=self.reach_target.tolist(),
                          reach_subphase='ARM_REACH',
                          reach_settle_episodes=self.reach_settle_episodes,
                          reach_settle_total_s=self.reach_settle_calls*self.dt)
        motion, violation = None, False
        if not self.reach_only:
            motion = self._public_motion(obs)
            # Admission and descent deadlines also run during a rate-settle
            # early return; pausing must never extend either bounded window.
            if self.lower_authorized and (self.calls-self.lower_authorized_call)*self.dt >= self.lower_window_s:
                return self._finish('lower_window_expired')
            if (not self.lower_authorized and self.lower_wait_start is not None
                    and (self.calls-self.lower_wait_start)*self.dt >= 3.):
                return self._finish('lower_quiet_window_not_formed')
            violation = motion['linear_norm'] >= .06 or motion['angular_norm'] >= .12
            budget_exit = self._base_budget_exit(obs)
            if budget_exit is not None:
                return self._finish(budget_exit)
            expected_visible = self._target_expected_visible(q)
            if sampled:
                if selected is None:
                    self.reach_visual_valid, self.reach_valid_samples = False, 0
                else:
                    displacement = float(np.linalg.norm(np.asarray(selected['body_point'])-self.reach_target))
                    self.debug['reach_target_displacement_m'] = displacement
                    if displacement > .12:
                        # A distant component is unmatched at every amplitude and
                        # never read as the localized target having moved: that
                        # reading would let another object replace the one goal.
                        self.debug['reach_unassociated_component'] = True
                        self.reach_visual_valid, self.reach_valid_samples = False, 0
                    else:
                        self.reach_valid_samples += 1
                        self.reach_visual_valid = self.reach_valid_samples >= 2
            expected_out_of_view = not expected_visible
            # The stationary gate already authorized this single fixed goal and
            # q_goal from two stationary raw detections, at any amplitude.
            # Missing candidates, a shape rejection, an unavailable camera or an
            # out-of-frame prediction are diagnostics only: requiring continued
            # shape detections would be an impossible condition while the arm
            # camera sweeps away. Nothing here re-solves IK, retargets the goal,
            # authorizes lowering or resets a deadline.
            self.reach_lost_start = None
            self.debug.update(expected_out_of_view=expected_out_of_view,
                              out_of_view_reach_allowed=True,
                              reach_visual_valid=self.reach_visual_valid,
                              reach_valid_samples=self.reach_valid_samples,
                              reach_visual_association=('locally_associated' if self.reach_visual_valid
                                                        else 'unconfirmed_diagnostic_only'),
                              reach_camera_available=sorted(self.camera_available))
            if not self.reach_visual_valid:
                self.state_reason = ('bounded_joint_reach_visual_association_optional_no_lowering'
                                     if self.lowering_m == 0. else
                                     'bounded_joint_reach_visual_association_optional_fixed_lowering')
            # Single-factor pause on the unchanged instantaneous thresholds. The
            # drift budgets above and the scheduled detector have already run on
            # this call, and the state stays REACH because the wheel hold
            # depends on it.
            settling = self._reach_settle(q, motion, violation)
            if settling is not None:
                return settling
        age = (self.calls-self.reach_start)*self.dt
        error = float(np.max(np.abs(q[:6]-self.reach_q)))
        self.debug.update(reach_age_s=age, reach_timeout_s=self.reach_timeout_s,
                          reach_deadline_s=self._reach_deadline_s(), reach_joint_error_rad=error)
        if not self.reach_only and age >= self._reach_deadline_s():
            return self._finish('reach_joint_motion_timeout')
        if self.reach_progress_anchor is None:
            self.reach_progress_anchor = (self.calls, error)
        progress_age = (self.calls-self.reach_progress_anchor[0])*self.dt
        if progress_age >= 3.:
            if not self.reach_only and error > .04 and self.reach_progress_anchor[1]-error < .02:
                return self._finish('reach_joint_progress_below_002rad_in_3s')
            self.reach_progress_anchor = (self.calls, error)
        # Measured-state tether prevents accumulating a large command error.
        step = np.clip(self.reach_q-self.arm_command[:6], -.30*self.dt, .30*self.dt)
        proposed = self.arm_command[:6] + step
        self.arm_command[:6] = np.clip(proposed, q[:6]-.10, q[:6]+.10)
        self.arm_command[6:] += np.clip(self.jaw_target - self.arm_command[6:],
                                       -.02*self.dt, .02*self.dt)
        if self.lowering_m > 0. and not self.reach_only:
            outcome = self._lowering(obs, error, motion)
            return self._action() if outcome is None else outcome
        if error < .04:
            settled = True
            if not self.reach_only:
                # Zero-lowering completion needs the arm and the base to be
                # actually settled for the whole window, so a transient actual
                # joint value at a hard limit can never finish the reach.
                arm_speed = float(np.max(np.abs(obs[36+self.arm_obs_ids])))
                settled = (arm_speed < .05 and motion['tangent_speed_m_s'] < .01
                           and motion['linear_norm'] < .06 and motion['angular_norm'] < .12)
                self.debug.update(reach_arm_speed_rad_s=arm_speed, reach_observation_settled=settled,
                                  reach_observation_requirements={'joint_error_rad': .04,
                                                                  'arm_speed_rad_s': .05,
                                                                  'tangent_speed_m_s': .01,
                                                                  'linear_norm_m_s': .06,
                                                                  'angular_norm_rad_s': .12, 'window_s': 2.})
            if settled:
                self.reach_at_goal_start = self.calls if self.reach_at_goal_start is None else self.reach_at_goal_start
                if not self.reach_only and (self.calls-self.reach_at_goal_start)*self.dt >= 2.:
                    return self._finish('reach_observation_complete')
            else:
                self.reach_at_goal_start = None
        else:
            self.reach_at_goal_start = None
        return self._action()

    def _solve_lower_delta(self, leg_q):
        """Leg-angle delta that lowers the body by lowering_m with the wheels planted.

        Solved from the pose measured at authorization rather than from nominal
        defaults: the loaded legs sit well below their unloaded angles, so the
        descent must be relative to where the robot actually is. Holding each
        wheel's (x, y) keeps the wheelbase and track, which makes the body drop by
        the requested amount. Raises instead of commanding an unsolvable descent.
        """
        by_corner = {corner: np.array([leg_q[self.leg.joint_names.index(name)]
                                       for name in legs.leg_joint_names(corner)])
                     for corner in legs.CORNERS}
        solved, residual = legs.descend_all(by_corner, self.lowering_m)
        if residual > 1e-6:
            raise RuntimeError(f"Lowering IK residual {residual:.2e} m at {self.lowering_m:.4f} m")
        delta = np.array([solved[name.split('_')[0]][legs.LEG_LINKS.index(name.split('_')[1])]
                          for name in self.leg.joint_names]) - leg_q
        rise = {corner: float(legs.foot_body_xyz(corner, *solved[corner])[2]
                              - legs.foot_body_xyz(corner, *by_corner[corner])[2])
                for corner in legs.CORNERS}
        if max(abs(value - self.lowering_m) for value in rise.values()) > 1e-4:
            raise RuntimeError(f"Solved descent misses {self.lowering_m:.4f} m: {rise}")
        self.debug['lower_solved_delta_rad'] = delta.tolist()
        self.debug['lower_solved_residual_m'] = residual
        self.debug['lower_solved_rise_per_corner_m'] = rise
        return delta

    def _lowering(self, obs, error, motion):
        """Admit and drive the fixed-amplitude original leg reference descent.

        Authorization needs one continuous public quiet window of actual arm,
        base and leg observations. It is latched: no later image, missing
        detection or unmatched component can re-arm it, copy q0 again or refresh
        a deadline. Alpha then rises by at most dt/3 per step towards the CLI
        amplitude, with the same arm goal and wheel anchor. A reference
        increment is not a claim that the body descended; only public leg joint
        response is measured. Returns an action to stop, or None to continue.
        """
        leg_q = obs[12+self.leg_obs_ids] + self.leg_defaults
        gravity = obs[9:12]
        tilt = float(np.arccos(np.clip(-gravity[2]/np.linalg.norm(gravity), -1., 1.)))
        if not self.lower_authorized:
            arm_speed = float(np.max(np.abs(obs[36+self.arm_obs_ids])))
            quiet = bool(error < .04 and arm_speed < .05 and motion['tangent_speed_m_s'] < .01
                         and motion['linear_norm'] < .06 and motion['angular_norm'] < .12
                         and tilt <= .10 and self.reach_settle_start is None)
            if error < .04 and self.lower_wait_start is None:
                self.lower_wait_start = self.calls
            if quiet:
                self.lower_quiet_calls += 1
                self.lower_leg_window.append(leg_q.copy())
                del self.lower_leg_window[:-self.lower_window_calls]
            else:
                self.lower_quiet_calls, self.lower_leg_window = 0, []
            self.lower_quiet_window_s = self.lower_quiet_calls*self.dt
            # Instantaneous contact-solver leg rates alternate sign at high
            # frequency, so sustained leg displacement over the window is the
            # stillness evidence, not any single raw velocity sample.
            window_full = len(self.lower_leg_window) >= self.lower_window_calls
            leg_range = (float(np.max(np.ptp(np.asarray(self.lower_leg_window), axis=0)))
                         if window_full else None)
            self.debug.update(lower_quiet_tick=quiet,
                              lower_arm_speed_rad_s=arm_speed, lower_tilt_rad=tilt,
                              lower_leg_window_range_rad=leg_range,
                              lower_admission_requirements={'joint_error_rad': .04,
                                                            'arm_speed_rad_s': .05,
                                                            'tangent_speed_m_s': .01,
                                                            'linear_norm_m_s': .06,
                                                            'angular_norm_rad_s': .12,
                                                            'tilt_rad': .10,
                                                            'leg_window_range_rad': .02,
                                                            'window_s': .5, 'max_wait_s': 3.},
                              lower_admission_claim='continuous public quiet window over actual arm, '
                                                    'base and leg observations; not a stillness claim')
            if self.lower_wait_start is not None:
                self.debug['reach_subphase'] = 'WAIT_LOWER_QUIET'
            if self.lower_quiet_calls >= self.lower_window_calls and leg_range is not None and leg_range < .02:
                self.lower_authorized, self.lower_authorized_call = True, self.calls
                self.lower_reference_q0 = leg_q.copy()
                self.lower_delta = self._solve_lower_delta(leg_q)
                self.debug.update(lower_authorized_quiet_window_s=self.lower_quiet_window_s,
                                  lower_authorized_leg_window_range_rad=leg_range)
            elif self.lower_wait_start is not None and (self.calls-self.lower_wait_start)*self.dt >= 3.:
                return self._finish('lower_quiet_window_not_formed')
            else:
                self.state_reason = 'waiting_for_public_quiet_window_before_fixed_lowering'
                return None
        if self.grasp_start is not None:
            # Once the grasp has begun it owns the leg reference: re-entering the
            # alpha gate below would ramp alpha straight back up and cancel the lift.
            return self._grasp(obs)
        # Bounded joint response check against the commanded increment. Both
        # measures describe leg joints only, never actual body height.
        increment_error = float(np.max(np.abs((leg_q-self.lower_reference_q0)
                                             - self.lowering_alpha*self.lower_delta)))
        denominator = float(np.dot(self.lower_delta, self.lower_delta))
        self.lower_increment_error_rad = increment_error
        self.lower_beta = (float(np.dot(leg_q-self.lower_reference_q0, self.lower_delta)/denominator)
                           if denominator > 0. else None)
        elapsed = (self.calls-self.lower_authorized_call)*self.dt
        self.debug.update(lower_elapsed_s=elapsed, lower_hold_s=(0. if self.lower_hold_start is None else
                                                                 (self.calls-self.lower_hold_start)*self.dt),
                          lower_response_limits={'increment_error_rad': .06, 'fault_s': .2,
                                                 'beta_after_half_alpha_s': 1., 'beta_min': .15,
                                                 'window_s': self.lower_window_s, 'hold_s': 2.},
                          lower_response_claim='leg joint response to the fixed reference increment; '
                                               'no claim about actual chassis descent')
        if increment_error > .06:
            self.lower_increment_fault_start = (self.calls if self.lower_increment_fault_start is None
                                                else self.lower_increment_fault_start)
            if (self.calls-self.lower_increment_fault_start+1)*self.dt >= .2:
                return self._finish('lower_joint_increment_error_exceeded')
        else:
            self.lower_increment_fault_start = None
        if self.lowering_alpha >= .5:
            self.lower_alpha_half_call = (self.calls if self.lower_alpha_half_call is None
                                          else self.lower_alpha_half_call)
            if ((self.calls-self.lower_alpha_half_call)*self.dt >= 1.
                    and self.lower_beta is not None and self.lower_beta < .15):
                return self._finish('lower_reference_no_joint_progress')
        if self.lowering_alpha >= 1.:
            self.lower_hold_start = self.calls if self.lower_hold_start is None else self.lower_hold_start
            hold_s = (self.calls-self.lower_hold_start)*self.dt
            self.state_reason = 'holding_fixed_lowered_reference_after_bounded_descent'
            self.debug.update(reach_subphase='LOWER_HOLD', lower_hold_s=hold_s)
            if hold_s >= 2.:
                if self.grasp:
                    return self._grasp(obs)
                return self._finish('reach_lowering_hold_complete')
        else:
            self.state_reason = 'incrementing_fixed_leg_reference_after_authorized_quiet_window'
            self.debug['reach_subphase'] = 'LOWER_REFERENCE'
            # Alpha only advances while the actual arm is at the fixed goal and
            # no rate settle is running; a stop never clears or reverses it.
            if error < .04 and self.reach_settle_start is None:
                self.lowering_alpha = min(1., self.lowering_alpha+self.dt/3.)
        if elapsed >= self.lower_window_s:
            return self._finish('lower_window_expired')
        return None

    def _grasp(self, obs):
        """Close the jaws on whatever the reach placed them around, then lift.

        This asserts no grasp. The policy cannot see the object, so it commands the
        only two things it controls -- finger closure, and a bounded return of the
        leg reference along the path it descended -- and the evaluator's own object
        and gripper records decide whether anything was actually held.
        """
        if self.grasp_start is None:
            self.grasp_start = self.calls
        if self.grasp_stage is None:
            self.grasp_stage, self.grasp_stage_start = 'CLOSE', self.calls
        stage_s = (self.calls-self.grasp_stage_start)*self.dt
        total_s = (self.calls-self.grasp_start)*self.dt
        slack = float(np.max(np.abs(self.arm_command[6:])))
        self.debug.update(reach_subphase='GRASP_'+self.grasp_stage, grasp_stage_s=stage_s,
                          grasp_total_s=total_s, grasp_budget_s=self.grasp_budget_s,
                          grasp_jaw_target=self.jaw_target.tolist(),
                          grasp_jaw_command=self.arm_command[6:].tolist(),
                          grasp_finger_slack_m=slack, lowering_alpha=self.lowering_alpha,
                          grasp_claim='the two controlled actions only; whether an object is '
                                      'between the fingers is measured, not claimed')
        if total_s >= self.grasp_budget_s:
            return self._finish('reach_grasp_timeout')

        if self.grasp_stage == 'CLOSE':
            self.jaw_target = np.zeros(2)
            self.state_reason = 'closing_jaws_at_the_lowered_reach_goal'
            if slack <= 1e-3:
                self.grasp_stage, self.grasp_stage_start = 'CLOSE_HOLD', self.calls
            return self._action()

        if self.grasp_stage == 'CLOSE_HOLD':
            self.state_reason = 'holding_closed_jaws_before_lifting'
            if stage_s >= self.grasp_hold_s:
                self.grasp_stage, self.grasp_stage_start = 'LIFT', self.calls
            return self._action()

        if self.grasp_stage == 'LIFT':
            self.state_reason = 'raising_the_body_along_the_descent_reference'
            self.lowering_alpha = max(0., self.lowering_alpha-self.dt/self.grasp_lift_s)
            if self.lowering_alpha <= 0.:
                self.grasp_stage, self.grasp_stage_start = 'LIFT_HOLD', self.calls
            return self._action()

        self.state_reason = 'holding_the_lifted_stance_after_the_grasp_attempt'
        if stage_s >= self.grasp_hold_s:
            return self._finish('reach_grasp_lift_complete')
        return self._action()

    def act(self, proprio, image):
        self.calls += 1
        self.alpha = float(np.clip((self.calls-self.settle_calls)/self.ramp_calls, 0., 1.))
        if self.done_reason is not None:
            return self._action()
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            return self._finish('invalid_proprio')
        q = arm_joints_from_proprio(obs, self.names, self.defaults)
        self.debug = {'gripper_body': fk(q)[:3, 3].tolist()}
        gravity = obs[9:12]
        if np.linalg.norm(gravity) < .5 or -gravity[2]/np.linalg.norm(gravity) < np.cos(.25):
            return self._finish('tilt_limit_exceeded')
        if self.calls <= self.settle_calls:
            return self._action()
        if self.pause_for_stance:
            self._phase('STANCE_TRANSITION')
            self.state = 'STANCE_TRANSITION'
            self.state_reason = 'reference_transition_requires_wheel_stop'
            self.stance_verified, self.stance_quiet_calls, self.stance_check_start = False, 0, None
            return self._action()
        tilt = np.arccos(np.clip(-gravity[2]/np.linalg.norm(gravity), -1., 1.))
        if tilt > .12 or np.linalg.norm(obs[3:5]) > .45:
            if self.reach_q is not None:
                return self._finish('reach_unstable_posture')
            self.recover_until = self.calls + 50
        if self.calls < self.recover_until or (self.recover_until and tilt > .085):
            self._phase('RECOVER_STANCE')
            self.state = 'RECOVER_STANCE'
            self.state_reason = 'public_tilt_or_rate_requires_recovery'
            self.recovery_start = self.calls if self.recovery_start is None else self.recovery_start
            if (self.calls-self.recovery_start)*self.dt >= 5.:
                return self._finish('stance_recovery_timeout')
            return self._action()
        self.recover_until = 0
        self.recovery_start = None
        if not self.stance_verified and not self.reach_only:
            self.stance_check_start = self.calls if self.stance_check_start is None else self.stance_check_start
            leg_speed = float(np.max(np.abs(obs[36+self.leg_obs_ids])))
            stable = tilt < .085 and np.linalg.norm(obs[3:6]) < .12 and leg_speed < .20
            self.stance_quiet_calls = self.stance_quiet_calls+1 if stable else 0
            self.debug.update(stance_leg_speed_rad_s=leg_speed, stance_quiet_calls=self.stance_quiet_calls)
            if self.stance_quiet_calls*self.dt < .5:
                self._phase('WAIT_FOR_STABLE_STANCE')
                self.state, self.state_reason = 'WAIT_FOR_STABLE_STANCE', 'verify_actual_post_transition_joint_motion'
                if (self.calls-self.stance_check_start)*self.dt >= 8.:
                    return self._finish('post_transition_stance_not_stable')
                return self._action()
            self.stance_verified = True
        # Short target propagation uses only public body twist. A fresh visual
        # confirmation is required before entering the stationary reach stage.
        if self.target is not None and self.reach_q is None and not self.reach_only:
            up = -gravity / np.linalg.norm(gravity)
            velocity = obs[:3] - np.dot(obs[:3], up)*up
            self.target -= self.dt * (velocity + np.cross(obs[3:6], self.target))
        selected = None
        sampled = not self.reach_only and self.calls % self.vision_stride == 0
        if sampled:
            self.vision_samples += 1
            self.last_vision_sample = self.calls
            selected = self._detect(obs, image, q)
        if self.reach_q is not None:
            return self._reach(obs, q, sampled, selected)
        if not self.reach_only and self.target is not None and not self.view_locked:
            # Tilting only the wrist placed its downward ray through the body
            # in the near field. Extend the shoulder/elbow so the camera clears
            # the nose, and aim using measured camera/arm geometry.
            self.near_view = bool(self.near_view or self.target[0] < 1.6)
            if self.near_view:
                self.arm_command[1] += np.clip(1.1-self.arm_command[1], -.40*self.dt, .40*self.dt)
                self.arm_command[2] += np.clip(-.8-self.arm_command[2], -.40*self.dt, .40*self.dt)
            camera = ee_camera_transform(q)
            delta = self.target-camera[:3, 3]
            camera_pitch = np.clip(np.arctan2(-delta[2], delta[0]) + .087-q[1]-q[2], 0., 1.21)
            self.arm_command[4] += np.clip(camera_pitch-self.arm_command[4], -.45*self.dt, .45*self.dt)
            self.debug.update(near_view=self.near_view, camera_pitch_target=float(camera_pitch))
        if self.reach_only:
            self.target = np.array([self.standoff, 0., -.30])
            self.brake_start = self.brake_start or self.calls
        elif self.brake_start is None and (self.target is None or self.confirmations < 2
                                         or self.calls-self.last_seen > round(.5/self.dt)):
            self._phase('WAIT_FOR_TARGET')
            self.state = 'WAIT_FOR_TARGET'
            self.state_reason = 'missing_or_stale_visual_target'
            self.wait_start = self.calls if self.wait_start is None else self.wait_start
            if (self.calls-self.wait_start)*self.dt >= 5.:
                return self._finish('no_valid_target_timeout')
            return self._action()
        self.wait_start = None
        if (self.brake_start is None and self.target is not None
                and self.target[0] <= self.standoff+.025 and abs(self.target[1]) < .22):
            self.brake_start = self.calls
            self.view_locked = True
            self.stationary_gate.reset(self.calls)
        if self.brake_start is not None:
            self._phase('BRAKE')
            self.state = 'BRAKE'
            self.state_reason = 'wait_for_actual_stillness_and_new_raw_visual_samples'
            quiet = (np.linalg.norm(obs[:3]) < .06 and np.linalg.norm(obs[3:6]) < .12
                     and np.max(np.abs(obs[36+self.arm_obs_ids])) < .05)
            self.quiet_calls = self.quiet_calls+1 if quiet else 0
            self.debug['arm_speed_rad_s'] = float(np.max(np.abs(obs[36+self.arm_obs_ids])))
            if (self.calls-self.brake_start)*self.dt >= 8.:
                return self._finish('brake_or_stationary_vision_timeout')
            if self.reach_only:
                if (self.calls-self.brake_start)*self.dt < 1. or self.quiet_calls*self.dt < .2:
                    return self._action()
            else:
                # Waiting one second also flushes moving-arm camera latency.
                # The independent gate clears its sample evidence on motion;
                # its output never includes the approach EMA in self.target.
                brake_wait_complete = (self.calls-self.brake_start)*self.dt >= 1.
                gate = self.stationary_gate.update(
                    self.calls, bool(quiet and brake_wait_complete),
                    point_body=None if selected is None else selected['body_point'],
                    source=None if selected is None else selected['source'], frame_token=None)
                self.debug['stationary_target_gate'] = gate
                if not gate['ready']:
                    return self._action()
                self.target = np.asarray(gate['target_body'], dtype=float)
            if not self.reach_only and self.target[0] > self.standoff+.05:
                self.brake_retries += 1
                if self.brake_retries > 2:
                    return self._finish('brake_rebound_retry_limit')
                self.brake_start, self.quiet_calls = None, 0
                self.stationary_gate.reset(self.calls)
                self.state = 'REAPPROACH_AFTER_SETTLING'
                self.state_reason = 'fresh_stationary_target_confirms_base_rebound'
                return self._action()
            up = -gravity/np.linalg.norm(gravity)
            desired = self.target + .12*up
            fit = solve_ik(desired, seed=q[:6], max_nfev=120)
            predicted = fk(fit.joints)[:3, 3]
            # A nearest feasible solution can still earn proximity; report the
            # residual explicitly and never present it as a successful IK pose.
            self.debug.update(ik_requested=desired.tolist(), ik_error_m=fit.position_error,
                              estimated_target_distance_m=float(np.linalg.norm(predicted-self.target)))
            if np.linalg.norm(predicted-self.target) > .30:
                return self._finish('stationary_target_unreachable')
            self.reach_q, self.reach_start = fit.joints, self.calls
            self.reach_target = self.target.copy()
            self.reach_visual_valid, self.reach_valid_samples = True, 2
            # Anchor the public up direction and zero the local drift integrals
            # that bound how far the base may move during the fixed reach.
            self.reach_up_anchor = up.copy()
            self.reach_base_displacement, self.reach_base_yaw = np.zeros(3), 0.
            travel = float(np.max(np.abs(self.reach_q-q[:6])))
            self.reach_timeout_s = max(15., travel/.30+10.)
            self._phase('REACH')
            self.state, self.state_reason = 'REACH_READY', 'stationary_raw_target_accepted_for_ik'
            return self._action()
        x, y, _ = self.target
        bearing = float(np.arctan2(y, x))
        self.debug.update(target_body=self.target.tolist(), bearing_rad=bearing,
                          visual_age_s=(self.calls-self.last_seen)*self.dt)
        if x < self.standoff-.08:
            return self._finish('target_too_close_outside_brake_window')
        self.state = 'ARC_APPROACH'
        self.state_reason = 'follow_visible_target_with_existing_arc_control'
        forward = self.forward_cmd * np.clip((x-self.standoff)/.6, .2, 1.)
        forward *= np.clip(1.-abs(bearing)/1.1, .15, 1.)
        # Measured skid steering is much slower than straight rolling. Keep
        # translation small until bearing is low enough to retain the target.
        forward *= np.clip(1.-abs(bearing)/.25, .04, 1.)
        turn = np.clip(self.turn_gain*bearing, -self.turn_cap, self.turn_cap)
        if x < 1.1 and abs(y) < .22:
            turn = 0.
            forward = self.forward_cmd*np.clip((x-self.standoff)/.25, .3, 1.)
            self._phase('CREEP')
            self.state = 'CREEP_WITHIN_ARM_WINDOW'
            self.state_reason = 'straight_near_field_motion_inside_brake_lateral_window'
            if self.creep_anchor is None:
                self.creep_anchor = (self.calls, float(x))
            elapsed = (self.calls-self.creep_anchor[0])*self.dt
            progress = self.creep_anchor[1]-float(x)
            self.debug.update(creep_progress_m=progress, creep_window_s=elapsed)
            if elapsed >= 3. and sampled and selected is not None:
                if progress < .01:
                    return self._finish('near_field_forward_progress_below_1cm_in_3s')
                self.creep_anchor = (self.calls, float(x))
        else:
            self._phase('APPROACH')
        return self._action(self.alpha*np.clip(forward+self.sides*turn, -.6, .6))

    def describe(self):
        return dict(mode='reach_probe' if self.reach_only else 'first_reach',
                    inputs='public proprio and head/ee RGB-D; static action and robot geometry',
                    target_selection='small bearing then distance; temporal visual match',
                    forward_cmd=self.forward_cmd, turn_cap=self.turn_cap, turn_gain=self.turn_gain, standoff=self.standoff,
                    lowering_m=self.lowering_m,
                    grasp_contract={'enabled': self.grasp,
                                    'close_travel_m': [0., 0.],
                                    'stages': ['CLOSE', 'CLOSE_HOLD', 'LIFT', 'LIFT_HOLD'],
                                    'budget_s': self.grasp_budget_s,
                                    'lift_s': self.grasp_lift_s,
                                    'jaw_target_m': self.jaw_target.tolist(),
                                    'scope': 'commands finger closure and a bounded return of the '
                                             'leg reference along the descent path; it makes no '
                                             'contact, force or held-object claim -- the evaluator '
                                             'records decide'},
                    settle_calls=self.settle_calls, calls=self.calls, state=self.state,
                    done_reason=self.done_reason,
                    visual_reach_contract='stationary two-sample raw localization of one yellow '
                    'component, then ONE fixed bounded joint-feedback reach; the goal is never '
                    're-solved or retargeted. Shape-assisted local association within .12 m is '
                    'optional diagnostics only: there is no continuous visual identity or tracking '
                    'guarantee, and at every lowering amplitude a missing, rejected, unmatched or '
                    'out-of-frame detection does not stop the bounded joint motion and never '
                    'authorizes lowering',
                    reach_base_budgets={'displacement_m': .03, 'yaw_rad': .05, 'gravity_rad': .05,
                                        'basis': 'approximate local integration of public twist, '
                                                 'no vertical component, not world odometry'},
                    reach_settle_contract={'linear_norm_m_s': .06, 'angular_norm_rad_s': .12,
                                           'quiet_s': .2, 'episode_s': 2., 'total_s': 3.,
                                           'scope': 'whole reach at any amplitude; unchanged thresholds '
                                                    'pause the fixed goal and freeze alpha instead of '
                                                    'stopping the run on one sample, time keeps counting, '
                                                    'and a pause is not a stillness claim',
                                           'episodes': self.reach_settle_episodes,
                                           'settled_s': self.reach_settle_calls*self.dt},
                    lowering_contract={'amplitude_m': self.lowering_m, 'alpha': self.lowering_alpha,
                                       'alpha_rate_per_step': self.dt/3.,
                                       'admission': {'window_s': .5, 'max_wait_s': 3.,
                                                     'joint_error_rad': .04, 'arm_speed_rad_s': .05,
                                                     'tangent_speed_m_s': .01, 'linear_norm_m_s': .06,
                                                     'angular_norm_rad_s': .12, 'tilt_rad': .10,
                                                     'leg_window_range_rad': .02},
                                       'authorized': bool(self.lower_authorized),
                                       'authorized_step': self.lower_authorized_call,
                                       'window_s': self.lower_window_s, 'hold_s': 2.,
                                       'response': {'increment_error_rad': .06, 'fault_s': .2,
                                                    'beta_min': .15, 'beta_after_half_alpha_s': 1.},
                                       'scope': 'one latched authorization from a continuous public quiet '
                                                'window; a fixed CLI amplitude on the original leg '
                                                'reference, never chosen from score or simulation truth. '
                                                'Leg increment error and beta measure joint response '
                                                'only and claim no actual chassis descent'},
                    stage_stats={'state_calls': dict(self.state_calls), 'transitions': list(self.state_transitions),
                                 'phase': self.phase, 'phase_age_s': (self.calls-self.phase_start)*self.dt,
                                 'brake_retries': self.brake_retries, 'vision_samples': self.vision_samples},
                    claim='proximity attempt only; no implemented grasp or delivery, no verified '
                          'contact and no score claim', debug=self.debug)