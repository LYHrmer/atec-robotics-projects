"""Bounded close-and-small-lift probe appended to one normal Task B first reach.

The frozen ``FirstReachPolicy`` owns every action and every failure up to its own
normal ``reach_lowering_hold_complete``; this wrapper never edits it, re-solves
its goal, resets one of its failures or relocates anything. Only that single
normal completion opens three bounded phases: slew the fingers to one fixed
physical preload target, latch one small paired joint lift from PUBLIC MEASURED
arm q, and hold for a short observation. That finger target is a finite position
request past the closed stop, so the unchanged position actuators ask for
holding force; no joint limit, stop, actuator stiffness/damping or helper
preload range is modified, and the actual finger motion and any actual force
remain whatever the original physics produces. Inputs after entry are the public
proprio vector only: no reward, score, contact sensor, object or simulator
state, and no image is read by the probe.

A stable non-empty jaw width and a joint-space lift are NOT a claim of contact,
grasp, transport, delivery or Task B completion. This candidate is an
experiment, not a proven grasp: the real check is an external audit of the same
object's recorded pose and mesh lower surface.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from task_b.arm_kinematics import (arm_joints_from_proprio, arm_targets_to_action,
                                  gripper_targets, pinch_position)
from task_b.control import ARM_TERM, LEG_TERM
from task_b.first_reach import FirstReachPolicy
# Shared static Piper chain limits, the same source arm_kinematics itself reads.
# No Task E table pose, world IK or score-based verification is used here.
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

NORMAL_FIRST_REACH_REASON = 'reach_lowering_hold_complete'
REQUIRED_LOWERING_M = .02
#: Mirrors the evaluator's fallback wheel-hold states so the first-reach prefix
#: keeps exactly the anchor behaviour the frozen policy had on its own.
BRAKE_STATES = ('BRAKE', 'REACH_READY', 'REACH', 'REACH_VISUAL_HOLD')
ARM6 = tuple('arm_joint' + str(index) for index in range(1, 7))
PROBE_PHASES = ('PROBE_CLOSE', 'PROBE_LIFT', 'PROBE_OBSERVE')

# ---------------------------------------------------------------------------
# Fixed constants of this bounded clamp-and-small-lift candidate. None of them
# is tuned on a reward, a score, an object pose or any privileged simulator
# observation.
# ---------------------------------------------------------------------------
CLOSE_PRELOAD_M = .025          # physical q7=-.025 m, q8=+.025 m
FINGER_SLEW_M_S = .05           # per finger, from the previous actual command
CLOSE_MAX_S, CLOSE_MIN_ACTIVE_S = 3., 1.4
WIDTH_WINDOW = 10               # consecutive 50 Hz measurements
WIDTH_RANGE_M = .002
WIDTH_MIN_M, WIDTH_MAX_M = .006, .065
EMPTY_HOLD_S = .2
PAIRED_DELTA_RAD = .20          # q2 -= .20, q3 += .20; all other angles unchanged
LIFT_MAX_S = 5.
LIFT_RATE_RAD_S = .10
LIFT_TETHER_RAD = .10
LIFT_Q2_FEEDBACK_GAIN = 1.0
LIFT_Q2_FEEDBACK_CAP_RAD = .08
LIFT_ERROR_RAD = .04
LIFT_QDOT_RAD_S = .05
LIFT_RISE_M = .015
LIFT_ARRIVAL_S = .3
LIFT_PROGRESS_WINDOW_S, LIFT_PROGRESS_RAD = 2., .01
OBSERVE_S = 2.
QUIET_LINEAR_M_S, QUIET_ANGULAR_RAD_S = .06, .12
PAUSE_QUIET_S, PAUSE_EPISODE_S, PAUSE_TOTAL_S = .2, 2., 2.
BASE_BUDGETS = {'displacement_m': .03, 'yaw_rad': .05, 'gravity_rad': .05}
TOTAL_PROBE_S = 10.

CANDIDATE_CONSTANTS = {
    'candidate': 'bounded clamp-and-small-lift candidate; an experiment, not a grasp success and not '
                 'tuned on score, object state or simulator truth',
    'finger_close_target': {'source': f'task_b.arm_kinematics.gripper_targets(0, preload_m={CLOSE_PRELOAD_M})',
                            'physical_q7_m': -CLOSE_PRELOAD_M, 'physical_q8_m': CLOSE_PRELOAD_M,
                            'slew_m_per_s': FINGER_SLEW_M_S,
                            'note': f'a finite position target {CLOSE_PRELOAD_M} m beyond the closed stop '
                                    'requests holding force from the unchanged position actuators; the '
                                    'original joint limits, stops, actuator stiffness/damping and the '
                                    'gripper_targets allowed preload range are untouched, so the actual '
                                    'motion and any actual force stay whatever the original physics gives',
                            'close_target_commanded_before_lift': True},
    'close': {'max_s': CLOSE_MAX_S, 'min_active_s': CLOSE_MIN_ACTIVE_S,
              'width_window_samples': WIDTH_WINDOW, 'width_range_m': WIDTH_RANGE_M,
              'width_median_window_m': [WIDTH_MIN_M, WIDTH_MAX_M], 'empty_hold_s': EMPTY_HOLD_S,
              'nominal_slew_s': f'about 1.2 s to slew the nominal .035 m half-command to the '
                                f'{-CLOSE_PRELOAD_M} m half-target at {FINGER_SLEW_M_S} m/s, plus the '
                                f'{WIDTH_WINDOW} consecutive width samples (.2 s at 50 Hz), which '
                                f'together equal the {CLOSE_MIN_ACTIVE_S} s minimum active close'},
    'lift': {'paired_delta_rad': PAIRED_DELTA_RAD, 'joints': ['arm_joint2', 'arm_joint3'],
             'signs': [-1., 1.], 'max_s': LIFT_MAX_S, 'rate_rad_per_s': LIFT_RATE_RAD_S,
             'tether_rad': LIFT_TETHER_RAD, 'arrival_error_rad': LIFT_ERROR_RAD,
             'arrival_qdot_rad_per_s': LIFT_QDOT_RAD_S, 'arrival_rise_m': LIFT_RISE_M,
             'arrival_window_s': LIFT_ARRIVAL_S, 'progress_window_s': LIFT_PROGRESS_WINDOW_S,
             'progress_rad': LIFT_PROGRESS_RAD,
             'goal_source': 'public measured arm q latched once after closure; no Cartesian IK, '
                            'no grasp-point shift and no silent clipping of the goal',
             'q2_position_feedback_gain': LIFT_Q2_FEEDBACK_GAIN,
             'q2_position_feedback_cap_rad': LIFT_Q2_FEEDBACK_CAP_RAD,
             'feedback_basis': 'bounded public measured q2 tracking correction; the fixed actual '
                               'joint goal, arrival thresholds and original actuator settings stay unchanged'},
    'observe': {'window_s': OBSERVE_S},
    'motion': {'linear_norm_m_s': QUIET_LINEAR_M_S, 'angular_norm_rad_s': QUIET_ANGULAR_RAD_S,
               'hard_tilt_rad': .25, 'instability_tilt_rad': .12, 'instability_rate_rad_s': .45,
               'pause_quiet_s': PAUSE_QUIET_S, 'pause_episode_s': PAUSE_EPISODE_S,
               'pause_total_s': PAUSE_TOTAL_S, 'base_budgets_from_entry': dict(BASE_BUDGETS)},
    'total_probe_s': TOTAL_PROBE_S,
    'required_first_reach': {'reach_only': False, 'lowering_m': REQUIRED_LOWERING_M,
                            'entry_reason': NORMAL_FIRST_REACH_REASON},
}

STOP_REASONS = {
    'grasp_probe_observation_complete': 'NORMAL end of the experiment after the bounded observation; '
                                        'it reports that the observation phase ran, and is not a '
                                        'grasp, lift, transport, delivery or score claim',
    'grasp_probe_empty_close': f'measured jaw width stayed below {WIDTH_MIN_M} m after the closed target '
                               'was commanded',
    'grasp_probe_close_not_confirmed': 'the closed target was not commanded, or no stable non-empty '
                                       f'width window formed, inside {CLOSE_MAX_S} s',
    'grasp_probe_lift_goal_outside_joint_limits': 'the latched paired goal left the static joint '
                                                  'limits, so nothing was commanded',
    'no_joint_lift_progress': f'a full {LIFT_PROGRESS_WINDOW_S} s active lift window improved the goal '
                              f'error by less than {LIFT_PROGRESS_RAD} rad while it exceeded '
                              f'{LIFT_ERROR_RAD} rad',
    'grasp_probe_lift_not_reached': f'the arrival conditions were not held for {LIFT_ARRIVAL_S} s inside '
                                    f'{LIFT_MAX_S} s',
    'grasp_probe_empty_during_lift': f'jaw width collapsed below {WIDTH_MIN_M} m during the lift',
    'grasp_probe_empty_during_observation': f'jaw width collapsed below {WIDTH_MIN_M} m during the '
                                            'observation',
    'grasp_probe_pause_timeout': f'one public body-motion pause reached {PAUSE_EPISODE_S} s',
    'grasp_probe_pause_budget_exceeded': f'cumulative pausing reached {PAUSE_TOTAL_S} s across the probe',
    'grasp_probe_base_displacement_budget_exceeded': 'integrated tangential drift since entry passed '
                                                     f'{BASE_BUDGETS["displacement_m"]} m',
    'grasp_probe_base_yaw_budget_exceeded': f'integrated yaw since entry passed {BASE_BUDGETS["yaw_rad"]} rad',
    'grasp_probe_gravity_changed': 'public gravity direction turned more than '
                                   f'{BASE_BUDGETS["gravity_rad"]} rad since entry',
    'grasp_probe_total_budget_exceeded': f'the {TOTAL_PROBE_S} s probe budget, pauses included, ran out',
    'invalid_proprio': 'public proprio was not 84 finite values',
    'tilt_limit_exceeded': 'unchanged hard public tilt limit of .25 rad',
    'reach_unstable_posture': 'unchanged reach instability limits: tilt > .12 rad or roll/pitch rate '
                              'norm > .45 rad/s',
}


class GraspProbePolicy:
    """One frozen first reach, then a bounded close, small lift and observation.

    Composition only: the inner :class:`FirstReachPolicy` is constructed with the
    caller's own parameters and called exactly once per step while it runs. Its
    actions and every one of its failure reasons pass through unchanged, and only
    ``reach_lowering_hold_complete`` is intercepted.
    """

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, ramp_calls=100, reach_only=False,
                 forward_cmd=.20, turn_cap=.20, standoff=.56, turn_gain=.4,
                 lowering_m=REQUIRED_LOWERING_M):
        if reach_only:
            raise ValueError('The grasp probe needs the real visual first reach; reach_only must be False')
        if not np.isfinite(lowering_m) or abs(float(lowering_m)-REQUIRED_LOWERING_M) > 1e-9:
            raise ValueError('The grasp probe is defined only for the audited .02 m lowered reach')
        self._inner = FirstReachPolicy(schema, observation_joint_names, defaults, dt=dt,
                                       settle_calls=settle_calls, ramp_calls=ramp_calls,
                                       reach_only=False, forward_cmd=forward_cmd, turn_cap=turn_cap,
                                       standoff=standoff, turn_gain=turn_gain, lowering_m=lowering_m)
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.dt = float(dt)
        self.settle_calls, self.ramp_calls = int(settle_calls), max(int(ramp_calls), 1)
        self.arm, self.leg = schema.term(ARM_TERM), schema.term(LEG_TERM)
        self.finger_target = gripper_targets(0., preload_m=CLOSE_PRELOAD_M)
        if not np.allclose(self.finger_target, [-CLOSE_PRELOAD_M, CLOSE_PRELOAD_M], atol=1e-12):
            raise ValueError('Unexpected physical finger preload target from gripper_targets')
        # Public observation order, never articulation order.
        self.arm_obs_ids = np.array([self.names.index(name) for name in ARM6])
        self.finger_obs_ids = np.array([self.names.index('arm_joint'+str(i)) for i in (7, 8)])
        # Use the original physical Piper limits, as FirstReach does. The
        # training soft margin is not a physical stop: the measured shoulder
        # can be at 3.14 rad and its valid 3.04 rad lift goal exceeds that margin.
        self.static_lower = JOINT_LOWER.copy()
        self.static_upper = JOINT_UPPER.copy()
        if np.any(self.static_lower > self.static_upper):
            raise ValueError('Empty static joint window for the arm')
        self.calls, self.alpha = 0, 0.
        self.state, self.state_reason = self._inner.state, 'first_reach_prefix_owned_by_the_frozen_policy'
        self.done_reason, self.debug = None, {}
        self.phase, self.phase_start, self.entry_call = None, None, None
        self.arm_command, self.held_arm_command = None, None
        self.close_active_calls, self.close_target_commanded = 0, False
        self.width, self.width_window = None, deque(maxlen=WIDTH_WINDOW)
        self.q7, self.q8, self.empty_start = None, None, None
        self.lift_start_q, self.lift_goal, self.lift_limit_violations = None, None, {}
        self.lift_active_calls, self.lift_arrival_start = 0, None
        self.lift_progress_anchor, self.lift_clamped_calls = None, 0
        self.lift_error, self.lift_qdot, self.pinch_rise = None, None, None
        self.q2_position_feedback_rad = 0.
        self.finger_qdot = None
        self.pinch_reference, self.predicted_rise = None, None
        self.observe_active_calls = 0
        self.pause_start, self.pause_calls = None, 0
        self.pause_quiet_calls, self.pause_episodes, self.pause_cause = 0, 0, None
        self.base_displacement, self.base_yaw = np.zeros(3), 0.
        self.up_anchor, self.gravity_change = None, 0.
        self.motion = None
        self._hold = False

    # -- public attributes the evaluator drives or reads ---------------------

    @property
    def pause_for_stance(self):
        """Public stance-transition bit; the frozen policy keeps owning it."""
        return self._inner.pause_for_stance

    @pause_for_stance.setter
    def pause_for_stance(self, value):
        self._inner.pause_for_stance = bool(value)

    @property
    def wheel_hold_requested(self):
        """Mirror the frozen prefix, then hold the SAME anchor for the probe.

        During the first reach this is exactly the evaluator's fallback state
        test, so nothing about the prefix changes. From probe entry it stays
        true for every remaining call, so the wheel anchor is never released or
        re-anchored across CLOSE, LIFT and OBSERVE.
        """
        return self._hold

    # -- action assembly ----------------------------------------------------

    def _action(self):
        """Probe arm command, the child's achieved lowered legs, zero wheels.

        The leg slice is read straight off the child's own latched alpha and
        fixed delta, so the achieved lowered reference stays exactly where the
        first reach left it, and the base is never raised.
        """
        out = np.zeros(self.schema.total_dim, dtype=np.float32)
        out[self.arm.start:self.arm.stop] = arm_targets_to_action(
            self.arm_command, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        out[self.leg.start:self.leg.stop] = self._inner.lowering_alpha*self._inner.lower_delta/self.leg.scale
        return out

    def _stop(self, reason):
        """Request a normal evaluator stop while holding the last command.

        Every reason here is an experimental stopping reason, never an official
        termination and never a score, grasp or delivery claim.
        """
        self.done_reason = self.state_reason = reason
        self.state = 'STOPPED'
        self._hold = True
        action = self._action()
        self._record()
        return action

    # -- first-reach prefix -------------------------------------------------

    def _prefix(self, proprio, images):
        """One and only one inner call this step; its action is returned as is."""
        action = self._inner.act(proprio, images)
        self.state, self.state_reason = self._inner.state, self._inner.state_reason
        reason = self._inner.done_reason
        if reason is not None:
            self.arm_command = self._inner.arm_command.copy()
            if reason != NORMAL_FIRST_REACH_REASON:
                self.done_reason = reason  # propagated unchanged, never reset
            else:
                # ONLY a normal completion of the frozen reach enters the probe.
                # No score, image, object state or GT target takes part in this.
                self.entry_call = self.phase_start = self.calls
                self.phase = self.state = 'PROBE_CLOSE'
                self.state_reason = 'entering_the_bounded_close_and_small_lift_probe'
                self.held_arm_command = self._inner.arm_command[:6].copy()
                gravity = np.asarray(proprio, dtype=float)[9:12]
                self.up_anchor = -gravity / np.linalg.norm(gravity)
        self._hold = True if self.phase is not None else self._inner.state in BRAKE_STATES
        self._record()
        return action

    # -- bounded probe ------------------------------------------------------

    def _budgets(self, obs, up):
        """Integrate public twist since entry against the unchanged budgets.

        Small-angle integration of the public linear/angular velocity in the
        current body frame with the vertical component removed, exactly as the
        first reach does. It is a local drift estimate, not world odometry, and
        it keeps running during pauses.
        """
        velocity = obs[:3] - np.dot(obs[:3], up)*up
        self.base_displacement = self.base_displacement + self.dt*velocity
        self.base_yaw += self.dt*float(np.dot(obs[3:6], up))
        if self.up_anchor is None:
            self.up_anchor = up.copy()
        self.gravity_change = float(np.arccos(np.clip(np.dot(up, self.up_anchor), -1., 1.)))
        if float(np.linalg.norm(self.base_displacement)) > BASE_BUDGETS['displacement_m']:
            return 'grasp_probe_base_displacement_budget_exceeded'
        if abs(self.base_yaw) > BASE_BUDGETS['yaw_rad']:
            return 'grasp_probe_base_yaw_budget_exceeded'
        if self.gravity_change > BASE_BUDGETS['gravity_rad']:
            return 'grasp_probe_gravity_changed'
        return None

    def _empty(self, width):
        """Empty-jaw watchdog, armed once the closed target has been commanded."""
        if not self.close_target_commanded or width >= WIDTH_MIN_M:
            self.empty_start = None
            return None
        if self.empty_start is None:
            self.empty_start = self.calls
        if (self.calls-self.empty_start+1)*self.dt < EMPTY_HOLD_S:
            return None
        return {'PROBE_CLOSE': 'grasp_probe_empty_close',
                'PROBE_LIFT': 'grasp_probe_empty_during_lift',
                'PROBE_OBSERVE': 'grasp_probe_empty_during_observation'}[self.phase]

    def _pause(self, forced):
        """Finite pause on the unchanged public motion thresholds.

        A violating public twist freezes the finger and arm commands while the
        lowered reference and the wheel anchor stay exactly as they are. The
        width, arrival and observation windows restart afterwards; phase
        deadlines, the body budgets and the total probe budget never do.
        Returns a stop reason, 'FROZEN', or None when the caller may continue.
        """
        violation = bool(forced or self.motion['linear_norm'] >= QUIET_LINEAR_M_S
                         or self.motion['angular_norm'] >= QUIET_ANGULAR_RAD_S)
        if self.pause_start is None:
            if not violation:
                return None
            self.pause_start, self.pause_quiet_calls = self.calls, 0
            self.pause_episodes += 1
            self.pause_cause = ('public_stance_transition_request' if forced else
                                'public_linear_and_angular_norm'
                                if self.motion['linear_norm'] >= QUIET_LINEAR_M_S
                                and self.motion['angular_norm'] >= QUIET_ANGULAR_RAD_S
                                else 'public_linear_norm'
                                if self.motion['linear_norm'] >= QUIET_LINEAR_M_S
                                else 'public_angular_norm')
        self.pause_calls += 1
        self.pause_quiet_calls = 0 if violation else self.pause_quiet_calls+1
        self.width_window.clear()
        self.lift_arrival_start = None
        self.observe_active_calls = 0
        if (self.calls-self.pause_start+1)*self.dt >= PAUSE_EPISODE_S:
            return 'grasp_probe_pause_timeout'
        if self.pause_calls*self.dt >= PAUSE_TOTAL_S:
            return 'grasp_probe_pause_budget_exceeded'
        if self.pause_quiet_calls*self.dt >= PAUSE_QUIET_S:
            self.pause_start, self.pause_quiet_calls = None, 0
            return None
        return 'FROZEN'

    def _probe(self, proprio):
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            return self._stop('invalid_proprio')
        try:
            q = arm_joints_from_proprio(obs, self.names, self.defaults)
        except (ValueError, KeyError):
            return self._stop('invalid_proprio')
        gravity = obs[9:12]
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < .5 or -gravity[2]/norm < np.cos(.25):
            return self._stop('tilt_limit_exceeded')
        up = -gravity/norm
        tilt = float(np.arccos(np.clip(-gravity[2]/norm, -1., 1.)))
        if tilt > .12 or float(np.linalg.norm(obs[3:5])) > .45:
            return self._stop('reach_unstable_posture')
        linear, angular = obs[:3], obs[3:6]
        self.motion = {'linear_norm': float(np.linalg.norm(linear)),
                       'angular_norm': float(np.linalg.norm(angular)),
                       'tangent_speed_m_s': float(np.linalg.norm(linear-np.dot(linear, up)*up)),
                       'tilt_rad': tilt}
        self.width = float(q[6]-q[7])
        self.q7, self.q8 = float(q[6]), float(q[7])
        arm_qdot = np.asarray(obs[36+self.arm_obs_ids], dtype=float)
        self.finger_qdot = np.asarray(obs[36+self.finger_obs_ids], dtype=float)
        budget_exit = self._budgets(obs, up)
        if budget_exit is not None:
            return self._stop(budget_exit)
        empty = self._empty(self.width)
        if empty is not None:
            return self._stop(empty)
        # Phase deadlines and the total budget run through pauses.
        elapsed = (self.calls-self.phase_start)*self.dt
        if self.phase == 'PROBE_CLOSE' and elapsed >= CLOSE_MAX_S:
            return self._stop('grasp_probe_close_not_confirmed')
        if self.phase == 'PROBE_LIFT' and elapsed >= LIFT_MAX_S:
            return self._stop('grasp_probe_lift_not_reached')
        if (self.calls-self.entry_call)*self.dt >= TOTAL_PROBE_S:
            return self._stop('grasp_probe_total_budget_exceeded')
        paused = self._pause(bool(self._inner.pause_for_stance))
        if paused is not None:
            if paused != 'FROZEN':
                return self._stop(paused)
            self.state_reason = 'bounded_pause_freezing_finger_and_arm_commands_on_public_body_motion'
            action = self._action()
            self._record()
            return action
        if self.phase == 'PROBE_CLOSE':
            action = self._close(q)
        elif self.phase == 'PROBE_LIFT':
            action = self._lift(q, arm_qdot, up)
        else:
            action = self._observe(q, arm_qdot, up)
        if self.done_reason is None:  # a phase helper may already have stopped
            self._record()
        return action

    def _close(self, q):
        """Hold the six achieved joint commands and gradually close the fingers.

        The lift needs all three: the closed physical target actually commanded,
        at least ``CLOSE_MIN_ACTIVE_S`` of active closing, and the stable
        non-empty measured width window. Nominally the fingers need about 1.2 s
        to slew from .035 m to the -.025 m half-target at .05 m/s and the window
        needs .2 s more, so the nominal 1.4 s matches the minimum active close
        exactly. That window only permits the bounded lift below. It is NOT a
        claim of contact or grasp: no contact sensor, score or object state is
        read, and the fingers are never re-squeezed or retried.
        """
        self.state, self.state_reason = 'PROBE_CLOSE', 'slewing_fingers_to_the_fixed_physical_preload_target'
        self.close_active_calls += 1
        self.arm_command[:6] = self.held_arm_command
        self.arm_command[6:] += np.clip(self.finger_target-self.arm_command[6:],
                                        -FINGER_SLEW_M_S*self.dt, FINGER_SLEW_M_S*self.dt)
        self.close_target_commanded = bool(
            self.close_target_commanded
            or np.max(np.abs(self.arm_command[6:]-self.finger_target)) <= 1e-12)
        self.width_window.append(self.width)
        window = list(self.width_window)
        stable = bool(len(window) == WIDTH_WINDOW and float(np.ptp(window)) < WIDTH_RANGE_M
                      and WIDTH_MIN_M <= float(np.median(window)) <= WIDTH_MAX_M)
        if (self.close_active_calls*self.dt >= CLOSE_MIN_ACTIVE_S and self.close_target_commanded
                and stable):
            return self._enter_lift(q)
        return self._action()

    def _enter_lift(self, q):
        """Latch the measured pose once and build the fixed paired joint goal."""
        self.lift_start_q = q[:6].copy()
        goal = self.lift_start_q.copy()
        goal[1] -= PAIRED_DELTA_RAD
        goal[2] += PAIRED_DELTA_RAD
        violations = {name: [float(goal[index]), float(self.static_lower[index]),
                             float(self.static_upper[index])]
                      for index, name in enumerate(ARM6)
                      if not (self.static_lower[index]-1e-9 <= goal[index] <= self.static_upper[index]+1e-9)}
        self.lift_limit_violations = violations
        if violations:
            # An unreachable goal stops the probe; it is never silently clipped.
            return self._stop('grasp_probe_lift_goal_outside_joint_limits')
        self.lift_goal = goal
        self.pinch_reference = pinch_position(q)
        # Static body-frame FK prediction of the paired motion, for diagnostics
        # only. It is not measured motion and is not the probe's arrival test.
        self.predicted_rise = float(np.linalg.norm(pinch_position(goal)-pinch_position(self.lift_start_q)))
        self.phase = self.state = 'PROBE_LIFT'
        self.phase_start = self.calls
        self.state_reason = 'latched_measured_paired_joint_lift_goal_after_stable_width_window'
        self.lift_active_calls, self.lift_arrival_start, self.lift_progress_anchor = 0, None, None
        return self._action()

    def _track_lift_goal(self, q):
        """Move the six commands toward the latched goal under both bounds."""
        # Satisfy the three bounds simultaneously. Sequential clipping can
        # jump farther than the rate limit after a sudden measured q change.
        rate = LIFT_RATE_RAD_S*self.dt
        low = np.maximum(np.maximum(self.arm_command[:6]-rate, q[:6]-LIFT_TETHER_RAD),
                         self.static_lower)
        high = np.minimum(np.minimum(self.arm_command[:6]+rate, q[:6]+LIFT_TETHER_RAD),
                          self.static_upper)
        if np.any(low > high):
            return self._stop('grasp_probe_rate_tether_intersection_empty')
        command_goal = self.lift_goal.copy()
        q2_correction = float(np.clip(LIFT_Q2_FEEDBACK_GAIN*(self.lift_goal[1]-q[1]),
                                      -LIFT_Q2_FEEDBACK_CAP_RAD, LIFT_Q2_FEEDBACK_CAP_RAD))
        command_goal[1] += q2_correction
        self.q2_position_feedback_rad = q2_correction
        self.arm_command[:6] = np.clip(command_goal, low, high)
        self.arm_command[6:] = self.finger_target  # the closed target is retained
        return None

    def _lift_measurements(self, q, arm_qdot, up):
        self.lift_error = float(np.max(np.abs(q[:6]-self.lift_goal)))
        self.lift_qdot = float(np.max(np.abs(arm_qdot)))
        # Body-frame FK of the measured joints, projected on the public up
        # direction. It measures the arm relative to the base, not world height.
        self.pinch_rise = float(np.dot(pinch_position(q)-self.pinch_reference, up))

    def _lift(self, q, arm_qdot, up):
        self.state, self.state_reason = 'PROBE_LIFT', 'driving_the_latched_paired_joint_goal_under_rate_and_tether'
        self.lift_active_calls += 1
        stopped = self._track_lift_goal(q)
        if stopped is not None:
            return stopped
        self._lift_measurements(q, arm_qdot, up)
        if (self.lift_error < LIFT_ERROR_RAD and self.lift_qdot < LIFT_QDOT_RAD_S
                and self.pinch_rise >= LIFT_RISE_M):
            if self.lift_arrival_start is None:
                self.lift_arrival_start = self.calls
            if (self.calls-self.lift_arrival_start+1)*self.dt >= LIFT_ARRIVAL_S:
                self.phase = self.state = 'PROBE_OBSERVE'
                self.phase_start, self.observe_active_calls = self.calls, 0
                self.state_reason = 'holding_the_measured_lift_goal_for_the_bounded_observation'
                return self._action()
        else:
            self.lift_arrival_start = None
        if self.lift_progress_anchor is None:
            self.lift_progress_anchor = (self.lift_active_calls, self.lift_error)
        anchor_calls, anchor_error = self.lift_progress_anchor
        if (self.lift_active_calls-anchor_calls)*self.dt >= LIFT_PROGRESS_WINDOW_S:
            if self.lift_error > LIFT_ERROR_RAD and anchor_error-self.lift_error < LIFT_PROGRESS_RAD:
                return self._stop('no_joint_lift_progress')
            self.lift_progress_anchor = (self.lift_active_calls, self.lift_error)
        return self._action()

    def _observe(self, q, arm_qdot, up):
        """Hold the same goal, closed target, leg reference and wheel anchor.

        The experiment ends here: no transport, opening, regrasp or further
        lowering follows, and no image, object motion or score is inspected.
        """
        self.state = 'PROBE_OBSERVE'
        self.state_reason = 'bounded_observation_of_the_held_measured_lift_goal'
        self.observe_active_calls += 1
        stopped = self._track_lift_goal(q)
        if stopped is not None:
            return stopped
        self._lift_measurements(q, arm_qdot, up)
        if self.observe_active_calls*self.dt >= OBSERVE_S:
            return self._stop('grasp_probe_observation_complete')
        return self._action()

    # -- diagnostics --------------------------------------------------------

    def _phase_remaining_s(self):
        if self.phase is None:
            return None
        if self.phase == 'PROBE_CLOSE':
            return max(0., CLOSE_MAX_S-(self.calls-self.phase_start)*self.dt)
        if self.phase == 'PROBE_LIFT':
            return max(0., LIFT_MAX_S-(self.calls-self.phase_start)*self.dt)
        return max(0., OBSERVE_S-self.observe_active_calls*self.dt)

    def _record(self):
        """Merge the child's diagnostics, then add the probe's own."""
        debug = dict(self._inner.debug)
        debug.update(
            state=self.state, reason=self.state_reason, done_reason=self.done_reason,
            first_reach_state=self._inner.state, first_reach_reason=self._inner.state_reason,
            first_reach_done_reason=self._inner.done_reason,
            first_reach_debug_timing=('live' if self.phase is None
                                      else 'frozen_snapshot_of_the_normal_completion_call'),
            probe_phase=self.phase, probe_entry_call=self.entry_call, probe_call=self.calls,
            probe_q2_position_feedback_rad=self.q2_position_feedback_rad,
            probe_phase_elapsed_s=(None if self.phase is None
                                   else (self.calls-self.phase_start)*self.dt),
            probe_phase_remaining_s=self._phase_remaining_s(),
            probe_elapsed_s=(None if self.entry_call is None
                             else (self.calls-self.entry_call)*self.dt),
            probe_remaining_s=(None if self.entry_call is None else
                               max(0., TOTAL_PROBE_S-(self.calls-self.entry_call)*self.dt)),
            probe_paused=self.pause_start is not None, probe_pause_cause=self.pause_cause,
            probe_pause_episodes=self.pause_episodes,
            probe_pause_episode_s=(0. if self.pause_start is None
                                   else (self.calls-self.pause_start+1)*self.dt),
            probe_pause_total_s=self.pause_calls*self.dt,
            probe_pause_quiet_s=self.pause_quiet_calls*self.dt,
            probe_held_arm_command_rad=(None if self.held_arm_command is None
                                        else self.held_arm_command.tolist()),
            probe_arm_command=(None if self.arm_command is None else self.arm_command.tolist()),
            probe_finger_target_m=self.finger_target.tolist(),
            probe_finger_target_commanded=bool(self.close_target_commanded),
            probe_close_active_s=self.close_active_calls*self.dt,
            probe_measured_q7_m=self.q7, probe_measured_q8_m=self.q8,
            probe_measured_width_m=self.width,
            probe_measured_finger_qdot_m_s=(None if self.finger_qdot is None
                                            else self.finger_qdot.tolist()),
            probe_width_window_m=[float(value) for value in self.width_window],
            probe_width_window_range_m=(float(np.ptp(list(self.width_window)))
                                        if len(self.width_window) == WIDTH_WINDOW else None),
            probe_width_window_median_m=(float(np.median(list(self.width_window)))
                                         if len(self.width_window) == WIDTH_WINDOW else None),
            probe_empty_width_s=(0. if self.empty_start is None
                                 else (self.calls-self.empty_start+1)*self.dt),
            probe_lift_latched_q_rad=(None if self.lift_start_q is None else self.lift_start_q.tolist()),
            probe_lift_goal_rad=(None if self.lift_goal is None else self.lift_goal.tolist()),
            probe_lift_paired_delta_rad={'arm_joint2': -PAIRED_DELTA_RAD, 'arm_joint3': PAIRED_DELTA_RAD},
            probe_lift_static_limit_lower_rad=self.static_lower.tolist(),
            probe_lift_static_limit_upper_rad=self.static_upper.tolist(),
            probe_lift_limit_violations=self.lift_limit_violations,
            probe_lift_command_clamped_calls=self.lift_clamped_calls,
            probe_lift_active_s=self.lift_active_calls*self.dt,
            probe_lift_arrival_s=(0. if self.lift_arrival_start is None
                                  else (self.calls-self.lift_arrival_start+1)*self.dt),
            probe_lift_max_q_error_rad=self.lift_error, probe_lift_max_qdot_rad_s=self.lift_qdot,
            probe_pinch_rise_m=self.pinch_rise,
            probe_pinch_reference_body=(None if self.pinch_reference is None
                                        else self.pinch_reference.tolist()),
            probe_predicted_pinch_travel_m=self.predicted_rise,
            probe_predicted_travel_claim='static body-frame FK of the latched paired goal; a prediction '
                                         'for the recorded pose, never a claim of actual motion',
            probe_observe_active_s=self.observe_active_calls*self.dt,
            probe_public_motion=self.motion,
            probe_base_displacement_m=float(np.linalg.norm(self.base_displacement)),
            probe_base_yaw_rad=abs(float(self.base_yaw)),
            probe_base_gravity_change_rad=float(self.gravity_change),
            probe_base_budgets=dict(BASE_BUDGETS),
            probe_base_integration_claim='approximate small-angle integration of public twist since '
                                         'probe entry, no vertical component; not world odometry',
            probe_lowering_alpha=self._inner.lowering_alpha,
            probe_lowered_reference_basis='the child\'s own latched lowering alpha and fixed leg delta '
                                          'are reused verbatim; the probe never writes either, so the '
                                          'achieved lowered reference cannot move and the base is not raised',
            wheel_hold_requested=bool(self._hold),
            probe_candidate_constants=CANDIDATE_CONSTANTS,
            probe_claim='bounded close, small joint lift and observation from public proprio only; a '
                        'stable non-empty jaw width and a joint-space lift are not contact, grasp, '
                        'transport, delivery or score evidence')
        self.debug = debug

    def act(self, proprio, images):
        self.calls += 1
        self.alpha = float(np.clip((self.calls-self.settle_calls)/self.ramp_calls, 0., 1.))
        if self.done_reason is not None:
            self._record()
            return self._action()
        if self.phase is None:
            return self._prefix(proprio, images)
        return self._probe(proprio)

    def describe(self):
        return dict(mode='grasp_probe',
                    inputs='the frozen first reach owns all vision; after entry the probe reads only '
                           'public proprio. No reward, score, contact sensor, object or world state, '
                           'and no image, enters this probe',
                    entry_contract='actions and every failure reason are the frozen FirstReachPolicy '
                                   'ones until it returns reach_lowering_hold_complete by itself; only '
                                   'that normal completion opens the probe, any other reason is '
                                   'propagated unchanged, and one step never calls the inner policy '
                                   'twice. Nothing here triggers on score, resets a failure, relocates '
                                   'an object or re-solves a ground-truth target',
                    candidate_constants=CANDIDATE_CONSTANTS,
                    phases={'PROBE_CLOSE': 'hold the six achieved joint commands, slew both fingers to '
                                           f'the fixed physical preload target (q7={-CLOSE_PRELOAD_M} m, '
                                           f'q8={CLOSE_PRELOAD_M} m) at <={FINGER_SLEW_M_S} m/s, and '
                                           f'require the closed target to have actually been commanded, '
                                           f'>={CLOSE_MIN_ACTIVE_S} s of active closing and '
                                           f'{WIDTH_WINDOW} consecutive width samples with range '
                                           f'<{WIDTH_RANGE_M} m and median in '
                                           f'[{WIDTH_MIN_M},{WIDTH_MAX_M}] m. A finite target beyond the '
                                           'closed stop requests holding force from the unchanged '
                                           'position actuators; limits, stops and the helper preload '
                                           'range are not modified. This permits the bounded lift only; '
                                           'it is not a claim of contact or grasp, and there is no '
                                           're-squeeze or retry',
                            'PROBE_LIFT': 'latch measured arm q once, apply the fixed paired goal '
                                          f'(q2-{PAIRED_DELTA_RAD} rad, q3+{PAIRED_DELTA_RAD} rad, other '
                                          'angles unchanged) after a static limit check, and drive it '
                                          f'under a {LIFT_RATE_RAD_S} rad/s per-axis rate and a '
                                          f'+/-{LIFT_TETHER_RAD} rad measured tether while the closed '
                                          'finger target is retained. No Cartesian IK or grasp-point '
                                          'shift takes part',
                            'PROBE_OBSERVE': 'hold the same measured goal, closed target, lowered leg '
                                             'reference, wheel anchor and motion bounds for '
                                             f'{OBSERVE_S} s, then end. The experiment stops there: no '
                                             'transport, opening, regrasp or further lowering'},
                    stop_reasons=STOP_REASONS,
                    probe_state={'phase': self.phase, 'phase_order': list(PROBE_PHASES),
                                 'entry_call': self.entry_call,
                                 'calls': self.calls, 'state': self.state,
                                 'done_reason': self.done_reason,
                                 'phase_elapsed_s': (None if self.phase is None
                                                     else (self.calls-self.phase_start)*self.dt),
                                 'phase_remaining_s': self._phase_remaining_s(),
                                 'probe_elapsed_s': (None if self.entry_call is None
                                                     else (self.calls-self.entry_call)*self.dt),
                                 'pause_episodes': self.pause_episodes,
                                 'pause_total_s': self.pause_calls*self.dt,
                                 'held_arm_command_rad': (None if self.held_arm_command is None
                                                          else self.held_arm_command.tolist()),
                                 'finger_target_m': self.finger_target.tolist(),
                                 'measured_width_m': self.width,
                                 'latched_lift_q_rad': (None if self.lift_start_q is None
                                                        else self.lift_start_q.tolist()),
                                 'lift_goal_rad': (None if self.lift_goal is None
                                                   else self.lift_goal.tolist()),
                                 'lift_max_q_error_rad': self.lift_error,
                                 'lift_max_qdot_rad_s': self.lift_qdot,
                                 'pinch_rise_m': self.pinch_rise,
                                 'predicted_pinch_travel_m': self.predicted_rise,
                                 'wheel_hold_requested': bool(self._hold),
                                 'lowering_alpha': self._inner.lowering_alpha},
                    wheel_and_leg_contract='the same wheel anchor request and the child\'s own achieved '
                                           'lowered leg reference are kept continuously through CLOSE, '
                                           'LIFT and OBSERVE; the base is never raised, the wheels are '
                                           'never released or re-anchored and no extra locomotion is '
                                           'commanded',
                    verification='real validation is external: a fresh original-environment run plus an '
                                 'independent audit of the same object\'s recorded pose and mesh lower '
                                 'surface before and after the lift. This policy cannot and does not '
                                 'certify a grasp',
                    claim='proximity, bounded finger closure and a small joint-space lift attempt only; '
                          'no verified contact, no grasp success, no transport or delivery and no score '
                          'claim. A normal end reports that the observation phase ran, nothing more',
                    first_reach=self._inner.describe(),
                    debug=self.debug)
