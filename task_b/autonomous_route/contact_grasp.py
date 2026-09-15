"""Open-jaw wrist rotation and one extra leg reference before the frozen force75 probe.

This is a subclass of the audited :class:`~task_b.grasp_probe.GraspProbePolicy`.
Both that module and the frozen ``FirstReachPolicy`` are read-only here: nothing
is monkeypatched, no module constant is mutated, and the inherited
``act``/call count, safety helpers and the ENTIRE original PROBE_CLOSE /
PROBE_LIFT / PROBE_OBSERVE sequence are reused rather than reimplemented.

Why this candidate exists. In the recorded force75 runs two actual fingers
touched the sloping shoulder of the mustard bottle, the gripper rose and the
object stayed on the ground and slipped out of the jaws. The recorded shapes
carry mu=1 and .5 kg. Those are OFFLINE observations of past runs and static
asset data used to pick this candidate's fixed numbers; they are never read at
runtime. After the prefix this policy still sees only the public proprio vector
that the frozen first reach already receives, plus the RGB-D the first reach
itself owns: no score, reward, contact force, net force, object pose, ground
truth, map or seed enters any decision.

The candidate is: the frozen ``FirstReachPolicy(.02)`` prefix, unchanged; then
verify the jaw is fully open; then rotate ONLY arm_joint6 by a fixed +pi/6
relative to the PUBLIC MEASURED q6; then add one more nominal .02 m leg
reference on top of the .02 m the child already completed; then hand the new,
rotated and deeper pose to the unchanged force75 close/lift/observe probe.

Total NOMINAL reference lowering is .04 m: .02 m owned by the frozen child plus
one separate .02 m copy of the child's own leg delta added here. That is a
joint-space reference sum, NOT a measurement or a claim that the actual body
descended .04 m or descended at all; only public leg joint response is checked.
The original ``FirstReachPolicy`` constructor still accepts at most .03 m and is
not touched: the extra increment is applied by this module to the leg action
slice, never by re-parameterising the child.

Nothing here is a grasp, contact, lift, transport, delivery or score claim. An
open jaw width is not proof of collision clearance, a rotated wrist is not proof
of a better approach angle, and a completed reference increment is not proof of
body height. This remains an experimental contact candidate whose only real
verification is an external audit of the recorded object pose and mesh.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from task_b.arm_kinematics import arm_joints_from_proprio
from task_b.control import LEG_JOINT_PATTERN
from task_b.grasp_probe import (ARM6, BASE_BUDGETS, CLOSE_PRELOAD_M, FINGER_SLEW_M_S,
                                GraspProbePolicy, NORMAL_FIRST_REACH_REASON, PAUSE_EPISODE_S,
                                PAUSE_QUIET_S, PAUSE_TOTAL_S, PROBE_PHASES, QUIET_ANGULAR_RAD_S,
                                QUIET_LINEAR_M_S, TOTAL_PROBE_S)

#: The only accepted lowering amplitude for the frozen prefix of this candidate.
CONTACT_LOWERING_M = .02
#: Fully open physical finger pair (q7, q8) in metres: the same pair the frozen
#: first reach already commands, so this phase verifies the current public
#: observation instead of opening anything new.
OPEN_TARGET_M = (.035, -.035)
OPEN_MAX_S, OPEN_HOLD_S = 1., .2
OPEN_WIDTH_M, OPEN_FINGER_M = .066, .032
#: Open-width watchdog, armed once the wrist goal is latched.
WIDTH_KEEP_M, WIDTH_LOST_S = .064, .2

WRIST_DELTA_RAD = np.pi/6       # fixed, relative to PUBLIC measured q6, latched once
ROTATE_MAX_S = 7.
WRIST_RATE_RAD_S = .10          # command slew, arm_joint6 only
WRIST_TETHER_RAD = .10          # |command - public measured q6|
WRIST_ERROR_RAD = .04
WRIST_PROGRESS_WINDOW_S, WRIST_PROGRESS_RAD = 2., .02

HOLD_ERROR_RAD = .04            # the five non-q6 axes against their held commands
QUIET_ARM_QDOT_RAD_S = .12
QUIET_TANGENT_M_S, QUIET_TILT_RAD = .01, .10
#: One COMPLETE .5 s window of public arm/leg positions: ceil(.5/dt)+1 samples,
#: 26 at the official 50 Hz, both endpoints included. Any ineligible sample, a
#: pause or a phase change clears it; it is never carried between phases.
QUIET_WINDOW_S, QUIET_ARM_SPAN_RAD, QUIET_LEG_SPAN_RAD = .5, .002, .02

LOWER_MAX_S = 5.
LOWER_ALPHA_DIVISOR = 3.        # at most dt/3 per active tick, as the child does
LOWER_INCREMENT_ERROR_RAD, LOWER_INCREMENT_FAULT_S = .06, .2
LOWER_BETA_MIN, LOWER_BETA_AFTER_HALF_S = .15, 1.
#: Original physical leg-chain limits, per link type.
LEG_LIMITS_RAD = {'hip': (-.87, .87), 'thigh': (-.94, 4.69), 'calf': (-2.82, -.43)}

#: One global deadline from the first normal prefix completion through the final
#: inherited observation, pauses included. Phase deadlines stay in force too, and
#: nothing retries, re-anchors, restarts or extends any of them.
CONTACT_TOTAL_S = 25.

CONTACT_PHASES = ('CONTACT_OPEN', 'CONTACT_ROTATE', 'CONTACT_LOWER')
PHASE_MAX_S = {'CONTACT_OPEN': OPEN_MAX_S, 'CONTACT_ROTATE': ROTATE_MAX_S,
               'CONTACT_LOWER': LOWER_MAX_S}
PHASE_DEADLINE_REASON = {'CONTACT_OPEN': 'contact_open_not_reached',
                         'CONTACT_ROTATE': 'contact_wrist_not_reached',
                         'CONTACT_LOWER': 'contact_lower_not_reached'}

CONTACT_STOP_REASONS = {
    'contact_open_not_reached': f'the public jaw width did not hold >= {OPEN_WIDTH_M} m with '
                                f'q7 >= {OPEN_FINGER_M} m and q8 <= {-OPEN_FINGER_M} m for a complete '
                                f'{OPEN_HOLD_S} s inside {OPEN_MAX_S} s. It is an observation of the '
                                'already open jaw, never a collision-clearance proof',
    'contact_wrist_goal_outside_joint_limits': f'public measured q6 + {WRIST_DELTA_RAD:.6f} rad left the '
                                               'original static arm_joint6 limits, so nothing was '
                                               'commanded and the goal was NOT clipped',
    'contact_wrist_command_bounds_infeasible': f'the {WRIST_RATE_RAD_S} rad/s slew on the previous q6 '
                                               f'command, the {WRIST_TETHER_RAD} rad measured tether and '
                                               'the original hard limits had no common value, so the '
                                               'previous valid command is held instead of clipped '
                                               'sequentially or jumped',
    'contact_wrist_no_progress': f'a full {WRIST_PROGRESS_WINDOW_S} s active rotation window improved the '
                                 f'fixed-goal error by less than {WRIST_PROGRESS_RAD} rad while it stayed '
                                 f'above {WRIST_ERROR_RAD} rad',
    'contact_wrist_not_reached': f'the fresh arrival evidence (goal error, public arm speed, held-axis '
                                 f'errors, body quiet and one complete {QUIET_WINDOW_S} s arm/leg position '
                                 f'window) did not form inside {ROTATE_MAX_S} s',
    'contact_open_width_lost': f'public jaw width stayed below {WIDTH_KEEP_M} m for {WIDTH_LOST_S} s '
                               'during the rotation or the extra lowering; the candidate stops there and '
                               'never closes or lifts',
    'contact_lower_reference_outside_joint_limits': 'the latched public start leg q, or that q plus the '
                                                    'extra fixed delta, left the original leg hard '
                                                    'limits, so no extra reference was commanded',
    'contact_lower_increment_error': f'|public leg increment - alpha*extra delta| stayed above '
                                     f'{LOWER_INCREMENT_ERROR_RAD} rad for {LOWER_INCREMENT_FAULT_S} s; a '
                                     'joint-response fault, not a body-height measurement',
    'contact_lower_no_joint_progress': f'projected leg-joint progress beta stayed below {LOWER_BETA_MIN} '
                                       f'for {LOWER_BETA_AFTER_HALF_S} s after alpha reached .5; a '
                                       'joint-response fault, not a body-height measurement',
    'contact_lower_not_reached': f'fresh admission, the extra reference increment and the final fresh '
                                 f'{QUIET_WINDOW_S} s quiet window did not complete inside {LOWER_MAX_S} s',
    'contact_total_budget_exceeded': f'the single {CONTACT_TOTAL_S} s budget from the normal prefix '
                                     'completion through the final inherited observation, pauses '
                                     'included, ran out',
}

CONTACT_CONSTANTS = {
    'candidate': 'fully open jaw, one fixed relative wrist rotation and one extra fixed leg reference '
                 'inserted between the frozen first reach and the UNCHANGED inherited force75 probe',
    'offline_evidence_only': {'observation': 'in the recorded force75 runs two actual fingers contacted '
                                             'the sloping shoulder, the gripper rose and the object '
                                             'stayed grounded and slipped out',
                              'recorded_shape_properties': {'mu': 1., 'mass_kg': .5},
                              'note': 'these are offline analyses of past runs and static asset data used '
                                      'to choose fixed numbers; they are NOT policy inputs. At runtime '
                                      'this policy reads only the public proprio and the RGB-D the frozen '
                                      'first reach owns: no score, reward, contact or net force, object '
                                      'pose, ground truth, map or seed'},
    'required_first_reach': {'lowering_m': CONTACT_LOWERING_M, 'reach_only': False,
                             'entry_reason': NORMAL_FIRST_REACH_REASON,
                             'note': f'{CONTACT_LOWERING_M} m is the default AND the only accepted value; '
                                     'it is forwarded to the untouched FirstReachPolicy constructor, '
                                     'which still accepts at most .03 m and is not modified'},
    'open': {'max_s': OPEN_MAX_S, 'hold_s': OPEN_HOLD_S, 'target_m': list(OPEN_TARGET_M),
             'slew_m_per_s': FINGER_SLEW_M_S, 'width_min_m': OPEN_WIDTH_M,
             'finger_min_m': OPEN_FINGER_M,
             'claim': 'the fingers are ALREADY at this open pair in the frozen prefix, so the phase '
                      'verifies the current public observation. A width is not proof of collision '
                      'clearance'},
    'rotate': {'max_s': ROTATE_MAX_S, 'axis': 'arm_joint6', 'delta_rad': float(WRIST_DELTA_RAD),
               'rate_rad_per_s': WRIST_RATE_RAD_S, 'tether_rad': WRIST_TETHER_RAD,
               'arrival_error_rad': WRIST_ERROR_RAD,
               'progress_window_s': WRIST_PROGRESS_WINDOW_S, 'progress_rad': WRIST_PROGRESS_RAD,
               'goal_source': 'PUBLIC measured q6 at the end of CONTACT_OPEN plus the fixed +pi/6; '
                              'latched once and never re-chased against later measured q',
               'bounds': 'the rate bound on the previous command, the measured tether and the original '
                         'static hard limits are intersected simultaneously; an empty intersection stops '
                         'and holds the previous valid command'},
    'lower': {'max_s': LOWER_MAX_S, 'alpha_rate_per_step': 'dt/' + repr(LOWER_ALPHA_DIVISOR),
              'extra_delta_source': 'an unchanged copy of the frozen child lower_delta; with the child '
                                    f'lowering parameter {CONTACT_LOWERING_M} m this is precisely one '
                                    f'more nominal {CONTACT_LOWERING_M} m reference',
              'leg_hard_limits_rad': {key: list(value) for key, value in LEG_LIMITS_RAD.items()},
              'response': {'increment_error_rad': LOWER_INCREMENT_ERROR_RAD,
                           'fault_s': LOWER_INCREMENT_FAULT_S, 'beta_min': LOWER_BETA_MIN,
                           'beta_after_half_alpha_s': LOWER_BETA_AFTER_HALF_S},
              'claim': 'the child supplies its own completed .02 m reference and this module ADDS a '
                       'separate copy on the leg action slice: the child term is never replaced, edited '
                       'or counted twice. Increment error and beta describe public leg joint response '
                       'only and claim nothing about actual chassis height'},
    'quiet_window': {'window_s': QUIET_WINDOW_S, 'arm_span_rad': QUIET_ARM_SPAN_RAD,
                     'leg_span_rad': QUIET_LEG_SPAN_RAD, 'arm_qdot_rad_per_s': QUIET_ARM_QDOT_RAD_S,
                     'held_axis_error_rad': HOLD_ERROR_RAD, 'tangent_speed_m_s': QUIET_TANGENT_M_S,
                     'linear_norm_m_s': QUIET_LINEAR_M_S, 'angular_norm_rad_s': QUIET_ANGULAR_RAD_S,
                     'tilt_rad': QUIET_TILT_RAD,
                     'note': 'one COMPLETE window of public samples, both endpoints included. Each of '
                             'CONTACT_ROTATE arrival, CONTACT_LOWER admission and the CONTACT_LOWER '
                             'final settle needs its OWN fresh window: the frozen FirstReach gate and any '
                             'previous window are never reused, and moving descent samples are excluded '
                             'from the final one'},
    'reference_lowering_m': {'first_reach': CONTACT_LOWERING_M, 'extra': CONTACT_LOWERING_M,
                             'total_reference': 2*CONTACT_LOWERING_M,
                             'claim': 'a NOMINAL joint-space reference sum only; no actual body-height '
                                      'claim of any kind'},
    'inherited_probe': 'PROBE_CLOSE / PROBE_LIFT / PROBE_OBSERVE are inherited verbatim: the same '
                       f'[-{CLOSE_PRELOAD_M}, +{CLOSE_PRELOAD_M}] m finger target, close 2.4/4 s, paired '
                       'q2-.20/q3+.20 measured lift, q2 feedback cap .12/tau .10, q2 tether .18 with .10 '
                       'on the other axes, .10 rad/s arm slew, .04 rad fixed-goal arrival, 15 mm FK rise, '
                       'the original .5 s arrival window and the 2 s observation. The LIFT reference and '
                       'R0 therefore latch the NEW, rotated and deeper public measured q normally',
    'budgets': {'contact_total_s': CONTACT_TOTAL_S, 'phase_deadlines_s': dict(PHASE_MAX_S),
                'inherited_probe_total_s': TOTAL_PROBE_S,
                'base_budgets_from_contact_entry': dict(BASE_BUDGETS),
                'pause_episode_s': PAUSE_EPISODE_S, 'pause_total_s': PAUSE_TOTAL_S,
                'pause_quiet_s': PAUSE_QUIET_S,
                'note': 'phase deadlines and both global budgets count pause time; there is no fallback '
                        'retry, re-anchoring, restart or deadline extension, and every failure holds the '
                        'last valid action'},
    'claim': 'an experimental contact candidate only: no verified contact, grasp, lift, transport, '
             'delivery or score claim, and no proof of collision clearance or body height',
}


class ContactGraspPolicy(GraspProbePolicy):
    """Frozen ``FirstReachPolicy(.02)``, open jaw, +pi/6 wrist, extra .02 m, then the inherited probe.

    The public interface is the inherited one: ``act(proprio, images)``, ``state``,
    ``done_reason``, ``debug``, ``describe()``, ``alpha``, ``pause_for_stance`` and
    ``wheel_hold_requested``. Only ``_prefix`` interception, ``_probe`` dispatch and
    ``_action``/``_record``/``describe`` extension are added; the inherited close,
    lift and observation code is reused unchanged.
    """

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, ramp_calls=100, reach_only=False,
                 forward_cmd=.20, turn_cap=.20, standoff=.56, turn_gain=.4,
                 lowering_m=CONTACT_LOWERING_M):
        # Initialise every added field BEFORE the super constructor, so any
        # dynamic dispatch into the overridden _record/_action can never see a
        # half-built instance. The dt- and name-dependent ones follow below.
        self._init_contact_fields()
        if np.ndim(lowering_m) != 0:
            raise ValueError('lowering_m must be one finite scalar equal to '
                             + repr(CONTACT_LOWERING_M) + ' m')
        try:
            lowering = float(lowering_m)
        except (TypeError, ValueError):
            raise ValueError('lowering_m must be one finite scalar equal to '
                             + repr(CONTACT_LOWERING_M) + ' m')
        if not np.isfinite(lowering) or abs(lowering-CONTACT_LOWERING_M) > 1e-9:
            # This candidate is defined at one amplitude only, and the accepted
            # value is forwarded verbatim: never clipped or rounded onto it.
            raise ValueError('The contact-grasp candidate is defined only for a first reach lowered by '
                             + repr(CONTACT_LOWERING_M) + ' m; got ' + repr(lowering_m))
        super().__init__(schema, observation_joint_names, defaults, dt=dt,
                         settle_calls=settle_calls, ramp_calls=ramp_calls, reach_only=reach_only,
                         forward_cmd=forward_cmd, turn_cap=turn_cap, standoff=standoff,
                         turn_gain=turn_gain, lowering_m=lowering)
        # One COMPLETE .5 s window, both endpoints included: 26 samples at 50 Hz.
        self.contact_quiet_samples = int(np.ceil(QUIET_WINDOW_S/self.dt))+1
        self.contact_arm_window = deque(maxlen=self.contact_quiet_samples)
        self.contact_leg_window = deque(maxlen=self.contact_quiet_samples)
        self.open_target = np.array(OPEN_TARGET_M, dtype=float)
        # Public observation order for the leg joints, in ACTION leg-name order.
        self.contact_leg_obs_ids = np.array([self.names.index(name) for name in self.leg.joint_names])
        self.contact_leg_defaults = np.array([self.defaults[name] for name in self.leg.joint_names])
        lower, upper = [], []
        for name in self.leg.joint_names:
            match = LEG_JOINT_PATTERN.match(name)
            if match is None:
                raise ValueError('Unrecognized leg joint ' + repr(name))
            low, high = LEG_LIMITS_RAD[match.group('link')]
            lower.append(low)
            upper.append(high)
        self.contact_leg_lower = np.array(lower, dtype=float)
        self.contact_leg_upper = np.array(upper, dtype=float)

    def _init_contact_fields(self):
        """Every added field, safe to read before the super constructor runs."""
        self.contact_phase, self.contact_phase_start = None, None
        self.contact_entry_call, self.contact_handoff_call = None, None
        self.contact_open_hold_calls, self.contact_width_lost_start = 0, None
        self.wrist_start_q, self.wrist_goal = None, None
        self.wrist_command, self.wrist_error = None, None
        self.wrist_limit_violation = None
        self.contact_hold_command = None
        self.rotate_active_calls, self.wrist_progress_anchor = 0, None
        self.contact_quiet_samples = 0
        self.contact_arm_window, self.contact_leg_window = deque(), deque()
        self.contact_arm_span, self.contact_leg_span = None, None
        self.contact_quiet_ready, self.contact_quiet_tick = False, False
        self.contact_body_quiet = False
        self.contact_arm_qdot, self.contact_hold_error = None, None
        self.contact_leg_q = None
        self.contact_lower_stage, self.contact_lower_authorized_call = None, None
        self.contact_leg_q0, self.contact_extra_delta = None, None
        self.contact_extra_alpha = 0.
        self.contact_extra_limit_violations = {}
        self.contact_increment_error, self.contact_increment_fault_start = 0., None
        self.contact_beta, self.contact_alpha_half_call = None, None

    # -- action assembly ----------------------------------------------------

    def _action(self):
        """The inherited action plus the SEPARATE extra leg reference increment.

        The parent already supplies the child's own completed lowering term; this
        only ADDS ``contact_extra_alpha*extra_delta/leg.scale`` on the same leg
        slice, so the child term is never replaced, edited or counted twice.
        Before authorization there is no extra delta and nothing is added, which
        is also why the first-reach prefix action is untouched.
        """
        out = super()._action()
        delta = getattr(self, 'contact_extra_delta', None)
        if delta is not None:
            alpha = float(getattr(self, 'contact_extra_alpha', 0.))
            out[self.leg.start:self.leg.stop] += alpha*delta/self.leg.scale
        return out

    def _clear_contact_windows(self):
        """Drop the fresh arm/leg quiet window; readiness is never carried over."""
        self.contact_arm_window.clear()
        self.contact_leg_window.clear()
        self.contact_arm_span, self.contact_leg_span = None, None
        self.contact_quiet_ready = False

    def _hold_open_fingers(self):
        """Slew the finger commands toward the open pair at the original limit."""
        self.arm_command[6:] += np.clip(self.open_target-self.arm_command[6:],
                                        -FINGER_SLEW_M_S*self.dt, FINGER_SLEW_M_S*self.dt)

    # -- first-reach prefix -------------------------------------------------

    def _prefix(self, proprio, images):
        """One inherited prefix call; its action is returned byte-for-byte.

        Every non-normal reason keeps propagating unchanged through the parent.
        Only the parent's own normal ``reach_lowering_hold_complete`` handoff,
        which has just entered the inherited PROBE_CLOSE, is converted into
        CONTACT_OPEN before the next action is produced. The parent's probe entry
        clock is set back to None so its 12 s window has truthfully not begun,
        while its public gravity anchor, held arm command and wheel hold from
        this same call are kept exactly as they are.
        """
        action = super()._prefix(proprio, images)
        if self.contact_entry_call is None and self.phase == 'PROBE_CLOSE':
            self.contact_entry_call = self.calls
            self.contact_hold_command = self.held_arm_command.copy()
            self.entry_call = None
            self._enter_contact_phase(
                'CONTACT_OPEN', 'verifying_the_fully_open_jaw_before_any_wrist_rotation')
            self._hold = True
            self._record()
        return action

    def _enter_contact_phase(self, phase, reason):
        """Enter one added phase and clear the window a phase change invalidates."""
        self.contact_phase = self.phase = self.state = phase
        self.contact_phase_start = self.phase_start = self.calls
        self.state_reason = reason
        self._clear_contact_windows()

    # -- added preparation phases -------------------------------------------

    def _probe(self, proprio):
        """Global deadline once per post-prefix tick, then dispatch.

        The inherited ``_probe`` owns CLOSE/LIFT/OBSERVE completely, including its
        single ``_budgets`` integration; the added path below integrates the same
        budgets exactly once itself, so no tick double-integrates.
        """
        if self.contact_entry_call is not None:
            if (self.calls-self.contact_entry_call)*self.dt >= CONTACT_TOTAL_S:
                return self._stop('contact_total_budget_exceeded')
        if self.contact_phase is None:
            return super()._probe(proprio)
        return self._contact_tick(proprio)

    def _contact_tick(self, proprio):
        """One added-phase tick: unchanged safety, budgets, pause, then the phase."""
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
        self.contact_leg_q = np.asarray(obs[12+self.contact_leg_obs_ids]
                                        + self.contact_leg_defaults, dtype=float)
        # Exactly one integration of the inherited body budgets on this tick,
        # from the same anchor the contact entry recorded, pauses included.
        budget_exit = self._budgets(obs, up)
        if budget_exit is not None:
            return self._stop(budget_exit)
        if (self.calls-self.contact_phase_start)*self.dt >= PHASE_MAX_S[self.contact_phase]:
            return self._stop(PHASE_DEADLINE_REASON[self.contact_phase])
        lost = self._width_watchdog()
        if lost is not None:
            return self._stop(lost)
        paused = self._pause(bool(self._inner.pause_for_stance))
        if paused is not None:
            if paused != 'FROZEN':
                return self._stop(paused)
            # Commands, the extra alpha and the extra reference all freeze; the
            # fresh windows are cleared and the deadlines keep counting.
            self._clear_contact_windows()
            self.contact_quiet_tick = False
            self.state_reason = 'bounded_pause_freezing_contact_commands_and_the_extra_lowering_alpha'
            action = self._action()
            self._record()
            return action
        if self.contact_phase == 'CONTACT_OPEN':
            action = self._contact_open(q)
        elif self.contact_phase == 'CONTACT_ROTATE':
            action = self._contact_rotate(q, arm_qdot)
        else:
            action = self._contact_lower(q, arm_qdot)
        if self.done_reason is None:  # a phase helper may already have stopped
            self._record()
        return action

    def _width_watchdog(self):
        """Open-width watchdog for the rotation and the extra lowering.

        Armed once the wrist goal is latched. Losing the open width stops the
        candidate where it is: it never closes and never lifts instead.
        """
        if self.wrist_goal is None or self.width is None or self.width >= WIDTH_KEEP_M:
            self.contact_width_lost_start = None
            return None
        if self.contact_width_lost_start is None:
            self.contact_width_lost_start = self.calls
        if (self.calls-self.contact_width_lost_start+1)*self.dt < WIDTH_LOST_S:
            return None
        return 'contact_open_width_lost'

    def _contact_open(self, q):
        """Verify the already open jaw while every other command is retained.

        The frozen prefix has already commanded this open pair, so this phase is
        an observation of the current public width, not a new opening motion. A
        width is never a claim that the fingers clear the object.
        """
        self.state = self.contact_phase
        self.state_reason = 'verifying_the_fully_open_jaw_before_any_wrist_rotation'
        self.arm_command[:6] = self.contact_hold_command
        self._hold_open_fingers()
        eligible = bool(self.width >= OPEN_WIDTH_M and self.q7 >= OPEN_FINGER_M
                        and self.q8 <= -OPEN_FINGER_M)
        self.contact_open_hold_calls = self.contact_open_hold_calls+1 if eligible else 0
        if self.contact_open_hold_calls*self.dt >= OPEN_HOLD_S:
            return self._enter_rotate(q)
        return self._action()

    def _enter_rotate(self, q):
        """Latch the fixed wrist goal from PUBLIC measured q6, once."""
        start = float(q[5])
        goal = start + float(WRIST_DELTA_RAD)
        low, high = float(self.static_lower[5]), float(self.static_upper[5])
        self.wrist_start_q, self.wrist_goal = start, goal
        if not (low-1e-9 <= goal <= high+1e-9):
            self.wrist_limit_violation = {'arm_joint6_goal_rad': goal, 'lower_rad': low, 'upper_rad': high}
            self.wrist_goal = None
            return self._stop('contact_wrist_goal_outside_joint_limits')
        # The five other axes keep the commands they already hold; only q6 moves.
        self.contact_hold_command = self.arm_command[:6].copy()
        self.wrist_command = float(self.arm_command[5])
        self._enter_contact_phase('CONTACT_ROTATE',
                                  'latched_the_fixed_relative_wrist_goal_from_public_measured_q6')
        self.rotate_active_calls, self.wrist_progress_anchor = 0, None
        return self._action()

    def _wrist_step(self, q):
        """Advance the q6 command inside all three bounds simultaneously."""
        rate = WRIST_RATE_RAD_S*self.dt
        low = max(self.wrist_command-rate, float(q[5])-WRIST_TETHER_RAD, float(self.static_lower[5]))
        high = min(self.wrist_command+rate, float(q[5])+WRIST_TETHER_RAD, float(self.static_upper[5]))
        if low > high:
            # Hold the previous valid command: no sequential clipping, no jump.
            return self._stop('contact_wrist_command_bounds_infeasible')
        self.wrist_command = float(np.clip(self.wrist_goal, low, high))
        return None

    def _quiet_tick(self, q, arm_qdot):
        """Maintain ONE fresh, complete public arm/leg/body quiet window.

        Every sample must satisfy the fixed-goal wrist error, the public arm
        speed cap, the held-axis errors and the public body-quiet bounds; an
        ineligible sample clears the whole window. Readiness additionally needs a
        complete window whose per-axis arm span is within .002 rad and whose
        per-axis leg span is within .02 rad. It is a public-observation window,
        not a stillness claim, and it is never shared between phases.
        """
        self.contact_hold_error = float(np.max(np.abs(q[:5]-self.contact_hold_command[:5])))
        self.wrist_error = abs(float(q[5])-self.wrist_goal)
        self.contact_arm_qdot = float(np.max(np.abs(arm_qdot)))
        self.contact_body_quiet = bool(self.motion['tangent_speed_m_s'] < QUIET_TANGENT_M_S
                                      and self.motion['linear_norm'] < QUIET_LINEAR_M_S
                                      and self.motion['angular_norm'] < QUIET_ANGULAR_RAD_S
                                      and self.motion['tilt_rad'] <= QUIET_TILT_RAD)
        self.contact_quiet_tick = bool(self.wrist_error < WRIST_ERROR_RAD
                                       and self.contact_arm_qdot <= QUIET_ARM_QDOT_RAD_S
                                       and self.contact_hold_error < HOLD_ERROR_RAD
                                       and self.contact_body_quiet)
        if not self.contact_quiet_tick:
            self._clear_contact_windows()
            return False
        self.contact_arm_window.append(np.asarray(q[:6], dtype=float).copy())
        self.contact_leg_window.append(self.contact_leg_q.copy())
        if len(self.contact_arm_window) < self.contact_quiet_samples:
            self.contact_arm_span, self.contact_leg_span = None, None
            self.contact_quiet_ready = False
            return False
        self.contact_arm_span = float(np.max(np.ptp(np.asarray(self.contact_arm_window), axis=0)))
        self.contact_leg_span = float(np.max(np.ptp(np.asarray(self.contact_leg_window), axis=0)))
        self.contact_quiet_ready = bool(self.contact_arm_span <= QUIET_ARM_SPAN_RAD
                                        and self.contact_leg_span <= QUIET_LEG_SPAN_RAD)
        return self.contact_quiet_ready

    def _contact_rotate(self, q, arm_qdot):
        """Move ONLY q6 toward the fixed latched goal under all three bounds."""
        self.state = self.contact_phase
        self.state_reason = 'rotating_only_q6_toward_the_fixed_relative_wrist_goal'
        self.rotate_active_calls += 1
        stopped = self._wrist_step(q)
        if stopped is not None:
            return stopped
        self.arm_command[:6] = self.contact_hold_command
        self.arm_command[5] = self.wrist_command
        self._hold_open_fingers()
        if self._quiet_tick(q, arm_qdot):
            self._enter_contact_phase(
                'CONTACT_LOWER', 'reacquiring_a_fresh_public_quiet_window_before_the_extra_leg_reference')
            self.contact_lower_stage = 'ADMIT'
            return self._action()
        if self.wrist_progress_anchor is None:
            self.wrist_progress_anchor = (self.rotate_active_calls, self.wrist_error)
        anchor_calls, anchor_error = self.wrist_progress_anchor
        if (self.rotate_active_calls-anchor_calls)*self.dt >= WRIST_PROGRESS_WINDOW_S:
            if self.wrist_error > WRIST_ERROR_RAD and anchor_error-self.wrist_error < WRIST_PROGRESS_RAD:
                return self._stop('contact_wrist_no_progress')
            self.wrist_progress_anchor = (self.rotate_active_calls, self.wrist_error)
        return self._action()

    def _contact_lower(self, q, arm_qdot):
        """Admit, drive and settle ONE extra fixed leg reference increment."""
        self.state = self.contact_phase
        self.arm_command[:6] = self.contact_hold_command
        self.arm_command[5] = self.wrist_command  # the last valid rotated command
        self._hold_open_fingers()
        ready = self._quiet_tick(q, arm_qdot)
        if self.contact_lower_stage == 'ADMIT':
            self.state_reason = 'waiting_for_a_fresh_public_quiet_window_before_the_extra_leg_reference'
            if ready:
                stopped = self._authorize_extra_lowering()
                if stopped is not None:
                    return stopped
            return self._action()
        # Public leg joint response to the commanded increment. Both measures
        # describe leg joints only, never actual chassis height.
        self.contact_increment_error = float(np.max(np.abs(
            (self.contact_leg_q-self.contact_leg_q0)
            - self.contact_extra_alpha*self.contact_extra_delta)))
        denominator = float(np.dot(self.contact_extra_delta, self.contact_extra_delta))
        self.contact_beta = (float(np.dot(self.contact_leg_q-self.contact_leg_q0,
                                          self.contact_extra_delta)/denominator)
                             if denominator > 0. else None)
        if self.contact_increment_error > LOWER_INCREMENT_ERROR_RAD:
            if self.contact_increment_fault_start is None:
                self.contact_increment_fault_start = self.calls
            if (self.calls-self.contact_increment_fault_start+1)*self.dt >= LOWER_INCREMENT_FAULT_S:
                return self._stop('contact_lower_increment_error')
        else:
            self.contact_increment_fault_start = None
        if self.contact_extra_alpha >= .5:
            if self.contact_alpha_half_call is None:
                self.contact_alpha_half_call = self.calls
            if ((self.calls-self.contact_alpha_half_call)*self.dt >= LOWER_BETA_AFTER_HALF_S
                    and self.contact_beta is not None and self.contact_beta < LOWER_BETA_MIN):
                return self._stop('contact_lower_no_joint_progress')
        if self.contact_extra_alpha >= 1.:
            if self.contact_lower_stage != 'SETTLE':
                # A fresh final window: no moving descent sample may enter it.
                self.contact_lower_stage = 'SETTLE'
                self._clear_contact_windows()
                ready = False
            self.state_reason = 'holding_the_completed_extra_leg_reference_for_a_fresh_public_quiet_window'
            if ready:
                return self._enter_probe_close()
            return self._action()
        self.state_reason = 'incrementing_the_extra_fixed_leg_reference_after_the_fresh_quiet_window'
        # Alpha advances only on active, body-quiet ticks with valid arm
        # readiness and a retained open width; a pause freezes it, and no stop
        # ever clears or reverses it.
        if self.contact_quiet_tick and self.width >= WIDTH_KEEP_M:
            self.contact_extra_alpha = float(min(1., self.contact_extra_alpha
                                                 + self.dt/LOWER_ALPHA_DIVISOR))
        return self._action()

    def _authorize_extra_lowering(self):
        """Latch the public start leg q and one unchanged copy of the child delta.

        Because the child lowering parameter is .02 m, that copy is precisely one
        more nominal .02 m reference. It is a reference, not a height
        measurement, and the child's own term stays exactly where it is.
        """
        start = self.contact_leg_q.copy()
        extra = np.asarray(self._inner.lower_delta, dtype=float).copy()
        target = start + extra
        violations = {}
        for index, name in enumerate(self.leg.joint_names):
            low, high = float(self.contact_leg_lower[index]), float(self.contact_leg_upper[index])
            if not (low-1e-9 <= start[index] <= high+1e-9 and low-1e-9 <= target[index] <= high+1e-9):
                violations[name] = [float(start[index]), float(target[index]), low, high]
        self.contact_extra_limit_violations = violations
        if violations:
            return self._stop('contact_lower_reference_outside_joint_limits')
        self.contact_leg_q0, self.contact_extra_delta = start, extra
        self.contact_lower_authorized_call = self.calls
        self.contact_lower_stage = 'DESCEND'
        self._clear_contact_windows()
        self.state_reason = 'authorized_the_extra_fixed_leg_reference_after_a_fresh_public_quiet_window'
        return None

    def _enter_probe_close(self):
        """Hand the prepared pose to the inherited, unchanged force75 probe.

        Only the CLOSE-phase-specific evidence and the inherited probe clocks are
        reset. The body displacement/yaw/gravity anchor, the cumulative pause
        counts, the contact entry and the extra alpha and reference are all
        preserved, so the extra leg term stays at alpha 1 through CLOSE, LIFT,
        OBSERVE and every stop.
        """
        self.contact_handoff_call = self.calls
        self.contact_phase = None
        self.phase = self.state = 'PROBE_CLOSE'
        self.state_reason = 'entering_the_inherited_unchanged_close_lift_and_observation_probe'
        self.held_arm_command = self.arm_command[:6].copy()
        self.entry_call = self.phase_start = self.calls
        self.close_active_calls, self.close_target_commanded = 0, False
        self.width_window.clear()
        self.empty_start = None
        self._clear_contact_windows()
        return self._action()

    # -- diagnostics --------------------------------------------------------

    def _phase_remaining_s(self):
        if self.contact_phase is not None:
            return max(0., PHASE_MAX_S[self.contact_phase]
                       - (self.calls-self.contact_phase_start)*self.dt)
        return super()._phase_remaining_s()

    def _contact_debug(self):
        """The added diagnostics, plus corrections of now-inaccurate parent text."""
        elapsed = (None if self.contact_entry_call is None
                   else (self.calls-self.contact_entry_call)*self.dt)
        return dict(
            contact_phase=(self.contact_phase if self.contact_phase is not None else self.phase),
            contact_phase_order=list(CONTACT_PHASES)+list(PROBE_PHASES),
            contact_lower_stage=self.contact_lower_stage,
            contact_entry_call=self.contact_entry_call,
            contact_handoff_call=self.contact_handoff_call,
            contact_phase_elapsed_s=(None if self.contact_phase is None
                                     else (self.calls-self.contact_phase_start)*self.dt),
            contact_phase_remaining_s=(None if self.contact_phase is None
                                       else self._phase_remaining_s()),
            contact_phase_deadlines_s=dict(PHASE_MAX_S),
            contact_elapsed_s=elapsed,
            contact_remaining_s=(None if elapsed is None else max(0., CONTACT_TOTAL_S-elapsed)),
            contact_total_s=CONTACT_TOTAL_S,
            first_reach_lowering_m=self._inner.lowering_m,
            first_reach_lowering_alpha=self._inner.lowering_alpha,
            extra_lowering_m=CONTACT_LOWERING_M,
            total_reference_lowering_m=2*CONTACT_LOWERING_M,
            reference_lowering_basis='the frozen child owns its own .02 m reference at its own alpha, and '
                                     'this module ADDS one separate copy of the child leg delta at '
                                     'contact_extra_alpha. The nominal references sum to .04 m; this is '
                                     'NOT a measurement or a claim that the body descended at all',
            contact_extra_alpha=float(self.contact_extra_alpha),
            contact_extra_delta_rad=(None if self.contact_extra_delta is None
                                     else self.contact_extra_delta.tolist()),
            contact_extra_delta_source='an unchanged copy of the frozen child lower_delta, mapped onto the '
                                       'leg action slice as alpha*delta/leg_scale in action leg-name '
                                       'order; the child term is not replaced or counted twice',
            contact_extra_leg_joint_order=list(self.leg.joint_names),
            contact_open_hold_s=self.contact_open_hold_calls*self.dt,
            contact_open_requirements={'width_m': OPEN_WIDTH_M, 'q7_m': OPEN_FINGER_M,
                                       'q8_m': -OPEN_FINGER_M, 'hold_s': OPEN_HOLD_S,
                                       'max_s': OPEN_MAX_S},
            contact_open_claim='the frozen prefix already commands this open pair, so the phase confirms '
                               'the current public width; it is never a collision-clearance proof',
            contact_open_width_lost_s=(0. if self.contact_width_lost_start is None
                                       else (self.calls-self.contact_width_lost_start+1)*self.dt),
            contact_open_width_keep_m=WIDTH_KEEP_M,
            contact_wrist_start_q_rad=self.wrist_start_q,
            contact_wrist_goal_rad=self.wrist_goal,
            contact_wrist_delta_rad=float(WRIST_DELTA_RAD),
            contact_wrist_command_rad=self.wrist_command,
            contact_wrist_error_rad=self.wrist_error,
            contact_wrist_bounds={'rate_rad_per_s': WRIST_RATE_RAD_S, 'tether_rad': WRIST_TETHER_RAD,
                                  'hard_lower_rad': float(self.static_lower[5]),
                                  'hard_upper_rad': float(self.static_upper[5]),
                                  'note': 'intersected simultaneously; an empty intersection stops and '
                                          'holds the previous valid command'},
            contact_wrist_limit_violation=self.wrist_limit_violation,
            contact_rotate_active_s=self.rotate_active_calls*self.dt,
            contact_held_arm_command_rad=(None if self.contact_hold_command is None
                                          else self.contact_hold_command.tolist()),
            contact_held_axis_error_rad=self.contact_hold_error,
            contact_arm_qdot_rad_s=self.contact_arm_qdot,
            contact_body_quiet=bool(self.contact_body_quiet),
            contact_quiet_tick=bool(self.contact_quiet_tick),
            contact_quiet_samples=len(self.contact_arm_window),
            contact_quiet_required_samples=self.contact_quiet_samples,
            contact_quiet_window_s=max(0., (len(self.contact_arm_window)-1)*self.dt),
            contact_quiet_required_s=QUIET_WINDOW_S,
            contact_quiet_arm_span_rad=self.contact_arm_span,
            contact_quiet_leg_span_rad=self.contact_leg_span,
            contact_quiet_span_limits_rad={'arm': QUIET_ARM_SPAN_RAD, 'leg': QUIET_LEG_SPAN_RAD},
            contact_quiet_ready=bool(self.contact_quiet_ready),
            contact_quiet_basis='one COMPLETE window of public arm/leg positions with the wrist error, '
                                'held-axis errors, arm speed and body bounds satisfied on every sample. '
                                'CONTACT_ROTATE arrival, CONTACT_LOWER admission and the final settle each '
                                'need their OWN fresh window; the frozen FirstReach gate is never reused '
                                'and a pause or a phase change clears it',
            contact_lower_authorized_call=self.contact_lower_authorized_call,
            contact_lower_start_leg_q_rad=(None if self.contact_leg_q0 is None
                                           else self.contact_leg_q0.tolist()),
            contact_lower_leg_q_rad=(None if self.contact_leg_q is None else self.contact_leg_q.tolist()),
            contact_lower_increment_error_rad=float(self.contact_increment_error),
            contact_lower_beta=self.contact_beta,
            contact_lower_limits_rad={'hip': list(LEG_LIMITS_RAD['hip']),
                                      'thigh': list(LEG_LIMITS_RAD['thigh']),
                                      'calf': list(LEG_LIMITS_RAD['calf'])},
            contact_lower_limit_violations=self.contact_extra_limit_violations,
            contact_lower_response_limits={'increment_error_rad': LOWER_INCREMENT_ERROR_RAD,
                                           'fault_s': LOWER_INCREMENT_FAULT_S,
                                           'beta_min': LOWER_BETA_MIN,
                                           'beta_after_half_alpha_s': LOWER_BETA_AFTER_HALF_S},
            contact_lower_response_claim='public leg joint response to the extra fixed reference; no claim '
                                         'about actual chassis height',
            contact_lower_alpha_gate='the extra alpha advances by at most dt/3 only on active ticks whose '
                                     'wrist error, held-axis errors, public arm speed and public '
                                     f'body-quiet bounds all hold and whose width is >= {WIDTH_KEEP_M} m; '
                                     'a pause or invalid arm readiness freezes it, and no stop clears or '
                                     'reverses it',
            contact_probe_clock_started=self.entry_call is not None,
            contact_probe_clock_basis=('the inherited 12 s probe clock has NOT started; it begins only at '
                                       'the inherited PROBE_CLOSE entry'
                                       if self.entry_call is None else
                                       'the inherited 12 s probe clock started at the inherited '
                                       'PROBE_CLOSE entry and covers CLOSE/LIFT/OBSERVE only'),
            contact_preserved_budgets={'base_budgets_from_contact_entry': dict(BASE_BUDGETS),
                                       'base_displacement_m': float(np.linalg.norm(self.base_displacement)),
                                       'base_yaw_rad': abs(float(self.base_yaw)),
                                       'base_gravity_change_rad': float(self.gravity_change),
                                       'gravity_anchor_up': (None if self.up_anchor is None
                                                             else self.up_anchor.tolist()),
                                       'pause_episodes': self.pause_episodes,
                                       'pause_total_s': self.pause_calls*self.dt,
                                       'pause_episode_s': PAUSE_EPISODE_S,
                                       'pause_budget_s': PAUSE_TOTAL_S,
                                       'note': 'the anchor, the drift integrals and the cumulative pause '
                                               'counts are carried from the contact entry through the '
                                               'inherited probe and are never reset or re-anchored'},
            contact_wheel_hold_requested=bool(self._hold),
            contact_wheel_hold_basis='the inherited wheel anchor requested at the prefix handoff stays '
                                     'requested continuously through CONTACT_OPEN/ROTATE/LOWER and the '
                                     'inherited probe; it is never released or re-engaged',
            contact_candidate_constants=CONTACT_CONSTANTS,
            # Corrections of inherited text that no longer describes this policy.
            probe_lowered_reference_basis='the child\'s own latched lowering alpha and fixed leg delta are '
                                          'still reused verbatim and never written, AND this subclass adds '
                                          'one SEPARATE extra leg reference of its own on the same action '
                                          'slice, so the commanded leg reference is deeper than the '
                                          'child\'s. The base is never raised and no actual body height is '
                                          'claimed',
            probe_wrist_motion_basis='this subclass DOES move arm_joint6: one fixed +pi/6 goal relative to '
                                     'public measured q6, latched once before the inherited probe, which '
                                     'then latches its own lift reference from the new rotated and deeper '
                                     'public measured q',
            probe_first_reach_lowering_basis='the amplitude this instance actually gave the frozen '
                                             'FirstReachPolicy, which owns the descent, the hold and its '
                                             'own scaled lowering reference. This contact candidate '
                                             f'accepts {CONTACT_LOWERING_M} m ONLY; the inherited '
                                             'grasp-probe metadata that also lists .03 m describes the '
                                             'parent class, not this subclass',
            probe_claim='experimental contact candidate: a frozen first reach, a verified open jaw, one '
                        'fixed relative wrist rotation, one extra fixed leg reference and then the '
                        'unchanged close/lift/observation probe. No contact, grasp, transport, delivery, '
                        'collision-clearance or body-height claim of any kind')

    def _record(self):
        super()._record()
        self.debug.update(self._contact_debug())

    def describe(self):
        info = super().describe()
        contact_phases = {
            'CONTACT_OPEN': f'retain the inherited six arm commands and the achieved leg reference, slew '
                            f'the fingers toward the open pair {list(OPEN_TARGET_M)} m at '
                            f'<={FINGER_SLEW_M_S} m/s and require public width >={OPEN_WIDTH_M} m with '
                            f'q7>={OPEN_FINGER_M} m and q8<={-OPEN_FINGER_M} m for a complete '
                            f'{OPEN_HOLD_S} s inside {OPEN_MAX_S} s. The prefix already commands this '
                            'pair, so the phase verifies the current observation and proves no collision '
                            'clearance. On success the fixed wrist goal is latched from public measured '
                            f'q6 +{float(WRIST_DELTA_RAD):.6f} rad, rejected rather than clipped if it '
                            'leaves the original q6 hard limits',
            'CONTACT_ROTATE': f'move ONLY arm_joint6 toward that fixed goal, which never chases later '
                              f'measured q, under the {WRIST_RATE_RAD_S} rad/s slew, the '
                              f'{WRIST_TETHER_RAD} rad measured tether and the original hard limits '
                              'intersected at once (an empty intersection stops on the previous valid '
                              'command). The other five arm commands, the open fingers and the achieved '
                              'child leg reference are held. Arrival needs one fresh COMPLETE quiet '
                              f'window ({self.contact_quiet_samples} samples at {1./self.dt:.0f} Hz) on '
                              'the thresholds in contact_candidate_constants.quiet_window, losing the '
                              'open width stops without closing or lifting, and '
                              f'{ROTATE_MAX_S} s bounds the phase',
            'CONTACT_LOWER': 'hold the last valid rotated wrist command, the five other commands and the '
                             'open fingers; independently reacquire a fresh window with the same '
                             'conditions, never the frozen FirstReach gate or the previous phase window; '
                             'latch public leg q in action leg-name order; validate that q and q+delta '
                             'against the original leg hard limits; then raise a SEPARATE alpha by at '
                             f'most dt/{LOWER_ALPHA_DIVISOR:.0f} per active body-quiet tick and add '
                             'alpha*delta/leg_scale on the leg slice on top of the child\'s completed '
                             'term. Public leg joint response is bounded as in '
                             'contact_candidate_constants.lower.response, at alpha 1 one more fresh '
                             f'complete window is required before the inherited probe, {LOWER_MAX_S} s '
                             'bounds the phase and no actual chassis descent is claimed'}
        info.update(mode='contact_grasp',
                    contact_candidate_constants=CONTACT_CONSTANTS,
                    phases={**contact_phases, **info['phases']},
                    phase_order=list(CONTACT_PHASES)+list(PROBE_PHASES),
                    stop_reasons={**info['stop_reasons'], **CONTACT_STOP_REASONS},
                    inputs='the frozen first reach owns all vision; after its normal completion this '
                           'policy reads only the public proprio vector. No reward, score, contact or net '
                           'force, object pose, ground truth, map, seed or image enters the added phases. '
                           'The recorded finger contact on the sloping shoulder and the mu=1 / .5 kg shape '
                           'properties are OFFLINE evidence behind the fixed numbers, not runtime inputs',
                    entry_contract='the inherited prefix action is returned unchanged on every prefix '
                                   'call including its normal completion, and every non-normal reason '
                                   'propagates unchanged. Only reach_lowering_hold_complete converts the '
                                   'newly entered inherited PROBE_CLOSE into CONTACT_OPEN, records the '
                                   'contact entry call and public gravity anchor and sets the inherited '
                                   'probe clock back to not-started; the inherited post-prefix arm '
                                   'command, wheel anchor and child fields are untouched',
                    lowering_contract='this instance handed the unchanged FirstReachPolicy constructor '
                                      f'exactly {CONTACT_LOWERING_M} m, the only amplitude this candidate '
                                      'accepts, and that constructor still permits at most .03 m and is '
                                      'not modified. The frozen policy keeps owning its own descent, hold, '
                                      'alpha and leg delta, all of which stay exactly as it reports them. '
                                      'This subclass then ADDS a separate extra leg reference of one '
                                      'unchanged copy of that delta on the leg action slice, so a second '
                                      f'nominal {CONTACT_LOWERING_M} m reference exists and the nominal '
                                      f'references sum to {2*CONTACT_LOWERING_M} m. That sum is '
                                      'joint-space reference only: it is not a measurement and no actual '
                                      'body-height claim is made',
                    contact_lowering_contract={'first_reach_lowering_m': self._inner.lowering_m,
                                               'first_reach_alpha': self._inner.lowering_alpha,
                                               'extra_lowering_m': CONTACT_LOWERING_M,
                                               'total_reference_lowering_m': 2*CONTACT_LOWERING_M,
                                               'accepted_lowering_m': [CONTACT_LOWERING_M],
                                               'extra_alpha': float(self.contact_extra_alpha),
                                               'extra_delta_source': 'an unchanged copy of the child '
                                                                     'lower_delta, added on the leg '
                                                                     'action slice and never counted '
                                                                     'twice or written back',
                                               'persistence': 'the extra alpha and reference persist '
                                                              'through CLOSE, LIFT, OBSERVE and every '
                                                              'stop, and freeze during a pause'},
                    inherited_probe_contract=CONTACT_CONSTANTS['inherited_probe'],
                    budget_contract=CONTACT_CONSTANTS['budgets'],
                    wheel_and_leg_contract='the inherited wheel anchor is requested continuously from the '
                                           'prefix handoff through the added phases and the inherited '
                                           'probe, never released or re-engaged; the leg reference is the '
                                           'child\'s achieved one plus this module\'s separate extra '
                                           'increment, and the base is never raised and no extra '
                                           'locomotion is commanded',
                    contact_state={'contact_phase': self.contact_phase,
                                   'lower_stage': self.contact_lower_stage,
                                   'contact_entry_call': self.contact_entry_call,
                                   'contact_handoff_call': self.contact_handoff_call,
                                   'contact_elapsed_s': (None if self.contact_entry_call is None else
                                                         (self.calls-self.contact_entry_call)*self.dt),
                                   'contact_total_s': CONTACT_TOTAL_S,
                                   'wrist_start_q_rad': self.wrist_start_q,
                                   'wrist_goal_rad': self.wrist_goal,
                                   'wrist_command_rad': self.wrist_command,
                                   'wrist_error_rad': self.wrist_error,
                                   'quiet_samples': len(self.contact_arm_window),
                                   'quiet_required_samples': self.contact_quiet_samples,
                                   'quiet_arm_span_rad': self.contact_arm_span,
                                   'quiet_leg_span_rad': self.contact_leg_span,
                                   'quiet_ready': bool(self.contact_quiet_ready),
                                   'extra_alpha': float(self.contact_extra_alpha),
                                   'lower_authorized_call': self.contact_lower_authorized_call,
                                   'lower_start_leg_q_rad': (None if self.contact_leg_q0 is None
                                                             else self.contact_leg_q0.tolist()),
                                   'lower_increment_error_rad': float(self.contact_increment_error),
                                   'lower_beta': self.contact_beta,
                                   'probe_clock_started': self.entry_call is not None,
                                   'wheel_hold_requested': bool(self._hold),
                                   'arm_axis_order': list(ARM6)},
                    claim='experimental contact candidate only: a frozen first reach, a verified open '
                          'jaw, one fixed relative wrist rotation, one extra fixed leg reference and the '
                          'unchanged close/lift/observation probe. No verified contact, grasp success, '
                          'transport, delivery or score claim, no proof of collision clearance and no '
                          'actual body-height claim. Real validation stays external: an original-'
                          'environment run plus an independent audit of the recorded object pose, '
                          'quaternion and mesh lower surface',
                    debug=self.debug)
        return info
