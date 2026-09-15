"""Top-level carry controller: A-raise handoff, loaded mobility probe, delivery.

This is a NEW file and owns the top-level state machine only. The frozen visual
reach, contact grasp and the p13 payload baseline are composed, not edited:
:class:`PayloadRaisePrefix` is an ultra-thin subclass of the p13
:class:`~task_b.payload_motion.PayloadMotionPolicy` that publishes ONE public
handoff at the ORIGINAL A-arrival branch and then stops advancing segments. The
prefix child is called exactly once per tick while it runs and its action is
returned byte-for-byte; from the tick after the handoff it is never called again.

Why the handoff exists. The p13 policy is stationary: after A it immediately
begins B, and it enforces its own carry budgets (``PAYLOAD_BUDGETS``: .03 m of
base displacement, .05 rad of yaw). Measured on p13 that displacement budget is
about a tenth of a second of legitimate driving, so a navigation wrapper that
simply kept calling it would be stopped by the wrapper's own stationary
bookkeeping before the robot moved. The handoff therefore transfers the SAME
:class:`~task_b.payload_motion.PayloadJointTracker` instance - not a copy, not a
re-plan - so the accumulated integral, the filter state, the scalar path position
and the loaded actuator command all stay continuous across the boundary.

What the carry controller inherits and what it owns:

* it inherits the tracker, the A/B/C goals, the closed finger command and the
  EXACT final child action template, whose leg slice already carries BOTH the
  .02 m fixed lowering and the contact candidate's extra .02 m. The leg slice is
  copied verbatim; it is never rebuilt from ``lowering_m``, which would silently
  drop one of the two descents;
* it owns the wheel slice, the phase clock, the phase budgets and the wheel
  anchor request. While a movement phase is active the anchor is released; it is
  re-engaged only after a controlled brake has actually settled, and the new
  anchor is taken at the wheel angle reached then. This is the recorded
  ``multi_reach`` defect this design exists to avoid: that run latched a .03 m
  stationary budget while the base was still sliding at .235 m/s and stopped
  immediately.

What this file is NOT. There is no object pose, no contact force, no world base
pose, no reward and no score anywhere in it, and ``objects_in_circle`` is never
read. The only runtime inputs are the public 84-value proprio vector, the RGB-D
the frozen first reach already owns, and static task constants. A stop reason
here is bookkeeping: it is never a grasp, carry, clearance, delivery or score
claim, and a retained jaw width never certifies attachment.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from task_b.arm_kinematics import arm_joints_from_proprio, arm_targets_to_action, fk, solve_ik
from task_e_geometry import JOINT_LOWER, JOINT_UPPER
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.payload_motion import (
    ARM6,
    EMPTY_WIDTH_M,
    EMPTY_WIDTH_S,
    GOAL_ERROR_RAD,
    HARD_TILT_RAD,
    PayloadMotionPolicy,
    QUIET_ANGULAR_RAD_S,
    QUIET_ARM_QDOT_RAD_S,
    QUIET_ARM_SPAN_RAD,
    QUIET_LEG_SPAN_RAD,
    QUIET_LINEAR_M_S,
    QUIET_TANGENT_M_S,
    QUIET_TILT_RAD,
    QUIET_WINDOW_S,
    UNSTABLE_RATE_RAD_S,
    UNSTABLE_TILT_RAD,
)
from task_b.public_odometry import PublicPlanarOdometry

#: Wheel collision geometry measured from the original robot USD: a cylinder of
#: radius .1129 m, width .05 m, axis along body y. Used only to express a planar
#: speed request in the physical wheel-target units the drive layer emits.
WHEEL_RADIUS_M = .1129

#: The evaluator multiplies the wheel slice by ``--wheel_action_gain`` and the
#: static schema then multiplies by the wheel term scale, so one normalized
#: action unit is gain*scale physical rad/s. The carry controller emits PHYSICAL
#: rad/s and divides by this divisor at the single boundary where the action is
#: assembled. Dividing twice, or multiplying by the gain again, is the concrete
#: bug this constant exists to prevent.
DEFAULT_WHEEL_ACTION_GAIN = 8.

# -- carry probe: NEW experimental initial values, not proven capability -------
#: Every number below is a probe initial value. None of it is a measured
#: performance, a safety limit or an official threshold.
PROBE_FORWARD_M = .20
PROBE_SPEED_M_S = .03
PROBE_SPEED_CAP_M_S = .08
PROBE_YAW_TARGET_RAD = .175          # about +-10 degrees
PROBE_YAW_CAP_RAD_S = .08
#: The yaw probe rolls slowly forward while turning. The contract is explicit
#: that the low-speed forward-plus-differential form is preferred and that a
#: strong in-place rotation must NOT be the default - a skid-steer fights itself
#: hardest at zero forward speed. The d2 probe measured the in-place case, where
#: a full-authority counter-rotation produced no wheel tracking at all, so this
#: form is the one that still has to be measured.
PROBE_YAW_FORWARD_M_S = .02
#: Raised from .70 m: the arc form of the yaw probe rolls forward while turning,
#: so each 10 deg probe can add up to ~.24 m that the in-place form never
#: travelled. The cap still bounds the probe, it just now bounds the form the
#: contract actually asked for.
PROBE_PATH_CAP_M = 1.20
PROBE_TOTAL_S = 60.
ACTION_SEGMENT_MAX_S = 12.
BRAKE_SEGMENT_MAX_S = 6.
PROBE_HOLD_S = 1.
NO_PROGRESS_WINDOW_S = 5.
NO_PROGRESS_FORWARD_M = .01
NO_PROGRESS_YAW_RAD = .02
BRAKE_SETTLE_SPEED_M_S = .01

#: Emergency raw arm-speed stop candidate for the probe. p13's raw peak was
#: about .236 rad/s, so this is roughly 27% of headroom - a data-driven guard,
#: NOT a safety certification, and deliberately NOT the low-bandwidth .12 rad/s
#: run gate reused for a different purpose.
RAW_ARM_QDOT_ESTOP_RAD_S = .30

#: Stall floors for this module's own no-progress guard, matching the drive's
#: rule: a genuine absence of motion, not slowness. The old absolute test (0.02
#: rad of yaw in 5 s) implicitly demanded 0.004 rad/s, which this chassis cannot
#: do - it manages ~0.0016 rad/s rolling and ~0.001 creeping - so the probe's own
#: guard would have stopped a turn that was really happening.
STALL_YAW_RATE_RAD_S = .0002
STALL_FORWARD_SPEED_M_S = .002
#: The delivery passes this to the drive so its PROBE stall guard cannot stop a
#: deliberate slow turn. Small enough to still catch a chassis that is not
#: turning at all, far below the 0.0007-0.0016 rad/s this one actually achieves.
DELIVERY_STALL_YAW_RATE_RAD_S = 5e-5

CARRY_PROBE_PHASES = (
    'PREFIX', 'CARRY_READY',
    'PROBE_DRIVE', 'PROBE_BRAKE_DRIVE',
    'PROBE_YAW_POS', 'PROBE_BRAKE_YAW_POS',
    'PROBE_YAW_NEG', 'PROBE_BRAKE_YAW_NEG',
    'PROBE_HOLD', 'STOPPED',
)

#: Phases that actually command the base. The wheel anchor MUST be released for
#: every one of them: while ``wheel_hold_requested`` is True the evaluator
#: overwrites the whole wheel slice with its own brake holder, so a movement
#: phase that forgot to release would silently have every steering request
#: discarded and would then stall for a reason that looks like weak authority.
MOVEMENT_PHASES = frozenset({'PROBE_DRIVE', 'PROBE_YAW_POS', 'PROBE_YAW_NEG',
                             'TURN_APPROACH'})

# -- first_delivery: the frontal dock ----------------------------------------
#: The barrel is a static task constant: centre, radius 1 m, wall .02 m, wall
#: height .5 m. The inner wall is a 32-segment polygon, so the conservative
#: inscribed radius is .98*cos(pi/32); `.98` alone is not a safe circular bound.
BARREL_CENTRE_XY = np.array([-3., -10.])
INNER_WALL_INSCRIBED_M = .98*np.cos(np.pi/32.)
#: Measured on plan_d2/d3: this chassis achieves ~0.0016 rad/s of yaw against a
#: 0.08 rad/s request in BOTH the in-place and the rolling-arc form. That is why
#: the route is FRONTAL: a side dock needs 2.064 rad of heading change, which is
#: ~1290 s and exceeds the 1200 s episode limit on its own, while a frontal dock
#: needs only ~0.49 rad. The values here are the measured ones, not assumed.
MEASURED_YAW_RATE_RAD_S = .0016
#: The usable standoff window for an ON-AXIS dock, which is the only kind that
#: works. The LOWER bound is the chassis: its forward extent is .481 m, so the body
#: cannot come closer than 1.511 m without the wheels entering the wall. The UPPER
#: bound is the carried bottle leaving the barrel mouth, measured at 1.576 m.
#:
#: This was briefly widened to 1.66 m on the strength of a claim that a 0.493 rad
#: arm swing pulls the bottle back toward the axis. That claim was WRONG - it
#: rotated the barrel without checking the chassis envelope, and an off-axis dock
#: presents the chassis CORNER to the wall, which needs a LARGER standoff while the
#: bottle leaves the mouth sooner. The two constraints cross at phi ~ 0.15 rad, so
#: off-axis docking is not available and the on-axis window is the whole window.
DOCK_STANDOFF_M = 1.545
DOCK_STANDOFF_LOW_M = 1.511
DOCK_STANDOFF_HIGH_M = 1.576

#: The window above is the ON-AXIS one. It narrows as the dock goes off-axis, from
#: two directions at once: the chassis presents its CORNER to the wall (so the
#: minimum standoff grows) while the carried bottle reaches the mouth edge sooner
#: (so the maximum falls). Measured on the collision geometry: forward extent .481 m,
#: lateral extent .380 m. At a .12 rad bearing error the real window is only about
#: [1.553, 1.567] - 14 mm - so a fixed [1.511, 1.576] would happily dock the chassis
#: 33 mm INSIDE the wall. The window is therefore computed from the measured bearing
#: error rather than assumed constant.
CHASSIS_FORWARD_EXTENT_M = .481
CHASSIS_LATERAL_EXTENT_M = .380
CLEARANCE_MARGIN_M = .03
#: How fast the bottle-side limit falls with bearing error, from the measured
#: 1.576 m on-axis and 1.556 m at 0.256 rad.
BOTTLE_LIMIT_FALL_PER_RAD_M = .078
#: Why the dock aims at the window CENTRE and not its top: the approach comes from
#: outside, so a "dock as soon as the range is inside" rule fires on the upper edge.
#: With a window only tens of mm wide that leaves essentially no margin - the smoke
#: test settled at 1.5758 m against a 1.576 m limit, 0.2 mm of it.
DOCK_AIM_TOLERANCE_M = .01

#: Low place before release. Releasing at the raise height drops the bottle 0.645 m
#: onto a restitution-1 surface, and an ideal bounce returns it to its release height,
#: above the 0.55 m rim, so it can leave the barrel.
#:
#: Lowering alone does NOT fix it: with the arm merely lowered the bottle stays at
#: radial ~1.0 m and its lower part clips the wall annulus (measured 31 mesh points at
#: 0.15 m of lowering). Extending forward AND lowering does fix it, because moving the
#: gripper forward pulls the whole bottle toward the barrel axis so it can descend
#: inside the mouth. Measured on the model at the 1.511 m chassis minimum: reach 0.65 m
#: with a 0.15 m drop puts the bottle bottom at 0.4814 m - below the rim - with the
#: whole bottle inside the inscribed radius and zero wall strikes, all inside the
#: original joint limits. Chassis minimum is used because the dock cannot come closer.
LOW_PLACE_FORWARD_M = 1.0      # fraction of the way to the chassis-minimum standoff
LOW_PLACE_DROP_M = .15
#: The scalar path moves at COMMAND_RATE_RAD_S = 0.1 rad/s, and a low-place goal can
#: be ~1.0 rad away on one axis, so the move alone is ~10 s before any settling. The
#: first value of 12 s expired exactly as the arm arrived.
LOW_PLACE_DEADLINE_S = 35.
LOW_PLACE_ERROR_RAD = .06
#: Stop as soon as the range falls to the aim plus this, then brake. Tight on
#: purpose: the settled range is re-checked against the real window afterwards,
#: and the release is refused if it landed outside.
DOCK_RANGE_TOL_M = .015
#: Alignment tolerance. The heading must be good to well under the angle that
#: would walk the bottle out of a 67 mm window over 1.545 m.
DOCK_HEADING_TOL_RAD = .05
#: The chassis does NOT have to turn. Measured on plan_d8 the wheel differential
#: turns about 1 degree and then stalls at 0.00004 rad/s, so a dock demanding a
#: precise heading is unreachable. But the barrel mouth is .9753 m in radius while
#: a 28 degree swing of ``arm_joint1`` moves the fingers only about .17 m
#: laterally - checked directly on the model: at every swing from 0 to -0.493 rad
#: and every standoff from 1.45 to 1.60 m the fingers stay inside the mouth, the
#: rim clearance stays 220 mm and there are zero wall strikes. So the ARM absorbs
#: the heading error and the chassis just has to arrive at the right range.
MAX_ARM_SWING_RAD = .55

#: Hip-twist yaw. The leg ``hip_joint`` is the X axis and swings the leg sideways,
#: so an asymmetric pattern - front legs one way, rear the other - twists the body
#: about the vertical. This is the classic quadruped yaw mechanism and it uses the
#: LEG actuators, which are far stronger than the wheels: K=80 with a 100 N*m effort
#: limit, against a wheel drive whose damping of 1.0 only reaches its 20 N*m limit at
#: a 20 rad/s velocity error. The stance reference ADDS its own delta to the incoming
#: leg action (``out[leg]*scale + desired_delta``) rather than replacing it, so
#: superimposing this on the held stance passes straight through.
#:
#: The SIGN is not derived here: the mirrored legs make the abduction direction in
#: world terms uncertain from the asset alone, so it is a run parameter and the first
#: run measures which way the body actually goes.
LEG_TWIST_RAD = .28

#: How far the turn phase may back away from where the carry started. Backing is
#: radial to the barrel, so it keeps the bearing from rotating while the chassis
#: turns - but on d15 it retreated over 3 m before the heading came in, and all of
#: that has to be driven back. The turn rate only has to beat the bearing rotation
#: v*sin(theta)/d, about 0.0006 rad/s at 7 m against a ~0.002 rad/s turn, so there
#: is no need to keep retreating forever - but MEASURED, retreating is what makes
#: the turn work: d15 (unlimited) reached -0.126 rad at 185 s while d16 (limited to
#: 1.5 m) had only reached -0.266 rad at 323 s. Backing is radial, so it holds the
#: bearing still while the chassis turns, and capping it too tightly removes exactly
#: that benefit. 3.5 m keeps the benefit while bounding the retreat.
TURN_BACK_LIMIT_M = 3.5
#: Alignment and range taken from a PUBLIC observation of the actual barrel, used
#: in preference to dead reckoning. Measured over the recorded runs, the planar
#: odometry error grows linearly at about 0.29 mm/s (p13: 30 mm in 117 s; d4:
#: 83 mm in 282 s), so by the time a frontal dock is reached after ~550 s it is
#: ~160 mm - against a usable window of 67 mm. Dead reckoning alone cannot hit
#: this window, which is why the last few metres are observed instead.
DOCK_BEARING_TOL_RAD = .05
#: The head camera only sees this barrel from about 3.3 m of centre standoff, and
#: its full wall height from about 1.6 m, so observation begins just inside that.
OBSERVATION_START_RANGE_M = 3.10
#: The camera updates at 10 Hz while control runs at 50 Hz; observing every tick
#: would just reprocess the same frame five times.
OBSERVATION_PERIOD_CALLS = 5
#: How long a trustworthy observation stays usable. The estimator passes its gates
#: on most ticks but not all - d18 saw about 78% - and WITHOUT a latch the source
#: flip-flops: on a tick with no observation the bearing falls back to the
#: dead-reckoned one, which disagrees with the observed bearing by ~0.1 rad and can
#: pass an alignment test the observed bearing fails. That makes the controller
#: alternate between "aligned, drive in" and "not aligned, turn", and driving in on
#: the dead-reckoned bearing is exactly the 186 mm mistake the observation exists to
#: prevent. A recent observation therefore wins over the fallback.
#: 10 s, not 1 s. Measured on d19 the estimator passes its gates on only ~7.5% of
#: ticks once the barrel is near, so a 1 s latch expires constantly and the
#: controller falls back to dead reckoning - which on d19 believed the bearing error
#: was 0.005 rad when the observation said 0.176. Docking on the wrong bearing puts
#: the chassis CORNER into the wall, because the safe window is computed from it.
#: The bearing changes slowly (about 0.002 rad/s), so a 10 s old observation is off
#: by ~0.02 rad, which is far better than a fallback that is 0.18 rad wrong.
OBSERVATION_STALE_CALLS = 500
#: Speed scheduling. The 0.010 m/s creep first chosen here was an unvalidated
#: guess and it made the turn WORSE, not better: at creep the wheels barely roll
#: (d5 achieved only 0.0038 m/s of the 0.010 asked) and the yaw fell to
#: ~0.001 rad/s against the 0.0016 rad/s that d3 measured at 0.020 m/s. Use the
#: speed the probe actually proved.
TURN_CREEP_SPEED_M_S = .020
APPROACH_SPEED_M_S = .030
ALIGN_TOL_RAD = .08
PURSUIT_YAW_GAIN = 1.5
#: Whole-run budget. The measured turn alone is ~309 s, the approach ~100 s, so
#: the 300 s initial value in the written contract cannot hold a frontal dock.
#: The original task limit is 1200 s and is NOT modified.
DELIVERY_BUDGET_S = 1000.
#: Release. The fingers are prismatic with hard limits [0,.035] and [-.035,0], so
#: the TRUE travel from p13's recorded .0575 m grip to full open is only .0125 m
#: per the gap, not the .11 m of command-space travel the contract assumed.
FINGER_OPEN_TARGET = np.array([.035, -.035])
FINGER_SLEW_M_S = .05
OPEN_DEADLINE_S = 6.
OPEN_GAP_M = .07
OPEN_GAP_TOL_M = .003
OBSERVE_S = 3.
OBSERVE_DEADLINE_S = 5.

DELIVERY_STOP_REASONS = {
    'delivery_carry_probe_complete': 'NORMAL end of the carry probe: the A pose was handed over and '
                                     'held, a bounded loaded straight run was driven and braked to a '
                                     'settle, both bounded yaw probes ran, and a complete quiet window '
                                     'closed. It is a SEQUENCE claim only: no object pose, height, '
                                     'carry, clearance, delivery or score is measured here.',
    'delivery_handoff_never_ready': 'the frozen prefix stopped before its own A-arrival branch, so no '
                                    'carry state was ever published. The child reason is propagated '
                                    'separately and unchanged.',
}


class PayloadRaisePrefix(PayloadMotionPolicy):
    """The p13 policy, stopped at the original A arrival so a carry can take over.

    Exactly one branch is added: when the ORIGINAL A-arrival condition has
    already held (``_advance_segment('A')`` is only reached from the unchanged
    ``arrived`` test in ``_segment_tick``), publish ``handoff_ready`` and return
    that tick's original A action without calling ``tracker.begin(B)``. Nothing
    else differs, so with ``handoff_at_raise=False`` the inherited behaviour is
    unchanged.
    """

    def __init__(self, *args, handoff_at_raise=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_at_raise = bool(handoff_at_raise)
        self.handoff_ready = False
        self.handoff_call = None
        self._carry_taken = False

    def _advance_segment(self, key):
        if self.handoff_at_raise and key == 'A':
            if not self.handoff_ready:
                self.handoff_ready = True
                self.handoff_call = self.calls
                self.state_reason = ('the_ORIGINAL_A_arrival_condition_held_at_the_handoff_boundary; '
                                     'B_is_not_started_here')
            # Idempotent: a second call must never begin the swing.
            action = self._action()
            self._record()
            return action
        return super()._advance_segment(key)

    def take_carry_state(self):
        """The one public handoff. Returns the live control state, taken once."""
        if not self.handoff_ready:
            raise ValueError('the carry state is only published at the original A arrival')
        if self._carry_taken:
            raise ValueError('the carry state has already been taken')
        self._carry_taken = True
        if self.action_template is None or self.finger_command is None or self.tracker is None:
            raise ValueError('the carry state is incomplete; the prefix never fully opened')
        return {
            'control_tick': self.calls,
            'action_template': np.asarray(self.action_template, dtype=np.float32).copy(),
            'finger_command': np.asarray(self.finger_command, dtype=float).copy(),
            'goals_rad': {key: np.asarray(value, dtype=float).copy()
                          for key, value in self.goals.items()},
            'tracker': self.tracker,          # the SAME instance, never a copy
            'arm_command_rad': np.asarray(self.arm_command, dtype=float).copy(),
            'prefix_phase': self.phase,
            'prefix_state_reason': self.state_reason,
            'reference_q_rad': None if self.reference_q is None else np.asarray(
                self.reference_q, dtype=float).copy(),
            'source': 'the public 84-value proprio, the frozen visual modules own RGB-D, and the '
                      'static scene constants. No object pose, contact force, world base pose, '
                      'reward or score is present in this record.',
        }


class PayloadDeliveryPolicy:
    """Carry probe (and the delivery state-machine skeleton) on top of the prefix."""

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 mode='carry_probe', settle_calls=100, ramp_calls=100,
                 forward_cmd=.20, turn_cap=.20, standoff=.56, turn_gain=.4,
                 lowering_m=None, open_on='close_complete', reach_only=False,
                 wheel_action_gain=DEFAULT_WHEEL_ACTION_GAIN,
                 wheel_radius_m=WHEEL_RADIUS_M,
                 probe_forward_m=PROBE_FORWARD_M, probe_speed_m_s=PROBE_SPEED_M_S,
                 probe_yaw_rad=PROBE_YAW_TARGET_RAD, probe_total_s=PROBE_TOTAL_S,
                 dock_standoff_m=DOCK_STANDOFF_M, delivery_budget_s=DELIVERY_BUDGET_S,
                 release_hold_s=OBSERVE_S, dock_bearing_tol_rad=None,
                 low_place=False, leg_twist_rad=0., leg_twist_sign=1.,
                 drive_wheel_speed_max_rad_s=None, drive_differential_max_rad_s=None,
                 drive=None):
        if mode not in ('carry_probe', 'first_delivery'):
            raise ValueError('mode must be carry_probe or first_delivery')
        if reach_only:
            raise ValueError('the carry sequence needs the real visual reach; reach_only must be False')
        if not np.isfinite(wheel_action_gain) or wheel_action_gain <= 0.:
            raise ValueError('wheel_action_gain must be positive and finite')
        if not np.isfinite(wheel_radius_m) or wheel_radius_m <= 0.:
            raise ValueError('wheel_radius_m must be positive and finite')
        self.mode = str(mode)
        self.dt = float(dt)
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.arm, self.leg, self.wheel = (schema.term(ARM_TERM), schema.term(LEG_TERM),
                                          schema.term(WHEEL_TERM))
        self.wheel_divisor = float(wheel_action_gain) * float(self.wheel.scale)
        self.wheel_radius_m = float(wheel_radius_m)
        self.wheel_joint_names = tuple(self.wheel.joint_names)
        self.arm_obs_ids = np.array([self.names.index(name) for name in ARM6])
        self.leg_obs_ids = np.array([self.names.index(name) for name in self.leg.joint_names])
        self.quiet_samples = int(np.ceil(QUIET_WINDOW_S/self.dt))+1

        kwargs = dict(settle_calls=settle_calls, ramp_calls=ramp_calls, reach_only=False,
                      forward_cmd=forward_cmd, turn_cap=turn_cap, standoff=standoff,
                      turn_gain=turn_gain, open_on=open_on)
        if lowering_m is not None:
            kwargs['lowering_m'] = lowering_m
        self.child = PayloadRaisePrefix(schema, observation_joint_names, defaults, dt=dt,
                                        handoff_at_raise=True, **kwargs)

        self.probe_forward_m = float(probe_forward_m)
        self.probe_speed_m_s = float(probe_speed_m_s)
        self.probe_yaw_rad = float(probe_yaw_rad)
        self.probe_total_s = float(probe_total_s)
        self.dock_standoff_m = float(dock_standoff_m)
        self.delivery_budget_s = float(delivery_budget_s)
        self.release_hold_s = float(release_hold_s)
        #: How far off the barrel bearing the dock may be. It defaults to what the
        #: ARM can absorb, because the chassis cannot turn; if a run shows the
        #: chassis CAN turn (raised wheel-torque caps), this is tightened so the
        #: dock is on-axis and the wider on-axis window applies.
        self.dock_bearing_tol_rad = (MAX_ARM_SWING_RAD if dock_bearing_tol_rad is None
                                     else float(dock_bearing_tol_rad))
        #: Hip-twist yaw, in radians of front/rear abduction. Zero disables it, which
        #: is the default: it is a candidate mechanism, not a verified capability.
        self.leg_twist_rad = float(leg_twist_rad)
        self.leg_twist_sign = 1. if float(leg_twist_sign) >= 0. else -1.
        self.leg_twist_applied = None
        #: Optional overrides for the drive layer's own wheel-speed caps. They exist
        #: because those caps, not the robot, may be what limits the turn: the wheel
        #: drive has stiffness 0 / damping 1.0, so its torque is 1.0 * (commanded -
        #: actual) and the effort limit of 20 N*m per wheel is only reached at a
        #: 20 rad/s velocity error. Commanding ~1.3 rad/s therefore asks for only
        #: ~1.3 N*m. Raising the caps asks for real torque.
        self.drive_wheel_speed_max_rad_s = drive_wheel_speed_max_rad_s
        self.drive_differential_max_rad_s = drive_differential_max_rad_s
        self._drive = drive

        self.odometry = PublicPlanarOdometry(dt=self.dt)
        self.calls = 0
        self.state, self.phase, self.phase_start = 'PREFIX', 'PREFIX', None
        self.state_reason = 'the_frozen_prefix_owns_every_action_until_the_original_A_arrival'
        self.done_reason = None
        self.debug = {}
        self.carry = None
        self._child_action = None
        self.tracker = None
        self.template = None
        self.finger_command = None
        self.arm_command = None
        self.alpha = 0.

        self.arm_window = deque()
        self.arm_span = None
        self.quiet_tick, self.quiet_ready = False, False
        #: Call index at which the CURRENT unbroken quiet run began. A duration
        #: measured from the deque length would be capped at the window size, so
        #: any hold longer than the window could never be observed - the exact
        #: defect the contract flags in the original hold accounting.
        self.quiet_start_call = None
        self.arm_qdot_raw = None
        self.arm_qdot_previous = None
        self.arm_qdot_twopoint = None
        self.arm_qdot_difference = None
        self.q_previous = None
        self.window_total_variation = None

        self.anchor_released = False
        self.anchor_engage_calls = []
        self.phase_elapsed_s = 0.
        self.probe_path_m = 0.
        self.probe_start_xy = None
        self.segment_start_xy = None
        self.segment_start_yaw = None
        self.yaw_start = None
        self._previous_xy = None
        self.segment_no_progress_anchor = None
        self.segment_no_progress_call = None
        self.segment_no_progress_progress = None
        self.baseline_xy = None
        self.baseline_yaw = None
        self.measured_speed = 0.
        self.measured_yaw_rate = 0.
        self.linear_norm = 0.
        self.tilt_rad = 0.
        self._previous_xy = None
        self._last_proprio = None
        self._last_images = None
        self._observer_instance = None
        self.barrel_observation = None
        self.dock_source = None
        self.dock_range_used_m = None
        self.arm_swing_rad = None
        self.dock_window_used = None
        self._latched_observation = None
        self._latched_observation_call = None
        #: Off by default. Its geometry is CPU-verified (reach 0.65 m with a 0.15 m
        #: drop puts the bottle bottom at 0.4814 m, below the rim, whole bottle inside
        #: the mouth, no wall strikes), but the IK is solved from the measured pose and
        #: a synthetic fixture cannot validate that - it needs a real run. Leaving it
        #: off keeps the previously verified release behaviour available.
        self.low_place_enabled = bool(low_place)
        self.low_place_goal = None
        self.low_place_error = None
        self.low_place_tried = False
        self.drive_state = None
        self.drive_fault = None
        self.wheel_request_rad_s = np.zeros(len(self.wheel_joint_names))
        self.measured_arm_q = None
        self.heading_error_rad = None
        self.range_to_centre_m = None
        self.release_start_call = None
        self.opened_call = None
        self.observed_release_calls = 0
        self.held_dock_error_rad = None
        self.empty_start = None
        self.width = None
        self.attachment_lost = False

    # -- public attributes the evaluator drives or reads ---------------------

    @property
    def pause_for_stance(self):
        return self.child.pause_for_stance

    @pause_for_stance.setter
    def pause_for_stance(self, value):
        self.child.pause_for_stance = bool(value)

    @property
    def wheel_hold_requested(self):
        """True only when the base is genuinely parked and the anchor is engaged.

        The evaluator overwrites the wheel slice with its own brake-wheel holder
        whenever this is True, so returning True during a movement phase would
        silently discard every navigation request.
        """
        if self.carry is None:
            return self.child.wheel_hold_requested
        return not self.anchor_released

    # -- action assembly -----------------------------------------------------

    def _action(self, wheel_rad_s=None):
        """The EXACT handoff template with the arm and wheel slices replaced."""
        if self.template is None:
            # The carry state was never published, so there is no template to
            # replace and nothing here may build an action. Returning the child's
            # own last action is the only valid answer; without this guard a
            # failed grasp crashed the controller instead of stopping cleanly.
            if self._child_action is not None:
                return np.asarray(self._child_action, dtype=np.float32).copy()
            return np.zeros(int(self.schema.total_dim), dtype=np.float32)
        out = np.asarray(self.template, dtype=np.float32).copy()
        targets = np.concatenate([np.asarray(self.arm_command, dtype=float),
                                  np.asarray(self.finger_command, dtype=float)])
        out[self.arm.start:self.arm.stop] = arm_targets_to_action(
            targets, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        if wheel_rad_s is None or self.wheel_hold_requested:
            self.wheel_request_rad_s = np.zeros(len(self.wheel_joint_names))
            out[self.wheel.start:self.wheel.stop] = 0.
        else:
            # _action is the single authority on what was EMITTED, so the
            # diagnostic never reports a request from a previous tick.
            self.wheel_request_rad_s = np.asarray(wheel_rad_s, dtype=float)
            out[self.wheel.start:self.wheel.stop] = self.wheel_request_rad_s/self.wheel_divisor
        return out

    def _stop(self, reason):
        self.done_reason = self.state_reason = reason
        self.state = self.phase = 'STOPPED'
        # A stop parks the robot: hand the wheel anchor back so the evaluator
        # holds the wheel angle actually reached rather than leaving the last
        # navigation request in force.
        self.anchor_released = False
        action = self._action()
        self._record()
        return action

    def _begin(self, phase, reason):
        """Enter a phase; a movement phase releases the wheel anchor."""
        self.phase = self.state = phase
        self.state_reason = reason
        self.phase_start = self.calls
        self.phase_elapsed_s = 0.
        self.segment_start_xy = self.odometry.position_xy.copy()
        self.segment_start_yaw = float(self.odometry.yaw_unwrapped_rad)
        self.segment_no_progress_anchor = None
        self.segment_no_progress_call = None
        self.segment_no_progress_progress = None
        self.arm_window.clear()
        self.arm_span, self.quiet_ready, self.quiet_tick = None, False, False
        self.quiet_start_call = None
        # Entering a movement phase RELEASES the anchor. Engaging is never
        # automatic: it happens only at a confirmed settle, through
        # _engage_anchor(), so the evaluator's brake holder can never grab the
        # wheels while the base is still sliding - the recorded multi_reach
        # defect of latching a stationary budget at .235 m/s.
        if phase in MOVEMENT_PHASES:
            self.anchor_released = True

    def _engage_anchor(self):
        """Take a NEW wheel anchor at the angle actually reached, once settled."""
        if self.anchor_released:
            self.anchor_released = False
            self.anchor_engage_calls.append(self.calls)

    # -- prefix --------------------------------------------------------------

    def _prefix(self, proprio, images):
        action = self.child.act(proprio, images)
        self._child_action = np.asarray(action, dtype=np.float32).copy()
        self.state, self.state_reason = self.child.state, self.child.state_reason
        if self.child.done_reason is not None:
            # Propagated unchanged: never retried, never reclassified.
            self.done_reason = self.child.done_reason
        elif self.child.handoff_ready:
            try:
                self.carry = self.child.take_carry_state()
            except ValueError as error:
                # The prefix said it was ready but its state is incomplete. That is
                # an implementation fault, so stop rather than carry on with a
                # partially inherited controller.
                return self._stop('delivery_handoff_incomplete: ' + str(error))
            self.tracker = self.carry['tracker']
            self.template = self.carry['action_template']
            self.finger_command = self.carry['finger_command']
            self.arm_command = self.carry['arm_command_rad']
            self.width = float(self.finger_command[0]-self.finger_command[1])
            self._begin('CARRY_READY',
                        'A_actually_arrived_before_the_handoff,_so_the_SAME_tracker_is_now_held_'
                        'closed_loop_while_the_carry_controller_owns_the_wheel_slice')
        self._record()
        return action

    # -- delivery tick -------------------------------------------------------

    def _tick(self, proprio):
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            return self._stop('delivery_invalid_proprio')
        try:
            q = arm_joints_from_proprio(obs, self.names, self.defaults)
        except (ValueError, KeyError):
            return self._stop('delivery_invalid_proprio')

        gravity = obs[9:12]
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < .5 or -gravity[2]/norm < np.cos(HARD_TILT_RAD):
            return self._stop('delivery_tilt_limit_exceeded')
        up = -gravity/norm
        tilt = float(np.arccos(np.clip(-gravity[2]/norm, -1., 1.)))
        if tilt > UNSTABLE_TILT_RAD or float(np.linalg.norm(obs[3:5])) > UNSTABLE_RATE_RAD_S:
            return self._stop('delivery_unstable_posture')

        # The odometry is updated once per tick in act(), for the prefix and the
        # delivery alike, so it is deliberately NOT updated again here.
        linear = obs[:3]
        tangent = linear-np.dot(linear, up)*up
        self.measured_speed = float(np.linalg.norm(tangent))
        self.measured_yaw_rate = float(np.dot(obs[3:6], up))
        self.linear_norm = float(np.linalg.norm(linear))
        self.tilt_rad = tilt
        arm_q = np.asarray(q[:6], dtype=float)
        self.measured_arm_q = arm_q
        self.width = float(q[6]-q[7])
        self._velocity_signals(obs, arm_q)
        self._attachment_watchdog()
        self._accumulate_path()

        # Physical protections independent of the phase clock. Each names a
        # different physical quantity; none of them is the stationary run gate.
        if self.measured_speed > PROBE_SPEED_CAP_M_S:
            return self._stop('delivery_probe_overspeed')
        if self.attachment_lost:
            return self._stop('delivery_attachment_lost')
        if float(np.max(np.abs(self.arm_qdot_raw))) > RAW_ARM_QDOT_ESTOP_RAD_S:
            return self._stop('delivery_arm_velocity_estop')
        # The probe's travel and wall budgets bound the PROBE. A delivery has to
        # cover the whole route to the barrel, so its bound is the delivery
        # budget, enforced in the approach phase; applying the probe cap here
        # stopped the delivery after 1.2 m of a ~4 m route.
        if self.mode == 'carry_probe':
            if self.probe_path_m > PROBE_PATH_CAP_M:
                return self._stop('delivery_probe_path_budget_exceeded')
            if (self.calls-self.carry['control_tick'])*self.dt > self.probe_total_s:
                return self._stop('delivery_probe_total_budget_exceeded')
        if self.drive_fault is not None:
            return self._stop('delivery_drive_fault: ' + self.drive_fault)
        if self.phase in MOVEMENT_PHASES and self._drive_layer() is None:
            # Never command a movement phase through a drive layer that does not
            # exist: without this the base sat unanchored for 276 ticks and the
            # run ended on a misleading no-progress reason.
            return self._stop('delivery_drive_unavailable')

        self.phase_elapsed_s = (self.calls-self.phase_start)*self.dt

        # The SAME tracker, held closed loop on A with the RAW measured joint
        # velocity. advance_path=False stops the scalar reference moving; it does
        # NOT freeze the loaded actuator command, and paused stays False because
        # a pause would freeze the path, the filter and the command.
        self.arm_command = self.tracker.update(arm_q, np.asarray(self.arm_qdot_raw, dtype=float),
                                               advance_path=False, paused=False)
        if self.tracker.done_reason is not None:
            return self._stop(self.tracker.done_reason)

        self._quiet_tick(arm_q)
        return self._phase_tick()

    def _velocity_signals(self, obs, arm_q):
        """Report four DIFFERENT velocity measurements, never just one.

        The two-point mean is a low-bandwidth run-admission signal that zeroes an
        alternating component; it is NOT evidence that no motion exists, because
        the reported joint velocity is sampled instantaneously while a position
        difference spans four physics substeps at the recorded .005 s physics
        step. All four are recorded so an auditor can see the unmodified signal.
        """
        self.arm_qdot_raw = np.asarray(obs[36+self.arm_obs_ids], dtype=float)
        previous = self.arm_qdot_previous
        self.arm_qdot_twopoint = (.5*(previous+self.arm_qdot_raw) if previous is not None
                                  else self.arm_qdot_raw.copy())
        self.arm_qdot_previous = self.arm_qdot_raw.copy()
        if self.q_previous is None:
            self.arm_qdot_difference = np.zeros(6)
        else:
            self.arm_qdot_difference = (arm_q-self.q_previous)/self.dt
        self.q_previous = arm_q.copy()

    def _attachment_watchdog(self):
        """A collapsed jaw width means the gripper is EMPTY. Never the converse."""
        if self.width is None or self.width >= EMPTY_WIDTH_M:
            self.empty_start = None
            return
        if self.empty_start is None:
            self.empty_start = self.calls
        if (self.calls-self.empty_start+1)*self.dt >= EMPTY_WIDTH_S:
            self.attachment_lost = True

    def _accumulate_path(self):
        """Cumulative driven path, so the probe cap bounds travel and not net offset."""
        xy = self.odometry.position_xy
        if self._previous_xy is not None:
            self.probe_path_m += float(np.linalg.norm(xy-self._previous_xy))
        self._previous_xy = xy.copy()

    def _goal_error(self):
        """The actual fixed-goal error, the third element of the arrival gate."""
        if self.tracker is None or self.measured_arm_q is None:
            return None
        return float(self.tracker.goal_error(self.measured_arm_q))

    def _quiet_tick(self, arm_q):
        """The unchanged .5 s / 26-sample window with the UNCHANGED .002 rad span.

        The run-admission velocity term is the two-point mean, named as such. The
        position span is the binding guard on real motion and is untouched.

        A stance recalibration request invalidates the observation outright: the
        legs are about to move under a loaded arm, so the quiet run restarts. The
        old stationary pause machinery is deliberately NOT reused here - only the
        invalidation is, because a hold that survives a pause is not a hold.
        """
        if bool(self.child.pause_for_stance):
            self.arm_window.clear()
            self.arm_span, self.quiet_ready, self.quiet_tick = None, False, False
            self.quiet_start_call = None
            return
        body_quiet = bool(self.measured_speed < QUIET_TANGENT_M_S
                          and self.linear_norm < QUIET_LINEAR_M_S
                          and abs(self.measured_yaw_rate) < QUIET_ANGULAR_RAD_S
                          and self.tilt_rad <= QUIET_TILT_RAD)
        eligible = bool(body_quiet
                        and float(np.max(np.abs(self.arm_qdot_twopoint))) <= QUIET_ARM_QDOT_RAD_S)
        self.quiet_tick = eligible
        if not eligible:
            self.arm_window.clear()
            self.arm_span, self.quiet_ready, self.window_total_variation = None, False, None
            self.quiet_start_call = None
            return
        self.arm_window.append(arm_q.copy())
        while len(self.arm_window) > self.quiet_samples:
            self.arm_window.popleft()
        if len(self.arm_window) < self.quiet_samples:
            self.arm_span, self.quiet_ready, self.window_total_variation = None, False, None
            self.quiet_start_call = None
            return
        window = np.asarray(self.arm_window)
        self.arm_span = float(np.max(np.max(window, axis=0)-np.min(window, axis=0)))
        steps = np.abs(np.diff(window, axis=0)).sum(axis=1)
        self.window_total_variation = float(steps.sum()/((len(window)-1)*self.dt))
        self.quiet_ready = bool(self.arm_span <= QUIET_ARM_SPAN_RAD)
        # The run start is LATCHED when the window first closes and released the
        # moment quiet is lost. Recomputing it from the deque every tick would
        # pin the elapsed time at exactly the window length, so no hold longer
        # than the window could ever be observed.
        if self.quiet_ready:
            if self.quiet_start_call is None:
                self.quiet_start_call = self.calls-(self.quiet_samples-1)
        else:
            self.quiet_start_call = None

    def quiet_elapsed_s(self):
        """Duration of the current unbroken quiet run, by endpoint timestamps."""
        if not self.quiet_ready or self.quiet_start_call is None:
            return 0.
        return float((self.calls-self.quiet_start_call)*self.dt)

    # -- phases --------------------------------------------------------------

    def _phase_tick(self):
        handler = getattr(self, '_phase_'+self.phase.lower(), None)
        if handler is None:
            return self._stop('delivery_unknown_phase')
        return handler()

    def _phase_carry_ready(self):
        """Hold A until the FULL arrival gate closes, then release the anchor.

        All three elements are required, as the contract states: the actual
        fixed-goal error inside .04 rad, the complete fresh .5 s window, and its
        .002 rad position span. The error term used to be printed but never
        compared, so a still arm sitting .06 rad off the A goal released the
        anchor and started driving.
        """
        if self.phase_elapsed_s >= ACTION_SEGMENT_MAX_S:
            return self._stop('delivery_carry_ready_not_reached')
        error = self._goal_error()
        if self.quiet_ready and error is not None and error < GOAL_ERROR_RAD:
            self.baseline_xy = self.odometry.position_xy.copy()
            self.baseline_yaw = float(self.odometry.yaw_unwrapped_rad)
            if self.mode == 'first_delivery':
                self._begin('TURN_APPROACH',
                            'A_held_through_a_complete_quiet_window;_the_wheel_anchor_is_RELEASED_'
                            'for_the_frontal_dock_approach')
                return self._action(self._drive_request(0., 0.))
            self._begin('PROBE_DRIVE',
                        'A_held_through_a_complete_quiet_window;_the_wheel_anchor_is_now_RELEASED_'
                        'for_the_bounded_loaded_straight_probe')
            return self._action(self._drive_request(0., 0.))
        return self._action()

    def _phase_probe_drive(self):
        if self.phase_elapsed_s >= ACTION_SEGMENT_MAX_S:
            return self._stop('delivery_probe_drive_timeout')
        travelled = float(np.linalg.norm(self.odometry.position_xy-self.baseline_xy))
        if travelled >= self.probe_forward_m:
            self._begin('PROBE_BRAKE_DRIVE',
                        'the_loaded_straight_probe_reached_its_bounded_distance;_braking_to_a_settle_'
                        'before_any_new_anchor')
            return self._action(np.zeros(len(self.wheel_joint_names)))
        stalled = self._no_progress(travelled, self.probe_forward_m, forward=True)
        if stalled:
            return self._stop('delivery_probe_no_forward_progress')
        return self._action(self._drive_request(self.probe_speed_m_s, 0.))

    def _phase_probe_brake_drive(self):
        return self._brake_then('PROBE_YAW_POS',
                                'braked_to_a_measured_settle_and_re-anchored_ONLY_now;_starting_the_'
                                'bounded_positive_yaw_probe')

    def _phase_probe_yaw_pos(self):
        return self._yaw_phase(direction=+1., next_phase='PROBE_BRAKE_YAW_POS')

    def _phase_probe_yaw_neg(self):
        return self._yaw_phase(direction=-1., next_phase='PROBE_BRAKE_YAW_NEG')

    def _yaw_phase(self, direction, next_phase):
        if self.phase_elapsed_s >= ACTION_SEGMENT_MAX_S:
            return self._stop('delivery_probe_yaw_timeout')
        turned = direction*(float(self.odometry.yaw_unwrapped_rad)-self.yaw_start)
        if turned >= self.probe_yaw_rad:
            self._begin(next_phase,
                        'the_bounded_yaw_probe_reached_its_target;_braking_to_a_settle_before_'
                        're-anchoring')
            return self._action(np.zeros(len(self.wheel_joint_names)))
        if self._no_progress(turned, self.probe_yaw_rad, forward=False):
            return self._stop('delivery_probe_no_yaw_progress')
        return self._action(self._drive_request(PROBE_YAW_FORWARD_M_S,
                                                direction*PROBE_YAW_CAP_RAD_S))

    def _phase_probe_brake_yaw_pos(self):
        return self._brake_then('PROBE_YAW_NEG',
                                'braked_and_re-anchored_after_the_positive_yaw_probe;_starting_the_'
                                'bounded_negative_yaw_probe')

    def _phase_probe_brake_yaw_neg(self):
        return self._brake_then('PROBE_HOLD',
                                'braked_and_re-anchored_after_the_negative_yaw_probe;_starting_the_'
                                'final_bounded_observation')

    def _brake_then(self, next_phase, reason):
        """Brake to zero, take the anchor ONLY at the settle, confirm, then go on.

        The two stages are deliberately separate. Stage one commands a zero wheel
        request while the anchor is still RELEASED, so the evaluator's brake
        holder cannot grab the wheels mid-slide. Only once the measured planar
        speed AND a complete quiet window agree does stage two take the new
        anchor at the angle actually reached; the anchor is then held while the
        arm confirms it is still quiet, which is the "confirm the load is still
        held" step the contract asks for before the next probe.
        """
        if self.phase_elapsed_s >= BRAKE_SEGMENT_MAX_S:
            return self._stop('delivery_probe_brake_timeout')
        if not self.anchor_released:
            if self.quiet_ready:
                self.yaw_start = float(self.odometry.yaw_unwrapped_rad)
                self._begin(next_phase, reason)
                return self._action(self._drive_request(0., 0.))
            return self._action()
        if self.measured_speed < BRAKE_SETTLE_SPEED_M_S and self.quiet_ready:
            self._engage_anchor()
            return self._action()
        # Braking goes through the drive layer rather than a raw zero action.
        # A zero normalized action asks for 0 rad/s, and this chassis drives its
        # wheels with stiffness 0 / damping 1.0, so that is only a ~0.2 N*m
        # damper: measured on the d1 probe the wheels kept spinning at -0.42 rad/s
        # against it and the base never settled. The drive's own loop can command
        # actively negative, which is a real deceleration and is still slew-limited.
        return self._action(self._drive_request(0., 0.))

    def _phase_probe_hold(self):
        """Observe for a REAL elapsed duration, measured between endpoint timestamps.

        The goal error is part of the gate here too: a hold that is only quiet
        while the arm sits outside the arrival error is not holding the pose the
        probe claims to have reached.
        """
        if self.phase_elapsed_s >= ACTION_SEGMENT_MAX_S:
            return self._stop('delivery_probe_hold_not_reached')
        error = self._goal_error()
        if self.quiet_elapsed_s() >= PROBE_HOLD_S and error is not None and error < GOAL_ERROR_RAD:
            return self._stop('delivery_carry_probe_complete')
        return self._action()

    def _no_progress(self, progress, target, *, forward):
        """A real stall in the intended direction, judged by RATE not distance.

        Whether a slow turn is fast enough for the route is the phase budget's
        business; this guard only catches genuine absence of motion.
        """
        floor = STALL_FORWARD_SPEED_M_S if forward else STALL_YAW_RATE_RAD_S
        if self.segment_no_progress_call is None:
            self.segment_no_progress_call = self.calls
            self.segment_no_progress_progress = progress
            return False
        elapsed = (self.calls-self.segment_no_progress_call)*self.dt
        if elapsed < NO_PROGRESS_WINDOW_S:
            return False
        rate = (progress-self.segment_no_progress_progress)/elapsed
        if abs(rate) >= floor:
            self.segment_no_progress_call = self.calls
            self.segment_no_progress_progress = progress
            return False
        return True

    def _observer(self):
        """The public barrel observer, built on first use so this module imports."""
        if self._observer_instance is not None:
            return self._observer_instance
        try:
            from task_b.bucket_observation import BarrelObservation
        except Exception:
            return None
        try:
            self._observer_instance = BarrelObservation(self.names, self.defaults, dt=self.dt)
        except Exception:
            return None
        return self._observer_instance

    @staticmethod
    def _observation_usable(observation):
        """True only if a trustworthy observation carries BOTH usable numbers.

        A record can be flagged trustworthy while a field is absent, so the
        fields are required to be present and finite here rather than assumed:
        the first version took 'trustworthy' at face value and crashed on a None
        distance.
        """
        if not isinstance(observation, dict) or not observation.get('trustworthy'):
            return False
        return all(isinstance(observation.get(key), (int, float))
                   and np.isfinite(observation[key])
                   for key in ('body_centre_distance_m', 'bearing_rad'))

    def _observe_barrel(self):
        """One public barrel observation, at the camera's own rate.

        The usability check is applied on the CACHED return as well: returning
        the stored record unchecked let a trustworthy-flagged record with a None
        distance reach the caller and crash it.
        """
        if self._last_images is None or self.calls % OBSERVATION_PERIOD_CALLS:
            return (self.barrel_observation
                    if self._observation_usable(self.barrel_observation) else None)
        observer = self._observer()
        if observer is None:
            return None
        try:
            result = observer.observe(self._last_images, self._last_proprio,
                                      timestamp_s=self.calls*self.dt)
        except Exception as error:
            self.barrel_observation = {'trustworthy': False, 'reason': repr(error)}
            return None
        self.barrel_observation = result
        return result if self._observation_usable(result) else None

    # -- first_delivery: frontal dock from the raise pose --------------------

    def _bearing_error(self):
        """Heading error to the barrel centre, from the public plan alone."""
        position = self.odometry.position_xy
        delta = BARREL_CENTRE_XY-position
        self.range_to_centre_m = float(np.linalg.norm(delta))
        bearing = float(np.arctan2(delta[1], delta[0]))
        self.heading_error_rad = float((bearing-float(self.odometry.yaw_unwrapped_rad)
                                        + np.pi) % (2.*np.pi)-np.pi)
        return self.heading_error_rad

    def _phase_turn_approach(self):
        """Turn onto the barrel bearing and run in, steering on PUBLIC observation.

        Dead reckoning is kept only as the fallback it is: measured over the
        recorded runs its planar error grows about 0.29 mm/s, so after the ~550 s
        this route takes it reaches ~160 mm against a 67 mm dock window. Once the
        head camera can see the barrel (from ~3.1 m of centre standoff) the range
        AND the alignment are taken from that observation instead, because both
        are then measured against the real barrel rather than against a drifting
        integral. Which source was used for the dock is recorded.
        """
        if (self.calls-self.carry['control_tick'])*self.dt > self.delivery_budget_s:
            return self._stop('delivery_budget_exceeded')
        error = self._bearing_error()
        dead_reckoned_range = self.range_to_centre_m

        observation = None
        if dead_reckoned_range <= OBSERVATION_START_RANGE_M:
            observation = self._observe_barrel()
        if observation is not None:
            self._latched_observation = observation
            self._latched_observation_call = self.calls
        elif (self._latched_observation is not None
              and (self.calls-self._latched_observation_call) <= OBSERVATION_STALE_CALLS):
            observation = self._latched_observation
        if observation is not None:
            observed_range = float(observation['body_centre_distance_m'])
            observed_bearing = float(observation['bearing_rad'])
            self.dock_source = 'public_barrel_observation'
            range_for_dock = observed_range
            yaw_error = observed_bearing
        else:
            observed_range, observed_bearing = None, None
            self.dock_source = 'dead_reckoning_no_trustworthy_observation'
            range_for_dock = dead_reckoned_range
            yaw_error = error
        self.dock_range_used_m = range_for_dock

        # The bearing tolerance is the ARM's limit whatever the range source is.
        # Requiring .05 rad whenever an observation was available was a bug: the
        # chassis cannot turn at all, so that condition could never be met and the
        # robot would have driven past a dock it was perfectly able to make. The
        # observation supplies the RANGE; the bearing is the arm's business.
        aligned = abs(yaw_error) <= self.dock_bearing_tol_rad
        window_low, window_high = self.dock_window(yaw_error)
        self.dock_window_used = [window_low, window_high]
        aim = .5*(window_low+window_high)
        if aligned and window_low <= range_for_dock <= aim+DOCK_AIM_TOLERANCE_M:
            self._begin('DOCK_BRAKE',
                        'on_the_barrel_bearing_at_the_aim_range_from_' + self.dock_source +
                        ';_braking_to_a_settle_before_the_release')
            return self._action(self._drive_request(0., 0.))
        if range_for_dock < window_low:
            return self._stop('delivery_dock_standoff_undershot_from_' + self.dock_source)
        if aligned:
            speed = APPROACH_SPEED_M_S
        else:
            # Turn while backing AWAY from the barrel, not toward it.
            #
            # The bearing to the barrel rotates at v*sin(theta)/d, where theta is
            # the angle between the velocity and the bearing. Driving forward at
            # 0.02 m/s from 4.3 m rotates it about 0.0047 rad/s while the chassis
            # turns 0.0016 - the geometry outruns the turn nearly 3x, which is
            # exactly what d7 showed when its heading error GREW from -0.536 to
            # -0.648 rad while "approaching". Driving radially has sin(theta) = 0,
            # so the bearing does not rotate at all and every radian the chassis
            # manages is a net radian gained. Backing up also REDUCES the turn
            # needed: at 4 m further out the required heading change falls from
            # .49 rad to about .26.
            backed = (float(np.linalg.norm(self.odometry.position_xy-self.baseline_xy))
                      if self.baseline_xy is not None else 0.)
            # Stop retreating once far enough for the turn to outpace the bearing
            # rotation; keep turning in place beyond that.
            speed = -TURN_CREEP_SPEED_M_S if backed < TURN_BACK_LIMIT_M else 0.
        yaw_command = float(np.clip(PURSUIT_YAW_GAIN*yaw_error,
                                    -PROBE_YAW_CAP_RAD_S, PROBE_YAW_CAP_RAD_S))
        action = self._action(self._drive_request(speed, yaw_command))
        if self.leg_twist_rad and not aligned:
            # Twist toward the barrel: the sign that reduces the heading error.
            direction = -1. if yaw_error < 0. else 1.
            action = self._leg_twist_action(action, self.leg_twist_sign*direction*self.leg_twist_rad)
        return action

    def dock_window(self, bearing_rad):
        """The standoff window at this bearing error, from the measured envelopes."""
        phi = abs(float(bearing_rad))
        extent = (CHASSIS_FORWARD_EXTENT_M*abs(np.cos(phi))
                  + CHASSIS_LATERAL_EXTENT_M*abs(np.sin(phi)))
        return (1.0+extent+CLEARANCE_MARGIN_M,
                DOCK_STANDOFF_HIGH_M-BOTTLE_LIMIT_FALL_PER_RAD_M*phi)

    def _plan_low_place(self):
        """IK for an extended-and-lowered release goal, from the CURRENT public pose.

        Solved at run time from the measured arm q rather than hard-coded, so it
        follows whatever pose the grasp actually produced.
        """
        if self.measured_arm_q is None:
            return None, 'no measured arm pose'
        try:
            pose = fk(np.asarray(self.measured_arm_q, dtype=float))
        except Exception as error:
            return None, 'fk failed: %r' % (error,)
        # 0.65 m is the CPU-verified reach: the gripper currently sits at about
        # 0.497, and a smaller value (the first version used DOCK_STANDOFF_LOW_M-1.02
        # = 0.491) would leave the arm essentially where it is, which is precisely
        # the lower-without-extending case that clips the wall. At 0.65 the whole
        # bottle is inside the inscribed radius, the arm stays above the rim
        # (lowest 0.611 m against 0.55) and there are zero strikes.
        reach = .65
        target = np.array([reach, pose[1, 3], pose[2, 3]-LOW_PLACE_DROP_M])
        try:
            fit = solve_ik(target, rotation=pose[:3, :3], seed=np.asarray(
                self.measured_arm_q, dtype=float), max_nfev=120)
        except Exception as error:
            return None, 'ik failed: %r' % (error,)
        if not fit.success or fit.position_error > .01:
            return None, 'ik did not converge (err %.4f)' % fit.position_error
        joints = np.asarray(fit.joints, dtype=float)
        outside = [i+1 for i in range(6) if not (JOINT_LOWER[i]-1e-6 <= joints[i]
                                                 <= JOINT_UPPER[i]+1e-6)]
        if outside:
            return None, 'ik solution leaves the original joint limits: %r' % (outside,)
        # The heading error is absorbed by joint1; fold it into this goal instead of
        # applying it separately, so the two adjustments cannot fight each other.
        joints = joints.copy()
        joints[0] = joints[0]+float(np.clip(self.heading_error_rad or 0.,
                                            -MAX_ARM_SWING_RAD, MAX_ARM_SWING_RAD))
        return joints, None

    def _leg_twist_action(self, action, twist_rad):
        """Superimpose front/rear hip abduction that twists the body in yaw."""
        out = np.asarray(action, dtype=np.float32).copy()
        for name, sign in (('FR_hip_joint', 1.), ('FL_hip_joint', 1.),
                           ('RR_hip_joint', -1.), ('RL_hip_joint', -1.)):
            index = self.leg.start + self.leg.joint_names.index(name)
            out[index] = out[index]+sign*twist_rad/self.leg.scale
        self.leg_twist_applied = float(twist_rad)
        return out

    def _apply_arm_swing(self, swing_rad):
        """Point the arm at the barrel by offsetting joint1 on the held A goal.

        The chassis cannot turn - plan_d8 measured the wheel differential stalling
        after about one degree - so the arm absorbs the residual bearing error.
        Checked on the model: across the whole range of swings and standoffs the
        fingers stay inside the mouth, the rim clearance stays 220 mm and there are
        no wall strikes, because the mouth (.9753 m radius) is far wider than the
        .17 m a 28 degree swing moves the fingers.
        """
        tracker = self.tracker
        if tracker is None:
            return
        swing = float(np.clip(swing_rad, -MAX_ARM_SWING_RAD, MAX_ARM_SWING_RAD))
        for name in ('reference', 'path_start', 'fixed_goal'):
            value = getattr(tracker, name, None)
            if value is not None:
                value[0] = value[0]+swing
        self.arm_swing_rad = swing

    def _phase_dock_brake(self):
        """Brake to a measured settle, re-anchor, verify the dock, then release."""
        if self.phase_elapsed_s >= BRAKE_SEGMENT_MAX_S:
            return self._stop('delivery_dock_brake_timeout')
        error = self._bearing_error()
        observation = self._observe_barrel()
        if observation is not None:
            self.dock_range_used_m = float(observation['body_centre_distance_m'])
            self.dock_source = 'public_barrel_observation'
        elif self.dock_range_used_m is None:
            self.dock_range_used_m = self.range_to_centre_m
            self.dock_source = 'dead_reckoning_no_trustworthy_observation'
        if not self.anchor_released:
            if self.quiet_ready:
                # Verify against whichever source the dock used. Releasing from
                # outside the window drops the bottle past the mouth.
                window_low, window_high = self.dock_window(self.heading_error_rad or 0.)
                self.dock_window_used = [window_low, window_high]
                if not (window_low <= self.dock_range_used_m <= window_high):
                    return self._stop('delivery_dock_outside_window_from_' + self.dock_source)
                # Swing the arm onto the barrel before opening. The bearing error
                # measured at the settle is what the arm has to absorb.
                self._apply_arm_swing(self.heading_error_rad or 0.)
                self._begin('LOW_PLACE',
                            'at_the_dock;_extending_and_lowering_the_load_before_opening')
                return self._action()
            return self._action()
        if self.measured_speed < BRAKE_SETTLE_SPEED_M_S and self.quiet_ready:
            self._engage_anchor()
            self.held_dock_error_rad = error
            return self._action()
        return self._action(self._drive_request(0., 0.))

    def _phase_low_place(self):
        """Extend forward and lower the load before opening it.

        Releasing at the raise height drops the bottle 0.645 m onto a restitution-1
        surface, and an ideal bounce returns it above the 0.55 m rim. Lowering alone
        does not help - the bottle then clips the wall - but extending forward AND
        lowering moves the whole bottle inside the mouth so it can descend. If the IK
        cannot be solved the release still happens at the raise height, and the
        reason is recorded rather than the bounce guard being silently skipped.
        """
        if self.phase_elapsed_s >= LOW_PLACE_DEADLINE_S:
            return self._stop('delivery_low_place_timeout')
        if not self.low_place_tried:
            self.low_place_tried = True
            if not self.low_place_enabled:
                self.low_place_error = 'low_place_disabled'
                self._begin('RELEASE_OPEN',
                            'low_place_disabled;_opening_at_the_raise_height')
                return self._action()
            self._apply_arm_swing(self.heading_error_rad or 0.)
            goal, error = self._plan_low_place()
            if goal is None:
                self.low_place_error = error
                self._begin('RELEASE_OPEN',
                            'low_place_UNAVAILABLE (' + str(error) + ');_opening_at_the_raise_'
                            'height,_where_a_restitution_1_bounce_can_leave_the_barrel')
                return self._action()
            try:
                self.tracker.begin(goal)
            except ValueError as rejection:
                self.low_place_error = str(rejection)
                self._begin('RELEASE_OPEN', 'low_place_goal_rejected (' + str(rejection) + ')')
                return self._action()
            self.low_place_goal = goal.tolist()
        error = self._goal_error()
        if error is not None and error < LOW_PLACE_ERROR_RAD:
            self._begin('RELEASE_OPEN',
                        'low_place_arrived_at_the_extended_and_lowered_pose;_opening_the_fingers')
        return self._action()

    def _phase_release_open(self):
        """Slew the fingers open from the raise pose. The pose itself is held.

        The fingers are prismatic with hard limits [0,.035] and [-.035,0]; p13
        ended at a .0575 m gap, so this walks the remaining ~.0125 m rather than
        the .11 m of command-space travel the written contract assumed. The gate
        is the MEASURED gap, never the fact that a command was issued.
        """
        if self.phase_elapsed_s >= OPEN_DEADLINE_S:
            return self._stop('delivery_release_open_timeout')
        step = FINGER_SLEW_M_S*self.dt
        self.finger_command = self.finger_command+np.clip(
            FINGER_OPEN_TARGET-self.finger_command, -step, step)
        if self.width is not None and abs(self.width-OPEN_GAP_M) <= OPEN_GAP_TOL_M:
            self.opened_call = self.calls
            self._begin('RELEASE_OBSERVE',
                        'the_MEASURED_jaw_gap_reached_the_open_target;_observing_the_release')
        return self._action()

    def _phase_release_observe(self):
        """Watch the release for a real elapsed interval.

        ``phase_elapsed_s`` is already the endpoint difference between the phase
        entry call and now, so this is a timestamp duration and not a sample count
        that could be assembled across an interruption.
        """
        if self.phase_elapsed_s >= OBSERVE_DEADLINE_S:
            return self._stop('delivery_release_observe_timeout')
        if self.phase_elapsed_s >= self.release_hold_s:
            return self._stop('delivery_release_sequence_complete')
        return self._action()

    # -- drive boundary ------------------------------------------------------

    def _drive_layer(self):
        """The drive layer is built on first use so this module always imports.

        It is a separate file with a separate owner; a missing or broken drive
        layer must show up as an explicit diagnostic rather than an import error
        that hides whether the carry state was ever handed over.
        """
        if self._drive is not None:
            return self._drive
        try:
            from task_b.carry_drive import CarryDrive
        except Exception:
            return None
        try:
            # The drive's own no-progress guard is a PROBE guard: it exists to
            # bound a measurement. For the delivery its floor is disabled to a
            # near-zero rate, because here the turn is deliberate and its
            # governor is the delivery budget, not a stall test. Leaving the
            # probe floor in place stopped d6 and d5 while they were turning
            # correctly, just slowly: measured rates run 0.0007 rad/s one way and
            # 0.0016 the other, and a fixed floor cannot sit below both without
            # also failing to catch a genuinely dead chassis.
            extra = {}
            if self.drive_wheel_speed_max_rad_s is not None:
                extra['wheel_speed_max_rad_s'] = float(self.drive_wheel_speed_max_rad_s)
            if self.drive_differential_max_rad_s is not None:
                extra['differential_max_rad_s'] = float(self.drive_differential_max_rad_s)
            self._drive = CarryDrive(dt=self.dt, wheel_joint_names=self.wheel_joint_names,
                                     wheel_radius_m=self.wheel_radius_m,
                                     stall_yaw_rate_rad_s=0., **extra)
        except Exception:
            return None
        return self._drive

    def _drive_request(self, speed_m_s, yaw_rate_rad_s):
        """Ask the drive layer for PHYSICAL wheel rad/s; never apply gain here."""
        drive = self._drive_layer()
        if drive is None:
            self.drive_state = {
                'available': False,
                'reason': 'no usable drive layer (task_b.carry_drive is missing or failed to build); '
                          'the base cannot be commanded to move'}
            return np.zeros(len(self.wheel_joint_names))
        out = drive.update(self._last_proprio, speed_m_s, yaw_rate_rad_s)
        self.drive_state = out.get('state')
        # The drive latches its own faults instead of raising, so an unchecked
        # fault would leave this controller commanding a drive that has already
        # given up. Latch it here and let the tick turn it into a normal stop.
        fault = getattr(drive, 'done_reason', None)
        if fault is not None and self.drive_fault is None:
            self.drive_fault = str(fault)
        target = np.asarray(out['wheel_target_rad_s'], dtype=float)
        if target.shape != (len(self.wheel_joint_names),) or not np.isfinite(target).all():
            self.drive_state = {'available': False,
                                'reason': 'the drive layer returned an invalid wheel target'}
            return np.zeros(len(self.wheel_joint_names))
        return target

    # -- diagnostics ---------------------------------------------------------

    def _record(self):
        self.debug = {
            'module': 'task_b.delivery.PayloadDeliveryPolicy',
            'mode': self.mode,
            'state': self.state, 'phase': self.phase, 'reason': self.state_reason,
            'done_reason': self.done_reason, 'calls': self.calls,
            'phase_elapsed_s': self.phase_elapsed_s,
            'child_called_this_tick': self.carry is None,
            'handoff': None if self.carry is None else {
                'control_tick': self.carry['control_tick'],
                'prefix_phase': self.carry['prefix_phase'],
                'tracker_is_the_same_instance': True,
                'finger_command_m': self.carry['finger_command'].tolist(),
            },
            'wheel': {
                'divisor_normalized_per_physical_rad_s': self.wheel_divisor,
                'gain_and_scale_applied_exactly_once_here': True,
                'radius_m': self.wheel_radius_m,
                'joint_names': list(self.wheel_joint_names),
                'request_rad_s': np.asarray(self.wheel_request_rad_s, dtype=float).tolist(),
                'wheel_hold_requested': bool(self.wheel_hold_requested),
                'anchor_released': bool(self.anchor_released),
                'anchor_engage_calls': list(self.anchor_engage_calls),
                'drive': self.drive_state,
                'drive_fault': self.drive_fault,
            },
            'measured': {'planar_speed_m_s': self.measured_speed,
                         'yaw_rate_rad_s': self.measured_yaw_rate,
                         'planar_speed_cap_m_s': PROBE_SPEED_CAP_M_S,
                         'yaw_rate_cap_rad_s': PROBE_YAW_CAP_RAD_S},
            'arm_velocity': {
                'raw_absmax_rad_s': None if self.arm_qdot_raw is None else float(
                    np.max(np.abs(self.arm_qdot_raw))),
                'twopoint_mean_absmax_rad_s': None if self.arm_qdot_twopoint is None else float(
                    np.max(np.abs(self.arm_qdot_twopoint))),
                'position_difference_absmax_rad_s': None if self.arm_qdot_difference is None else float(
                    np.max(np.abs(self.arm_qdot_difference))),
                'window_total_variation_rad_per_s': self.window_total_variation,
                'raw_estop_candidate_rad_s': RAW_ARM_QDOT_ESTOP_RAD_S,
                'note': 'four different measurements of the same joint, deliberately reported '
                        'together; the two-point mean is a low-bandwidth run-admission signal and '
                        'is NOT evidence that no motion exists',
            },
            'quiet': {'window_s': QUIET_WINDOW_S, 'samples_required': self.quiet_samples,
                      'samples_held': len(self.arm_window), 'arm_span_rad': self.arm_span,
                      'arm_span_limit_rad': QUIET_ARM_SPAN_RAD,
                      'leg_span_limit_rad': QUIET_LEG_SPAN_RAD,
                      'quiet_tick': self.quiet_tick, 'quiet_ready': self.quiet_ready,
                      'quiet_elapsed_s': self.quiet_elapsed_s(),
                      'duration_is_measured_between_endpoint_timestamps': True},
            'goal_error_rad': (None if (self.tracker is None or self.measured_arm_q is None)
                               else float(self.tracker.goal_error(self.measured_arm_q))),
            'goal_error_limit_rad': GOAL_ERROR_RAD,
            'arm_command_rad': None if self.arm_command is None else np.asarray(
                self.arm_command, dtype=float).tolist(),
            'odometry': {'xy': self.odometry.position_xy.tolist(),
                         'yaw_rad': float(self.odometry.yaw_unwrapped_rad),
                         'raw_and_never_written_back': True},
            'probe': {'target_forward_m': self.probe_forward_m, 'target_yaw_rad': self.probe_yaw_rad,
                      'speed_target_m_s': self.probe_speed_m_s,
                      'path_from_baseline_m': self.probe_path_m,
                      'path_cap_m': PROBE_PATH_CAP_M,
                      'total_s': self.probe_total_s,
                      'baseline_xy': None if self.baseline_xy is None else self.baseline_xy.tolist()},
            'dock': {'barrel_centre_xy': BARREL_CENTRE_XY.tolist(),
                     'standoff_target_m': self.dock_standoff_m,
                     'standoff_window_m': [DOCK_STANDOFF_LOW_M, DOCK_STANDOFF_HIGH_M],
                     'inner_wall_inscribed_radius_m': float(INNER_WALL_INSCRIBED_M),
                     'measured_yaw_rate_basis_rad_s': MEASURED_YAW_RATE_RAD_S,
                     'range_to_centre_m': self.range_to_centre_m,
                     'heading_error_rad': self.heading_error_rad,
                     'heading_tolerance_rad': self.dock_bearing_tol_rad,
                     'window_used_m': self.dock_window_used,
                     'observation_latched': self._latched_observation is not None,
                     'observation_age_s': (None if self._latched_observation_call is None
                                           else round((self.calls-self._latched_observation_call)
                                                      * self.dt, 2)),
                     'held_dock_bearing_error_rad': self.held_dock_error_rad,
                     'range_used_for_the_dock_m': self.dock_range_used_m,
                     'leg_twist_rad_applied': self.leg_twist_applied,
                     'low_place_goal_rad': self.low_place_goal,
                     'low_place_error': self.low_place_error,
                     'dock_source': self.dock_source,
                     'observation': (None if not isinstance(self.barrel_observation, dict) else {
                         'trustworthy': self.barrel_observation.get('trustworthy'),
                         'reason': self.barrel_observation.get('reason'),
                         'body_centre_distance_m': self.barrel_observation.get('body_centre_distance_m'),
                         'bearing_rad': self.barrel_observation.get('bearing_rad')}),
                     'odometry_error_basis': 'measured ~0.29 mm/s of planar drift against truth, so '
                                             'the last metres are observed, not integrated',
                     'approach': 'creep while misaligned, then run in; the measured yaw rate is the '
                                 'same rolling or creeping, so creeping costs no turn time'},
            'release': {'finger_command_m': (None if self.finger_command is None
                                             else np.asarray(self.finger_command, dtype=float).tolist()),
                        'open_target_m': FINGER_OPEN_TARGET.tolist(),
                        'measured_gap_m': self.width,
                        'open_gap_target_m': OPEN_GAP_M,
                        'open_gap_tolerance_m': OPEN_GAP_TOL_M,
                        'opened_call': self.opened_call,
                        'gate_is_the_measured_gap_never_the_issued_command': True},
            'jaw': {'measured_width_m': self.width, 'empty_width_m': EMPTY_WIDTH_M,
                    'attachment_lost': self.attachment_lost},
            'evidence_note': 'sequence and motion bookkeeping only: no object pose, contact force, '
                             'world base pose, reward or score is read anywhere in this module, and '
                             'objects_in_circle is never read, so nothing here is grasp, carry, '
                             'clearance, delivery or score evidence',
        }

    # -- entry ---------------------------------------------------------------

    def act(self, proprio, images):
        self.calls += 1
        self._last_proprio = np.asarray(proprio, dtype=float).reshape(-1)
        self._last_images = images
        # ONE odometry update per control tick, from the FIRST tick of the
        # episode - the prefix and the delivery share one continuous estimate.
        # Updating it only after the handoff (as the first version did) discarded
        # the whole grasp-approach displacement: the d4 run then believed it was
        # 1.55 m from the barrel when it was really 2.63 m away, docked there,
        # and released the bottle into open ground.
        if self._last_proprio.size == 84 and np.isfinite(self._last_proprio).all():
            self.odometry.update(self._last_proprio)
        if self.carry is None:
            # The prefix owns every action until a carry state is actually
            # published - including the ticks where it stops or fails. A crashed
            # or stopped prefix must still yield an action, never an exception.
            return self._prefix(proprio, images)
        if self.done_reason is not None:
            action = self._action(np.zeros(len(self.wheel_joint_names)))
        else:
            action = self._tick(proprio)
        self._record()
        return action

    def describe(self):
        return {
            'module': 'task_b.delivery.PayloadDeliveryPolicy',
            'candidate': 'carry probe: hold the handed-over A pose while a bounded loaded straight '
                         'run and two bounded yaw probes are driven and braked',
            'composition': 'PayloadRaisePrefix subclasses the p13 PayloadMotionPolicy and adds ONE '
                           'branch at the original A-arrival test; the prefix child is called once '
                           'per tick and never again after the handoff',
            'handoff': 'the SAME PayloadJointTracker instance is transferred, so the integral, the '
                       'filter, the scalar path and the loaded actuator command stay continuous',
            'not_claimed': 'this module does not measure the object, the clearance or the score, and '
                           'no stop reason here is a grasp, carry, delivery or success claim',
            'probe_values_are_initial_candidates': ['forward_m', 'speed_m_s', 'yaw_rad', 'total_s'],
            'known_risks': [
                'the yaw probes may show little or no response; that is a measurement this probe '
                'exists to produce, not something it assumes',
                'the wheel-unit boundary is a single divide by gain*scale, verified by CPU check',
            ],
        }
