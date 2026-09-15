"""Bounded loaded arm raise, side swing and side extension after a contact grasp.

Composition only. :class:`~task_b.contact_grasp.ContactGraspPolicy` is built with
the caller's own parameters, called exactly once per tick while it runs, and its
actions and every one of its stop reasons pass through unchanged. The payload
sequence opens at ONE of two child states, selected by ``open_on`` and recorded
in ``debug``: the contract default ``observation_complete`` waits for the child's
own normal reason ``grasp_probe_observation_complete``; ``close_complete`` takes
over at the child's own normal internal CLOSE->LIFT progression instead, for the
measured reason documented at :data:`PAYLOAD_ENTRY_POINTS`. Neither entry ever
converts a stop reason into a success. Once the payload opens the child is never
called again, so its 25 s contact and 12 s probe clocks can no longer fire.
``contact_grasp.py``, ``grasp_probe.py`` and ``first_reach.py`` are read-only here.

What this module is NOT. The child's normal completion is a sequence-completion
reason, not proof that the original bottle is held: it means the inherited probe
closed, requested its small paired lift and finished its bounded observation.
Nothing in this file measures the object. There is no object pose, no contact
force, no net force, no world base pose, no reward and no score in any decision:
the only runtime inputs are the public 84-value proprio vector and the RGB-D the
frozen first reach already owns. ``payload_motion_observation_complete`` claims
that the three planned arm segments and the final bounded hold completed, and
nothing more. It is not a grasp, carry, transport, delivery, clearance or score
claim, and a retained jaw width never certifies attachment.

Geometry provenance. The waypoint formulas below are static: the body-mounted
Piper FK/IK and the original shared joint limits, driven by the child's own
PUBLIC measured ``lift_start_q``. The private static clearance study of an
earlier contact pose (bottle minimum z about .6307 m over a .55 m rim, root
radius about .8512 m, about .0744 m body/wheel wall clearance at 1.45 m from the
bucket centre, under a RIGID ATTACHMENT ASSUMPTION) is planning evidence only:
it is not imported, not a runtime constant, and it does not transfer to the new
deeper +pi/6 wrist contact without an independent recheck.

The reusable pieces are deliberately small and public so a later delivery
wrapper can reuse them: :func:`build_payload_goals` is a pure static planner and
:class:`PayloadJointTracker` is a single-goal loaded arm tracker whose
``advance_path=False, paused=False`` mode keeps the load-compensated closed loop
running on the SAME reference while that wrapper handles its own separately
audited navigation.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from task_b.arm_kinematics import ARM_JOINT_NAMES, arm_joints_from_proprio, arm_targets_to_action, fk, solve_ik
from task_b.contact_grasp import CONTACT_LOWERING_M, ContactGraspPolicy
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

#: The contract entry: the one child REASON that opens the payload sequence.
#: Every other reason, normal-looking or not, is propagated unchanged, never
#: retried and never reclassified as a success.
NORMAL_CONTACT_REASON = 'grasp_probe_observation_complete'

#: Where the payload sequence may take over from the child.
#:
#: ``'observation_complete'`` is the audited contract default: wait for the
#: child's own normal reason.
#:
#: ``'close_complete'`` is a DELIBERATE, EVIDENCE-BASED alternative, not a
#: reclassification of any failure. In the recorded run
#: ``plan_p5_contact_wrist30_lower04_seed42_01`` the object was independently
#: confirmed lifted and following the gripper, yet the frozen probe still stopped
#: on ``grasp_probe_lift_not_reached``: its ACTUAL goal error had already fallen
#: to .0328 rad, below its own .04 rad threshold, but measured arm |qdot| stayed
#: at .2263 rad/s against a .12 rad/s cap, so its .5 s position window never
#: collected a single sample and 242 of its ~249 lift ticks rode the rate/tether
#: bound. The frozen probe compensates q2 only (gain 3, cap .12, actually
#: saturating near -.0985) and has no q3 term at all, so its arrival gate cannot
#: converge while a real .5 kg load hangs on the wrist. This entry therefore
#: takes over at the child's own NORMAL internal CLOSE->LIFT progression, where
#: the child has latched a stable width window and its measured lift_start_q,
#: and lets the load-compensated tracker below (q2 gain 4 / cap .14 and q3
#: gain 2 / cap .08) own the loaded motion instead. Taking over at a normal
#: internal transition is NOT the same as treating a stop reason as success, and
#: it is not evidence of a grasp: the grasp evidence is external and offline.
PAYLOAD_ENTRY_POINTS = ('observation_complete', 'close_complete')

ARM6 = ARM_JOINT_NAMES[:6]

# -- waypoint formula constants (static, no scene or object input) ------------
RAISE_RADIUS_M = .35            # planar radius of A from the arm mount
RAISE_HEIGHT_M = .45            # A height in body frame
RAISE_PITCH_RAD = -.20          # A tilt about the horizontal normal of d
MOUNT_XY = np.array([.2, 0.])   # body-frame planar position of the arm mount
SWING_JOINT1_RAD = np.pi/2      # B replaces ONLY joint1
EXTEND_POSITION_M = np.array([.2, .52, .40])
EXTEND_PITCH_RAD = -.40         # relative to R0 with the specified yaw, never added to A's
IK_POSITION_TOLERANCE_M = .005
IK_ORIENTATION_TOLERANCE_RAD = .04
DEGENERATE_DIRECTION_M = 1e-6

# -- scalar path and bounded load compensation --------------------------------
PATH_MAX_AXIS_RATE_RAD_S = .10  # sets the scalar segment duration only
COMMAND_RATE_RAD_S = .10        # per-axis actuator command slew
FILTER_TAU_S = .10
#: Low-pass time constant for the MEASURED JOINT VELOCITY used by the damping term.
#:
#: Measured on the recorded runs: the public qdot carries a component at exactly
#: 25 Hz, i.e. the 50 Hz Nyquist rate, with about .068 rad/s of amplitude, while
#: the measured POSITION at that frequency is negligible (.00015 rad). Feeding
#: that Nyquist-rate content into damp*qdot and then through the .002 rad/tick
#: command rate limiter rectifies it into a sustained oscillation: in p6/p7/p8 the
#: q2 1.67 Hz amplitude sits at .0005-.0011 rad while the scalar path is moving,
#: then switches on to .0086-.0098 rad the moment the reference freezes and stays
#: there with no decay. Filtering the velocity before it is used for damping
#: attenuates the Nyquist-rate component by ~6x while keeping ~86% of the damping
#: authority at the 1.67 Hz structural mode, which is the mode that must decay.
DAMPING_VELOCITY_TAU_S = .04
#: Selected-axis load compensation: axis index -> (gain, raw cap, damping).
#: q2 (index 1) and q3 (index 2) only; every other axis adds exactly nothing.
#:
#: The damping term is a MEASURED correction to the original static-torque design.
#: The contract's proportional gains were derived from a static torque bound
#: (q2 about 13.60069 Nm, q3 about 6.56302 Nm at K=80, so .170009/.082038 rad of
#: offset) with no stability margin for the original lightly damped D=4 drives.
#: In the recorded run ``plan_p6_payload_closeentry_seed42_01`` segment A did
#: reach position - the scalar path finished (alpha=1) and the actual fixed-goal
#: error fell to .0226 rad, well inside the unchanged .04 rad threshold, while the
#: object was independently confirmed raised .59 m and following the gripper for
#: 19.0 s - yet measured q2 |qdot| held at .2690 rad/s against the .12 rad/s cap,
#: so the .5 s position window never collected a sample and A expired on its wall
#: deadline. The recorded raw q2 correction (-.0571) was SMALLER in magnitude than
#: its own filtered value (-.1001), i.e. the raw term was already swinging back
#: through zero while the filter lagged: an outer-loop limit cycle, not too little
#: compensation. Bounded derivative damping on the same two axes attacks exactly
#: the quantity that failed. It does NOT touch the original actuators, the .04 rad
#: arrival threshold, the .12 rad/s cap or any other admission rule.
#:
#: The damping magnitudes come from a CPU sweep against a ZERO-LAG position
#: follower (task_b_takeover_20260914/sweep_damping.json), which is the worst
#: case for a one-step discrete derivative and is NOT the original robot. There
#: the loop settles cleanly to a residual |qdot| near 2e-5 rad/s at .05/.03,
#: still completes but rides the command rate limit at .08/.05 and .11/.07, and
#: fails outright at .20/.12 and above; the analytical bound for that plant,
#: filter_weight*damp/dt < 1, is damp < .11. The proportional gains are NOT
#: reduced: the .14 rad cap needs roughly gain 4 for the correction to saturate
#: while the actual error is still inside the unchanged .04 rad threshold, so
#: lowering the gain would break arrival rather than fix the oscillation.
#: Settling in that synthetic plant is a necessary bound, never a guarantee about
#: the real loaded joint.
#:
#: The gains were REDUCED from the contract's 4 and 2, and a gravity feed-forward
#: was added, for a measured reason. The load hangs on a lightly damped joint: with
#: the recorded drive constants K=80 / D=4 the arm plus .5 kg payload sits at a
#: 1.67 Hz resonance with damping ratio only .262 (J about .727 kg m^2), which is
#: exactly the sustained oscillation seen in p6/p7/p8 - it switches on the moment
#: the scalar path finishes, never decays, and measures .019 rad peak to peak in
#: q2 with the carried bottle wobbling about 17 mm. Holding that against gravity
#: with proportional gain alone forces the gain high enough to excite the mode, and
#: the .10 rad/s command slew then turns the loop into a relay that sustains it.
#:
#: CORRECTION, measured after the fact. The feed-forward was introduced expecting it
#: to carry the gravity offset and make a low gain affordable. In the recorded runs
#: it is INERT: ``arm_command - reference`` evaluates to about zero on every axis
#: (p10 and p11 both report a q2 feed-forward of -0.0000), because at the close pose
#: that child hands over, measured arm_joint2 sits pinned on its hard upper limit
#: (3.1400065 against a 3.1399998 limit), so the child's command already equals the
#: measured position and there is no offset left to forward. The feed-forward term is
#: kept because it is correct in principle and costs nothing, but it is NOT what
#: fixed the oscillation and must not be credited for it. What actually tamed the
#: 1.67 Hz mode between p9 and p10 was this gain reduction together with the damping
#: increase below, and what then removed the residual static error so the raise could
#: pass its arrival gate was the slow integral trim, not the feed-forward.
#:
#: Screened on the identified joint model (task_b_takeover_20260914/plant_model.py,
#: a MODEL not the robot), which said gains of 1.5 and below hold the loop still
#: while 2.0 still limit-cycles. That model screen motivated the gain reduction and
#: p9 -> p10 confirmed it on the real robot: at gain 1.25 and damping .15 the measured
#: arm |qdot| fell from .2721 rad/s to .0832 and the first full 26-sample window of
#: the whole campaign formed. No acceptance rule changes here:
#: the .04 rad actual-goal error, the .12 rad/s cap and the complete .5 s
#: .002 rad position window are all untouched, and they are still measured on the
#: real arm, so a wrong feed-forward shows up as a failed arrival and not as a
#: silent pass.
#: Gain and damping were given more margin after ``plan_p11_payload_trim_seed42_01``.
#: There the raise segment PASSED for the first time (26-sample window, arrival, then
#: a residual |qdot| of .0845 rad/s), but the SWING did not: it arrived essentially
#: perfectly - actual goal error .0004 rad - while measured |qdot| held at a constant
#: .151 rad/s against the .12 cap, 1.26x over, for the full 24 s budget. That segment
#: rotates joint1 by about pi/2, which reconfigures the arm, moves the structural mode
#: from 1.67 Hz toward 1.5 Hz and invalidates a feed-forward measured in the close
#: pose, so the same loop that is quiet for the raise is re-excited by the swing. The
#: response is more margin in the two directions that shrink a limit cycle - a lower
#: proportional gain and more damping on the same measured-velocity-filtered term.
FEEDBACK_AXES = {1: (.9, .14, .20), 2: (.9, .08, .20)}

#: Slow integral trim added to the feed-forward, in rad per (rad of error * s).
#:
#: Measured need. In ``plan_p10_payload_ff_seed42_01`` the feed-forward removed
#: the oscillation completely - measured arm |qdot| fell from .1783 rad/s to
#: .0832 (the velocity-sense floor, matching the .070 the identified model
#: predicted) and a full 26-sample window held from step 4400 on - but the actual
#: goal error then sat FLAT at .0461 rad for six seconds, just outside the
#: unchanged .04 rad arrival threshold. That is a feed-forward MISMATCH of about
#: .104 rad: the offset was measured at the close pose, and the raise pose
#: carries a different gravity torque.
#:
#: Raising the proportional gain would close that gap, but the identified model
#: puts the limit-cycle boundary between gain 1.5 and 2.0, and the mismatch needs
#: roughly gain 2.5. The integral trim instead drives the residual to zero at
#: constant loop gain: with gain 1.25 and this rate the error decays with a
#: 2.25 s time constant, which fits inside the segment wall budget. Its gain at
#: the 1.67 Hz mode is only .095 against the 1.25 proportional gain, so it adds
#: almost nothing at the frequency that had to be tamed.
#:
#: Bounded and wound up against saturation: the trim only accumulates while the
#: proportional correction is NOT saturated, is clamped to the cap below, and
#: freezes with everything else during a pause. No acceptance rule changes.
INTEGRAL_GAIN_PER_S = 1.
INTEGRAL_CAP_RAD = .20
#: Per-axis measured-command tether: q2 is widened to .18, the rest keep .10.
TETHER_RAD = np.array([.10, .18, .10, .10, .10, .10])

# -- completion, quiet windows and finite deadlines ---------------------------
GOAL_ERROR_RAD = .04            # fixed actual-goal arrival error, never relaxed
QUIET_ARM_QDOT_RAD_S = .12
#: The arm velocity gate is applied to the MEAN OF TWO CONSECUTIVE public samples.
#:
#: Measured reason, and this is a measurement correction, not a threshold change.
#: In plan_p11 and plan_p12 the swing segment arrived with an actual goal error of
#: .0066 rad and a measured arm POSITION excursion of only .0008 rad - the arm was
#: physically still to within .046 degrees - while the reported arm qdot read a
#: CONSTANT .151-.152 rad/s that alternated sign on EVERY sample:
#:   qdot: -.1511 +.1529 -.1513 +.1526 -.1515 +.1526 ...
#:   q   : 1.645002 1.645806 1.644993 1.645794 1.644958 1.645756 ...
#: Two consecutive samples differ by .0008 rad, which is .041 rad/s, so the reported
#: velocity contradicts the reported position by a factor of 3.7. This is a
#: Nyquist-rate (25 Hz at the official 50 Hz) artifact: the same 25 Hz component was
#: measured systemically across legs and wheels too, so it is a solver/sense artifact
#: rather than arm dynamics. Gain and damping changes do not touch it - lowering the
#: proportional gain from 1.25 to .9 and raising damping from .15 to .20 left the
#: swing artifact at .152, unchanged.
#:
#: Averaging two consecutive samples removes a Nyquist component EXACTLY: an
#: alternating sequence (+a, -a) has mean zero, and the two-point kernel has gain
#: cos(pi*f/fs), which is 0 at 25 Hz and .9945 at the 1.67 Hz structural mode, so real
#: motion passes through essentially untouched. The .12 rad/s threshold, the .5 s
#: window and the .002 rad position span are all UNCHANGED and remain the binding
#: guard on real motion: any genuine movement of .004 rad/s or more still breaks the
#: position window. Both the raw and the de-Nyquisted maxima are reported in debug so
#: an auditor can always see the unmodified signal.
def _denyquist(previous, current):
    """Mean of two consecutive samples: exact zero gain at the Nyquist rate."""
    return .5*(np.asarray(previous, dtype=float)+np.asarray(current, dtype=float))
QUIET_WINDOW_S = .5             # ceil(.5/dt)+1 samples: 26 at the official 50 Hz
QUIET_ARM_SPAN_RAD, QUIET_LEG_SPAN_RAD = .002, .02
QUIET_TANGENT_M_S, QUIET_TILT_RAD = .01, .10
QUIET_LINEAR_M_S, QUIET_ANGULAR_RAD_S = .06, .12
#: Wall deadline per segment = planned scalar travel duration + this slack.
#:
#: Raised from the original 6 s after the recorded run
#: ``plan_p7_payload_damped_seed42_01``. There the damping fix had already done
#: its job - measured q2 |qdot| fell from .2690 rad/s (p6) to .1146 rad/s, below
#: the .12 rad/s cap, the raw correction magnitude (.0656) was once again LARGER
#: than its filtered value (.0422) so the limit cycle was gone, the tick was
#: eligible (quiet_tick true) and the actual fixed-goal error was .0234 rad inside
#: the unchanged .04 rad threshold - but segment A expired at exactly its 20.96 s
#: wall deadline holding only 1 of the 26 window samples it needs. A real .5 kg
#: load simply takes longer than 6 s to decay below the velocity cap.
#:
#: This lengthens a FINITE deadline only. It relaxes no acceptance criterion: the
#: .04 rad actual-goal error, the .12 rad/s hard velocity cap and the complete
#: .5 s / 26-sample position window are all unchanged, the per-segment budget is
#: still fixed when the plan is created and never refreshed on a phase change, and
#: the separate no-progress watchdog still stops a genuine stall.
#: Raised again from 14 s: the swing needed its full 24 s budget to settle at all,
#: and a lower proportional gain settles more slowly. Still a finite wall budget
#: only - the .04 rad error, the .12 rad/s cap and the .002 rad window are untouched.
SEGMENT_SLACK_S = 20.
PROGRESS_WINDOW_S, PROGRESS_RAD = 3., .01
FINAL_HOLD_S, FINAL_HOLD_DEADLINE_S = 2., 5.
TOTAL_SLACK_S = 5.              # total budget = three segment budgets + this

# -- unchanged body safety, budgets and pause response ------------------------
PAUSE_LINEAR_M_S, PAUSE_ANGULAR_RAD_S = .06, .12
PAUSE_QUIET_S, PAUSE_EPISODE_S, PAUSE_TOTAL_S = .2, 2., 6.
HARD_TILT_RAD = .25
UNSTABLE_TILT_RAD, UNSTABLE_RATE_RAD_S = .12, .45
PAYLOAD_BUDGETS = {'displacement_m': .03, 'yaw_rad': .05, 'gravity_rad': .05}
EMPTY_WIDTH_M, EMPTY_WIDTH_S = .006, .2

PAYLOAD_PHASES = ('PAYLOAD_RAISE', 'PAYLOAD_SWING', 'PAYLOAD_EXTEND', 'PAYLOAD_HOLD')
SEGMENT_GOAL_KEY = {'PAYLOAD_RAISE': 'A', 'PAYLOAD_SWING': 'B', 'PAYLOAD_EXTEND': 'C'}
SEGMENT_DEADLINE_REASON = {'PAYLOAD_RAISE': 'payload_raise_not_reached',
                           'PAYLOAD_SWING': 'payload_swing_not_reached',
                           'PAYLOAD_EXTEND': 'payload_extend_not_reached'}
SEGMENT_STALL_REASON = {'PAYLOAD_RAISE': 'payload_raise_no_progress',
                        'PAYLOAD_SWING': 'payload_swing_no_progress',
                        'PAYLOAD_EXTEND': 'payload_extend_no_progress'}

PAYLOAD_STOP_REASONS = {
    'payload_motion_observation_complete': 'NORMAL end of this experiment: the three planned arm '
                                           f'segments each met the fixed {GOAL_ERROR_RAD} rad actual-goal '
                                           f'error with a complete {QUIET_WINDOW_S} s position window, and '
                                           f'the final loaded hold observed {FINAL_HOLD_S} s of valid quiet. '
                                           'It is a SEQUENCE claim only: no object pose, height, carry, '
                                           'clearance, delivery or score is measured anywhere in this module',
    'payload_plan_invalid': 'the static waypoint plan built from the child public measured lift_start_q '
                            'failed its own geometry, joint-limit or IK-accuracy validation, so the '
                            'payload sequence stopped BEFORE any new motion and no fallback posture was '
                            'substituted',
    'payload_handoff_state_incomplete': 'the child reported its normal reason without the public '
                                        'lift_start_q, lift_goal and arm_command the payload reference '
                                        'requires, so nothing was planned or commanded',
    'payload_command_bounds_infeasible': f'the {COMMAND_RATE_RAD_S} rad/s command slew, the per-axis '
                                        'measured tether and the original hard joint limits had no common '
                                        'value on some axis, so the previous command is held instead of '
                                        'clipped sequentially or shifted',
    'payload_raise_not_reached': 'segment A did not form its fixed-goal arrival evidence inside its own '
                                 f'planned duration plus {SEGMENT_SLACK_S} s, pauses included',
    'payload_swing_not_reached': 'segment B did not form its fixed-goal arrival evidence inside its own '
                                 f'planned duration plus {SEGMENT_SLACK_S} s, pauses included',
    'payload_extend_not_reached': 'segment C did not form its fixed-goal arrival evidence inside its own '
                                  f'planned duration plus {SEGMENT_SLACK_S} s, pauses included',
    'payload_raise_no_progress': f'segment A left more than {GOAL_ERROR_RAD} rad of actual fixed-goal error '
                                 f'and improved it by less than {PROGRESS_RAD} rad over {PROGRESS_WINDOW_S} s '
                                 'of active tracking',
    'payload_swing_no_progress': f'segment B left more than {GOAL_ERROR_RAD} rad of actual fixed-goal error '
                                 f'and improved it by less than {PROGRESS_RAD} rad over {PROGRESS_WINDOW_S} s '
                                 'of active tracking',
    'payload_extend_no_progress': f'segment C left more than {GOAL_ERROR_RAD} rad of actual fixed-goal error '
                                  f'and improved it by less than {PROGRESS_RAD} rad over {PROGRESS_WINDOW_S} s '
                                  'of active tracking',
    'payload_hold_not_reached': f'the final loaded hold did not accumulate {FINAL_HOLD_S} s of valid quiet '
                                f'observation inside its {FINAL_HOLD_DEADLINE_S} s deadline',
    'payload_empty_gripper': f'public jaw width stayed below {EMPTY_WIDTH_M} m for {EMPTY_WIDTH_S} s, which '
                             'indicates an EMPTY gripper; a larger width never certifies the opposite',
    'payload_pause_timeout': f'one public body-motion pause reached {PAUSE_EPISODE_S} s',
    'payload_pause_budget_exceeded': f'cumulative pausing reached {PAUSE_TOTAL_S} s across the payload '
                                     'sequence',
    'payload_base_displacement_budget_exceeded': 'integrated tangential drift since the payload handoff '
                                                 f'passed {PAYLOAD_BUDGETS["displacement_m"]} m',
    'payload_base_yaw_budget_exceeded': 'integrated yaw since the payload handoff passed '
                                        f'{PAYLOAD_BUDGETS["yaw_rad"]} rad',
    'payload_gravity_changed': 'public gravity direction turned more than '
                               f'{PAYLOAD_BUDGETS["gravity_rad"]} rad since the payload handoff',
    'payload_total_budget_exceeded': 'the one finite total payload budget, fixed when the plan was created '
                                     'and never refreshed on a phase change, ran out',
    'payload_stance_recalibration': 'the original stance layer requested a new transition pause after the '
                                    'payload handoff, which this loaded sequence does not attempt to ride out',
    'payload_invalid_proprio': 'public proprio was not 84 finite values, or its arm mapping failed',
    'payload_tilt_limit_exceeded': 'unchanged hard public tilt limit of ' + repr(HARD_TILT_RAD) + ' rad',
    'payload_unstable_posture': f'unchanged instability limits: tilt > {UNSTABLE_TILT_RAD} rad or '
                                f'roll/pitch rate norm > {UNSTABLE_RATE_RAD_S} rad/s',
}


def _rodrigues(axis, angle):
    """Right-handed rotation matrix about a finite non-degenerate axis."""
    vector = np.asarray(axis, dtype=float).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        raise ValueError('rotation axis must be three finite values')
    norm = float(np.linalg.norm(vector))
    if norm < DEGENERATE_DIRECTION_M:
        raise ValueError('rotation axis is degenerate')
    if not np.isfinite(angle):
        raise ValueError('rotation angle must be finite')
    return Rotation.from_rotvec(float(angle)*vector/norm).as_matrix()


def _orientation_error_rad(achieved, desired):
    """Geodesic angle between two rotation matrices, in radians."""
    return float(np.linalg.norm(Rotation.from_matrix(
        np.asarray(achieved, dtype=float) @ np.asarray(desired, dtype=float).T).as_rotvec()))


def _six(value, name):
    """Accept a finite six- or eight-axis arm coordinate and slice to six."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size not in (6, 8) or not np.isfinite(array).all():
        raise ValueError(name + ' must be six or eight finite arm coordinates')
    return array[:6].copy()


def build_payload_goals(reference_q, current_q):
    """Plan the three fixed loaded arm goals from PUBLIC measured coordinates.

    ``reference_q`` is the child's public measured ``lift_start_q``; ``current_q``
    is the current public measured arm q, used only as an IK seed. This is a pure
    static function: it reads the body-mounted Piper FK/IK and the original shared
    joint limits, mutates no policy or simulator, and takes no object, world or
    score input. The new +pi/6 contact wrist angle enters automatically through
    ``reference_q``; no example joint vector from any private report is used.

    Returns ``A``, ``B``, ``C`` as six-axis arrays plus a ``diagnostics`` mapping
    of the actual IK errors and every formula parameter. Raises ``ValueError`` on
    a degenerate direction, a joint-limit violation or an IK accuracy failure; the
    owner catches that and stops before commanding any new motion.
    """
    reference = _six(reference_q, 'reference_q')
    seed = _six(current_q, 'current_q')
    pose0 = fk(reference)
    rotation0 = pose0[:3, :3]
    planar = pose0[:2, 3]-MOUNT_XY
    span = float(np.linalg.norm(planar))
    if span < DEGENERATE_DIRECTION_M:
        raise ValueError('the reference gripper sits on the arm mount axis, so the raise direction '
                         'd is degenerate and no payload waypoint is defined')
    direction = planar/span
    # A: raise on the SAME planar bearing before any side swing.
    position_a = np.array([MOUNT_XY[0]+RAISE_RADIUS_M*direction[0],
                           RAISE_RADIUS_M*direction[1], RAISE_HEIGHT_M])
    axis_a = np.array([-direction[1], direction[0], 0.])
    rotation_a = _rodrigues(axis_a, RAISE_PITCH_RAD) @ rotation0
    fit_a = solve_ik(position_a, rotation_a, seed=seed, max_nfev=180)
    goal_a = np.asarray(fit_a.joints, dtype=float).reshape(-1)[:6].copy()
    # B: copy A and replace ONLY joint1. The other five stay exactly at A.
    goal_b = goal_a.copy()
    goal_b[0] = SWING_JOINT1_RAD
    # C: the extension pitch is relative to R0 under the specified yaw, never
    # stacked on top of A's, and never applied about a gripper-local axis.
    rotation_c = (_rodrigues([-1., 0., 0.], EXTEND_PITCH_RAD)
                  @ _rodrigues([0., 0., 1.], np.pi/2-float(reference[0])) @ rotation0)
    fit_c = solve_ik(EXTEND_POSITION_M, rotation_c, seed=goal_b, max_nfev=180)
    goal_c = np.asarray(fit_c.joints, dtype=float).reshape(-1)[:6].copy()

    goals = {'A': goal_a, 'B': goal_b, 'C': goal_c}
    desired = {'A': (position_a, rotation_a), 'C': (EXTEND_POSITION_M, rotation_c)}
    diagnostics = {
        'reference_q_rad': reference.tolist(),
        'seed_q_rad': seed.tolist(),
        'reference_gripper_position_m': pose0[:3, 3].tolist(),
        'raise_direction_d': direction.tolist(),
        'raise_axis_nA': axis_a.tolist(),
        'formula': {'A_position_m': position_a.tolist(), 'A_pitch_rad': RAISE_PITCH_RAD,
                    'A_radius_m': RAISE_RADIUS_M, 'A_height_m': RAISE_HEIGHT_M,
                    'mount_xy_m': MOUNT_XY.tolist(),
                    'B_joint1_rad': SWING_JOINT1_RAD,
                    'B_rule': 'copy qA and replace ONLY joint1; the other five stay at qA',
                    'C_position_m': EXTEND_POSITION_M.tolist(), 'C_pitch_rad': EXTEND_PITCH_RAD,
                    'C_yaw_rad': float(np.pi/2-float(reference[0])),
                    'C_rule': 'Rot([-1,0,0],-.40) @ Rz(pi/2-reference_q[0]) @ R0, body-fixed LEFT '
                              'multiplication; the -.40 is relative to R0 and is NOT added to A'},
        'ik': {}, 'goals_rad': {key: value.tolist() for key, value in goals.items()},
        'tolerances': {'position_m': IK_POSITION_TOLERANCE_M,
                       'orientation_rad': IK_ORIENTATION_TOLERANCE_RAD},
        'static_only': True,
        'is_grasp_or_clearance_evidence': False,
        'note': 'static body-mounted FK/IK on the public measured reference; no object pose, world '
                'base pose, contact force, reward or score takes part, and no clearance is certified',
    }
    for key, fit in (('A', fit_a), ('C', fit_c)):
        achieved = fk(goals[key])
        position, rotation = desired[key]
        diagnostics['ik'][key] = {
            'solver_position_error_m': float(fit.position_error),
            'solver_orientation_error_rad': float(fit.orientation_error),
            'solver_success': bool(fit.success),
            'recomputed_position_error_m': float(np.linalg.norm(achieved[:3, 3]-position)),
            'recomputed_orientation_error_rad': _orientation_error_rad(achieved[:3, :3], rotation),
        }
    for key in ('A', 'B', 'C'):
        achieved = fk(goals[key])
        diagnostics['goal_gripper_position_m'] = diagnostics.get('goal_gripper_position_m', {})
        diagnostics['goal_gripper_position_m'][key] = achieved[:3, 3].tolist()

    # Validate everything AFTER the diagnostics exist, so a caught failure is
    # still fully explained, and raise instead of substituting any posture.
    violations = {}
    for key, goal in goals.items():
        if not np.isfinite(goal).all():
            violations[key] = 'non-finite goal'
            continue
        outside = [ARM6[index] for index in range(6)
                   if not (JOINT_LOWER[index]-1e-9 <= goal[index] <= JOINT_UPPER[index]+1e-9)]
        if outside:
            violations[key] = {'outside_original_joint_limits': outside,
                               'goal_rad': goal.tolist(),
                               'lower_rad': JOINT_LOWER[:6].tolist(),
                               'upper_rad': JOINT_UPPER[:6].tolist()}
    for key in ('A', 'C'):
        entry = diagnostics['ik'][key]
        if entry['recomputed_position_error_m'] > IK_POSITION_TOLERANCE_M:
            violations.setdefault(key, {})
            violations[key] = {'ik_position_error_m': entry['recomputed_position_error_m'],
                               'limit_m': IK_POSITION_TOLERANCE_M}
        elif entry['recomputed_orientation_error_rad'] > IK_ORIENTATION_TOLERANCE_RAD:
            violations[key] = {'ik_orientation_error_rad': entry['recomputed_orientation_error_rad'],
                               'limit_rad': IK_ORIENTATION_TOLERANCE_RAD}
    diagnostics['violations'] = violations
    if violations:
        raise ValueError('invalid payload waypoint plan: ' + repr(violations))
    return {'A': goal_a, 'B': goal_b, 'C': goal_c, 'diagnostics': diagnostics}


class PayloadJointTracker:
    """One fixed six-axis goal, one scalar path and bounded load compensation.

    Owns exactly two things: scalar interpolation of the desired-actual reference
    toward ONE fixed goal, and the selected-axis proportional load correction that
    keeps a loaded arm near that reference. It does not own body safety, fingers,
    wheels, legs, navigation, phase ordering or any success claim, and it never
    reads an object, world pose, force, reward or score.
    """

    def __init__(self, dt, initial_reference, initial_command, feedforward=None):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.:
            raise ValueError('dt must be one finite positive value')
        self.reference = _six(initial_reference, 'initial_reference')
        self.command = _six(initial_command, 'initial_command')
        self.static_lower, self.static_upper = JOINT_LOWER[:6].copy(), JOINT_UPPER[:6].copy()
        if np.any(self.static_lower > self.static_upper):
            raise ValueError('empty static joint window for the arm')
        if feedforward is None:
            self.feedforward = np.zeros(6)
        else:
            self.feedforward = _six(feedforward, 'feedforward').copy()
        self.tether = TETHER_RAD.copy()
        self.gains = np.zeros(6)
        self.caps = np.zeros(6)
        self.damping = np.zeros(6)
        for index, (gain, cap, damp) in FEEDBACK_AXES.items():
            self.gains[index], self.caps[index], self.damping[index] = gain, cap, damp
        self.filter_weight = float(1.-np.exp(-self.dt/FILTER_TAU_S))
        self.damping_velocity_weight = float(1.-np.exp(-self.dt/DAMPING_VELOCITY_TAU_S))
        self.filtered_rate = np.zeros(6)
        self.integral = np.zeros(6)
        # Initialise the selected-axis filtered correction from the loaded
        # handoff offset (command - reference), so the holding reference the child
        # had already built up is not dropped to zero at the first payload tick.
        self.raw_correction = np.zeros(6)
        self.filtered_correction = np.zeros(6)
        offset = self.command-self.reference
        if not np.any(self.feedforward):
            # No feed-forward: the correction itself must carry the whole holding
            # offset, so it is seeded from the child's admitted command-reference.
            for index in FEEDBACK_AXES:
                self.filtered_correction[index] = float(np.clip(offset[index], -self.caps[index],
                                                                self.caps[index]))
        # With a feed-forward that offset is already carried, so the correction
        # starts at exactly zero and nothing is counted twice.
        self.handoff_filtered_correction = self.filtered_correction.copy()
        self.fixed_goal, self.path_start = None, None
        self.path_alpha, self.planned_duration_s = 0., 0.
        self.done_reason = None
        self.segments_begun = 0
        self.bounds_violation = None
        self.clamp_reason = None

    # -- segment control ----------------------------------------------------

    def begin(self, goal):
        """Latch ONE fixed six-axis goal and reset only the scalar path state.

        The start of the scalar path is the PREVIOUS desired-actual reference, so
        consecutive segments chain through audited goals rather than through a
        biased measurement. The command and the correction filter are deliberately
        NOT reset: the load compensation is continuous across A, B, C and the hold.
        The owner may only call this after the previous segment actually arrived.
        """
        if self.done_reason is not None:
            raise ValueError('this tracker already stopped: ' + self.done_reason)
        candidate = _six(goal, 'goal')
        outside = [ARM6[index] for index in range(6)
                   if not (self.static_lower[index]-1e-9 <= candidate[index]
                           <= self.static_upper[index]+1e-9)]
        if outside:
            raise ValueError('payload goal leaves the original joint limits: ' + repr(outside))
        start = self.reference.copy()
        travel = float(np.max(np.abs(candidate-start)))
        duration = travel/PATH_MAX_AXIS_RATE_RAD_S
        if not np.isfinite(duration):
            raise ValueError('payload goal produced a non-finite scalar duration')
        # Atomic: nothing above mutated state, so an invalid goal leaves the
        # previous segment exactly as it was.
        self.fixed_goal, self.path_start = candidate, start
        self.planned_duration_s = float(max(duration, self.dt))
        self.path_alpha = 0.
        self.segments_begun += 1
        return self.planned_duration_s

    def update(self, q, qdot, *, advance_path=True, paused=False):
        """One bounded tracking tick; returns the six-axis actuator command.

        ``advance_path=False, paused=False`` is the A-complete holding mode: the
        scalar reference stops moving but the load-compensated closed loop keeps
        running on the SAME reference. That is a path decision only and is never a
        waiver of the owner's body-motion, jaw or budget guards.

        A pause freezes the reference, the correction filter and the command with
        no hidden catch-up. If the rate/tether/limit intersection is ever empty the
        previous command is held and a stable ``done_reason`` is set, which no
        later update can revive.
        """
        if self.done_reason is not None:
            return self.command.copy()
        measured = _six(q, 'q')
        rate = _six(qdot, 'qdot')
        if self.fixed_goal is None:
            raise ValueError('begin(goal) must latch a fixed goal before update()')
        if paused:
            return self.command.copy()
        if advance_path:
            self.path_alpha = float(min(1., self.path_alpha+self.dt/self.planned_duration_s))
            self.reference = self.path_start+self.path_alpha*(self.fixed_goal-self.path_start)
        # One proportional-plus-damping correction against the CURRENT scalar
        # reference, on the selected axes only, then one filter update. Other axes
        # stay at exactly zero. The damping term uses the PUBLIC measured joint
        # velocity and opposes motion, so it can only remove energy from the outer
        # loop; it never widens a bound or shifts a goal.
        # The damping term uses a low-passed measured velocity, so Nyquist-rate
        # solver noise cannot be rectified into the command by the rate limiter.
        # The position path is untouched: the measured position carries no
        # significant content at that frequency.
        self.filtered_rate = self.filtered_rate+self.damping_velocity_weight*(rate-self.filtered_rate)
        proportional = self.gains*(self.reference-measured)
        self.raw_correction = np.clip(proportional-self.damping*self.filtered_rate,
                                      -self.caps, self.caps)
        # Integral trim, accumulated ONLY on an axis that HAS proportional gain
        # and is not saturated. The gain test matters: an axis with zero gain has
        # a zero proportional term, which would otherwise pass the saturation test
        # and silently acquire integral action it was never given.
        unsaturated = (self.gains > 0.) & (np.abs(proportional) <= self.caps)
        self.integral = np.clip(
            self.integral+np.where(unsaturated, INTEGRAL_GAIN_PER_S*(self.reference-measured)*self.dt, 0.),
            -INTEGRAL_CAP_RAD, INTEGRAL_CAP_RAD)
        self.filtered_correction = (self.filtered_correction
                                    + self.filter_weight*(self.raw_correction-self.filtered_correction))
        candidate = self.reference+self.feedforward+self.integral+self.filtered_correction
        slew = COMMAND_RATE_RAD_S*self.dt
        previous_command = self.command.copy()
        low = np.maximum(np.maximum(previous_command-slew, measured-self.tether), self.static_lower)
        high = np.minimum(np.minimum(previous_command+slew, measured+self.tether), self.static_upper)
        empty = [ARM6[index] for index in range(6) if low[index] > high[index]+1e-12]
        if empty:
            self.bounds_violation = {'axes': empty, 'low_rad': low.tolist(), 'high_rad': high.tolist(),
                                     'measured_rad': measured.tolist(),
                                     'previous_command_rad': self.command.tolist()}
            self.done_reason = 'payload_command_bounds_infeasible'
            return self.command.copy()
        self.command = np.clip(candidate, low, high)
        # Which bound actually bound each axis, and how far the candidate was from
        # what was applied. The integral's own saturation test only compares the
        # PROPORTIONAL term with its cap, so it cannot see that the FINAL command
        # was clamped by slew, tether or a hard limit; the independent audit
        # demonstrated windup to the full integral cap under exactly that
        # condition. This records it; the A/B/C integral behaviour is unchanged.
        self.clamp_reason = {
            'per_axis': [
                ('slew_high' if abs(high[index]-(previous_command[index]+slew)) <= 1e-12
                 else 'tether_high' if abs(high[index]-(measured[index]+self.tether[index])) <= 1e-12
                 else 'hard_upper') if self.command[index] >= high[index]-1e-12
                else ('slew_low' if abs(low[index]-(previous_command[index]-slew)) <= 1e-12
                      else 'tether_low' if abs(low[index]-(measured[index]-self.tether[index])) <= 1e-12
                      else 'hard_lower') if self.command[index] <= low[index]+1e-12
                else 'free' for index in range(6)],
            'candidate_minus_applied_max_rad': float(np.max(np.abs(candidate-self.command))),
            'integral_absmax_rad': float(np.max(np.abs(self.integral))),
            'integral_cap_rad': INTEGRAL_CAP_RAD,
        }
        return self.command.copy()

    # -- diagnostics --------------------------------------------------------

    def goal_error(self, q):
        """Max absolute actual error to the FIXED goal; None before begin()."""
        if self.fixed_goal is None:
            return None
        return float(np.max(np.abs(_six(q, 'q')-self.fixed_goal)))

    def describe(self):
        """Serializable copies only; no mutable internal state is exposed."""
        return {
            'unit': 'PayloadJointTracker',
            'applied_command_clamp': self.clamp_reason,
            'owns': 'one fixed six-axis goal, one scalar reference path and the selected-axis '
                    'load correction; NOT body safety, fingers, wheels, legs, navigation or success',
            'dt_s': self.dt,
            'path_max_axis_rate_rad_s': PATH_MAX_AXIS_RATE_RAD_S,
            'command_rate_rad_s': COMMAND_RATE_RAD_S,
            'filter_tau_s': FILTER_TAU_S,
            'filter_weight': self.filter_weight,
            'damping_velocity_tau_s': DAMPING_VELOCITY_TAU_S,
            'damping_velocity_weight': self.damping_velocity_weight,
            'filtered_rate_rad_s': self.filtered_rate.tolist(),
            'feedback_axes': {ARM6[index]: {'gain': gain, 'raw_cap_rad': cap,
                                            'damping_rad_per_rad_s': damp}
                              for index, (gain, cap, damp) in FEEDBACK_AXES.items()},
            'other_axes_added_correction': 0.,
            'feedforward_basis': 'the child own measured holding offset, arm_command - reference: a '
                                 'public quantity already admitted by the child, introduced here as a '
                                 'gravity feed-forward so the proportional term no longer has to carry '
                                 'the whole .5 kg load',
            'damping_basis': 'measured public joint velocity, opposing motion only; added after the '
                             'recorded p6 outer-loop limit cycle and it changes no threshold, no '
                             'original actuator and no admission rule',
            'tether_rad': self.tether.tolist(),
            'static_lower_rad': self.static_lower.tolist(),
            'static_upper_rad': self.static_upper.tolist(),
            'reference_rad': self.reference.tolist(),
            'command_rad': self.command.tolist(),
            'fixed_goal_rad': None if self.fixed_goal is None else self.fixed_goal.tolist(),
            'path_start_rad': None if self.path_start is None else self.path_start.tolist(),
            'path_alpha': self.path_alpha,
            'planned_duration_s': self.planned_duration_s,
            'raw_correction_rad': self.raw_correction.tolist(),
            'filtered_correction_rad': self.filtered_correction.tolist(),
            'feedforward_rad': self.feedforward.tolist(),
            'integral_trim_rad': self.integral.tolist(),
            'integral_gain_per_s': INTEGRAL_GAIN_PER_S,
            'integral_cap_rad': INTEGRAL_CAP_RAD,
            'handoff_filtered_correction_rad': self.handoff_filtered_correction.tolist(),
            'segments_begun': self.segments_begun,
            'bounds_violation': self.bounds_violation,
            'done_reason': self.done_reason,
        }


class PayloadMotionPolicy:
    """Frozen contact-grasp prefix, then one bounded loaded A -> B -> C -> hold.

    The child is called exactly once per tick while it runs and its action is
    returned byte-for-byte, including on the tick it completes normally. After
    that the child is never called again. Public phases are ``PREFIX``,
    ``PAYLOAD_RAISE``, ``PAYLOAD_SWING``, ``PAYLOAD_EXTEND``, ``PAYLOAD_HOLD``
    and ``STOPPED``.
    """

    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, ramp_calls=100, reach_only=False,
                 forward_cmd=.20, turn_cap=.20, standoff=.56, turn_gain=.4,
                 lowering_m=CONTACT_LOWERING_M, open_on='observation_complete'):
        if reach_only:
            raise ValueError('the payload sequence needs the real visual reach; reach_only must be False')
        if open_on not in PAYLOAD_ENTRY_POINTS:
            raise ValueError('open_on must be one of ' + repr(PAYLOAD_ENTRY_POINTS))
        self.open_on = str(open_on)
        self._child = ContactGraspPolicy(schema, observation_joint_names, defaults, dt=dt,
                                         settle_calls=settle_calls, ramp_calls=ramp_calls,
                                         reach_only=False, forward_cmd=forward_cmd,
                                         turn_cap=turn_cap, standoff=standoff,
                                         turn_gain=turn_gain, lowering_m=lowering_m)
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.dt = float(dt)
        self.arm = schema.term(ARM_TERM)
        self.leg = schema.term(LEG_TERM)
        self.wheel = schema.term(WHEEL_TERM)
        self.arm_obs_ids = np.array([self.names.index(name) for name in ARM6])
        self.finger_obs_ids = np.array([self.names.index('arm_joint'+str(i)) for i in (7, 8)])
        self.leg_obs_ids = np.array([self.names.index(name) for name in self.leg.joint_names])
        self.leg_defaults = np.array([self.defaults[name] for name in self.leg.joint_names])
        self.quiet_samples = int(np.ceil(QUIET_WINDOW_S/self.dt))+1

        self.calls, self.alpha = 0, 0.
        self.settle_calls, self.ramp_calls = int(settle_calls), max(int(ramp_calls), 1)
        self.state, self.state_reason = self._child.state, 'contact_grasp_prefix_owned_by_the_frozen_child'
        self.done_reason, self.debug = None, {}
        self.phase, self.phase_start = 'PREFIX', None
        self.entry_call = None
        self.action_template = None       # the EXACT final child action
        self.finger_command = None        # the child's closed finger targets, never reopened
        self.arm_command = None
        self.tracker = None
        self.goals, self.plan_diagnostics, self.plan_error = None, None, None
        self.reference_q = None
        self.payload_reference_basis = None
        self.segment_start, self.segment_deadline_s = None, None
        self.total_deadline_s = None
        self.goal_error, self.arm_qdot, self.width = None, None, None
        self.arm_qdot_previous, self.arm_qdot_denyquist = None, None
        self.arm_window, self.leg_window = deque(), deque()
        self.arm_span, self.leg_span = None, None
        self.quiet_ready, self.quiet_tick, self.body_quiet = False, False, False
        self.quiet_start_call = None
        self.arrival_ready = False
        self.hold_quiet_calls = 0
        self.progress_anchor, self.progress_call = None, None
        self.pause_start, self.pause_calls = None, 0
        self.pause_quiet_calls, self.pause_episodes, self.pause_cause = 0, 0, None
        self.empty_start = None
        self.base_displacement, self.base_yaw = np.zeros(3), 0.
        self.up_anchor, self.gravity_change = None, 0.
        self.motion = None
        self._hold = False

    # -- public attributes the evaluator drives or reads ---------------------

    @property
    def pause_for_stance(self):
        """The frozen child keeps owning the public stance-transition bit."""
        return self._child.pause_for_stance

    @pause_for_stance.setter
    def pause_for_stance(self, value):
        self._child.pause_for_stance = bool(value)

    @property
    def wheel_hold_requested(self):
        """Mirror the child, then hold the SAME anchor for the whole payload run.

        This module is stationary: the anchor is never released or re-anchored,
        and a future navigation wrapper is the only owner allowed to release it.
        """
        return self._hold if self.entry_call is not None else self._child.wheel_hold_requested

    # -- action assembly ----------------------------------------------------

    def _action(self):
        """The EXACT final child action with only the arm slice replaced.

        The leg slice is the child's own final achieved reference, which already
        carries BOTH its .02 m lowering and the contact candidate's extra .02 m;
        it is copied verbatim rather than rebuilt from ``lower_delta``, which
        would silently drop the extra term. Wheels are zeroed for this stationary
        experiment and the fingers keep the child's closed targets.
        """
        out = np.asarray(self.action_template, dtype=np.float32).copy()
        out[self.wheel.start:self.wheel.stop] = 0.
        targets = np.concatenate([self.arm_command, self.finger_command])
        out[self.arm.start:self.arm.stop] = arm_targets_to_action(
            targets, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        return out

    def _stop(self, reason):
        """Request a normal evaluator stop while holding the last command.

        Every reason is an experimental stopping reason: never an official
        termination and never a grasp, carry, delivery or score claim. The fingers
        keep holding their closed targets even here.
        """
        self.done_reason = self.state_reason = reason
        self.state = 'STOPPED'
        self._hold = True
        self._clear_windows()
        action = self._action()
        self._record()
        return action

    def _clear_windows(self):
        """Drop the quiet windows; a failure or phase change never keeps them."""
        self.arm_window.clear()
        self.leg_window.clear()
        self.arm_span, self.leg_span = None, None
        self.quiet_ready, self.arrival_ready = False, False
        self.quiet_start_call = None

    def quiet_elapsed_s(self):
        """Duration of the current unbroken quiet run, by endpoint timestamps.

        A count of accumulated calls cannot substitute for this: it can be built
        up across pauses, and the window deque is bounded, so a duration read
        from its length is capped at the window and can never express a longer
        hold. The run start is latched when the complete window first closes.
        """
        if not self.quiet_ready or self.quiet_start_call is None:
            return 0.
        return float((self.calls-self.quiet_start_call)*self.dt)

    # -- contact-grasp prefix -----------------------------------------------

    def _prefix(self, proprio, images):
        """One and only one child call this step; its action is returned as is."""
        action = self._child.act(proprio, images)
        self.state, self.state_reason = self._child.state, self._child.state_reason
        reason = self._child.done_reason
        if reason is not None:
            if reason != NORMAL_CONTACT_REASON:
                # Propagated unchanged: no retry, reset, alternative target or
                # reclassification of a frozen child safety outcome.
                self.done_reason = reason
            else:
                stopped = self._enter_payload(proprio, action)
                if stopped is not None:
                    self.done_reason = stopped
        elif self.open_on == 'close_complete' and self._close_complete():
            # The child's OWN normal CLOSE -> LIFT progression, with its stable
            # width window and measured lift_start_q already latched. No stop
            # reason exists yet, so nothing is being reclassified.
            stopped = self._enter_payload(proprio, action)
            if stopped is not None:
                self.done_reason = stopped
        self._record()
        return action

    def _close_complete(self):
        """True on the child's normal CLOSE->LIFT transition, never on a stop."""
        child = self._child
        return bool(child.done_reason is None
                    and getattr(child, 'phase', None) == 'PROBE_LIFT'
                    and getattr(child, 'lift_start_q', None) is not None
                    and getattr(child, 'lift_goal', None) is not None
                    and getattr(child, 'arm_command', None) is not None)

    def _enter_payload(self, proprio, action):
        """Latch the child's final state and build the static plan, once.

        Returns a stop reason instead of moving if the handoff state is
        incomplete or the plan fails validation. No new motion is commanded on
        this tick either way: the child's own action is what the caller receives.
        """
        child = self._child
        reference_q = getattr(child, 'lift_start_q', None)
        lift_goal = getattr(child, 'lift_goal', None)
        arm_command = getattr(child, 'arm_command', None)
        if reference_q is None or lift_goal is None or arm_command is None:
            return 'payload_handoff_state_incomplete'
        try:
            reference_q = _six(reference_q, 'child lift_start_q')
            goal_reference = _six(lift_goal, 'child lift_goal')
            command = np.asarray(arm_command, dtype=float).reshape(-1)
            if command.size != 8 or not np.isfinite(command).all():
                return 'payload_handoff_state_incomplete'
        except ValueError:
            return 'payload_handoff_state_incomplete'
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            return 'payload_invalid_proprio'
        try:
            current = arm_joints_from_proprio(obs, self.names, self.defaults)
        except (ValueError, KeyError):
            return 'payload_invalid_proprio'
        # The EXACT final child action, including both lowering terms on the legs.
        self.action_template = np.asarray(action, dtype=np.float32).copy()
        self.finger_command = command[6:].copy()   # closed targets; never reopened here
        self.arm_command = command[:6].copy()
        self.reference_q = reference_q
        self.entry_call = self.phase_start = self.calls
        self._hold = True
        gravity = obs[9:12]
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < .5:
            return 'payload_invalid_proprio'
        # Independent payload anchors, started once here. They are new budgets
        # for a new stage, not a reset of any child failure.
        self.up_anchor = -gravity/norm
        self.base_displacement, self.base_yaw, self.gravity_change = np.zeros(3), 0., 0.
        try:
            plan = build_payload_goals(reference_q, current[:6])
        except ValueError as error:
            self.plan_error = str(error)
            return 'payload_plan_invalid'
        self.goals = {key: plan[key] for key in ('A', 'B', 'C')}
        self.plan_diagnostics = plan['diagnostics']
        # The desired-actual reference is one of the child's OWN admitted
        # quantities, and the previous actuator command is the child's own last
        # arm command, so the load at the handoff is preserved rather than
        # re-derived from a fresh biased measurement.
        #
        # On the contract entry that is the child's completed lift_goal, which
        # already passed its actual-goal admission. On the close_complete entry
        # the child has only just latched that goal and has NOT yet moved to it,
        # so its measured lift_start_q is the honest desired-actual reference
        # there; using lift_goal instead would open with a .2 rad reference step
        # the child never actually achieved.
        reference = goal_reference if self.open_on == 'observation_complete' else reference_q
        self.payload_reference_basis = ('child lift_goal (contract entry)'
                                       if self.open_on == 'observation_complete'
                                       else 'child measured lift_start_q (close_complete entry)')
        # The gravity feed-forward is the child's OWN measured holding offset:
        # the difference between the actuator command it was already using to
        # hold this load and the desired-actual reference itself. It is a public
        # quantity the child admitted, not a new sensor, and it means the
        # proportional term only has to correct the residue rather than carry the
        # whole .5 kg. See FEEDBACK_AXES for why a low gain is required.
        #
        # A useful consequence: on the first payload tick the correction is still
        # zero, so the candidate command is exactly
        # reference + (arm_command - reference) = arm_command. The payload stage
        # therefore starts from the child's own last actuator command with no
        # discontinuity, instead of stepping the command and re-settling.
        self.tracker = PayloadJointTracker(self.dt, reference, self.arm_command,
                                           feedforward=self.arm_command-reference)
        try:
            budget = 0.
            for key in ('A', 'B', 'C'):
                start = self.tracker.reference if key == 'A' else self.goals[chr(ord(key)-1)]
                travel = float(np.max(np.abs(self.goals[key]-_six(start, 'segment start'))))
                budget += max(travel/PATH_MAX_AXIS_RATE_RAD_S, self.dt)+SEGMENT_SLACK_S
            self.total_deadline_s = float(budget+TOTAL_SLACK_S)
            self.tracker.begin(self.goals['A'])
        except ValueError as error:
            self.plan_error = str(error)
            return 'payload_plan_invalid'
        self.phase = self.state = 'PAYLOAD_RAISE'
        self.state_reason = 'raising_the_loaded_arm_before_any_side_swing'
        self.segment_start = self.calls
        self.segment_deadline_s = self.tracker.planned_duration_s+SEGMENT_SLACK_S
        self.progress_anchor, self.progress_call = None, None
        return None

    # -- bounded payload sequence -------------------------------------------

    def _payload_tick(self, proprio):
        """One payload tick: unchanged safety, budgets, pause, then the phase."""
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            return self._stop('payload_invalid_proprio')
        try:
            q = arm_joints_from_proprio(obs, self.names, self.defaults)
        except (ValueError, KeyError):
            return self._stop('payload_invalid_proprio')
        gravity = obs[9:12]
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(norm) or norm < .5 or -gravity[2]/norm < np.cos(HARD_TILT_RAD):
            return self._stop('payload_tilt_limit_exceeded')
        up = -gravity/norm
        tilt = float(np.arccos(np.clip(-gravity[2]/norm, -1., 1.)))
        if tilt > UNSTABLE_TILT_RAD or float(np.linalg.norm(obs[3:5])) > UNSTABLE_RATE_RAD_S:
            return self._stop('payload_unstable_posture')
        linear, angular = obs[:3], obs[3:6]
        self.motion = {'linear_norm': float(np.linalg.norm(linear)),
                       'angular_norm': float(np.linalg.norm(angular)),
                       'tangent_speed_m_s': float(np.linalg.norm(linear-np.dot(linear, up)*up)),
                       'tilt_rad': tilt}
        self.width = float(q[6]-q[7])
        self.arm_qdot = np.asarray(obs[36+self.arm_obs_ids], dtype=float)
        self.arm_qdot_denyquist = _denyquist(self.arm_qdot_previous
                                             if self.arm_qdot_previous is not None else self.arm_qdot,
                                             self.arm_qdot)
        self.arm_qdot_previous = self.arm_qdot.copy()
        leg_q = np.asarray(obs[12+self.leg_obs_ids]+self.leg_defaults, dtype=float)
        exit_reason = self._budgets(obs, up)
        if exit_reason is not None:
            return self._stop(exit_reason)
        if self.total_deadline_s is not None and \
                (self.calls-self.entry_call)*self.dt >= self.total_deadline_s:
            return self._stop('payload_total_budget_exceeded')
        empty = self._empty_watchdog()
        if empty is not None:
            return self._stop(empty)
        if bool(self._child.pause_for_stance):
            # A loaded arm does not ride out a new stance recalibration.
            return self._stop('payload_stance_recalibration')
        paused = self._pause()
        if paused is not None:
            if paused != 'FROZEN':
                return self._stop(paused)
            self._clear_windows()
            self.quiet_tick = False
            self.arm_command = self.tracker.update(q[:6], self.arm_qdot, paused=True)
            self.state_reason = 'bounded_pause_freezing_the_loaded_path_filter_and_command'
            action = self._action()
            self._record()
            return action
        self._quiet_tick(q, leg_q)
        if self.phase == 'PAYLOAD_HOLD':
            return self._hold_tick(q)
        return self._segment_tick(q)

    def _segment_tick(self, q):
        """Drive one fixed goal, then admit arrival on complete fresh evidence."""
        key = SEGMENT_GOAL_KEY[self.phase]
        elapsed = (self.calls-self.segment_start)*self.dt
        if elapsed >= self.segment_deadline_s:
            return self._stop(SEGMENT_DEADLINE_REASON[self.phase])
        self.arm_command = self.tracker.update(q[:6], self.arm_qdot)
        if self.tracker.done_reason is not None:
            return self._stop(self.tracker.done_reason)
        self.goal_error = self.tracker.goal_error(q[:6])
        stalled = self._progress_watchdog()
        if stalled is not None:
            return self._stop(stalled)
        arrived = (self.tracker.path_alpha >= 1.-1e-12 and self.goal_error is not None
                   and self.goal_error < GOAL_ERROR_RAD and self.quiet_ready)
        self.arrival_ready = bool(arrived)
        if arrived:
            return self._advance_segment(key)
        self.state_reason = ('tracking_the_fixed_' + key + '_goal_under_the_scalar_path_and_bounded_load'
                             '_compensation')
        action = self._action()
        self._record()
        return action

    def _advance_segment(self, key):
        """Only an ACTUAL arrival opens the next segment; A always precedes B."""
        if key == 'A':
            nxt, phase, reason = 'B', 'PAYLOAD_SWING', 'swinging_only_joint1_after_the_raise_actually_settled'
        elif key == 'B':
            nxt, phase, reason = 'C', 'PAYLOAD_EXTEND', 'extending_over_the_side_at_height_after_the_swing'
        else:
            self.phase = self.state = 'PAYLOAD_HOLD'
            self.state_reason = 'holding_the_extended_loaded_pose_for_the_bounded_observation'
            self.phase_start = self.segment_start = self.calls
            self.hold_quiet_calls = 0
            self._clear_windows()
            action = self._action()
            self._record()
            return action
        try:
            self.tracker.begin(self.goals[nxt])
        except ValueError as error:
            self.plan_error = str(error)
            return self._stop('payload_plan_invalid')
        self.phase = self.state = phase
        self.state_reason = reason
        self.phase_start = self.segment_start = self.calls
        self.segment_deadline_s = self.tracker.planned_duration_s+SEGMENT_SLACK_S
        self.progress_anchor, self.progress_call = None, None
        self._clear_windows()
        action = self._action()
        self._record()
        return action

    def _hold_tick(self, q):
        """Keep the SAME C closed loop running and count valid quiet observation."""
        if (self.calls-self.phase_start)*self.dt >= FINAL_HOLD_DEADLINE_S:
            return self._stop('payload_hold_not_reached')
        # advance_path is False: the scalar reference stops moving, but the loaded
        # actuator command is NOT frozen just because alpha reached 1.
        self.arm_command = self.tracker.update(q[:6], self.arm_qdot, advance_path=False)
        if self.tracker.done_reason is not None:
            return self._stop(self.tracker.done_reason)
        self.goal_error = self.tracker.goal_error(q[:6])
        # The COMPLETE fresh window is required, not the per-tick eligibility
        # flag: quiet_tick alone would bypass the .002 rad position span. The
        # duration is the endpoint time difference of the current unbroken quiet
        # run, so a pause or a lost window restarts it and cannot be papered over
        # by accumulating calls.
        valid = (self.goal_error is not None and self.goal_error < GOAL_ERROR_RAD
                 and self.quiet_ready)
        self.hold_quiet_calls = self.hold_quiet_calls+1 if valid else 0
        self.state_reason = 'bounded_observation_of_the_held_loaded_extension'
        if self.quiet_elapsed_s() >= FINAL_HOLD_S:
            return self._stop('payload_motion_observation_complete')
        action = self._action()
        self._record()
        return action

    # -- guards, budgets and windows ----------------------------------------

    def _quiet_tick(self, q, leg_q):
        """Maintain ONE complete fresh arm/leg position window per phase."""
        motion = self.motion
        self.body_quiet = bool(motion['tangent_speed_m_s'] < QUIET_TANGENT_M_S
                              and motion['linear_norm'] < QUIET_LINEAR_M_S
                              and motion['angular_norm'] < QUIET_ANGULAR_RAD_S
                              and motion['tilt_rad'] <= QUIET_TILT_RAD)
        eligible = bool(self.body_quiet
                        and float(np.max(np.abs(self.arm_qdot_denyquist))) <= QUIET_ARM_QDOT_RAD_S)
        self.quiet_tick = eligible
        if not eligible:
            self._clear_windows()
            return
        self.arm_window.append(np.asarray(q[:6], dtype=float).copy())
        self.leg_window.append(np.asarray(leg_q, dtype=float).copy())
        while len(self.arm_window) > self.quiet_samples:
            self.arm_window.popleft()
        while len(self.leg_window) > self.quiet_samples:
            self.leg_window.popleft()
        if len(self.arm_window) < self.quiet_samples:
            self.arm_span, self.leg_span, self.quiet_ready = None, None, False
            return
        arm = np.asarray(self.arm_window)
        leg = np.asarray(self.leg_window)
        self.arm_span = float(np.max(np.max(arm, axis=0)-np.min(arm, axis=0)))
        self.leg_span = float(np.max(np.max(leg, axis=0)-np.min(leg, axis=0)))
        self.quiet_ready = bool(self.arm_span <= QUIET_ARM_SPAN_RAD
                                and self.leg_span <= QUIET_LEG_SPAN_RAD)
        if self.quiet_ready:
            if self.quiet_start_call is None:
                self.quiet_start_call = self.calls-(self.quiet_samples-1)
        else:
            self.quiet_start_call = None

    def _progress_watchdog(self):
        """Require real reduction of the ACTUAL fixed-goal error while it is large.

        A legitimate body-rate pause resets this comparison but never the wall
        deadline. A bookkeeping change in the interpolation is not a stall.
        """
        if self.goal_error is None or self.goal_error <= GOAL_ERROR_RAD:
            self.progress_anchor, self.progress_call = None, None
            return None
        if self.progress_anchor is None:
            self.progress_anchor, self.progress_call = self.goal_error, self.calls
            return None
        if (self.calls-self.progress_call)*self.dt < PROGRESS_WINDOW_S:
            return None
        if self.progress_anchor-self.goal_error < PROGRESS_RAD:
            return SEGMENT_STALL_REASON[self.phase]
        self.progress_anchor, self.progress_call = self.goal_error, self.calls
        return None

    def _empty_watchdog(self):
        """A collapsed public jaw width indicates an EMPTY gripper.

        A larger width is NOT the converse: it never certifies attachment. The
        fingers keep their closed targets whatever this returns.
        """
        if self.width is None or self.width >= EMPTY_WIDTH_M:
            self.empty_start = None
            return None
        if self.empty_start is None:
            self.empty_start = self.calls
        if (self.calls-self.empty_start+1)*self.dt < EMPTY_WIDTH_S:
            return None
        return 'payload_empty_gripper'

    def _pause(self):
        """Finite body-rate pause response with its own bounded histories."""
        motion = self.motion
        excited = bool(motion['linear_norm'] >= PAUSE_LINEAR_M_S
                       or motion['angular_norm'] >= PAUSE_ANGULAR_RAD_S)
        if self.pause_start is None:
            if not excited:
                return None
            self.pause_start, self.pause_quiet_calls = self.calls, 0
            self.pause_episodes += 1
            self.pause_cause = ('linear' if motion['linear_norm'] >= PAUSE_LINEAR_M_S else 'angular')
        self.pause_calls += 1
        if (self.calls-self.pause_start+1)*self.dt >= PAUSE_EPISODE_S:
            return 'payload_pause_timeout'
        if self.pause_calls*self.dt >= PAUSE_TOTAL_S:
            return 'payload_pause_budget_exceeded'
        self.pause_quiet_calls = 0 if excited else self.pause_quiet_calls+1
        if self.pause_quiet_calls*self.dt >= PAUSE_QUIET_S:
            self.pause_start, self.pause_quiet_calls = None, 0
            # The progress comparison restarts after a legitimate pause; the wall
            # deadline deliberately does not.
            self.progress_anchor, self.progress_call = None, None
            return None
        return 'FROZEN'

    def _budgets(self, obs, up):
        """Integrate the public twist since the payload handoff, no vertical term."""
        linear, angular = obs[:3], obs[3:6]
        tangential = linear-np.dot(linear, up)*up
        self.base_displacement = self.base_displacement+tangential*self.dt
        self.base_yaw = float(self.base_yaw+float(np.dot(angular, up))*self.dt)
        self.gravity_change = float(np.arccos(np.clip(np.dot(up, self.up_anchor), -1., 1.)))
        if float(np.linalg.norm(self.base_displacement)) > PAYLOAD_BUDGETS['displacement_m']:
            return 'payload_base_displacement_budget_exceeded'
        if abs(self.base_yaw) > PAYLOAD_BUDGETS['yaw_rad']:
            return 'payload_base_yaw_budget_exceeded'
        if self.gravity_change > PAYLOAD_BUDGETS['gravity_rad']:
            return 'payload_gravity_changed'
        return None

    # -- diagnostics --------------------------------------------------------

    def _record(self):
        child = self._child
        tracker = None if self.tracker is None else self.tracker.describe()
        self.debug = {
            'state': self.state, 'reason': self.state_reason, 'done_reason': self.done_reason,
            'phase': self.phase, 'phase_order': list(PAYLOAD_PHASES), 'calls': self.calls,
            'phase_age_s': None if self.phase_start is None else (self.calls-self.phase_start)*self.dt,
            'payload_entry_call': self.entry_call,
            'payload_age_s': None if self.entry_call is None else (self.calls-self.entry_call)*self.dt,
            'child_phase': getattr(child, 'contact_phase', None) or getattr(child, 'phase', None),
            'child_state': child.state, 'child_reason': child.state_reason,
            'child_done_reason': child.done_reason,
            'child_called_this_tick': self.entry_call is None or self.calls == self.entry_call,
            'open_on': self.open_on,
            'payload_reference_basis': self.payload_reference_basis,
            'reference_q_rad': None if self.reference_q is None else self.reference_q.tolist(),
            'plan_diagnostics': self.plan_diagnostics,
            'plan_error': self.plan_error,
            'goals_rad': None if self.goals is None else {k: v.tolist() for k, v in self.goals.items()},
            'tracker': tracker,
            'actual_goal_error_rad': self.goal_error,
            'goal_error_limit_rad': GOAL_ERROR_RAD,
            'arm_command_rad': None if self.arm_command is None else np.asarray(self.arm_command).tolist(),
            'finger_command_m': None if self.finger_command is None else self.finger_command.tolist(),
            'measured_width_m': self.width,
            'empty_width_m': EMPTY_WIDTH_M,
            'arm_qdot_rad_s': None if self.arm_qdot is None else self.arm_qdot.tolist(),
            'arm_qdot_denyquist_rad_s': (None if self.arm_qdot_denyquist is None
                                         else self.arm_qdot_denyquist.tolist()),
            'arm_qdot_raw_absmax_rad_s': (None if self.arm_qdot is None
                                          else float(np.max(np.abs(self.arm_qdot)))),
            'arm_qdot_limit_rad_s': QUIET_ARM_QDOT_RAD_S,
            'arm_qdot_gate_basis': 'the mean of two consecutive public samples; the raw value is '
                                   'reported alongside and the .12 rad/s threshold is unchanged',
            'quiet_window_s': QUIET_WINDOW_S, 'quiet_samples_required': self.quiet_samples,
            'quiet_samples_held': len(self.arm_window),
            'arm_window_span_rad': self.arm_span, 'arm_span_limit_rad': QUIET_ARM_SPAN_RAD,
            'leg_window_span_rad': self.leg_span, 'leg_span_limit_rad': QUIET_LEG_SPAN_RAD,
            'quiet_ready': self.quiet_ready, 'quiet_tick': self.quiet_tick,
            'body_quiet': self.body_quiet, 'arrival_ready': self.arrival_ready,
            'hold_quiet_s': self.hold_quiet_calls*self.dt, 'hold_required_s': FINAL_HOLD_S,
            'hold_quiet_elapsed_s': self.quiet_elapsed_s(),
            'hold_duration_basis': 'endpoint timestamps of the current unbroken quiet run; the '
                                   'call count is kept for continuity but no longer decides',
            'segment_deadline_s': self.segment_deadline_s,
            'segment_age_s': None if self.segment_start is None else (self.calls-self.segment_start)*self.dt,
            'total_deadline_s': self.total_deadline_s,
            'progress_anchor_rad': self.progress_anchor,
            'progress_window_s': PROGRESS_WINDOW_S, 'progress_required_rad': PROGRESS_RAD,
            'pause_s': self.pause_calls*self.dt, 'pause_episodes': self.pause_episodes,
            'pause_cause': self.pause_cause,
            'pause_episode_limit_s': PAUSE_EPISODE_S, 'pause_total_limit_s': PAUSE_TOTAL_S,
            'base_displacement_m': float(np.linalg.norm(self.base_displacement)),
            'base_yaw_rad': self.base_yaw, 'gravity_change_rad': self.gravity_change,
            'payload_budgets': dict(PAYLOAD_BUDGETS),
            'motion': self.motion,
            'wheel_hold_requested': bool(self.wheel_hold_requested),
            'evidence_note': 'sequence bookkeeping only: no object pose, contact force, world base pose, '
                             'reward or score is read anywhere, so nothing here is grasp, carry, '
                             'clearance, delivery or score evidence',
        }

    def act(self, proprio, images):
        """One tick. Exactly one child call while it runs, then none ever again."""
        self.calls += 1
        self.alpha = float(np.clip((self.calls-self.settle_calls)/self.ramp_calls, 0., 1.))
        if self.done_reason is not None:
            # Already stopped: hold the last command without touching the child.
            if self.action_template is None:
                return np.asarray(self._child.act(proprio, images), dtype=np.float32)
            return self._action()
        if self.entry_call is None:
            return self._prefix(proprio, images)
        return self._payload_tick(proprio)

    def describe(self):
        """Static, honest description of this candidate; not a result."""
        return {
            'module': 'task_b.payload_motion.PayloadMotionPolicy',
            'candidate': 'frozen contact-grasp prefix, then a bounded loaded raise, joint1 side swing, '
                         'side extension and a final bounded loaded hold, all while stationary',
            'composition': 'ContactGraspPolicy is composed, not subclassed or copied; it is called '
                           'exactly once per tick while it runs and its action is returned byte-for-byte, '
                           'including on its normal-completion tick. After that it is never called again, '
                           'so its completed 25 s contact and 12 s probe clocks cannot fire in this stage',
            'entry_point': {'selected': self.open_on, 'available': list(PAYLOAD_ENTRY_POINTS),
                            'close_complete_is_not_a_failure_reclassification': True},
            'child_interception': {'opens_payload': NORMAL_CONTACT_REASON,
                                   'every_other_reason': 'propagated unchanged, no retry, no reset, no '
                                                         'alternative target, no reclassification',
                                   'normal_reason_is_not_grasp_proof': True},
            'phases': {
                'PREFIX': 'the frozen child owns every action, wheel-hold bit and stance semantics',
                'PAYLOAD_RAISE': 'scalar path to the fixed A goal with bounded load compensation; A must '
                                 'ACTUALLY settle before any side swing',
                'PAYLOAD_SWING': 'the same tracker driven to B, which is A with ONLY joint1 replaced',
                'PAYLOAD_EXTEND': 'the same tracker driven to the fixed C extension over the side',
                'PAYLOAD_HOLD': f'advance_path=False closed loop on C for {FINAL_HOLD_S} s of valid quiet '
                                f'observation inside {FINAL_HOLD_DEADLINE_S} s',
                'STOPPED': 'one finite experimental stop reason, holding the last command and fingers',
            },
            'waypoint_formulas': {
                'reference': 'the child PUBLIC measured lift_start_q; the new +pi/6 contact wrist angle '
                             'enters automatically through it, and no private example q is used',
                'A': 'p=[.2+.35*dx, .35*dy, .45], n=[-dy, dx, 0], R=Rot(n,-.20) @ R0, bounded IK seeded '
                     'by the current public q',
                'B': 'copy qA, replace ONLY joint1 with +pi/2',
                'C': 'p=[.2,.52,.40], R=Rot([-1,0,0],-.40) @ Rz(pi/2-reference_q[0]) @ R0; the -.40 is '
                     'relative to R0 and is never added on top of A',
                'validation': f'original joint limits, IK position error <= {IK_POSITION_TOLERANCE_M} m '
                              f'and orientation error <= {IK_ORIENTATION_TOLERANCE_RAD} rad; a failed plan '
                              'stops BEFORE new motion with no fallback posture',
            },
            'tracking': {
                'scalar_path': f'one alpha per segment from the previous desired-actual goal, duration '
                               f'max|goal-start|/{PATH_MAX_AXIS_RATE_RAD_S} s; axes never advance '
                               'independently',
                'feedback_axes': {ARM6[index]: {'gain': gain, 'raw_cap_rad': cap,
                                                'damping_rad_per_rad_s': damp}
                                  for index, (gain, cap, damp) in FEEDBACK_AXES.items()},
                'other_axes': 'zero added correction',
                'integral_trim': f'slowly accumulated at {INTEGRAL_GAIN_PER_S} rad per rad-second, '
                                 f'clamped to {INTEGRAL_CAP_RAD} rad, and only while the proportional '
                                 'correction is unsaturated; it removes the feed-forward mismatch '
                                 'without raising the loop gain into the 1.67 Hz mode',
                'feed_forward': 'the child measured holding offset arm_command - reference, supplied '
                                'on every axis so the proportional term does not have to carry the '
                                'load; the correction is then seeded at zero, never at the same '
                                'offset twice',
                'tether_rad': TETHER_RAD.tolist(),
                'damping_velocity_filter': f'tau {DAMPING_VELOCITY_TAU_S} s on the measured velocity that '
                                          'feeds the damping term only, so Nyquist-rate solver noise cannot '
                                          'be rectified into the command; the position path is unfiltered',
                'filter': f'tau {FILTER_TAU_S} s, weight 1-exp(-dt/tau), initialised at the handoff from '
                          'child.arm_command - child.lift_goal clipped to each cap, and kept across '
                          'A/B/C and the final hold; only a pause freezes it',
                'command_bounds': f'simultaneous intersection of previous command +/-{COMMAND_RATE_RAD_S}*dt, '
                                  'measured q +/- per-axis tether and the original hard limits; an empty '
                                  'intersection holds the previous command and stops',
                'actuators': 'unchanged original K=80/D=4/100 N drives; no gain, limit or friction edit',
            },
            'continuity': {
                'legs': 'the EXACT final child action is the template, so both the child .02 m lowering '
                        'and the contact candidate extra .02 m are preserved; lower_delta is never '
                        'used to rebuild them',
                'fingers': 'the child closed finger targets are retained on every tick and every stop; '
                           'this module never opens the jaw',
                'wheels': 'zeroed for this stationary experiment, with the SAME wheel anchor held '
                          'throughout; only a future navigation wrapper may release it',
            },
            'completion': {
                'per_segment': f'alpha=1 AND actual max six-axis fixed-goal error < {GOAL_ERROR_RAD} rad '
                               f'AND arm qdot <= {QUIET_ARM_QDOT_RAD_S} rad/s AND one complete '
                               f'{QUIET_WINDOW_S} s window with arm span <= {QUIET_ARM_SPAN_RAD} rad and '
                               f'leg span <= {QUIET_LEG_SPAN_RAD} rad, under the original body quiet '
                               'conditions; no object state gates anything',
                'deadlines': f'per-segment planned duration + {SEGMENT_SLACK_S} s including pauses and '
                             'settling, never refreshed; one total budget fixed at plan creation',
                'normal_reason': 'payload_motion_observation_complete is a SEQUENCE claim only',
            },
            'reusable_units': {
                'build_payload_goals': 'pure static planner: (reference_q, current_q) -> A/B/C + '
                                       'diagnostics; raises ValueError instead of returning a fallback',
                'PayloadJointTracker': 'one fixed goal, one scalar path, selected-axis load compensation; '
                                       'advance_path=False, paused=False keeps the loaded closed loop on '
                                       'the SAME reference for a future navigation stage',
                'not_implemented_here': ['navigation', 'bucket approach', 'release', 'delivery', 'scoring'],
            },
            'inputs': {'allowed': ['public 84-value proprio', 'the RGB-D the frozen child already owns'],
                       'never_read': ['object pose', 'object orientation', 'contact force', 'net force',
                                      'world base pose', 'reward', 'score', 'map', 'seed',
                                      'private experiment reports']},
            'geometry_provenance': 'the private static clearance study of the EARLIER contact pose is '
                                   'planning evidence under a rigid attachment assumption only; it is not '
                                   'imported, is not a runtime constant, and requires an independent '
                                   'recheck for the new deeper +pi/6 contact',
            'stop_reasons': dict(PAYLOAD_STOP_REASONS),
            'not_a_claim': 'no grasp, lift, carry, transport, clearance, delivery, score or success is '
                           'demonstrated by this module; a retained jaw width never certifies attachment',
        }
