"""Oracle grasp probe: can the Piper jaws actually hold a Task B object?

Camera-free. The evaluator parks the base beside one object using its true pose
and hands this policy that object's measured body-frame position; everything
after that -- the bounded arm reach, the wheels-planted descent, the finger
closure and the lift -- is commanded through the ordinary action terms and
executed by the official environment. Nothing here writes simulator state.

This is an ORACLE probe, not a policy result: it is told where the object is, so
it measures grip physics, not perception or navigation. It asserts no grasp; the
evaluator's object and gripper records decide whether anything was held.
"""
from __future__ import annotations

import numpy as np

from task_b import leg_kinematics as legs
from task_e_geometry import top_grasp_rotation
from task_b.arm_kinematics import ARM_JOINT_NAMES, GRASP_DEPTH, arm_targets_to_action, fk, solve_ik
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM, wheel_side

PHASES = ("REACH", "CLOSE", "CLOSE_HOLD", "LIFT", "LIFT_HOLD", "CARRY",
           "PLACE_RAISE", "PLACE_DRIVE", "PLACE_IN", "RELEASE", "DONE")


class GraspProbePolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 grasp_point_body, descent_delta_rad, preload_m=.025, reach_s=3.,
                 approach_clearance_m=.12, carry_s=0., carry_cmd=.10,
                 target_xy=None, carry_stop_m=1.2, place_lift_m=.40, place_forward_m=.10,
                 place_s=2.5, carry_stall_s=2.5, place_drive_s=2.0,
                 close_s=2., close_hold_s=1.5, lift_s=2., lift_hold_s=2.):
        schema.validate()
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        point = np.asarray(grasp_point_body, dtype=float).reshape(-1)
        if point.size != 3 or not np.isfinite(point).all():
            raise ValueError("grasp_point_body must be three finite body-frame metres")
        descent = np.asarray(descent_delta_rad, dtype=float).reshape(-1)
        if descent.shape != (12,) or not np.isfinite(descent).all():
            raise ValueError("descent_delta_rad must be the 12 leg-joint deltas from the standing "
                             "pose to the pose the evaluator descended to")
        # The probe allows a deeper preload than arm_kinematics.gripper_targets caps
        # at Task E (0.025 m). The finger servo is only as strong as its position
        # error, and the mustard bottle needed more than the ~4 N that cap produced:
        # it slipped out as soon as the lift began. This is a grip-force knob, not a
        # trajectory, so it is bounded by what stays physically sensible.
        if not np.isfinite(preload_m) or not 0. <= preload_m <= .08:
            raise ValueError("preload_m must lie in [0, .08] m")
        for name, value in (("reach_s", reach_s), ("close_s", close_s),
                            ("close_hold_s", close_hold_s), ("lift_s", lift_s),
                            ("lift_hold_s", lift_hold_s)):
            if not np.isfinite(value) or value < 0.:
                raise ValueError(f"{name} must be finite and non-negative")
        self.schema, self.names, self.defaults = schema, tuple(observation_joint_names), dict(defaults)
        self.dt = float(dt)
        self.grasp_point_body = point
        self.descent_delta = descent
        self.preload_m = float(preload_m)
        self.approach_clearance_m = float(approach_clearance_m)
        self.turn_gain, self.turn_cap = 1.5, .35
        if not np.isfinite(carry_s) or carry_s < 0.:
            raise ValueError("carry_s must be finite and non-negative")
        if not np.isfinite(carry_cmd) or not 0. < abs(carry_cmd) <= .6:
            raise ValueError("carry_cmd must be finite and non-zero, in [-0.6, 0.6]")
        self.carry_s, self.carry_cmd = float(carry_s), float(carry_cmd)
        # Delivery: carry to within carry_stop_m of the target, then raise the held
        # object above the bin's 0.50 m lip and let go. The bin's outer wall radius
        # is 1.0 m, which is also the reward radius, so the object has to end up
        # INSIDE the wall, not merely pushed against it.
        self.target_xy = None if target_xy is None else np.asarray(target_xy, dtype=float).reshape(2)
        if self.target_xy is not None and not np.isfinite(self.target_xy).all():
            raise ValueError("target_xy must be two finite world-frame metres")
        self.carry_stop_m, self.place_lift_m = float(carry_stop_m), float(place_lift_m)
        self.carry_timeout_s = 60.
        self.place_s = float(place_s)
        self.place_forward_m = float(place_forward_m)
        self.carry_stall_s = float(carry_stall_s)
        self.place_drive_s = float(place_drive_s)
        self.carry_anchor, self.carry_stall_start = None, None
        self.pose_xy, self.pose_yaw = None, None
        self.carry_distance_m = 0.0
        self.place_raise_q = None
        self.place_raise_error_m = None
        self.place_in_q = None
        self.place_in_error_m = None
        self.place_start_q = None
        # Jaw travel in metres, as arm_kinematics.gripper_targets defines it: open is
        # the joint's full travel, closed is the stop. The close commands PAST the
        # stop by preload_m, so the servo keeps pressing instead of settling at zero
        # error -- measured on the mustard bottle, closing exactly at the stop left
        # roughly 4 N per finger, which slipped while lifting a 0.5 kg object.
        self.jaw_open = np.array([.035, -.035])
        self.jaw_closed = np.array([-self.preload_m, self.preload_m])
        self.reach_s = float(reach_s)
        self.close_s, self.close_hold_s = float(close_s), float(close_hold_s)
        self.lift_s, self.lift_hold_s = float(lift_s), float(lift_hold_s)

        self.arm, self.leg, self.wheel = (schema.term(ARM_TERM), schema.term(LEG_TERM),
                                          schema.term(WHEEL_TERM))
        self.leg_obs_ids = np.array([self.names.index(n) for n in self.leg.joint_names])
        self.arm_obs_ids = np.array([self.names.index(n) for n in self.arm.joint_names])
        self.arm_defaults = np.array([self.defaults[n] for n in ARM_JOINT_NAMES])
        # Right wheels positive, left negative, for the differential turn.
        self.sides = np.array([1. if wheel_side(n) == 'right' else -1.
                               for n in self.wheel.joint_names])

        # Solve the top-down grasp once, before the run. The evaluator has already
        # lowered the body and measured the object in that stance, so the reach goes
        # straight to the grasp point: the jaw midpoint lands on it, which puts the
        # gripper_base GRASP_DEPTH above it with the approach axis pointing down.
        #
        # The fingers slide along the gripper's local +Y (per the USD: arm_link7
        # spans y in [-0.0265, 0] and arm_link8 y in [0, +0.0265]), so the jaw axis
        # handed to top_grasp_rotation must be the body's +Y: perpendicular to the
        # approach, straddling the object left-right. Pointing it along the approach
        # would close the jaws front-to-back instead.
        jaw_axis = (0., 1.)
        rotation_target = top_grasp_rotation(jaw_axis)
        goal = point + np.array([0., 0., GRASP_DEPTH])
        # Two targets, not one. The finger tips sit 21 mm BEYOND the jaw midpoint,
        # so driving straight to the grasp pose rakes them across the object's top
        # and knocks it over -- measured on a mustard bottle, which tipped every run.
        # Instead: arrive directly above the grasp point, then descend vertically.
        approach = point + np.array([0., 0., GRASP_DEPTH + self.approach_clearance_m])
        fit = solve_ik(goal, rotation=rotation_target, seed=self.arm_defaults[:6], max_nfev=400)
        pre = solve_ik(approach, rotation=rotation_target, seed=np.asarray(fit.joints), max_nfev=400)
        self.reach_pre_q = np.asarray(pre.joints, dtype=float)
        self.ik_pre_error_m = float(pre.position_error)
        if not pre.success or self.ik_pre_error_m > .01:
            raise RuntimeError(f"Pre-grasp IK failed: position error {self.ik_pre_error_m:.4f} m")
        self.reach_q = np.asarray(fit.joints, dtype=float)
        self.ik_position_error_m = float(fit.position_error)
        self.ik_orientation_error_rad = float(fit.orientation_error)
        self.ik_ok = bool(fit.success)
        solved = fk(self.reach_q)
        self.jaw_axis_body = solved[:3, 1].tolist()
        self.approach_axis_body = solved[:3, 2].tolist()
        if abs(solved[2, 1]) > .25:
            raise RuntimeError(f"Solved jaw axis {np.round(solved[:3, 1], 3)} is not horizontal")
        if abs(float(solved[:3, 1] @ np.array([1., 0., 0.]))) > .4:
            raise RuntimeError(
                f"Solved jaw axis {np.round(solved[:3, 1], 3)} runs along the approach instead of "
                "across it; the fingers would close front-to-back and could not straddle the object")
        if not self.ik_ok or self.ik_position_error_m > .01:
            raise RuntimeError(
                f"Top-down grasp IK failed from the standing stance: position error "
                f"{self.ik_position_error_m:.4f} m, orientation error "
                f"{self.ik_orientation_error_rad:.4f} rad. The target is "
                f"{GRASP_DEPTH:.4f} m above the grasp point.")

        self.calls = 0
        self.alpha = 0.0
        self.phase = "REACH"
        self.phase_start = 0
        self.jaw_target = self.jaw_open.copy()
        self.done_reason = None
        self.debug = {}
        # The arm is a position servo with finite stiffness. Loaded, it settles
        # short of the commanded pose -- measured 0.084 rad on arm_joint2, worth
        # ~5 cm at the jaws. An integral trim holds the solved pose; the tether
        # still bounds how far the command may run ahead of the real joint.
        self.arm_trim = np.zeros(6)
        self.trim_gain, self.trim_cap, self.tether = 2.0, .15, .25
        self.reach_tolerance_rad = .02
        self.reach_timeout_s = 8.
        self.reach_joint_error_rad = None
        self.pre_grasp_reached = False
        self.arm_measured = None

    def set_pose(self, position, yaw):
        """ORACLE: the evaluator supplies the true base pose each step.

        The probe is told where it is so that the delivery mechanics can be tested
        on their own, without odometry error mixed in. Nothing here reads state.
        """
        self.pose_xy = np.asarray(position, dtype=float).reshape(3)[:2].copy()
        self.pose_yaw = float(yaw)

    def _advance(self):
        elapsed = (self.calls - self.phase_start) * self.dt
        if self.phase == "REACH":
            # Leave REACH only once the arm has actually arrived, not on a timer.
            arrived = (self.reach_joint_error_rad is not None
                       and self.reach_joint_error_rad <= self.reach_tolerance_rad)
            if arrived and not self.pre_grasp_reached:
                self.pre_grasp_reached = True          # switch to the vertical descent
                self.phase_start = self.calls
                return
            if arrived:
                self._enter("CLOSE")
            elif elapsed >= self.reach_timeout_s:
                self.done_reason = "grasp_probe_arm_did_not_reach_the_solved_pose"
                self._enter("DONE")
        elif self.phase == "CLOSE" and elapsed >= self.close_s:
            self._enter("CLOSE_HOLD")
        elif self.phase == "CLOSE_HOLD" and elapsed >= self.close_hold_s:
            self._enter("LIFT")
        elif self.phase == "LIFT" and elapsed >= self.lift_s:
            self._enter("LIFT_HOLD")
        elif self.phase == "LIFT_HOLD":
            if elapsed >= self.lift_hold_s:
                # A zero carry_s keeps the probe a stationary grip test.
                self._enter("CARRY" if (self.carry_s > 0. or self.target_xy is not None) else "DONE")
                if self.phase == "DONE":
                    self.done_reason = "grasp_probe_complete"
        elif self.phase == "CARRY":
            if self.target_xy is None:
                if elapsed >= self.carry_s:
                    self.done_reason = "grasp_probe_complete"
                    self._enter("DONE")
            elif self.pose_xy is not None and np.linalg.norm(self.pose_xy - self.target_xy) <= self.carry_stop_m:
                self._enter("PLACE_RAISE")
            elif self._carry_stalled():
                # Carried as far as it will go -- the held object is against the bin
                # wall and the wheels are stalled. That is the cue to raise it over.
                self.debug["carry_stalled"] = True
                self._enter("PLACE_RAISE")
            elif elapsed >= self.carry_timeout_s:
                self.done_reason = "grasp_probe_carry_timed_out"
                self._enter("DONE")
        elif self.phase == "PLACE_RAISE":
            if self.place_raise_q is None and self.place_raise_error_m is None:
                self._solve_place_raise()
            elif elapsed >= self.place_s and (self.reach_joint_error_rad is not None
                                              and self.reach_joint_error_rad <= self.reach_tolerance_rad):
                # With the object lifted clear of the lip it no longer blocks the
                # vehicle, so the rest of the approach is driven rather than reached
                # for. The arm's inward move only achieved 47% of its target.
                self._enter("PLACE_DRIVE" if self.place_drive_s > 0. else "PLACE_IN")
        elif self.phase == "PLACE_DRIVE":
            # Stop before the BODY reaches the bin. The base_link box is about 0.43 m
            # long and 0.24 m half-wide, so its front corner reaches ~0.49 m ahead of
            # the origin: at 1.484 m from the centre it already touched the wall and
            # tripped illegal_contact at 3.16 N.
            arrived = (self.pose_xy is not None and self.target_xy is not None
                       and np.linalg.norm(self.pose_xy - self.target_xy) <= self.carry_stop_m)
            if arrived or elapsed >= self.place_drive_s:
                self._enter("PLACE_IN")
        elif self.phase == "PLACE_IN":
            # Raise and move inward as two separate targets: asking for both at once
            # exceeded the arm's reach from the real grasp pose (0.039 m short).
            if self.place_in_q is None and self.place_in_error_m is None:
                self._solve_place_in()
            elif elapsed >= self.place_s and (self.reach_joint_error_rad is not None
                                              and self.reach_joint_error_rad <= self.reach_tolerance_rad):
                self._enter("RELEASE")
        elif self.phase == "RELEASE" and elapsed >= 0.8:
            self.done_reason = "grasp_probe_complete"
            self._enter("DONE")

    def _enter(self, phase):
        self.phase, self.phase_start = phase, self.calls
        if phase == "CLOSE":
            self.jaw_target = self.jaw_closed.copy()
        elif phase == "RELEASE":
            self.jaw_target = self.jaw_open.copy()
        elif phase == "PLACE_RAISE":
            self.place_raise_q, self.place_raise_error_m = None, None
            self.place_start_q = (None if self.arm_measured is None
                                  else self.arm_measured.copy())

    def _carry_stalled(self):
        """True when the base has stopped advancing, measured from the oracle pose."""
        if self.pose_xy is None:
            return False
        if self.carry_anchor is None:
            self.carry_anchor, self.carry_stall_start = self.pose_xy.copy(), self.calls
            return False
        if float(np.linalg.norm(self.pose_xy - self.carry_anchor)) > .01:
            self.carry_anchor, self.carry_stall_start = self.pose_xy.copy(), self.calls
            return False
        return (self.calls - self.carry_stall_start) * self.dt >= self.carry_stall_s

    def _solve_place_raise(self):
        """Raise the held object clear of the bin lip, from where the arm actually is."""
        pose = fk(self.arm_measured)
        jaw_mid = pose[:3, 3] + GRASP_DEPTH * pose[:3, 2]
        self.place_from_body_xyz = jaw_mid.tolist()
        goal = pose[:3, 3] + np.array([0., 0., self.place_lift_m])
        fit = solve_ik(goal, rotation=pose[:3, :3], seed=self.arm_measured, max_nfev=400)
        self.place_raise_q = np.asarray(fit.joints, dtype=float)
        self.place_raise_error_m = float(fit.position_error)
        self.place_raised_body_xyz = fk(self.place_raise_q)[:3, 3].tolist()

    def _solve_place_in(self):
        """Then move the raised object inward, past the bin wall."""
        pose = fk(self.arm_measured)
        goal = pose[:3, 3] + np.array([self.place_forward_m, 0., 0.])
        fit = solve_ik(goal, rotation=pose[:3, :3], seed=self.arm_measured, max_nfev=400)
        self.place_in_q = np.asarray(fit.joints, dtype=float)
        self.place_in_error_m = float(fit.position_error)
        # Best effort: a short raise still moves the object and the run's own records
        # say where it ended up, which is more informative than aborting here.

    def act(self, proprio):
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            raise ValueError(f"grasp_probe requires finite proprio84, got {obs.size}")
        self.calls += 1
        self._advance()

        # Arm: integral trim toward the solved grasp pose. The plain command is
        # clamped to the measured joint plus a tether, so the trim can only push
        # as far as the arm is allowed to follow.
        measured = (obs[12 + self.arm_obs_ids] + self.arm_defaults)[:6]
        self.arm_measured = measured
        # Stage 1 aims at the pre-grasp pose; stage 2, once that is reached, at the
        # grasp pose itself. Converging on the pre-grasp pose before switching is what
        # makes the final descent vertical.
        # The place motions are interpolated over place_s rather than jumped to. A
        # 0.40 m raise commanded at once moved at ~0.57 m/s and threw the object out
        # of the jaws inside 0.7 s; the grip only survives a gentle lift.
        if self.phase == "PLACE_IN" and self.place_in_q is not None:
            progress = min(1., (self.calls - self.phase_start) * self.dt / self.place_s)
            target_q = self.place_raise_q + progress * (self.place_in_q - self.place_raise_q)
        elif self.phase == "PLACE_RAISE" and self.place_raise_q is not None:
            progress = min(1., (self.calls - self.phase_start) * self.dt / self.place_s)
            target_q = self.place_start_q + progress * (self.place_raise_q - self.place_start_q)
        elif self.phase in ("PLACE_IN", "RELEASE") and self.place_raise_q is not None:
            target_q = self.place_raise_q
        else:
            target_q = self.reach_pre_q if not self.pre_grasp_reached else self.reach_q
        error = target_q - measured
        self.reach_joint_error_rad = float(np.max(np.abs(error)))
        self.arm_trim = np.clip(self.arm_trim + self.dt * self.trim_gain * error,
                                -self.trim_cap, self.trim_cap)
        arm_command = np.clip(target_q + self.arm_trim,
                              measured - self.tether, measured + self.tether)

        # Legs: descend during LOWER, hold, then return along the same path at LIFT.
        # The evaluator performed the descent before measuring, so the policy starts
        # by HOLDING it -- alpha 0 commands the descended stance, not the standing
        # one. Only the lift moves the legs back, and it does so at the end.
        if self.phase in ("LIFT", "LIFT_HOLD", "DONE"):
            self.alpha = min(1., self.alpha + self.dt / max(self.lift_s, 1e-6))

        jaws = self.jaw_target
        full_arm = np.r_[arm_command, jaws]

        action = np.zeros(self.schema.total_dim, dtype=np.float64)
        action[self.arm.start:self.arm.stop] = arm_targets_to_action(
            full_arm, self.arm.joint_names, self.defaults, scale=self.arm.scale)
        action[self.leg.start:self.leg.stop] = (1. - self.alpha) * self.descent_delta / self.leg.scale
        # Carry: drive all four wheels the same way. Which way the base actually
        # travels is a property of the chassis, not an assumption -- the evaluator
        # records the base displacement so the sign is measured, never inferred.
        if self.phase == "CARRY":
            action[self.wheel.start:self.wheel.stop] = self._carry_command()
        elif self.phase == "PLACE_DRIVE":
            # Straight in: the heading is already lined up on the bin, and this
            # chassis cannot steer.
            action[self.wheel.start:self.wheel.stop] = self.carry_cmd

        self.debug = {
            'phase': self.phase, 'phase_s': (self.calls - self.phase_start) * self.dt,
            'grasp_point_body': self.grasp_point_body.tolist(),
            'arm_target_rad': full_arm[:6].tolist(), 'arm_measured_rad': measured.tolist(),
            'arm_trim_rad': self.arm_trim.tolist(),
            'reach_joint_error_rad': self.reach_joint_error_rad,
            'achieved_gripper_body': fk(measured)[:3, 3].tolist(),
            'achieved_jaw_mid_body': (fk(measured)[:3, 3] + GRASP_DEPTH * fk(measured)[:3, 2]).tolist(),
            'jaw_target_m': self.jaw_target.tolist(),
            'lift_alpha': self.alpha, 'carry_distance_m': self.carry_distance_m,
            'place_raise_from_body_z': getattr(self, 'place_raise_from_body_z', None),
            'place_raise_error_m': self.place_raise_error_m,
            'ik_position_error_m': self.ik_position_error_m,
            'ik_orientation_error_rad': self.ik_orientation_error_rad,
            'predicted_gripper_body': fk(self.reach_q)[:3, 3].tolist(),
            'pre_grasp_reached': self.pre_grasp_reached,
            'ik_pre_error_m': self.ik_pre_error_m,
            'jaw_axis_body': self.jaw_axis_body, 'approach_axis_body': self.approach_axis_body,
        }
        return action.astype(np.float32)

    def _carry_command(self):
        """Steer toward the target, then drive. Oracle pose in, four wheel commands out."""
        zero = np.zeros(len(self.wheel.joint_names))
        if self.pose_xy is None or self.target_xy is None:
            return zero + self.carry_cmd
        delta = self.target_xy - self.pose_xy
        distance = float(np.linalg.norm(delta))
        self.carry_distance_m = distance
        if distance <= 1e-6:
            return zero
        bearing = float(np.arctan2(delta[1], delta[0]))
        error = float(np.arctan2(np.sin(bearing - self.pose_yaw), np.cos(bearing - self.pose_yaw)))
        turn = float(np.clip(self.turn_gain * error, -self.turn_cap, self.turn_cap))
        # Slow down near the target so the stop lands inside the placement window.
        forward = self.carry_cmd * float(np.clip((distance - self.carry_stop_m) / .8, .15, 1.))
        forward *= float(np.clip(1. - abs(error) / 1.2, .0, 1.))
        return np.clip(forward + self.sides * turn, -.6, .6)

    def describe(self):
        return dict(mode='grasp_probe', oracle=True,
                    inputs='the measured body-frame position of one object, supplied by the '
                           'evaluator; public proprio; static action and robot geometry',
                    grasp_point_body=self.grasp_point_body.tolist(),
                    phases=list(PHASES), reach_s=self.reach_s,
                    close_s=self.close_s, lift_s=self.lift_s,
                    descent_delta_rad=self.descent_delta.tolist(),
                    carry_s=self.carry_s, carry_cmd=self.carry_cmd,
                    target_xy=None if self.target_xy is None else self.target_xy.tolist(),
                    carry_stop_m=self.carry_stop_m, place_lift_m=self.place_lift_m,
                    place_forward_m=self.place_forward_m,
                    place_in_q_done=self.place_in_q is not None,
                    ik_position_error_m=self.ik_position_error_m,
                    ik_orientation_error_rad=self.ik_orientation_error_rad,
                    jaw_axis_body=self.jaw_axis_body, approach_axis_body=self.approach_axis_body,
                    phase=self.phase, calls=self.calls, done_reason=self.done_reason,
                    claim='ORACLE grip test. It measures whether the jaws hold an object when '
                          'they are placed on it; it is not a perception, navigation or policy '
                          'result, and it claims no score.')
