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
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM

PHASES = ("REACH", "CLOSE", "CLOSE_HOLD", "LIFT", "LIFT_HOLD", "DONE")


class GraspProbePolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 grasp_point_body, descent_delta_rad, preload_m=.025, reach_s=3.,
                 approach_clearance_m=.12,
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

    def _phase_of(self, seconds):
        return seconds

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
        elif self.phase == "LIFT_HOLD" and elapsed >= self.lift_hold_s:
            self.done_reason = "grasp_probe_complete"
            self._enter("DONE")

    def _enter(self, phase):
        self.phase, self.phase_start = phase, self.calls
        if phase == "CLOSE":
            self.jaw_target = self.jaw_closed.copy()

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

        self.debug = {
            'phase': self.phase, 'phase_s': (self.calls - self.phase_start) * self.dt,
            'grasp_point_body': self.grasp_point_body.tolist(),
            'arm_target_rad': full_arm[:6].tolist(), 'arm_measured_rad': measured.tolist(),
            'arm_trim_rad': self.arm_trim.tolist(),
            'reach_joint_error_rad': self.reach_joint_error_rad,
            'achieved_gripper_body': fk(measured)[:3, 3].tolist(),
            'achieved_jaw_mid_body': (fk(measured)[:3, 3] + GRASP_DEPTH * fk(measured)[:3, 2]).tolist(),
            'jaw_target_m': self.jaw_target.tolist(),
            'lift_alpha': self.alpha,
            'ik_position_error_m': self.ik_position_error_m,
            'ik_orientation_error_rad': self.ik_orientation_error_rad,
            'predicted_gripper_body': fk(self.reach_q)[:3, 3].tolist(),
            'pre_grasp_reached': self.pre_grasp_reached,
            'ik_pre_error_m': self.ik_pre_error_m,
            'jaw_axis_body': self.jaw_axis_body, 'approach_axis_body': self.approach_axis_body,
        }
        return action.astype(np.float32)

    def describe(self):
        return dict(mode='grasp_probe', oracle=True,
                    inputs='the measured body-frame position of one object, supplied by the '
                           'evaluator; public proprio; static action and robot geometry',
                    grasp_point_body=self.grasp_point_body.tolist(),
                    phases=list(PHASES), reach_s=self.reach_s,
                    close_s=self.close_s, lift_s=self.lift_s,
                    descent_delta_rad=self.descent_delta.tolist(),
                    ik_position_error_m=self.ik_position_error_m,
                    ik_orientation_error_rad=self.ik_orientation_error_rad,
                    jaw_axis_body=self.jaw_axis_body, approach_axis_body=self.approach_axis_body,
                    phase=self.phase, calls=self.calls, done_reason=self.done_reason,
                    claim='ORACLE grip test. It measures whether the jaws hold an object when '
                          'they are placed on it; it is not a perception, navigation or policy '
                          'result, and it claims no score.')
