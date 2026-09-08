"""Task E RGB-D / joint-feedback pick-and-place candidate.

No simulator state is imported or read. ``AlgSolution.predicts`` consumes only
the published observations and cumulative score. Perception can be injected for
offline controller checks; deployment imports task_e_perception. This candidate
must be evaluated in physics before any score or reliability claim is made.
"""
from collections import defaultdict
import json
import os
import numpy as np

try:
    from .task_e_geometry import (DEFAULT_JOINT_POS, TABLE_TOP_Z, fk,
                                  joints_from_proprio, action_from_joint_targets,
                                  solve_grasp_ik)
except ImportError:
    from task_e_geometry import (DEFAULT_JOINT_POS, TABLE_TOP_Z, fk,
                                 joints_from_proprio, action_from_joint_targets,
                                 solve_grasp_ik)


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class AlgSolution:
    """Finite-state observation-only controller, with at most 3 tries/object."""

    PICK_ORDER = (2, 1, 3)
    JAWS = {1: (-1., 0.), 2: (0., -1.), 3: (0., -1.)}
    BASKET_XY = {1: (1.12, -.32), 2: (1.04, -.30), 3: (1.00, -.27)}
    MAX_ATTEMPTS = 3
    GRASP_DEPTH = .115

    def __init__(self, perception_fn=None):
        if perception_fn is None:
            try:
                from .task_e_perception import estimate_objects
            except ImportError:
                from task_e_perception import estimate_objects
            perception_fn = estimate_objects
        self.estimate_objects = perception_fn
        self.debug = os.environ.get('ATEC_TASKE_DEBUG', '1') != '0'
        self.reset()

    def get_action_spec(self):
        return {}

    def reset(self, **kwargs):
        self.state = 'INIT'
        self.state_step = 0
        self.total_steps = 0
        self.current_object = None
        self.attempts = defaultdict(int)
        self.completed = set()
        self.grasped_once = set()
        self.samples = defaultdict(list)
        self.last_detections = {}
        self._waypoints = []
        self._waypoint_index = 0
        self._motion_settle = 0
        self._hold_q = DEFAULT_JOINT_POS[:6].copy()
        self._last_score = 0.
        self._attempt_score = 0.
        self._pre_release_score = 0.
        self._grasp_was_known = False
        self._lift_verified = False
        self._contact = None
        self._motion_goal = None
        self._done = False
        self._motion_steps = 0
        self._last_failure = ''

    def _log(self, event, **fields):
        if self.debug:
            print('[TASKE_VISION] ' + json.dumps(dict(
                event=event, step=self.total_steps, state=self.state,
                object=self.current_object, score=round(self._last_score, 3),
                **fields), ensure_ascii=False), flush=True)

    def _set_state(self, state):
        self.state, self.state_step = state, 0
        self._motion_settle = 0
        self._motion_steps = 0
        self._log('state')

    def _read_vision(self, obs):
        try:
            raw = self.estimate_objects(obs)
        except (ValueError, KeyError, RuntimeError) as error:
            self._log('vision_unavailable', reason=str(error))
            return {}
        detections = {}
        for key, value in raw.items():
            pos = _numpy(value).astype(float).reshape(-1)
            if int(key) in self.PICK_ORDER and pos.size == 3 and np.all(np.isfinite(pos)):
                detections[int(key)] = pos
        self.last_detections = detections
        return detections

    @staticmethod
    def _pinch_from_joints(q):
        pose = fk(q)
        return pose[:3, 3] + AlgSolution.GRASP_DEPTH * pose[:3, 2]

    def _plan_motion(self, goal, q, *, start=None, spacing=.025):
        """Plan contact-space line into bounded IK joint waypoints."""
        goal = np.asarray(goal, dtype=float)
        start = self._pinch_from_joints(q) if start is None else np.asarray(start, dtype=float)
        count = max(1, int(np.ceil(np.linalg.norm(goal-start) / spacing)))
        count = min(count, 32)
        seed = q[:6].copy()
        poses = []
        worst_error = 0.
        for alpha in np.linspace(1./count, 1., count):
            contact = start + alpha * (goal-start)
            try:
                result = solve_grasp_ik(contact, self.JAWS[self.current_object], seed,
                                        grasp_depth=self.GRASP_DEPTH,
                                        tilt_degrees=(0., 5., 10., 15., 20., 25., 30., 35., 40., 50., 60.))
            except ValueError as error:
                self._log('plan_rejected', reason=str(error), goal=goal.tolist())
                return False
            worst_error = max(worst_error, result.ik.position_error)
            if result.ik.position_error > .018 or result.ik.orientation_error > .18:
                self._log('plan_rejected', position_error=result.ik.position_error,
                          orientation_error=result.ik.orientation_error, contact=contact.tolist())
                return False
            seed = result.ik.joints
            poses.append(seed.copy())
        self._waypoints = poses
        self._waypoint_index = 0
        self._motion_goal = goal.copy()
        self._motion_settle = 0
        self._motion_steps = 0
        self._log('planned', waypoints=len(poses), goal=np.round(goal, 5).tolist(),
                  worst_position_error=round(worst_error, 5))
        return True

    def _motion_command(self, q, *, slow=False):
        """Feedback-limited target; advance only after each waypoint is reached."""
        self._motion_steps += 1
        if not self._waypoints:
            return self._hold_q.copy(), True
        target = self._waypoints[self._waypoint_index]
        error = float(np.max(np.abs(target-q[:6])))
        tolerance = .025 if slow else .04
        if error < tolerance and self._waypoint_index < len(self._waypoints)-1:
            self._waypoint_index += 1
            target = self._waypoints[self._waypoint_index]
            error = float(np.max(np.abs(target-q[:6])))
        last = self._waypoint_index == len(self._waypoints)-1
        self._motion_settle = self._motion_settle+1 if last and error < .025 else 0
        delta = .035 if slow else .085
        command = q[:6] + np.clip(target-q[:6], -delta, delta)
        self._hold_q = target.copy()
        return command, self._motion_settle >= 12

    def _retract(self, q, reason=''):
        self._last_failure = reason
        # First move vertically clear in contact space while keeping jaws open.
        start = self._pinch_from_joints(q)
        goal = start.copy()
        goal[2] = max(goal[2], TABLE_TOP_Z+.27)
        if self.current_object is not None and self._plan_motion(goal, q):
            self._set_state('RETRACT')
        else:
            self._waypoints = [DEFAULT_JOINT_POS[:6].copy()]
            self._waypoint_index = 0
            self._set_state('HOME')
        if reason:
            self._log('retry', reason=reason, attempts=self.attempts[self.current_object])

    def _select_and_plan(self, q, score):
        candidates = [i for i in self.PICK_ORDER if i not in self.completed and
                      self.attempts[i] < self.MAX_ATTEMPTS and len(self.samples[i]) >= 3]
        if not candidates:
            return False
        self.current_object = candidates[0]
        points = np.asarray(self.samples[self.current_object][-7:])
        self._contact = np.median(points, axis=0)
        # Reject unstable detection; do not substitute a nominal spawn position.
        if np.max(np.ptp(points[:, :2], axis=0)) > .045:
            self.samples[self.current_object].clear()
            return False
        self.attempts[self.current_object] += 1
        self._attempt_score = score
        self._grasp_was_known = self.current_object in self.grasped_once
        self._lift_verified = False
        self._log('selected', contact=np.round(self._contact, 5).tolist(),
                  attempt=self.attempts[self.current_object])
        above = self._contact.copy()
        above[2] += .12
        if not self._plan_motion(above, q):
            self._retract(q, 'pregrasp_unreachable')
            return True
        self._set_state('PREGRASP')
        return True

    def _start_descent(self, q, obs):
        # Re-observe immediately above the target, before closing onto its old pose.
        observed = self._read_vision(obs).get(self.current_object)
        if observed is not None and np.linalg.norm(observed[:2]-self._contact[:2]) < .04:
            self._contact = observed.copy()
        if self._plan_motion(self._contact, q, spacing=.012):
            self._set_state('DESCEND')
        else:
            self._retract(q, 'descent_unreachable')

    def _action(self, arm, closed=False):
        target = np.empty(8)
        target[:6] = arm
        # Position targets beyond the physical closed limit create holding force,
        # as in the existing expert. Only command targets, never alter limits.
        close = -.025 if self.current_object == 2 else -.015
        target[6:] = [close, -close] if closed else [.035, -.035]
        return {'action': action_from_joint_targets(target).tolist(), 'giveup': bool(self._done)}

    def predicts(self, obs, current_score):
        proprio = _numpy(obs['proprio']).reshape(-1)
        q = joints_from_proprio(proprio)
        score = float(_numpy(current_score).reshape(-1)[0])
        if score + 1e-4 < self._last_score:
            self.reset()
        self._last_score = score
        self.total_steps += 1
        self.state_step += 1
        if score >= 17.99:
            self._done = True
            return self._action(q[:6], closed=False)
        if self._done:
            return self._action(q[:6], closed=False)

        arm, closed = self._hold_q.copy(), False
        if self.state == 'INIT':
            arm = q[:6]+np.clip(DEFAULT_JOINT_POS[:6]-q[:6], -.06, .06)
            home_error = float(np.max(np.abs(q[:6]-DEFAULT_JOINT_POS[:6])))
            if self.state_step >= 80 and home_error < .04:
                self.samples.clear()
                self._hold_q = DEFAULT_JOINT_POS[:6].copy()
                self._set_state('LOCATE')
            elif self.state_step >= 400:
                self._done = True
                self._log('initial_tracking_failed', joint_error=home_error)

        elif self.state == 'LOCATE':
            if self.state_step % 5 == 1:
                for i, point in self._read_vision(obs).items():
                    if i not in self.completed:
                        self.samples[i].append(point)
            if self.state_step >= 16 and self._select_and_plan(q, score):
                pass
            elif self.state_step >= 150:
                remaining = [i for i in self.PICK_ORDER if i not in self.completed and
                             self.attempts[i] < self.MAX_ATTEMPTS]
                if remaining:
                    # A bounded return to the high home view can clear occlusions.
                    for i in remaining:
                        self.attempts[i] += 1
                    self._waypoints = [DEFAULT_JOINT_POS[:6].copy()]
                    self._waypoint_index = 0
                    self._set_state('HOME')
                else:
                    self._done = True
                    self._log('exhausted', completed=sorted(self.completed), attempts=dict(self.attempts))

        elif self.state in ('PREGRASP', 'DESCEND', 'LIFT', 'TRANSPORT', 'PLACE', 'RETRACT', 'HOME'):
            moving_state = self.state
            closed = moving_state in ('LIFT', 'TRANSPORT', 'PLACE')
            arm, reached = self._motion_command(q, slow=moving_state in ('DESCEND', 'PLACE'))
            if self._motion_steps > 700:
                if moving_state == 'HOME':
                    self._done = True
                    self._log('home_tracking_failed')
                elif moving_state == 'RETRACT':
                    self._waypoints = [DEFAULT_JOINT_POS[:6].copy()]
                    self._waypoint_index = 0
                    self._set_state('HOME')
                else:
                    self._retract(q, 'joint_tracking_timeout_'+moving_state.lower())
                closed = False
            elif reached:
                if moving_state == 'PREGRASP':
                    self._start_descent(q, obs)
                elif moving_state == 'DESCEND':
                    self._hold_q = q[:6].copy()
                    self._set_state('CLOSE')
                elif moving_state == 'LIFT':
                    self._set_state('VERIFY_LIFT')
                elif moving_state == 'TRANSPORT':
                    place = np.r_[self.BASKET_XY[self.current_object], TABLE_TOP_Z+.15]
                    if self._plan_motion(place, q, spacing=.015):
                        self._set_state('PLACE')
                    else:
                        self._pre_release_score = score
                        self._set_state('OPEN')
                elif moving_state == 'PLACE':
                    self._pre_release_score = score
                    self._set_state('OPEN')
                elif moving_state == 'RETRACT':
                    self._waypoints = [DEFAULT_JOINT_POS[:6].copy()]
                    self._waypoint_index = 0
                    self._set_state('HOME')
                elif moving_state == 'HOME':
                    self.samples.clear()
                    self._set_state('LOCATE')

        elif self.state == 'CLOSE':
            closed = True
            if self.state_step >= 70:
                jaw_width = float(q[6]-q[7])
                self._log('closed', jaw_width=round(jaw_width, 5))
                if jaw_width < .006:
                    self._retract(q, 'empty_gripper')
                    closed = False
                else:
                    lift = self._contact.copy()
                    lift[2] = TABLE_TOP_Z+.27
                    if self._plan_motion(lift, q, spacing=.012):
                        self._set_state('LIFT')
                    else:
                        self._retract(q, 'lift_unreachable')
                        closed = False

        elif self.state == 'VERIFY_LIFT':
            closed = True
            if self.state_step >= 30:
                jaw_width = float(q[6]-q[7])
                new_grasp_score = score >= self._attempt_score+2.9
                self._lift_verified = (new_grasp_score or self._grasp_was_known) and jaw_width >= .006
                if new_grasp_score:
                    self.grasped_once.add(self.current_object)
                self._log('lift_verified', verified=self._lift_verified,
                          jaw_width=round(jaw_width, 5), gain=round(score-self._attempt_score, 3))
                if self._lift_verified:
                    carry = np.r_[self.BASKET_XY[self.current_object], TABLE_TOP_Z+.27]
                    if self._plan_motion(carry, q, spacing=.025):
                        self._set_state('TRANSPORT')
                    else:
                        self._retract(q, 'transport_unreachable')
                        closed = False
                else:
                    self._retract(q, 'lift_not_confirmed')
                    closed = False

        elif self.state == 'OPEN':
            if self.state_step >= 70:
                self._set_state('VERIFY_PLACE')

        elif self.state == 'VERIFY_PLACE':
            if self.state_step >= 35:
                required_gain = 3. if self._grasp_was_known else 6.
                scored = score >= self._attempt_score+required_gain-.1
                # Score may arrive during PLACE before the jaws actually open.
                detection = self._read_vision(obs).get(self.current_object)
                visible_inside = detection is not None and (
                    abs(detection[0]-1.08) < .19 and abs(detection[1]+.30) < .10
                    and TABLE_TOP_Z <= detection[2] <= TABLE_TOP_Z+.16)
                if scored or (self._lift_verified and visible_inside):
                    self.completed.add(self.current_object)
                    self._log('object_complete', gain=round(score-self._attempt_score, 3),
                              verified_by='score' if scored else 'vision')
                else:
                    self._log('place_unconfirmed', gain=round(score-self._attempt_score, 3))
                self._retract(q)

        return self._action(arm, closed=closed)
