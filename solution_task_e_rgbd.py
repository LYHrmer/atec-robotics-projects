"""Task E RGB-D controller with object-specific collision clearance.

Runtime inputs are the public RGB-D images, joint observations and score.
Kinematic constants describe the published robot, not live simulator state.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .solution_task_e_vision import AlgSolution as BaseSolution
    from .task_e_geometry import TABLE_TOP_Z, unproject_depth, action_from_joint_targets, fk, solve_ik
    from .task_e_perception import estimate_objects_detailed
except ImportError:
    from solution_task_e_vision import AlgSolution as BaseSolution
    from task_e_geometry import TABLE_TOP_Z, unproject_depth, action_from_joint_targets, fk, solve_ik
    from task_e_perception import estimate_objects_detailed


def estimate_grasp_contacts(obs):
    objects, detail = estimate_objects_detailed(obs)
    bottle = detail['detections'].get('2')
    if bottle is not None:
        # The tall cap can collide with the palm above a waist grasp. Locate
        # its upper footprint in depth and grasp the neck instead.
        objects.pop(2, None)
        if bottle['shape_cost'] <= 2.0 and bottle['pixels'] >= 400:
            lower, upper = np.asarray(bottle['visible_bounds'])
            depth = obs.get('image', obs)['video_depth']
            if hasattr(depth, 'detach'):
                depth = depth.detach().cpu().numpy()
            points = unproject_depth(depth)
            within = ((points[:, 0] > lower[0]-.004) & (points[:, 0] < upper[0]+.004)
                      & (points[:, 1] > lower[1]-.004) & (points[:, 1] < upper[1]+.004)
                      & (points[:, 2] > upper[2]-.016) & (points[:, 2] < upper[2]+.005))
            cap = points[within]
            if len(cap) >= 25:
                cap_lower, cap_upper = np.percentile(cap, [2, 98], axis=0)
                objects[2] = np.r_[(cap_lower[:2]+cap_upper[:2])/2., cap_upper[2]-.035]

    banana = detail['detections'].get('3')
    if banana is not None:
        objects.pop(3, None)
        lower, upper = np.asarray(banana['visible_bounds'])
        if (banana['shape_cost'] <= 3.0 and banana['pixels'] >= 180
                and .13 < upper[0]-lower[0] < .25):
            # The asset's collision hull spans its visual concavity. The near
            # end is narrower than the middle across the 56-mm finger slab.
            objects[3] = np.array([lower[0]+.053, (lower[1]+upper[1])/2.-.004, TABLE_TOP_Z+.0255])
    return objects


class AlgSolution(BaseSolution):
    def __init__(self):
        super().__init__(perception_fn=estimate_grasp_contacts)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._descent_positions = deque(maxlen=35)

    def _set_state(self, state):
        super()._set_state(state)
        self._descent_positions.clear()

    def _plan_motion(self, goal, q, *, start=None, spacing=.025):
        goal = np.array(goal, dtype=float, copy=True)
        if self.current_object == 3 and self.state == 'CLOSE':
            return self._plan_gentle_lift(goal, q)
        if (self.current_object == 2 and self.state in ('CLOSE', 'VERIFY_LIFT')
                and abs(goal[2]-(TABLE_TOP_Z+.27)) < 1e-4):
            # The neck-held bottle hangs below the pinch point. Clear the
            # basket rim with its bottom before translating into the basket.
            goal[2] = TABLE_TOP_Z+.36
        return super()._plan_motion(goal, q, start=start, spacing=spacing)

    def _plan_gentle_lift(self, goal, q):
        initial_rotation = fk(q)[:3, :3]
        start = self._pinch_from_joints(q)
        height = max(.001, goal[2]-start[2])
        count = max(1, int(np.ceil(height/.007)))
        seed = q[:6].copy()
        joints, angles = [], []
        previous_angle = 0.
        for alpha in np.linspace(1./count, 1., count):
            contact = start+alpha*(goal-start)
            risen = contact[2]-start[2]
            # Lift the newly closed grasp vertically first, then move toward
            # the robot with the smallest tilt needed by the finite wrist.
            contact[0] += .055*np.clip((risen-.06)/max(height-.06, .001), 0., 1.)
            fit = None
            for angle in (previous_angle, previous_angle+1., previous_angle+2.):
                if angle > 35.:
                    continue
                rotation = Rotation.from_euler('y', angle, degrees=True).as_matrix() @ initial_rotation
                position = contact-self.GRASP_DEPTH*rotation[:, 2]
                candidate = solve_ik(position, rotation, seed, multi_start=False,
                                     orientation_weight=.4, max_nfev=70,
                                     position_tolerance=.002, orientation_tolerance=.025)
                if candidate.success:
                    fit, previous_angle = candidate, angle
                    break
            if fit is None:
                self._log('gentle_lift_unreachable', contact=contact.tolist(), previous_angle=previous_angle)
                return False
            seed = fit.joints
            joints.append(seed.copy())
            angles.append(previous_angle)
        self._waypoints = joints
        self._waypoint_index = 0
        self._motion_goal = contact.copy()
        self._motion_settle = self._motion_steps = 0
        self._log('gentle_lift_planned', waypoints=len(joints), final_tilt=angles[-1],
                  initial_fixed_segments=angles.count(0.))
        return True

    def _start_descent(self, q, obs):
        if self.current_object != 3:
            return super()._start_descent(q, obs)
        # Keep the clear-view median: fingers occlude the near endpoint above
        # the banana, making a new bounding-box estimate biased.
        if self._plan_motion(self._contact, q, spacing=.010):
            self._set_state('DESCEND')
        else:
            self._retract(q, 'banana_descent_unreachable')

    def _motion_command(self, q, *, slow=False):
        if self.current_object == 3 and self.state in ('LIFT', 'TRANSPORT'):
            slow = True
        arm, reached = super()._motion_command(q, slow=slow)
        if self.current_object == 2 and self.state in ('LIFT', 'TRANSPORT', 'PLACE'):
            arm = q[:6] + np.clip(arm-q[:6], -.025, .025)
        if self.state == 'DESCEND' and not reached:
            self._descent_positions.append(q[:6].copy())
            if len(self._descent_positions) == 35 and self._motion_steps >= 90:
                motion = float(np.max(np.ptp(np.asarray(self._descent_positions), axis=0)))
                difference = self._pinch_from_joints(q)-self._contact
                if motion < .020 and np.linalg.norm(difference[:2]) < .025 and -.01 <= difference[2] <= .085:
                    self._log('contact_stop', joint_motion=motion, contact_error=difference.tolist())
                    self._hold_q = q[:6].copy()
                    return q[:6].copy(), True
        return arm, reached

    def _action(self, arm, closed=False):
        target = np.empty(8)
        target[:6] = arm
        close = -.025 if self.current_object == 2 else -.015
        target[6:] = [close, -close] if closed else [.035, -.035]
        return {'action': action_from_joint_targets(target).tolist(), 'giveup': bool(self._done)}
