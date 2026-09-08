"""Independent Task E candidate with continuous sugarbox transfer planning.

All sensing, grasping, object-specific lift paths and control outputs come from
the RGB-D controller. Only the sugarbox's carry and placement plans use the
reviewed Opus helper. Load this file with tools/eval_task_e.py --solution.
"""
from task_e_held_motion import HeldMotionMixin
from solution_task_e_rgbd import AlgSolution as RGBDSolution


class AlgSolution(HeldMotionMixin, RGBDSolution):
    def _plan_motion(self, goal, q, *, start=None, spacing=.025):
        if self.current_object == 1 and self.state in ('VERIFY_LIFT', 'TRANSPORT'):
            return self._plan_held_motion(goal, q, spacing=spacing)
        return super()._plan_motion(goal, q, start=start, spacing=spacing)
