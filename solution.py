"""Experimental faster confirmations and continuous empty sugarbox retraction.

Only public observations/current_score enter this wrapper. The successful Opus
transfer controller still owns every transition, score check and failure path.
The physical effect of earlier contact confirmation requires simulation tests.
"""
from collections import deque

import numpy as np

from solution_task_e_vision import _numpy
from task_e_geometry import joints_from_proprio
from solution_baseline import AlgSolution as OpusSolution


class AlgSolution(OpusSolution):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._confirm_jaw_widths = deque(maxlen=10)

    def _set_state(self, state):
        super()._set_state(state)
        self._confirm_jaw_widths.clear()

    def _plan_motion(self, goal, q, *, start=None, spacing=.025):
        # This call happens after the base VERIFY_PLACE check and with open
        # jaws. Preserve the measured wrist pose while clearing the basket.
        if self.current_object == 1 and self.state == 'VERIFY_PLACE':
            if self._plan_held_motion(goal, q, spacing=spacing):
                return True
        return super()._plan_motion(goal, q, start=start, spacing=spacing)

    def predicts(self, obs, current_score):
        score = float(_numpy(current_score).reshape(-1)[0])
        # Let the base reset first on an episode/score reset. No old jaw window
        # or accelerated counter should be carried into its INIT observation.
        if score + 1e-4 < self._last_score:
            return super().predicts(obs, current_score)

        state = self.state
        actual_step = self.state_step + 1  # base increments on this observation
        q = joints_from_proprio(_numpy(obs['proprio']).reshape(-1))
        width = float(q[6] - q[7])
        threshold = None
        valid = bool(np.isfinite(score) and np.all(np.isfinite(q)))

        if state == 'CLOSE':
            # Range, not endpoint difference: an oscillation must not pass.
            self._confirm_jaw_widths.append(width)
            stable = (len(self._confirm_jaw_widths) == 10
                      and np.all(np.isfinite(self._confirm_jaw_widths))
                      and np.ptp(self._confirm_jaw_widths) < .0015)
            if valid and actual_step >= 35 and stable:
                # The base still rejects width < .006 as an empty grasp.
                threshold = 70
        elif state == 'VERIFY_LIFT':
            if (valid and actual_step >= 10 and width > .006
                    and score >= self._attempt_score + 2.9):
                threshold = 30
        elif state == 'OPEN':
            if valid and actual_step >= 20 and width > .065:
                threshold = 70
        elif state == 'VERIFY_PLACE':
            required_gain = 3. if self._grasp_was_known else 6.
            if (valid and actual_step >= 10
                    and score >= self._attempt_score + required_gain - .1):
                threshold = 35

        if threshold is None or actual_step >= threshold:
            return super().predicts(obs, current_score)

        # Advance only the base's existing confirmation gate. No score,
        # completion flag, motion waypoint or total_steps value is fabricated.
        self.state_step = threshold - 1
        self._log('fast_confirmation', actual_state_step=actual_step,
                  original_threshold=threshold, jaw_width=round(width, 6))
        try:
            return super().predicts(obs, current_score)
        finally:
            # Normally the base transitions and resets state_step to zero.
            # Preserve elapsed-time meaning if it returns early without one.
            if self.state == state:
                self.state_step = actual_step
