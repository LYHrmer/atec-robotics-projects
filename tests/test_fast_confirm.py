"""Observation-gate regressions; these are not physical grasp/score tests."""
import unittest
from unittest.mock import patch

import numpy as np

from task_e_geometry import DEFAULT_JOINT_POS
from solution import AlgSolution
from solution_baseline import AlgSolution as OpusSolution


def observation(width=.04):
    joints = DEFAULT_JOINT_POS.copy()
    joints[6:] = (width / 2., -width / 2.)
    return {'proprio': joints - DEFAULT_JOINT_POS}


class FastConfirmationTests(unittest.TestCase):
    def policy(self, state, *, object_id=2, elapsed=0):
        policy = AlgSolution()
        policy.debug = False
        policy.current_object = object_id
        policy._contact = np.array([1.0, .1, .85])
        policy._set_state(state)
        policy.state_step = elapsed
        policy._read_vision = lambda obs: {}
        return policy

    def test_close_minimum_and_stability_window(self):
        policy = self.policy('CLOSE')
        with patch.object(policy, '_plan_motion', return_value=True):
            for _ in range(34):
                policy.predicts(observation(), 0.)
                self.assertEqual(policy.state, 'CLOSE')
            policy.predicts(observation(), 0.)
        self.assertEqual(policy.state, 'LIFT')
        self.assertEqual(policy.total_steps, 35)
        self.assertEqual(policy.state_step, 0)
        self.assertFalse(policy.completed)

    def test_close_oscillation_nonfinite_and_original_timeout(self):
        for widths in ([.04, .045] * 5, [.04] * 9 + [float('nan')]):
            with self.subTest(widths=widths):
                policy = self.policy('CLOSE', elapsed=25)
                for width in widths:
                    policy.predicts(observation(width), 0.)
                self.assertEqual(policy.state, 'CLOSE')
                self.assertEqual(policy.state_step, 35)
        policy = self.policy('CLOSE', elapsed=69)
        with patch.object(policy, '_plan_motion', return_value=True):
            policy.predicts(observation(), 0.)
        self.assertEqual(policy.state, 'LIFT')  # unchanged original fallback

    def test_close_stable_empty_jaws_still_reject_grasp(self):
        policy = self.policy('CLOSE', elapsed=25)
        with patch.object(policy, '_plan_motion', return_value=True):
            for _ in range(10):
                policy.predicts(observation(0.), 0.)
        self.assertEqual(policy.state, 'RETRACT')
        self.assertEqual(policy._last_failure, 'empty_gripper')
        self.assertFalse(policy.completed)

    def test_lift_waits_for_score_width_and_ten_observations(self):
        for elapsed, width, score, expected in [
                (8, .04, 3., 'VERIFY_LIFT'),
                (9, .04, 0., 'VERIFY_LIFT'),
                (9, .005, 3., 'VERIFY_LIFT'),
                (9, .04, 3., 'TRANSPORT')]:
            with self.subTest(elapsed=elapsed, width=width, score=score):
                policy = self.policy('VERIFY_LIFT', elapsed=elapsed)
                with patch.object(policy, '_plan_motion', return_value=True):
                    policy.predicts(observation(width), score)
                self.assertEqual(policy.state, expected)
                self.assertFalse(policy.completed)

    def test_open_requires_observed_width_and_minimum_time(self):
        for elapsed, width, expected in [(18, .07, 'OPEN'),
                                         (19, .065, 'OPEN'),
                                         (19, .066, 'VERIFY_PLACE')]:
            with self.subTest(elapsed=elapsed, width=width):
                policy = self.policy('OPEN', elapsed=elapsed)
                policy.predicts(observation(width), 0.)
                self.assertEqual(policy.state, expected)
                self.assertFalse(policy.completed)

    def test_place_distinguishes_grasp_score_and_full_attempt_gain(self):
        for known, attempt_score, score, expected in [
                (False, 0., 3., 'VERIFY_PLACE'),
                (False, 0., 6., 'RETRACT'),
                (True, 3., 3., 'VERIFY_PLACE'),
                (True, 3., 6., 'RETRACT')]:
            with self.subTest(known=known, score=score):
                policy = self.policy('VERIFY_PLACE', elapsed=9)
                policy._grasp_was_known = known
                policy._attempt_score = attempt_score
                with patch.object(policy, '_plan_motion', return_value=True):
                    policy.predicts(observation(.07), score)
                self.assertEqual(policy.state, expected)
                self.assertEqual(bool(policy.completed), expected == 'RETRACT')

    def test_episode_reset_clears_history_and_does_not_skip_init(self):
        policy = self.policy('CLOSE', elapsed=34)
        policy._last_score = 6.
        policy._confirm_jaw_widths.extend([.04] * 10)
        policy.predicts(observation(), 0.)
        self.assertEqual(policy.state, 'INIT')
        self.assertEqual(policy.state_step, 1)
        self.assertEqual(policy.total_steps, 1)
        self.assertEqual(len(policy._confirm_jaw_widths), 0)
        self.assertFalse(policy.completed)

    def test_only_sugar_post_place_changes_planner_with_fallback(self):
        for object_id, state, held_ok in [(1, 'VERIFY_PLACE', True),
                                          (1, 'VERIFY_PLACE', False),
                                          (2, 'VERIFY_PLACE', True),
                                          (1, 'CLOSE', True)]:
            with self.subTest(object_id=object_id, state=state, held_ok=held_ok):
                policy = self.policy(state, object_id=object_id)
                with patch.object(policy, '_plan_held_motion', return_value=held_ok) as held:
                    with patch.object(OpusSolution, '_plan_motion', return_value=True) as base:
                        self.assertTrue(policy._plan_motion([1., -.3, 1.1], DEFAULT_JOINT_POS))
                special = object_id == 1 and state == 'VERIFY_PLACE'
                self.assertEqual(held.call_count, int(special))
                self.assertEqual(base.call_count, int(not special or not held_ok))


if __name__ == '__main__':
    unittest.main()
