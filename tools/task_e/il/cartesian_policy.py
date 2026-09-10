"""Cartesian-goal residual servo student: analytic DLS plus a learned residual.

The high-level machinery of the per-joint BC candidate is reused, but the
checkpoint schema, the network inputs and the commanded action are different:
the joint-space waypoint is hidden from the network behind an FK Cartesian
goal, and the network only shifts the analytic DLS step within a bounded
residual. Moving phases never call the teacher servo and never fall back to it.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import numpy as np
import torch

from solution import AlgSolution as PlanningAndGraspingPolicy
from tools.task_e.il.cartesian import (CARTESIAN_FEATURE_NAMES, CARTESIAN_INTERFACE_VERSION,
                                       analytic_step, cartesian_features, command_from_residual,
                                       goal_from_joint_target, make_cartesian_model)
from tools.task_e.il.common import motion_settings, sha256
from tools.task_e.il.policy import LearnedServoSolution


class CartesianResidualSolution(LearnedServoSolution):
    """Reuses the prediction/metrics plumbing; loads its own checkpoint schema."""

    def __init__(self):
        self._il_checkpoint = Path(os.environ['ATEC_IL_CHECKPOINT']).expanduser().resolve(strict=True)
        checkpoint = torch.load(self._il_checkpoint, map_location='cpu', weights_only=True)
        if (checkpoint.get('cartesian_interface_version') != CARTESIAN_INTERFACE_VERSION
                or list(checkpoint.get('feature_names', ())) != list(CARTESIAN_FEATURE_NAMES)):
            raise ValueError('Incompatible Cartesian residual servo checkpoint')
        self._il_residual_scale = np.asarray(checkpoint['residual_scale'], dtype=float)
        if self._il_residual_scale.shape != (6,) or not np.all(self._il_residual_scale > 0.):
            raise ValueError('Checkpoint residual scale must be six positive radians')
        self._il_model = make_cartesian_model().eval()
        self._il_model.load_state_dict(checkpoint['model_state_dict'])
        self._il_dls_only = os.environ.get('ATEC_IL_DLS_ONLY', '0') == '1'
        self._il_qdot = np.zeros(6)
        self._il_rows = []
        self._il_phase_calls = {}
        self._il_total_calls = self._il_motion_calls = self._il_learned_calls = 0
        self._il_dls_only_calls = self._il_contact_stops = self._il_effective_calls = 0
        self._il_rate_clamps = self._il_joint_clamps = 0
        self._il_analytic_sum = self._il_residual_sum = 0.
        self._il_stats_path = Path(os.environ['ATEC_IL_METRICS']).expanduser().resolve()
        if self._il_stats_path.exists():
            raise FileExistsError(self._il_stats_path)
        self._il_stats_path.parent.mkdir(parents=True, exist_ok=True)
        # Skip LearnedServoSolution.__init__: it loads the per-joint schema.
        PlanningAndGraspingPolicy.__init__(self)
        atexit.register(self.save_metrics)
        self.save_metrics()

    def _motion_command(self, q, *, slow=False):
        # Waypoint acceptance, settling and the descent contact stop are kept
        # exactly as in the rule-based controller. Only the six-joint command
        # changes: analytic DLS toward the Cartesian goal, plus a learned
        # residual, then physical guards.
        slow, limit = motion_settings(self, slow)
        self._motion_steps += 1
        self._il_motion_calls += 1
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
        self._hold_q = target.copy()
        reached = self._motion_settle >= 12

        goal_position, goal_rotation = goal_from_joint_target(target)
        analytic = analytic_step(q, goal_position, goal_rotation, limit)
        features = cartesian_features(q, self._il_qdot, goal_position, goal_rotation,
                                      analytic, limit, self.state, self.current_object)
        if self._il_dls_only:
            residual = np.zeros(6)
        else:
            with torch.inference_mode():
                output = self._il_model(torch.from_numpy(features)).numpy().astype(float)
            residual = output*self._il_residual_scale
            if (not np.isfinite(residual).all()
                    or np.max(np.abs(residual)-self._il_residual_scale) > 1e-6):
                raise ValueError('Invalid learned residual; refusing teacher fallback')
        arm, rate_clamped, joint_clamped = command_from_residual(q, analytic, residual, limit)
        baseline, _, _ = command_from_residual(q, analytic, np.zeros(6), limit)
        self._il_rate_clamps += int(rate_clamped)
        self._il_joint_clamps += int(joint_clamped)
        self._il_analytic_sum += float(np.mean(np.abs(analytic)))
        self._il_residual_sum += float(np.mean(np.abs(residual)))
        self._il_effective_calls += int(np.max(np.abs(arm-baseline)) > 1e-9)
        self._il_phase_calls[self.state] = self._il_phase_calls.get(self.state, 0)+1
        self._il_rows.append({'step': self.total_steps, 'features': features,
                              'analytic': analytic.copy(), 'residual': residual.copy(),
                              'command_delta': arm-q[:6], 'q': np.asarray(q).copy(),
                              'goal_position': goal_position, 'limit': limit,
                              'phase': self.state, 'object_id': self.current_object or 0})

        if self.state == 'DESCEND' and not reached:
            self._descent_positions.append(q[:6].copy())
            if len(self._descent_positions) == 35 and self._motion_steps >= 90:
                motion = float(np.max(np.ptp(np.asarray(self._descent_positions), axis=0)))
                difference = self._pinch_from_joints(q)-self._contact
                if motion < .020 and np.linalg.norm(difference[:2]) < .025 and -.01 <= difference[2] <= .085:
                    self._hold_q = q[:6].copy()
                    self._il_contact_stops += 1
                    self._log('il_rule_contact_stop', joint_motion=motion, contact_error=difference.tolist())
                    return q[:6].copy(), True
        if self._il_dls_only:
            self._il_dls_only_calls += 1
        else:
            self._il_learned_calls += 1
        return arm, reached

    def save_metrics(self):
        if self._il_rows:
            servo_path = self._il_stats_path.with_suffix('.servo.npz')
            temporary_trace = servo_path.with_suffix('.tmp')
            with temporary_trace.open('wb') as handle:
                np.savez_compressed(handle, **{key: np.asarray([row[key] for row in self._il_rows])
                                               for key in self._il_rows[0]})
            temporary_trace.replace(servo_path)
        commanded = self._il_learned_calls+self._il_dls_only_calls
        result = {'experiment': 'Cartesian-goal residual BC of six-joint arm servo only',
                  'mode': 'dls_only_ablation' if self._il_dls_only else 'learned_residual',
                  'checkpoint': str(self._il_checkpoint), 'checkpoint_sha256': sha256(self._il_checkpoint),
                  'residual_scale_rad': self._il_residual_scale.tolist(),
                  'policy_calls': self._il_total_calls, 'moving_phase_calls': self._il_motion_calls,
                  'servo_command_calls': commanded,
                  'learned_residual_action_calls': self._il_learned_calls,
                  'dls_only_action_calls': self._il_dls_only_calls,
                  'residual_changed_command_calls': self._il_effective_calls,
                  'learned_fraction_all_calls': self._il_learned_calls/max(1, self._il_total_calls),
                  'learned_fraction_moving_calls': self._il_learned_calls/max(1, self._il_motion_calls),
                  'teacher_action_fallback_calls': 0,
                  'rule_contact_stop_calls': self._il_contact_stops,
                  'rate_limit_clamped_calls': self._il_rate_clamps,
                  'joint_limit_clamped_calls': self._il_joint_clamps,
                  'mean_abs_analytic_delta_rad': self._il_analytic_sum/max(1, commanded+self._il_contact_stops),
                  'mean_abs_residual_rad': self._il_residual_sum/max(1, commanded+self._il_contact_stops),
                  'servo_calls_by_phase': dict(sorted(self._il_phase_calls.items())),
                  'finalized_before_sim_shutdown': getattr(self, '_il_finalized', False),
                  'servo_input_output_trace': str(self._il_stats_path.with_suffix('.servo.npz')),
                  'last_observed_score': getattr(self, '_last_score', 0.),
                  'not_learned': ['RGBD perception', 'IK path planning', 'Cartesian goal from FK',
                                  'analytic DLS step', 'waypoint acceptance', 'rule contact stop',
                                  'grasp/release state machine', 'gripper', 'initial homing', 'holding',
                                  'rate and joint-limit guards'],
                  'learned': 'bounded six-joint residual added to the analytic DLS step',
                  'official_result': 'Use evaluator result.json; last observed score can precede terminal reward.',
                  'uses_simulator_truth': False}
        temporary = self._il_stats_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(self._il_stats_path)
        print('TASK_E_IL_CARTESIAN '+json.dumps(result), flush=True)
