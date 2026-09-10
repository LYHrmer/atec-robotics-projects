"""BC servo candidate with no teacher-action fallback during moving phases."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import numpy as np
import torch

from solution import AlgSolution as PlanningAndGraspingPolicy
from solution_task_e_vision import _numpy
from tools.task_e.il.common import FEATURE_NAMES, INTERFACE_VERSION, make_model, motion_settings, servo_features, sha256


class LearnedServoSolution(PlanningAndGraspingPolicy):
    def __init__(self):
        self._il_checkpoint = Path(os.environ['ATEC_IL_CHECKPOINT']).expanduser().resolve(strict=True)
        checkpoint = torch.load(self._il_checkpoint, map_location='cpu', weights_only=True)
        if (checkpoint['interface_version'] != INTERFACE_VERSION
                or checkpoint['feature_names'] != FEATURE_NAMES):
            raise ValueError('Incompatible BC servo checkpoint')
        self._il_model = make_model().eval()
        self._il_model.load_state_dict(checkpoint['model_state_dict'])
        self._il_qdot = np.zeros(6)
        self._il_rows = []
        self._il_total_calls = self._il_motion_calls = self._il_learned_calls = self._il_contact_stops = 0
        self._il_stats_path = Path(os.environ['ATEC_IL_METRICS']).expanduser().resolve()
        if self._il_stats_path.exists():
            raise FileExistsError(self._il_stats_path)
        self._il_stats_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        atexit.register(self.save_metrics)
        self.save_metrics()

    def predicts(self, obs, current_score):
        proprio = _numpy(obs['proprio']).reshape(-1)
        if proprio.size != 24 or not np.isfinite(proprio).all():
            raise ValueError('Expected finite Task E 24-dimensional proprioception')
        self._il_qdot = proprio[8:14].copy()
        self._il_total_calls += 1
        response = super().predicts(obs, current_score)
        if self._il_total_calls % 250 == 0 or response['giveup']:
            self.save_metrics()
        return response

    def _motion_command(self, q, *, slow=False):
        # Keep waypoint acceptance and descent-contact state transitions. The
        # actual six-joint command is produced exclusively by the trained MLP.
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
        features = servo_features(q, self._il_qdot, target, limit, self.state, self.current_object)
        with torch.inference_mode():
            delta = self._il_model(torch.from_numpy(features)).squeeze(-1).numpy()*limit
        if not np.isfinite(delta).all() or np.max(np.abs(delta)) > limit+1e-6:
            raise ValueError('Invalid learned servo output; refusing teacher fallback')
        arm = q[:6]+delta
        self._il_rows.append({'step':self.total_steps, 'features':features, 'delta':delta.copy(),
                              'q':np.asarray(q).copy(), 'target':np.asarray(target).copy(),
                              'limit':limit, 'phase':self.state, 'object_id':self.current_object or 0})
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
        self._il_learned_calls += 1
        return arm, reached

    def save_metrics(self):
        if self._il_rows:
            servo_path = self._il_stats_path.with_suffix('.servo.npz')
            temporary_trace = servo_path.with_suffix('.tmp')
            with temporary_trace.open('wb') as handle:
                np.savez_compressed(handle, **{key:np.asarray([row[key] for row in self._il_rows])
                                              for key in self._il_rows[0]})
            temporary_trace.replace(servo_path)
        result = {'experiment': 'hierarchical BC of six-joint arm servo only',
                  'checkpoint': str(self._il_checkpoint), 'checkpoint_sha256': sha256(self._il_checkpoint),
                  'policy_calls': self._il_total_calls, 'moving_phase_calls': self._il_motion_calls,
                  'learned_arm_action_calls': self._il_learned_calls,
                  'learned_fraction_all_calls': self._il_learned_calls/max(1,self._il_total_calls),
                  'learned_fraction_moving_calls': self._il_learned_calls/max(1,self._il_motion_calls),
                  'teacher_action_fallback_calls': 0, 'rule_contact_stop_calls': self._il_contact_stops,
                  'finalized_before_sim_shutdown':getattr(self,'_il_finalized',False),
                  'servo_input_output_trace':str(self._il_stats_path.with_suffix('.servo.npz')),
                  'last_observed_score': getattr(self,'_last_score',0.),
                  'not_learned': ['RGBD perception', 'IK path planning', 'waypoint acceptance',
                                  'grasp/release state machine', 'gripper', 'initial homing', 'holding'],
                  'official_result': 'Use evaluator result.json; last observed score can precede terminal reward.',
                  'uses_simulator_truth': False}
        temporary = self._il_stats_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(self._il_stats_path)
        print('TASK_E_IL_POLICY '+json.dumps(result),flush=True)

    def finalize(self):
        self._il_finalized = True
        self.save_metrics()
