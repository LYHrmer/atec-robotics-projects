"""Collect aligned teacher servo samples through the existing public policy API."""
from __future__ import annotations

import atexit
import argparse
import json
import os
from pathlib import Path
import numpy as np

from solution import AlgSolution as Teacher
from solution_task_e_vision import _numpy
from tools.task_e.il.common import FEATURE_NAMES, INTERFACE_VERSION, motion_settings, servo_features, sha256


class CollectingSolution(Teacher):
    def __init__(self):
        self.data_dir = Path(os.environ['ATEC_IL_DATA_DIR']).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=False)
        self.episode_id = os.environ.get('ATEC_IL_EPISODE_ID', self.data_dir.name)
        self._il_buffer = []
        self._il_chunks = self._il_samples = self._il_contact_stops = 0
        self._il_qdot = np.zeros(6)
        root = Path(__file__).resolve().parents[3]
        source_files = ['solution.py', 'solution_baseline.py', 'solution_task_e_rgbd.py',
                        'solution_task_e_vision.py', 'task_e_geometry.py', 'task_e_perception.py',
                        'task_e_held_motion.py', 'tools/task_e/il/common.py', 'tools/task_e/il/collect.py']
        self._il_metadata = {'interface_version': INTERFACE_VERSION, 'episode_id': self.episode_id,
                             'feature_names': FEATURE_NAMES, 'label': 'teacher arm target minus pre-action q, divided by joint limit',
                             'source_hashes': {name: sha256(root/name) for name in source_files},
                             'scope': 'Only learned servo supervision; hand-coded RGBD perception, IK path, state machine and gripper remain.',
                             'uses_simulator_truth': False, 'sample_alignment': 'features and labels from same pre-env.step observation'}
        seed_parser = argparse.ArgumentParser(add_help=False)
        seed_parser.add_argument('--seed', type=int, default=42)
        self._il_metadata['rollout_seed'] = seed_parser.parse_known_args()[0].seed
        super().__init__()
        atexit.register(self.flush)
        self.flush()

    def predicts(self, obs, current_score):
        proprio = _numpy(obs['proprio']).reshape(-1)
        if proprio.size != 24 or not np.isfinite(proprio).all():
            raise ValueError('Expected finite Task E 24-dimensional proprioception')
        self._il_qdot = proprio[8:14].copy()
        response = super().predicts(obs, current_score)
        if self.total_steps % 250 == 0 or response['giveup']:
            self.flush()
        return response

    def _motion_command(self, q, *, slow=False):
        phase, object_id = self.state, self.current_object
        _, limit = motion_settings(self, slow)
        arm, reached = super()._motion_command(q, slow=slow)
        if self._waypoints:
            target = np.asarray(self._waypoints[self._waypoint_index]).copy()
            # The existing descent contact-stop is a rule-based safety transition,
            # retained in the student. It is not a servo training label.
            contact_stop = (phase == 'DESCEND' and reached
                            and np.max(np.abs(np.asarray(arm)-q[:6])) < 1e-10
                            and np.max(np.abs(target-q[:6])) >= .025)
            if contact_stop:
                self._il_contact_stops += 1
            else:
                features = servo_features(q, self._il_qdot, target, limit, phase, object_id)
                delta = np.asarray(arm)-q[:6]
                if np.max(np.abs(delta)) > limit + 1e-6:
                    raise ValueError('Teacher action exceeds the declared servo limit')
                self._il_buffer.append({'features': features, 'label': delta/limit, 'delta': delta,
                                        'q': q.copy(), 'qdot': self._il_qdot.copy(), 'target': target,
                                        'limit': limit, 'phase': phase, 'object_id': object_id or 0,
                                        'step': self.total_steps, 'score': self._last_score})
                self._il_samples += 1
        return arm, reached

    def flush(self):
        if self._il_buffer:
            path = self.data_dir/f'chunk_{self._il_chunks:04d}.npz'
            with path.open('xb') as handle:
                np.savez_compressed(handle, **{key: np.asarray([row[key] for row in self._il_buffer])
                                               for key in self._il_buffer[0]})
            self._il_chunks += 1
            self._il_buffer.clear()
        metadata = dict(self._il_metadata, chunks=self._il_chunks, samples=self._il_samples,
                        rule_contact_stops_excluded=self._il_contact_stops,
                        last_observed_score=getattr(self, '_last_score', 0.),
                        last_policy_step=getattr(self, 'total_steps', 0),
                        finalized_before_sim_shutdown=getattr(self, '_il_finalized', False))
        temporary = self.data_dir/'metadata.tmp'
        temporary.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary.replace(self.data_dir/'metadata.json')
        print('TASK_E_IL_DATA '+json.dumps({'episode_id': self.episode_id, 'samples': self._il_samples,
                                          'chunks': self._il_chunks}), flush=True)

    def finalize(self):
        """The evaluator must call this before Kit/Isaac shutdown, which can skip atexit."""
        self._il_finalized = True
        self.flush()
