"""Fast synthetic smoke drive of ContactGraspPolicy (no physics, no GPU).

Reuses the frozen audit_grasp_probe fixture/plant conventions. Passing proves
only that the added phases execute and sequence; it is NOT grasp evidence.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(ROOT))
from task_b.audit_first_reach import fixture
from task_b.arm_kinematics import ARM_JOINT_NAMES, solve_ik
from task_b.control import ARM_TERM, LEG_TERM, WHEEL_TERM
from task_b.payload_motion import PayloadMotionPolicy
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

DT = .02
JAW_HALF = .02   # synthetic obstruction half-width; not a contact model


def build():
    schema, names, defaults = fixture()
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .035, -.035])))
    # Original scene default_joint_pos, read from a recorded environment_metadata
    # (offline static asset data, never a runtime input).
    for side, hip, thigh, calf in (('FR', -.1, .8, -1.5), ('FL', .1, .8, -1.5),
                                   ('RR', -.1, 1., -1.5), ('RL', .1, 1., -1.5)):
        defaults[f'{side}_hip_joint'] = hip
        defaults[f'{side}_thigh_joint'] = thigh
        defaults[f'{side}_calf_joint'] = calf
    terms = tuple(replace(term, joint_names=term.joint_names[::-1])
                  if term.name == ARM_TERM else term for term in schema.terms)
    schema = replace(schema, terms=terms,
                     default_joint_pos=np.array([defaults[name] for name in names]))
    return schema, names, defaults


def main():
    schema, names, defaults = build()
    arm, leg, wheel = (schema.term(name) for name in (ARM_TERM, LEG_TERM, WHEEL_TERM))
    point = np.array([.5, .179, -.28])
    fit = solve_ik(point + [0., 0., .12], seed=[0., 1.1, -.8, 0., .95, 0.], max_nfev=120)
    images = {key: value for source in ('ee', 'head') for key, value in (
        (source + '_rgb', np.zeros((24, 32, 3), np.uint8)),
        (source + '_depth', np.ones((24, 32))))}

    def detector(*unused, **kwargs):
        return [{'body_point': point.tolist(), 'height_span_m': .18,
                 'forward_planar_distance_m': float(np.linalg.norm(point[:2])),
                 'optical_depth_m': 1., 'depth_samples': 100, 'pixel_area': 100,
                 'pixel_uv': [16., 12.], 'bbox_xywh': [11, 2, 10, 20],
                 'aspect': 2., 'kind': 'synthetic_supported_yellow_component'}], {
                     'yellow_components': 1, 'rejected': {},
                     'association_mode': 'synthetic_detector_output'}

    class PositionPlant:
        def __init__(self):
            self.q = dict(defaults)
            self.qdot = dict.fromkeys(names, 0.)

        def observe(self):
            obs = np.zeros(84)
            obs[11] = -1.
            for index, name in enumerate(names):
                obs[12 + index] = self.q[name] - defaults[name]
                obs[36 + index] = self.qdot[name]
            return obs

        def respond(self, action):
            old = dict(self.q)
            for term in (arm, leg):
                for name, value in zip(term.joint_names, action[term.start:term.stop]):
                    index = ARM_JOINT_NAMES.index(name) if name in ARM_JOINT_NAMES else None
                    target = defaults[name] + float(value) * term.scale
                    if index is not None and index < 6:
                        target = float(np.clip(target, JOINT_LOWER[index], JOINT_UPPER[index]))
                    self.q[name] = target
            # Synthetic jaw obstruction so the inherited close is not empty and the
            # child can progress to its own PROBE_LIFT. This is NOT a contact model
            # and NOT an object: it only keeps the width finite on CPU.
            self.q['arm_joint7'] = float(np.clip(self.q['arm_joint7'], JAW_HALF, .035))
            self.q['arm_joint8'] = float(np.clip(self.q['arm_joint8'], -.035, -JAW_HALF))
            self.qdot = {name: (self.q[name] - old[name]) / DT for name in names}

        def step(self, policy):
            obs = self.observe()
            action = policy.act(obs, images)
            self.respond(action)
            return np.asarray(action), obs

    trace, seen = [], []
    with patch('task_b.first_reach.detect_yellow_candidates', detector), \
            patch('task_b.visual_approach.detect_yellow_candidates', detector), \
            patch('task_b.first_reach.solve_ik', return_value=fit):
        policy = PayloadMotionPolicy(schema, names, defaults, settle_calls=0, ramp_calls=1,
                                     forward_cmd=.05, turn_cap=.6, standoff=.50, turn_gain=2.,
                                     lowering_m=.02, open_on='close_complete')
        plant = PositionPlant()
        last_state, prev_leg = None, None
        for call in range(9000):
            if policy.done_reason:
                break
            action, _ = plant.step(policy)
            state = policy.state
            if state != last_state:
                debug = policy.debug or {}
                trace.append({'call': call, 'state': state, 'reason': policy.state_reason,
                              'q6': round(float(plant.q['arm_joint6']), 5),
                              'width': round(float(plant.q['arm_joint7'] - plant.q['arm_joint8']), 5),
                              'leg_action': [round(float(v), 5) for v in action[leg.start:leg.stop]]})
                seen.append(state)
                last_state = state
        report = {
            'module': 'task_b.payload_motion.PayloadMotionPolicy(open_on=close_complete)',
            'synthetic': True,
            'is_grasp_evidence': False,
            'calls_executed': call + 1,
            'final_state': policy.state,
            'done_reason': policy.done_reason,
            'states_seen': seen,
            'contact_phases_reached': [s for s in seen if s.startswith('CONTACT_')],
            'payload_phases_reached': [x for x in seen if x.startswith('PAYLOAD_')],
            'plan_error': (policy.debug or {}).get('plan_error'),
            'plan_ik': ((policy.debug or {}).get('plan_diagnostics') or {}).get('ik'),
            'goals_rad': (policy.debug or {}).get('goals_rad'),
            'tracker': (policy.debug or {}).get('tracker'),
            'final_debug': {k: (policy.debug or {}).get(k) for k in (
                'actual_goal_error_rad', 'arm_qdot_rad_s', 'quiet_ready', 'quiet_tick',
                'body_quiet', 'arm_window_span_rad', 'segment_age_s', 'segment_deadline_s')},
            'transitions': trace,
            'final_q6': round(float(plant.q['arm_joint6']), 5),
            'final_width': round(float(plant.q['arm_joint7'] - plant.q['arm_joint8']), 5),
        }
    out = Path('/home/lybm/ATEC_Experiments_20260910/task_b_takeover_20260914/smoke_contact_grasp.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'transitions'}, indent=2))
    for entry in trace:
        print(entry)


if __name__ == '__main__':
    main()
