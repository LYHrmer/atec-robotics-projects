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
from task_b.contact_grasp import ContactGraspPolicy
from task_e_geometry import JOINT_LOWER, JOINT_UPPER

DT = .02


def build():
    schema, names, defaults = fixture()
    defaults.update(dict(zip(ARM_JOINT_NAMES, [.07, 1.1, -.8, .01, .02, .03, .035, -.035])))
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
        policy = ContactGraspPolicy(schema, names, defaults, settle_calls=0, ramp_calls=1,
                                    forward_cmd=.05, turn_cap=.6, standoff=.50, turn_gain=2.,
                                    lowering_m=.02)
        plant = PositionPlant()
        last_state, prev_leg = None, None
        for call in range(4000):
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
            'module': 'task_b.contact_grasp.ContactGraspPolicy',
            'synthetic': True,
            'is_grasp_evidence': False,
            'calls_executed': call + 1,
            'final_state': policy.state,
            'done_reason': policy.done_reason,
            'states_seen': seen,
            'contact_phases_reached': [s for s in seen if s.startswith('CONTACT_')],
            'reached_probe_close_after_contact': 'PROBE_CLOSE' in seen[seen.index('CONTACT_LOWER') + 1:]
                                                 if 'CONTACT_LOWER' in seen else False,
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
