"""Release-audit isolation tests using NumPy, without Isaac Sim or PyTorch.

The evaluator launches the SDK at import time. Extract only the actual audit
function from its AST; its action arithmetic runs on the small CPU tensor
adapter below. These tests establish accounting/diagnostic behavior, not
physical release success.
"""
import ast
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


class CpuTensor(np.ndarray):
    """Only the tensor operations needed by the production audit function."""

    def __new__(cls, values, dtype=float):
        return np.asarray(values, dtype=dtype).view(cls)

    @property
    def device(self):
        return 'cpu'

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self.copy()

    def numpy(self):
        return np.asarray(self)

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis).view(type(self))


def load_audit_function():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'eval_task_e.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'post_terminal_release_audit')
    cpu_torch = SimpleNamespace(
        as_tensor=lambda value, device=None, dtype=float: CpuTensor(value, dtype=dtype),
        clamp=np.clip,
        isfinite=np.isfinite,
        inference_mode=nullcontext,
    )
    namespace = {'np': np, 'torch': cpu_torch, 'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['post_terminal_release_audit']


class Scene(dict):
    # A nonzero environment origin catches world/local membership confusion.
    env_origins = CpuTensor([[5., 2., 0.]])


class FakeEnvironment:
    def __init__(self, outside_step=None):
        default = CpuTensor([[0., 1.2, -1.5, 0., 1.2, 0., .035, -.035]])
        q = default.clone()
        q[0, :6] += .03
        q[0, 6:] = [.022, -.024]
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=default))
        scene = Scene(robot=self.robot)
        for name in ('object_1', 'object_2', 'object_3'):
            scene[name] = SimpleNamespace(data=SimpleNamespace(
                root_pos_w=CpuTensor([[6.08, 1.7, .90]])))
        cfg = SimpleNamespace(
            actions=SimpleNamespace(joint_arm=SimpleNamespace(scale=.5, use_default_offset=True)),
            terminations=SimpleNamespace(basket_success=SimpleNamespace(params={
                'center': (1.08, -.3, .74), 'half_x': .2, 'half_y': .11, 'table_top_z': .8266,
            })),
        )
        self.unwrapped = SimpleNamespace(scene=scene, cfg=cfg, step_dt=.02)
        self.calls = 0
        self.outside_step = outside_step

    def step(self, action):
        self.calls += 1
        self.robot.data.joint_pos[:] = self.robot.data.default_joint_pos + .5 * action
        obj = self.unwrapped.scene['object_3']
        obj.data.root_pos_w[0, 0] = 7. if self.calls == self.outside_step else 6.08
        # Deliberately large rewards and repeated termination must be ignored.
        return {}, 999., True, False, {}


class PostTerminalReleaseTests(unittest.TestCase):
    def run_audit(self, env, steps, *, running=lambda: True):
        official = {'score': 18., 'steps': 2689, 'sim_seconds': 53.78}
        original = copy.deepcopy(official)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'result.json').write_text('OFFICIAL RESULT')
            (output / 'telemetry.npz').write_bytes(b'OFFICIAL TELEMETRY')
            with redirect_stdout(io.StringIO()):
                report = load_audit_function()(env, output, steps, official, running)
            with np.load(output / 'post_terminal_release.npz') as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            self.assertEqual((output / 'result.json').read_text(), 'OFFICIAL RESULT')
            self.assertEqual((output / 'telemetry.npz').read_bytes(), b'OFFICIAL TELEMETRY')
            self.assertEqual(official, original)
            self.assertEqual(json.loads((output / 'post_terminal_release.json').read_text()), report)
        return report, arrays

    def test_150_steps_frozen_arm_ramp_scale_local_bounds_and_isolation(self):
        env = FakeEnvironment()
        report, data = self.run_audit(env, 150)
        self.assertEqual(env.calls, 150)
        self.assertEqual(data['qpos'].shape, (151, 8))
        self.assertLessEqual(np.max(abs(np.diff(data['target_qpos'][:, 6:], axis=0))), .00200001)
        np.testing.assert_allclose(data['target_qpos'][:, :6],
                                   np.broadcast_to(data['target_qpos'][0, :6], (151, 6)))
        np.testing.assert_allclose(data['qpos'], data['target_qpos'])
        np.testing.assert_allclose(data['target_qpos'][-1, 6:], [.035, -.035])
        self.assertTrue(report['final_all_inside'])
        self.assertTrue(report['final_jaws_open'])
        self.assertTrue(report['continuous_last_1s_all_inside'])
        self.assertEqual(report['diagnostic_sim_seconds'], 3.)

    def test_last_second_transient_exit_not_hidden_by_final_membership(self):
        report, _ = self.run_audit(FakeEnvironment(outside_step=120), 150)
        self.assertTrue(report['final_all_inside'])
        self.assertFalse(report['continuous_last_1s_all_inside'])
        self.assertFalse(report['continuous_last_1s_membership']['object_3'])

    def test_short_audit_does_not_claim_one_second_continuous_window(self):
        report, data = self.run_audit(FakeEnvironment(), 20)
        self.assertIsNone(report['continuous_last_1s_all_inside'])
        self.assertIsNone(report['continuous_window_seconds'])
        self.assertEqual(len(data['qpos']), 21)

    def test_application_stop_is_bounded_and_retains_initial_record(self):
        env = FakeEnvironment()
        report, data = self.run_audit(env, 150, running=lambda: False)
        self.assertEqual(env.calls, 0)
        self.assertEqual(report['executed_steps'], 0)
        self.assertEqual(report['stop_reason'], 'app_stopped')
        self.assertEqual(len(data['qpos']), 1)
        self.assertIsNone(report['continuous_last_1s_all_inside'])


if __name__ == '__main__':
    unittest.main()
