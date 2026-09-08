"""Deterministic geometry regression tests; no Isaac Sim or injected score.

Run: python -m unittest discover -s tests -p test_geometry.py -v
These establish geometry/interface correctness, not grasp success in physics.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import task_e_geometry as geo


class PiperKinematicsTests(unittest.TestCase):
    def test_zero_pose_matches_independent_usd_export(self):
        # UsdGeom.XformCache GetLocalToWorldTransform('/piper/gripper_base'),
        # transposed into column-vector convention by export_piper_kinematics.py.
        # This fixture is authored USD data, not an expected value made by fk().
        authored = np.array([
            [.08709366995972789, .000010267292873801598, .9962001267554268, .05614253133535385],
            [-.000004146302503482003, .9999999999419629, -.00000994396196643557, -.00000005960464477539063],
            [-.9962001267997078, -.0000032644909379382905, .08709366999724455, .2131931334733963],
            [0., 0., 0., 1.],
        ])
        actual = geo.fk(np.zeros(6), base_position=np.zeros(3),
                        base_quaternion=[1., 0., 0., 0.])
        np.testing.assert_allclose(actual, authored, atol=7e-7, rtol=0.)

    def test_full_spatial_jacobian_matches_finite_difference(self):
        epsilon = 1e-6
        for q in (np.array([-.5, 1.2, -1.5, .2, .8, .2]),
                  np.array([.4, 2.1, -2.3, -.7, -.4, 1.1])):
            with self.subTest(joints=q.tolist()):
                _, analytic = geo.fk(q, return_jacobian=True)
                numerical = np.empty((6, 6))
                for index in range(6):
                    delta = np.eye(6)[index] * epsilon
                    positive, negative = geo.fk(q + delta), geo.fk(q - delta)
                    numerical[:3, index] = (positive[:3, 3] - negative[:3, 3]) / (2 * epsilon)
                    numerical[3:, index] = Rotation.from_matrix(
                        positive[:3, :3] @ negative[:3, :3].T).as_rotvec() / (2 * epsilon)
                np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=0.)

    def test_grasp_ik_selected_workspace_points_and_bounds(self):
        # One far-side and one near-side point for the three asset geometries.
        cases = ((.90, .29, .045, [-1., 0.]),
                 (1.10, .25, .045, [-1., 0.]),
                 (.90, .20, .110, [0., -1.]),
                 (1.10, .14, .110, [0., -1.]),
                 (.90, .09, .030, [0., -1.]),
                 (1.10, .03, .030, [0., -1.]))
        for x, y, height, jaw in cases:
            with self.subTest(x=x, y=y, height=height):
                contact = np.array([x, y, geo.TABLE_TOP_Z + height])
                result = geo.solve_grasp_ik(contact, jaw)
                self.assertTrue(result.ik.success, result)
                self.assertTrue(np.all(result.ik.joints >= geo.JOINT_LOWER))
                self.assertTrue(np.all(result.ik.joints <= geo.JOINT_UPPER))
                pose = geo.fk(result.ik.joints)
                self.assertLess(np.linalg.norm(pose[:3, 3] - result.position), .003)
                actual_contact = pose[:3, 3] + .115 * pose[:3, 2]
                self.assertLess(np.linalg.norm(actual_contact - contact), .008)
                self.assertGreaterEqual(result.finger_floor_clearance, .002)

    def test_unreachable_target_is_not_reported_successful(self):
        result = geo.solve_ik([5., 0., geo.TABLE_TOP_Z+.2], multi_start=False)
        self.assertFalse(result.success)
        self.assertGreater(result.position_error, 2.)
        with self.assertRaisesRegex(ValueError, 'below the table'):
            geo.solve_grasp_ik([1., .06, geo.TABLE_TOP_Z+.001], [0., -1.])

    def test_differential_step_reduces_error_without_violating_step_bound(self):
        q = np.array([-.4, 1.4, -1.5, .2, .7, .1])
        pose = geo.fk(q)
        desired = pose[:3, 3] + np.array([.008, -.006, .005])
        updated = geo.differential_ik(q, desired, pose[:3, :3], max_joint_delta=.03)
        self.assertLess(np.linalg.norm(geo.fk(updated)[:3, 3]-desired),
                        np.linalg.norm(pose[:3, 3]-desired))
        self.assertLessEqual(np.max(np.abs(updated-q)), .03+1e-12)


class ObservationGeometryTests(unittest.TestCase):
    def test_relative_joint_observation_and_action_scale(self):
        defaults = np.array([0., 1.2, -1.5, 0., 1.2, 0., .035, -.035])
        relative = np.array([.2, -.1, .3, -.4, -.2, .5, -.01, .01])
        # Extra proprio fields are velocities and previous actions, not joints.
        proprio = np.r_[relative, np.full(16, 123.)]
        np.testing.assert_allclose(geo.joints_from_proprio(proprio), defaults+relative)
        np.testing.assert_allclose(geo.action_from_joint_targets(defaults+relative), 2.*relative)
        np.testing.assert_allclose(geo.action_from_joint_targets(defaults), np.zeros(8))

    def test_camera_center_ray_matches_world_convention(self):
        k, transform = geo.video_camera_parameters()
        self.assertAlmostEqual(k[0, 0], 640.*24./20.955)
        np.testing.assert_array_equal(k[:2, 2], [320., 240.])
        # Independent analytic rotation around world +Y. Camera forward is +X.
        angle = 2.*np.arctan2(.290, .957)
        expected_forward = np.array([np.cos(angle), 0., -np.sin(angle)])
        np.testing.assert_allclose(transform[:3, 2], expected_forward, atol=1e-12)
        depth = np.zeros((480, 640))
        depth[240, 320] = 1.3
        actual = geo.unproject_depth(depth)
        expected = np.array([-.2, 0., geo.TABLE_TOP_Z+.8])+1.3*expected_forward
        np.testing.assert_allclose(actual[0], expected, atol=1e-12)

    def test_depth_roundtrip_filters_invalid_and_preserves_optical_depth(self):
        depth = np.zeros((480, 640, 1))
        expected_pixels = np.array([[41, 53], [240, 320], [431, 574]])
        depths = np.array([1.15, 1.4, 1.75])
        for (row, col), value in zip(expected_pixels, depths):
            depth[row, col, 0] = value
        depth[1, 2, 0], depth[2, 3, 0], depth[3, 4, 0] = np.nan, np.inf, -1.
        points, pixels = geo.unproject_depth(depth, return_pixels=True)
        uv, recovered = geo.project_world(points)
        np.testing.assert_array_equal(pixels, expected_pixels)
        np.testing.assert_allclose(uv, expected_pixels[:, ::-1], atol=1e-10)
        np.testing.assert_allclose(recovered, depths, atol=1e-12)
        # Off-axis pixels must not reinterpret optical depth as Euclidean range.
        radial = np.linalg.norm(points[0]-geo.VIDEO_CAM_POS)
        self.assertGreater(radial, depths[0]+.02)


if __name__ == '__main__':
    unittest.main()
