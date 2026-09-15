"""CPU-only visibility check for the existing first-reach IK candidate.

These are static planning cases, not simulator trajectories or success proof.
No official environment state enters a controller, and no GPU is initialized.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from task_b.arm_kinematics import (
    ee_camera_transform, fk, head_camera_transform, solve_ik,
)


def projection(point, pose, focal_length, width=640, height=480):
    camera = pose[:3, :3].T @ (point - pose[:3, 3])
    focal = width * focal_length / 20.955
    pixel = ([float(width / 2 + focal * camera[0] / camera[2]),
              float(height / 2 + focal * camera[1] / camera[2])]
             if camera[2] > 0 else None)
    in_fov = bool(pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height)
    return dict(camera_xyz_m=camera.tolist(), pixel_uv=pixel,
                optical_depth_above_detector_minimum=bool(camera[2] > .10),
                point_inside_image=in_fov,
                point_inside_observable_range=bool(in_fov and camera[2] > .10))


def main():
    repo = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
    cases = []
    target = np.array([.50, .179, -.280])
    for wrist in (.95, 1.21):
        q = np.array([0., 1.1, -.8, 0., wrist, 0.])
        fit = solve_ik(target + [0., 0., .12], seed=q, max_nfev=120)
        duration = float(np.max(np.abs(fit.joints - q)) / .30)
        samples = []
        for time in np.linspace(0., duration, 15):
            # Match the per-joint command slew limit, without claiming physical tracking.
            candidate = q + np.clip(fit.joints - q, -.30 * time, .30 * time)
            samples.append(dict(time_s=float(time), joints_rad=candidate.tolist(),
                                ee=projection(target, ee_camera_transform(candidate), 15.),
                                head=projection(target, head_camera_transform(), 24.)))
        cases.append(dict(initial_q_rad=q.tolist(), visual_target_body_m=target.tolist(),
                          assumed_up_body=[0., 0., 1.], goal_q_rad=fit.joints.tolist(),
                          requested_gripper_body_m=(target + [0., 0., .12]).tolist(),
                          ik_position_residual_m=fit.position_error,
                          predicted_gripper_to_visual_surface_m=float(np.linalg.norm(fk(fit.joints)[:3, 3] - target)),
                          minimum_command_duration_s=duration, samples=samples))
    sources = ['task_b/arm_kinematics.py', 'task_b/visual_approach.py', 'task_e_geometry.py']
    result = dict(
        status='static_cpu_candidate_only',
        assumptions=['640 x 480 public image geometry; focal length EE 15 / head 24; aperture 20.955',
                     'target x=.50 is the planned standoff, not a measured successful stop',
                     'target y=.179 and z=-.280 approximate the latest visual surface estimate',
                     'body is fixed and level, measured joint tracking and collisions are not simulated',
                     'point projection does not certify full-object visibility or detector success'],
        source_sha256={s: hashlib.sha256((repo/s).read_bytes()).hexdigest() for s in sources},
        cases=cases)
    output = Path(__file__).with_name('geometry_probe.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), cases=[dict(
        initial_wrist_rad=c['initial_q_rad'][4],
        predicted_distance_m=c['predicted_gripper_to_visual_surface_m'],
        minimum_command_s=c['minimum_command_duration_s'],
        final_ee=c['samples'][-1]['ee'], final_head=c['samples'][-1]['head']) for c in cases]), indent=2))


if __name__ == '__main__':
    main()
