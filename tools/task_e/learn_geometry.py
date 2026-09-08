"""Small CPU exercises using the published Task E geometry and measured results.

The camera depth, contact point and IK target below are constructed examples,
not sensor recordings or additional task-success measurements.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from task_e_geometry import (
    DEFAULT_JOINT_POS, TABLE_TOP_Z, action_from_joint_targets, fk,
    joints_from_proprio, project_world, solve_ik, top_grasp_rotation,
    unproject_depth, video_camera_parameters,
)


def vector(value):
    return np.array2string(np.asarray(value), precision=6, suppress_small=True)


def camera_example():
    print('\n[1] 相机：人工设置一个深度像素，检查往返投影')
    k, world_from_camera = video_camera_parameters(640, 480)
    depth = np.zeros((480, 640))
    depth[240, 400] = 1.0
    points = unproject_depth(depth, stride=1)
    pixels, optical_depth = project_world(points)
    expected_camera = np.array([(400-k[0, 2])/k[0, 0], 0., 1.])
    expected_world = world_from_camera[:3, :3] @ expected_camera + world_from_camera[:3, 3]
    np.testing.assert_allclose(points[0], expected_world, atol=1e-12)
    np.testing.assert_allclose(pixels[0], [400, 240], atol=1e-9)
    print(f'焦距 fx={k[0, 0]:.6f} 像素；主点={vector(k[:2, 2])}')
    print(f'相机坐标 / m：{vector(expected_camera)}')
    print(f'世界坐标 / m：{vector(points[0])}')
    print(f'重新投影 / 像素：{vector(pixels[0])}；光轴深度={optical_depth[0]:.6f} m')


def contact_example():
    print('\n[2] 夹持点：目标点不动，比较竖直与倾斜夹爪')
    contact = np.array([1.05, .18, TABLE_TOP_Z+.060])
    vertical = top_grasp_rotation([0., -1.])
    tilted = Rotation.from_euler('y', 20., degrees=True).as_matrix() @ vertical
    print(f'人工夹持点 / m：{vector(contact)}')
    for name, rotation in [('竖直', vertical), ('绕世界 Y 轴倾斜 20 度', tilted)]:
        origin = contact-.115*rotation[:, 2]
        np.testing.assert_allclose(origin+.115*rotation[:, 2], contact, atol=1e-12)
        print(f'{name}：接近方向={vector(rotation[:, 2])}；末端原点={vector(origin)}')
    print('这里只验证接触偏移；还没有证明这个人工目标可达或无碰撞。')


def ik_example():
    print('\n[3] IK：由一组合法关节角构造位姿，再反求关节解')
    reference = DEFAULT_JOINT_POS[:6]+np.array([.10, .05, .03, .02, -.08, .06])
    target = fk(reference)
    result = solve_ik(target[:3, 3], target[:3, :3], seed=DEFAULT_JOINT_POS[:6], multi_start=False)
    assert result.success, 'This known reachable example should satisfy the pose tolerances.'
    print(f'构造目标的关节角 / rad：{vector(reference)}')
    print(f'反求的关节角 / rad：{vector(result.joints)}')
    print(f'位置误差={result.position_error:.3e} m；朝向误差={result.orientation_error:.3e} rad')
    print('检查的是末端位姿误差；一般情况下，不要求反求角度与构造角度完全相同。')


def action_example():
    print('\n[4] 动作：第二臂关节从默认 1.2 rad 指向 1.3 rad')
    target = DEFAULT_JOINT_POS.copy()
    target[1] += .1
    action = action_from_joint_targets(target)
    np.testing.assert_allclose(DEFAULT_JOINT_POS+.5*action, target, atol=1e-12)
    print(f'关节目标：{vector(target)}')
    print(f'八维 action：{vector(action)}')
    relative = np.zeros(24)
    relative[1] = .04
    print(f'若相对位置观测第 2 项为 0.04，实际第 2 关节为 {joints_from_proprio(relative)[1]:.2f} rad。')
    print('action 的 0.2 是位置换算量，不是关节速度。')


def metrics_example():
    print('\n[5] 实测对比：从公开结果重新计算百分比')
    report = json.loads((ROOT/'results/motion_comparison/comparison.json').read_text())
    for key, name, unit in [
        ('duration_s', '仿真完成时间', 's'),
        ('wrist_joint6_path_rad', '腕关节总行程', 'rad'),
        ('arm_acceleration_rms_rad_s2', '全程差分加速度 RMS', 'rad/s²'),
    ]:
        baseline = report['baseline']['overall'][key]
        candidate = report['candidate']['overall'][key]
        change = (candidate/baseline-1)*100
        print(f'{name}：{baseline:.5f} → {candidate:.5f} {unit}；变化 {change:+.2f}%')
    print('数据来自同 seed 42 的两个 18 分运行；不是新的仿真测试。')


def main():
    examples = dict(camera=camera_example, contact=contact_example, ik=ik_example,
                    action=action_example, metrics=metrics_example)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', nargs='?', choices=['all', *examples], default='all')
    args = parser.parse_args()
    for name, function in examples.items():
        if args.example in ('all', name):
            function()


if __name__ == '__main__':
    main()
