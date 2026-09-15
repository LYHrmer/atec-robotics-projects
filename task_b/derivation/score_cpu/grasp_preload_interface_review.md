# Task B 夹爪预紧：原接口独立核对

2026-09-14，本地只读检查。没有改控制器、审计器、官方环境、执行器或 Git；没有启动 GPU。新的组合规格已收到：preload 0.025 m、paired lift 0.20 rad，结果应归因整个候选。

## 指令没有被 Python 动作链裁到关节硬限

原 `source/atec_rl_lab/atec_rl_lab/tasks/task_base/action_base.py:18` 默认 arm 是 position、scale=0.5、clip=None；真实 grasp 运行 metadata 也记录此配置。

本机 IsaacLab 的 `envs/mdp/actions/joint_actions.py:169` 先做 `raw*scale+offset`，只在显式 `cfg.clip is not None` 时裁剪；`:197` 将处理结果交给 `set_joint_position_target`。`assets/articulation/articulation.py:1079` 仅写目标缓存；`:1869` 调用执行器后把目标转存 PhysX 缓存；`:263` 下发位置目标。`actuators/actuator_pd.py:118` 的隐式执行器计算近似 effort 供诊断，但直接返回输入 control_action，不修改位置目标。

因此上述原 Python 链没有将 q7=-preload、q8=+preload 裁到闭止挡。实际 q7 的物理范围仍为 [0,0.035] m、q8 为 [-0.035,0] m，由原 PhysX 关节约束决定；预紧目标超出此范围不等于实际越界。不能把“目标超出闭止挡”误判为修改物理限位。

## 原 Task B 与 Task E 并非相同夹爪驱动

| 配置 | 刚度 | 阻尼 | effort limit |
|---|---:|---:|---:|
| B2w 附挂 Piper，含两指 | 80 | 4 | 100 |
| Task E 独立 Piper | 800 | 80 | 100 |

来源为原 `assets/robots/b2w.py:34` 与 `assets/robots/piper.py:44`。Task E `tasks/task_e/env_cfg.py:233` 还关闭了机械臂重力。这里刚度相差 10 倍、阻尼相差 20 倍，不能直接照搬 Task E 成功的预紧量并假定夹持力相同。

B2w USD 中两指的原 authored 数值是 stiffness=100、damping=1、maxForce=10；启动时 `articulation.py:1763`、`:1764`、`:1772` 将隐式执行器配置的刚度、阻尼和 effort_limit_sim 写入模拟器，所以运行配置应按 80/4/100 分析，不能误取 USD 尚未覆盖的值。

## 静态驱动力估算及实际限制

两指是 prismatic force drive。忽略运动、约束耦合及碰撞求解误差，原近似模型为 `F7 = 80*(-p-q7)-4*qdot7`、`F8 = 80*(p-q8)-4*qdot8`。以下仅取速度为零、约 49 mm 对称夹持宽度，即 |q|=24.5 mm：

| preload p | 零宽闭止挡每指驱动力幅值 | 49 mm 宽度每指驱动力幅值 | 理想双指摩擦承重 0.5 kg 所需 μ 下限 |
|---|---:|---:|---:|
| 0.015 m | 1.20 N | 3.16 N | 0.776 |
| 0.025 m | 2.00 N | 3.96 N | 0.619 |
| 0.040 m | 3.20 N | 5.16 N | 0.475 |
| 0.050 m | 4.00 N | 5.96 N | 0.411 |

原 Mustard 质量由 `assets/objects/task_b/object.py:55` 设置为 0.5 kg。μ 下限按 `mg/(|F7|+|F8|)` 计算，假设对称横向接触、驱动力全部成为有效法向力、没有加速度或力矩负担。它既不是实际摩擦系数，也不是已测接触力或抓握成功条件。原 Task B 关闭了 base 的材料随机化事件，不能沿用 base 默认事件里的 0.8/0.6 当手指—瓶体接触参数。

0.015→0.025 只增加每指约 0.8 N；49 mm 宽度时增幅约 25.3%。它能作为一次有界候选，但无法保证消除滑脱。100 N effort limit 在上述静态估算中远未饱和；这不是建议直接追求该力上限。

## 范围与下一候选

`task_b/arm_kinematics.py:139` 的 `gripper_targets` 明确限定 opening∈[0,0.07] m、preload∈[0,0.025] m。这个 0.025 上限来自本地控制 API，不是所检查原官方动作接口的幅值限制。更大的有限位置目标在上述原接口中没有被禁止或自动裁剪，但会超出现有本地 helper 的约定，需要明确的新候选边界与相应审计；不能悄悄放宽 helper，也不能称其天然违反官方任务规则。

当前建议采用收到的 0.025 m 有界规格。0.025 从全开每指 0.035 到 -0.025，按原 0.05 m/s 需 1.2 s，仍可在最早 1.4 s 的 CLOSE 门槛前下发完整目标。若未来 preload>0.035 m，闭合 slew 会超过 1.4 s，必须先确认 `close_target_commanded` 再准许 LIFT，否则 LIFT 直接保持最终 finger_target 会造成跳变。当前新规格已明确此条件。

如 0.025 仍滑脱，再依据实际宽度、同一物体位姿和原材料/接触诊断决定是否设一个新的 0.04 或 0.05 上界；这些值这里只是力估算对照，不是本轮授权实现。新实验同时改 paired delta=0.20，不应把变化单独归因于 preload。保持同一原环境、固定硬限、slew、公共姿态预算、非空宽度与真实物体离地审计。
