# ATEC 工作与复习总览

把行走和操作项目放在同一入口，便于回顾实习中的 ATEC 工作，并围绕运动控制算法岗位复习。

| 项目 | 实际方法 | 已验证结果 | 资料 |
| --- | --- | --- | --- |
| Task A：D1+G2 连续越障 | 基础 D1 策略、残差 PPO、RGB-D 导航和有预算的恢复 | 原始赛道 286.002 m，505.66 s，仅触发 reach_goal_x | [结果与 RL 复习](TASKA_D1G2.md) |
| Task E：三物体抓取入篮 | RGB-D 几何感知、IK、关节轨迹、反馈状态机 | v3 seed 42 / seed 0 均为 18/18，分别 53.78 s / 62.36 s | [项目导读](LEARNING_GUIDE.md)、[算法说明](ALGORITHM.md)、[面试讲解](INTERVIEW.md) |

Task A 的 [完整复现仓库](https://github.com/LYHrmer/atec-taska-d1g2) 和 [通关视频](https://github.com/LYHrmer/atec-taska-d1g2/releases/tag/v2026.09.09-local-pass) 为私有，需要访问权限。公开的 [结果摘要](taska/acceptance_summary.json) 来自同一实际运行。
Task E 的代码、详细文档、结果与 [视频 Release](https://github.com/LYHrmer/atec-taske-rgbd-manipulation/releases/tag/fast-smooth-18) 在当前仓库公开。

## 建议的复习顺序

1. 坐标系、相机反投影、外参、末端位姿，然后学习机身角速度、重力与视觉相对位姿。
2. Task E 的 IK、轨迹与状态机，再看 Task A 的导航速度、动作尺度、动作历史与低层策略接口。
3. Task A 已实际训练残差 PPO：理解非对称 actor–critic、奖励、课程地形及训练与评测的分布差异。
4. Task E 当前满分方案没有训练 policy.pt，也不是已完成的模仿学习实验。后续可用其轨迹学习 BC/ACT 数据采集、观测动作对齐、训练验证划分和闭环评测；不要把未来方向写成已有成果。
5. 对齐接触、姿态、视觉接受/拒绝和动作时间戳，区分“身体先失稳”与“感知先失效”。

项目结果来自 Isaac Sim / Isaac Lab 本机仿真，没有声称实机部署或官方在线验收。Task A 有同配置失败记录，Task E 的两个满分种子也不能代表任意场景成功率。
开发使用 Codex/Opus 辅助分析、实现与检查，最终结论由实际训练、运行和验收数据支持。讲项目时应分清上游基础、集成训练工作、AI 辅助与尚未完成的内容。
