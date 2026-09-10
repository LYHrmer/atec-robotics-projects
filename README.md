# ATEC 机器人仿真：Task A 越障与 Task E 抓取

这个仓库记录两项机器人任务：让 D1+G2 走完整条复杂赛道，让 Piper 机械臂把三个物体放入篮中。两项任务的代码、文档和验收证据统一管理，便于复现、学习和回顾工程取舍。

## 已经完成什么

| 任务 | 机器人怎样完成任务 | 已验证的本机结果 | 代码与运行 | 视频与证据 |
| --- | --- | --- | --- | --- |
| **Task A：连续越障** | D1+G2 用相机估计行进方向，在已有行走策略上训练动作修正 | 提速版 **457.32 s / 286.005 m**，比原 505.66 s 少 **9.56%**；均只触发终点 | [Task A 入口](task_a/README.md) | [新旧横屏视频](docs/VIDEO_COMPARISON.md) · [速度验收](task_a/evidence/speed_070_050_02/speed_acceptance_audit.json) |
| **Task E：三物体抓取入篮** | Piper 从彩色与深度图像定位物体，用运动学规划动作并检查抓放反馈 | seed 42 / seed 0 均为 **18/18 分**，分别 **53.78 s / 62.36 s** | [Task E 入口](docs/TASK_E.md) | [演示视频](media/task_e_fast_seed42_color.mp4) · [seed 42](results/fast_seed42_package.json) · [seed 0](results/fast_seed0_package.json) |
| **Task E：模仿学习实验** | 从规则教师演示中学习六轴关节动作修正，保留感知、规划与抓放规则 | 独立 seed 1 **18/18 分、52.84 s**；同场景原规则 / 纯 DLS 为 62.96 / 63.16 s | [方法、数据与复现](docs/TASK_E_IMITATION.md) | [三组视频](docs/VIDEO_COMPARISON.md#task-e) · [比较记录](results/task_e_il/closed_loop/comparison_summary.json) |

Task A 两个速度配置各有一次完整通过，原配置也存在失败；Task E 只验证了少量指定初态。这些结果来自 Isaac Sim / Isaac Lab 本机仿真，尚未证明广泛场景成功率、实机效果或官方线上成绩。

Task B 的要求、旧方案与硬件可行性见 [评估说明](docs/TASK_B_FEASIBILITY.md)：可以继续研发，当前尚无正分或通关证据。

Task E 的学习实验每组只运行一次，省时主要来自放置阶段减少等待；本轮满分终止时尚未完成独立的松爪留篮检查。原规则方案和历史视频继续保留。

## 先选一条阅读路线

| 你想做什么 | Task A | Task E |
| --- | --- | --- |
| **先看懂机器人在做什么** | [行走与导航导读](task_a/docs/LEARNING_GUIDE.md) | [从抓取视频开始](docs/LEARNING_GUIDE.md) |
| **跟着代码理解实现** | [代码、模型与运行入口](task_a/README.md) | [一次控制调用的代码导读](docs/CODE_WALKTHROUGH.md) |
| **复现已有结果** | [环境、依赖与启动](task_a/README.md) | [本地复现](docs/TASK_E.md#本地复现) |
| **复习或准备项目介绍** | [训练过程、失败分析与自测](docs/TASKA_D1G2.md) | [算法原理](docs/ALGORITHM.md) · [面试讲解](docs/INTERVIEW.md) · [自测题](docs/SELF_CHECK.md) |

想把两项任务一起学习，按 [跨任务学习路线](docs/ATEC_PROJECTS.md) 连接坐标系、感知、控制和实验验证。

## 本地运行从哪里开始

本机验证环境为 Ubuntu、Python 3.10、Isaac Sim 4.5、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128，硬件为 i9-14900HX、RTX 5060 Laptop 8 GB、32 GB 内存。Task A 完整通过时显存采样峰值为 6210 MiB。

先按各任务运行文档配置环境与外部资产：[Task A](task_a/README.md) · [Task E](docs/TASK_E.md#本地复现)。Task A 的机器人原始资源及基础策略需要按其说明准备；训练后的 [残差模型](task_a/weights/model_1999_final.pt) 已随本仓库保存。

配置完成后，从仓库根目录使用统一入口 `bash run.sh task-a` 或 `bash run.sh task-e`，按对应运行文档追加参数。Task A 的相对输出路径以 `task_a/` 为起点，Task E 以仓库根目录为起点。

还没有配置仿真时，可以从仓库根目录运行 Task E 几何算例，只需 Python、NumPy 和 SciPy：

```bash
PYTHONNOUSERSITE=1 python3 tools/task_e/learn_geometry.py
```

算例检查像素反投影、夹持点换算、正逆运动学和已有运动指标，不生成新的比赛成绩；逐项解释见 [五个动手练习](docs/HANDS_ON.md)。

## 文件放在哪里

```text
task_a/              Task A 代码、训练权重、运行说明与验收证据
run.sh               两项任务的统一启动入口
solution.py          Task E 官方策略入口；同目录策略文件共同运行
scripts/evaluate.sh  Task E 评测入口
tools/task_e/        Task E 评测、几何学习和运动分析工具
datasets/task_e_il/  Task E 完整教师演示，按回合划分训练与验证
weights/task_e_il/   Task E 模仿学习实验权重
docs/               跨任务学习路线、Task E 深入文档、Task A 成果说明
results/            Task E 结果、源码记录和逐步运动数据
media/              Task E 视频、截图与教学图
```

Task A 的运行说明集中在 `task_a/`；Task E 保留根目录的官方调用接口。完整视频和历史版本见 [Releases](https://github.com/LYHrmer/atec-robotics-projects/releases)。

## 几个常用词

| 术语 | 这里指什么 |
| --- | --- |
| RGB-D | 彩色图像加深度图，帮助估计三维位置 |
| 残差策略 | 在已有控制动作上学习一份修正；Task A 用强化学习，Task E 新实验用教师演示训练 |
| IK，逆运动学 | 给定机械臂末端的位置和朝向，反求关节角 |
| seed，随机种子 | 固定一次随机场景生成过程，便于重复比较 |
| 仿真时间 | 模拟世界里经过的时间，与电脑实际计算耗时不同 |

## 来源与许可

项目基于 [ATEC 官方仿真环境](https://github.com/atecup/ATEC2026_Simulation_Challenge)。Task E 代码许可与来源见 [LICENSE](LICENSE)、[ATTRIBUTIONS.md](ATTRIBUTIONS.md)；Task A 的依赖来源与使用范围见 [任务说明](task_a/README.md)。开发使用 Codex / Opus 辅助分析、实现和检查，结论以保存的运行证据为准。
