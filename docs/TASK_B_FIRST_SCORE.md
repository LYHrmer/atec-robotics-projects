# Task B 首个有效接近分

[项目首页](../README.md) · [在线播放](https://lyhrmer.github.io/atec-robotics-projects/task-b.html) · [后续执行方案](TASK_B_EXECUTION_PLAN.md) · [历史实验](TASK_B_EXPERIMENTS.md)

**2026 年 9 月 14 日，B2wPiper 在本机原始 Task B 环境取得了 1 个有效接近分。** 机器人用公开 RGB-D 接近一个黄色物体，停稳后伸臂，再缓慢降低机身，使夹爪本体进入官方 0.20 m 距离阈值。此次没有抓起、投递或通关证明。

## 结果与原始证据

| 项目 | 本次实测 |
| --- | --- |
| 任务 / 回合 | `ATEC-TaskB-B2wPiper` / `plan_p2_lower02_seed42_01` |
| 场景 | seed 42，原起点与 18 物体布局，单环境 |
| 首分时刻 | 第 **3278** 个环境步后，**65.56 仿真秒** |
| 原始分项 | `grasped_objects=1`，`objects_in_circle=0` |
| 首分距离 | `gripper_base` 到 `object_10` 根位置 **0.199966689 m** |
| 最终距离 | **0.191444200 m** |
| 记录范围 | **3378 步 / 67.56 仿真秒**，首分后继续记录 100 步，即 2 秒 |
| 停止原因 | `positive_score_observation_window_complete`；没有触发原任务终止 |
| 独立审计 | `verified_positive_score`，`failed_checks=[]`，退出码 0 |

官方的 `grasped_objects` 名称容易引起误解：它检查的是夹爪本体到物体根位置的三维距离，不检查物体是否被夹住或抬起。此次 `score_raw_total=0.9999999776482582` 是环境浮点奖励累计；对应原分项中的 **1 个接近分**。`object_10` 按官方资产配置是芥末瓶，但策略通过图像选择目标，不读取该编号或物体真值。

[原始结果](../results/task_b_positive/plan_p2_lower02_seed42_01/result.json)、[得分事件](../results/task_b_positive/plan_p2_lower02_seed42_01/scoring_events.json) 和 [独立得分审计](../results/task_b_positive/plan_p2_lower02_seed42_01/independent_positive_audit.json) 均已保存。审计使用动作执行后、任何复位前的同期状态重算距离，并对全部 3378 步重放奖励几何，未发现不一致；全程没有非法接触或跌倒终止。它是本机记录审计，尚无官方线上认证。

## 机器人怎样取得这一分

1. 从实际落地关节姿态切换到紧凑站姿，用腿关节反馈保持支撑；前伸腕部相机寻找黄色候选。
2. 用公开 RGB-D 做弧线接近与近场微进。停车后清除运动期间的旧确认，重新取得两次静止定位。
3. 制动期间保持四个轮关节的角度锚点，抑制真实回退；轮速度仍通过原动作接口与原执行器执行。
4. 对已确认目标只求解一次机械臂关节目标，再进行有界关节反馈运动。锁定后的形状跟踪只辅助诊断；近场出框或启发式漏检不等于物体已移动，也不能授权下探。姿态、公开速度、累计机身运动、关节进展与时间预算继续约束运动。
5. 实际臂位置、速度及底盘状态连续 25 次控制观测满足准入条件后，按预先固定的 **0.02 m 名义参考量**缓慢增加腿关节参考，不依据奖励或真值选择下降幅度。这些观测占用 0.50 秒控制周期，首末观测相隔 0.48 秒。

名义参考与真实运动分别验收。[独立下降审计](../results/task_b_lowering_runtime_audit.json) 以第 3191 步授权前状态为基准：

| 时刻 | 参考进度 | 机身实际下降 | 夹爪实际下降 |
| --- | ---: | ---: | ---: |
| 首分事件，step 3278 | 58.67% | 11.551 mm | 11.561 mm |
| 最终状态，step 3378 | 100% | 20.273 mm | 20.310 mm |

授权后机身 / 夹爪最大水平漂移分别约 **5.49 / 5.44 mm**。第 3340 步达到参考终点，评测器随后按首分观察窗口收尾；最终 `LOWER_HOLD` 仅 **0.74 秒**，没有完成控制器计划的 2 秒下降到位保持，也没有触发 `reach_lowering_hold_complete`。

## 复现命令

在仓库根目录运行，需先准备 [运行入口说明](../task_b/README.md) 中的 Isaac Lab 环境及官方资产。以下来自实际启动脚本，保留全部运行参数，仅将原绝对输出目录改为新的复现目录并省略日志重定向；`settle_calls` 与 `ramp_calls` 均采用默认 100。seed 固定不保证 GPU 仿真逐步一致。

```bash
export ATEC_TASK_ROOT=/home/lybm/ATEC2026_Simulation_Challenge
export ATEC_PYTHON=/home/lybm/miniforge3/envs/isaaclab/bin/python
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2

bash run.sh task-b \
  --mode first_reach --seed 42 --max_steps 5000 \
  --wheel_action_gain 8 \
  --reach_forward 0.05 --reach_turn_cap 0.6 --reach_turn_gain 2 \
  --reach_standoff 0.50 --reach_lowering 0.02 \
  --stance_profile compact --stance_hold --brake_wheel_hold \
  --score_hold_steps 100 --video --rgb_interval 250 \
  --output runs/task_b_first_score_seed42_repro_01
```

`--output` 必须尚不存在。`--score_hold_steps 100` 只控制评测器何时结束记录，分数不传入策略或动作生成；该参数不修改官方终止条件。单 GPU 串行运行，每次使用新进程。

独立审计需要完整运行目录；仓库中的小文件摘要不包含全部遥测与源码快照。下载完整证据包并解压后，在仓库根目录运行：

```bash
"$ATEC_PYTHON" task_b/audit_positive.py \
  /path/to/positive_package_20260914 \
  --output /tmp/task_b_first_score_reaudit.json
```

## 视频、来源与分工

[网页视频](https://lyhrmer.github.io/atec-robotics-projects/task-b.html) 保留同一次运行的全部 675 帧，按 1× 播放；原双相机录像为 1280×480、10 fps。媒体时长 67.50 秒与仿真记录 67.56 秒来自不同采样口径，得分时刻以事件记录为准。[转码记录](videos/task_b_first_score_20260914_provenance.json) 保存来源哈希及处理参数。

- [原始录像：task_b_first_score_20260914_original.mp4](https://github.com/LYHrmer/atec-robotics-projects/releases/download/task-b-first-score-20260914/task_b_first_score_20260914_original.mp4)
- [完整证据：task_b_first_score_20260914_evidence.tar.gz](https://github.com/LYHrmer/atec-robotics-projects/releases/download/task-b-first-score-20260914/task_b_first_score_20260914_evidence.tar.gz)
- [发布版本：task-b-first-score-20260914](https://github.com/LYHrmer/atec-robotics-projects/releases/tag/task-b-first-score-20260914)

GPT-6 Astra ultra 负责根据失败证据制定阶段方案与窄接口；实际调用的 Claude Opus 编写静止确认、轮保持、固定目标伸臂、有限停稳和缓降等具体模块；原生代理负责基础运动学、集成修复、独立检查、GPU 实测和证据发布。[具体分工](COLLABORATION.md) 区分设计、实现与验收。

策略输入限于公开本体观测、原相机 RGB-D 和静态结构信息。世界位姿、物体位姿与接触力只作离线诊断；本轮未修改 Task B 原物理、动作、奖励或终止规则。加载源码、动作配置及上游对应关系纳入审计；USD 资产、驱动等不在源码哈希覆盖内，且评测器使用了已记录的只读相机兼容补丁，详见 [审计范围](../results/task_b_positive/plan_p2_lower02_seed42_01/public_package.json)。

这是一条 seed 42 的有效正分记录，不能估计成功率，也没有证明物理夹持、抬升、投递或 18 件通关。后续按 [执行方案](TASK_B_EXECUTION_PLAN.md) 分别验收这些目标；更高成绩通过核验后，GitHub 对应任务展示与视频更新为最佳成绩，较低成绩和失败原片本机归档。本轮保留旧基础实验数据与既有 A/E 发布，不改写 Git 历史。
