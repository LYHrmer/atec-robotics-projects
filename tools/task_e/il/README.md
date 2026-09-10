# Task E 行为克隆数据管线与伺服对照

**主实验已完成训练及闭环比较。** 当前结果、公开权重与复现入口见
[模仿学习中文说明](../../../docs/TASK_E_IMITATION.md)：seed 1 学生得到 18/18、52.84 s，
同场景原规则 / DLS-only 分别为 62.96 / 63.16 s。下面保留采集器和早期管线对照的用法。

本实验学习运动阶段的六轴伺服动作。RGB-D 感知、IK 路径、路点判定、抓放状态机和夹爪仍沿用
`solution.py` 的规则方案，因此不是整个 Task E 策略的模仿学习，也不是端到端视觉网络。

首版 `train_bc.py` / `solution_task_e_il_bc.py` 只用作数据管线对照：其教师标签恰好等于
`clip(features[..., 0], -1, 1)`，网络学的是输入中已经存在的饱和函数，不能作为主要学习成果。
后续笛卡尔目标残差实验隐藏教师关节路点，用独立 DLS 基线比较学习增益，见 `CARTESIAN_NOTES.md`。

早期管线网络的输入为每个关节的当前角度、速度、与路径路点的误差、动作限幅、关节编号、运动阶段和物体编号。
路点来自当前 RGB-D 估计与机器人运动学；输入中没有教师当步动作或仿真真值。输出是六轴关节目标增量，
训练标签来自教师在同一执行前观测上的实际动作。部署时移动阶段不调用教师伺服生成动作，也不回退到教师。

历史 telemetry 记录的是执行后的关节状态，且缺少完整路径上下文，不能直接当作严格对齐的本实验监督数据。
以下采集器通过公开 `predicts(obs, score)` 接口记录对应的执行前数据，不改仿真器、任务物理或原规则方案。

从仓库根目录、现有 Isaac Lab 环境运行。每个数据与结果目录必须尚不存在：

```bash
ATEC_IL_DATA_DIR=runs/task_e_il/data_seed42 ATEC_IL_EPISODE_ID=teacher_seed42 \
  bash run.sh task-e --headless --solution solution_task_e_il_collect.py --seed 42 \
  --max_steps 6000 --output runs/task_e_il/teacher_seed42 --snapshot_interval 600

ATEC_IL_DATA_DIR=runs/task_e_il/data_seed0 ATEC_IL_EPISODE_ID=teacher_seed0 \
  bash run.sh task-e --headless --solution solution_task_e_il_collect.py --seed 0 \
  --max_steps 6000 --output runs/task_e_il/teacher_seed0 --snapshot_interval 600
```

先检查两次 evaluator 的 `result.json`，确认演示来源与分数。采集器元数据的最后一次已观察分数可能早于最终奖励；
任务终点以 evaluator 为准。数据每 250 步保存一次，evaluator 在关闭 Isaac 前显式调用 `finalize()` 保存尾部。
Kit 关闭时可能跳过 Python 的 `atexit`，因此训练器要求元数据的 `finalized_before_sim_shutdown=true`；
早期未显式结束的采集可能缺少最后一段，必须重新采集，不能把它当完整演示。

整条 seed 42 演示用于训练、整条 seed 0 演示用于验证；脚本拒绝同 seed 或同 episode 混入两组。

```bash
PYTHONNOUSERSITE=1 "$ATEC_PYTHON" tools/task_e/il/train_bc.py \
  --train runs/task_e_il/data_seed42 --validation runs/task_e_il/data_seed0 \
  --output runs/task_e_il/train_01
```

若需要额外测试早期管线，可以用独立 seed 1 运行以下命令；本轮没有运行这个旧学生的闭环测试，
已发布的成绩来自主笛卡尔残差模型：

```bash
ATEC_IL_CHECKPOINT=runs/task_e_il/train_01/servo_bc.pt \
ATEC_IL_METRICS=runs/task_e_il/bc_seed1_metrics.json \
  bash run.sh task-e --headless --solution solution_task_e_il_bc.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/bc_seed1 --snapshot_interval 600
```

汇报 evaluator 的总分、结束原因与时间，并附 metrics 的 learned 动作占比、规则接触停止数和教师动作回退次数。
未学习的 INIT/持位/夹爪阶段仍有规则动作；分母分别给出所有策略调用和运动阶段调用，不能用后者掩盖前者。
主实验的真实闭环成绩及局限见页首链接，不能由早期管线的离线误差推断通关。
