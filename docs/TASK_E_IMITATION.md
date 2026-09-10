# Task E 模仿学习：让网络学习机械臂的动作修正

[仓库首页](../README.md) · [Task E 已有满分方案](TASK_E.md) · [数据说明](../datasets/task_e_il/README.md)

**已经完成真实演示采集、CPU 训练和三组闭环比较；学生在独立 seed 1 得到 18/18 分、52.84 s。** 同一 seed 的原规则教师和 DLS-only 对照也均为 18/18，分别用时 62.96 s 和 63.16 s。本实验训练一个小网络，学习机械臂运动时的六轴动作修正；视觉定位、路径规划和抓放流程继续使用已有规则，因此这里只研究局部伺服的模仿学习。目前每组只有一次闭环结果。

可以直接查看 [模型权重](../weights/task_e_il/cartesian_bc_v1.pt)、[训练报告](../results/task_e_il/offline_v1/training_report.json)、[学生最终成绩](../results/task_e_il/closed_loop/cartesian_bc_seed1/result.json)、[完整演示数据](../datasets/task_e_il/) 和 [独立录像下载](https://github.com/LYHrmer/atec-robotics-projects/releases/tag/task-e-imitation-20260910)。

## 网络到底学了什么

把“机械臂完成一次抓取”拆开看：先从 RGB-D 图像找到物体，再规划一串经过的姿态，最后让六个关节逐步跟上目标。本实验替换最后一步中的一部分。

规则规划器仍产生关节路点。**FK（正向运动学）**把路点换算成“夹爪应该到哪里、朝向哪里”；**DLS（阻尼最小二乘）**根据当前姿态先算出一次解析关节动作。学生看到当前状态、末端姿态误差和 DLS 动作，补上一个六维**残差**，也就是对这次动作的修正量。

```mermaid
flowchart LR
    A[RGB-D 与当前关节状态] --> B[规则感知与 IK 路径]
    B --> C[FK 换算末端目标]
    C --> D[DLS 解析动作]
    A --> E[45 维特征]
    C --> E
    D --> E
    E --> F[小网络预测六轴残差]
    D --> G[相加并限制动作幅度]
    F --> G
    G --> H[执行关节动作]
```

训练标签和执行关系可以写成：

```text
监督标签 = 教师当步关节增量 − 同一状态下的 DLS 关节增量
学生动作 = 当前关节角 + 限幅（DLS 关节增量 + 网络残差）
```

这里的教师是本仓库 [solution.py](../solution.py) 中已有的规则策略。采集器在动作执行前记录状态和教师实际给出的动作；训练时使用教师动作做标签。部署时，**网络不输入教师关节路点、教师当步动作或仿真物体真值**。不过，规则路径和关节路点到达判定仍保留，不能把整个控制系统称为摆脱规则的独立学生。

| 部分 | 本实验的处理 |
| --- | --- |
| RGB-D 感知、IK 路径、FK 末端目标 | 规则计算，没有训练视觉网络 |
| 路点切换、抓放状态机、夹爪、接触停止 | 沿用规则逻辑 |
| 初始化、持位和物理关节/动作限幅 | 沿用规则逻辑 |
| DLS 解析控制 | 保留，承担基础运动计算 |
| DLS 上的六轴关节增量残差 | **由演示训练得到** |

网络结构为 `45 → 128 → 128 → 6` 的 MLP（多层全连接网络），使用 Tanh 激活。45 维输入包括关节位置和速度、末端位置/朝向误差、DLS 动作、限幅、关节范围内的位置、运动阶段和物体编号。各关节残差的幅度由训练标签的 99.5% 分位数确定，再受物理边界约束；推理时没有教师动作回退。

早期的 [joint-servo BC 管线](../tools/task_e/il/README.md) 也保留了代码，但其标签可直接从输入用 `clip` 饱和公式得到，主要用于检查采集、训练和部署是否连通。**本页的主实验是笛卡尔目标残差模型**，不把前者当作主要学习成果。BC（行为克隆）指用演示中的状态—动作对进行监督学习。

## 数据与已经得到的结果

两条新采集演示均来自同一个满分规则教师。只记录运动阶段中对齐的伺服样本；初始化、持位等步骤不会都变成训练帧。

| 用途 | 环境 seed | 伺服样本数 | 教师整回合结果 |
| --- | ---: | ---: | --- |
| 训练 | 42 | 2310 | [18/18，2689 步，53.78 s](../results/task_e_il/offline_v1/teacher_seed42/result.json) |
| 验证及选择最佳轮次 | 0 | 2769 | [18/18，3118 步，62.36 s](../results/task_e_il/offline_v1/teacher_seed0/result.json) |
| 独立闭环比较 | 1 | 不参与本次训练或选模 | 规则教师、学生和 DLS 对照结果见下表 |

同一回合没有被随机拆到训练和验证两组。seed 0 已用于选择模型，不能再称为未见测试集。数据没有原始 RGB-D 图像，不能直接用于端到端视觉模仿学习；字段、采集完整性和文件哈希见 [数据说明](../datasets/task_e_il/README.md)。

在本机 CPU 上训练 200 轮，程序计时 **5.88 s**，按验证损失选中第 **197** 轮。计时包含特征派生、训练和后续统计处理，不包括原始文件读入、仿真演示采集或闭环测试。

| seed 0 离线指标 | DLS，不加学习残差 | DLS + 学习残差 |
| --- | ---: | ---: |
| 对教师动作的平均绝对误差，rad | 0.0020088 | **0.0008477** |
| PLACE 放置阶段的平均绝对误差，rad | **0.0004565** | 0.0006097 |

整体误差降低 **57.8%**，含义是“在保存的教师状态上，动作更接近教师”。PLACE 阶段的平均误差反而增加，不能据此宣称任务得分、速度或成功率改善。[独立 CPU 审计](../results/task_e_il/offline_v1/independent_cpu_audit.json) 还检查了教师标签没有进入网络输入，并发现目标误差和速度均置零时仍有非零学习残差；其闭环影响需要实际运行判断。原训练报告中的限幅次数受 float32 舍入影响，部署精度的统计以 [float64 更正报告](../results/task_e_il/offline_v1/float64_guard_count_audit.json) 为准。

## 闭环比较：以最终结果文件为准

闭环是让学生真的控制机械臂，再用它造成的新状态计算下一步动作。单步误差会改变后续观测，连续执行可能走到演示没有覆盖的状态，这也是模仿学习需要实际回合评测的原因。[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html) 专门讨论了这种由策略动作引起的状态分布变化；本实验目前使用 BC，没有实现 DAgger。

| seed 1 策略 | 总分 | 控制步数 / 仿真时间 | 当前结论 |
| --- | --- | --- | --- |
| 原规则教师 `solution.py` | **18/18** | 3148 / 62.96 s | [最终结果](../results/task_e_il/closed_loop/rule_seed1/result.json) |
| 笛卡尔残差学生 `cartesian_bc_v1.pt` | **18/18** | 2642 / 52.84 s | [最终结果](../results/task_e_il/closed_loop/cartesian_bc_seed1/result.json) |
| DLS-only：同一执行链，残差设零 | **18/18** | 3158 / 63.16 s | [最终结果](../results/task_e_il/closed_loop/dls_seed1/result.json) |

<!-- CLOSED_LOOP_RESULTS: update only from finalized evaluator result.json and published evidence. -->

三组物体初态相同，均正常终止且未修改任务物理。**在这一次 seed 1 比较中，学生比原规则教师少用 10.12 s（16.07%），比 DLS-only 少用 10.32 s（16.34%）。** 原规则教师使用原关节伺服公式；DLS-only 使用与学生相同的执行链、将学习残差设零，因而后者更直接检验残差的影响。这个对照支持“残差在本次运行中缩短了任务时间”，仍不足以证明跨 seed 的速度优势或成功率改善。

[独立审计](../results/task_e_il/closed_loop_independent_audit.json) 表明，相比原规则，省时主要来自物体 2 的 PLACE 阶段减少等待，不能解释为机械臂普遍运动更快。**学生得到 18 分时仍处于 PLACE、仍发送闭爪指令；本轮未验证三个物体全部释放后稳定留篮。** 满分来自本地官方环境的计分与终止结果，和额外的释放检查应分开看。

[学生调用统计](../results/task_e_il/closed_loop/cartesian_bc_seed1/metrics.json) 显示：2642 次策略调用中，2263 次是运动伺服调用，这些调用全部使用学习残差，且相对 DLS 真正改变了动作，占所有调用的 **85.65%**。教师动作回退和规则接触停止均为 0 次；物理动作限幅触发 1078 次，关节限位触发 280 次。其余初始化、持位和夹爪等规则环节仍是系统的一部分。

[DLS 对照统计](../results/task_e_il/closed_loop/dls_seed1/metrics.json) 则记录 2807 次 DLS 动作、0 次学习残差动作、0 次教师动作回退，确认对照确实关闭了残差。

## 复现：先离线，再闭环

命令都在仓库根目录执行。先按 [Task E 环境设置](TASK_E.md#本地复现) 配好 `ATEC_PYTHON` 和 `ATEC_TASK_ROOT`。离线训练只需兼容的 Python、NumPy、SciPy、PyTorch，使用 CPU；闭环评测还需要 Isaac Lab 与官方资产。每次选择新的输出目录和 metrics 文件。

**1. 使用已经公开的数据重训。** 不必重新采集两次仿真：

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
"$ATEC_PYTHON" tools/task_e/il/train_cartesian_bc.py \
  --train datasets/task_e_il/teacher_seed42 \
  --validation datasets/task_e_il/teacher_seed0 \
  --output runs/task_e_il/cartesian_reproduction_01 --epochs 200
```

输出包括 `cartesian_bc.pt` 和 `training_report.json`。公开 v1 权重单独保存在 `weights/task_e_il/cartesian_bc_v1.pt`；实际耗时和浮点结果可能随环境变化。需要重新采集时，参照 [采集命令和完整结束要求](../tools/task_e/il/README.md)。

**2. 对公开权重做 CPU 输入边界审计。** 这会检查整条验证演示，不启动仿真：

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
"$ATEC_PYTHON" tools/task_e/il/audit_cartesian_model.py \
  --dataset datasets/task_e_il/teacher_seed0 \
  --checkpoint weights/task_e_il/cartesian_bc_v1.pt \
  --output runs/task_e_il/cartesian_audit_reproduction_01.json
```

**3. 依次运行同一 seed 的教师、学生和 DLS 对照。** 每组分别保存结果与独立录像：

```bash
bash run.sh task-e --headless --solution solution.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/rule_reproduction_01 \
  --video runs/task_e_il/rule_reproduction_01.mp4

ATEC_IL_CHECKPOINT=weights/task_e_il/cartesian_bc_v1.pt \
ATEC_IL_METRICS=runs/task_e_il/student_reproduction_01_metrics.json \
ATEC_IL_DLS_ONLY=0 \
bash run.sh task-e --headless --solution solution_task_e_il_cartesian.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/student_reproduction_01 \
  --video runs/task_e_il/student_reproduction_01.mp4

ATEC_IL_CHECKPOINT=weights/task_e_il/cartesian_bc_v1.pt \
ATEC_IL_METRICS=runs/task_e_il/dls_reproduction_01_metrics.json \
ATEC_IL_DLS_ONLY=1 \
bash run.sh task-e --headless --solution solution_task_e_il_cartesian.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/dls_reproduction_01 \
  --video runs/task_e_il/dls_reproduction_01.mp4
```

测试自己重训的模型时，将两处 `ATEC_IL_CHECKPOINT` 换成第 1 步输出的 `cartesian_bc.pt`。DLS 对照仍加载同一检查点来核对接口，但实际使用零残差。最终得分读取各输出目录内的 `result.json`；采集器或 metrics 中的 `last_observed_score` 可能早于最后一步奖励，不能替代最终结果。

## 按什么顺序读代码

1. [原项目导读](LEARNING_GUIDE.md)：先理解 RGB-D、运动学和抓放状态机各自负责什么。
2. [数据字段](../datasets/task_e_il/README.md) → [collect.py](../tools/task_e/il/collect.py)：弄清动作执行前的状态怎样与教师标签对齐。
3. [cartesian.py](../tools/task_e/il/cartesian.py)：依次看 `goal_from_joint_target`、`derive_sample`、`cartesian_features`，再看残差相加与限幅。
4. [train_cartesian_bc.py](../tools/task_e/il/train_cartesian_bc.py)：看整集划分、按阶段/物体平衡的损失，以及如何选择最佳轮次。
5. [cartesian_policy.py](../tools/task_e/il/cartesian_policy.py) → [audit_cartesian_model.py](../tools/task_e/il/audit_cartesian_model.py)：看学生如何进入真实动作链、DLS 对照怎样开关，以及独立审计检查了哪些边界。

学习时可以用三个问题检查理解：网络是否看到了教师当步答案？把残差清零以后还剩哪些能力？离线误差降低是否在同一 seed 的实际得分和终止结果中得到体现？
