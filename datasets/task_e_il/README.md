# Task E 模仿学习演示数据

本目录保存两条完整结束、教师得分均为 18/18 的演示：

| 数据目录 | 采集 seed | 运动帧数 | 原回合控制步数 | 用途 |
| --- | --- | --- | --- | --- |
| `teacher_seed42/` | 42 | 2310 | 2689 | 训练 |
| `teacher_seed0/` | 0 | 2769 | 3118 | 独立整集验证 |

同一回合的相邻帧没有被随机拆到训练与验证两组。验证 seed 0 用于选择训练轮次，不能再称为未见测试集。
教师满分只说明演示来源；本目录不声称学生模型闭环通过。

每个 `chunk_*.npz` 保存教师调用中的**执行前**关节状态、速度、路径路点、阶段、物体编号、动作增量及控制步号。
采集器使用公开 `predicts(obs, score)` 接口，不读取仿真器对象真值。采集尾部在关闭 Isaac 前显式 `finalize()`，
`metadata.json` 中的 `finalized_before_sim_shutdown` 均为 `true`。

这是从 RGB-D 规则教师得到的低维伺服演示，**不包含原始相机画面，不是端到端视觉数据集**。
感知、IK 规划、抓放状态机和夹爪均仍为规则；主模型仅学习解析 DLS 控制器上的六轴动作残差。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `q`, `qdot` | 执行前绝对关节位置与六轴速度 |
| `target` | 教师路径中的六轴关节路点，仅供离线生成 FK 笛卡尔目标、监督标签和审计 |
| `delta` | 同一执行前观测下，教师实际给出的六轴关节目标增量 |
| `limit` | 当前阶段的动作增量限幅 |
| `phase`, `object_id`, `step`, `score` | 阶段、物体编号、策略控制步号和当时已观察到的分数 |
| `features`, `label` | 早期饱和函数 BC 管线对照字段，**不是主笛卡尔模型的输入/标签** |

主训练器由 `q/qdot/target/delta` 重新生成 45 维笛卡尔特征与 DLS 残差标签。
网络没有直接输入 `target` 关节值或教师当步动作；它接收当前关节状态、笛卡尔位姿误差、解析 DLS 步及阶段信息。
数据中保留 `target` 是为了独立复现特征与监督转换，不能将其直接喂给主网络。

从统一仓库根目录，使用现有兼容 Python 环境在 CPU 重训：

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
"$ATEC_PYTHON" tools/task_e/il/train_cartesian_bc.py \
  --train datasets/task_e_il/teacher_seed42 \
  --validation datasets/task_e_il/teacher_seed0 \
  --output runs/task_e_il/cartesian_reproduction_01 --epochs 200
```

输出目录必须尚不存在。原 v1 检查点保存在
[`weights/task_e_il/cartesian_bc_v1.pt`](../../weights/task_e_il/cartesian_bc_v1.pt)，
原训练报告、两条教师结果、源码清单及 CPU 独立审计在
[`results/task_e_il/offline_v1/`](../../results/task_e_il/offline_v1/)。
其中原训练报告的离线外层 guard 次数受 float32 舍入影响，部署精度的更正统计见
[`float64_guard_count_audit.json`](../../results/task_e_il/offline_v1/float64_guard_count_audit.json)。

所有演示块、元数据、权重和原报告均按字节复制，SHA256 见
[`file_manifest.json`](../../results/task_e_il/offline_v1/file_manifest.json)。
原报告中的绝对路径和 `runs/` 路径是历史运行位置；当前公开文件以本目录和清单的相对路径为准。
教师 `source_manifest.json` 保留原快照哈希，原始相机、完整日志、私人机器人资产和视频未复制到本数据集。
