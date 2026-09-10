# Task B：移动抓取实验（尚未通关）

[返回项目首页](../README.md) · [实际实验记录](../docs/TASK_B_EXPERIMENTS.md) · [分工与验收](../docs/COLLABORATION.md)

这是原始 Task B 官方环境 `ATEC-TaskB-B2wPiper` 的**引导实验**：在完全不改动官方物理、资源、动作、奖励与终止条件的前提下，先用四种开环策略检查站立、轮驱、转向与下蹲，再尝试公开 RGB-D 驱动的目标接近。Claude Opus 实现的稳定控制器可独立开启，所有尝试均保留实际奖励与终止记录。

**它不是抓取方案，也不是通关记录。** 站住 30 秒、开走几米、或者蹲下去，都不能算“抓取成功”，更不能算 Task B 通过。唯一与投递计分相关的官方奖励项是 `objects_in_circle`；`grasped_objects` 只表示夹爪本体进入了官方距离阈值，属于“接近”，不是“抓住”。任何结论都只能来自 `result.json` 里实际记录的数字。

## 基础模式

前 100 次调用全零动作（settle），随后 100 次调用线性增到目标（ramp）。全零是官方的静止指令：腿和臂是 `use_default_offset=True` 的位置项，全零即请求默认关节角；轮子是速度项，默认偏置为零。

| `--mode` | 动作 | 想测的问题 |
| --- | --- | --- |
| `hold` | 全零 | 默认姿态能不能自己站住，站多久 |
| `forward` | 四个轮子同时 +0.10（官方 scale=5.0，即 0.5 rad/s） | 会不会动、往哪动。**正号=前进只是待测假设**，实际方向由记录的 base 位移判定 |
| `turn` | 右轮 +0.10、左轮 −0.10（FR/RR 对 FL/RL） | 差速能否转向。**转向正负同样是待测假设** |
| `crouch` | 轮子为零，腿关节从**实际默认值**平滑插值到 hip 0 / thigh 1.0 / calf −2.0，并被真实软限位裁剪；臂与夹爪保持官方默认 | 机身能压多低，会不会撞出官方 `illegal_contact` |

`crouch` 是实验性的：CPU 有限数值搜索找到夹爪可达机身下方约 0.294 m 的姿态（不是已证明的全局下界，也未验证碰撞），直立时可能碰不到地面，所以需要往下压。是否可行未知，一旦触发官方非法接触，评测立即停止。

动作下标不是猜的：程序从真实 action manager 的 term 顺序、term 内关节顺序、mode、scale、`use_default_offset` 里读出来，并校验 B2w 共 24 维（腿 12 + 轮 4 + 臂 8）、四个轮关节为 `FR/FL/RR/RL_foot_joint`。连续轮的位置软限位可为非有限值，因其使用速度控制，不参加位置限位检查；JSON 中记为 `null`。腿和臂的位置限位仍必须有限、有序。**schema 与预期不符时直接报错，不做任何猜测。**

基础策略只接收公开 `proprio` 与静态关节/动作 schema。`visual_approach` 另外接收公开 RGB-D，使用移动 Piper FK 将黄色高物体候选投影到机身坐标；这只是颜色/形状启发式，不是已验证的芥末瓶语义识别。机器人根位姿、物体位姿、接触力都只写进产物文件做诊断，不进策略。

## 视觉与稳定控制

- `--mode visual_approach`：腕部 RGB-D 检测、两帧确认、差速接近；目标丢失立即停车。默认腕部视角存在近场盲区。`--vision_head` 可启用原头部 RGB-D 接续，需查具体运行记录是否已经验证。
- `--stabilize`：加入 [Claude 稳定模块](STABILITY_DESIGN.md)，默认使用未校准参考的 baseline。限速带来的存活改善，不等于有效转向。
- `--stability_profile neutral`：在开启稳定模块时，选择参考站姿校准候选；实际结果见实验记录。
- `--stability_legs off`：关闭腿部补偿，用于区分轮速限制与腿修正的作用。

观测关节顺序从 **ObservationManager 已解析的配置副本**读取；每一步另外核对观测相对角加默认值与记录关节角的一致性。原 `task.cfg` 中未解析的 `slice(None)` 不能代表观察顺序。这个一致性核对只用于诊断，不向策略提供真实位姿。

## 跑一次

需要已装好的 Isaac Lab 环境和官方任务仓库（只读引用，不修改、不复制其中任何文件）。本机验证环境：Python 3.10、Isaac Sim 4.5、Isaac Lab 2.3.2、单张 8 GB GPU。

```bash
export ATEC_TASK_ROOT=/home/lybm/ATEC2026_Simulation_Challenge
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$ATEC_TASK_ROOT:$ATEC_TASK_ROOT/source/atec_rl_lab"

export ATEC_PYTHON=/home/lybm/miniforge3/envs/isaaclab/bin/python
bash run.sh task-b --mode hold --seed 42 --max_steps 500 --output runs/task_b_hold_seed42
```

只有 `--output` 是必填，且**必须是新目录**（已存在就直接报错，不会覆盖旧结果）。其余：`--mode`（默认 `hold`）、`--seed`（默认 42）、`--max_steps`（默认 1500 步 = 30 仿真秒）、`--settle_calls`/`--ramp_calls`（默认 100/100）、`--wheel_cmd`（默认 0.10）、`--rgb_interval`（默认 100 步存一次公开 RGB）、`--crouch_fraction`（默认 1，0–1 范围内缩放下蹲幅度）、`--video`（双相机拼接为 1280×480、10 fps 原速录像）。其余是 AppLauncher 官方参数，如 `--device`。

评测**只在离屏运行**：脚本强制 `headless`，并强制 `enable_cameras`（官方 image 观测组需要相机）。Isaac Sim 4.5 上会自动选用随包的 headless 相机 experience，并复用 Task A 的只读递归相机位姿兼容补丁。它不会在移动相机上写入覆盖父节点变换的 Fabric 属性，日志前缀为 `[D1G2]`。

配置只动三处：`scene.num_envs=1`、`sim.device`、`sim.use_fabric`，然后照官方流程调用一次 `apply_safe_action_spec`（不带任何 participant spec）。物体布局由 `--seed` 经官方 `TaskBEnvB2WCfg(seed=...)` 决定，`env.reset(seed=...)` 用同一个种子。

## 产物与读法

| 文件 | 内容 |
| --- | --- |
| `result.json` | 模式、种子、步数、停止原因、终止时激活的终止项、分数定义与数值、逐项奖励合计、机身位移与最低高度、终止后状态、策略描述、诚实解读 |
| `trace.jsonl` | 每一步一行：奖励、逐项奖励、步前机身位姿、非法接触最大力、终止标志 |
| `telemetry.npz` | 每一步的 `action`/`q`/`qdot`/`base_xyz`/`base_quat`/`alpha`/`illegal_force`，以及关节名、奖励项名、终止项名、非法接触体名 |
| `environment_metadata.json` | 真实动作 schema、关节限位、奖励与终止项参数、非法接触体与阈值、观测维度；诊断真值（env origin、初始机身位姿、18 个物体初始位置）单独放在 `diagnostics_not_visible_to_policy` 下 |
| `source_manifest.json` | 本进程实际加载的引导代码与 `atec_rl_lab` 源码的 SHA-256 |
| `head_rgb_*.png` / `ee_rgb_*.png` | 稀疏公开 RGB 帧（step 0、每 `--rgb_interval` 步、最后一帧）；不保存原始相机大数组 |

几个必须注意的语义：

- **分数定义。** 官方奖励按 dt 缩放，`score_raw_total` = Σ(env reward / step_dt)，与官方 `scripts/play_atec_task.py` 的累加方式一致。逐项数值取自 `RewardManager.get_active_iterable_terms`（已去掉 dt）；`reward_term_sum_vs_env_reward_max_abs_error` 是两者的一致性自检。
- **首个终止即停止。** 包含 settle 阶段：没有“忽略预热”，也不会跨过终止继续跑。`terminated_during_settle` 会标出这种情况。官方终止项为 `time_out` / `illegal_contact` / `fall`（`minimum_height=0.0`，即世界系 base z < 0；旧报告里 0.24 的说法是错的）/ `objects_in_circle_done`。
- **逐步状态是“步前”状态。** 每行记录的 `q`/`base_xyz` 是算这一步动作时的状态。终止那一步之后官方环境已经把该 env 的关节复位（Task B 没有根位姿复位事件，所以机身位姿不变），`terminal_pre_reset` 通过实例级只读观察器，在官方复位之前复制状态、各接触体的历史最大接触力和终止项；复位后的状态另记在 `terminal_post_step_state_after_official_reset` 里。
- **线速度不作声明。** 轮半径未经本仓库核实，只报告轮关节角速度（rad/s）与实测机身位移，不换算 m/s。
- **哈希覆盖范围。** `source_manifest.json` 里的“未改动物理”是**配置层面的断言**（只设 `num_envs`/`device`/`use_fabric`），不是测量结论；哈希只覆盖本进程加载的 `.py`，**不覆盖** USD/USDA 资源与贴图、Isaac Lab/Isaac Sim 与 Kit、以及运行期打的相机 Fabric 补丁（该补丁是有意安装的），也不能证明本地原始仓库与上游一致。

## 已知前提

B2w 有四个轮子，避开了 Tron2 两轮平衡的学习问题；原 B2 步态策略不可靠。B2w+Piper 的 USD 资源在本机可解析，Piper 挂载体到臂的平移为 (0.2, 0, 0.1)、旋转为单位矩阵。旧 F16 报告里的 `arm_joint2=-0.3` 超出该关节下限 0，不得再用。

代码采用 MIT 许可，保留 ATEC 版权与来源说明；任务定义与机器人、物体资源由各自项目提供。
