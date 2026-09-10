# Task A：工程对照与论文方法参考

[仓库首页](../README.md) · [Task A 运行入口](../task_a/README.md) · [完整视频](VIDEO_COMPARISON.md) · [展示数据 JSON](data/task_a_comparison.json)

**当前证据支持：D1+G2 在原始 Task A 上已有两种速度配置的完整通过；新增基础策略单次对照在 31.30 m 处停滞。** 下面把本机工程对照、另一个仓库的实测和论文方法分开阅读。只有任务、机器人、输入和评测条件匹配时，性能数字才适合直接比较。

## 工程演进：失败记录也保留

七行均为 seed 42、原始起点的完整赛道尝试，但期间修改过导航、相机处理或恢复逻辑，不能把整张表当作单因素消融。距离取原日志的**最大前向进展**，不是实际行驶路径长度。

| 运行 | 最大前向进展 | 退出/完成仿真时间 | 结果 |
| --- | ---: | ---: | --- |
| 1399 + 初版 RGB-D | 120.272 m | 227.32 s | 停滞，未通关 |
| 1399 + 重力辅助 RGB-D，第 3 次 | 90.704 m | 180.40 s | 非法接触，未通关 |
| 1999 + 相同 RGB-D | 121.553 m | 233.72 s | 非法接触，未通关 |
| 1999 + 赛道方向约束 | 269.410 m | 508.26 s | 末段停滞，未通关 |
| 1999 + 有预算恢复，第 1 次 | 37.351 m | 69.30 s | 碎石段非法接触，未通关 |
| 1999 + 有预算恢复，第 2 次 | **286.002 m** | **505.66 s** | 唯一终止项 `reach_goal_x` |
| 1999 + 提速指令 0.70/0.50 | **286.005 m** | **457.32 s** | 唯一终止项 `reach_goal_x` |

这七行是解释工程变化的选定记录，并非全部尝试。其余失败与末段诊断见 [12 段历史录像清单](../media/task_a_historical_video_manifest.json) 和 [历史 Release](https://github.com/LYHrmer/atec-robotics-projects/releases/tag/task-a-experiment-history-20260909)。局部诊断通过不能拼接成全程通关，较早退出也不能算完成任务更快。

## 三组可以怎样比较

**1. 检查点：1399 → 1999。** 对应 `residual_1399_rgbd_gravity_course_03` 和 `residual_1999_rgbd_course_01`。11 个保存的源码/资产快照逐字节一致，初态、seed、30°相机、RGB-D 导航和 0.60/0.45 m/s 指令相同，只更换残差检查点。最大进展从 90.704 m 增至 121.553 m，增加 30.849 m；两次仍均非法接触失败，不能由此宣称通关率提高。

**2. 速度指令：0.60/0.45 → 0.70/0.50 m/s。** 同一个 1999 检查点，两次均完整通过，仿真用时减少 **48.34 s（9.56%）**。这是这两次运行的任务用时变化；视频倍速、电脑计算用时和稳定成功率分别看。[完整速度对照](../task_a/docs/SPEED_COMPARISON.md)

**3. 基础策略 vs 基础策略 + 残差。** 新增的基础策略使用同一个原始 ONNX、数值对齐的 PyTorch 后端，并保持已记录的导航、方向约束、恢复预算、相机、seed 42 和 0.60/0.45 m/s 指令；实际不加载残差模型。

| 同配置对照 | 最大前向进展 | 仿真时间 | 结果 |
| --- | ---: | ---: | --- |
| 基础策略，不加残差 | 31.296 m | 86.96 s 退出 | 停滞，未通关 |
| 加 1999 残差，第 1 次 | 37.351 m | 69.30 s 退出 | 非法接触，未通关 |
| 加 1999 残差，第 2 次 | 286.002 m | 505.66 s 完成 | 通关 |

新基础策略的 [result.json](../results/task_a_comparison/base_only_seed42_01/result.json)、[trace](../results/task_a_comparison/base_only_seed42_01/trace.jsonl)、[元数据摘要](../results/task_a_comparison/base_only_seed42_01/metadata.json) 和 [SHA256](../results/task_a_comparison/base_only_seed42_01/SHA256SUMS) 已保存。[新增基础策略完整失败录像](https://github.com/LYHrmer/atec-robotics-projects/releases/download/task-a-method-comparison-20260910/d1g2_taska_base_only_seed42.mp4)。退出原因是停滞，不是摔倒；86.96 s 也不能拿来与完整通过时间排名。

新旧共同的 11 个 Python 快照相同；合成机器人 USD 的差异仅为两条资产绝对路径，搬迁前后 D1/G2 主 USD 哈希相同。基础策略与残差使用不同适配分支，这不是纯粹交换一个权重文件。只有一次基础策略测试，且残差配置也有失败，当前不能宣称统计显著或稳定通关。

## 参考仓库：只在它自己的协议内比较

下表来自用户的 `wheel-legged-control-lab`，固定提交为 `05ce03d2ab68c1e4c503832f8e0602e9a4d0cabe`，并与其独立算术审计逐项核对。[原实验报告](https://github.com/LYHrmer/wheel-legged-control-lab/blob/05ce03d2ab68c1e4c503832f8e0602e9a4d0cabe/results/d1_v3_locomotion_report/README.md) · [审计 JSON](https://github.com/LYHrmer/wheel-legged-control-lab/blob/05ce03d2ab68c1e4c503832f8e0602e9a4d0cabe/results/d1_v3_locomotion_report/arithmetic_audit.json)

| MuJoCo D1 方法 | 速度 RMSE，m/s | 高度 RMSE，mm | 机械活动量，W | 质量达标 |
| --- | ---: | ---: | ---: | --- |
| LQR 零残差 | 0.07969 | 15.30 | 47.44 | 1/2 道路 |
| LQR + PPO | 0.08482 | 15.99 | 44.63 | 2/6 训练种子×道路 |
| 轮腿零残差 | 0.04016 | 9.84 | 16.03 | 2/2 道路 |
| 轮腿 + PPO | 0.03328 | 11.76 | 20.50 | 6/6 训练种子×道路 |

它使用**不含 G2 的 D1、MuJoCo、82 维 oracle 真值观测、60 s 命令跟踪任务**。每种学习结构有三个训练种子，每模型 32768 样本，结果对两条留出道路等权平均。质量达标同时检查完整时长与速度、偏航、高度、姿态误差，不等同于 ATEC 通关。

机械活动量是各子步 `sum(abs(力矩 × 关节速度))` 的均值，不是电池功率或 CoT。轮腿 PPO 的速度误差减少，但高度误差和机械活动量增加；跨 LQR/轮腿结构的差异还包含低层控制变化。**这些数据不与 D1+G2 Task A 混排，也不把真值状态控制称为视觉导航对照。**

## 三篇论文提供什么思路

| 方法与出处 | 主要思想 / 输入 | 原平台 | 本项目状态 |
| --- | --- | --- | --- |
| [Rudin 等，PMLR 2022](https://proceedings.mlr.press/v164/rudin22a.html) | GPU 并行 PPO 与课程；本体、命令及训练信息 | ANYmal 四足，仿真训练并上实机 | 思想参考；本项目采用并行残差 PPO，但未复现该论文完整协议 |
| [Johannink 等，2018 预印本 v2](https://arxiv.org/abs/1812.03201v2) | 已有反馈控制与学习残差叠加；机器人/任务状态 | 实体积木装配，涉及接触与不稳定物体 | 思想参考；本项目基础项是冻结行走网络，不是原装配反馈器 |
| [Bjelonic 等，2021 版本](https://arxiv.org/abs/2010.06322v2) | 全身 MPC 联合轮/机身运动与在线步态；状态、命令、模型和滚动约束 | 轮式 ANYmal，室内/室外 | 未移植到 D1+G2 或 Task A，没有本地性能成绩 |

Rudin 原文报告平地训练少于 4 分钟、崎岖地形约 20 分钟；GPU、机器人、训练目标和已有基础策略不同，不能与本项目训练时长排胜负。Bjelonic 原文的 CoT 最多减少 85%、预测误差最多改善 71%，仅相对于其论文自身基线。方法相似或引用论文，不等于已复现论文结果。

## 数据来源

[展示 JSON](data/task_a_comparison.json) 保存原始字段值、运行编号、原 `result.json` 的 SHA256 和来源路径。`score` 对应原 `raw_progress_score`，是本地原始进度累计分；`max_forward_m` 对应 `maximum_forward_progress_m`；`sim_seconds` 对应 `simulation_seconds`。完整通过记录的浮点进度分约为 26，不能与 Task E 的 18 分相混。

前六行原始记录位于本机 `/home/lybm/ATEC2026_Simulation_Challenge/logs/d1g2_taska_20260909/<运行编号>/result.json`。前三行尚未单独公开原始 JSON，本页和数据文件明确保留该限制，并提供对应历史原片；没有构造不存在的公开结果链接。其余已有 [末段停滞](../task_a/evidence/axis_course_01_stalled/result.json)、[恢复失败](../task_a/evidence/full_course_01_failed/result.json)、[原通过](../task_a/evidence/full_course_02/result.json)、[提速通过](../task_a/evidence/speed_070_050_02/result.json) 的原样证据。

本次只从原记录提取数字、核对源码快照及资产哈希，没有重跑历史物理。世界真值用于终止、原奖励、本地看门狗和离线核验，不作为导航或 actor 输入。完整成功、局部诊断、工程失败和外部论文结果应保持各自口径；后续多次冻结配置评测才能进一步回答稳定性。

## 展示数据接口

`schema_version=1`。`task_a_runs` 是八行数组：七行 `engineering_history` 与一行 `base_only_ablation`；`status` 为 `passed/failed/pending`，`completed` 为布尔值或 `null`。未完成实验的结果字段必须为 `null`，不能用 0 替代。

`comparisons` 用 `baseline_run_id` / `candidate_run_id` 连接三组对照；只有双方完成时才提供完成时间差。`external_reference.rows` 保存四行 MuJoCo 专属指标，`papers` 保存三篇思想与移植状态，论文未复现的 `local_performance` 为 `null`。JSON 的 `source_url` 指向可公开的证据或本页来源说明，原始本机文件另记于 `original_result_path`。
