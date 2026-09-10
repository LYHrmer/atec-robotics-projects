# Task B 能否继续做？

评估日期：2026-09-10。本文保留当时的只读可行性分析；随后已开始本机 B2wPiper 仿真，运行入口与新记录见 [Task B 实验](../task_b/README.md)。原工程源码保持不变。

**可以继续做，这台电脑足以承担单环境评测、视觉控制和分阶段的小网络训练。当前缺口是可靠的移动操作策略，尚无 Task B 通关证据；已有可核验评分运行仍为 0 分。** 建议先做出一个物体的完整拾取入圈闭环，再决定扩大到 18 个物体。

## 任务比 Task A、Task E 多了什么

官方场景有 18 个地面物体：糖盒、芥末瓶、香蕉各 6 个，随机分布在约 10×10 m 区域。机器人需要移动接近、操作物体，再送入固定收集圈。圈中心为 `(-3,-10)`，半径 1 m；容器有约 0.5 m 高的实体围壁，因此沿地面推到圈边不等于能够入圈。[官方环境配置](https://github.com/atecup/ATEC2026_Simulation_Challenge/blob/4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc/source/atec_rl_lab/atec_rl_lab/tasks/task_b/env_cfg.py)、[容器几何](https://github.com/atecup/ATEC2026_Simulation_Challenge/blob/4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc/source/atec_rl_lab/atec_rl_lab/tasks/task_b/terrain.py)。

评分与通关须分开看：

- 每个物体首次进入末端 0.20 m 范围可计一次接近分；源码中的 `grasped_objects` 不检测真实夹持。
- 每个物体首次进入收集圈、根位置高度在 `[0,0.5]` m 内可再计一次入圈分。两项均为每物体 1 分，原始累计分理论上最多 36。
- 成功终止要求 **18 个物体同时在圈内**；一次接近分、一次入圈分或较长存活都不等于通关。原时限 1200 s，50 Hz 控制、200 Hz 物理步进。

来源：[奖励实现](https://github.com/atecup/ATEC2026_Simulation_Challenge/blob/4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc/source/atec_rl_lab/atec_rl_lab/tasks/task_b/mdp/rewards.py)、[成功条件](https://github.com/atecup/ATEC2026_Simulation_Challenge/blob/4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc/source/atec_rl_lab/atec_rl_lab/tasks/task_b/mdp/terminations.py)、[基础时限与终止配置](https://github.com/atecup/ATEC2026_Simulation_Challenge/blob/4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc/source/atec_rl_lab/atec_rl_lab/tasks/task_base/envs_base_cfg.py)。

现有官方入口为 G1、Tron1Piper、Tron2ALegged、Tron2AWheel、B2Piper、B2wPiper，共 6 种，**没有 D1+G2**。若继续用 D1+G2，可以做保留任务规则的本地适配研究，但不能直接称为官方注册机器人结果。B2/B2w 的机身、髋部和大腿接触力超过 1 N 会触发非法接触终止。

## 这台电脑的能力

本机配置为 i9-14900HX、RTX 5060 Laptop 8 GB、32 GB RAM；此次只读查询可用磁盘约 **202 GiB**。

| 工作 | 判断与依据 |
| --- | --- |
| 单机器人、RGB-D、控制与录像 | 可以尝试。相同电脑已完成 Task A 全程视觉导航：505.66 s 仿真、1007.88 s 墙钟，采样峰值显存 6210 MiB、进程内存 7731 MiB。Task B 多出物体接触与相机负担，仍需单环境测量，不能沿用这个数字当作 Task B 实测。 |
| 本体状态输入的 PPO/残差训练 | 已有硬件实证：Task A 的 512 环境、600 次迭代共 7,372,800 步，用时约 910 s，采样峰值显存 4107 MiB。Task B 应从较少环境启动，再按显存增长扩大。 |
| 直接训练端到端视觉移动抓取 | 8 GB 显存与示范数据量都会限制规模。当前没有可核验的 Task B 专用示范集或训练成功权重；不建议将它作为第一步。 |
| 保存实验数据 | 现有空间支持阶段性实验，但应限制原始相机保存范围。按双相机、640×480、RGB uint8+depth float32、10 Hz、1200 s 估算，一轮未压缩数组约 52 GB，尚未计入额外数据。 |

因此，目前没有证据表明需要先换电脑或租更大显卡。优先解决控制与数据问题，再依据新的 Task B 实测决定是否扩容。

## 已有成果：哪些是真的，哪些只能作为线索

旧 Task B 主要在远程 GPU 上测试，不能当成本机 Task B 的性能基准。

| 已存运行 | 可核验事实 | 能支持的结论 |
| --- | --- | --- |
| `taskb_candidate_E_optionC_long_default_fix_audit` | 同一冻结包重复 5 次，均 0 分、第 149 步非法接触；`fall=0`，无崩溃 | 包可加载，行为未解决任务；“submission readiness pass”只表示打包检查通过。 |
| `taskb_score_improvement_diagnosis/probe_s2_solution_task_b` | 存活至第 1887 步，37.72 s，接近分和入圈分均为 0 | 扫场基线改善存活，但没有完成有效物体交互。 |
| `taskb_policy_backbone_smoke/fresh_process_remote` | 一次前进测试存活 305 步、位移约 0.296 m；测试最终均因非法接触结束 | 旧 B2 平地网络有短时运动能力，尚不是可靠的带臂移动底座。 |
| `taskb_policy_backbone_smoke/f13_contact_attribution` | 第 126 步 `FR_thigh` 接触峰值 15.6545 N，触发非法接触 | 应检查关节映射、动作尺度、带臂负载与步态；仅看机身高度不足以诊断失败。 |

旧 F21–F23 报告把 `base_z≈0.237 m` 解释成“高度终止、轮驱路线物理无解”，**这项归因证据不足**：当前官方 `fall.minimum_height=0.0`，保存的 F23 worker 未设置 0.24 m 阈值，且 warmup 忽略返回的终止标志。新测试需要检查有效初态、reset 后状态和实际终止项。部分该阶段本地目录只保留启动信息与 worker，缺少报告所述完整输出；本评估不把这些报告的推断当作已证实的物理结论。

F0 等使用物体真值的脚本只可复用为明确标记的离线/诊断工具，不能接入最终感知控制策略。F24 的 `REMOTE_CHECKPOINT_FOUND` 列表主要是 B2 平地和 Task E ACT 文件，也不能证明存在 Task B 专用成功模型。

## 可复用的部分与需要重做的部分

| 来源 | 可复用 | 需要重新验证 |
| --- | --- | --- |
| Task A D1+G2 | 训练脚手架、奖励/终止审计、关节命名映射、相机时序与深度坐标处理、RGB-D 里程计、日志证据链 | 1999 权重针对 D1+G2，不能直接驱动 B2/Piper；地图、相机外参、网格方向先验和导航目标均要按 Task B 检查。若仍用 D1，移动中摆臂改变负载，也超出原固定臂训练分布。 |
| Task E Piper | RGB-D 物体定位、FK/IK、路径检查、抓取反馈状态机与释放验证；三类物体有重合 | Task E 使用固定桌边底座和固定外部相机。Task B 是移动底座、低位地面物体，必须重建底座—机械臂—相机变换、关节零位与可达空间。G2 还需要自己的运动学模型。 |
| 旧 Task B | `demo/test/solution_task_b_candidate_E.py`、`demo/test/solution_task_b.py`、45→12 网络加载、84维本体/24维动作适配与接触诊断 | 旧候选不具备正分证据；`demo/policy.pt` 是 B2 平地基础网，不能当作 Task B 成品。原路径已移至 `demo/test/`，旧脚本导入路径可能失效。 |

Task E 现已同时保留原几何方案与 [模仿学习实验](TASK_E_IMITATION.md)。后者只学习六轴 DLS 动作修正，保留视觉、规划和抓放规则。它提供了示范采集与行为克隆的实现参考；固定桌面姿态、权重与旧 ACT 文件不能直接作为 Task B 移动操作策略。

## 建议下一步

1. **先确定评测口径和机器人。** 官方机器人路线先短测 B2Piper/B2wPiper 的有效站立、移动及地面可达性；若选择复用已会走的 D1+G2，单列为自定义机器人本地结果。
2. **建立独立、可复现的短测入口。** 每次用新进程核对初态、关节顺序、尺度、50 Hz 历史和真实终止项；原任务地形、碰撞与计分保持不变。
3. **先验收一个物体的完整闭环。** 合法视觉定位 → 安全接近 → 实际接触/抬起 → 越过围壁 → 松爪入圈。记录两个计分项与释放后物体位置，再扩展多物体、多种子。
4. **训练只补已定位的缺口。** 底座失稳则训练带臂的移动技能；低位抓取失败则先做运动学与接触诊断，再考虑示范/IL或局部残差。保留已工作的层，避免一开始全视觉、全身、18物体联合训练。

这是有依据的继续研发路线，尚不能承诺 Task B 满分或完成日期。初次评估未开启新实验；后续实测及实际失败归因已转入 [Task B 实验记录](../task_b/README.md)。

## 证据定位

原工程根目录：`/home/lybm/ATEC2026_Simulation_Challenge`。以下路径均相对该目录，供本机核验；不要求把原始日志或私人资产上传公开仓库。

| 证据文件 | SHA256 |
| --- | --- |
| `logs/taskb_candidate_E_optionC_long_default_fix_audit/repeated_eval_summary.json` | `609105ffce6fa2ee0e9f443381857c3989ff18d3b2be843946ac145ab4a3a76d` |
| `logs/taskb_score_improvement_diagnosis/probe_results.json` | `d90f3dfe9ca91759f38b56b1eae16ffc96f73f14f41c94ba050033a998c664e1` |
| `logs/taskb_policy_backbone_smoke/fresh_process_remote/movement_summary.json` | `da45bb9112041e5d119b53670153a27f20a54a44fba8b5a16b1e8c430d3d4c87` |
| `logs/taskb_policy_backbone_smoke/f13_contact_attribution/result.json` | `ce24d2e9c4d79a4bbc63ea59083f89a84c36a3f291f37c4b296d481783efd6d5` |
| `logs/d1g2_taska_training_20260909/rough_512_01/result.json` | `f0ff6a923effc1c0402756e41f6b8f51b49667f15b66a6e382a5fc6f272f7c32` |
| `demo/policy.pt` | `3408897c12072838a4b1194eef6782d98c2c9ea8b07cc0712a8b65e1f748e38e` |
| `demo/test/solution_task_b_candidate_E.py` | `abc1c5c16145530b83204964474d9093e4b0de5cc66b08a7859df90ed197e4c8` |

Task A 硬件评测来源：`logs/d1g2_taska_20260909/residual_1999_axis_recovery_course_02/audit_dependencies/run_acceptance_audit.json`。Task E 成果与方法见本仓库 [Task E 说明](TASK_E.md)。

本次通过只读 GitHub API 核对官方 `main` 提交 `4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc`：Task B 的 `env_cfg.py`、`terrain.py`、`mdp/rewards.py`、`mdp/terminations.py` 和 `task_base/envs_base_cfg.py` 与当前本机文件逐字节一致。评估依据是这一明确版本的公开实现，不替代赛事平台未提供的后续资格规定。
