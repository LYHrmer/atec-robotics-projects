本次是原赛道最后台阶的 109→131 m 诊断短测。48.14 s 时机器人到达 x=131.006561 m，正常触发 `diagnostic_segment_exit`。`reached_goal`、`goal_triggered`、`full_course_reached_goal` 均为 false，因此这份证据不代表完整 Task A 通关。

只读 CPU 验收读取了全部 98 行 trace，并核对 14 份归档源码的 SHA-256。没有运行仿真或重新估计整段轨迹，也没有修改已冻结的控制代码。完整数据见 [cpu_acceptance_audit.json](cpu_acceptance_audit.json)，逐样本误差见 [cpu_acceptance_samples.jsonl](cpu_acceptance_samples.jsonl)。

| 重锚时刻 | 候选最终参考时刻 | 整个缺口 | 其中惯性跨接 | 缺口三维路程 |
|---|---|---|---|---|
| 42.9 s | 42.7 s | 0.3 s | 0.1 s | 0.139454 m |
| 43.7 s | 43.5 s | 0.5 s | 0.3 s | 0.271672 m |

两次事件均由候选连续两帧通过原 PnP 检查后提交，明确标记 `inertial_reanchor`、`visually_connected_to_previous_segment=false`。候选建立 4 次，其中 2 次是失败后重建；原始坐标系、完整 XYZ、bootstrap 元信息及 episode 计数保持继承。总消耗为 2/4 次重锚、0.8/4 s 失联时间、0.411126/2 m 三维路程；每个缺口也都低于 2 s/1 m 限制。所有采样中均无预算停止。

50 Hz 时序方面，全部 trace 的 `proprio_samples=step+1`，包含 t=0 初始样本；camera 快照中的计数等于 `camera_step+1`。姿态时间戳匹配当前控制时间，位置年龄匹配绝对时间差，episode 各计数单调。归档主程序在每次 `env.step` 后仅调用一次 `observe_proprio(..., steps*0.02)`，随后相机使用同一时间戳，未额外积分 0.1 s。保存的 proprio 为 10 Hz，因此这份验收不能逐项重算全部 50 Hz 积分，只能结合调用代码和累计计数验证时序。

视觉失联期间独立航向更新 4 次。归档 `VisualNavigator.observe_heading` 仅更新 yaw、航向新鲜标记与计数，不更新 XY 或视觉年龄。动作策略只在 episode 开始前 reset 一次，恢复模块不访问 actor 或动作历史。

误差均在运行后用相同相机时间戳的诊断真值计算。XY 使用诊断入口 [109,0] 作为评估原点；z=0 是 2 s bootstrap 时人为指定的相对高度基准，评估时才映射到该帧实际高度 0.456367 m。该高度从未传入控制器。

| 相机时刻 | XY 误差 | XYZ 误差 | 航向误差 |
|---|---|---|---|
| 10.0 s | 0.2040 m | 0.4217 m | −0.1412° |
| 42.5 s，重锚前 | 0.1606 m | 0.4074 m | −0.1215° |
| 43.0 s，第一次重锚后 | 0.1636 m | 0.4202 m | +0.1027° |
| 44.0 s，第二次重锚后 | 0.1680 m | 0.4226 m | −0.2268° |
| 48.1 s，最后相机帧 | 0.1744 m | 0.4282 m | −0.0176° |

42.5→44.0 s 的误差向量变化范数为 0.01680 m。末端 z 误差 +0.3911 m 大部分在重锚前已经存在：10 s 为 +0.3691 m，42.5 s 为 +0.3744 m。导航只使用 XY 和航向；不能把局部桥接厘米级误差描述成整段定位精度。trace 仅 2 Hz，事件记录能给出精确重锚时刻和候选起点，但部分重锚提交位姿本身未被逐帧记录；JSON 分别标明候选起点误差和首次记录的重锚后误差，没有将后者冒称为事件时刻误差。

输入边界审查未发现世界位姿、真实 yaw 或地形高度进入 actor/nav：主程序只向策略传入关节状态、gyro、projected gravity 和 command；惯性桥接增加原 proprio 的 body linear velocity；轴约束使用 RGB-D、K、gravity 和已有估计 yaw。已知赛道网格方向是明确的静态先验。诊断测试框架另外使用真值做入口落点高度采样、x≥131 退出检测、日志、评分、原始终止条件和停滞监测，这些用途必须保留披露，不能声称整个 runner 完全不读真值。原环境 `done` 在诊断退出之前处理，诊断成功不会被写成原任务成功。

原验证机器上的历史验收命令如下。它依赖原日志目录内的 `audit_success.py` 和未上传的逐帧传感器归档，不能在当前公开目录直接执行。这里保留验收报告与逐样本结果；启动任务请按 [运行说明](../../README.md)。

```bash
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lybm/miniforge3/envs/isaaclab/bin/python logs/d1g2_taska_20260909/last_stairs_camera_recovery_02/audit_success.py
```

冻结恢复模块 SHA-256：`e97117fbcbc19c0402cbc342ef3b4590615e5e44b3877337b8ab872e0124792b`。
