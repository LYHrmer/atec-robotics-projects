# Task B 自主投递路线（已归档：第一次完整尝试）

本目录是 ATEC2026 仿真赛 Task B 的**第一次完整自主尝试**，开发于 2026-09-14，目标是在官方
`ATEC-TaskB-B2wPiper` 环境中，只凭**公开观测**（本体感受 proprio + 机器人自带 RGB-D 相机）
从零走完整条链路：低曲率行进与航向调度 → 有界重复接近 → 开爪腕部旋转加接触抓取 → 负载下
举臂/侧摆 → 负载搬运 → 用公开 RGB-D 观察桶壁 → 顶层投递状态机。多个模块以**子类化或组合**
经过审计的基类来扩展，而不是改动基类。**本路线已被取代（SUPERSEDED）**：现在出货的路线是
`task_b/grasp_probe.py`——一个用 oracle 定位探针的抓取流程，它确实能完成一次完整投递
（见 `task_b/results/first_delivery_video.json`，其中 `objects_in_circle = 1.0`）。本目录
中的**任何文件都不被出货路径 import**，保留下来只作为记录。

## 文件清单

| 文件 | 行数 | 本模块负责什么（据其自身 docstring） |
| --- | --- | --- |
| `locomotion.py` | 951 | 低曲率前进/弧线轮速候选；只重写四轮动作切片，其余动作逐项保留 |
| `heading_schedule.py` | 524 | 转向优先的速度调度：按 bearing 紧缩前进、把差速留满 |
| `multi_reach.py` | 526 | 只用公开观测的有界重复接近尝试（组合冻结的 `FirstReachPolicy`） |
| `contact_grasp.py` | 918 | 开爪腕部旋转加额外一次腿参考，再交给冻结的 force75 探针（`GraspProbePolicy` 子类） |
| `payload_motion.py` | 1416 | 抓取后的有界负载举升/侧摆/侧伸；纯组合，子类动作与停止原因原样透传 |
| `carry_drive.py` | 881 | 持物搬运阶段的低速平面与偏航率轮控，闭环在**实测公开 twist** 上 |
| `bucket_observation.py` | 804 | 公开 RGB-D 观察桶壁，并附诚实的质量数据（不发动作） |
| `delivery.py` | 1511 | 顶层 carry 状态机：A 举升交接、负载移动探针、投递 |
| `official_locomotion.py` | 180 | 用官方 locomotion checkpoint 驱动 B2wPiper；仅诊断 |
| `public_odometry.py` | 199 | 公开 twist 的平面航位推算（只读 proprio `[0:3]`/`[3:6]`/`[9:12]`） |
| `evaluate.py` | 797 | **本路由自带的早期版本**评测器（出货的 `task_b/evaluate.py` 为 1146 行，不同） |
| `grasp_probe.py` | 1015 | **本路由自带的早期版本**抓取探针（出货的 `task_b/grasp_probe.py` 为 413 行，不同） |
| `audit_contact_grasp.py` | 522 | 上述候选的独立 CPU 反例审计（合成，无物理/无接触/无奖励） |
| `audit_delivery.py` | 1256 | 投递契约的独立 CPU 反例审计 |
| `audit_grasp_probe.py` | 936 | 有界抓取探针的独立 CPU 反例审计 |
| `audit_heading_schedule.py` | 92 | 航向调度的合成 CPU 检查（无仿真器） |
| `audit_locomotion.py` | 184 | 行进候选的 CPU 边界检查（合成，非物理测试） |
| `audit_multi_reach.py` | 720 | 两目标包装器的独立 CPU 反例审计 |
| `audit_public_odometry.py` | 226 | 合成时序检查与离线公开里程计诊断 |
| `HEADING_SCHEDULE_DESIGN.md` | 215 | heading schedule 的设计文档 |
| `LOCOMOTION_DESIGN.md` | 54 | locomotion 候选的中文设计文档 |

## 为什么停下：底盘没有可用的偏航权限

`official_locomotion.py` 的 docstring 直接给出了结论，原文引用：

> "Why this exists. Task B cannot be cleared while the base turns at ~0.0007 rad/s:
> one delivery costs ~1050 s and the episode limit is 1200 s, so at most one object
> per episode and never eighteen."

`delivery.py` 里写的是另一个（更晚的、被它标为实测的）数值：`MEASURED_YAW_RATE_RAD_S = .0016`，
注释为 "Measured on plan_d2/d3: this chassis achieves ~0.0016 rad/s of yaw against a
0.08 rad/s request in BOTH the in-place and the rolling-arc form"，并据此把路线改成正面停靠：
"a side dock needs 2.064 rad of heading change, which is ~1290 s and exceeds the 1200 s
episode limit on its own, while a frontal dock needs only ~0.49 rad."

**能核实的**：0.0016 rad/s 这个量级可由回合自身的遥测核实——`plan_d2_carry_probe` 与
`plan_d3_carry_probe` 的 `stop_reason` 就记着 `no_yaw_progress:+0.0086_in_5.00s` 与
`+0.0081_in_5.00s`，即 0.0017 / 0.0016 rad/s。**不能核实的**：`~1050 s` 每趟与 `1200 s`
回合上限只是源码自己的说法——这些 `result.json` 不记录环境的 `episode_length_s`，且本批回合
的 `max_steps = 50000`（`dt = .02`，即 1000 s）。两个偏航率数字（~0.0007 与 ~0.0016 rad/s）
在源码中并存，本 README 无法判定哪个是最终口径。

**记录在案的结局**：`/home/lybm/ATEC_Experiments_20260910/task_b_score` 下我核对了 **46** 个
`result.json`，其中属于本路线六个模式（`carry_probe`、`first_delivery`、`multi_reach`、
`contact_grasp`、`payload_motion`、`official_locomotion`）的有 **21** 个。这 21 个里 **19** 个的
`reward_term_totals_raw` 是 `grasped_objects: 1.0` 配 `objects_in_circle: 0.0`；剩下 2 个是
`official_locomotion` 纯行进探针，两项都是 0.0。全部 46 个回合的 `objects_in_circle` 都是 0.0，
**没有任何回合与此模式相矛盾**。七个真正开到行车阶段的回合（3 个 carry_probe + 4 个
first_delivery）中，5 个停在 `delivery_drive_fault: no_yaw_progress`，另两个分别停在
`delivery_probe_brake_timeout` 与 `delivery_release_sequence_complete`；其余回合更早就停在
抓取/举升/搬运阶段。任务描述里的"约 60 个回合"我**无法从该目录复现**（实测 46 个）。

## 这里仍然有用的东西

* `public_odometry.py`：可复用件。其 docstring 自称在两次已完成回合上重放得到 XY 误差
  16.7 mm 与 19.8 mm、yaw 误差约 0.008 rad——这是它自己的有限回合测量，不是精度保证。
* `bucket_observation.py`：桶壁观察与**诚实的质量数据**（它明白写出深度从未在真实帧上跑过，
  因为 p13 只存了 RGB）。它给出的几何事实仍值得留：头部相机在约 3.3-3.5 m 以内才看得见该桶，
  且桶面在中性光照下无彩度，颜色检测不可用，线索必须是深度表面朝向。
* `heading_schedule.py`：航向调度本身，以及它记录的两条硬结论——"停轮不是物理恢复"
  （pulse 变体在 `sin(tilt)=0.12` 停轮仍在 step 447 终止），以及"先把距离花完会让目标离开相机
  FOV"。
* `locomotion.py`：H1–H3 三条可证伪假设，和它查到的执行器配置
  （`ImplicitActuatorCfg(stiffness=0.0, damping=1.0, effort_limit_sim=20.0, velocity_limit_sim=50.0)`）。
* `payload_motion.py`：`build_payload_goals`（纯静态规划器）与 `PayloadJointTracker`，其 docstring
  明确说这两个是 "deliberately small and public so a later delivery wrapper can reuse them"。
* 代码里的自我警告值得保留原话：`carry_drive.py` 写 "This module is a **candidate and a
  hypothesis, not verified capability**."；`locomotion.py` 写 "Nothing here has been run in
  simulation."；`heading_schedule.py` 写 "None of it is verified safety, and no motion, yaw,
  alignment or Task B score is claimed."；`bucket_observation.py` 写 "Status: candidate and
  hypothesis. Nothing here is a verified result."

## 注意

本目录中多个模块自我描述为**从未在仿真中验证过的候选**（`locomotion.py`、`heading_schedule.py`、
`carry_drive.py`、`bucket_observation.py` 等），`audit_*.py` 全部是**合成 CPU 检查**（无物理、
无接触、无物体、无奖励），并且本归档**没有产生任何分数**。请勿把这里的任何内容读成结果。
