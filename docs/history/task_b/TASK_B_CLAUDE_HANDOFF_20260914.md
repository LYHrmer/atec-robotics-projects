# Task B 接手事实与操作入口

记录于 2026-09-14 10:31 UTC 左右。交接后先检查当前文件、进程、结果，本文不是实时进程监视器。

## 用户目标与权限

用户要求 Claude 接替继续做 Task B，优先完成第一次真实投递，再争取更高分及通关。用户已授权正常开发、仿真、修复、最终 GitHub 发布，不需要重复询问是否继续。不要只给计划。原始物理、资产、评分和终止条件必须保持真实可核验。

用户反复追加优化时间后，最新明确的禁止上传截止为 **2026-09-14 11:23:29 UTC / 北京时间 19:23:29**。之前的 10:20、10:23 UTC 等截止均已被覆盖；截止前继续本机优化，不 push、不发布 release。到点意味着可以发布，不意味着必须停止探索或立即上传。最终远端 Task B 展示只保留最高且经过验证的成绩，失败与较低分实验留本地；保留已有 Task A / E 视频与成果，不重写 Git 历史。

## 当前真实结果

- 最高已验证是 **单回合 1 个接近分，0 投递，尚未证明真实抓起**；不同回合的 1 分不能累加。
- 原评分的 `grasped_objects` 实际只检测夹爪基座距物体根 <= 0.20 m，并不证明抓住。首次 `objects_in_circle` 才是该物体投递分。
- 桶中心为原任务常量 (-3, -10)，评分要求物体根在半径 1 m 内且 z 在 [0, .5]。实际桶内半径约 .98 m、桶沿世界 z 约 .55 m；不能低持瓶直接穿墙进圈。全 18 个物体同时在圈内才满足完整任务。
- 最近 `plan_p4_grasp_force75_contactdiag_seed42_01` 仍为 1 分、空夹停止。双指确实连续接触，夹爪上升但瓶底最高只离原基线约 .865 mm，随后滑脱。根位置轻微升高、夹爪高度和非空夹缝都不能替代物体抓持证据。
- 实际只读 PhysX 读回：瓶重 .5 kg，机器人/瓶摩擦系数均静 1、动 1、恢复系数 1；附着在 B2w 的 Piper 所有 arm_joint 实际 K=80、D=4、max_force=100。Task E 独立机械臂的 K=800/D=80 不能照搬当成本环境事实。
- 现有 finger_contact_forces_w 来自传感器 net_forces_w，**只有法向力，不含摩擦力**，且没有 object10 专用过滤。结合几何可诊断接触，不能直接把它当完整抓持支撑力。

## 工作区与运行

- 项目：`/home/lybm/ATEC_Robotics_Projects_20260910`，当前分支 `organize-atec`。
- 原挑战只读：`/home/lybm/ATEC2026_Simulation_Challenge`，原提交 `4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc`。
- 运行 Python：`/home/lybm/miniforge3/envs/isaaclab/bin/python`。
- 私有实验根：`/home/lybm/ATEC_Experiments_20260910`。
- 历史流水：`TASK_B_SCORE_ACTIVE.md`，从顶部读最新条目，旧条目不是当前状态。
- GPU 为 RTX 5060 Laptop 8 GB，**同时只跑一个 Isaac 实例**；CPU 几何、代码、审查可以并行。
- 每个正式成绩必须原环境重新启动、独立新输出目录。不得把物体/机器人搬到目标附近或从中途检查点拼接成正式通关。
- 策略只能使用公开 proprio84、RGB-D 和原场景静态常量/机器人几何；物体世界状态、接触力、世界底盘位姿、奖励只给独立诊断，不能决定策略动作、目标筛选或状态推进。
- 两仓库原先都没有 `.codegraph/`。若现在有则先用 CodeGraph；没有就不用主动建索引。

## 写作中的实际 Opus 任务：先接好再编辑

旧根代理已启动一个真实 Claude CLI Opus 子任务，**仅拥有 `task_b/contact_grasp.py`**：

- UUID `049398b4-7feb-4f9a-94c5-31fd175f59d1`，记录 PID 60914（接手时必须核验是否仍是该进程，不得盲杀复用 PID）。
- 驱动：`task_b_plan_opus/run_opus.py --task contact_grasp`。
- 日志与完成凭据：`task_b_plan_opus/invocation_049398b4-7feb-4f9a-94c5-31fd175f59d1.json`、同 UUID 的 `result_*.json` / `answer_*.md` / `stderr_*.txt`。
- 最新更新：实际 Opus 已自然结束，exit_code=0 / is_error=false，contact_grasp.py 最终 SHA c8684cccc0294de2491ba0692443190b8cb7ce48d041994f4d4416b3f5069af7。原始产物与 receipt 保存于 task_b_plan_opus/contact_grasp_opus_original.py、contact_grasp_source_receipt.json。尚未经过根代理编译或独立审计，不能称验证通过。
- 先确认旧子任务退出，再接管该文件，防止并发覆盖。自然结束后保存原始源码与 SHA，检查 `is_error`，再审查编译。若明确失败则沿同一契约自行实现，勿重启长篇泛化规划。
- 旧独立代理正在保存 `task_b/audit_contact_grasp.py`；在没有真实模块和执行报告前不能称审计通过。用户现在要移交，旧代理已被要求保存后停止写入。

## 紧接着做什么

1. 阅读已完成的 Astra ultra 方案 `task_b_astra_review_20260914/claude_takeover_plan.md` 以及具体实现契约 `task_b_plan_opus/contact_grasp_prompt.md`。方案开头的“contact 尚未落盘”是更早的快照，以本交接及实时文件状态为准。
2. 接好 ContactGrasp 产物及独立审计。新方案复用完整 FirstReach(.02) 前缀，张开夹爪 → 仅腕 q6 +30° → 再额外降低 .02 m 腿参考（总名义 .04）→ 复用冻结的闭夹、抬升、观察。不改原 FirstReach 的 .03 限制或原物理参数。
3. 确认 CPU 审计与边界检查通过、无其他 GPU 后，用已备脚本 `/tmp/taskb_contact_grasp_20260914.sh` 跑第一轮真实测试。它设置 `--mode contact_grasp --seed 42 --max_steps 8000 --reach_lowering .02 --score_hold_steps 0`，新目录 `task_b_score/plan_p5_contact_wrist30_lower04_seed42_01`。若该目录已存在并有运行结果，读取结果，不覆盖；新建后缀用于新实验。
4. 根据真实 quat + 原瓶完整 mesh 算物体底面离地和相对夹爪持续跟随，分别以准备阶段前、闭夹前作为基线，避免把下压/推挤误认抬升。记录双指法向、宽度、arm q/qdot、身体漂移、原终止标志，识别单侧没夹住、斜面挤出、负载跟踪或车身问题。不要靠放宽“成功”阈值宣称抓住。
5. 抓持代码运行的同时，可以实现 `task_b_plan_opus/payload_motion_prompt.md` 中明确的高举模块并做 CPU 检查；**尚未确认抓稳前不把整段运输测试当作有效载荷试验**。目前 payload_motion.py 尚未完成，不能直接假设 mode 已接通。
6. 确认抓稳后：A 安全高举 → 保持载荷、公共里程计导航 → 桶外切向停靠 → B 侧摆 → C 越过桶沿 → 张手释放 → 持续观察 → 原始评分与完整物理证据复核。先闭环首个投递，再扩展重复抓取及更多物体。

## 可复用模块与边界

- `task_b/first_reach.py` SHA `5efb6324f3b9f6b2f328d060e84ea60564b7580dab6b81e00a03b57d74d2b897`，冻结；完整前缀已真实跑通。
- `task_b/grasp_probe.py` SHA `bf8a4204c5badd5c1708c2804e9e4a085e29535f88cfba40cb527b20abb4e306`，冻结；89 项 CPU 检查通过，但真实瓶子会滑脱。闭夹 targets [-.075,+.075] 是原位置动作下的有限预载，真实关节仍受原硬限；不能改 K/D/摩擦代替改控制。
- `task_b/public_odometry.py` SHA `9bf8b464415a5b211c8d5e9d744e625b2d7da11c6635cb9dab02f4979216967b`，12 项 CPU 检查通过。`PublicPlanarOdometry(dt=.02).update(proprio84)`，固定原出生 (-10,-10,yaw0)，积分前一观测时段公开切向速度和绕重力轴角速度，`target_body_xy()` 支持静态航点。两段实际回放末端位置误差约 17 mm，仅为离线诊断，不保证载荷运输精度，不得据此喂 GT 校正。
- `payload_motion_prompt.md` 已定义 `build_payload_goals(reference_q,current_q)`、`PayloadJointTracker`、组合式 `PayloadMotionPolicy`。新 ContactGrasp 正常结束的同 tick 动作原样返回，之后不再调用子策略，避免已结束的 25/12 秒时钟误杀高举。必须保存完整最后 action，不能漏掉第二层腿下降。失败不能被重分类为成功。
- PayloadJointTracker 的 `advance_path=False, paused=False` 仍闭环持载，供 A 后导航使用；paused 才冻结路径/滤波/命令。q2 P4/cap .14/tether .18、q3 P2/cap .08/tether .10，tau .1，命令最大轴速度 .10 rad/s，保持实际误差 .04 和完整 .5 秒位置静稳窗口。都是方案参数，不是已实测载荷成果。
- 桶侧停靠初始规划：底盘距桶中心 1.45 m、桶在左侧，先到切向前置点再转向最后直行。完整刚体假设几何报告只是规划依据，新夹持姿态必须复算瓶体/轮子/车体到墙与桶沿净空。低速导航起步 .04–.05 m/s，明确制动再启用静止预算。
- `multi_reach.py` 是后备多接近分路线，不是投递路线。它已看见第二物体，但倒车到重新静止时尚有惯性，触发 .03 m 漂移预算；不要优先花 GPU 刷接近分，亦不要复制其制动切换缺陷到投递导航。
- `task_b/evaluate.py` 已本地接入 contact_grasp mode，以及只读材料/驱动力/双指法向诊断；还没有完整导航投递模式。原有默认行为不应被实验模式破坏。

## 关键证据和命令

最新力诊断：`task_b_score_cpu/grasp_force75_contactdiag_interpretation.md`、`grasp_force75_contactdiag_force_review.json`、`grasp_force75_contactdiag_independent_mesh.json`。

新夹持几何：`task_b_astra_review_20260914/contact_lowering_conditional_plan.md`、`contact_lowering_geometry.json`、`contact_acquisition_decision.md`。特别注意：原 .03 高度直接转腕路径会擦左指，因此采用先 .02 高度转腕再额外下降；新路径最终开口余量仍很小，不能宣称必然无碰撞。

高举/停靠：`task_b_astra_review_20260914/first_delivery_execution_contract.md`、`task_b_score_cpu/payload_raise_geometry.json`、`q6_30_raise_reparameterization.json`。旧报告若与更新 prompt 相冲突，以最新明确公式及实际反馈为准，不能照抄旧硬编码 q。

实际驱动诊断：`task_b_diagnostics/live_materials_drives_20260914/environment_metadata.json` 的 physics_diagnostics；对应 1-step GPU 实验已经退出。

在项目根可用：

```bash
PYTHONNOUSERSITE=1 /home/lybm/miniforge3/envs/isaaclab/bin/python -m task_b.audit_grasp_probe --output /home/lybm/ATEC_Experiments_20260910/task_b_score_cpu/grasp_probe_takeover_audit.json
python3 /home/lybm/ATEC_Experiments_20260910/taskb_progress.py /home/lybm/ATEC_Experiments_20260910/task_b_score/plan_p5_contact_wrist30_lower04_seed42_01
```

`python -m task_b.audit_positive RUN_DIRECTORY --output REPORT` 用来核验原始正分；审计器 SHA `f31f81582633b4fc53157b96e403a1fea4b1e571911a504c2923aeef555904ca`，不要改它来让结果通过。接近分审核通过也不等于真实投递已通过。

## GitHub 状态与交付

- 统一仓库 `https://github.com/LYHrmer/atec-robotics-projects`，Pages `https://lyhrmer.github.io/atec-robotics-projects/task-b.html`。
- 最后已推 main 的提交 `7b45099b0981364ac2ccda18c1dbe3a51fc51e57`，仅展示首次 1 接近分。
- 现有 draft release `task-b-first-score-20260914` 尚未发布；原片、证据等附件已上传草稿，页面对应下载链接在草稿公开前不能当作已可访问。截止前不要发布，之后根据最佳真实结果统一更新。
- 当前有很多未提交文件。先 `git status --short`，只暂存本次明确产物；heading_schedule/locomotion 的旧未追踪实验不要顺手 `git add -A` 带入。
- 发布前保留最高结果代码快照、seed/参数、官方分项、真实投递次数、失败限制、原视频及可播放网页视频。发布后核验 Pages 实际播放/拖动和 release 附件，不把还在草稿的链接写成已可下载。
- 继续把真实进展写入 TASK_B_SCORE_ACTIVE.md，注明每轮改动、源码 SHA、命令、结果、独立证据和失败原因。最终报告区分接近、抓起、投递、通关，不夸大。
