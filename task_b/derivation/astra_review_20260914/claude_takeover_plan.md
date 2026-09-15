# Claude 接手：首次真实物理投递

本方案由实际 GPT-6-Astra ultra 制定。当前工作区为 `/home/lybm/ATEC_Robotics_Projects_20260910`；私有实验与 Opus 合同位于 `/home/lybm/ATEC_Experiments_20260910`。交接时只读核验：`public_odometry.py` 已实现；`contact_grasp.py` 与 `payload_motion.py` 尚未落盘，前者实际 Opus 会话正在生成，后者已有完整提示。先核对正在运行的会话与文件所有权，避免两个进程同时写同一文件。

**可直接转交 Claude 的总指令：**

请接替我持续完成 Task B 的首次真实物理投递，按下列 Astra 方案直接实施、审计并运行，不停在计划或接近得分：当前最好只有1个接近分、0投递，尚无任何真实抓起证据。先接收正在生成的 ContactGrasp 薄模块，完成独立CPU审计，再以新进程运行原环境，用同一物体的实际四元数、原mesh瓶底高度及相对夹爪跟随至少1秒证明物理抓持；只有该证据成立才启动高举、公共里程计导航、桶外侧停、越沿张开放下并核验原objects_in_circle。夹持失败时依据真实双指法向及接触几何选择一次有界修正，不能只加力、无限扫位置或降低验收门槛。复用已完成的public_odometry与已备好的payload合同，只补当前缺失的小模块，不重建通用控制框架。所有策略只用原公开proprio/RGB-D及静态任务几何，GT/物体位姿/接触力/奖励仅用于外部审核，原物理、奖励、终止不改；GPU严格串行，代码与每次完整运行证据冻结。2026-09-14 11:23:29 UTC（北京时间19:23:29）之前绝不上传；之后仅发布已核验的Task B最佳展示，较低结果留本机，保留既有A/E，不改写Git历史。除实际阻塞外继续自主推进，不能将接近、非空夹爪或关节到位冒充抓取/投递。

## 按依赖完成，不并列扩建框架

1. **先完成当前接触候选。** 读取 `task_b_plan_opus/contact_grasp_prompt.md`，只新增 `task_b/contact_grasp.py`，继承冻结 GraspProbe。同参冻结 FirstReach(.02)正常完成后，OPEN→公开实测q6相对+30°→独立额外 .02 腿参考→原force75闭夹/小抬升；总参考下降 .04 不等于实际下降 .04，不扩大 FirstReach 的 .03 API上限。原夹紧preload .075、q2 P3/cap .12/tether .18/tau .1、paired .20及实际验收保持。前缀动作、轮锚、单位、率/实际q tether/硬限交集、完整静止窗口、失败传播均由独立审计验证。准备加probe≤25秒，body/pause预算不因转阶段清零。

2. **真实抓持是载物准入。** 原force75末端曾上升54.4毫米，瓶底最高仅增加0.864毫米，最终空夹；材料实际μs=μd=1，瓶质量 .5kg。双指均接触但法向对物体有明显向下分量，抬升期仰角26.7°/29.6°，所以当前改接触位置。新的运行应分别保留准备前、闭夹前、抬升后的实际object/gripper quat与原mesh变换；不得以闭夹前重置基线掩盖准备阶段已经推倒或移动瓶子。外部准入：同瓶mesh最低面比原支承升≥10毫米、root升≥15毫米，并至少1秒保持相对夹爪位置稳定（漂移≤20毫米）、无原非法接触/终止。官方近距分和策略normal完成均不足以证明抓起。原始指法向传感器不含摩擦，不能当全部接触支撑。

3. **若仍滑脱，收敛到具体根因。** 先区分准备动作未执行到位与实际夹持失败；前者修真实执行原因或有界跟踪，不能降物理验收门槛。准备到位后，若法向仍落在斜肩，按本次实际姿态重放原mesh，检查公开RGB-D可估的瓶体窄轴与手指深度，再选择一个有完整旋转/下探路径依据的腕角或接触深度修正；参数选择不能读取运行中GT。若双侧法向已接近水平而仍滑动，核对实际开口、位置参考产生的夹紧量、成对抬臂是否伴随横向拖拽/转动，优先改为保持夹爪方向的短程上举，不能继续无依据加夹力。每轮只回答一个明确假设；同一接触路线最多再做两项有数据依据的修正，若均失败，转向公开RGB-D定位的直壁侧夹/另一可达瓶体重新取得接触，而非放宽“离地/跟随”标准。每次候选先审姿态与原硬限，再单GPU实测。

4. **抓持成立后高举A，并验证载荷整条路径。** 完整合同为 `task_b_plan_opus/payload_motion_prompt.md`，只新增 `task_b/payload_motion.py`。三个小接口：`build_payload_goals(reference_q,current_q)`、`PayloadJointTracker(dt,initial_reference,initial_command)`（`begin(goal)`；`update(q,qd,advance_path=True,paused=False)`）、薄`PayloadMotionPolicy`。组合ContactGrasp，仅拦截normal `grasp_probe_observation_complete`，之后不再调用已完成child，避免其25秒旧时钟影响新阶段。保存child完整最终leg action，包含两层下降；不能只读FirstReach.lower_delta。由新的公开lift_start_q定义R0，按合同A高举→B仅q1侧摆→C侧伸；A实际到位前不能低位横摆。静态载荷界支持q2 P4/cap .14/tether .18、q3 P2/cap .08/tether .10、tau .1、command rate .10，固定actual goal误差 .04保持。先远离桶实测完整A/B/C；连续瓶底/夹爪跟随不过则停止，不进入运输。旧几何余量基于旧刚性夹持假设，须对新夹点重新核验。

5. **薄投递编排：A后运输，B/C在桶边执行。** `PublicPlanarOdometry(dt=.02).update(proprio)`必须从回合第一帧每控制步恰好调用一次；只按原固定公开spawn(-10,-10,yaw0)初始化，不能在抓起后重置或用GT校准。它已有两个完整回合约17毫米终点XY误差的离线证据，不能当运输精度保证。公开桶心c=(-3,-10)、外半径1、实体桶沿z≈.55。由当前估计p定义n=(p-c)/||p-c||、t=R90*n，侧停点c+1.45n，预停点=侧停点−1.0t；先去预停点，在桶外约1.76米处制动转向t，再前进约1米至侧停点，使桶在body+Y。低速初候选 .04–.05m/s、上限 .08，yaw≤.08rad/s，配合公开RGB-D检查行进障碍。Tracker的advance_path=False仍保持A目标闭环；移动控制需独立有限路程/时限，不能直接套静止模块.03米漂移预算导致必停，也不能删除姿态/夹爪/载荷保持条件。只有实际开始移动时释放轮锚；制动后先消除滑行，再形成静止窗口。原multi第二目标曾因尚以 .235m/s滑行便建立 .03米静止预算而停，勿重犯。

6. **过沿、释放、原规则复核。** 桶外底盘中心距约1.45米、方向沿切线，A持载稳定后B侧摆、C body gripper目标[.2,.52,.40]；姿态与IK按payload合同从公开R0生成，不能硬编码旧seed的关节姿态或物体坐标。旧候选完整瓶底约.631米、车侧余约74毫米、整瓶最大径向约.901米，仅为静态设计依据；应审核新夹点、真实倾角、整瓶与整臂越沿余量，不能只检查gripper origin。保持实际到位与轮锚，以原.05m/s指令slew张开至70毫米，观察至少3秒。必须关联此前夹持的同一物体，核验官方objects_in_circle事件、实际掉入实体桶及后续保持、无原失败；不将接近项grasped_objects再次当投递。

## 复用证据与运行纪律

- 首分完整运行：`task_b_score/plan_p2_lower02_seed42_01`；原 `task_b/audit_positive.py`已验证1个接近分。最新失败与法向诊断：`plan_p4_grasp_force75_seed42_01`、`plan_p4_grasp_force75_contactdiag_seed42_01`。启动新候选前核对最终result与源码快照，不能只看live阶段。
- 实际瓶体审计：本目录`audit_grasp_runtime.py`，独立报告在`task_b_score_cpu`；联系当前审计负责人复用，不能私改测试使其“通过”。接触静态依据见`contact_acquisition_decision.md`、`contact_force75_wrist_depth.json`、`contact_rotation_clearance_followup.json`；高举依据见`task_b_score_cpu/payload_raise_geometry.json`。
- 原环境每次新进程、单GPU，无中途重置规避原失败；停止原因和最后触发proprio必须入档。RGB-D/公开关节控制与GT诊断文件严格分离。接手先收尾当前Opus文件所有权，再实施缺失模块，禁止重复同时编辑。
- 19:23:29北京时间前只本机实验。之后展示内容以该任务经过核验的最佳成绩/完成度为准：Task B较低视频与证据本机保留，既有A/E保持；未验证的“成功”不发布。截止时间不是抓持成功证据，不因时间将结果写高；之后按用户已有发布授权与最佳结果策略处理，无需重复索取已给出的许可。
