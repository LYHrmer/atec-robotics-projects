# Task B 首个芥末瓶真实抓取：最小实施方案

2026-09-10，CPU 源码/静态 USD 核查。原任务不修改；运行物体坐标、root pose、root quaternion、接触传感器仅用于评估，不进入下面的控制方案。目标先取得一次有视觉证据的夹持与抬升，再做真实投放；`grasped_objects` 的接近分不能代替抓住。

## 已可直接使用的代码

统一工程：`/home/lybm/ATEC_Robotics_Projects_20260910`。

| 函数 | 适用方式 / 必须传入参数 |
| --- | --- |
| `task_b.arm_kinematics.fk(q6_or_q8, return_jacobian=False)` | 输出 `T_body_gripper_base`；明确采用 B2w 静态 arm mount `t=(.2,0,.1), R=I`。不读取世界位姿。 |
| `arm_joints_from_proprio(proprio84, observation_joint_names24, default_joint_positions_by_name)` | 从 `proprio[12:36]` 取相对 q，按观测 joint_names 映射，加静态默认角，输出 arm_joint1..8。B2w 的臂默认全 0。 |
| `ee_camera_transform(q)` | 输出 `T_body_ROS_ee_camera = FK(q) @ [Rz(-π/2),(-.05,0,.06)]`，与官方 EE 安装一致。 |
| `solve_ik(gripper_position_body, rotation_body, seed=q[:6], multi_start=True, max_nfev=90, position_tolerance=.005, orientation_tolerance=.08)` | 只在阶段开始多起点求解；必须检查 success 和误差，失败不能强发最近解。连续路点用前一解作 seed、`multi_start=False`。 |
| `differential_ik(q, p_body, R_body, damping=.025, max_joint_delta=.025, orientation_weight=.20)` | 50 Hz 小步伺服建议起点；只保证关节限位，不保证碰撞或底盘平衡。 |
| `arm_targets_to_action(targets8, action_arm_joint_names, defaults_by_name, scale=.5)` | 输出实际 arm term 顺序的归一化 8 维，仅替换 arm slice，保留腿与轮动作。须先验证当前 position/default-offset/clip=None schema。 |
| `task_e_geometry.unproject_depth(depth, intrinsics=K, world_from_camera=T_body_cam, mask=mask, return_pixels=True)` | 函数参数虽叫 world，矩阵可直接给 body_from_camera，输出就是 body 点；务必显式传 K/T，不能使用 Task E 默认视频相机。 |
| `task_e_perception._point_components(points, voxel_size=.004, connection_radius=.009)`、`_features(points, rgb)` | 可复用连接组件/形状特征。在重力消倾角坐标里计算瓶高与水平轴。私有函数接口需固定来源 SHA。 |

新增 helper 已完成 CPU 验证：30 个随机关节姿态对 B2w USD 直接关节链，位置最大误差 `7.98e-8 m`，姿态最大误差 `6.46e-7 rad`；数值 Jacobian 最大差 `1.96e-10`；与 Task E 刚体坐标转换后的 DLS 差 `2.39e-15`；8 个 IK 往返目标通过。见 `results/task_b_arm_kinematics_cpu_audit.json`。这不是碰撞或抓取成功证明。

禁止直接复用：Task E 的 `solve_ik`/`differential_ik`（内部固定桌面世界基座）、`joints_from_proprio`（前 8 维且加 Task E 默认角）、`estimate_objects_detailed`（固定桌面 ROI、每类只配一个目标）、完整 `AlgSolution`（固定篮位置与 Task E 每事件 3 分口径）。Task E IL 残差训练分布也是桌面操作，首轮先用解析 IK/FSM，不宣称可直接迁移其学习效果。

## 合法观测与移动坐标

官方 Task B `env_cfg.py:270–280` 确定观测关节顺序；`envs_base_cfg.py:150–243` 定义：

- `proprio[0:3]` body 线速度；`[3:6]` body 角速度；`[6:9]`环境命令；`[9:12]` projected gravity。
- `[12:36]` 相对关节角，`[36:60]` 相对关节速度，`[60:84]`上次动作。**观测/动作/USD 三种排列必须分开。**
- `image` 包含 `head_rgb/head_depth/ee_rgb/ee_depth`，640×480、10 Hz；不是 Task E `video_rgb/video_depth`。
- depth 是沿光轴距离，反投影 `[(u-cx)d/fx,(v-cy)d/fy,d]`，不能当径向距离。EE `fx=fy=640*15/20.955≈458.12`，head `640*24/20.955≈733.00`。原 raster K 主点 `(320,240)`；若采用 OpenCV 整数像素中心，统一显式改成 `(319.5,239.5)`，只改一次并记录元数据，不能与其他模块混用。
- 对抓取/局部趋近，完全不需要 world root pose：每次用当前 q 得 `T_body_cam`，图像目标直接变换为当前 body 坐标。目标进入近场后轮速清零，稳定再抓。不得长期持有旧 body 点而不随底盘运动更新。
- 重力给 `up_body=-g/||g||`。在 depth 中拟合与 up 相容的地面平面，得到当前 body 离地高度与物体离地高度；这比把某次 world base_z 写死可靠。世界 yaw 不是一次局部抓取的必需输入。
- 相机使用 Task A 无副作用 pose reader，避免创建 camera 自身 Fabric world 属性而破坏随父 link 运动。图像和用于 FK 的 q 尽量同时间；初版仅静止底盘与静止 arm 的 scan 时刻建目标。

## 芥末瓶定位与候选高度

原资产：`atec_robot_model/objects/task_b/006_mustard_bottle.usd`；外部 cfg scale=1，内部 mesh scale=.01。几何尺寸局部 xyz 为 `.096024 × .191301 × .058249 m`，长轴 Y，初始 wxyz `[0,0,-.707,.707]`使瓶盖朝上。质量 `.5 kg`、凸包碰撞、contact offset `.01 m`。Object7–12 是六个瓶，但控制器不能读取对象索引的场景位置。

官方地面顶部是 **z=.045 m**。配置初始根 z=.10 导致瓶底穿地，故必须等落地稳定；保持竖直时理论瓶顶约 `.2363 m`，此值只是静态核算，不能代替视觉测量。

最小合法 pipeline：

1. 初始至少 100 控制步稳态，arm 官方默认位，读取 EE RGB-D。先用饱和黄色 HSV mask 找候选，连通域与深度有效性去噪；这一步只称“黄色高物体”，不能把黄色香蕉/地砖误写为分类成功。
2. 反投影至 body，利用 g 与观测地面平面得到物体高度；近场以 3D 连通组件分离重叠投影，不做每类只选一个的 Hungarian 约束。
3. 芥末候选要求可见竖向跨度与 19.1 cm 静态模型相容、水平短边约 5–6 cm、长边约 9.6 cm，结合黄色支持；遮挡严重则拒绝/再观察。取前方最近的可见候选，跨至少三帧匹配；不能用 seed 重建位置。
4. 优先尝试 **瓶颈上方下降夹持**：沿用 Task E `solution_task_e_rgbd.py:20–38` 的“深度定位盖帽 footprint，夹持点为观测顶部下方 .035 m”思路。原函数的桌面 ROI 与固定相机不可复制。近场先要求组件≥400像素、盖帽顶部 .016 m 带内≥25点；点数不够则靠近再测。
5. 瓶身水平尺寸 96 mm 大于夹爪最大 70 mm，必须使闭合轴沿视觉估计的薄边，不能沿长边夹。侧夹瓶腰候选离地约 .095–.125 m（世界 z约 .14–.17）更难伸到；瓶颈候选离地约 .156 m更有利于未深蹲的 reach。实际以 depth 接触点为准。

静态 IK 窗口测试（无运行真值）：竖直向下接近、body 接触点 y=0 时，x=.40/.55、z=−.25/−.20 均有姿态解；z=−.30 时失败约3.5–3.8 cm；x=.70、z=−.25 时失败约5.9 cm。完整 `task_b_static_pick_ik_grid.json` 同目录。可先把目标趋近到 **body x≈.55、|y|<.03、z不低于约−.25**，再逐个路点确认 IK；这些是采样结果，不是完整可达边界。

此前完整 crouch 已在 195 步触发真实非法接触，不能拿该姿态继续抓。建议先测更浅 fraction=.6 并检查实际稳定高度；若仍够不到，就改变视觉选点/站位，不能放宽官方终止。目标接近的第一版可先停在 x≈.8 m，仅验视觉与方向，再加近场 .55 m 和 arm 动作。

## 抓取与抬升最小 FSM

阶段顺序沿用 Task E，但替换局部坐标/奖励口径：

1. `SETTLE → SCAN → APPROACH → STOP_AND_RESCAN`：底盘低速视觉趋近，丢目标停车；停止后以 EE RGB-D 重测瓶颈/薄边。抓取期间四轮零速，浅蹲角固定；拒绝重力倾斜/速度仍明显变化的帧。
2. `PREGRASP`：接触点上方 `.10–.12 m`，夹爪打开。重力朝下 `approach=-up_body`，jaw 与视觉薄边对齐并投影到 approach 垂面；`R=[jaw×approach,jaw,approach]`。gripper_base 目标为 `contact − .115*approach`。路点间距 `.025 m`，近物体降到 `.010–.012 m`；每点 bounded IK 通过后再执行。
3. `DESCEND`：轮零、实际 q 闭环每步≤`.025 rad`，夹爪开。上方再次定位目标，不能盲用起步旧点。接近后 arm 目标保持实际 q。
4. `CLOSE`：`gripper_targets(0)` 先合拢；如需复用 Task E 压紧方式，显式 `preload_m=.025` 请求 `[−.025,+.025]`，实际 hard limits仍是 joint7 `[0,.035]`、joint8 `[−.035,0]`，未改物理。默认offset0/scale.5时开指令 `[+.07,−.07]`，压紧指令 `[−.05,+.05]`。至少观察约70步，与 Task E初始等待一致。
5. `TEST_LIFT`：先竖直抬 `.06–.10 m`，维持小步 arm 运动；q7−q8 未完全归零可作为接触迹象，**不是夹到瓶的证明**。用 EE/头部 RGB-D 观察瓶是否随手移动、相对瓶手位置是否稳定。旧地面位置不可见也不单独算成功。
6. 仅当视觉随动与夹爪宽度一致，进入 `CARRY`。Task B 一次 proximity 奖励仅+1且可能提前发出，不能复制 Task E `score_gain>=2.9` 的 lift 验证，更不能用累计分数替代释放证据。

首次 bounded 实验只做一次候选、一次抓/抬，最多一次重新扫描重试；无目标/IK失败/空抓/滑落均保存明确原因，不擅自进入长距离运输。

## 过原围墙与真实投放

`tasks/task_b/terrain.py:37–60`：桶半径1.00 m、内半径.98 m、墙厚.02 m；底板厚.05 m，再上建高.5 m墙，故 **墙顶=.55 m，不是.50 m**。mesh32边形内壁最小径向距离约.97528 m。terrain生成的净XY平移为(−10,−10)，桶中心是公开地图的(−3,−10)；但没有合法自身位姿时，不能直接把这世界坐标减去 root 真值来导航。

合法最小路线：保持瓶直立高持，从允许 RGB-D 中识别橙色桶墙/边沿并趋近；或用现成 RGB-D相对里程计到公开地图位置，必须独立验证其 Task B 外参/纹理。接近桶前停车，通过局部可见墙面 depth 保持轮/腿在外侧，再伸臂越沿。

- 运输高度检查对象的最低点和手指最低点。若夹持点接近瓶中心，瓶中心须 `>.55+.191301/2+.03=.67565 m`；可规划 .70–.75 m。若夹瓶颈（观测顶部−.035），瓶在夹持点下方约 `.1563 m`，因此接触点须 `>.55+.1563+.03=.7363 m`。用观测几何更新该包络，倾斜时计算旋转后的全部角点。
- 在底盘局部坐标里，上述高度用观测地面平面与 gravity 转换，不用 true base_z。先竖直提升至包络过沿，再水平伸进桶，最后下降释放；不能边低持边穿墙。
- 临近桶底盘中心到桶中心可先取约1.5 m的试验停距，但最终以可见墙距离、前轮/腿静态包络和可达性核验，不能把此数当已验证安全距离。释放前瓶中心径向≤.88 m可留静态横截面+约3 cm余量，最好更靠内。
- 夹爪明确打开、撤离并等待，再以可见瓶落在桶内且静止、官方 `objects_in_circle` 首次增量联合记录一次实际投放。18个物体同时在圈内才是 Task B 全通关。

## 关键证据来源

- 原 `assets/robots/b2w.py:31–71`：Piper安装/相机/关节名；B2w主USD SHA `516e1b838869289f4b61eef8846ecee167dbd51f55746dca023b05f887fd03db`。
- `task_e_geometry.py` SHA `857940ed4722aace8c02fc5f6391b224c7f4e6bda681ee86510d6233dbf429c4`：Piper共同关节链；Task B helper 与实际USD直接核验，未将 Task E 桌面基座当 Task B base。
- `task_e_perception.py`：`_point_components/_features` 可复用，`estimate_objects_detailed`固定ROI/单实例分配不可直接复用。
- `solution_task_e_vision.py:177–365`：重测、路点、关闭等待与验证阶段；Task E 奖励3分阈值和固定篮位置需要删除/替换。
- 原 `assets/objects/task_b/object.py:42`、`tasks/task_b/env_cfg.py:73–89`、`tasks/task_b/terrain.py:25–60`：瓶/地形静态几何，不是运行目标位置。

目前结论：运动学工具可用；合法视觉趋近、浅蹲稳定性、碰撞自由 arm 路径及一次抓取尚须真实回合逐级验证，没有首瓶成功或 Task B 得分声明。
