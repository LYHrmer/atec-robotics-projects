# Task B 交接：从 oracle 到官方视觉策略

> 目标：把已经打通的「抓起 → 投递」从 **oracle 定位**换成**视觉**，做成官方策略成绩。
> 本文件是自包含交接，读完即可上手。

## 1. 现状一句话

M2（真实抓起）与 M3（投递）已在 oracle 定位下打通并推送，**官方 `objects_in_circle` 真的加过分**；
现在缺的只是把真值换成视觉。

## 2. 路径与环境

| 项目 | 值 |
| --- | --- |
| 交付物仓库 | `/tmp/atec-upload`（GitHub `LYHrmer/atec-robotics-projects`），本地=远端=`e04446f` |
| 官方 cfg（只读） | `/home/lybm/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_b/env_cfg.py` |
| Python | `/home/lybm/miniforge3/envs/isaaclab/bin/python` |
| 无渲染跑（快、稳） | 加 `--camera_free` |
| 开相机跑 | 不加 `--camera_free`（渲染已获用户批准） |
| CPU 审计（上 GPU 前先跑） | `task_b/audit_stance.py`（14 项）、`task_b/audit_grasp.py`（27 项） |

```bash
export ATEC_TASK_ROOT=/home/lybm/ATEC2026_Simulation_Challenge
export ATEC_PYTHON=/home/lybm/miniforge3/envs/isaaclab/bin/python
export PYTHONNOUSERSITE=1
bash run.sh task-b --mode <MODE> --output <必须是不存在的新目录> ...
```

## 3. 已推送成果

| commit | 内容 |
| --- | --- |
| `672225a` | **M2**：夹爪能夹住并抬起物体（升高 73 mm 且全程钳口-物体距离恒定，手指停在瓶身宽度 0.05641 m） |
| `64ed373` | **M3**：官方 `objects_in_circle` 在 step 2195 加分，score 2.0，全程 0 非法接触 |
| `6884533` | 站姿包线（机身可降 0.253 m，9 个官方接触体受力全 0）+ 无渲染能力 + 地面高度修正 |
| `e04446f` | 相机挂载审计 |

证据：`task_b/results/{oracle_grasp_probe,oracle_delivery,stance_envelope,camera_mount_audit}.json`

## 4. 已确证的约束（规划时必须遵守）

1. **目标就是那个桶。** `TARGET_CENTER (-3,-10)` 就是 `terrain.py` 建的桶（`trash_bin_x=7`）。
   奖励半径 **1.0 = 桶壁外沿半径**（`bin_diameter=2.0`），壁高 0.50（顶面 z=0.55）。
   - **推不进去**：顶在壁外侧时 root 在 r ≈ 1.02 > 1.0。
   - **举着也不算**：奖励还要求 root z ≤ 0.5。必须松爪让它**掉进去**。
2. **底盘完全不能差速转向。** 命令 ±1.75 rad/s 只出 0.05 rad/s，yaw 一动不动（轮子失速）。
   M1 能转是因为用了 `--wheel_action_gain 8`。**假设只能直行 + 极慢原地转。**
3. **机身不能贴到桶。** `base_link` 前角伸出约 0.49 m，距桶心 1.484 m 就碰壁
   （曾以 3.16 N 触发 `illegal_contact` 终止）。接近停在 **1.58 m**，靠已抬高的物体送进去。
4. **地面在 z = +0.0449**，不是 0。所有基于 z=0 的物体高度结论都偏低 4.5 cm。
5. **相机挂载模型是错的**（见第 6 节）。
6. 物体静置高度（实测）：糖盒 root z 0.0902、芥末瓶 0.1402、香蕉 0.0601；
   顶部 ≈ root + 0.0464 / 0.0957 / 0.0193。
7. 物体朝向只有 **3 种固定值**（`SUGAR_QUAT` / `OTHER_QUAT`），不是任意角——
   糖盒窄边 45 mm 沿**世界 x**，芥末/香蕉窄边 58/74 mm 沿**世界 y**。钳口轴必须对准窄边。

## 5. 已证伪 / 别重走

- 「可达 0.053 m」（128 环境构建）无效；单环境可信值 = 指尖最多到机身原点下方 **0.2937 m**。
- 「物体掉穿地板是 teleport 造成的」错——真因是 128 环境全塌到同一原点。
- 无渲染跑之所以一直空转，是 Isaac Lab 上游 bug：`create_new_stage` 不在 `_SIM_APP_CFG_TYPES` 里，
  被配置过滤器丢掉 → `SimulationApp._wait_for_viewport` 永久循环。已修。
- 不要试图泊车时同时对准「物体」和「桶」——钳口对准与航向对准会互相打架。

## 6. 建议的下一步（按顺序）

### 6.1 相机挂载标定（低风险，gate 住抓取精度）

- **问题**：头相机实测机身偏移 `(0.636,-0.042,0.380)` vs 模型 `(0.422,0.025,0.062)`（差 **0.39 m**）；
  腕相机差约 **0.05 m**。M1 的视觉定位误差是 **0.033 m**（水平 0.008、竖直 −0.033），
  量级与腕相机吻合 → **腕相机挂载很可能就是定位误差的来源**。
- **做法**：写一个模式让手臂扫过一组关节位姿，每个位姿停够 ≥25 步（相机周期 0.5 s），
  只保留手臂静止的采样点（相邻步关节变化 < 1e-5）：
  - 头相机：`offset = mean(R_base^T · (cam_w − base_w))`
  - 腕相机：`offset = mean(fk(q)^{-1} · (R_base^T · (cam_w − base_w))`
  然后更新 `task_b/arm_kinematics.py` 的 `head_camera_transform` / `ee_camera_transform`，
  再用真值重测定位误差。
- **测量坑**：机器人在出生时会**弹跳**（地形 restitution=1.0），相机 10 Hz 更新，
  弹跳期间采样最多滞后 5 步，反推偏移会摆动 **0.1 m**。只在稳定段测（离散度可到 0.0008 m）。
- 评测器已经会记录相机真实位姿（`evaluate.py::camera_state`，存进 `telemetry.npz` 的
  `camera_{head,ee}_{pos,quat}`）。

### 6.2 橙色桶检测器

`bin_color = [255,128,0]`，直径 2 m，平地上很好找。用于闭环转向与最终投放对准。

### 6.3 架构：**抓完再原地转身对准桶**

```
接近物体（只需面向物体，钳口自然对准其窄边）
  → 合爪（预压 25 mm）→ 抬升
  → 原地转身，用橙色桶闭环对准
  → 直行 → 抬过沿 → 松爪
```

理由：转身只需几十度且可视觉闭环，能把「钳口对准」（物体窄边）与「航向对准」（桶）
解耦。若强行在泊车时同时满足两者，机身姿态会被锁死，抓取会失败。

### 6.4 M4（18 件）

放在最后，不要提前做。

## 7. 可直接复用的模块（均已 CPU 审计通过）

| 文件 | 内容 |
| --- | --- |
| `task_b/leg_kinematics.py` | 腿部 FK + 「轮子原地、机身垂直升降」求解器（从 USD 读的关节系，可正负双向） |
| `task_b/stance_descend.py` | 站姿下降策略（`--mode stance_descend`） |
| `task_b/grasp_probe.py` | oracle 探针全序列：伸臂/合爪/抬升/搬运/抬过沿/松爪（`--mode grasp_probe`） |
| `task_b/first_reach.py` | 视觉路径；已加 `--reach_grasp`、两段式垂直伸臂、解出的几何下降量 |
| `task_b/evaluate.py` | `--camera_free`、`--mode stance_descend`、`--mode grasp_probe`、`camera_state()` |

几个已固化的关键设计（别改回去）：

- **伸臂必须两段式**：指尖比钳口中点多伸 21 mm，直着压向抓取位姿会把它 rake 倒。
  先到抓取点正上方，再垂直下降。
- **手臂位置伺服会下垂**（stiffness 80）：实测 joint2 差 0.084 rad ≈ 钳口差 5 cm。
  必须用积分补偿，且伸臂相位要**按到位收敛**结束，不能按时间。
- **合爪要预压越过行程终点 25 mm**：位置伺服的力量正比于误差，刚好停在终点会握力不足（≈4 N），
  0.5 kg 物体会滑脱。
- **投放要轻且拆步**：抬升+内移合成一个 IK 目标会超臂展；一次性抬 0.40 m 速度 0.57 m/s 会把物体甩掉。
  现在是 2.5 s 平滑插值 → 直行内移 → 松爪。

## 8. 环境纪律（都是踩过的坑）

- **GPU 串行**，一次只跑一个 Isaac 进程。
- 不要碰 `/home/lybm/ATEC2026_Simulation_Challenge` 里**未提交的文件**（用户自己的在途工作）。
- 任何 API 调用前**断言形状与单位**；动手前先验证前提（有一轮 8 项预检抓出 6 个 bug）。
- 真值只能进**探针**，且必须在结果文件里明确标注 `oracle`——探针结论不是策略成绩。
- 修了工具/模型就**重跑并把新旧数字并列**，不要加免责声明。
- 出结论前用**两种独立方法交叉验证**：本轮多次靠这个抓到自己的错误
  （地面高度、弧度 vs 米、网格增量被当成绝对量）。
- 输出到**新目录**，绝不覆盖旧结果。

## 9. 协作约定

- 提交署 `LYHrmer <1507229006@qq.com>`（local config 已设好）。
- 推送走 **SSH**：`git push git@github.com:LYHrmer/atec-robotics-projects.git main`
  （HTTPS 在这台机器上拿不到凭据）。
- 提交信息用英文，结尾加 `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`。
