# Task B 交接：从 oracle 到官方视觉策略

> 目标：把已经打通的「抓起 → 投递」从 **oracle 定位**换成**视觉**，做成官方策略成绩。
> 本文件是自包含交接，读完即可上手。

## 1. 现状一句话

M2（真实抓起）与 M3（投递）已在 oracle 定位下打通并推送，**官方 `objects_in_circle` 真的加过分**；
现在缺的只是把真值换成视觉。

> **2026-09-15 更正（相机挂载一节）**：原先"头相机差 0.39 m、腕相机差约 0.05 m、
> 腕相机挂载就是 0.033 m 定位误差来源"的结论**是错的**，已由实验推翻。真正的原因是一个
> 诊断读数的 bug：`isaaclab CameraCfg.update_latest_camera_pose` 默认 `False`，
> 此时 `sensor.data.pos_w` 返回**初始化那一刻**的相机位姿、之后永不更新。旧审计拿这个
> 冻结位姿去比已经移动过的机身，差多少就等于机身走了多少。
> 实测两个挂载模型本来就是对的（与仿真差 3e-06 / 9e-06），定位残差的真正来源是
> **检测器的深度几何**（读的是物体前表面，当成轴心用）。
> 结论与证据：`task_b/results/camera_mount_calibration.json`、
> `task_b/results/camera_mount_calibration_audit.json`（33 项 CPU 检查全绿）。
> 第 6.1 节已改写为实际做过的事。

## 2. 路径与环境

| 项目 | 值 |
| --- | --- |
| 交付物仓库 | `/tmp/atec-upload`（GitHub `LYHrmer/atec-robotics-projects`）。本地/远端是否同步以 `git log -1` 和 `git ls-remote <repo> main` 为准 —— 这里原来写死的 `e04446f` 早已过期，不要再写死哈希 |
| 官方 cfg（只读） | `/home/lybm/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/tasks/task_b/env_cfg.py` |
| Python | `/home/lybm/miniforge3/envs/isaaclab/bin/python` |
| 无渲染跑（快、稳） | 加 `--camera_free` |
| 开相机跑 | 不加 `--camera_free`（渲染已获用户批准） |
| CPU 审计（上 GPU 前先跑） | `task_b/audit_stance.py`（14 项）、`task_b/audit_grasp.py`（27 项）、`task_b/audit_camera_calibration.py`（33 项，可只跑 CPU，也可 `--fit/--gate/--visual` 指向真实 run） |

注意 `audit_stance.py` 的 `leg_frames_match_usd` 需要 Isaac Sim 的 USD 动态库在
`LD_LIBRARY_PATH` 上，否则报 `ImportError: libtf.so`：

```bash
LD_LIBRARY_PATH=/home/lybm/miniforge3/envs/isaaclab/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin \
  $ATEC_PYTHON task_b/audit_stance.py     # 14/14
```


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
| `e04446f` | 相机挂载审计 —— **结论已于 2026-09-15 推翻**，见第 1、6.1 节；该文件的数字是诊断读数 bug 的产物 |

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
5. **相机挂载模型是对的**（2026-09-15 实测更正，见第 6.1 节）。曾经的"挂载错了"是
   诊断读数 bug：`update_latest_camera_pose` 默认 False → `pos_w` 冻结在初始化时刻。
   别再用旧审计给的 0.39 m / 0.05 m 做规划。
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

### 6.1 相机挂载标定 —— 已完成，结论是"模型本来是对的"

- **做法**（已实现）：`--mode camera_calibration` 让手臂扫过 9 个保持位姿
  （第一个就是机器人自己的默认臂姿，25 步斜坡 + 140 步保持，腿轮保持默认），
  评测器逐步记录仿真自己的 `pos_w/quat_w_world`；只取"手臂与机身连续 25 步不变"的采样点
  （相机 10 Hz、策略 50 Hz，故记录位姿最多滞后 5 步，25 步静窗把滞后消掉）。
  `task_b/audit_camera_calibration.py` 闭式解出两个挂载，33 项 CPU 检查。
- **必须先修的工具 bug**：`evaluate.py` 现在把场景相机的
  `update_latest_camera_pose` 设为 `True`。默认 `False` 时 `pos_w` 是**初始化位姿**，
  旧审计的 0.39 m 就是这么来的。官方 observation 读的是 annotator 不是位姿，
  奖励/动作/终止都不受影响；每个 result.json 里都记了
  `camera_pose_reader`，source manifest 也列了这一项。
- **结果**：头挂载拟合约 `(0.421607, 0.025001, 0.061850)` + Ry(30°)，
  与 `arm_kinematics` 现有常量最大元素差 **3.1e-06**；腕挂载拟合
  `(-0.04999, 0.00000, 0.05999)` + Rz(-90°)，差 **9.1e-06**。
  跨 16 个不同臂姿的离散度 < 0.8 mm / 0.8 mrad，说明挂载确实是刚性的、
  FK 链也是对的。**所以 arm_kinematics.py 的数值一个没改**，只把验证结论写进了
  docstring。（另：`base_link` 与 articulation root 重合，实测偏移 0.0。）
- **硬门槛（真值重测）**：另跑一集（seed 7，`first_reach`，会走会转），
  用 `|M_model·M_sim⁻¹·p − p|` 在 18 个真实物体位置上量模型引入的定位误差：
  **头 0.6 mm、腕 0.6 mm（max）**，比 1 cm 门槛低约 17 倍。**过。**
- **顺带查清了 0.033 m 是什么**：同一集的真实检测输出对真值——检测器**像素**只偏
  2.6 px，但 body 系残差 47 mm。把它投影到"物体→相机"射线上：**沿线分量 +33 mm，
  离散只有 2.6 mm**，而芥末瓶半宽是 29 mm。也就是说深度读的是**前表面**，被当成轴心用了。
  横向那 31 mm 就是那 2.6 px 在 5.6 m 处的等效。两者都是检测器几何，不是挂载。
  → **给 6.2 的具体要求**：把估计从"前表面点"沿视线推到"物体轴心"（等于减掉半个宽度），
  再交给 IK；否则钳口会落在物体后面约 3 cm。


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
- **SSH 直连会间歇性挂断**（`kex_exchange_identification: Connection closed by remote host`）。
  这台机器跑 FlClash 代理（`127.0.0.1:7890`，fake-IP DNS 把 github.com 解析成 `198.18.0.15`），
  TUN 路由时好时坏。**稳的做法是显式把 SSH 穿 HTTP CONNECT 代理**：

  ```bash
  cd /tmp/atec-upload
  env GIT_SSH_COMMAND="ssh -o ConnectTimeout=30 -o ProxyCommand='nc -X connect -x 127.0.0.1:7890 %h %p'" \
    git push git@github.com:LYHrmer/atec-robotics-projects.git HEAD:main
  ```

  `nc -X connect` 就是普通的 openbsd netcat。推完用同一条 `GIT_SSH_COMMAND` 跑
  `git ls-remote <repo> main` 核对远端哈希 == 本地 HEAD（这条也会偶尔需要重试）。
  代理本身是活的，`curl -x http://127.0.0.1:7890 https://github.com` 能返回 200 可用来判断。
- 提交信息用英文，结尾加 `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`。
- **大文件**：仓库单文件上限 100 MB，Release 附件宽松得多。**原片不进 git**，作为
  Release 附件发布并把 sha256 写进转码记录；仓库只放转码后的版本
  （投递录像现在是 1080p / 19 MB，原片 355 MB 在 Release；见 `docs/VIDEO_COMPARISON.md`）。
