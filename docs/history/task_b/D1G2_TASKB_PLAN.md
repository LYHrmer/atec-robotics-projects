# D1+G2 做 Task B 的方案(学 RL 实战项目)

目标:用**本末 D1+G2 机器人**,在 **DDT_Lab**(NP3O locomotion 框架)里搭出并训练一个
**Task B 投递任务**(把物体送进桶的圆圈),过程中学强化学习;完成并验证后**上传到
`github.com/LYHrmer/atec-robotics-projects`**。

## 事实基线(已核实)
- DDT_Lab 是**运动训练库**:D1、Tita、D1G2 都有注册好的 velocity-tracking 任务
  (`DDT-Velocity-Flat-D1-v0`、`DDT-Velocity-Rough-D1G2-v0` 等),算法 NP3O(约束 PPO)。
- **它没有 Task B**:没有桶、物体、投递奖励、抓取。Task B 要**自己搭**。
- `ddt_lab` 已安装可 import;GPU 8GB;训练在这台机器上跑过(Task A)。
- G2 是 7 自由度臂;Task A 里被当负载举着,只学 16 维底座残差。

## 阶段计划(每阶段:学什么 + 交付物)

### 阶段 0 — 地基:确认能训 + 读懂训练循环 【进行中】
- 确认 D1 locomotion 在本机训得起来(smoke 跑出 Learning iteration)。
- 一起读 `locomotion_env_cfg.py`:观测 / 动作 / 奖励 / 终止 / NP3O runner。
- **学**:一个 RL 任务由哪些部件组成、PPO 训练循环长什么样。
- **交付**:能走路的 D1 基线策略;你看懂训练日志和曲线。

### 阶段 1 — 定点到达(Task B 的"移动"一半)
- 把"跟随速度指令"改成"走到指定坐标(桶的位置)"。
- 新增:目标点观测、距离奖励、到达终止。
- **学**:reward shaping、goal-conditioned RL。
- **交付**:D1 从出生点自主走到指定点。

### 阶段 2 — 加桶+物体+投递奖励(Task B 核心)
- 复用 ATEC 的桶 USD 和物体 USD(不从零建资产)。
- 物体先"随车携带"(焊在夹爪/托盘),**跳过抓取**先跑通投递本质。
- 奖励 = 物体位置进入桶圆圈;成功判定 = object_in_circle。
- **学**:多目标/稠密-稀疏奖励、成功与终止设计。
- **交付**:D1+G2 带物体走过去、送进圈 —— 可展示、可上传的 Task B。

### 阶段 3 —(进阶,可选)抓取
- G2 臂主动抓起物体再运;或脚本抓取+学习运输的折中。
- 视阶段 2 完成度和时间决定。

### 阶段 4 — 打包上传
- 清理、写 README/文档、快照;推到 `LYHrmer/atec-robotics-projects`。
- **只在 Task B 真正跑通并独立验证后**。

## 纪律
- GPU 串行:一次只训一个。
- 每个训练回合几十分钟到几小时;到阶段 2 是**跨会话的多次训练**,不是一次搞定。
- 重点在阶段 1-2 的任务设计:我边写边讲每个 obs/reward/termination 为什么这么设。
- 不谎报进度:训不出来就说训不出来,附日志和曲线。

---

## 【已确认】平台可训(2026-09-15)

三个训练入口只有一个能用:
- ATEC `scripts/rsl_rl/train.py`(hydra)→ 挂在建环境
- DDT_Lab `scripts/np3o/train.py`(原生)→ 挂在建环境(同一点)
- **`tools/train_d1g2_taska_residual.py`(残差机制)→ 今天实测可训** ✅

实测(128 envs, 5 iter):`D1G2_TRAIN_READY` → `Learning iteration` → `training_batch_finished`,
1.12s/iter,GPU 3766/8151 MiB,RAM 24GB free。**这就是训 Task B 要用的机制。**

启动命令模板:
```
cd /home/lybm/ATEC2026_Simulation_Challenge
python tools/train_d1g2_taska_residual.py --headless \
  --assets_root /home/lybm/ATEC_D1G2_workspace2_20260908/workspace2/DDT_Lab \
  --policy <assets>/ddt_ros2_control/controller/rl_controller/config/d1/flat_lab.onnx \
  --num_envs 128 --iterations N --output <dir>
```

## 【下一会话执行】Task B 环境的设计蓝图

要仿照 `tools/d1g2_taska_train_env.py`(435 行,`build_d1g2_taska_train_cfg`)新建
`tools/d1g2_taskb_train_env.py`,改动如下:

**MVP 定义(先跳过抓取)**:物体"随车携带"(生成在机器人身上/焊在夹爪),
任务 = 带着物体走进桶的圆圈。本质上是 **goal-conditioned 导航**,直接复用冻结的运动底座。

| 部件 | Task A(现有) | Task B(要改成) |
|---|---|---|
| Scene | 课程地形 + 机器人 | 平地 + 机器人 + **桶(静态,center (-3,-10),r=1m)** + **物体** |
| Command/Goal | 前向速度 0.35-0.85 | **桶的位置作为目标**(观测里加 robot→barrel 的相对向量) |
| Observation | proprio + 速度指令 | proprio + **robot→barrel 相对位置** (+物体相对位置) |
| Reward | track_lin_vel + 地形 | **-‖object_xy − barrel_xy‖(稠密)+ 进圈大奖励(稀疏)** |
| Termination | 摔倒/超时 | 摔倒/超时 + **object_in_circle 成功终止** |
| 冻结底座 | flat_lab.onnx | flat_lab.onnx(不变);残差学"朝桶走+投递" |

**RL 教学点**:goal-conditioning(把目标喂进观测)、dense+sparse 混合奖励、成功判定。

**资产复用**:ATEC 的桶 USD 和物体 USD 在
`/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/`(桶几何:center (-3,-10)、
外半径 1m、壁厚 .02、rim z .55;物体:006_mustard_bottle.usd)。

**下一步第一件事**:读 `d1g2_taska_train_env.py` 全文,搞清 scene/command/reward manager
怎么组装,再照着加桶+物体+投递奖励。
