# ATEC2026 Task B 冲刺执行手册（给 Codex / Claude 直接读取）

> 角色定位：你是代码协作 Agent。你的任务不是自由发挥写一个“高分方案”，而是在现有仓库内按阶段推进 Task B，产出可验证、可回滚、可解释、可提交的代码与实验记录。
>
> 任务范围：只做 **Task B**。不要安排 Task D、不要继续优化 Task E、不要改动与 Task B 无关的训练/提交产物。
>
> 当前原则：**人类负责阶段闸门和线上提交决策；AI 负责代码审查、最小 diff、日志分析、参数搜索、候选评测和文档归档。**

---

## 0. 最高优先级约束

### 0.1 不得违反的硬约束

1. **只处理 Task B**，不要主动改 Task A / Task D / Task E。
2. 主入口必须保持为 `demo/solution.py` 中的 `AlgSolution`。
3. 必须兼容 `predicts(obs, current_score)`；若仓库/评测链路存在 `predict()` 调用差异，则实现 `predict()` 作为薄转发。
4. B2wPiper 动作布局必须运行时验证，预期为：
   - leg: 12 维
   - wheel: 4 维
   - arm: 8 维
   - total: 24 维
5. 不允许复制公开方案或大段复刻旧 baseline 结构。
6. 不允许把长航点表作为最终方案主体；coverage path 必须程序化生成。
7. 不允许自动线上提交。
8. 不允许覆盖或删除已经验证过的 safe-final。
9. 不允许在未通过 import smoke test 前启动长时间 Isaac 评测。
10. 若出现 `Episode done. score:0.00 time:0.00s`、import error、action 维度不一致、环境无法启动，立即停止后续实验，写入 FAILED 状态文件，等待人工决策。

### 0.2 输出形式约束

执行任何代码改动时：

- 默认只输出 **unified diff**，不要整文件重写。
- 每个阶段必须输出：
  - 修改文件列表
  - 验证命令
  - 日志路径
  - 状态文件路径
  - 结果摘要
  - 是否通过进入下一阶段的闸门
- 长跑实验必须写：
  - `/tmp/taskb_<stage>_status`
  - `/tmp/taskb_<stage>_latest`
  - `experiments/taskb_*/results.csv`
  - `experiments/taskb_*/leaderboard.md` 或 `ranking.md`

---

## 1. 总体冲刺策略

### 1.1 最终目标

在剩余约 3.5 天内，产出两个候选：

```text
safe-final:
  稳定版本，优先拿 grasped_objects 分，必须可提交。

risky-final:
  在 safe-final 基础上追加 bucket-push 增益，只在多次评测明显高于 safe-final 时才作为候选。
```

### 1.2 技术主线

不要把端到端 RL / ACT 全动作模仿学习作为主线。主线是：

```text
B2wPiper
→ 程序化覆盖规划
→ 轮式底盘航点跟踪
→ 低位机械臂周期扫描
→ current_score 反馈
→ 局部重扫
→ 卡住恢复
→ 参数搜索
→ 候选版本重复评测
→ 代码审查与提交冻结
```

### 1.3 背景判断

Task B 的高性价比路线不是一开始完整抓取/放置，而是先利用末端执行器覆盖物体附近区域，尽量稳定触发 grasped_objects 分。只有 safe 版本已经稳定后，才尝试 bucket push 冲 objects_in_circle 分。

---

## 2. 阶段总览

禁止按“第几天”执行。必须按阶段和闸门执行。

```text
Stage A: 链路清零
Stage B: safe 策略模块化
Stage C: 参数搜索主线
Stage D: feedback / recovery 增强
Stage E: 轻量 IL mode selector 后台分支
Stage F: bucket push / CEM 后台分支
Stage G: 候选版本竞赛评测
Stage H: 原创性审查与提交冻结
```

阶段推进规则：

```text
A 未通过 → 禁止进入 B/C/D/E/F/G/H
B 未通过 → 禁止参数搜索
C 未产生 safe 候选 → 禁止 risky push 进入提交候选
G 未完成重复评测 → 禁止线上提交
H 未通过 → 禁止线上提交
```

---

## 3. Stage A：链路清零

### 3.1 目标

排除所有假算法问题：

- `demo/solution.py` 指向错误
- `AlgSolution` import 失败
- `predict` / `predicts` 接口不兼容
- Task B 环境名错误
- B2wPiper action 维度错误
- wheel action 未生效
- episode 0 秒结束
- 远端 headless/camera 参数错误

### 3.2 禁止事项

Stage A 不写新算法，不重构策略，不调参数。

### 3.3 必跑命令

```bash
cd /root/ATEC2026_Simulation_Challenge

git status --short

PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python -c \
'from demo.solution import AlgSolution; s=AlgSolution(); print("IMPORT_OK", type(s).__name__)'
```

如果存在环境列表脚本：

```bash
PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python scripts/list_envs.py | grep ATEC-TaskB-B2wPiper
```

短测：

```bash
PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python scripts/play_atec_task.py \
  --task ATEC-TaskB-B2wPiper \
  --enable_cameras \
  --headless \
  --debug
```

若仓库已有脚本：

```bash
bash scripts/eval_task_b_baseline.sh
```

### 3.4 必须写状态

```bash
echo "RUNNING stage=A action=link_check" > /tmp/taskb_stageA_status
echo "/tmp/taskb_stageA_<timestamp>" > /tmp/taskb_stageA_latest
```

### 3.5 通过闸门

必须全部满足：

```text
[ ] IMPORT_OK
[ ] TaskB 环境可启动
[ ] episode 不是 0 秒立即结束
[ ] action_dim 运行时确认是 24
[ ] wheel action 导致 base pose 有实际变化
[ ] 日志中能看到持续 step
```

### 3.6 失败处理

如果出现：

```text
Episode done. score:0.00 time:0.00s
```

立即停止，不要长跑。只检查：

```text
1. demo/solution.py 是否仍指向 Task E 或其他 wrapper
2. AlgSolution 是否有 predicts(obs, current_score)
3. 是否需要兼容 predict(obs, current_score)
4. action 长度是否为 24
5. wheel slice 是否为 [12:16]
6. 是否给了非法 action / NaN / 超范围
7. 是否因 action spec 或 reset condition 立即终止
```

---

## 4. Stage B：safe 策略模块化

### 4.1 目标

把现有 Task B baseline 改造成原创、可解释、可调参、可记录的模块化 safe policy。

### 4.2 推荐目录结构

```text
demo/
├── solution.py
├── solution_task_b_entry.py
└── taskb_policy/
    ├── __init__.py
    ├── config.py
    ├── action_layout.py
    ├── state_estimator.py
    ├── coverage_planner.py
    ├── chassis_controller.py
    ├── arm_sweep_controller.py
    ├── score_tracker.py
    ├── stuck_recovery.py
    ├── bucket_push.py
    └── policy.py
```

### 4.3 文件职责

| 文件 | 职责 | 强制要求 |
|---|---|---|
| `solution.py` | 极薄入口 | 不写业务主体 |
| `solution_task_b_entry.py` | `AlgSolution` 包装 | 兼容 `predicts` 和 `predict` |
| `config.py` | 参数集中管理 | 不要散落 magic number |
| `action_layout.py` | 24 维 action 索引 | 首次 step 打印/断言 |
| `state_estimator.py` | dead-reckoning / pose tracking | 保存短窗位移用于 stuck |
| `coverage_planner.py` | 程序化覆盖路径 | 禁止长硬编码航点表 |
| `chassis_controller.py` | wheel cmd | 只管底盘 |
| `arm_sweep_controller.py` | arm scan pose | 只管机械臂 |
| `score_tracker.py` | current_score 反馈 | 记录 score event |
| `stuck_recovery.py` | 卡住检测恢复 | 防止无限 recovery |
| `bucket_push.py` | risky 增益 | feature flag 默认关闭 |
| `policy.py` | orchestration | 不直接塞满所有逻辑 |

### 4.4 ActionLayout 必须实现

```python
class ActionLayout:
    LEG = slice(0, 12)
    WHEEL = slice(12, 16)
    ARM = slice(16, 24)
    DIM = 24
```

要求：

```text
- 第一次 predicts 调用时 assert len(action) == 24
- 如果 obs 推断出的维度与 24 冲突，记录 ERROR 并返回 safe zero action
- wheel cmd、arm cmd、leg cmd 必须分别写入 slice，不允许散写 index
```

### 4.5 CoveragePlanner 要求

禁止最终方案主体使用固定长航点表。必须使用函数生成：

```python
generate_boustrophedon(
    xmin, xmax,
    ymin, ymax,
    strip_spacing,
    margin,
    direction="x_first" or "y_first",
)
```

推荐参数初值：

```text
xmin = -15
xmax = -5
ymin = -15
ymax = -5
strip_spacing = 1.6
margin = 0.4
waypoint_reached_radius = 1.4
```

调参范围：

```text
strip_spacing: 1.2 / 1.4 / 1.6 / 1.8
waypoint_reached_radius: 1.2 / 1.4 / 1.6
margin: 0.2 / 0.4 / 0.6
```

### 4.6 ChassisController 要求

控制逻辑必须简单、可解释、可调参：

```text
heading_error 大 → 原地转向
heading_error 中 → 边走边转
heading_error 小 → 前进
接近 waypoint → 降速
卡住 → 交给 StuckRecovery
```

推荐参数初值：

```text
wheel_fwd_cmd = 0.16
wheel_turn_cmd = 0.18
yaw_turn_only_threshold = 0.28
yaw_blend_threshold = 0.15
waypoint_slow_radius = 1.2
```

### 4.7 ArmSweepController 要求

safe 主线先不做复杂抓取。机械臂用于扩大 EE 覆盖范围。

至少三种模式：

```text
front_scan
left_bias_scan
right_bias_scan
```

建议：

```text
- 使用当前 baseline 中能稳定运行的 arm pose 作为 arm_low_pose
- 在少数关节上叠加小幅周期 sweep
- 不要引入视觉检测
- 不要让 arm sweep 导致机器人立即 reset
```

推荐参数：

```text
sweep_period = 3.2
sweep_amplitude = 0.10 ~ 0.16
bias_switch_steps = 120
```

### 4.8 Stage B 通过闸门

```text
[ ] import smoke 通过
[ ] TaskB 短测可启动
[ ] action_dim=24
[ ] coverage path 是程序化生成
[ ] config.py 集中管理参数
[ ] safe policy 能持续 step
[ ] 没有新增外部依赖
```

---

## 5. Stage C：参数搜索主线

### 5.1 目标

这是真正的“训练环节”。训练对象不是神经网络，而是启发式系统参数。

### 5.2 禁止事项

```text
- 不要端到端 PPO
- 不要 ACT 直接学 24 维 action
- 不要 RGB-D 视觉训练
- 不要同时跑多个 Isaac 评测抢 GPU
- 不要用线上提交机会做参数搜索
```

### 5.3 参数搜索目录

```text
experiments/taskb_param_search/
├── configs/
│   ├── B001.json
│   ├── B002.json
│   └── ...
├── logs/
│   ├── B001/
│   ├── B002/
│   └── ...
├── results.csv
└── leaderboard.md
```

### 5.4 results.csv 字段

```csv
run_id,config_id,candidate,score,max_score,time_s,first_score_step,score_events,stuck_count,recovery_count,termination,notes
```

### 5.5 搜索策略

分三轮：

```text
C1 粗搜索：
  每组短测，淘汰 0 分、0 秒退出、明显卡死参数。

C2 中搜索：
  保留前 30% 参数组，跑更长评测。

C3 复测：
  前 3~5 个候选重复评测，按平均分、最低分、方差排序。
```

优先搜索顺序：

```text
1. strip_spacing
2. wheel_fwd_cmd
3. yaw_turn_only_threshold
4. sweep_amplitude
5. local_rescan_steps
6. no_score_patience_steps
7. stuck_window_steps
```

每轮最多同时改变 2~3 类参数。不要一次扫所有参数。

### 5.6 需要实现的脚本

```text
scripts/taskb/run_param_search.sh
scripts/taskb/parse_taskb_log.py
scripts/taskb/make_leaderboard.py
```

如果时间不足，至少实现 `run_param_search.sh` 和一个简单的 `grep/awk` 版结果汇总。

### 5.7 Stage C 通过闸门

```text
[ ] 至少完成 8 组参数短测
[ ] results.csv 存在
[ ] leaderboard.md 存在
[ ] 选出至少 1 个 safe candidate
[ ] 该 candidate 不是单次偶然高分
```

---

## 6. Stage D：score feedback + stuck recovery

### 6.1 ScoreTracker

维护状态：

```python
last_score
last_score_step
score_events
steps_since_score
region_success_counter
```

触发逻辑：

```text
if current_score > last_score:
    记录 score event
    进入 local_rescan
    当前 arm mode 暂时保持
    当前区域优先补扫
else:
    steps_since_score += 1
```

参数初值：

```text
local_rescan_steps = 80 ~ 140
no_score_patience_steps = 180 ~ 260
rescan_spacing_gain = 0.6
```

### 6.2 StuckRecovery

检测信号：

```text
- 最近 N 步位移小
- waypoint 距离长期不下降
- current_score 长期不变
```

恢复动作：

```text
1. backoff: 后退短时间
2. turn: 原地转小角度
3. switch arm mode: front → left → right
4. skip waypoint: 跳过当前 waypoint
```

参数初值：

```text
stuck_window_steps = 120
min_progress_in_window = 0.35
recovery_backoff_steps = 20 ~ 35
recovery_turn_steps = 25 ~ 40
max_recovery_per_waypoint = 2
```

### 6.3 必须记录的日志

每次触发以下事件都要打印结构化日志：

```text
[TASKB][SCORE] step=... old=... new=... mode=... waypoint=...
[TASKB][MODE] step=... from=... to=... reason=...
[TASKB][STUCK] step=... window_progress=... wp_dist=... no_score=...
[TASKB][RECOVERY] step=... action=backoff/turn/skip count=...
```

### 6.4 Stage D 通过闸门

```text
[ ] 日志中能看到 SCORE event
[ ] 日志中能看到 MODE switch
[ ] stuck 时 recovery 能退出，不会无限循环
[ ] 加入 D 后分数不低于 Stage C 最佳 safe 版本
```

---

## 7. Stage E：轻量 IL mode selector 后台分支

### 7.1 定位

可选分支，不得影响 safe 主线。

不训练 24 维 action，只训练“模式选择器”。

### 7.2 输入特征

```text
current_score
score_delta
steps_since_score
waypoint_id
heading_error
distance_to_waypoint
stuck_flag
current_arm_mode
elapsed_steps
last_score_region
```

### 7.3 输出模式

```text
KEEP_CURRENT
SWITCH_LEFT_SCAN
SWITCH_RIGHT_SCAN
LOCAL_RESCAN
SKIP_WAYPOINT
RECOVERY
ENTER_PUSH_PHASE
```

### 7.4 模型建议

优先：

```text
Decision Tree
Rule distillation
Small MLP only if absolutely necessary
```

不建议：

```text
RGB-D policy
ACT full-action policy
PPO full TaskB
```

### 7.5 验收标准

IL 分支只有满足以下条件才进入候选：

```text
[ ] 至少 3 次重复评测
[ ] 平均分 > safe-final 平均分 * 1.10
[ ] 最低分不低于 safe-final 最低分
[ ] 没有新增大依赖
[ ] feature flag 可关闭
```

否则归档为实验，不提交。

---

## 8. Stage F：bucket push / CEM 后台分支

### 8.1 定位

只做 risky 增益，不影响 safe。

### 8.2 优先顺序

```text
F1: push 参数搜索
F2: CEM 黑盒优化 push 参数
F3: 局部 RL，仅在 GPU 空闲且 F1/F2 无进展时试
```

### 8.3 禁止事项

```text
- 不训练完整 Task B
- 不输出完整 24 维端到端 action
- 不让 push phase 破坏 safe coverage
- 不在 risky 未复测时用于首提交
```

### 8.4 push 参数

```text
enter_push_score_threshold = 10 ~ 14
bucket_near_radius = 2.5 ~ 3.5
push_phase_duration_steps = 180 ~ 260
push_speed_scale = 0.5 ~ 0.8
push_lateral_sweep_amp = 0.05 ~ 0.15
```

### 8.5 必须记录

```csv
candidate,score_before_push,score_after_push,push_delta,failed_reason
```

### 8.6 验收标准

```text
[ ] safe 已稳定
[ ] push 后平均分提高
[ ] push 不显著降低 grasp 分
[ ] 多次评测不是偶然高分
[ ] 可一键关闭
```

---

## 9. Stage G：候选版本竞赛评测

### 9.1 候选池

最多保留 4 个候选：

```text
safe_v1: coverage + arm sweep
safe_v2: safe_v1 + score feedback + recovery
safe_il: safe_v2 + mode selector
risky_push: safe_v2 + bucket push
```

### 9.2 评测结果表

```csv
candidate,run_id,score,time_s,first_score_step,max_score,stuck_count,recovery_count,giveup,termination,notes
```

### 9.3 排序指标

优先级从高到低：

```text
1. import / 启动稳定性
2. 最低分
3. 平均分
4. 方差
5. 最高分
6. 代码风险
7. 提交目录风险
```

不要按单次最高分排序。

### 9.4 决策规则

```text
if safe_v2 平均分 >= safe_v1 and 最低分 >= safe_v1:
    safe_v1 淘汰

if safe_il 平均分 < safe_v2 * 1.10:
    safe_il 不进入提交候选

if risky_push 最低分明显低于 safe_v2:
    risky_push 不能作为首提交

first_submit = 最稳定版本
second_submit_if_needed = 多次验证显著更高版本
do_not_submit = 任何 import/action/打包风险未清零的版本
```

---

## 10. Stage H：原创性审查与提交冻结

### 10.1 只允许做

```text
- import smoke test
- 打包目录检查
- 反抄袭风险审查
- 注释压缩
- 文档补齐
- 低风险参数回滚
```

### 10.2 禁止做

```text
- 新增算法
- 新增依赖
- 重写 solution.py
- 改 action layout
- 改入口方法名
- 未经评测替换 final candidate
```

### 10.3 原创性检查清单

写入：

```text
docs/taskb_originality_checklist.md
```

内容模板：

```markdown
# Task B 原创性与代码审查清单

## 模块结构

- [ ] `solution.py` 是薄入口
- [ ] 核心逻辑在 `taskb_policy/`
- [ ] planner / chassis / arm / score / recovery 分离
- [ ] 参数集中在 `config.py`

## 与旧 baseline 的差异

| 项目 | 旧状态 | 新状态 | 独立优化说明 |
|---|---|---|---|
| 路径 | 固定航点 | 程序生成 | 可调覆盖参数 |
| 机械臂 | 固定姿态 | 周期扫描 | 扩大 EE 覆盖 |
| 得分反馈 | 无 | current_score local rescan | 闭环调度 |
| 卡住处理 | 无/弱 | stuck recovery | 鲁棒性增强 |
| 推桶 | 无 | feature flag push | risky 增益 |

## 实验记录

- [ ] 至少 5 条实验记录
- [ ] 每条记录有假设、改动、结果、结论
- [ ] 有 safe/risky 对比
- [ ] 有失败实验归档
```

### 10.4 提交说明模板

写入：

```text
docs/taskb_final_submit_notes.md
```

模板：

```markdown
# Task B Final Submit Notes

本方案面向 ATEC2026 Task B，使用 B2wPiper。

## 方案组成

1. 程序化 boustrophedon coverage planner
2. 基于 heading error 的轮式底盘控制
3. 面向 grasped_objects 的低位机械臂周期扫描
4. 基于 current_score 的 local rescan
5. stuck detection and recovery
6. 可选 bucket-near push phase

## 独立优化点

- 不使用硬编码长航点表，改为程序生成覆盖路径
- 使用 current_score 做闭环策略切换
- 加入 arm sweep 偏置模式
- 加入 stuck recovery
- risky 版本加入 bucket push feature flag

## 工程约束

- 无新增大依赖
- 支持离线运行
- `AlgSolution` 入口保持不变
- safe 与 risky 可回滚
```

---

## 11. 线上提交前检查

### 11.1 打包目录

示例：

```bash
mkdir -p submit/taskb_safe
cp demo/solution.py submit/taskb_safe/solution.py
cp -r demo/taskb_policy submit/taskb_safe/taskb_policy
cp demo/solution_task_b_entry.py submit/taskb_safe/solution_task_b_entry.py
cp demo/requirements.txt submit/taskb_safe/requirements.txt 2>/dev/null || true
```

注意：

```text
- 不要上传 run.sh
- 不要上传 server.py
- 不要上传无关训练日志
- 不要上传旧 Task E 权重
- 不要上传 stale submit zip
```

### 11.2 打包后 import

```bash
cd submit/taskb_safe
PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python -c \
'from solution import AlgSolution; s=AlgSolution(); print("PKG_IMPORT_OK", type(s).__name__)'
```

### 11.3 最终检查

```text
[ ] safe commit hash 已记录
[ ] risky commit hash 已记录，若存在
[ ] results.csv 已归档
[ ] ranking.md 已归档
[ ] originality_checklist 已完成
[ ] final_submit_notes 已完成
[ ] import smoke 通过
[ ] 打包目录 import 通过
[ ] 没有 run.sh / server.py
[ ] 没有不必要大文件
[ ] 人工确认后才允许线上提交
```

---

## 12. Codex / Claude 执行 Prompt

### 12.1 总控 Prompt

```text
你现在接手 ATEC2026 Task B 冲刺。必须按阶段执行，不得按“第几天”松散安排。

总目标：
在 3.5 天内产出一个可提交的 Task B safe-final，并可选产出 risky-push 候选。最终代码必须体现独立优化，避免与公开方案或旧 baseline 高度相似。

硬约束：
1. 我不负责 Task D，你不要安排 Task D。
2. demo/solution.py / AlgSolution / predicts(obs,current_score) 是主入口，不得破坏。
3. 若需要兼容 predict(obs,current_score)，只能作为薄转发加入。
4. B2wPiper 预期 action 为 24 维：12 leg + 4 wheel + 8 arm。必须运行时验证。
5. 不得复制公开方案。
6. 不得输出整份重写代码，只输出最小 diff。
7. 不得使用长硬编码航点表作为最终方案主体。
8. 所有实验必须有状态文件、日志目录、results.csv 或 markdown 报告。
9. 不得自动线上提交。
10. 不得删除或覆盖 safe-final。
11. 若出现 import error、0 秒 episode、action 维度不一致，立即停止后续实验并写 FAILED 状态。

阶段：
A：入口、action、评测链路清零。
B：safe 启发式系统模块化重构。
C：自动参数搜索。
D：score feedback 与 stuck recovery。
E：后台轻量 IL mode selector，只学模式，不学 24 维 action。
F：后台 bucket push 参数搜索 / CEM / 可选局部 RL。
G：候选版本重复评测排序。
H：代码审查、原创性审查、提交冻结。

每个阶段必须输出：
- 状态文件路径；
- 日志目录；
- 修改文件列表；
- 验证命令；
- 结果摘要；
- 是否通过进入下一阶段的闸门。

现在先执行 Stage A，只做链路清零，不写新算法。
```

### 12.2 代码审查 Prompt

```text
请审查当前 Task B 代码，只输出“问题清单 + 最小修改建议”，不要重写整份代码。

重点：
1. 是否保持 demo/solution.py / AlgSolution / predicts(obs,current_score) 入口；
2. 是否兼容 predict(obs,current_score)；
3. B2wPiper action 是否严格为 24 维；
4. wheel slice 是否为 12:16；
5. 是否存在导致 Episode done. score:0.00 time:0.00s 的风险；
6. 是否存在和旧 baseline / 公开方案高度相似的结构；
7. 是否缺少实验记录或原创性证据。

输出格式：
A. 严重问题
B. 中等问题
C. 建议但非必须
D. 最小修复 diff 建议
E. 验证命令
```

### 12.3 日志诊断 Prompt

```text
下面是 Task B 评测日志。请只回答：

A. 失败类型：
- 入口错误
- action 维度错误
- wheel action 未生效
- 即时终止
- 导航失败
- EE 未覆盖物体
- score feedback 失效
- stuck recovery 失效
- 超时

B. 支持判断的日志证据；
C. 下一轮只允许改一个地方时，最值得改什么；
D. 对应验证命令；
E. 是否允许继续长跑。

不要给大改方案。
```

### 12.4 反抄袭审查 Prompt

```text
请从代码审查和原创性角度审查当前 Task B 方案，不要生成新代码。

重点：
1. 是否只是旧 baseline 改变量名；
2. 是否存在长硬编码航点表；
3. 是否缺少 programmatic coverage；
4. 是否缺少 score-triggered local rescan；
5. 是否缺少 stuck recovery；
6. 是否缺少实验记录；
7. 是否有公开方案相似风险；
8. 哪些地方需要最小重构以体现独立优化。

输出：
- 风险等级：低 / 中 / 高
- 高风险位置
- 原因
- 最小重构建议
- 还缺哪些独立性证据
```

---

## 13. 低 token 远端协作协议

### 13.1 状态查询模板

```bash
ssh -p <PORT> root@<HOST> '
cat /tmp/taskb_stage_status 2>/dev/null || true
LOG=$(cat /tmp/taskb_stage_latest 2>/dev/null || true)
echo "LOG=$LOG"
[ -n "$LOG" ] && tail -40 "$LOG/_stages.log" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h /root
'
```

### 13.2 日志原则

对话中不要粘贴长日志。只摘：

```text
- 最后 40 行
- ERROR / WARN
- Episode done
- score event
- action_dim
- termination reason
- status file
```

### 13.3 无人值守规则

无人值守脚本只能做：

```text
- 参数搜索
- 单候选重复评测
- 日志解析
- leaderboard 更新
```

无人值守脚本不能做：

```text
- 自动大改代码
- 自动切换 final candidate
- 自动删除 safe-final
- 自动线上提交
- 自动引入依赖
```

---

## 14. 最终提交策略

首提交：

```text
safe_v2:
  programmatic coverage
  + chassis control
  + arm sweep
  + score feedback
  + stuck recovery
  + 参数搜索最优配置
```

第二候选：

```text
risky_push:
  safe_v2 + bucket push
```

只有满足以下条件才提交 risky：

```text
[ ] risky 平均分高于 safe
[ ] risky 最低分不明显低于 safe
[ ] risky 没有新增 import/action/打包风险
[ ] risky 可通过 feature flag 回退到 safe
```

不要提交：

```text
- 单次最高分但方差大的版本
- 未完成 import smoke 的版本
- 有 action 维度疑点的版本
- 有公开方案相似风险且无实验记录的版本
- 任何最后一刻大改版本
```

---

## 15. 当前 Agent 的下一步

现在必须执行：

```text
Stage A: 链路清零
```

不要跳到算法优化。

请按以下格式汇报：

```markdown
# Stage A Report

## Status
PASS / FAILED / BLOCKED

## Commands Run
...

## Files Inspected
...

## Findings
...

## Action Layout Check
...

## Episode Startup Check
...

## Logs
status_file:
latest_log_dir:

## Gate Decision
Can proceed to Stage B: yes/no

## Minimal Next Diff
若需要，只给 unified diff；若不需要，写 none。
```
