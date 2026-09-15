# ATEC2026 Task B Stage-A-First 执行手册 v2

> 给 Codex / Claude 直接读取。  
> 当前目标不是写高分算法，而是先定位并清除 **Task B 0.00s episode / reset 后立即终止** 的根因。  
> 在 Stage A 未通过前，禁止进入模块化重构、参数搜索、IL、RL、CEM、bucket push。

---

## 0. 当前审查结论

上一版 `TaskB_Codex_Claude_Runbook` 的大方向没有错：阶段闸门、safe/risky 分离、参数搜索、原创性审查都合理。  
但它对当前工程状态过于超前。现在真实瓶颈是：

```text
Task B episode 出现 0.00s 结束；
接口和 action 维度看起来可能已基本正确；
剩余高概率问题是 reset 后物理不稳定、wheel 命令过大、fall / illegal_contact / action spec / termination 条件触发。
```

因此本版改为 **Stage-A-First**：

```text
先证明机器人能活
→ 再证明低速 wheel 能安全移动
→ 再恢复 coverage baseline
→ 再进入 safe 策略优化
```

---

## 1. 本版相对上一版的关键修正

### 1.1 立即采纳的修正

| 修正 | 原因 |
|---|---|
| 删除 Stage E / Stage F 主流程 | 当前 Stage A 未清零，IL / RL / CEM / bucket push 都会分散注意力 |
| 暂停十文件模块化重构 | 现在最需要最小 diff，不是架构美化 |
| 增加 reset 后 settle / wheel ramp 消融 | 0.00s 很可能与 reset 后物理不稳定和 wheel 命令过大有关 |
| Stage A 不再要求 CSV / leaderboard | Stage A 只定位根因，不做参数优化 |
| 明确 zero-action survival test | 先判断“不动是否也死”，否则不能怪控制策略 |
| 明确 low-wheel 对照 high-wheel | 验证 `WHEEL_FWD_SPEED=0.6` 是否过激，而不是主观猜测 |

### 1.2 仍保留但冻结的内容

以下内容不是错误，但当前不执行：

```text
- 大规模参数搜索
- safe/risky 候选竞争
- 轻量 IL mode selector
- CEM / 局部 RL
- bucket push
- 完整 taskb_policy/ 十文件拆分
```

这些放入 Parking Lot，只有 Stage A/B 通过后再恢复。

---

## 2. 绝对硬约束

1. 只处理 **Task B**。
2. 不处理 Task D。
3. 不继续优化 Task E。
4. 不写新算法。
5. 不做模块化大重构。
6. 不做参数搜索。
7. 不做 IL / RL / ACT / CEM。
8. 不做 bucket push。
9. 不新增依赖。
10. 不自动线上提交。
11. 不删除或覆盖当前可回滚版本。
12. 只允许最小 diff。
13. 若出现 `score:0.00 time:0.00s`，立即停止长跑，写 FAILED 状态。
14. 未明确 termination reason 前，不得继续凭感觉改策略。

---

## 3. 当前唯一 Active Stage

```text
Active:
  Stage A First：0.00s 根因消融

Frozen:
  Stage B：最小 safe policy
  Stage C：小规模参数消融
  Stage D：提交审查

Parking Lot:
  IL mode selector
  RL / CEM
  bucket push
  十文件模块化
```

---

## 4. Stage A First 总目标

回答 5 个问题：

```text
Q1. demo/solution.py / AlgSolution / predicts 是否真的没问题？
Q2. action len 是否真的为 B2wPiper 需要的 24 维？
Q3. zero action / safe hold action 下，机器人是否也会 0.00s 死？
Q4. reset 后 settle N steps 是否能避免立即 fall / illegal_contact？
Q5. wheel_fwd_cmd=0.6 是否比 0.10/0.16 更容易触发立即终止？
```

只有这些问题有结论，才能进入 Stage B。

---

## 5. Stage A 状态文件规范

所有实验必须写状态文件：

```bash
echo "RUNNING stage=A substage=<A0/A1/A2/A3/A4> reason=<short>" > /tmp/taskb_stageA_status
echo "<log_dir>" > /tmp/taskb_stageA_latest
```

失败时：

```bash
echo "FAILED stage=A substage=<A0/A1/A2/A3/A4> reason=<reason> log=<log_dir>" > /tmp/taskb_stageA_status
```

通过时：

```bash
echo "PASS stage=A reason=survival_and_low_wheel_ok log=<log_dir>" > /tmp/taskb_stageA_status
```

Stage A 只需要：

```text
/tmp/taskb_stageA_status
/tmp/taskb_stageA_latest
docs/taskb_stageA_report.md
关键日志
```

不要创建：

```text
results.csv
leaderboard.md
configs/B001.json
candidate ranking
```

---

## 6. Stage A0：入口、接口、action 维度检查

### 6.1 目标

确认不是最基础的入口错误。

### 6.2 必查项

```text
[ ] demo/solution.py 是否指向 Task B
[ ] AlgSolution 是否可 import
[ ] 是否实现 predicts(obs, current_score)
[ ] 若评测可能调用 predict(obs, current_score)，是否有薄转发
[ ] action 是否是 list / np.ndarray 等可序列化格式
[ ] action len 是否为 24
[ ] action 中是否有 NaN / inf
[ ] wheel slice 是否为 [12:16]
[ ] arm slice 是否为 [16:24]
```

### 6.3 必跑命令

根据实际 conda 环境调整 python 路径：

```bash
cd /root/ATEC2026_Simulation_Challenge

PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python - <<'PY'
from demo.solution import AlgSolution
s = AlgSolution()
print("IMPORT_OK", type(s).__name__)
print("has_predicts", hasattr(s, "predicts"))
print("has_predict", hasattr(s, "predict"))
PY
```

### 6.4 若需要最小接口兼容 diff

只允许类似这种薄转发：

```diff
 class AlgSolution:
+    def predict(self, obs, current_score=0.0):
+        return self.predicts(obs, current_score)
```

禁止在 A0 阶段重写策略主体。

### 6.5 A0 通过条件

```text
[ ] IMPORT_OK
[ ] AlgSolution 存在
[ ] predicts 存在
[ ] predict 兼容或确认评测不需要
[ ] 首次 action 长度可打印/断言为 24
```

---

## 7. Stage A1：zero-action survival test

### 7.1 目标

判断机器人在不主动运动时是否也会立即终止。

如果 zero action 都 0.00s 死，根因大概率不是 wheel 速度，而是：

```text
- reset 初始状态本身不稳定
- default action / default joint target 不安全
- action spec 不匹配
- robot/env 选择错误
- illegal contact / fall 由初始姿态触发
```

### 7.2 实验要求

临时加入一个 debug mode：

```text
TASKB_DEBUG_MODE=zero
```

在 debug zero 模式下：

```text
前 300 steps：
  leg = 0
  wheel = 0
  arm = 当前 baseline 最安全动作或 0
  giveup = False
```

### 7.3 允许的最小代码形态

只加开关，不重构：

```python
self.debug_mode = os.environ.get("TASKB_DEBUG_MODE", "")
```

逻辑：

```python
if self.debug_mode == "zero":
    action = make_safe_zero_action()
    return {"action": action, "giveup": False}
```

### 7.4 必看日志

```text
- 是否 Episode done. score:0.00 time:0.00s
- termination reason
- first 10 steps 是否存在
- base height / base pose 是否异常
- fall / illegal_contact / action spec / reset 报错
```

### 7.5 A1 结论判定

| 结果 | 判断 |
|---|---|
| zero action 也 0.00s 死 | 不要调 wheel；优先查 env/robot/default action/reset/termination |
| zero action 能活 | 可以进入 A2/A3，验证 settle 和 wheel |
| zero action 能活但不动后超时 | 正常，说明 reset 生存性可接受 |

---

## 8. Stage A2：settle-only test

### 8.1 目标

验证 reset 后是否需要物理 settle。

### 8.2 实验设置

```text
TASKB_DEBUG_MODE=settle
TASKB_SETTLE_STEPS=100
```

行为：

```text
0 ~ settle_steps:
  输出 safe hold action
settle_steps 之后:
  仍不导航，继续 safe hold
```

### 8.3 推荐 safe hold

优先级：

```text
1. 当前 baseline 已知最安全的 action
2. zero action
3. 保持 leg/arm 默认，wheel=0
```

不要在 A2 阶段加入航点跟踪。

### 8.4 A2 结论判定

| 结果 | 判断 |
|---|---|
| A1 死，A2 活 | settle 有用；后续所有策略必须加 settle |
| A1 活，A2 活 | reset 基本稳定；进入 low wheel |
| A2 仍死 | 查 default action / 初始姿态 / env config |

---

## 9. Stage A3：settle + low-wheel ramp test

### 9.1 目标

验证低速 wheel 是否安全。

### 9.2 实验设置

```text
TASKB_DEBUG_MODE=low_wheel
TASKB_SETTLE_STEPS=100
TASKB_RAMP_STEPS=100
TASKB_WHEEL_FWD_CMD=0.10
```

然后再测：

```text
TASKB_WHEEL_FWD_CMD=0.16
```

### 9.3 控制逻辑

```python
if step < settle_steps:
    wheel = 0
elif step < settle_steps + ramp_steps:
    alpha = (step - settle_steps) / ramp_steps
    wheel = alpha * target_wheel_cmd
else:
    wheel = target_wheel_cmd
```

### 9.4 关键要求

```text
- 不做 coverage
- 不转弯
- 不 sweep arm
- 只测试前进 wheel 是否安全
```

### 9.5 A3 通过条件

```text
[ ] 0.10 能活过 settle + ramp
[ ] 0.16 能活过 settle + ramp
[ ] base pose 有实际位移
[ ] 没有 fall / illegal_contact
```

如果 0.10 都导致立即终止，不要进入 coverage。继续查 wheel layout 或 robot action spec。

---

## 10. Stage A4：high-wheel comparison

### 10.1 目标

验证当前 baseline 的 `WHEEL_FWD_SPEED=0.6` 是否过激。

### 10.2 实验设置

```text
TASKB_DEBUG_MODE=high_wheel
TASKB_SETTLE_STEPS=100
TASKB_RAMP_STEPS=100
TASKB_WHEEL_FWD_CMD=0.60
```

### 10.3 判定

| A3 低速 | A4 高速 | 结论 |
|---|---|---|
| 活 | 死 | wheel speed 过激基本成立 |
| 活 | 活 | 0.6 不是立即终止主因，但仍可能不适合稳定得分 |
| 死 | 死 | wheel layout / action spec / robot state 仍有问题 |
| 死 | 活 | 实验不可信，需检查代码分支是否真的生效 |

---

## 11. 推荐最小修复方向

在 A0-A4 之前不要盲改。  
若 A3 证明低速可活，A4 证明高速易死，则 Stage B 最小修复方向为：

```text
1. 加 settle_steps
2. 加 wheel ramp
3. 将启动 wheel_fwd_cmd 从 0.6 降到 0.10~0.16
4. coverage 速度使用慢启动
```

推荐参数：

```text
settle_steps = 100
ramp_steps = 100
wheel_fwd_cmd_safe = 0.12
wheel_fwd_cmd_max = 0.16
wheel_turn_cmd_safe = 0.10~0.18
```

注意：这些是候选参数，不是事实结论。必须用 A3/A4 验证。

---

## 12. Stage A Report 模板

必须生成：

```text
docs/taskb_stageA_report.md
```

模板如下：

```markdown
# Task B Stage A Report

## Status

PASS / FAILED / BLOCKED

## Scope

本阶段只定位 0.00s episode 根因，不做策略优化、不做参数搜索、不做 IL/RL。

## Environment

- commit:
- branch:
- python:
- task:
- robot:
- date:

## A0: Entry and Action Check

### Commands

```bash
...
```

### Result

- IMPORT_OK:
- has_predicts:
- has_predict:
- action_len:
- wheel_slice:
- arm_slice:

### Conclusion

...

## A1: Zero-Action Survival Test

- debug_mode:
- steps_target:
- episode_result:
- time_s:
- termination:
- first_steps_seen:
- conclusion:

## A2: Settle-Only Test

- settle_steps:
- episode_result:
- time_s:
- termination:
- conclusion:

## A3: Low-Wheel Ramp Test

| target_wheel_cmd | settle_steps | ramp_steps | survived | base_moved | termination | conclusion |
|---:|---:|---:|---|---|---|---|
| 0.10 | 100 | 100 | | | | |
| 0.16 | 100 | 100 | | | | |

## A4: High-Wheel Comparison

| target_wheel_cmd | settle_steps | ramp_steps | survived | base_moved | termination | conclusion |
|---:|---:|---:|---|---|---|---|
| 0.60 | 100 | 100 | | | | |

## Root Cause Hypothesis

- confirmed:
- likely:
- ruled_out:
- unknown:

## Gate Decision

Can proceed to Stage B: yes / no

Reason:

## Minimal Next Diff

```diff
...
```

## Logs

- status_file:
- latest_log_dir:
- key_log:
```

---

## 13. 给 Codex / Claude 的总控 Prompt v2

直接复制以下内容：

```text
你现在接手 ATEC2026 Task B，但当前只允许执行 Stage A First。不要按“第几天”安排，不要进入后续阶段。

当前核心问题：
Task B episode 出现 0.00s 结束。接口和 action 维度可能已基本正确，但仍需用最小消融确认是否为 reset 后物理不稳定、wheel 命令过大、fall/illegal_contact 或 action spec 问题。

当前禁止：
1. 不写新算法。
2. 不做模块化拆分。
3. 不新增依赖。
4. 不做参数搜索。
5. 不做 IL/RL/ACT/CEM。
6. 不做 bucket push。
7. 不做十文件 taskb_policy 重构。
8. 不自动线上提交。
9. 不覆盖 safe 版本。
10. 不粘贴整份重写文件，只允许最小 diff。

Stage A First 必须完成以下消融：

A0：入口与 action 检查
- demo/solution.py 是否指向 Task B
- AlgSolution 是否可 import
- predicts / predict 是否兼容
- 首次 action len 是否为 24
- wheel slice 是否为 12:16
- arm slice 是否为 16:24

A1：zero-action survival test
- 前 300 steps 输出 safe zero / hold action
- 判断 reset 后机器人不动是否也会 0.00s 终止

A2：settle-only test
- settle_steps=100
- 只输出 safe hold，不导航
- 判断物理 settle 是否必要

A3：settle + low-wheel ramp test
- settle_steps=100
- ramp_steps=100
- wheel_fwd_cmd 从 0 缓慢升到 0.10，再测 0.16
- 判断低速 wheel 是否安全、base pose 是否移动

A4：high-wheel comparison
- 同样 settle/ramp
- target wheel_fwd_cmd=0.60
- 验证当前 baseline 速度是否过激

输出要求：
1. 如需改代码，只给 unified diff。
2. 给出每个实验的运行命令。
3. 写 /tmp/taskb_stageA_status。
4. 写 /tmp/taskb_stageA_latest。
5. 生成 docs/taskb_stageA_report.md。
6. 表格总结 A0-A4：实验、是否 0.00s、termination、base 是否移动、结论。
7. 明确 Gate Decision：是否允许进入 Stage B。

若任何实验出现 import error、action 维度错误、0.00s 且原因不明，立即停止后续实验，写 FAILED 状态并等待人工决策。
```

---

## 14. Stage A 日志诊断 Prompt

当你拿到日志时，用这个 prompt：

```text
下面是 ATEC2026 Task B Stage A 日志。请只做根因诊断，不要提出新策略。

请输出：

A. 实验编号：
A0 / A1 / A2 / A3 / A4

B. 失败类型：
- import error
- action len mismatch
- action NaN / inf
- action spec violation
- reset physics unstable
- wheel command too aggressive
- fall
- illegal_contact
- timeout
- unknown

C. 证据：
引用日志中最关键的 3~8 行。

D. 当前能排除什么：
例如：接口错误已排除 / wheel 过激未排除 / reset 本身不稳定未排除。

E. 下一步只允许做一个动作：
例如：继续 A2 / 降 wheel 到 0.10 / 打印 termination reason / 停止等待人工。

F. 是否允许长跑：
yes / no

禁止：
- 不要写新算法
- 不要进入 Stage B
- 不要建议 IL/RL/参数搜索
```

---

## 15. Stage B 进入条件

只有 Stage A report 写明以下内容，才允许进入 Stage B：

```text
[ ] zero-action 或 settle-only 能稳定存活
[ ] low-wheel ramp 能使 base 移动
[ ] action_dim=24 已确认
[ ] wheel slice 已确认
[ ] termination reason 已知或 0.00s 已消除
[ ] 已找到一个不会立即 reset 的 wheel_fwd_cmd 范围
```

进入 Stage B 后，优先级也不是十文件模块化，而是最小 safe 修复：

```text
1. settle_steps
2. wheel ramp
3. wheel_fwd_cmd 降速
4. 恢复 coverage baseline
5. 观察是否出现非零 score
```

---

## 16. Parking Lot：当前不执行

### 16.1 IL mode selector

暂不执行。只有以下条件满足才恢复：

```text
- safe baseline 已稳定非零
- 有足够日志
- 至少还有充足算力和时间
- 不影响 safe-final
```

### 16.2 RL / CEM

暂不执行。当前做 RL/CEM 是错误优先级。

### 16.3 bucket push

暂不执行。只有 grasped_objects safe 分稳定后才考虑。

### 16.4 十文件模块化

暂不执行。等最小 safe 修复验证后，再做低风险整理。

---

## 17. 最终行动指令

现在不要再讨论高分策略。执行顺序固定为：

```text
A0
→ A1
→ A2
→ A3
→ A4
→ Stage A Report
→ 人工判断是否进入 Stage B
```

当前最可能有价值的两个修复方向是：

```text
1. reset 后加入 settle_steps；
2. wheel_fwd_cmd 从 0.6 降到 0.10~0.16，并使用 ramp。
```

但它们必须通过 A0-A4 消融验证，不能直接当作结论。

---

## 18. 给 Agent 的汇报格式

每次回复必须使用：

```markdown
# Stage A Progress

## Current Substage
A0 / A1 / A2 / A3 / A4

## Status
PASS / FAILED / BLOCKED / RUNNING

## Commands Run
...

## Files Changed
...

## Minimal Diff
```diff
...
```

## Evidence
...

## Conclusion
...

## Next Single Action
...

## Gate
Can proceed to next substage: yes/no
Can proceed to Stage B: yes/no
```

不要输出泛泛总结，不要说“可以考虑”。只给证据和下一步单动作。
