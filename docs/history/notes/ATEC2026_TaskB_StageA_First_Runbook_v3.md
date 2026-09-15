# ATEC2026 Task B Stage-A-First 执行手册 v3

> 给 Codex / Claude 直接读取。  
> 当前目标不是写高分策略，而是先定位并清除 **Task B episode 0.00s / reset 后立即终止 / 多 episode 越跑越不稳定** 的根因。  
> 在 Stage A 未通过前，禁止进入模块化重构、参数搜索、IL、RL、CEM、bucket push。

---

## 0. 当前版本定位

### 0.1 v3 相对 v2 的关键修正

v2 的主线正确，但还需要两个细化：

1. **A1 拆成 A1a / A1b**
   - A1a：zero-action quick death test，目标 10~20 steps，用来判断是否几步内立即死。
   - A1b：zero-action survival test，目标 300 steps，或至少超过历史 ep1=5.5s，用来判断 zero-action 是否稳定生存。

2. **A0 不再承担 action 正确性的最终结论**
   - A0 只做纯 Python import / interface 检查。
   - action_len=24、NaN/inf、action range、真实 termination reason 必须在真实环境 A1 中确认。
   - 不允许用 dummy obs 强行证明 action 正确。

### 0.2 当前已知异常现象

已观察到类似：

```text
ep1 = 5.5s
ep2 = 0.08s
ep3 = 0.24s
之后 = 0.00s
```

这个现象说明：

```text
如果只是 wheel_fwd_cmd=0.6 过大，ep1 通常也应该很快死。
所以更高优先级的疑点是：
1. env.reset() 后物理状态不稳定；
2. 多 episode 后仿真状态残留；
3. default action / hold action 不安全；
4. reset 后立即接触异常；
5. fall / illegal_contact / action spec 被触发；
6. 真实 action 与预期 action 不一致。
```

因此当前不能凭感觉调 wheel，也不能直接上 coverage。必须先做 A0 → A1a → A1b → A2 → A3 → A4 消融。

---

## 1. 绝对硬约束

### 1.1 当前禁止事项

1. 不写新算法。
2. 不做模块化拆分。
3. 不新增依赖。
4. 不做参数搜索。
5. 不做 IL / RL / ACT / CEM。
6. 不做 bucket push。
7. 不做十文件 `taskb_policy/` 重构。
8. 不自动线上提交。
9. 不覆盖 safe 版本。
10. 不粘贴整份重写文件，只允许最小 diff。
11. 不用 dummy obs 宣称 action 正确。
12. 不在 0.00s 根因未知时继续长跑。
13. 不用单次成功证明系统稳定。

### 1.2 当前允许事项

只允许：

```text
- import / interface 检查
- 最小 debug mode 开关
- zero-action / hold-action survival test
- settle-only test
- low-wheel ramp test
- high-wheel 对照
- 关键日志打印
- 状态文件与 Stage A report
```

---

## 2. 当前唯一 Active Stage

```text
Active:
  Stage A First：0.00s 根因消融

Frozen:
  Stage B：最小 safe 修复
  Stage C：小规模参数消融
  Stage D：提交审查

Parking Lot:
  IL mode selector
  RL / CEM
  bucket push
  十文件模块化
  大规模参数搜索
```

### 2.1 阶段推进规则

```text
A0 未过 → 禁止 A1
A1a 未过 → 禁止 A1b/A2/A3/A4
A1b 未过 → 禁止 A2/A3/A4
A2 未过 → 禁止 A3/A4
A3 未过 → 禁止恢复 coverage
Stage A report 未完成 → 禁止 Stage B
```

---

## 3. Stage A 需要回答的问题

```text
Q1. demo/solution.py / AlgSolution / predicts 是否能正常 import 和实例化？
Q2. 真实环境第一次调用时，action len 是否为 24？
Q3. action 是否包含 NaN / inf / 明显越界值？
Q4. zero-action / safe-hold 下是否也会几步内死亡？
Q5. zero-action 活过 quick test 后，是否能稳定活过 300 steps 或超过历史 ep1=5.5s？
Q6. reset 后 settle N steps 是否改善生存性？
Q7. low wheel ramp 是否能让 base 安全移动？
Q8. high wheel=0.6 是否明显比 0.10/0.16 更容易终止？
Q9. termination reason 到底是 fall、illegal_contact、action spec、timeout，还是 unknown？
Q10. 多 episode 失败是否与仿真进程残留相关？
```

---

## 4. Stage A 状态文件规范

每个子阶段都要写：

```bash
echo "RUNNING stage=A substage=<A0/A1a/A1b/A2/A3/A4> reason=<short>" > /tmp/taskb_stageA_status
echo "<log_dir>" > /tmp/taskb_stageA_latest
```

失败时：

```bash
echo "FAILED stage=A substage=<A0/A1a/A1b/A2/A3/A4> reason=<reason> log=<log_dir>" > /tmp/taskb_stageA_status
```

通过时：

```bash
echo "PASS stage=A substage=<A0/A1a/A1b/A2/A3/A4> reason=<short> log=<log_dir>" > /tmp/taskb_stageA_status
```

Stage A 只需要：

```text
/tmp/taskb_stageA_status
/tmp/taskb_stageA_latest
docs/taskb_stageA_report.md
关键日志
```

Stage A 不需要：

```text
results.csv
leaderboard.md
configs/B001.json
candidate ranking
大规模实验表
```

---

## 5. 推荐日志目录

```bash
mkdir -p /tmp/taskb_stageA_$(date +%Y%m%d_%H%M%S)
```

每次实验至少保存：

```text
run.log
env.log
stageA_notes.md
```

若能拿到 termination reason，必须保存原始行。

---

## 6. Stage A0：纯 Python 入口与接口检查

### 6.1 A0 目标

A0 只确认：

```text
- demo.solution 能 import
- AlgSolution 能实例化
- 是否有 predicts
- 是否有 predict
- predict 是否只是薄转发
- 是否存在明显 Task E contamination
```

A0 不确认：

```text
- 真实 action_len 一定正确
- wheel slice 一定正确
- 环境一定接受 action
- 不会触发 termination
```

这些必须在真实环境 A1 中确认。

### 6.2 A0 必跑命令

根据实际 conda 环境调整 python 路径：

```bash
cd /root/ATEC2026_Simulation_Challenge

echo "RUNNING stage=A substage=A0 reason=import_interface_check" > /tmp/taskb_stageA_status

PYTHONPATH=$PWD /root/miniconda3/envs/isaaclab/bin/python - <<'PY'
from demo.solution import AlgSolution

s = AlgSolution()
print("IMPORT_OK", type(s).__name__)
print("has_predicts", hasattr(s, "predicts"))
print("has_predict", hasattr(s, "predict"))

if hasattr(s, "predict") and hasattr(s, "predicts"):
    print("PREDICT_AND_PREDICTS_BOTH_EXIST")
PY
```

### 6.3 A0 文件检查

检查：

```text
demo/solution.py
demo/solution_task_b.py 或当前 Task B 入口文件
```

重点看：

```text
- solution.py 是否仍指向 Task E
- 是否 import 了错误任务文件
- 是否依赖不存在的权重或路径
- 实例化是否有副作用，例如启动训练、加载大文件、访问网络
```

### 6.4 若需要 predict 薄转发

只允许最小 diff：

```diff
 class AlgSolution:
+    def predict(self, obs, current_score=0.0):
+        return self.predicts(obs, current_score)
```

禁止在 A0 重写策略主体。

### 6.5 A0 通过条件

```text
[ ] IMPORT_OK
[ ] AlgSolution 可实例化
[ ] predicts 存在
[ ] predict 存在或确认评测只调用 predicts
[ ] 未发现明显 Task E contamination
[ ] 无实例化副作用
```

### 6.6 A0 汇报格式

```markdown
# Stage A0 Report

## Status
PASS / FAILED

## Import Check
- demo.solution import:
- AlgSolution instantiate:
- has predicts:
- has predict:
- predict thin wrapper:

## Files Inspected
- demo/solution.py
- related Task B solution file:

## Risk Found
- entry mismatch:
- Task E contamination:
- import side effect:
- missing method:

## Gate
Can proceed to A1a: yes/no

## Next Single Action
Run A1a zero-action quick death test with real env.
```

---

## 7. Stage A1a：zero-action quick death test

### 7.1 A1a 目标

判断真实环境中，zero-action / safe-hold 是否在几步内立即死亡。

这是 quick death test，不是稳定性证明。

### 7.2 A1a 实验设置

推荐使用环境变量控制，不写死：

```bash
export TASKB_DEBUG_MODE=zero_quick
export TASKB_DEBUG_MAX_STEPS=20
```

行为：

```text
0~20 steps：
  leg = 0 或当前最安全 hold
  wheel = 0
  arm = 0 或当前最安全 hold
  giveup = False
```

### 7.3 A1a 必须记录

真实环境第一次调用 `predicts()` 时，必须打印：

```text
[TASKB][A1A] first_call=True
[TASKB][ACTION] len=...
[TASKB][ACTION] min=...
[TASKB][ACTION] max=...
[TASKB][ACTION] has_nan=...
[TASKB][ACTION] has_inf=...
[TASKB][ACTION] wheel_slice=12:16 values=...
[TASKB][ACTION] arm_slice=16:24 values=...
```

如果能获得 robot/base 信息，也打印：

```text
[TASKB][BASE] step=... pos=... quat/euler=... height=...
```

如果能获得 termination：

```text
[TASKB][TERM] reason=...
```

### 7.4 A1a 运行命令模板

根据实际脚本替换：

```bash
cd /root/ATEC2026_Simulation_Challenge

LOG_DIR=/tmp/taskb_stageA_A1a_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "RUNNING stage=A substage=A1a reason=zero_quick" > /tmp/taskb_stageA_status
echo "$LOG_DIR" > /tmp/taskb_stageA_latest

TASKB_DEBUG_MODE=zero_quick \
TASKB_DEBUG_MAX_STEPS=20 \
PYTHONPATH=$PWD \
/root/miniconda3/envs/isaaclab/bin/python scripts/play_atec_task.py \
  --task ATEC-TaskB-B2wPiper \
  --enable_cameras \
  --headless \
  --debug \
  2>&1 | tee "$LOG_DIR/run.log"
```

若仓库没有 `scripts/play_atec_task.py`，使用当前可运行的 Task B eval 脚本，但必须保留环境变量和日志。

### 7.5 A1a 判定

| 结果 | 判断 | 下一步 |
|---|---|---|
| 10~20 steps 内死 | zero-action 也 quick death，优先查 reset/default/action spec/termination | 停止，补 termination reason |
| 能活过 20 steps | 没有立即死亡 | 进入 A1b |
| import/action error | 接口或 action 构造错误 | 回 A0 |
| action_len 非 24 | action layout 错误 | 停止修复 |
| has_nan/inf=True | action 数值错误 | 停止修复 |

---

## 8. Stage A1b：zero-action survival test

### 8.1 A1b 目标

验证 zero-action / safe-hold 是否能稳定生存。

A1a 活过 20 steps 只能说明“不立即死”，不能证明系统稳定。  
A1b 需要跑更久：

```text
目标：300 steps
或至少超过历史 ep1=5.5s 对应时长
```

### 8.2 A1b 实验设置

```bash
export TASKB_DEBUG_MODE=zero_survival
export TASKB_DEBUG_MAX_STEPS=300
```

### 8.3 推荐重复次数

至少两次：

```text
A1b-run1
A1b-run2
```

如果出现 run1 活、run2 死，必须记录：

```text
- episode index
- 是否同一 Isaac 进程
- 是否重启进程后恢复
- 是否存在仿真状态残留
```

### 8.4 A1b 运行命令模板

```bash
cd /root/ATEC2026_Simulation_Challenge

LOG_DIR=/tmp/taskb_stageA_A1b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "RUNNING stage=A substage=A1b reason=zero_survival" > /tmp/taskb_stageA_status
echo "$LOG_DIR" > /tmp/taskb_stageA_latest

TASKB_DEBUG_MODE=zero_survival \
TASKB_DEBUG_MAX_STEPS=300 \
PYTHONPATH=$PWD \
/root/miniconda3/envs/isaaclab/bin/python scripts/play_atec_task.py \
  --task ATEC-TaskB-B2wPiper \
  --enable_cameras \
  --headless \
  --debug \
  2>&1 | tee "$LOG_DIR/run.log"
```

### 8.5 A1b 判定

| 结果 | 判断 | 下一步 |
|---|---|---|
| 能稳定活过 300 steps | reset 生存性基本可接受 | 进入 A2 |
| 中途 fall/illegal_contact | zero-action 不稳定 | 查 default/hold/reset |
| run 间差异大 | 可能有状态残留或随机 reset 不稳定 | 尝试重启 Isaac 进程对照 |
| 仍 0.00s | 不是 wheel 速度主因 | 停止，查 termination |

---

## 9. Stage A2：settle-only test

### 9.1 A2 目标

验证 reset 后 settle 是否有助于稳定。

### 9.2 A2 实验设置

```bash
export TASKB_DEBUG_MODE=settle_only
export TASKB_SETTLE_STEPS=100
export TASKB_DEBUG_MAX_STEPS=300
```

行为：

```text
0~100 steps:
  safe hold / wheel=0

100~300 steps:
  仍 safe hold / wheel=0
```

A2 仍不导航，不做 coverage，不 sweep arm。

### 9.3 A2 判定

| 结果 | 判断 |
|---|---|
| A1b 死，A2 活 | settle 显著有效，后续所有策略必须加 settle |
| A1b 活，A2 活 | reset/hold 基本稳定 |
| A2 仍死 | 查 default action / initial pose / termination |
| A2 run 间差异大 | 查进程残留、seed、episode reset |

---

## 10. Stage A3：settle + low-wheel ramp test

### 10.1 A3 目标

验证低速 wheel 是否能安全让 base 移动。

### 10.2 A3 实验设置

先测 0.10：

```bash
export TASKB_DEBUG_MODE=low_wheel
export TASKB_SETTLE_STEPS=100
export TASKB_RAMP_STEPS=100
export TASKB_WHEEL_FWD_CMD=0.10
export TASKB_DEBUG_MAX_STEPS=400
```

再测 0.16：

```bash
export TASKB_WHEEL_FWD_CMD=0.16
```

### 10.3 A3 控制逻辑

```python
if step < settle_steps:
    wheel = 0
elif step < settle_steps + ramp_steps:
    alpha = (step - settle_steps) / max(1, ramp_steps)
    wheel = alpha * target_wheel_cmd
else:
    wheel = target_wheel_cmd
```

### 10.4 A3 关键要求

```text
- 不做 coverage
- 不转弯
- 不 sweep arm
- 不做 recovery
- 只测试前进 wheel 是否安全
```

### 10.5 A3 必须记录

```text
[TASKB][WHEEL] target=... alpha=... values=...
[TASKB][BASE] step=... pos=... moved=...
[TASKB][TERM] reason=...
```

### 10.6 A3 通过条件

```text
[ ] 0.10 能活过 settle + ramp
[ ] 0.16 能活过 settle + ramp
[ ] base pose 有实际位移
[ ] 没有 fall / illegal_contact
[ ] 没有 action spec violation
```

如果 0.10 都导致立即终止，不要进入 A4；查 wheel layout 或 action spec。

---

## 11. Stage A4：high-wheel comparison

### 11.1 A4 目标

验证当前 baseline 的 `WHEEL_FWD_SPEED=0.6` 是否过激。

### 11.2 A4 实验设置

```bash
export TASKB_DEBUG_MODE=high_wheel
export TASKB_SETTLE_STEPS=100
export TASKB_RAMP_STEPS=100
export TASKB_WHEEL_FWD_CMD=0.60
export TASKB_DEBUG_MAX_STEPS=400
```

### 11.3 A4 判定

| A3 low wheel | A4 high wheel | 结论 |
|---|---|---|
| 活 | 死 | wheel=0.6 过激基本成立 |
| 活 | 活 | 0.6 不是立即终止主因，但仍可能不利于稳定得分 |
| 死 | 死 | wheel layout / action spec / robot state 仍有问题 |
| 死 | 活 | 实验不可信，检查 debug 分支是否真的生效 |

---

## 12. 多 episode / 进程残留检查

如果出现：

```text
ep1 能活，ep2/ep3 很快死，之后全 0.00s
```

必须增加一个对照：

```text
同一 Isaac 进程连续 episodes
vs
每次重启 Isaac 进程单 episode
```

### 12.1 需要回答

```text
- 重启 Isaac 进程后是否恢复？
- 是否只有连续 episode 才恶化？
- reset 是否没有完全清理 robot/object/contact 状态？
- 是否 GPU/physics warning 在多 episode 后出现？
```

### 12.2 判定

| 现象 | 推断 |
|---|---|
| 重启进程后又能活一次 | 高度怀疑 reset 状态残留 |
| 同一进程后续全部 0.00s | 高度怀疑 env reset/episode cleanup 问题 |
| 重启进程也 0.00s | 更像代码/action/default/reset 初态问题 |
| 只有 high wheel 后续变差 | 可能 high wheel 把状态打坏或触发未清理接触 |

---

## 13. 推荐最小 debug mode 实现约束

如果需要改代码，只加最小调试开关。

### 13.1 允许新增

```python
import os
```

允许读取：

```python
TASKB_DEBUG_MODE
TASKB_DEBUG_MAX_STEPS
TASKB_SETTLE_STEPS
TASKB_RAMP_STEPS
TASKB_WHEEL_FWD_CMD
```

### 13.2 不允许

```text
- 新增复杂类
- 新增 taskb_policy 包
- 改 coverage planner
- 改大量业务逻辑
- 删除原 baseline
- 引入外部库
```

### 13.3 建议 debug mode

```text
zero_quick
zero_survival
settle_only
low_wheel
high_wheel
```

### 13.4 Debug action 构造原则

优先使用当前 baseline 的 action 构造函数，避免 debug action 和真实 action 格式不一致。

必须保证：

```text
len(action) == 24
wheel slice = 12:16
arm slice = 16:24
giveup = False
```

---

## 14. Stage A Report 模板

必须生成：

```text
docs/taskb_stageA_report.md
```

模板：

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
- Isaac process mode: single / restarted each run

## A0: Entry and Interface Check

### Commands

```bash
...
```

### Result

- IMPORT_OK:
- AlgSolution instantiate:
- has_predicts:
- has_predict:
- predict_thin_wrapper:
- TaskB entry confirmed:
- TaskE contamination:

### Conclusion

...

## A1a: Zero-Action Quick Death Test

- debug_mode:
- target_steps:
- survived_20_steps:
- episode_result:
- time_s:
- termination:
- action_len_first_call:
- has_nan:
- has_inf:
- action_min:
- action_max:
- wheel_slice_values:
- conclusion:

## A1b: Zero-Action Survival Test

| run | target_steps | survived | time_s | termination | notes |
|---:|---:|---|---:|---|---|
| 1 | 300 | | | | |
| 2 | 300 | | | | |

### Conclusion

...

## A2: Settle-Only Test

- settle_steps:
- target_steps:
- survived:
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

## Multi-Episode / Process Residue Check

- same_process_multiple_episodes:
- restarted_process_each_run:
- residue_suspected:
- evidence:

## Root Cause Hypothesis

- confirmed:
- likely:
- ruled_out:
- unknown:

## Gate Decision

Can proceed to Stage B: yes / no

Required conditions:

- [ ] A0 import/interface passed
- [ ] A1a passed or termination reason known
- [ ] A1b stable or instability explained
- [ ] action_len=24 confirmed in real env
- [ ] no NaN/inf
- [ ] low-wheel ramp can move base safely
- [ ] termination reason known or 0.00s eliminated

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

## 15. Stage B 进入条件

只有 Stage A report 明确满足以下条件，才允许进入 Stage B：

```text
[ ] A0 import/interface 通过
[ ] 真实环境 action_len=24
[ ] action 无 NaN/inf
[ ] zero-action quick death 已排除，或 termination reason 已知
[ ] zero-action survival 结果稳定，或不稳定原因已定位
[ ] low-wheel ramp 能让 base 移动
[ ] 找到不会立即 reset 的 wheel_fwd_cmd 范围
[ ] 0.00s 已消除，或根因已明确
```

进入 Stage B 后，仍只做最小 safe 修复：

```text
1. 加 settle_steps
2. 加 wheel ramp
3. 将启动 wheel_fwd_cmd 从 0.6 降到已验证安全范围
4. 恢复 coverage baseline
5. 观察是否出现非零 score
```

不要立即十文件模块化。

---

## 16. 给 Codex / Claude 的总控 Prompt v3

直接复制：

```text
你现在接手 ATEC2026 Task B，但当前只允许执行 Stage A First。不要按“第几天”安排，不要进入后续阶段。

当前核心问题：
Task B episode 出现 0.00s 结束，并且已观察到 ep1=5.5s、ep2=0.08s、ep3=0.24s、之后全 0.00s 的不稳定现象。这说明根因未必只是 wheel_fwd_cmd=0.6，也可能是 env.reset() 后物理状态不稳定、多 episode 状态残留、default action 不安全、fall/illegal_contact 或 action spec 问题。

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
11. 不用 dummy obs 宣称 action 正确。

Stage A First 必须按顺序完成：

A0：纯 Python 入口与接口检查
- demo/solution.py 是否指向 Task B
- AlgSolution 是否可 import
- predicts / predict 是否兼容
- 注意：A0 不证明 action 正确

A1a：zero-action quick death test
- 真实环境运行 10~20 steps
- 第一次 predicts 调用必须打印 action_len、min/max、NaN/inf、wheel slice、arm slice
- 判断 zero-action 是否几步内立即终止

A1b：zero-action survival test
- 只有 A1a 存活后才跑
- 目标 300 steps，或至少超过历史 ep1=5.5s
- 至少重复两次，记录 run 间差异
- 判断 zero-action 是否稳定生存

A2：settle-only test
- settle_steps=100
- 仍只输出 safe hold，不导航
- 判断 reset 后 settle 是否必要

A3：settle + low-wheel ramp test
- settle_steps=100
- ramp_steps=100
- wheel_fwd_cmd 从 0 缓慢升到 0.10，再测 0.16
- 判断低速 wheel 是否安全、base pose 是否移动

A4：high-wheel comparison
- 同样 settle/ramp
- target wheel_fwd_cmd=0.60
- 验证当前 baseline 速度是否过激

额外要求：
如果出现 ep1 能活、后续 episode 快速死亡的现象，必须增加“同一 Isaac 进程连续 episodes vs 每次重启 Isaac 进程单 episode”的对照，判断是否存在 reset 状态残留。

输出要求：
1. 如需改代码，只给 unified diff。
2. 给出每个实验的运行命令。
3. 写 /tmp/taskb_stageA_status。
4. 写 /tmp/taskb_stageA_latest。
5. 生成 docs/taskb_stageA_report.md。
6. 表格总结 A0/A1a/A1b/A2/A3/A4：是否 0.00s、termination、base 是否移动、结论。
7. 明确 Gate Decision：是否允许进入 Stage B。

若任何实验出现 import error、action 维度错误、NaN/inf、0.00s 且原因不明，立即停止后续实验，写 FAILED 状态并等待人工决策。
```

---

## 17. 日志诊断 Prompt v3

拿到日志后，直接用：

```text
下面是 ATEC2026 Task B Stage A 日志。请只做根因诊断，不要提出新策略。

请输出：

A. 实验编号：
A0 / A1a / A1b / A2 / A3 / A4

B. 失败类型：
- import error
- action len mismatch
- action NaN / inf
- action spec violation
- reset physics unstable
- default hold action unsafe
- multi-episode reset residue
- wheel command too aggressive
- wheel layout wrong
- fall
- illegal_contact
- timeout
- unknown

C. 证据：
引用日志中最关键的 3~8 行。

D. 当前能排除什么：
例如：接口错误已排除 / wheel 过激未排除 / reset 本身不稳定未排除。

E. 下一步只允许做一个动作：
例如：继续 A1b / 降 wheel 到 0.10 / 打印 termination reason / 重启 Isaac 对照 / 停止等待人工。

F. 是否允许长跑：
yes / no

禁止：
- 不要写新算法
- 不要进入 Stage B
- 不要建议 IL/RL/参数搜索
```

---

## 18. Agent 汇报格式

每次回复必须使用：

```markdown
# Stage A Progress

## Current Substage
A0 / A1a / A1b / A2 / A3 / A4

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

---

## 19. 当前最终行动顺序

```text
A0
→ A1a
→ A1b
→ A2
→ A3
→ A4
→ Multi-episode / process residue check（如需要）
→ Stage A Report
→ 人工判断是否进入 Stage B
```

当前最可能有价值的修复方向仍是：

```text
1. reset 后加入 settle_steps；
2. wheel_fwd_cmd 从 0.6 降到 0.10~0.16，并使用 ramp；
3. 若连续 episode 越跑越差，检查 reset 状态残留或改为每次重启进程评测。
```

但这些必须通过 A0-A4 证明，不能直接当作结论。
