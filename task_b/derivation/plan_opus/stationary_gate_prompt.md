# 明确的代码任务：实现停稳后的公开视觉目标确认门槛

用户指定 GPT-6 Astra ultra 制定方案，Claude Opus 做具体代码执行。你这轮只实现下面两个文件，不再扩展阅读整个项目或制定总方案。主代理正在并行集成与准备仿真。没有 .codegraph 目录，跳过CodeGraph。

工作目录 /home/lybm/ATEC_Robotics_Projects_20260910。
你独占且只允许写：
- task_b/stationary_target_gate.py
- task_b/audit_stationary_target_gate.py

不要编辑 first_reach.py、evaluate.py、README、任何官方源码；不要运行仿真、shell、网络、安装依赖、git。可以使用 Read/Write/Edit 编写以上文件。CPU审计由主代理用现有Python环境执行，所以你不需要Bash。请直接写出完整代码，不只输出建议。

## 接口合同（由 Astra ultra 确定）

Python 3.10+，不依赖Isaac/GPU。numpy可用，但优先简单标准库。

class StationaryTargetGate:
- __init__(dt=.02, sensor_period=.1)，其他合理默认参数可命名可配，但以上构造必须可用。
- reset(step): BRAKE进入时清除所有运动期目标/计数/时间，step是当前非负整数控制调用数。
- update(step, quiet, point_body=None, source=None, frame_token=None) -> dict。
  point_body仅在本次10Hz检测有新raw点时传入，形状3元素有限数；非检测tick传None。quiet由调用端公开机身线速度<.06m/s、角速度<.12rad/s、六臂qdot<.05rad/s判断，不让本模块看GT/奖励/物体root。
- 返回每次均具有ready(bool)、target_body(list3或None)、confirmations(int)、last_fresh_step(int或None)、age_s(float或None)、reason(str)、freshness_basis(str)。额外diagnostic字段可加。

行为：
1. quiet必须先连续 .2s，之后再等待完整1个sensor_period排除最后运动图像，才能接受第一有效静止点。reset后的第一调用不能立刻ready。
2. 同一source至少两个相隔>=sensor_period的新更新，位置差<=.04m才ready；空间跳变重新开始确认，不能平均成假目标。
3. 有frame_token时同一来源相同token重读永不增加确认；不同token但小于采样周期也不得提前放行。无token时不得伪造真实帧ID：freshness_basis明确'sensor_period_elapsed_assumption'；只根据已传入的检测样本与控制时长接受。静止时新图像可与旧像素完全相同，禁止要求像素变化。
4. quiet失效立即清空静止证据。source切换重建同源两帧证据，不能混相机成两帧。
5. 无候选(point=None)不增加确认、不刷新last_fresh；距上次有效样本>.5s时ready=false，之后需要重新确认，不允许用旧ready穿透。
6. 非有限/错误形状输入、时间倒退/重复step、无合法source应有明确安全行为或ValueError，禁止静默ready。重复调用同一step不能计数两次。step推进而无候选是正常情形。
7. target_body从停稳后观测得出，可用两次一致样本中位/平均；不得持有调用者可变数组引用、不得依赖旧目标。
8. 本模块只用于BRAKE停稳定位，**不能用于REACH运动期间**。臂向下时目标按几何预期出视野/低于检测最小深度，所以运动期失效监控由主状态机负责。不要给本模块加IK/动作/全局导航。

## CPU审计交付

audit_stationary_target_gate.py能从仓库根目录运行 `python -m task_b.audit_stationary_target_gate --output PATH`，输出真实测试JSON并返回非零失败码。只导入本模块/标准库/numpy，不导入evaluate/Isaac。
用明确时间线测试：运动前点丢弃；连续quiet+.1s刷新等待；不能两个快调用立刻ready；同token永不重算；无token使用明确假设标注；新token且同源两帧通过；像素不参与逻辑；source切换；跳变；丢点>.5s；quiet失效；非有限和形状；重复step/倒退；reset；防止返回或输入数组别名污染内部状态。

不要为了数量写镜像实现的测试；覆盖这些真实失效模式即可。保存真实代码后最终简洁报告文件、接口和你未执行CPU/GPU测试的事实，主代理随后验证。任务到此结束，不做其他设计。
