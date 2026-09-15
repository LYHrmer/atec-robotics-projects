# P2 最小实现合同：真实伸臂到位后的有限参考下降

2026-09-14，Astra 预备方案；**尚未放行实施或运行**。只有正在准备的 P1 新回合已经实际到达 q_goal、姿态稳定、轮保持有效、没有原非法接触终止，且独立诊断表明剩余距离是可用1–3 cm改善的小缺口时，才采用本文。P1 未到位就先解决其真实失效，不追加下探。

## 放行必须来自哪份实际证据

本轮 P1 result/trace/telemetry 中：真实视觉 gate→REACH；最大臂关节误差≤.04 rad并保持≥.5 s；公开平面速度<.01 m/s连续≥.5 s；轮保持未重锁anchor，公开累计平移/yaw/重力边界未超限；原终止为假。独立审计使用同一到位时间的实际 gripper_base和物体root，报告三维距离与水平/竖直分量；实际距离明显大于约.23 m或主要是水平错位时，不以加深到.03代替定位/IK修正。

旧hold回合2340的静态理想距离 .21683 m、假设下降2 cm后 .19686 m 只是参考。它不足以放行，也不能输入策略。下一P1实际距离若已得分，先完成正分审计/复现/交付，不为首分再改下探代码。

## 最少实现变化

**Opus 只改 FirstReachPolicy 与其 CPU 审计；root 负责 evaluate 中已有参数限制和元数据。** 继续使用同一个 reach_q、现有 lower_delta与StanceReference→StanceHold组合，不引入新腿模型、GT、reward/contact控制或新的搜索/导航逻辑。

1. 在首次真实 stationary_gate授权时建立一次固定目标/轨迹许可。`ARM_REACH`期间实际lowering_alpha=0，即使本回合配置L>0，也应执行已通过P1的有界关节反馈合同；不能再因为 `lowering_m>0` 从伸臂第一步起取消缺测许可。到达 q_goal后，另走下面的公开静止准入，**不是error<.04的同一步立即下探**。
2. 保存 `lower_ready_since`。须臂误差<.04 rad、臂qdot最大值<.05 rad/s、腿q在.5 s窗口内的每关节最大-最小值<.02 rad、平面速度<.01 m/s、角速度norm<.12 rad/s、tilt≤.10 rad，连续.5 s后才锁存 `lower_authorized=True`，保存此时腿q为q0。腿使用实际公开角度的有限时间变化范围，避免把最后子步高频qdot当作持续位移；这一修订基于后续locked回合的瞬时角速/姿态差异证据。目标此时已可能出框，不再要求一个几何上无法获得的额外视觉授权；使用最初确认的静止地面目标与严格限定的后续动作。3 s仍未形成静止窗口就结束，reason明确。
3. alpha仍按最多dt/3递增，仅在lower_authorized时增加；固定L≤.03，固定臂q_goal，保持同一轮anchor。沿用现有动作 `leg_delta=alpha*lower_delta/leg_scale`。到alpha=1后保持2 s并正常结束；不得以缺测或0分为理由在同回合提高L。**状态仍保持REACH，新增debug `reach_subphase=ARM_REACH/WAIT_LOWER_QUIET/LOWER_REFERENCE/LOWER_HOLD`，防止现有evaluate轮保持白名单在下探时意外release。**
4. 下探监控继续执行P1原倾角、公开base速度、角速度、累计平面位移/yaw/重力变化、q误差tether和有限时长规则。增加两项直接相关的关节检查：`max(abs((q_leg-q0)-alpha*lower_delta))>.06 rad`连续.2 s结束；在alpha≥.5满1 s之后，实际腿变化沿lower_delta的投影 `beta=dot(q_leg-q0,lower_delta)/dot(lower_delta,lower_delta)`仍<.15，记关节参考无进展并结束。beta是关节响应，不是机身下降量。新相机候选仍按已锁定目标空间门控记录，不重做IK追分；缺测不撤销已经限定的本次缓降动作。
5. 保持原ARM_REACH deadline；首次授权下探后，另给下降+末端保持总共最多6 s，整个目标尝试上限为原ARM deadline+6 s。时间或姿态/跟踪失败时锁存当前alpha并调用原明确停止出口，禁止把alpha瞬间归零。停止/冻结不称自动抬升恢复；暂不实现未经测试的回升轨迹。无GT或reward进入任何判断。

原有下探3 s只按实际允许增长的tick计累积；无进展或不静止暂停期间，墙上仿真时间仍计入6 s上限，不能无限等。lower_authorized、q0和deadline都不可被后续图像/状态名称重复重置。

## root 的必要配套修改

已有 `--brake_wheel_hold` 当前被限定为 first_reach且lowering=0，这个**CLI限制必须在P2获准时**放开到 first_reach且lowering∈[0,.03]；否则新回合在启动前就被拦。没有启用轮保持的 first_reach 下探仍不属于本轮获准配置。物理轮速覆盖维持已经验证的原顺序：正常gain→腿参考/保持→共同物理wheel速度/scale→可选stabilizer；不再改变单位或重排控制器。

元数据与trace记录L、lower_authorized时刻、q0、alpha、beta、增量跟踪误差、各阶段剩余时间。原奖励/对象root/真实下降仍只是独立验收数据。保存原first_reach和环境源码快照，所有回合新进程/新目录。

## 运行顺序与实际验收

根代理最新倾向在P1真正完成稳态后优先试固定L=.02，而不是每一厘米都重复接近；待实际稳态距离约.21–.23 m、水平误差较小且姿态/轮锚稳定时，记录依据并以新进程执行原3 s慢ramp。它连续经过1 cm参考阶段，不能称为已经独立验证过1 cm。若实际缺口更小，root仍可固定L=.01；L=.02稳定执行但仍不足时才考虑下一新进程L=.03。每回合只改变固定L，不从运行中得分/真值自动改变幅度。此选择须等P1真实稳态证据，不由旧静态预测单独放行。

每回合验收同时报告：实际腿关节响应；alpha对应期间实际base与gripper世界z变化；水平漂移；臂q误差；tilt；原终止；真实最小gripper-root距离与分项奖励。实际下降不足名义L，不额外自动补偿；只在独立审计后决定下一个固定L。下探过程中首个正分按原post-step/terminal-pre-reset证据审计，不能把beta、FK或视觉表面距离代替奖励。

## 必要 CPU 检查

- L>0但alpha=0时，已确认目标的ARM_REACH可在缺测下继续，与P1相同；未通过静止gate不得获得许可。
- 真实q未到位、腿/臂还在运动、底盘未静止时，alpha一直为0且准入超时有效；单次合格tick不能启动。
- 连续静止.5 s后只授权一次，alpha的增长率/上限和固定L不被目标缺测/重复样本/奖励改变。
- 下探3 s+保持2 s在有限deadline内完成；暂停不能无限延长；跟踪/姿态/无进展失败锁存alpha、不跳零。
- q0/腿索引按公开observation joint names映射，不能拿articulation顺序直接索引proprio。
- root检查 `REACH` 白名单下从臂运动到下探的wheel anchor不改变、物理轮速映射仍准确；初始控制配置无flag或lowering=0时保持已通过P1行为。

该合同不要求实现真实抓取、目标根估计、底盘高度闭环或通用下蹲策略；它只验证一个已确认目标旁的有限参考下降，不能扩写成18件能力。
