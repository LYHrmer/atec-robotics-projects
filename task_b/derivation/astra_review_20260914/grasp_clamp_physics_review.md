# Task B 闭夹预紧核对（2026-09-14，本地方案）

结论：下一候选使用现有 helper 上限 preload=.025 m、paired lift=.20 rad，保留原驱动、动作约束与所有实测门槛。当前证据只支持一次有界夹持试验，未证明瓶子能够被夹起。

原规则事实：

- `source/atec_rl_lab/atec_rl_lab/assets/objects/task_b/object.py` 的 `Mustard_cfg` 将质量覆盖为 **.5 kg**，接触偏移 .01 m、rest_offset=0；旧静态 USD 内 .1 kg 不是实际 Task B 配置质量。
- `assets/robots/b2w.py` 的附臂 `arm_joint.*` 使用 stiffness=80、damping=4、effort_limit_sim=100；不能套用独立 Piper 的 800/80。
- `tasks/task_b/env_cfg.py` 明确 `self.events.physics_material=None`，故基类 .8/.6 startup 随机材料不生效。Task B 将 `sim.physics_material` 设为地形材料：静/动摩擦均1、combine=multiply。IsaacLab `simulation_context.py` 创建并绑定 PhysicsScene/defaultMaterial。CPU 解析 mustard 与组合 b2w_piper 未发现 authored PhysicsMaterialAPI，瓶只绑定视觉 OmniPBR。该源码链支持默认材料推断，不冒充实时接触材料测量。
- 原 arm action 是 default-offset position，scale=.5、clip=None；真实回合 environment_metadata.json 也证实此 schema。原 q7 硬限 [0,.035] m、q8 [-.035,0] m；预紧只发送越过闭合端的有限位置参考，不改变硬限。
- USD q7/q8 均为沿关节 Z 的棱柱关节，变换后分别沿 gripper 的 -Y/+Y 合拢；轴基点 gripper local Z=.1358 m。原碰撞手指在 local Z 约 .0593–.1358 m，使用凸包碰撞。原 USD 驱动100/1/maxForce10会被上述 Task B actuator 配置覆盖，不能直接作为运行值。

名义零速度弹簧力只作量级检查：两指幅值和约 `80*(jaw_width+2*preload)` N。在夹缝49.4 mm时，.015/.025预紧分别约6.35/7.95 N；夹缝34.6 mm时约5.17/6.77 N。物体重力约4.905 N。即使默认摩擦为1，这也不是已测法向力或可靠抓力：瓶面倾斜、接触几何和阻尼会影响向上承载，接触偏移也不等于真实夹持厚度。

选择 .025 而非更大位置目标，是因为它已在现有 helper 的明确范围内，并提供可量化的约1.6 N名义总力增量。此次保持接触位置，避免同时引入再次下探/视觉对位变量。若新回合仍显示瓶底贴原支承面而夹爪滑出，下一步应改接触获取，不继续扫抬升幅度或宣称增加预紧必然有效。

close 最早1.4 s：单指 .035→-.025、.05 m/s需1.2 s，随后10个50 Hz样本覆盖约.2 s控制占用。最大3 s仍按真实宽度窗口判定。LIFT5 s、OBSERVE2 s、总10 s保持；总预算优先，三个阶段不能同时耗满各自上限后再额外延时。

本轮真实验收仍是同一对象的实际姿态、mesh最低面脱离原支承至少10 mm、root升至少15 mm，并随夹爪保持至少1 s。公开 pinch rise 是控制门槛，不是物体离地证据。原正分只证明官方末端距离事件。
