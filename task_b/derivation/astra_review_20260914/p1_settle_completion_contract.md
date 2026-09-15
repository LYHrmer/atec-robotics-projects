# P1 有限停稳收尾合同

2026-09-14；依据已完成的 `plan_p1_locked_seed42_01`，不修改控制器，不运行 GPU。

## 实际事实与未知项

本回合2,629个物理步、0分、无原任务终止；policy call2630在执行新物理步之前因 `reach_base_not_stationary` 结束。实际q2最终为3.1400087 rad；3.06258是控制命令，不是实测角度。全臂最大目标误差≤.04 rad从2621到2630仅约.18 s，最后误差约.01995 rad，臂仍有运动，尚未形成稳定到位证据。

最后完整pre-step2629的公开线速度norm=.02074 m/s、平面速度norm=.00117 m/s；整个REACH平面速度最大约.00479 m/s。末端到object_10根的最终实际距离约.21358 m，其中水平.01888 m、竖直高差.21274 m。原非法接触力记录全为0。不能称为非法接触造成停止，也不能把小于几厘米的剩余距离写成已得分。

独立审计发现最后50个公开样本roll角速符号翻转49次，幅值约±.109 rad/s；同一秒真实机身姿态相对首样本最大变化约.001054 rad。标准公开观测来源是root_ang_vel_b，未缩放、未加噪声；瞬时最后子步速度与控制周期姿态差分可能不同，不能直接把它叫传感器噪声。2630触发样本的public84此前未保存，**不能从相邻姿态唯一重建其瞬时linvel/omega，不能严格断言OR判断是哪项超限**。root已补充未来policy_stop_record的完整public84保存。

独立证据位于 `task_b_score_cpu/plan_p1_locked_seed42_01_reach_readiness.json`。本轮支持继续验证零下降的稳定收尾，仍不放行P2。

## 下一单因素决定

只把 `_reach` 中“瞬时norm(v)≥.06或norm(omega)≥.12立即退出”，改为**保持当前臂命令、等待有限停稳**。两个阈值均不提高；不同时增加下降、换IK、改轮保持或加末端减速。是否需要减速由下一轮实际表现决定。

## 给 Opus 的最小代码范围

仅 `task_b/first_reach.py` 和对应 `task_b/audit_first_reach.py`。不新增框架，不改evaluate/轮保持/检测器/原物理。root的观测记录补充属于诊断改动，不改变动作。

1. 增加本次reach的 `settle_active/start/quiet_since/total_seconds` 字段，只有新静止目标进入REACH时初始化。每步先记录公开线速度向量/norm、重力切平面速度norm、角速度向量/norm，再检查原累计位移/yaw/gravity预算及总deadline，保证等待时这些检查仍执行。
2. norm(v)≥.06或norm(omega)≥.12触发/维持settle；冻结当前 `arm_command`，不再推进关节目标，但仍按既有命令保持支撑，lowering_alpha严格为0。**policy.state保留REACH**，用reason/debug `reach_subphase=SETTLE_TRANSIENT_BASE_RATE` 标识，避免evaluate轮保持白名单release。
3. 两个原速率条件连续正常.20 s后恢复同一q_goal。等待期间若再次异常，清除quiet_since，但不清除本次settle开始时间、累计等待时间或reach总时限。每次等待最长2 s、整个reach累计等待最长3 s，超过则明确结束；不能反复给自己完整新时限。恢复时不重锁目标/轮anchor、不重求IK、不清空3 s关节进展历史或累计位移预算。
4. 终点原2 s观察计时，仅在 `max(abs(q-q_goal))<.04`、臂qdot最大绝对值<.05 rad/s、平面速度norm<.01 m/s、两个原速率条件正常且不在settle时持续累积；任何条件不满足就清零。该条件用于证明稳定到位，不把路过误差阈值当作稳态。此时仍执行原关节命令/tether，不用GT决定动作。
5. 原act中的tilt>.12、滚转/俯仰角速>.45恢复入口、.25 rad硬停，以及原最大位移.03 m/yaw.05 rad/重力方向变化.05 rad预算不变；它们不能被settle分支屏蔽。合成reach_only与lowering>0分支不扩大授权。本轮仍lowering0。

公开角速度瞬时波动本身仍记录；有限等待是为了区分短脉冲与持续运动，不宣称已经证明该信号错误。冻结命令不是主动抬升、刹车新算法或回撤。

## 必要 CPU 与下一实际回合验收

CPU只检查具体分支：一tick角速>.12进入settle且臂命令冻结/状态仍REACH；正常持续.2 s恢复原轨迹；持续异常2 s退出；多次事件累计3 s退出；等待中的位移预算/姿态硬停/总deadline仍生效；瞬时q误差进入.04但qdot过大不能计终点2 s；真正连续安静才能完成。输入/输出均有限，wait重入不能重置授权。

新进程使用上一回合完全相同参数：seed42、compact、stance_hold、brake_wheel_hold、gain8、forward.05、turn_cap.6、turn_gain2、standoff.50、lowering0及同5,000步上限。新目录建议 `plan_p1_settle_seed42_01`。只改变上述REACH收尾规则。

通过条件是实际臂稳定到位2 s、轮anchor持续保持、累计公开运动预算未超限、无原非法接触终止；同时独立验算实际末端—root距离/分项奖励。若稳定到位仍0分，再用该稳态窗口决定P2。以本轮最终姿态做刚性垂直下降1/2 cm估算约.20362/.19367 m，只是条件预测，不能代替下一回合实际收尾或下探证据。
