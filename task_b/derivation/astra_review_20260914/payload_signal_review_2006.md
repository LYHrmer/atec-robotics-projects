# Task B 新交接：payload 速度信号与到位门独立复核

2026-09-14；private、只读源码与已有日志、仅 CPU。未修改 production，未运行 GPU，未执行 Git，未读取私人推理记录。起点为桌面《ATEC_TaskB_当前情况汇总.md》20:04:32 版本。项目与实验根均无 `.codegraph/`。

**结论：保留当前 PayloadJointTracker，继续把导航接上；先修补静稳/HOLD 的信号契约和连续性，再做有限的移动携带实测。不需要退回重调机械臂。p13 已证明静止底盘上的姿态链与持续持有；本报告不把它升级为移动搬运或投递。**

“两采样均值”有工程价值，但它是验收信号变更。应删除“位置与速度互相矛盾必有一错”“位置范围除以窗口长度就是平均运动速率”“真实运动几乎完整通过”“验收规则未变”等表述。准确写法：**保留 .12 rad/s 数值阈值与 .5 s/.002 rad 位置窗口，把静稳速度门改为公开 qdot 的两点均值；原始 qdot 同时记录，需补独立的快速异常保护。**

证据可复现于 [CPU 脚本](payload_signal_cpu_2006.py)、[诊断 JSON](payload_signal_cpu_2006.json)、[HOLD 反例脚本](payload_hold_counterexamples_2006.py) 和 [反例结果](payload_hold_counterexamples_2006.json)。当前 `payload_motion.py` 与 p13 源码快照 SHA256 一致：`042d7b2190dd510660e55ecc43a050319d804e14dec8bdda32f2300dadc62eb0`。各原始文件 SHA 在诊断 JSON 中。

## 1. qdot 与位置差分不矛盾；25 Hz 来源尚未唯一确定

本地 metadata 明确 `physics_dt=.005 s`、`decimation=4`、公开控制采样 `.02 s`。`evaluate.py:198` 直接取 articulation 的 joint_vel，`evaluate.py:489` 和 `:594` 保存同一 pre_step 的 proprio/q/qdot。各轮 policy debug 的 raw qdot 与 telemetry 完全一致，没有发现错索引或前后时刻混接。

`(q[k]-q[k-1])/.02` 是控制采样间隔的平均变化率；qdot 是该边界报告的关节速度，并非这四个物理步的平均速度。在加减速、往返或求解器分步积分下，二者可以同时正确而幅值不同。PhysX 官方还专门说明 TGS 报告末次位置迭代子步的关节位置/速度，所报速度可能明显不同于整物理步对应的速度。该文支持“差异不等于数据错误”，不能凭它认定本轮唯一根因；本报告没有核实本次实际 solver 的完整运行配置。[PhysX Team, Implicit Spring Joint Drives, p.5](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf)

末尾稳定区的 q2 实测如下。p11/p12/p13 SWING 使用最后 201 个采样，首末相隔 4 s；p13 HOLD 此表使用 telemetry 100 样本，首末 1.98 s。25 Hz 幅值是去除常量/线性趋势后对离散 `(-1)^k` 的回归，不能区分真实 25 Hz、混叠、求解器周期或更高频率。

| 区间 | q 位置范围 rad | 总变差/时间 rad/s | q 的离散 25 Hz 幅值 rad | raw qdot 的离散 25 Hz 幅值 rad/s |
|---|---:|---:|---:|---:|
| p11 SWING 尾 4 s | .0008285 | .040393 | .0004039 | .152093 |
| p12 SWING 尾 4 s | .0008334 | .040610 | .0004061 | .153032 |
| p13 SWING 尾 4 s | .0009323 | .039416 | .0003939 | .149802 |
| p13 HOLD | .0016451 | .033851 | .0003386 | .131826 |

p11/p12 的相邻位置变化方向 **100% 交替**。p12 差分速度最大 .040829 rad/s，两点 qdot 均值最大仅 .001281 rad/s。位置小幅往返真实存在于采样到的仿真关节状态；其机械影响可能很小，但“总运动速率 ≤.0016 rad/s”不成立。范围只限制最大偏离，不限制往返的总路程。相同离散频段也见于腿/轮，支持共模或求解器解释的可能性，不能排除耦合动力学或混叠，更不能证明“所有高频成分都可无损删除”。

## 2. 两点均值确实改变门限语义；反馈与门限必须分别写清

对于 `y[k]=(v[k]+v[k-1])/2`，直接由数字滤波器频响得到 `H=e^(-jπf/fs) cos(πf/fs)`：50 Hz 下，1.67 Hz 增益 .9945、10 Hz .8090、20 Hz .3090、24 Hz .0628、25 Hz 0，群延迟 .01 s。这既能去掉交替分量，也会去掉真实交替高速运动；幅值很小的高频往返还能同时满足 .002 rad 范围门。此滤波器不是对“传感误差”的鉴别器。[SciPy 官方 freqz 定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html)

代码核验：

- [payload_motion.py:976](/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py:976) 更新 raw 和两点均值；首个 payload tick 用 current 两次初始化，之后与离线两点均值逐样本相等。
- [payload_motion.py:1094](/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py:1094) 仅以两点均值决定 `quiet_tick`，从而影响窗口收集、A/B/C 到位及最终 HOLD。
- [payload_motion.py:1016](/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py:1016) 与 `:1070` 仍给 tracker **raw qdot**；tracker 内先做 tau=.04 s 低通，再进入 q2/q3 阻尼修正，修正本身另有 tau=.10 s 低通。
- tau=.04 的实际离散滤波增益为 25 Hz **.2449（约 4.08 倍衰减）**、1.67 Hz **.9237**。代码旧注释中的“约 6 倍/保留 86%”与当前 dt/tau 不一致。两点均值没有代替这个阻尼滤波器。
- raw 姿态重力、机身角速度等硬保护仍在 `:961` 起执行。**没有独立 raw arm qdot 紧急阈值。** 原 .12 是静稳准入门，不是现存紧急门；不能声称“仍保留原 raw 臂速紧急门”。

## 3. 低频振荡显著改善已经观察到，精确因果未被独立证明

p9 RAISE 最后 4 s 的 q2 span 为 .041286 rad，去趋势后在 1.67 Hz 的正余弦拟合幅值为 .018490 rad；p10 同口径变为 .000380 rad / .00000145 rad。p11/p12 SWING 稳定尾段对应幅值约 2～3e-6 rad。数据确实支持低频大幅振荡已经显著压下，无需因 25 Hz 语义争议推翻这项进展。

但这些短窗不能给出精确模态辨识：4.02 s FFT 分辨率约 .249 Hz，本次 p6/p9 尾窗峰值格点约 1.493 Hz。`plant_model.py` 明说 **J=.727 是为了复现实测约 1.67 Hz 而拟合的值**。代入 `sqrt(K/J)/(2π)` 得 1.66954 Hz 是拟合回代；ζ=.26225、对应线性模型阻尼振动频率 1.61111 Hz。静态 sag 是合同扭矩界除以 K，不是该 J 独立预测出来的验证。它适合解释并筛选候选，不足以断言整个多关节/接触/限幅闭环已被一个惯量准确识别，更不足以断言其他姿态/负载稳定边界必为增益 1.5～2.0。

“重力前馈全轴空操作”也应缩小：p13 的 q2 前馈约 -7.7e-6 rad，确实近零，但 q3 为 **-.01658 rad**、q5 为 -.00467 rad；p10 q3 为 -.01541 rad。可说“q2 前馈近零，不能把 q2 改善归功于它”；不能说每轴全是零。降增益、增阻尼以及积分 trim 的组合实测有效，单因素因果尚未隔离。带滤波、限幅和离散延迟的外环阻尼也不具有从代码注释即可保证的“只能耗能”性质，宜改称经验改善。

## 4. HOLD 两个真实漏洞及本轮实际影响

[payload_motion.py:1064](/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py:1064) 的 `valid` 只要求 goal_error<.04 和 `quiet_tick`，**没有 `quiet_ready`**。因此位置窗口不通过时仍能累积 HOLD；CPU 在隔离对象上执行逐字 AST 提取的生产方法，构造 `quiet_ready=False、quiet_tick=True、hold_quiet_calls=99`，下一次调用得到正常完成。这不是纯注释问题。

[payload_motion.py:994](/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py:994) 的 FROZEN 分支清空位置窗口并提前返回，**没有清空 hold_quiet_calls**；正常 tick 的“无效则归零”不会执行。CPU 调用 `_pause`，FROZEN 后 50 个先前计数仍保留。未来遇到机身暂停时可跨暂停累积“连续静稳时间”。

本轮 p13 **pause_s=0**。HOLD 从 phase-transition step 5765 到停止观察 step 5865，实际经过 2.00 s；计数从 5766 开始，100 个有效样本首末只隔 **1.98 s**。不能把样本数乘 dt 一概当作观测端点时长。telemetry 的 75 个完整 26 样本窗全部通过，最大 arm span .00170064 rad；加上停止时保留的最终观察后为 **76/76** 个完整窗通过。故漏洞需要修补，但没有证据说它在 p13 造成了位置窗口不合格的虚假持有。

最小补丁交接要求：`valid` 使用完整 `quiet_ready`，任何暂停/失效/状态切换都清空 HOLD 连续计时；以首个完整合格窗口时刻建立 `valid_since`，用实际采样时间 `now-valid_since >=2.0` 验收。保留现有 26 样本端点正确的 .5 s 窗口，不把 25 样本称 .5 s。暂停冻结路径/滤波/命令而不补走漏掉的步数的设计保留；总/段 wall deadline 仍不得由暂停刷新。

## 5. 导航和释放前只补必要保护，不重开控制器调参

1. **保留 tracker 状态连续。** `advance_path=False, paused=False` 才是运输期间持续闭环保持当前固定参考的接口；每 tick 仍更新负载补偿。不要重建 tracker、清 integral/filter 或把整个 stationary policy 的 `.03 m/.05 rad` 位移/yaw预算直接套到导航上。导航需要自己的有界路程/时限/姿态守卫，静止姿态链本身未验证轮动载荷扰动。
2. **新增独立 raw 速度异常守卫。** p13 全 payload raw 六轴最大 .2359546 rad/s（EXTEND），p11/p12 最大约 .1905；**.30 rad/s 可作为下一次低速短程测试的保守单样本硬停候选**，比 p13 最大值约留 27% 余量。它是试验候选，不是已证明的机械安全速度；先 CPU replay 确认 p13 不被误杀并用合成交替尖峰确认均值零也能触发。不要恢复 raw .12 作为静稳门，否则已知 .152 的往返会重新卡死。同步记录 q 总变差/时间、span、raw/均值速度、低频残差；总变差可以先作为诊断，不为通关另行调一个宽阈值。
3. **释放前补最终限幅关联的 anti-windup 诊断/CPU 反例。** `:617` 当前“unsaturated”只看 P 项是否越 cap，未反馈最终 slew/tether/hard-limit 的夹限。p13 q2 有 345、q3 有 445 个样本，在最终命令与 `reference+ff+I+filtered` 候选不同时仍更新 I；q2 I 触及 -.20，末尾 q2/q3 I 为 -.15994/-.06146 rad。它未证明失稳，却说明“已有完整 anti-windup”不成立。新释放流程应记录 candidate-applied residual 与各限幅原因，CPU 覆盖加载→输出夹限→卸载：可按最终夹限方向与误差方向条件积分，或有界回算；让误差把命令拉回可行域的积分可以保留。释放时不能突然把补偿/滤波命令全部归零，仍需原 slew/tether 保持命令连续。
4. **下一次必要物理实测并入已有交接主线。** 完成 CPU 守卫与连续性反例后，保持原场景参数，用低速短直线携带→刹停→短转向，验证同一瓶相对夹爪漂移、mesh 最低面、车体守卫及 raw/均值速度是否仍有余量。再到桶边完成同一物体释放和 ≥3 s 观察。若只是要辨别 25 Hz 源头，可在该必要短试验中追加只读 200 Hz 物理步记录，而非另跑一轮改物理/改 solver 的“修复”。本轮交接不要求立即执行任何 GPU。

**已证实**：原始关节数据对应、离散交替成分、均值门替换、p13 HOLD 实际完整窗通过、HOLD 两处通用漏洞、低频大幅振荡改善、最终限幅未全面参与积分保护。

**合理推断**：边界采样/求解器/共同机身耦合可能解释速度差异；两点均值能帮助当前缓慢任务的静稳判定；保留现有闭环模块比从头调参更符合已有证据。

**仍未证实**：25 Hz 唯一物理来源、任意真实高频无害、1.67 Hz 精确结构模型与普适增益边界、移动携带及卸载后稳定性、桶沿完整几何净空、投递与跨 seed 可复现性。
