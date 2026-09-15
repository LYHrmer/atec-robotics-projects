# Task B 有界轮速与弧线控制候选

这是尚未执行仿真的独立候选。它只改四轮速度动作，逐项保留传入的腿和机械臂动作，不改变官方执行器、物理、资源、奖励或终止条件。CPU 检查不能证明转向、稳定性或得分。

## 实测与可检验假设

已有 `turn_neutral_seed42_01` 请求 ±1.5 rad/s，但末100步实际轮速均值接近零；`turn_seed42_03` 的较大纯转请求出现 FL/RR 与 FR/RL 的不对称响应，最终 RR_thigh 非法接触。`forward_seed42_02` 的四轮 +0.5 rad/s 请求实际约0.19–0.22 rad/s，并产生前进。以上均为单种子实验。

官方 `assets/robots/b2w.py` 配置为 `ImplicitActuatorCfg(stiffness=0,damping=1,effort_limit_sim=20,velocity_limit_sim=50)`。Isaac Lab 的 `actuators/actuator_pd.py::ImplicitActuator.compute` 根据速度误差计算**近似**扭矩；真正的隐式 PhysX 驱动并不由这段显式计算直接施加。平均轮速误差不能证明峰值扭矩、饱和、摩擦或支撑载荷。可检验的假设是：有界速度补偿与前进弧线可能改变此前弱转向的响应。

后续 gain3 与较早在 sin(tilt)=.12 停轮的 pulse 实验均出现 RR_thigh 非法接触。因此本模块的 .12 停轮阈值不是稳定保证，更不能代替腿姿控制；根代理的独立 `stance_hold` 可作为只改腿动作的组件组合。这个新模块本身尚无真实回合。

## 接口与单位

```python
controller = LocomotionController(schema, observation_joint_names, dt=.02)
action = controller.apply(incoming_action, proprio,
                          bearing_error=target_bearing, enabled=True)
```

`schema` 来自实际 ActionManager；关节名必须来自已解析的 ObservationManager 配置，而不是原始 `task.cfg`。输入是完整24维动作和公开84维 proprio。控制只使用 `[3:6]` 角速度、`[9:12]` 重力、按名字解析的 `[36:60]` 关节速度。`[6:9]` 是速度指令，不是加速度。原任务默认轮关节速度为0；本实现显式拒绝其他默认值。没有世界位置、物体真值或接触力输入。

- `bearing_error=None`：限幅和变化率限制后的轮动作透传，不增加反馈；这条模式允许原地反转轮请求。
- 提供 body-frame bearing：用传入四轮请求的平均值作为名义前进速度，弃用其原差速，生成有界偏航反馈。角度正方向是 body +Y；`turn_sign` 明确可配置，不能从漂移自动确认符号。
- 四轮请求全零、弧线模式平均请求为零、`enabled=False` 或姿态保护触发：当步轮动作归零并清补偿记忆。腿和臂保持传入值。
- 默认每轮物理目标上限3 rad/s（原 scale=5时为动作0.6），可显式配置但模块硬上限为10 rad/s；这都是候选的试验界限，不是官方限值或已验证安全范围。只提高 `max_wheel_rad_s` 不会自动提高名义请求、common/diff/assist上限。

## 反馈和边界

前进积分针对**名义请求轮速减实际轮速**，而不是已补偿的电机目标。因此实际轮速超过请求时会撤回补偿。偏航反馈使用目标 bearing 对应的角速度与公开 gyro；两类积分均有限幅、停止清零和无响应时的衰减。

弧线模式要求 `|diff| ≤ .8 |common|`。对四轮使用同一个插值比例保持这个约束和每轮变化率上限；从透传模式进入、或改变前进符号时，先清掉旧轮请求，避免把旧纯转带入弧线。清零边界允许立即制动。反转或对向滚动受到命令层面的限制，不表示真实轮子不会倒滑，也不表示底盘稳定。

原始动作在任何 gate 前先检查有限性及 float32 可表示范围。默认原始重力 sin(tilt)≥.12 或 roll/pitch 角速度范数≥1.5 rad/s 时停轮；非法重力、倒置及错误观测顺序不能驱动车轮。JSON 区分请求、前次发出目标和实际滤波轮速，记录跟踪不足及对角不对称。跟踪计数是带增减滞回的计数，不是“严格连续N帧”。

本模块把自己的上次输出视为实际上次轮目标，因此实测探针应由它唯一拥有轮命令。若下游还有限速器或另一个轮控制器，记录与反积分需要先对齐最终动作；不要盲叠既有 `stability.py` 的低轮速上限。只改腿的 `stance_hold` 不受这个限制。

## 建议第一轮试验

由根代理在原任务中串行测试500步：同种子、同站姿保持、相同100步静止期，一条仅前进且 bearing=0，另一条给小固定 bearing（例如0.15 rad）。先使用默认驱动上限。必须记录原始请求、最终动作、四轮实际速度、公开 gyro/重力、阶段及官方终止；真值仅在事后计算位移和偏航。

若轮速执行改善但 gyro/yaw 不响应，不能继续把问题归因于速度跟踪；若姿态先恶化或目标更快移出视野，应撤销该参数配置。更大驱动只作为额外的短试验，不根据CPU有界输出宣称安全。

## 验证和贡献

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' "$ATEC_PYTHON" task_b/audit_locomotion.py \
  --output results/task_b_locomotion_cpu_audit.json
```

35项CPU检查通过，包括独立打乱 articulation/observation/action 顺序、非默认轮动作scale、非wheel动作逐位保留、零轮即时停车、无效输入、倒置、名义速度超调撤回补偿、1500次受阻有界积分、600次变化弧线的曲率及变化率边界、gyro反馈符号、旧纯转退出和JSON有限性。另一次独立审阅发现有限float64极值会污染EMA，现已在滤波前拒绝超过float32范围的gyro/轮速，并验证严格模式拒绝、非严格模式停车及后续正常输入恢复。实际 `visual_neutral_seed42_01` 的1500帧公开 qdot 与诊断 qdot 按名字重排误差为0，上一动作字段误差也为0；诊断数组不进入控制器。

Claude Opus 真实CLI同一会话 `5dd69995-f42f-4c09-a02e-4ccb322d7283` 提供了原始模块。首次调用由编排代理误中断，随后原会话恢复，在写出模块后遇到账户当日额度错误，未交付测试和设计文档。Codex保留原稿SHA `529760fed60c67637858a3283a187da2f5cab9f7b3236d1202d7e038865c60c8`，修复上述反馈/切换/输入边界并完成本文件及CPU检查；没有把部分交付称作Claude完整成功或仿真验证。
