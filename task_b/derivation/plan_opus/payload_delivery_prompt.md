# 合同：`task_b/payload_delivery.py`（首次真实投递的导航/停靠/释放薄包装）

本文件是**实现契约**，不是结果。它只描述要写的模块必须做什么、不得做什么。任何"能跑通"
的说法都必须来自真实回合的原始事件，不能来自本文件。

## 0. 背景：现在缺的是哪一段

已由真实回合证实（见 `docs/TASK_B_GRASP_AND_LIFT.md`）：

- 真实抓起（整 mesh 最低面 +42.9 mm，跟随 2.88 s）；
- 负载高举（+619.9 mm，世界 z≈0.629 m > 桶沿 z≈0.55 m）；
- 搬运姿态全链 A→B→C→HOLD（44.5 s 持续持有）。

**完全未实现**：持载状态下导航到桶、桶外侧方停靠、越过桶沿释放。
官方 `objects_in_circle` 在所有回合中恒为 0.0。本模块就是补这一段。

## 1. 硬约束（违反即作废）

- 策略输入**只允许**：公开 84 维 proprio、公开 RGB-D、原场景静态常量、机器人静态几何。
- **禁止**进入策略：GT 物体位姿、接触力、世界底盘位姿真值、奖励、分数、seed 地图、
  `objects_in_circle` 事件本身。这些只能用于**事后独立审计**。
- **不得**修改原物理、资产、奖励项、终止条件、原关节限位、驱动 K/D。
- **不得**修改下列冻结模块（SHA 已核验）：
  `task_b/grasp_probe.py`、`task_b/first_reach.py`、`task_b/public_odometry.py`。
- 失败必须如实上报为失败，**不得**重分类成功，**不得**放宽任何验收阈值。
- 单张 8 GB GPU **严格串行**；代码与 CPU 审计可并行。

## 2. 组合方式（不要复制粘贴已有逻辑）

本模块是一个**薄包装**，按以下方式组合：

```
PayloadDeliveryPolicy
  └─ PayloadMotionPolicy          (task_b/payload_motion.py，本人所写，可增量修改)
       └─ ContactGraspPolicy      (task_b/contact_grasp.py)
            └─ FirstReachPolicy   (task_b/first_reach.py，冻结)
```

- 子策略每 tick **恰好调用一次**，其返回的 action **逐字节原样保留**（包括它正常结束的那一 tick）。
- 子策略正常结束后**永不再调用**（避免已完成的 25 s / 12 s 时钟误杀后续阶段）。
- 手臂目标值：复用 `PayloadJointTracker` 与 `build_payload_goals`，不要重写；
- 腿部：复用子策略 action 里的腿切片，**逐字节复制**，不要重建；
- 轮子：本模块**唯一**有权写入轮切片（见 §6）。

### 2.1 必须对 `payload_motion.py` 做的增量修改（默认值必须保持 p13 行为）

全部为**加法式**，默认参数下 p13 的行为必须逐字节不变：

1. 构造函数新增 `hold_after_raise=False`。
   为 `True` 时，`PAYLOAD_RAISE` 到位后**不**自动进入 `PAYLOAD_SWING`，而是进入新公开相位
   `PAYLOAD_TRANSIT_HOLD`：以 `tracker.update(q, qdot, advance_path=False, paused=False)`
   继续闭环保持 A 姿态（这是"持载原地不动"，不是暂停）。
2. 新增公开方法 `release_transit_hold()`：只允许从 `PAYLOAD_TRANSIT_HOLD` 调用一次，
   之后恢复原有 A→B→C 前进逻辑。非法相位调用必须抛异常，不得静默忽略。
3. 新增属性 `wheel_authority_external`（bool）。为 `True` 时
   `wheel_hold_requested` 必须返回 `False`（让出轮锚给本模块）；
   为 `False` 时语义与现在完全一致。
4. `describe()` 必须如实列出上面三项，并写明默认关闭。
5. 新增相位名必须同时加入 `PAYLOAD_PHASES` 与 `_record()` 的 debug 输出。

## 3. 公开相位顺序（原合同）

```
PREFIX -> PAYLOAD_RAISE -> PAYLOAD_TRANSIT_HOLD
       -> MOVE_PRE_DOCK -> BRAKE_PRE_DOCK -> TURN_TANGENT
       -> SIDE_DOCK     -> BRAKE_SIDE_DOCK
       -> PAYLOAD_SWING -> PAYLOAD_EXTEND -> PAYLOAD_HOLD
       -> OVER_RIM -> OPEN -> RELEASE_OBSERVE -> DONE
```

注：原合同把 `OVER_RIM` 排在 `SIDE_DOCK` 之后、`OPEN` 之前。本模块把侧摆/越沿伸展
（`PAYLOAD_SWING`/`PAYLOAD_EXTEND`）放在停靠**之后**执行——即"先停好再侧伸越沿"，
这与原合同"身体留在桶外，先抬高再侧伸越沿释放"的意图一致。必须在 `describe()` 里
如实写清这一处与旧文本的次序差异及理由。

## 4. 停靠几何（只用公开积分位姿与静态常量）

桶心 `C = (-3, -10)`（原任务公开静态常量），半径 1 m，墙厚 .02，墙高 .5，桶沿 z≈.55。

每个控制 tick 用 `task_b.public_odometry.PublicPlanarOdometry(dt=.02).update(proprio84)`
取得估计平面位姿 `p`（固定公开 spawn (-10,-10,yaw0)，**禁止** GT 校准）。

```
n     = (p - C) / ||p - C||          # 桶心指向机器人的单位向量
t     = R90 * n = (-n_y, n_x)        # 桶的切向
p_d   = C + standoff * n             # 侧停点（standoff 默认 1.45 m）
p_pre = p_d - 1.0 * t                # 预停点
```

行进意图：先到 `p_pre` → 制动 → 转到朝向 `t` → 沿 `t` 直行约 1 m 抵达 `p_d`，
此时**桶在 body +Y 侧**。必须先在外侧转向，避免贴桶原地转时前后轮扫入墙。

本模块必须把 `standoff` 做成构造参数（默认 `1.45`），因为桶沿净空正在用真实抓持姿态
重算；净空结果可能反过来调整该值。

## 5. 速度、速率与朝向限制（全部为硬上限，不得放宽）

- 目标平面速度：`.04`–`.05` m/s（构造参数，默认 `.045`）
- 平面速度**硬上限** `.08` m/s
- yaw 速率**硬上限** `.08` rad/s
- 轮命令 slew：复用 `first_reach.py` 的既有约定
  （`previous_wheels += clip(request - previous_wheels, -dt*.5, +dt*.5)`）
- 轮子归一化命令到动作：复用已验证 schema 映射
  （`common` 加在右侧、`common` 减在左侧；`wheel_side()` 判左右）。
  **符号已由真实回合实测确认**：正 `common` → 正 body-x 速度（前进）。
  见 §7 的启动方向自检。

## 6. 轮锚（wheel hold）语义——最容易出错的地方

原记录缺陷：`multi_reach` 在底盘仍以 .235 m/s 滑行时就建立 .03 m 静止预算，立刻自停。

本模块必须：

1. `PREFIX` / `PAYLOAD_RAISE` / `PAYLOAD_TRANSIT_HOLD` 全程**保持原轮锚**（轮命令为 0，
   `wheel_hold_requested` 沿用子策略/载荷模块语义）。
2. **只有真正开始发出行走动作的那一刻**才把 `wheel_authority_external = True` 并释放轮锚。
3. 每段行进结束后**必须重新制动**：先把轮请求斜坡降到 0，等到公开平面速度
   `< .01 m/s` **且** 持续 `.5 s` 之后，才允许建立该段的静止/到位预算。
   **绝不允许**在仍明显滑行时起算稳定窗口。
4. `TURN_TANGENT` 段同样是行进段，同样适用上面的制动规则。

## 7. 方向自检（把一次可能的废回合变成诊断）

公开里程计有约 17–20 mm 的两回合离线误差，**不是**精度保证。因此：

- 首次发出非零 `common` 后，累计公开位移达到 `0.05 m` 时，检查该位移在**期望前进方向**
  上的投影是否为正。
- 若为负或接近零 → 立刻按"正常停止原因"停下，reason 用 `payload_delivery_direction_fault`，
  并如实记录。**不得**在运行中自动翻转符号继续跑（那会把符号问题藏起来）。

## 8. 分段预算与看门狗（每段独立，不得照搬静止模块预算）

每段都要有：

- 有限**路程**预算（该段期望距离的 1.5 倍 + 0.3 m 余量）
- 有限**时间**预算（由路程 / 最小速度推出，再乘 2 倍余量）
- **无进展**判据：`PROGRESS_WINDOW_S` 内公开位移 < `PROGRESS_M` 即停

任一触发即以对应的确定 reason 正常停止（例如
`payload_delivery_pre_dock_not_reached` / `payload_delivery_side_dock_not_reached` /
`payload_delivery_no_progress`）。**超时不得当作到位。**

整回合预算 300 s（原任务 1200 s 上限不改动）。

## 9. 释放语义

`OVER_RIM` 到位后进入 `OPEN`：

- 保持实际到位姿态与轮锚；
- 以**有限 finger slew** 把 `arm_joint7/joint8` 张到 `.07 m`（复用 `gripper_targets`，
  不得改原关节限位）；
- 张开后原地观察 **≥ 3 s**（`RELEASE_OBSERVE`）。

外部审计必须关联**此前夹持的同一物体**，确认原 `objects_in_circle` 事件、物体掉入实体桶
且在圈内保持。**模块内部不得读取该事件来驱动状态转换。**

## 10. 必须提供的公开接口

```python
class PayloadDeliveryPolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 standoff=1.45, transit_speed=.045, speed_cap=.08, yaw_cap=.08,
                 target_open_m=.07, observe_s=3., episode_budget_s=300.,
                 hold_after_raise=True, **child_kwargs): ...

    def act(self, proprio, images): -> np.ndarray   # (24,) float32
    def describe(self) -> dict
    # 公开只读属性：phase, state, done_reason, debug, wheel_hold_requested, pause_for_stance
```

`debug` 至少包含：
`phase`、`phase_age_s`、公开积分位姿 `odom_xy`/`odom_yaw`、`n`/`t`、`p_d`/`p_pre`、
到段目标的距离、`speed`/`speed_cap`/`yaw_rate`/`yaw_cap`、
`wheel_request`/`wheel_action`、`wheel_authority_external`、
各段路程/时间/无进展预算与用量、`direction_check` 结果、
`finger_command_m`、`release_observe_s`。

`describe()` 必须包含一句明确的证据声明：本模块只做序列与运动记账，
**没有任何**物体位姿/接触力/世界底盘真值/奖励/分数输入，
因此其输出**不是**抓起、搬运、净空、投递或得分的证据。

## 11. 代码风格

- 中文注释不必要；与既有模块一致：**英文 docstring，说明"为什么"，不写"是什么"**。
- 不要写"这一步是为了通过测试"之类的注释。
- 模块顶部 docstring 必须包含：为什么需要新候选、已记录的具体失败、已知风险，以及
  "本文件是候选与假设，不是已验证结果"的声明。
- 不要 import 未使用的模块；不要留下 TODO。

## 12. 完成标准

- `python -c "import task_b.payload_delivery"` 通过；
- 用 `--help` 风格的极简 smoke：构造一个**合成 schema**（不要真 GPU）调用
  `act()` 若干 tick，确认不抛异常且 action 形状/范围合法；
- `hold_after_raise=False` 时 `PayloadMotionPolicy` 行为与修改前逐字节一致
  （用 p13 的记录 action 序列回归验证）。
- 把上面三项的实际输出写入交付说明，不要只写"应该没问题"。
