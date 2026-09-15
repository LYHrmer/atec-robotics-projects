# carry_probe 构建与 CPU 核验记录（P0）

生成：2026-09-14。按 `task_b_astra_review_20260914/next_delivery_plan_astra_2006.md` 的 P0 执行。
**本轮没有运行任何 GPU 实验。** 下面是实际命令的实际输出，不是预期结果。

---

## 1. 本轮新增/改动

| 文件 | 归属 | 内容 |
|---|---|---|
| `task_b/delivery.py` | 主执行（本机） | `PayloadRaisePrefix` 薄子类 + `PayloadDeliveryPolicy` 顶层状态机 |
| `task_b/evaluate.py` | 主执行（本机） | 新增 `--mode carry_probe` 与四个探针参数 |
| `task_b/carry_drive.py` | 并行 agent | 公开 twist 低速/偏航跟踪，输出物理轮 rad/s |
| `task_b/bucket_observation.py` | 并行 agent | 公共 RGB-D 桶壁/桶沿观测 |

**未改动**：`first_reach.py`、`grasp_probe.py`、`contact_grasp.py`、`public_odometry.py` 四个冻结模块，
原物理、资产、奖励、终止条件、关节限位、驱动 K/D。`payload_motion.py` 本轮**零改动**——
交接用的是继承而非修改。

## 2. A 到位公开交接

`PayloadRaisePrefix(PayloadMotionPolicy)` 只加一个分支：在原 `_advance_segment('A')` 里发布
`handoff_ready` 并返回该 tick 的原 A 动作，**不调用 `tracker.begin(B)`**。`take_carry_state()`
交出**同一个** `PayloadJointTracker` 实例（不是副本），所以积分、滤波、标量路径和已加载的
执行器命令全部连续。交接后顶层永久停止调用前缀子策略。

`carry_probe` 的相位链（实测，来自 CPU 冒烟）：

```
CARRY_READY → PROBE_DRIVE → PROBE_BRAKE_DRIVE → PROBE_YAW_POS → PROBE_BRAKE_YAW_POS
→ PROBE_YAW_NEG → PROBE_BRAKE_YAW_NEG → PROBE_HOLD → STOPPED
```

## 3. 轮命令单位

合同要求只在一个边界换算。这里：drive 层输出**物理 rad/s**，`delivery.py` 在装动作时
**除以 `gain × scale` = 8 × 5 = 40**，evaluator 再按原流程乘回 gain 8，schema 再乘 scale 5。

已对该链路做了实测往返校验：927 个移动 tick，**0 处不一致**。

## 4. 轮锚生命周期（本轮最重要的实现约束）

evaluator 在 `wheel_hold_requested` 为真时会用 `BrakeWheelHold` **整个覆盖轮切片**。
所以移动相位必须让轮锚释放，否则所有导航请求会被静默丢弃。

本实现：`_begin()` 进入移动相位时释放；**engaged 从不自动发生**，只在 `_brake_then()` 里
实测速度与完整静窗同时满足后由 `_engage_anchor()` 取**新**锚（在真实到达的轮角上）。
刹车阶段全程保持释放、自己发零请求——正是为了不重演 `multi_reach` 在 .235 m/s 滑行时
就锁死静止预算的缺陷。

实测：3 次刹车 → 3 个新锚（call 453 / 704 / 955），无一是滑行中取的。

## 5. CPU 核验的实际输出

### 5.1 交接分支回归 —— `task_b_score_cpu/delivery_handoff_check.py`

```
passed: True   failed: []
  the_subclass_defines_only_the_intended_members       PASS
  the_subclass_is_a_payload_motion_policy              PASS
  armed_handoff_holds_the_raise_phase                  PASS
  armed_handoff_publishes_ready                        PASS
  armed_handoff_does_not_begin_the_swing               PASS
  armed_handoff_reports_no_success_stop                PASS
  armed_handoff_is_idempotent                          PASS
  carry_state_transfers_the_same_tracker_instance      PASS
  carry_state_carries_the_public_provenance_and_goals  PASS
  carry_state_keeps_the_full_leg_template_untouched    PASS
  carry_state_can_only_be_taken_once                   PASS
  disarmed_handoff_still_begins_the_swing              PASS
  disarmed_handoff_never_publishes_ready               PASS
```

### 5.2 相位机合成冒烟 —— `task_b_score_cpu/delivery_probe_smoke.py`

```
passed: True   failed: []
phase_sequence: CARRY_READY → PROBE_DRIVE → PROBE_BRAKE_DRIVE → PROBE_YAW_POS
                → PROBE_BRAKE_YAW_POS → PROBE_YAW_NEG → PROBE_BRAKE_YAW_NEG
                → PROBE_HOLD → STOPPED
final_reason: delivery_carry_probe_complete
  the_probe_walks_the_intended_phase_order                 PASS
  the_probe_ends_on_its_normal_reason                      PASS
  the_base_actually_moved_in_the_synthetic_plant           PASS
  the_anchor_was_released_for_every_movement_tick          PASS
  no_new_anchor_is_taken_while_still_sliding               PASS
  anchors_are_engaged_only_at_settles                      PASS
  the_wheel_unit_boundary_round_trips_exactly_once         PASS
  a_blocked_base_stops_on_no_progress_not_a_timeout        PASS
  a_non_finite_observation_stops_cleanly                   PASS
  an_overspeed_reading_stops_the_run                       PASS
  a_weak_yaw_plant_surfaces_as_an_honest_diagnostic        PASS
```

## 6. 这些检查抓到的四个真实缺陷

都是**先失败、修好、再通过**，不是事后补写的：

1. **刹车后忘了重新释放轮锚。** 侧偏探针启动时锚仍是 engage，evaluator 会覆盖轮切片，
   转向请求全被丢弃 → 表现为"转向没进展"。这正是合同预言的陷阱。
2. **锚在刹车一开始就被取走**——即 `multi_reach` 那个缺陷的翻版。改为两段式：先零请求
   滑到停稳，确认后才取新锚。
3. **静稳计时用了 deque 长度**，而窗口上限就是 26 个采样（0.5 s），所以任何超过 0.5 s 的
   保持**永远无法达成**。改为在静稳行程开始时**锁存**起点 call，按端点时间差计时。
4. **诊断上报了上一 tick 的轮请求**（相位切换那 81 个 tick），使"单位往返"检查假失败。
   改为由 `_action` 单一来源记录真正发出的值。

第 3 条正是合同点名的那类错误（"原 `hold_quiet_calls * dt` 不能代替端点持续时间"）。

## 7. 明确没有验证的事

- **没有跑过真实机器人。** 合成冒烟的 plant 是虚构的一阶模型，其常数与 B2wPiper 无关，
  不构成任何机动性结论。
- `carry_drive.py` 与 `bucket_observation.py` 由并行 agent 交付，我**只验证了接口对接**
  （签名、返回键、形状、物理单位与符号方向），**没有独立复核它们的内部逻辑**，
  也没有验证它们在真实物理下的响应。按合同应由独立审查者另出报告。
- 桶视觉是否可用，取决于那两个模块对 p13 已存 RGB-D 的检查结论；本轮未复核。
- 探针参数（.20 m / .03 m/s / ±10° / 60 s）是**新试验初值**，不是已证明能力。

## 8. 下一步

用与 p13 相同的抓取前缀参数启动第一轮真实持载探针，新目录
`task_b_score/plan_d1_carry_probe_seed42_01`。通过判据见合同 P1。
