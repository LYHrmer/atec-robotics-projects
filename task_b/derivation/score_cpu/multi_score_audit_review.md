# Task B 多物体接近得分：评测与审计只读复核

日期：2026-09-14。只读生产代码，未修改官方环境、控制器或冻结审计器，未发布 Release。真实已验证回合仍以 `plan_p2_lower02_seed42_01` 的 1 个接近分为当前证据；下述 2—3 分数据全部是明确标记的 CPU 审计自检，不能进入成绩表。

## 最小评测安排

**已有记录器支持多次得分，无需为多分更改奖励、事件结构或审计准则。** 新长回合使用 `--score_hold_steps 0` 即关闭“首分后 N 步停止”，继续受 `max_steps`、官方终止和策略自身 `done_reason` 约束。

原有首分逻辑在 `evaluate.py` 中锁定 `first_positive_step`，之后的新分不会重新起算观察窗口。沿用首分实验的 `--score_hold_steps 100` 会在第一分后 2 秒停止，未必有时间接近第二物体。不要为了延长窗口，把 reward 或“已经计分的物体 ID”传给控制器。

只关评测截断还不够：原 `FirstReachPolicy` 只执行一次固定目标伸臂，降低完成并保持 2 秒后调用 `_finish('reach_lowering_hold_complete')`。多物体控制需要自己的“完成一次有界尝试→恢复/退回→凭公开图像关联下一目标”生命周期；尝试完成不等于获得一分，处理记录也不能冒充官方计分集合。

## 官方规则对多分的约束

- `GraspedObjectsByEE` 为 18 个物体各保留一次性 `_counted`；`gripper_base` 到物体根位置三维距离不超过 0.20 m 时，该物体第一次触发 +1。同一物体离开后重新进入不会再加分。
- `ObjectsInCircle` 有自己独立的一次性集合：中心 (-3,-10)、半径 1 m，物体根 z 在 [0,0.5]。一个物体的接近与入圈是两个独立奖励，同一步可同时发生。
- 多个物体可同一步加分。此时 `scoring_events` 只应有该步的一个事件，`reward_terms` 内的值可以为 2、3 等；事件条数不等于分数。
- 两项奖励配置均为每物体 1 分，纯接近最多 18 分，全部本地原始分最多 36。只有 18 物体**同时**在合法区域才触发 `objects_in_circle_done`；积累接近分不构成通关。
- 原 episode 的 `_counted` 必须连续保留。不能把两次新回合各 1 分累加后称作一个回合的 2 分；官方任一终止后仍须停止。

原始 float32 标量会有换算误差：两次 `.02 / .02` 实际可能累计为 `1.9999999552965164`。本轮只使用 `score_hold_steps=0`，不新增目标分停止。如以后需要评测侧里程碑停止，应使用明确的数值容差或原奖励项的整数累计，而非裸 `score >= 2`；显示整数分与原始累计值仍分别保存。

## 冻结审计器验证

审计器 SHA-256 仍为 `f31f81582633b4fc53157b96e403a1fea4b1e571911a504c2923aeef555904ca`。它逐步重放所有可用动作后物体集合，要求正分事件准确覆盖每个正奖励步，并在跨非终止步时用**下一步动作前状态**验证前一步动作后状态；终止步只接受 reset 前捕获。没有将同索引动作前状态当作得分状态。

[自检汇总](multi_score_audit_fixtures_20260914_v2/summary.json) 中 7 项均符合预期：

| CPU 合成样例 | 预期/实际 | 核验重点 |
| --- | --- | --- |
| 两步分别接近两个对象 | 退出 0；2 events / 2 proximity | 支持 2 个以上事件的循环结构 |
| 同一步接近三个对象 | 退出 0；1 event / 3 proximity | 不把事件数量误作分数 |
| 同一步一个对象接近并入圈 | 退出 0；1 event / 1 proximity + 1 delivery | 两个独立计分集合 |
| 同物体离开后重入，后续奖励仍为零 | 退出 0；累计仅 1 proximity | 一次性计分保持 |
| 第二次得分与最后一步 illegal 同步 | 退出 0；2 proximity，并记录质量限制 | 保留原奖励，终止前状态正确；不谎称完整无失败 |
| 从双得分记录中漏掉第二事件 | 退出 1 | 正奖励步与事件集合不一致 |
| 第二事件使用同索引动作前状态 | 退出 1 | 几何不能解释奖励，且与下一前态不一致 |

fixture 目录和 result 都带 `SYNTHETIC_ONLY.txt` / `fixture_notice`，正分打包器会拒绝它们。第一次 fixture 搭建时修改了 q 却未同步 proprio，被原关节映射检查拒绝；只修正了自检数据，未放宽审计器。原始失败自检留在 `multi_score_audit_fixtures_20260914/`。

补充记录边界：冻结审计器用 NPZ/trace/event 重算分项及唯一对象，当前没有单独将 `result.reward_term_totals_raw` 的冗余摘要与 NPZ 合计再比较一遍。真实 evaluator 的该字段直接由 NPZ 同源数组合计生成；新多分回合复核时顺手检查摘要即可。本轮不因这个冗余字段扩大或修改冻结准则。

## 新 multi_reach 入口的独立只读检查

已阅读 root 新增的 `evaluate.py` diff。新模式尚无完整控制模块，本结论只覆盖入口集成：

- **GT/score 隔离成立。** 构造仅传静态 schema/defaults、dt、CLI 参数；`act` 仍只收公开 proprio 和 RGB-D。`pause_for_stance` 来自使用公开关节的参考轨迹阶段。reward 与真值仍只在动作后记录/评测侧使用。
- **原模式路径保持。** first_reach/reach_probe 使用原 `FirstReachPolicy` 与空额外参数；原类没有 `wheel_hold_requested`，新 `getattr` 的回退仍等于原四状态判断。此次改动未修改奖励计算、官方配置或第一终止停止规则。静态路径一致不替代后续实际控制验收。
- **新模块可自动快照。** `task_b.multi_reach` 显式 import 和构造发生在 `write_source_manifest` 前，后者枚举全部已加载项目 Python；新的顶层模块/依赖自动进入源码快照，审计器也逐项校验。不要在 `act` 第一次运行时才懒导入新控制源码，否则首次 manifest 尚未覆盖该文件。
- **需要同步一处元数据说明。** `brake_wheel_hold_integration.active_states` 目前仍列旧 4 状态，但运行分支已经优先采用 `policy.wheel_hold_requested`。若 UNLOWER/RETRACT 等新阶段请求保持，元数据应说明该布尔优先、旧 states 只是回退，避免描述与实际行为不符。
- **新 CLI 约束已存在。** multi_reach 要求 1—3 次尝试、compact stance + stance hold + brake hold；降低幅度 [0,.03]，NaN/越界不能通过范围条件。
- **旧入口边界尚可前移。** `score_hold_steps<0` 仍静默等效关闭；FirstReach 的 `ramp_calls<=0` 被内部压成 1；compact 的 `settle_calls<20` 到构造后才报错。若要给 CLI 清晰错误，应在启动 AppLauncher 前验证；这些是已有的输入语义/启动成本问题，不是此次多模式引入的真值泄漏。

待新模块完成后的必要回归应聚焦：旧 first_reach 公共观测输入下动作不变；多目标过渡时 wheel_hold_requested 明确控制 engage/release 且不重复乘 gain；完成一次尝试不等于计分；所有新模块都在 manifest 中；实际新回合按冻结审计器检查多事件与唯一物体。无需改原任务，也不需要为多分另造宽松审计器。

**审查后 root 已修复并经只读确认：** 元数据新增 `activation_source` 说明 `policy.wheel_hold_requested` 优先，原四状态另存为 `fallback_active_states`；有布尔覆盖时 `active_states` 为 null。`score_hold_steps<0` 已在 AppLauncher 启动前报错。冻结审计器 SHA 未变。
