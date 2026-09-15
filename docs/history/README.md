# 历史工作记录（Task B）

这个目录保存 Task B 开发过程中的**过程性文档**：按日期写的方案说明、状态汇总、外部评审稿、
早期 Runbook 和排障记录。它们的价值是让人看到当时**为什么**那么决定、以及走过哪些错路，
不是结果本身。

**读之前请注意：**

- 这些是**当时的快照**，没有随之后的结果更新。文件里"尚未抓起""投递未验证"之类的说法，
  反映的是它写下时的状态。**当前结论一律以 [`task_b/results/`](../../task_b/results/) 下的
  证据文件为准**；两者冲突时，前者作废。
- 目录里任何数字都不是策略成绩。真正的官方得分只有一次：
  [`task_b/results/first_delivery_video.json`](../../task_b/results/first_delivery_video.json)
  记录的投递探针（真值定位）。
- 文件多为中文长文，其中 `TASK_B_SCORE_ACTIVE.md` 约 70 KB，是当时的主工作日志。

## task_b/ —— 按主题归档的方案与状态记录

| 文件 | 写于 | 内容 |
| --- | --- | --- |
| `TASK_B_SCORE_ACTIVE.md` | 09-14 | 主工作日志，体量最大，记录当时的试验串与判读 |
| `TASK_B_STRATEGY_CONSOLIDATED.md` | 09-15 | 阶段末的策略整合 |
| `TASK_B_STATUS_FOR_EXTERNAL_REVIEW.md` | 09-14 | 给外部评审的状态说明 |
| `TASK_B_CLAUDE_HANDOFF_20260914.md` | 09-14 | 当时的交接稿（已被仓库根目录的现行交接取代） |
| `TASK_B_POSITIVE_RELEASE_CHECKLIST_20260914.md` | 09-14 | 首个正分的发布清单 |
| `YAW_AUTHORITY_BLOCKER.md` | 09-14 | 底盘转向能力不足的专门分析 |
| `WHEEL_TORQUE_WAS_THE_LIMITER.md` | 09-14 | 把限制定位到轮端力矩 |
| `BARREL_OBSERVATION_VERIFIED_AND_ODOMETRY_DRIFT_MEASURED.md` | 09-14 | 桶的视觉观测与里程计漂移实测 |
| `MULTI_OBJECT_DESIGN.md` | 09-14 | 多物体回合的设计取舍 |
| `TRAINING_HANG_DIAGNOSIS.md` | 09-15 | 训练挂起的排查 |
| `task_b_first_pick_plan.md` | 09-10 | 最早的抓取方案 |
| `task_b_bootstrap_audit_notes.md` | 09-10 | 引导阶段审计笔记 |
| `task_b_plan_evidence_review.md` | 09-10 | 方案与证据的复核 |
| `D1G2_TASKB_PLAN.md` | 09-15 | 与 Task A（D1+G2）的联合规划 |
| `TASK_B_GRASP_AND_LIFT.md` | 09-14 | 真实抓起与负载高举的独立审计稿，标题里的「尚未投递」是当时状态 |

## notes/ —— 更早的对话与 Runbook 记录

`ATEC2026_TaskB_*Runbook*.md` 是最早一版的操作手册（多个版本并存，含一个下载产生的重复副本），
`ATEC_TaskB_*.md` 是中文方案与汇总稿。这些比上面那批更早，主要反映最初的规划口径。

## 相关

- 早期"自主完成整回合"路线的源码归档：[`task_b/autonomous_route/`](../../task_b/autonomous_route/)
- 设计所依据的一次性 CPU 推导：[`task_b/derivation/`](../../task_b/derivation/)
- 现行交接与已确证约束：[`task_b/HANDOVER.md`](../../task_b/HANDOVER.md)
