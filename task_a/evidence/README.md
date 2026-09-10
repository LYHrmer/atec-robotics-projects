# 实验记录

| 实验 | 范围 | 结果 |
| --- | --- | --- |
| [完整恢复方案第 2 次](full_course_02/result.json) | 原始起点、完整赛道 | 505.66 s / 286.002 m，唯一终止 reach_goal_x，通关 |
| [完整恢复方案第 1 次](full_course_01_failed/result.json) | 原始起点、完整赛道 | 69.30 s 碎石路非法接触失败 |
| [原视觉方案](axis_course_01_stalled/result.json) | 原始起点、完整赛道 | 最远 269.410 m，最后台阶局部停滞看门狗结束 |
| [最后台阶原视觉](last_stairs_camera_baseline_01/result.json) | 109→131 m 诊断 | 最远 x128.404，停滞 |
| [最后台阶恢复初版](last_stairs_camera_recovery_01/interruption.json) | 109→131 m 诊断 | 候选参考失联，预算停止；人工中断以测试修正，原 result 保留了 running 状态 |
| [最后台阶恢复修正](last_stairs_camera_recovery_02/result.json) | 109→131 m 诊断 | 48.14 s 抵达 x131.0066；不是全程通过 |
| [独立交付目录启动检查](package_smoke/result.json) | 原起点 100 控制步 | 正常结束于 max_steps；不是完整通关 |

`full_course_02/trace.jsonl` 保留完整运行的定期诊断及终止前状态；其中世界真值用于事后核验，不作为 actor 或导航输入。
原始 10 Hz RGB-D/本体传感器 NPZ 在用户本机日志目录保留，因体积较大未上传。
完整视频通过 Release 附件交付，文件摘要见 `acceptance_summary.json`。
GitHub 来源与模型文件见根 README、THIRD_PARTY_NOTICES 和 SHA256SUMS。
