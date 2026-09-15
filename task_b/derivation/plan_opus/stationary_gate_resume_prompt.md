恢复上一轮任务。上一轮已读接口，但供应商连接中途断开，两个目标文件均未写入。不要重新调查项目；Astra方案与前面接口仍有效。

为了避免一次生成太长，这一轮只直接 Write 完整的 task_b/stationary_target_gate.py，审计脚本下一轮再做（已有独立集成审计等你的模块）。你只写这一文件，不改其它文件，不再Read其它控制器。实现控制在约150–220行，不堆长注释，不展开总方案。用户明确让Opus负责具体代码，主代理正等此模块启动实测。

保留此前所有接口：StationaryTargetGate(dt=.02,sensor_period=.1)，reset(step)，update(step,quiet,point_body=None,source=None,frame_token=None)，始终返回ready/target_body/confirmations/last_fresh_step/age_s/reason/freshness_basis。quiet连续.2秒后等一完整采样周期；同源两有效样本间隔>=.1且位置差<=.04才ready；同token不重数；无token明确sensor_period_elapsed_assumption；没有候选不刷新计数，>.5秒失效需重建；quiet失效清除、source切换重建、点shape3且finite、重复step不计数、倒退报错、内外数组不别名。point=None是普通非检测tick，不要每次都重置两帧确认。无像素变化要求，不做IK/动作或GT。

写完后只报告文件路径和未执行测试的事实。本轮到此结束。
