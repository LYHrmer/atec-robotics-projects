D1+G2 Task A 本地完整赛道通过证据

最终仅触发 reach_goal_x：从 x=-141.0 到 x=145.0023498535，前进286.0023498535米；505.66秒仿真/1007.876秒实际用时，低于原1200秒任务预算。未触发同帧illegal_contact、fall或time_out。

1999 residual和基础ONNX、全部13份运行模块与已通过的末阶诊断代码匹配；27份原框架依赖及11个USD层重新核验。终止前状态由原runner的pre-reset捕获保存。完整trace以gzip存放，原始trace SHA在run_acceptance_audit.json和manifest.json中。

恢复发生2次，单次失联0.4/0.5秒，总0.9秒、0.484383米，均在既定预算内；明确属于inertial_reanchor，继承惯性漂移，不声称跨间隔纯视觉重定位。

GPU采样峰值6210MiB；进程RAM高水位7731.03125MiB（约7.55GiB）。温度/全机剩余RAM的stdout统计来源见hardware_audit.json。

这是一次原始规则下的本地完整赛道成功，使用自定义D1+G2机器人，未作线上提交。此前失败运行另有保留。
