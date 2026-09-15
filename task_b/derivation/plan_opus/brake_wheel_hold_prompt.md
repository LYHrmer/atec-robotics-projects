用户指定GPT-6 Astra ultra制定方案，Claude Opus实现明确代码。Astra已根据真实P1失败完成下面合同：现wheel stiffness=0/damping=1，速度指令0不能锁住轮位置，刹停后实际倒退3.6/10.5/11.8cm。只增加公开轮角位置反馈，不改官方执行器。

工作目录 /home/lybm/ATEC_Robotics_Projects_20260910，无.codegraph。
本轮只写两文件：task_b/brake_wheel_hold.py（<200行），task_b/audit_brake_wheel_hold.py（简短CPU审计）。不要读全仓、不要修改其它文件、不要运行shell/GPU/联网/git。直接用Write写这两个文件，主代理随后运行CPU和GPU。先模块，再审计，避免一个巨大输出；最终只报告文件和未运行测试。

实现BrakeWheelHold(dt=.02,kp=2.,kd=.25,max_speed=.6,slew=2.)。
- engage(q4)：保存4个有限公开轮角copy，active=True，last_command=0；active时重复engage不重锚。
- release()：清anchor/active/输出；inactive update返回0。
- update(q4,qdot4)->float：输出四轮同一**物理轮速目标rad/s**。逐轮e_i=atan2(sin(anchor_i-q_i),cos(anchor_i-q_i))，desired=clip(kp*mean(e)-kd*mean(qdot),-.6,.6)，last += clip(desired-last,-slew*dt,+slew*dt)。实际limit使用构造参数。
- 原始轮角可能跳±2π/±4π，必须wrap；不读任何世界位姿、GT、奖励、相机或Isaac。四轮均同号，不使用左右sides。
- shape必须(4,)，每项有限（拒绝嵌套形状）；构造dt,kp,max_speed,slew有限>0，kd有限>=0。
- debug保存wrapped_errors、mean_qdot、raw_desired、command_rad_s、saturated、active，JSON可序列化；describe()返回参数、物理单位、inputs和实验性质，不声称已得分。
- 不依赖schema或wheel_gain，本模块永远物理rad/s；root在evaluate做归一化。只用标准库或numpy现有依赖。

审计可用 `python -m task_b.audit_brake_wheel_hold --output PATH`，标准库unittest或直接assert均可，必须非零退出反映失败，保存JSON count/passed/checks且标为CPU。覆盖：正负位置误差方向；相差±2π/±4π的等效输入一致；cap与每步slew；NaN/shape/参数；engage复制、重复engage不重锁、release/inactive归零。加一个防单位混淆用例：本模块最大.6rad/s；以scale5归一化得到.12，若还在gain8前则应.015，模块内不得乘8。

不是设计任务，公式和接口已冻结。不要再提出另一套控制；直接实现上述小模块。
