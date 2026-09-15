你是实际 Claude Opus，被用户授权协助 ATEC 机器人项目。请承担两个有边界、可交付的设计任务。当前只写具体方案，不执行仿真，不写代码，不改文件，不发布 GitHub。只允许 Read、Glob、Grep 阅读指定项目与证据。不要读任何凭据或无关私密文件。用中文给出能直接整合进执行方案的设计，避免泛泛评审。

项目：/home/lybm/ATEC_Robotics_Projects_20260910
实验：/home/lybm/ATEC_Experiments_20260910/task_b_score
CPU 几何证据：/home/lybm/ATEC_Experiments_20260910/task_b_score_cpu
进度记录：/home/lybm/ATEC_Experiments_20260910/TASK_B_SCORE_ACTIVE.md（记录中的运行 PID 已因机器重启失效，不代表当前有运行）。

必须先读当前 task_b/first_reach.py、task_b/stance_hold.py、task_b/stance_reference.py，及 task_b/evaluate.py 的关键评分、输入、状态记录部分。读取 task_b_score 下 11 个 result.json，形成现有证据的客观认识。最新 first_reach_creep_seed42_01 没有 result.json，是重启中断，不可当成方法成功/失败。Task B 所有新完整实验仍为 0 分，从未在真实物体视觉闭环中完成进入 REACH 后的验证。唯一 reach_probe 为合成身体坐标目标，验证 FK 与真实位置一致，不是物体得分。

关键约束：Task B 是 B2wPiper；只可用公开 proprio obs 和相机 RGBD 做控制。诊断可读的真实 robot root/object pose/contact 不能输入策略。原始环境评分、终止、模型和物理不许修改。每轮新进程，因 env.reset 不重置 root pose。一个 GPU 仿真串行，不允许并发。官方每件首次 gripper_base 与物体根三维距离小于等于约 0.20m 计 proximity +1；这不等于真实抓取。每件首次进入收集区域另 +1；18 件同时位于合法区域才是最终通关。必须核对源码的实际严格比较号，不能以此提示替代源码。零分视频仅本地，正分及可验证结果才作为新 Task B 视频发布。

任务 A：设计第一分的最短可信闭环。重点检查当前 compact + stance_hold + wheel gain8、远距离接近、近距离 creep、锁定腕部相机停稳、两帧新 RGBD、位置 IK、至多 3cm 缓降的方案。指出三处最可能失败点，哪些控制/记录修改是下一次正式实验前必需，哪些仅失败后再做。逐阶段给进入条件、动作、退出条件、超时或失败判据，并给失败后的单变量调整；不要只列风险。明确当前近距离 creep 和 3cm 缓降尚未完整验证。不要假定降低机身会必然成功；结合实际 compact base 高度/俯仰变化、可达目标窗口、姿态反馈与惯性回退。优先复用当前代码，不扩建复杂框架。

任务 B：设计从第一分到真实抓起、单件投递、18 件通关的分阶段路线。请读项目 solution_task_e_il_cartesian.py、task_e_geometry.py、task_e_perception.py、tools/task_e/il/README.md 的相关部分，明确哪些 Task E 机制可复用、哪些权重/运动空间/夹爪动作不能直接迁移。每阶段写可测验收条件、失败分类、需要新增的数据/模块、Claude 与 Codex 的可并行分工（Claude 可以负责关键设计和有独占文件边界的实现，不应仅做审查；Codex 统合和唯一 GPU 调度，独立审计证据）。不要宣称现有 Task E 的残差 IL 已覆盖 Task B，也不要第一步就全量 RL。

请输出：1) 你实际核查的文件/实验与未核查项；2) 任务 A 的具体状态机和实验门槛，含优先级明确的必要修正；3) 任务 B 的分阶段工程方案与 Task E 复用边界；4) 明确的并行任务分工、输入输出接口与验收；5) 需要由主代理裁决的最多三个真实不确定点。控制篇幅，着重可操作性，不提供未经执行的结果。不得运行 shell/GPU，不得编辑文件。最终文字由外层调度保存。
