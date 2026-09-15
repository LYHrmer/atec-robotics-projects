# Task B bootstrap：独立 CPU 审查前置结论

2026-09-10；只读原工程与已安装 IsaacLab 源码，尚未运行新 bootstrap，也未修改 Opus 文件。以下是实际代码行为，不是从基类名称推测。

## 1. 实际 Task B 环境确实在 step 内自动 reset

- 当前 isolated Python 的 `find_spec("atec_rl_lab").origin` 指向 `/home/lybm/ATEC2026_Simulation_Challenge/source/atec_rl_lab/atec_rl_lab/__init__.py`。
- 该包 `tasks/task_b/__init__.py:48-54` 将 `ATEC-TaskB-B2wPiper` 注册到 `atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv`。
- `tasks/task_base/envs_base.py:6-16` 的确覆盖了 `step`，但第10行直接调用 `super().step(action)`，随后只添加 `Elapsed_Time`、`Step_dt`、`Episode_Length_s`，没有绕过、取消或替换自动 reset。
- 本机 `/home/lybm/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:204-208` 计算终止与奖励，`:216-221` 对终止环境调用 `_reset_idx(reset_env_ids)`，`:238-241` 才重算并返回 reset 后的观测。

因此：首次 `terminated/truncated=True` 的分数和布尔标志仍属于刚结束回合，但 step 返回后的 root/q/contact/RGB 可能已经属于 reset 后状态。不能把这些值不加标记地称为终止现场，尤其不能据此解释“机身瞬间跳到某高度”或判断非法接触链。

建议在独立评测器中为 **该环境实例** 包装 `_reset_idx`，仅作 pre-reset 诊断：

1. 保留原 bound method；包装函数先复制诊断，再原样调用并返回原 method(env_ids)，不改变 env_ids、调用次数、物理、reward、termination 或 reset。
2. 用 evaluator 的 `inside_env_step` 标记区分初始显式 `env.reset(seed)` 与 step 内自动 reset；每次 env.step 前清空本步 snapshot。不能把初始 reset 当一次失败。
3. snapshot 含控制步号、env_ids、joint q/qdot、root xyz/quat、物体位置、active termination flags、reward total/terms、接触历史；所有 tensor 都脱离可变缓冲区并复制。
4. 在有终止的 trace 行优先保存 `terminal_pre_reset`，同时将 step 返回后的诊断明确标为 `returned_after_auto_reset`。如果 observer 失败，则标记诊断缺失，仍调用原 reset；不能偷偷保留物理状态。
5. 不额外调用 `observation_manager.compute`、`scene.update`、step 或 reset。RGB只能标成“最后已渲染帧”；不能要求 observer 内 render 去制造另一时刻图像。
6. observer 的全部对象/world 信息仅进入日志，不进入 policy 或动作选择；结束后恢复原方法。
7. 首次终止包括settling的第1步即停止；不忽略warmup、不继续第二回合。

## 2. 非法接触必须看历史最大力

`/home/lybm/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:154-162` 的 `illegal_contact` 使用 `net_forces_w_history`，先取各body跨history的力范数最大，再与threshold比较。只存最后一帧 `net_forces_w` 可能漏掉造成终止的历史峰值。

Task B B2w配置监控 base、`.*_hip`、`.*_thigh`，原阈值1 N。记录实际解析的body ids/names；诊断预警阈值可以独立列出，但不能把旧worker的0.01 N诊断阈值当作官方非法接触阈值。`fall.minimum_height` 实际为0.0 m，没有0.24 m终止线。

## 3. 总reward除dt一次；分项已经除过dt

`/home/lybm/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py`：

- `:150`：`value = term_func(...) * weight * dt`；`:152` 加到返回总reward。
- `:157`：`_step_reward[:, term_idx] = value / dt`。因此该缓冲区或 `get_active_iterable_terms` 的分项已是每控制步原始计分增量，**不能再除dt**。
- `:119-120`：`Episode_Reward/...` 是累计dt-scaled reward再除 `max_episode_length_s`，不是原始任务总分，不能直接当分数。

正确例子：dt=.02、当前步接近1个物体，则返回reward=.02，原始总分增量1；分项 `_step_reward` 已是1。前一步都0、下一步再接近同一个物体为0。Task B只有两个weight=1计数项，记录的分项之和应与 returned_reward/dt 一致。

## 4. 动作顺序和默认偏置审查准则

- 实际动作扁平顺序来自 `action_manager.active_terms` / `_terms` 插入顺序，再接每项实际解析的 `_joint_names`；不能取USD遍历顺序、robot.data.joint_names顺序，也不能取 `ACTION_TERM_NAMES` 字典的leg/arm/wheel顺序。
- 原 ActionsCfg 声明是joint_leg、joint_wheel、joint_arm；本B2w预期12+4+8=24。轮关节名FR、FL、RR、RL，但评测器仍应按运行时manager核实。
- 本机 `.../envs/mdp/actions/joint_actions.py:169-177` 执行 `processed = raw * _scale + _offset`（然后clip）。`:194-195` 位置项在use_default_offset时使用实际 `default_joint_pos[:,_joint_ids]`；`:245`起速度项使用默认速度偏置。
- 当前实现没有公开 `JointAction.joint_names` 属性，运行时可验证后读 `_joint_names` / `_joint_ids`，或使用IO描述符；不要假定不存在的property。
- crouch希望q目标hip0/thigh1.0/calf−2.0时必须先用该joint真实限位约束，再反解 `(desired_q-offset)/scale`；arm/gripper保持官方defaults，不继承TaskE自己的默认角。
- 验证应采用重排term和joint、非单位scale和非零offset的静态schema，使用真正Isaac `JointAction.process_actions` 的源函数做CPU消费端校验，而非复制新控制器算法作为expected。

## 5. 当前零动作日志的范围

初次查看 `/home/lybm/ATEC_Experiments_20260910/task_b_initial/zero_survival.log` 仍处于 `Starting the simulation`，没有执行step或终止项。日志包含 `Environment seed: None`；这是原play入口限制，bootstrap应固定cfg.seed且env.reset(seed)。在确实产生步骤证据前，不把启动等待归因于姿态或接触。

## 6. Task B 的布局 seed 必须在配置构造时传入

`tasks/task_b/env_cfg.py:75-83` 在 `TaskBEnvCfg.__post_init__` 内立即以 `np.random.default_rng(seed=self.seed)` 生成18物体的固定初始位置。因此不能先无seed调用 `parse_env_cfg`、再修改 `cfg.seed`；后者虽会影响env随机数，却不会重新生成已写入scene的物体位置。优先直接 `TaskBEnvB2WCfg(seed=args.seed)`，并继续 `env.reset(seed=args.seed)`。CPU核验应检查seed是配置构造参数，实跑再比起始18物体布局；策略不能读取生成坐标或由seed复原对象位置。

## 7. 首轮初始化超时：优先修正 experience 版本

首轮zero_survival由root报告timeout240s、INT后15s强杀、exit137；没有任何控制步，不能记作有效0分回合或策略失败。

- 主日志实际加载 `/home/lybm/IsaacLab/apps/isaaclab.python.headless.rendering.kit`。此文件声明 `app.version="5.1.0"`，启用 `app.useFabricSceneDelegate=true`、`rtx.hydra.readTransformsFromFabricInRenderDelegate=true`、`fabricUseGPUInterop=true`。
- 同目录4.5专用 `apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit` 声明4.5.0，没有这三个5.1配置；extension registry和asset根也从107/5.1换为106/4.5。
- 首轮主日志反复报 `failed to get rt interface`、`Could not load any plugins to populate Fabric`、`FSD enabled but IUtils could not be created`。这与错用experience吻合，是优先嫌疑，仍需同条件重跑验证，不能宣称已证明具体卡住栈。
- 更详细 `/tmp/isaaclab/logs/isaaclab_2026-09-10_15-52-36.log` 已记录机器人初始化成功、27body/24joint与真实PD/限位；最后在head_camera XformPrimView初始化。因此最后可见spatial-tendon warning不是机器人初始化失败证据。当前没有资产下载等待、OOM或PhysX数值异常证据。
- 下一次最小变化：同样hold参数，显式追加 `--experience /home/lybm/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit`，先保留其他参数以验证版本差异。
- 注意 `XformPrimView.__init__` 无条件打印“Using Fabric”；实际 `_use_fabric` 是 `/physics/fabricEnabled`设置，不能从这行debug推断 `--disable_fabric` 失效。

## 8. 移动Task B不能直接复用旧Task E camera reader

`tools/task_e/fabric_compat.py` 当前仍在其legacy reader上调用 `XFormPrim.get_world_poses(usd=False)`。该Isaac4.5路径会创建camera自身 `_worldPosition/_worldOrientation`；在此前TaskA中已实证会阻断相机继承parent运动。

正式TaskB bootstrap应复用 `task_a/tools/d1g2_taska_camera_compat.py::prepare_camera_views` 的无副作用递归 `isaacsim.core.utils.xforms.get_world_pose` reader：在AppLauncher后、gym.make前安装，finally恢复。它按live Fabric父link+USD局部相机外参组合，仅读取camera pose给renderer/sensor，不可把camera/worldpose转送policy。首次移动实跑仍需确认RGBD随base/手臂更新；初始化通过不能代替视觉时序验收。

## 9. 4.5 重跑与 CPU 边界核验完成

- `zero_survival_isaac45.log`：逐步返回日志共 500 条，step 0..499 连续=True，全部 terminated/truncated=False=True；SHA-256 `2fad02252a7640d2972d4e48d78d7b9e22cf8647d1ff4d21bb7eb369f4ac198d`。
- `low_wheel_isaac45.log`：逐步返回日志共 500 条，step 0..499 连续=True，全部 terminated/truncated=False=True；SHA-256 `d8011524bd7f4e8250f6126f1d1c4f28ec1a2a6ca64473654bc018f048f1c6bf`。

两份旧 play 日志没有连续位姿/速度遥测，不能据此量化 low_wheel 是否移动或其速度符号。相同入口显式选 4.5 后完成 500 步，支持首轮启动问题与 experience 不兼容有关；初轮 exit137 仍不算策略失败。

统一工程新增 `task_b/diagnostics.py::PreResetRecorder`，10 项独立 CPU 边界全部通过；原方法一次调用/返回值和异常 identity、观察失败不中断 reset、buffer 深复制、初始 reset 排除及 close 恢复均验证。`task_b/audit_bootstrap.py` 另直接提取实际 Isaac ActionManager/JointAction/RewardManager 方法做 CPU 执行：12 种重排 schema、192 个动作物理值检查，最大误差 9.536743172944284e-9；总 reward 除 dt 一次与 active terms 原值一致。报告 `results/task_b_bootstrap_cpu_audit.json` 含精确消费端源码 SHA。CPU 审计不替代物理稳定性、碰撞、相机时序或任务抓投验收。
