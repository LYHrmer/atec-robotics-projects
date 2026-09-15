# force75 原环境双指接触诊断

Run: `plan_p4_grasp_force75_contactdiag_seed42_01`。控制器仍为 `bf8a4204c5badd5c1708c2804e9e4a085e29535f88cfba40cb527b20abb4e306`，仅新增离线诊断记录。本回合 3257 步、约 1 分接近分，无投递；`grasp_probe_empty_during_lift` 停止，原环境未 terminated/truncated，非法力为 0。

## 已观测到的事实

原环境 live PhysX getter 给出 robot 全 25 个 collision shape 与 object_10 的材料均为 `[static_friction=1, dynamic_friction=1, restitution=1]`，目标质量 0.5 kg。不是借用 Task Base 中本回合未执行的 0.8/0.6 随机化配置。

以下为世界坐标系**作用在手指上的法向合力**，单位 N；Z 正方向向上：

| 阶段 | 指 7 水平模长均值 | 指 7 Z 均值 | 指 8 水平模长均值 | 指 8 Z 均值 | 双指法向模长都 >1 N 的样本比例 |
|---|---:|---:|---:|---:|---:|
| CLOSE 最后 0.5 s | 11.44 | +5.75 | 11.45 | +6.77 | 100% |
| LIFT 非空样本 | 7.86 | +3.97 | 5.90 | +3.36 | 99.35% |
| LIFT 最后 0.5 s | 2.99 | +1.52 | 2.63 | +1.46 | 61.54% |

CLOSE 最后 0.5 s，两指法向模长均值分别 12.80 / 13.31 N。LIFT 期间存在连续 152 个双指都 >1 N 的样本，两个端点相距 3.02 s。非空期间两指法向仰角均值为 26.7° / 29.6°。末段接触合力消失，夹爪宽度降为零。

使用每一帧实际 object quaternion 和原 mustard mesh：目标 10 的 root 最高 +4.203 mm；mesh 最低表面最高仅 +0.865 mm，最终 −0.106 mm。夹爪实际世界高度则上升 54.253 mm。没有达到物体离开原支撑面或随夹爪上升的证据。

## 对下一候选的含义与边界

实测双指法向受力与斜面方向，结合爪宽变化和实际物体几何轨迹，支持肩部受压后沿表面滑脱的解释。仅有法向力不足或仅有一指未接触，并不能解释全部记录；继续单纯增加预紧量缺乏本轮证据支持。改变接触高度/朝向以获得更合适的侧壁接触，是合理的下一候选方向，仍需实际回合验证。

`net_forces_w` 的本地原始定义位于 `/home/lybm/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py:74`：它只包含作用在 sensor body 上的 normal contact forces，明确不包含 tangential/friction forces。世界向上的手指法向意味着其接触对侧获得相反的向下法向贡献；不能据此把 `-sum(Fz)` 称为物体获得的完整向上支撑力，也不能由材料 μ=1 断言摩擦已用足。

这些力没有按被接触物体过滤，因此严格证明的是两根手指各自有接触，尚不能单凭该传感器区分目标瓶体、其他物体或自接触。肩部解释是结合实际轨迹与既有接触几何的推断。每个控制 tick 只读取缓存的物理力，零值不能证明中间所有物理子步都没有接触。

全部接触力与实际 q/object/gripper 状态按同一行 pre-step 对齐；得分发生后的动作状态不使用同索引 pre-step 冒充。

详细输出：`grasp_force75_contactdiag_force_review.json`、`grasp_force75_contactdiag_independent_mesh.json`。工具：`analyze_finger_contacts.py` 与 Astra 的 `audit_grasp_runtime.py`。本报告没有修改控制器、物理、原环境或上传结果。
