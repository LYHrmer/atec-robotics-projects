# Task A：真实提速与视频对比

[返回 Task A](../README.md) · [视频对比入口](../../docs/VIDEO_COMPARISON.md)

同一个 D1+G2 模型在原始 Task A 完整赛道上，出现了更快的一次通过：**457.32 仿真秒，比原记录减少 48.34 秒（9.56%）**。这是真实控制运行用时的变化，与视频倍速分开记录。

| 比较项 | 原通过记录 | 提速通过记录 |
| --- | --- | --- |
| 运行编号 | residual_1999_axis_recovery_course_02 | speed_070_050_seed42_02 |
| 巡航 / 碎石指令 | 0.60 / 0.45 m/s | 0.70 / 0.50 m/s |
| 完整任务仿真时间 | 505.66 s | 457.32 s |
| 原始起点 → 终点 | −141 → 145.00235 m | −141 → 145.00452 m |
| 唯一终止项 | reach_goal_x | reach_goal_x |
| 模型 | 1999 残差检查点 | 相同检查点，无重新训练 |
| GPU 采样峰值 | 6210 MiB | 6212 MiB |
| 实际运行时间，不含启动 | 1007.88 s | 1000.18 s |

渲染、图像处理、视频编码和其他 CPU 工作影响实际运行时间，因此不能把任务内 9.56% 提速直接说成电脑计算提速。每个配置已有至少一次完整成功记录，原配置也曾失败；当前不足以比较稳定成功率。

## 怎样复现

按 [Task A 运行说明](../README.md) 完成仿真环境、机器人资源和原始场景资源准备后，在仓库根目录执行：

```bash
# 原配置
bash run.sh task-a --output outputs/baseline_compare_01 --video

# 提速配置
bash run.sh task-a-fast --output outputs/fast_compare_01 --video
```

每次使用新输出目录。`task-a-fast` 只在现有启动参数上指定 `--speed .70 --rough_speed .50`，不改物理、地形、计分、相机安装或原终点判定。碎石减速区间依旧由视觉估计的位置判断。

## 结果与证据

- [提速结果](../evidence/speed_070_050_02/result.json)、[完整 trace](../evidence/speed_070_050_02/trace.jsonl)。
- [原通过结果](../evidence/full_course_02/result.json)、[原完整 trace](../evidence/full_course_02/trace.jsonl)。
- 新旧原始录像、横屏展示版和明确标为 10× 的速览分别保存，详见 [视频对比页](../../docs/VIDEO_COMPARISON.md)。

## 这次还修复了什么

首轮新目录测试未导入原始赛道 MDL、纹理及天空 HDR，相机无法建立特征，69.90 s 后停滞退出，只有 2.17 m 进展。[原始失败结果](../evidence/scene_assets_missing_01/result.json) 保留，属于配置失败，不能拿来评价 0.70/0.50 m/s 的步态能力。

此前 100 步短测只覆盖 2 秒启动阶段，没有证明视觉初始化；缺资源的问题因此未被发现。本次导入原始 5 个场景文件并校验 SHA 后重新启动，才得到上表的完整通过。新的启动检查同时覆盖机器人与场景资源，在进入仿真前报告缺失；没有用新材质或修改地图来改善视觉条件。
