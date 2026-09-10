# Task A：让 D1+G2 走完整条越障赛道

[返回项目首页](../README.md) · [先读白话原理](docs/LEARNING_GUIDE.md) · [看通关证据](evidence/README.md)

机器人带着 G2 机械臂，沿赛道连续走过碎石、坡地和台阶。本项目复用 D1 的平地行走策略，再训练一个网络修正动作；前置相机负责估计行进方向。机械臂在这项任务中保持默认姿态。

**已在本机连续通过原始 Task A。** 这是一次完整通过记录，同配置也有碎石路失败记录；没有声称每次稳定成功、实机部署或官方在线验收。

| 已验证的内容 | 结果 |
| --- | --- |
| 原始起点 → 终点 | x=−141 → 145.00235 m，前进 286.00235 m |
| 任务内用时 | 505.66 仿真秒，实际运行约 16.8 分钟，不含启动 |
| 终止判定 | 唯一终止项 reach_goal_x，没有同帧失败 |
| 运行显存 | 采样峰值 6210 MiB，GPU 采样最高 61°C |

[完整视频](https://github.com/LYHrmer/atec-robotics-projects/releases/tag/task-a-local-pass-20260909) · [结果 JSON](evidence/full_course_02/result.json) · [训练权重](weights/model_1999_final.pt) · [完整轨迹](evidence/full_course_02/trace.jsonl)

**2026-09-10 提速实验：457.32 仿真秒完整通过，比上述原版少 48.34 秒（9.56%）。** 使用同一个训练模型，只把巡航/碎石段指令从 0.60/0.45 调为 0.70/0.50 m/s。新旧完整录像均保留；这是新配置的一次成功记录，仍需更多重复实验评估稳定性。见 [速度对比与复现](docs/SPEED_COMPARISON.md)。

## 跑一次项目

下面命令都从**统一仓库根目录**执行。先准备现有的 Isaac Lab 环境；本项目没有把仿真器和显卡驱动打包进 Git。原验证机器为 Ubuntu、Python 3.10、Isaac Sim 4.5、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128，完整版本见 [环境记录](provenance/environment_versions.json)。

**1. 指定 Python 和 Isaac Lab 路径。** 以下是原验证机器的路径，换电脑时改成自己的安装位置。

```bash
export ATEC_PYTHON=/home/lybm/miniforge3/envs/isaaclab/bin/python
export ISAACLAB_PATH=/home/lybm/IsaacLab
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 与启动器保持一致，避免用户目录的同名包覆盖仿真环境。Isaac Lab 之外，推理还需 `onnx` 和 OpenCV；录像还需 `imageio`、`imageio-ffmpeg` 和 Pillow。先检查现有环境：

```bash
"$ATEC_PYTHON" -c 'import onnx, cv2, imageio, imageio_ffmpeg; from PIL import Image; print("运行依赖已就绪")'
```

若有缺失，在同一个 Python 环境补装；下列 ONNX / OpenCV 版本取自完整通关时的环境记录：

```bash
"$ATEC_PYTHON" -m pip install onnx==1.22.0 opencv-python==4.11.0.86 imageio==2.37.3 imageio-ffmpeg==0.6.0 Pillow==11.3.0
```

运行现有模型不需要重新训练。重新训练时另需 `rsl-rl-lib==3.0.1`、`tensordict==0.13.0`，见 [训练记录](docs/training_history.md)。

**2. 导入原始机器人资源。** 需要用户提供的 workspace2.zip；导入器只提取运行所需的 18 个文件，并核对 SHA-256。这些第三方原始资源不在公开仓库中。

```bash
python3 task_a/scripts/setup_assets.py --from-zip /绝对路径/workspace2.zip
python3 task_a/scripts/setup_assets.py --check
```

已有 DDT_Lab 目录时，可改用 `--from-directory /绝对路径/DDT_Lab`。也可设置 `ATEC_D1G2_ASSET_ROOT` 直接使用外部目录，详见 [资源说明](assets/README.md)。导入的资源保留在本地，不提交 Git。

还需从官方任务资源目录导入 **5 个原始赛道材质、纹理与天空文件**。它们提供视觉导航所需的图像特征，仅有机器人模型不足以完成视觉闭环：

```bash
python3 task_a/scripts/setup_scene_assets.py --from-directory /绝对路径/ATEC2026_Simulation_Challenge/atec_robot_model
python3 task_a/scripts/setup_scene_assets.py --check
```

两个导入器都核对原始文件 SHA-256，启动器会在进入仿真前检查资源完整性。

**3. 启动。** 每次使用一个不存在的输出目录。

```bash
bash run.sh task-a --output outputs/course_01 --video
```

复现提速配置时改用 `bash run.sh task-a-fast --output outputs/fast_course_01 --video`。两个入口分别保留原参数和提速参数，均使用原任务终点判定。

结果位于 `task_a/outputs/course_01/`：`result.json` 给出最终判定，`trace.jsonl` 记录过程，`run.mp4` 是录像。只有 `reason=reach_goal_x` 且没有同帧失败才算原始任务通过。

只想检查新路径能否启动，可追加 `--max_steps 100` 并换一个输出目录；这是短测，不是通关测试。原有入口 `bash task_a/scripts/run_taska.sh ...` 仍可使用。

## 看懂代码，按这个顺序读

| 问题 | 对应文件 |
| --- | --- |
| 一次运行怎样启动、判定和记录？ | [run_d1g2_taska.py](tools/run_d1g2_taska.py) |
| 原始赛道怎样换成 D1+G2？ | [d1g2_taska_env.py](tools/d1g2_taska_env.py) |
| 基础动作怎样加上学到的修正？ | [d1g2_taska_residual.py](tools/d1g2_taska_residual.py) |
| 相机怎样决定向前走和转向？ | [视觉导航](tools/d1g2_taska_visual_navigation.py)、[RGB-D 位移估计](tools/d1g2_taska_rgbd.py) |
| 失去旧画面对应关系时怎么办？ | [有预算的参考重建](tools/d1g2_taska_visual_recovery.py) |
| 模型怎样训练？ | [训练入口](tools/train_d1g2_taska_residual.py)、[训练环境](tools/d1g2_taska_train_env.py) |

想先理解这些模块为什么存在，读 [白话原理与术语](docs/LEARNING_GUIDE.md)；想复盘失败和训练阶段，读 [训练记录](docs/training_history.md)；想准备面试，读 [自测与讲解要点](../docs/TASKA_D1G2.md)。

## 验证与文件来源

- 部署检查点为迭代编号 1999 的残差模型。图像由导航模块处理，低层策略不是直接接收原始图像的端到端视觉网络。
- 原任务的起点、地形、物理与终止保留。世界真值用于原终止、诊断和停滞看门狗，不作为 actor 或导航位姿输入。
- 原 505.66 s 通过运行在最后台阶使用两次短时本体积分连接新视觉片段，累计 0.9 s、0.4844 m 路径积分；提速运行累计为 0.7 s、0.3796 m。它们不是纯视觉重定位，也不是定位误差数值。
- 本目录有真实成功和失败证据。[迁移说明](docs/MIGRATION.md) 区分原通关运行与本次目录迁移短测；[来源声明](THIRD_PARTY_NOTICES.md) 说明框架、训练权重和外部资源。
