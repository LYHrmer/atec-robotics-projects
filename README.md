# ATEC Task E：RGB-D 三物体抓取与入篮

固定基 Piper 机械臂使用 RGB-D、运动学和反馈状态机，完成糖盒、芥末瓶与香蕉的抓取、搬运和入篮。加速版在本机 **seed 42 实测 18/18 分，2689 步，53.78 秒仿真时间**，比基线缩短 **13.03%**。打包后的同一算法在 **seed 0 得到 18/18 分、62.36 秒**。

这是可解释的几何控制项目，针对已知机器人、相机标定和物体集合；没有训练新的强化学习或模仿学习模型。成绩来自本地官方任务环境，不代表线上榜单成绩或所有随机场景的成功保证。

[![播放加速版三物体成功录像](media/fast_preview.jpg)](media/task_e_fast_seed42_color.mp4)

**[新版彩色视频](media/task_e_fast_seed42_color.mp4)** · **[新版原始录像](media/task_e_fast_seed42_original.mp4)** · **[优化对比](docs/OPTIMIZATION.md)** · **[关键算法](docs/ALGORITHM.md)** · **[面试讲解提纲](docs/INTERVIEW.md)**

[基线版本与视频](https://github.com/LYHrmer/atec-taske-rgbd-manipulation/releases/tag/baseline-18) 保留，可随时对照。

视频为 1280×720、25 fps，按 1 倍仿真时间连续展示，两路相机同时可见，结尾定格 2 秒。彩色展示版仅增强饱和度和对比度，使用 BT.709 色彩标记；没有剪掉失败片段、改变动作速度或编辑计分。原始文件与处理记录见 [视频来源说明](media/provenance.json)。

## 结果与证据

| 版本与运行 | 种子 | 得分 | 仿真时间 | 说明 |
| --- | ---: | ---: | ---: | --- |
| v3 加速源版本，录像评测 | 42 | **18/18** | **53.78 s** | 打包仅规范化导入路径，[结果及哈希](results/fast_seed42_source.json) |
| v3 加速独立包复核 | 42 | **18/18** | **53.78 s** | 经公开启动器复现，[结果](results/fast_seed42_package.json)；终止后松爪观察通过 |
| v3 加速独立包 | 0 | **18/18** | **62.36 s** | 发布的七个运行文件与该次实测逐字节一致，[结果](results/fast_seed0_package.json) |
| 基线独立 v2 包，录像评测 | 42 | **18/18** | **61.84 s** | 对应 `baseline-18` 标签，[结果及哈希](results/seed42_video.json) |
| v2 源版本 | 0 | **18/18** | 66.58 s | 同一算法与五个依赖；打包时仅将入口导入路径改为平级，[结果](results/seed0_source.json) |
| 远端香蕉实验分支 | 5 | 15/18 | 69.70 s | 抓起后搬运滑落，属于另一实验分支，[失败记录](results/far5_experiment.json) |

这些验证的覆盖范围有限，不据此报告统计成功率。53.78 秒是环境完成计分的时间；官方入篮事件可能发生在松爪前，因此计分结束与释放后留篮是两个不同的检查。seed 42 已追加 3 秒松爪诊断：最终夹宽约 70 mm，最后 1 秒三个物体持续在篮区内；[独立报告](results/fast42_release/post_terminal_release.json) 与 [释放后截图](media/fast_release_final.png) 单独提供，额外诊断不计入 53.78 秒。任务物理参数、计分和终止条件保持不变。场景真值只用于开发期间的离线诊断，不进入策略输入；发布代码不读取运行时真实物体位姿。

## 算法怎样工作

```text
固定外部 RGB-D → 深度反投影 → 三维连通簇与物体匹配
             → 物体专属接触点 → 有关节限制的 IK 轨迹
             → 关节反馈执行 → 公开得分确认抓起与入篮
```

- **瓶子净空**：由瓶顶深度定位瓶颈，抬高悬垂瓶身后再横移，解决瓶底撞篮沿的问题。
- **香蕉夹持**：考虑资产的凸包碰撞几何、夹爪开口和重心偏移，选择较稳的接触位置，先竖直提升，再逐步倾转。
- **持物姿态连续性**：在糖盒搬运与放置阶段插值位置和旋转，逐点检查 FK 误差、关节变化与桌面间隙，减少腕部分支跳变。
- **反馈状态机**：多帧定位、下降接触停滞判断、夹爪开度与得分联合确认，以及有限重试。
- **加速与减少转腕**：满足夹爪稳定、张开或公开得分条件后提前结束等待；糖盒空夹回撤沿当前姿态连续求解。全程腕关节行程减少 14.39%，但差分加速度 RMS 增加 4.36%，不据此声称全面降低加速度或冲击。

接口提供末端相机，但当前策略决策使用固定外部 RGB-D；末端画面用于录像与诊断。姿态连续路径检查也不是完整的物体避障规划，失败恢复仍有改进空间。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| [solution.py](solution.py) | v3 官方入口，反馈提前确认与糖盒连续回撤 |
| [solution_baseline.py](solution_baseline.py) | 保留 v2 入口，组合 RGB-D 与持物规划 |
| [solution_task_e_rgbd.py](solution_task_e_rgbd.py) | 接触点修正、瓶子净空、香蕉提升 |
| [solution_task_e_vision.py](solution_task_e_vision.py) | 状态机、反馈执行和任务确认 |
| [task_e_perception.py](task_e_perception.py) | RGB-D 三维分割与匹配 |
| [task_e_geometry.py](task_e_geometry.py) | 标定、FK/IK、动作尺度 |
| [task_e_held_motion.py](task_e_held_motion.py) | 连续持物姿态规划 |
| [tools/eval_task_e.py](tools/eval_task_e.py) | 本地评测、源码快照、逐步遥测、连续录像 |
| [tools/task_e/compare_motion.py](tools/task_e/compare_motion.py) | 相同种子下的耗时与关节运动对比 |

## 复现

先按 [ATEC 官方项目](https://github.com/atecup/ATEC2026_Simulation_Challenge) 配置 Isaac Lab 和比赛资产。本仓库不打包官方模型或机器人资产。策略运行只需 NumPy/SciPy，仿真评测另需官方环境、Pillow、PyTorch 与 FFmpeg。

本机验证环境：Ubuntu、Python 3.10.20、Isaac Sim 4.5、Isaac Lab 2.3.2、PyTorch 2.7.0+cu128、RTX 5060 Laptop。4.5 与新版 Isaac Lab 的相机接口兼容处理仅在本地评测工具中启用，不属于提交策略。

```bash
export ATEC_TASK_ROOT=/path/to/ATEC2026_Simulation_Challenge
export ATEC_PYTHON=/path/to/isaaclab/environment/bin/python

"$ATEC_PYTHON" -m pip install -r requirements.txt
bash scripts/evaluate.sh --check
bash scripts/evaluate.sh --solution solution.py --headless --seed 42 \
  --max_steps 6000 --output runs/seed42 --video runs/seed42.mp4
```

输出目录必须是新目录。录像先编码到临时文件，完成封装后才显示最终 MP4，避免打开录制中的不完整文件。评测会保存得分、停止原因、源码哈希、观测快照与诊断轨迹；策略只接收官方观测和累计得分。

无需仿真即可运行 20 项运动学、反馈确认与释放诊断测试：

```bash
PYTHONNOUSERSITE=1 "$ATEC_PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
```

本次运动对比的原始逐步数据已提供，可以独立重算：

```bash
"$ATEC_PYTHON" tools/task_e/compare_motion.py \
  results/baseline42_motion results/fast42_motion --relative-to . --output-dir runs/comparison
```

官方部署入口是 `AlgSolution.predicts(obs, current_score)`，输出包含 8 维 `action` 与布尔 `giveup`。当前方法无需 `policy.pt`。动作按官方位置控制定义换算，并非一律裁剪到 `[-1, 1]`。

## 面试展示

建议先播放视频，再解释“相机坐标如何变成关节目标”，最后用瓶子碰篮沿和香蕉滑落说明怎样依据证据定位问题。推导、实验边界和常见追问见 [算法说明](docs/ALGORITHM.md) 与 [面试提纲](docs/INTERVIEW.md)。

项目采用 Codex 与 Claude Opus 辅助实现和诊断，其中连续持物规划 helper 有 Opus 参与。重点是能够解释、验证和复现代码，不把 AI 生成初稿等同于已经验证的算法。

## 许可与来源

代码采用 MIT 许可，保留 ATEC 版权与来源说明。官方任务、Piper/YCB 等资产由各自项目提供，未随本仓库重新分发。详见 [LICENSE](LICENSE) 与 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。
