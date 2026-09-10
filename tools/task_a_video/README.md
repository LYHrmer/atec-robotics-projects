# 将 Task A 原录像整理为横屏展示

本目录只处理已经录制的视频和同次运行的传感器帧，不启动 Isaac Sim，不修改机器人策略，也不生成新成绩。原始 `d1g2_taska_full_run.mp4` 保留，横屏版本使用另一个文件名。

## 默认布局：原通关录像与同次前视

- 输出 1920×1080，16:9；完整过程保持原来的 10 fps、1× 仿真播放速度。
- 左侧显示第三人称画面，只裁去原录像顶部 40 像素黑色文字栏；剩余 640×440 画面等比放大到 1280×880。
- 右侧显示同次运行的真实前置 RGB、赛道进度、仿真时钟和地形说明。未做图像补帧、AI 重绘或颜色调整。
- 原录像全部 5057 帧保持原顺序，随后添加明确标注的 2 秒定格结果页。视频时长为 507.7 秒，任务成绩仍为 **505.66 仿真秒、286.002 米**。

前置相机取控制步 `5n` 保存的 RGB，第三人称原录像第 `n` 帧由控制步 `5n+1` 保存；两个保存调用的控制时间差为 0.02 仿真秒，不另行推断渲染器内部曝光时刻。过程数值取不晚于原录像标注控制时间的最近一条日志，未插值；距离为日志中的世界 X 减去原始起点 X。这些数据只用于事后展示，不送回控制器。

原第三人称录像开头存在绿色赛道标记遮挡，展示版保留这一原始情况。前置相机可帮助读者观察当时的路面。原录像最后一帧早于终止记录约 0.04 秒，结尾成绩页依据保存的验收 JSON，未将定格图冒称为终止时刻新画面。

## 从原始素材重新导出

需要 FFmpeg / ffprobe，以及 Python 的 NumPy、OpenCV 和 Pillow。默认中文字体为 Ubuntu 的 `NotoSansCJK-Regular.ttc`，可用 `--font`、`--font-index` 指定其他已安装字体。脚本不下载资源。

从仓库根目录执行，路径替换为自己的素材位置：

```bash
python3 tools/task_a_video/export_landscape.py \
  --source-video /path/to/d1g2_taska_full_run.mp4 \
  --sensors /path/to/residual_1999_axis_recovery_course_02/sensors \
  --output /path/to/d1g2_taska_landscape_1080p_1x.mp4
```

`sensors/` 是原机保存的 `frame_000000.npz` 到 `frame_025280.npz`，每隔 5 个控制步一份，共 5057 份。它们体积较大，没有随 Git 仓库上传；需要原始采集目录才能重新生成前置相机分栏。默认 trace 与 result 直接读取本仓库的完整通关证据，也可通过 `--trace`、`--result` 指定。

先检查排版而不编码整段，可追加：

```bash
--preview-dir /path/to/new_preview_directory --preview-only
```

脚本拒绝覆盖已有输出文件。完整导出会同时保存：

| 文件 | 用途 |
| --- | --- |
| `*.mp4` | H.264、yuv420p、faststart 横屏视频；默认 CPU libx264 medium / CRF 20 / 8 线程 |
| `*.provenance.json` | 原片、trace、result、字体、导出脚本与成片 SHA-256，尺寸、时间轴及实际编码命令 |
| `*.sensor_rgb_sha256.jsonl` | 每个前置 RGB 的控制步、来源文件名与像素 SHA-256 |

比对成绩时以 [验收记录](../../task_a/evidence/acceptance_summary.json) 为准。视频排版和播放速度不能作为控制算法变快的证据。

## 无前视模式：提速运行与成绩对照

提速运行 `speed_070_050_seed42_02` 没有采集前视 RGB。本模式只读取这次运行的第三人称原片、result 和 trace，右侧显示成绩对照卡，不读取或借用其他运行的前视画面。

本次 **457.32 仿真秒** 完成原始赛道，前进 **286.0045166 米**，唯一终止项仍为 `reach_goal_x`。相对原通关的 505.66 秒，减少 **48.34 秒 / 9.56%**。比较的是两次实际任务的仿真用时，视频均保持 1×；这不是通过加快视频播放制造的提速，也不代表随机场景稳定成功率。

```bash
python3 tools/task_a_video/export_landscape.py \
  --without-front-camera \
  --source-video /path/to/speed_070_050_seed42_02/run.mp4 \
  --trace /path/to/speed_070_050_seed42_02/trace.jsonl \
  --result /path/to/speed_070_050_seed42_02/result.json \
  --output /path/to/d1g2_taska_faster_landscape_1080p_1x.mp4 \
  --threads 4
```

对照结果默认使用仓库中原通关的 `task_a/evidence/full_course_02/result.json`，可用 `--baseline-result` 指定。无前视模式拒绝同时提供 `--sensors`，来源 JSON 中相机素材明确为 `null`。

本次原片共有 4573 帧、457.3 秒；横屏版保留每一帧并追加 2 秒结果页，得到 **459.3 秒、1920×1080、10 fps**。结尾画面沿用最后录制帧，终止数值来自验收 JSON，不生成新的仿真终止帧。CPU libx264 使用 4 线程；本机导出另将整个进程限制在 4 个 CPU 上。

新增模式后，默认带前视布局与第一版导出脚本在开头、碎石、台阶、末帧和结果页共 5 个关键帧的 RGB 像素完全一致。原视频、已导出的展示视频与其来源记录都保留，第一版脚本快照也单独保存以对应历史 SHA。

## 可选的 10× 全程速览

原通关的完整原速版（505.66 仿真秒，带前视布局）导出成功后，可以再做约 53 秒的速览。过程明确标注 **10×**，最后结果页仍定格 2 秒，原通关成绩保持 505.66 仿真秒。这个脚本仅用于该原通关版本，不用于上面的提速版。

```bash
python3 tools/task_a_video/export_preview.py \
  --source /path/to/d1g2_taska_landscape_1080p_1x.mp4 \
  --output /path/to/d1g2_taska_landscape_preview_10x.mp4
```

它先核对完整版本及其来源 JSON 的 SHA，再压缩过程时间戳；30 fps 输出会丢弃部分中间帧，不使用光流补帧。结果与完整录像请一起保留，速览用于快速观看。

## 网页直接播放

[播放页](https://lyhrmer.github.io/atec-robotics-projects/) 使用 `docs/videos/d1g2_taska_faster_720p_1x.mp4`，由提速版 1080p 横屏视频缩小至 1280×720。H.264 Main / Level 3.1、yuv420p、faststart，约 21 MB，便于浏览器加载。全部 4593 帧和 459.3 秒时间轴保留，没有倍速或删减，原 1080p 文件保留。

具体转码参数、原文件和网页文件 SHA-256、完整解码验证见 [来源记录](../../docs/videos/provenance.json)。`docs/index.html` 提供原生播放器；GitHub Pages 从 `main` 分支的 `/docs` 发布，`.nojekyll` 保持静态文件原样输出。Release 中的 1080p 链接是下载附件，网页播放器直接加载同站点的 MP4。
