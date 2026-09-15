# Task B 首次真实正分：发布清单

2026-09-14 只读准备。**当前仍为 0 分；以下是正分之后的待办，未写入产品页，未执行 Git 写入或发布。**

## 已确认的发布位置

- 本地仓库：`/home/lybm/ATEC_Robotics_Projects_20260910`，分支 `organize-atec`，推送目标 `origin/main`。
- GitHub：`LYHrmer/atec-robotics-projects`。API 已确认 Pages 使用 **main 的 /docs**，当前状态 `built`；无需新增部署 workflow。
- 当前 `/docs/index.html` 专门播放 Task A，A 播放地址为 `https://lyhrmer.github.io/atec-robotics-projects/`。保持其视频与来源说明，只增 Task B 导航。
- API 核实 8 个既有 Release 均存在。`task-b-bootstrap-20260910` 只有 `manifest.json`、`task_b_comparison.json`、`task_b_evidence.tar.gz`，没有 B 视频。

## 1. 先取得能发布的真实产物

- [ ] 完整新进程 run 结束并写出 `result.json`；读取实际 `score_raw_total>0`，不能用 CPU fixture、缓存审计布尔或策略 debug 的预测距离代替。
- [ ] 执行固定哈希的 `task_b/audit_positive.py` 得到退出码 0，分项奖励、唯一物体、得分步、动作后或 reset 前状态均核对通过。
- [ ] 如同一步或后续 illegal/fall，保留官方原奖励，公开发生时刻与 `quality_limits`；不能写成稳定完成或通关。只有 proximity 分时，明确“接近得分，尚未证明抓起/投递”。
- [ ] 用 `task_b/package_positive.py` 写全新目录，复核无 `INCOMPLETE`，`sha256sum --check SHA256SUMS` 通过。入口已拒绝 `SYNTHETIC_ONLY.txt` 与 `fixture_notice`。
- [ ] 将打包器中的审计器冻结 SHA 与实际 `task_b/audit_positive.py` 对齐；当前为 `f31f81582633b4fc53157b96e403a1fea4b1e571911a504c2923aeef555904ca`。

```bash
export ATEC_TASK_ROOT=/home/lybm/ATEC2026_Simulation_Challenge
export PYTHONNOUSERSITE=1
/home/lybm/miniforge3/envs/isaaclab/bin/python task_b/package_positive.py \
  /absolute/path/to/real_positive_run /absolute/path/to/new_positive_package \
  --auditor task_b/audit_positive.py
```

## 2. 实际需提交的文件

| 文件或位置 | 操作与内容 |
| --- | --- |
| `task_b/evaluate.py`、`first_reach.py`、`stance_hold.py`、`stance_reference.py`、`stationary_target_gate.py`、`brake_wheel_hold.py` | 提交最终实际使用的控制和记录代码；最终运行的 source snapshots 保留其确切版本，不能把后来工作树误写成得分版本 |
| `task_b/audit_positive.py`、`package_positive.py`、`POSITIVE_PACKAGE.md` | 提交验收与打包入口 |
| `task_b/audit_first_reach.py`、`audit_brake_wheel_hold.py` 及实际对应 CPU 结果 | 提交本轮控制边界验证；从已有输出整理，勿为了发布重新跑无关长测 |
| `task_b/locomotion.py`、`heading_schedule.py`、各自 `audit_*.py` / `*_DESIGN.md`，`results/task_b_locomotion_cpu_audit.json`、`task_b_heading_schedule_cpu_audit.json` | 已有 Opus 候选可一并保存，标明实际 GPU 使用与否；未使用者不能宣称贡献了本次得分 |
| 新 `results/task_b_positive/<run_id>/` | 保存小体积的原样 result、metadata、source_manifest、scoring_events、独立审计、public_package 清单；完整 telemetry、gzip trace、原视频和源码快照放 Release 证据包，给出下载链接 |
| 新 `docs/TASK_B_FIRST_SCORE.md` | 单次真实运行条件、精确命令、得分事件表、唯一物体和质量限制、源码 hash、证据包链接；不推断成功率 |
| 新 `docs/task-b.html` | 新 B 独立播放器，不改 A 页的主视频 |
| 新 `docs/videos/task_b_<run_id>_720p_1x.mp4`、`task_b_<run_id>_poster.jpg`、`task_b_<run_id>_provenance.json` | 网页版独立命名；provenance 绑定原片/新片 SHA、尺寸、帧数、时长及转码命令 |
| `docs/TASK_B_EXECUTION_PLAN.md`、`TASK_B_ASTRA_DECISIONS.md`、`COLLABORATION.md` | 保留计划与真实分工，追加此次证据和哪些阶段仍未完成 |

当前未提交文件由多个代理共同写入；发布时使用明确路径暂存，避免 `git add -A` 把临时物或零分视频带入。

## 3. 文案改动位置（以本次读取的行号为准）

| 文件、行 | 得分后如何改 |
| --- | --- |
| `README.md:3` | “两项机器人任务”改为 A/E 已验证任务与 B 移动操作实验的统一管理说明；不把三个任务都称作通关 |
| `README.md:15` | 将“当前仍为 0 分”替换为实际得分和范围，例如“Task B 本机原始得分 {score}：{proximity_count} 个接近分、{delivery_count} 个投递分；尚未通关”，附新播放器和证据链接 |
| `README.md:32`、`:55` | 阅读路线的“两项”仍指 A/E 时应明确 A/E；统一启动入口已支持 A/B/E，改为“三项任务” |
| `task_b/README.md:3`、`:5` | 顶部新增“首次正分记录 / 在线视频 / 打包说明”；下方基础模式保留为历史基础阶段 |
| `task_b/README.md:7`、`:9`、`:25` | 简述最终使用的 first_reach 状态链及观测边界；保留“接近不等于抓住”的说明 |
| `task_b/README.md:45`、`:59` | 补真实 first_reach 复现命令与新参数；产物表补 env_reward、scoring_events、final_state_before_close 和时序语义 |
| `docs/VIDEO_COMPARISON.md:49`、`:51` | 标题改为“Task B：首次接近得分”；新增此次原片/网页片/分数/时长/终止原因；历史零分片仍仅本机，不更改原 comparison JSON 的 video.published=false |
| `docs/TASK_B_EXPERIMENTS.md:1`、`:5` | 标明“基础阶段归档（2026-09-10）”，把“当前 0 分”限定为本页历史批次，顶部链接新正分报告；12 行旧实验数字与失败结论保留 |
| `docs/TASK_B_EXECUTION_PLAN.md:9`、`:13` | 追加新正分 run 的实际更新，旧 11 次完整零分批次保留为历史记录；M2—M4 仍按证据状态显示 |
| `docs/COLLABORATION.md:5` 附近表格 | 填入 Opus 本次实际实现、Codex 集成、独立审核分别做了什么及验证结果；不把候选模块全部算作已上线控制 |
| `docs/index.html:73` 的 footer nav | 仅增加 `<a href="task-b.html">Task B 接近得分录像</a>`；第 58 行 A 视频 src、海报、下载链接与第 6—7 行 A 标题描述继续保留 |

## 4. 浏览器视频与旧视频保留

已用 ffprobe 抽查现有本机 B 原片：**H.264 High / yuv420p / 1280×480 / 10 fps，80 秒约 200.9 MB**。兼容编码并不代表文件适合直接进 Git。正分后原片放新 Release，网页完整 1× 版另行压缩。不能将 Release 下载 URL 当作唯一 `<video src>`；采用已验证的 Pages 同源文件方式。

网页影片可保持双相机内容，按比例在 1280×720 中加上下留白；不删帧、不改变时间轴、不生成机器人动作。示例参数（实际正分后执行并记录 ffmpeg 版本、完整命令、SHA）：

```bash
ffmpeg -n -i /absolute/path/to/positive_run/public_cameras.mp4 \
  -vf 'pad=1280:720:0:120:color=0x101820' -an \
  -c:v libx264 -preset medium -crf 26 -pix_fmt yuv420p -movflags +faststart \
  docs/videos/task_b_RUN_ID_720p_1x.mp4
```

- [ ] 网页版实际尺寸、codec、pix_fmt、帧数、fps、duration 检查通过；文件明显小于普通 Git 单文件上限，尽量控制在几十 MB。不能根据上面的 CRF 预先宣称最终大小。
- [ ] `<video controls playsinline preload="metadata">` + `type="video/mp4"`，同源相对路径、真实运行海报、加载失败提示和下载入口；复用 A 页现有控制方式即可。
- [ ] 页面分开写“首次得分仿真时刻”“回合仿真时长”“媒体时长”。如得分与终止同步而末帧未录到，说明视频最后正常帧早于该事件，得分由 reset 前证据核验。
- [ ] 不覆盖 `docs/videos/d1g2_taska_faster_*`、原 `docs/videos/provenance.json`、`media/task_e_*.mp4`；为 B 使用独立 provenance 文件。
- [ ] 保留这 8 个既有 tag 及附件：`task-e-imitation-20260910`、`task-b-bootstrap-20260910`、`task-a-method-comparison-20260910`、`task-a-local-pass-20260909`、`task-a-experiment-history-20260909`、`robotics-comparison-20260910`、`fast-smooth-18`、`baseline-18`。

## 5. 最短发布顺序

1. 冻结真实 run → 审计 → 正分 package → checksum；将 package 封为独立 `task_b_positive_<run_id>.tar.gz`。只该新目录进入待发布素材，零分实验目录不进入。
2. 生成网页片和独立来源记录；完成上表文档/页面更新。用本地 HTTP 服务打开新 B 页，验证真实播放、拖动、全屏和错误/下载入口；A 页及 A/E 旧链接做回归点检。
3. 检查 `git diff --check` 与显式暂存清单，确保只提交预定代码、数据摘要、页面和 B 网页影片；提交并推送 `HEAD:main`。记录推送 commit，不重写旧提交或旧 tag。
4. 在**新 tag**（例如 `task-b-first-positive-YYYYMMDD`，实际先检查未占用）创建 GitHub Release，target 为该推送 commit。上传证据 tar.gz、原片和来源/校验清单；不使用 `--clobber` 覆盖历史附件。Release 文案写确切分数、分项和质量限制。多行发布说明使用临时正文文件和 `--notes-file`。
5. API 查询新 Release 和 Pages build，确认目标 commit / 附件名 / 大小；下载公开附件复算 SHA。打开 `https://lyhrmer.github.io/atec-robotics-projects/task-b.html`，确认视频实际开始播放且能跳转，浏览器无 404/解码错误；再次检查原 Task A 首页正常。
6. 最终回复给用户**仓库提交、新 Release、B 在线播放**三个实际链接，写明取得的是哪类分数、是否终止、尚未完成什么。只有上述公开状态已核实，才能说“已上传且可播放”。

本清单没有创建任何正分结论。真实 run、验证和公开链接尚缺时，保持当前页面事实为 0 分。
