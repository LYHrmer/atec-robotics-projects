# Task B 正分证据打包

`package_positive.py` 只准备本地待发布目录。它不会上传 GitHub，也不会运行仿真。当前是否已经得分，必须看真实回合的独立审计结果。

使用有 NumPy 的 Python 解释器，在项目根目录执行：

```bash
export ATEC_TASK_ROOT=/path/to/ATEC2026_Simulation_Challenge
python task_b/package_positive.py /path/to/finished_run /path/to/new_package \
  --auditor task_b/audit_positive.py
```

输入必须是已完成的真实运行目录；输出父目录必须存在，输出目录必须尚不存在且位于输入目录之外。工具固定已审查的 v2 审计器 SHA-256，运行该脚本生成**全新审计报告**。零分、审计退出非零、证据不一致、路径穿越、符号链接文件、源码快照与清单不符都会拒绝。含 `SYNTHETIC_ONLY.txt` 或 `result.json.fixture_notice` 的合成自检样本也明确拒绝。输入中的旧审计 JSON 不参加判定。

仅在独立审计通过后，工具才建立输出目录，并按白名单复制 `result.json`、环境元数据、源码清单及其实际项目源码快照、`scoring_events.json`、`telemetry.npz`。原始 `trace.jsonl` 压缩为确定性的 `trace.jsonl.gz`；存在的 `public_cameras.mp4` 也只在正分验证通过后复制。不会遍历复制输入目录中的其他视频、图片或文件，也不会碰 Task A/E 的产物。

`public_package.json` 记录各文件的 SHA-256、大小以及 trace 解压后的 SHA-256/大小，注明这只是本地官方环境原始分数，未经比赛服务器认证。接近分不等于抓取、投递或通关。`SHA256SUMS` 覆盖全部产物和 `public_package.json`（不含校验清单自身）。检查命令：

```bash
cd /path/to/new_package
sha256sum --check SHA256SUMS
gzip -dc trace.jsonl.gz | sha256sum
```

第二条命令应与 `public_package.json` 中 `files["trace.jsonl.gz"].uncompressed_sha256` 一致。打包中途失败的目录会留下 `INCOMPLETE`，不可发布；重新执行必须使用另一个新目录。审计检查的是记录之间的一致性，无法为任意输入日志的真实来源提供密码学证明，发布者仍须确认其来自实际仿真。打包成功也不代表已经发布、视频已经验证可播放，或任务已经通关。
