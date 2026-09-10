可冻结的精简 GitHub evidence 文件

以下文件均位于本 audit_dependencies 目录：

- README.md
- run_acceptance_audit.json
- final_result_snapshot.json
- recovery_budget_audit.json
- hardware_audit.json
- run_source_hash_audit.json
- validation_lineage.json
- manifest.json
- environment_versions.json
- robot_usd_dependencies.json
- control_and_termination_audit.json
- trace.jsonl.gz

建议一并保留 files/（27份依赖源码）、nav_heading_checks.json、independent_static_review.json，体积较小。运行时13份原始冻结模块在上一级 sources/；交付代码若已收录同SHA文件，可仅保留源哈希清单。权重和11个USD实体放交付资产目录，本evidence不重复复制。无需复制5000余帧sensors或大视频。

原始trace 7,801,097 bytes；trace.jsonl.gz为同内容压缩副本，完整未裁剪。
