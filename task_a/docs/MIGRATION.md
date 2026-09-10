# 合并到统一仓库

当前 `task_a/` 包含 Task A 的运行/训练代码、自训残差权重、原框架、通关和失败证据。Task E 的既有入口保留在仓库根目录；首页和 `run.sh` 负责统一导航。

## 保留了什么

- 源版本为原 Task A 提交 `cdc4dbad5c690d7b86d2d304eb1c9494bf103cdf`，完整通关运行编号 `residual_1999_axis_recovery_course_02`。
- 冻结的推理、导航、恢复、训练与框架源码直接迁入；不因整理目录而重新训练或修改控制逻辑。
- 原始结果、完整 trace、源码哈希和独立验收记录保留原值。它们证明的是原运行，不是把本次短测冒充完整通过。
- 从原始日志补入台阶诊断的 `cpu_acceptance_samples.jsonl`，修复旧验收页的样本链接；页面中的旧本机验收命令已标明其依赖范围。
- `provenance/original_private_manifest.json` 是原私有包的历史清单；当前公开文件以本目录 `SHA256SUMS` 为准。

## 外部依赖

原私有包中的 `assets/DDT_Lab/` 未公开复制，仍由用户本地的 workspace2.zip 或 DDT_Lab 目录导入。文件名单与期望 SHA 在 `provenance/external_assets.json`。
两份含第三方原始内容的证据副本也未公开复制：`sources/3_ddt_robot.py`、`sources/4_d1_with_g2_combined.usda`。它们的原运行哈希继续保留在原验收清单中；历史清单引用它们不代表当前公开目录应有实体。
旧私有包作为历史和资产备份保留；以后开发和文档更新集中到本仓库。

## 本次验证

资产导入、隔离后的 Python 包路径、CPU 推理和新目录 GPU 启动检查分别保存为 `provenance/merge_validation.json`。GPU 短测只验证迁移后的运行入口与资源路径，不用于新增成功率结论。
