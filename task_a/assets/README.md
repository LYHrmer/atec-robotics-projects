# 原始机器人资源

D1/G2 原始 USD、机器人配置和基础 ONNX 是外部依赖，导入目录 `DDT_Lab/` 不提交到 Git。
自训残差权重保存在仓库的 `../weights/`。完整运行说明见 [Task A](../README.md)。

视觉导航还需要原始 ATEC 场景资源：

```bash
python3 task_a/scripts/setup_scene_assets.py --from-directory /path/to/ATEC2026_Simulation_Challenge/atec_robot_model
python3 task_a/scripts/setup_scene_assets.py --check
```

[场景资源清单](../provenance/external_scene_assets.json) 固定了 5 个文件的 SHA-256：原赛道 MDL 材质、3 张纹理及天空 HDR。它们会导入本地 `task_a/atec_robot_model/`，不提交 Git。缺少它们时仿真仍可能启动，但相机图像无法支持既有视觉导航；启动器现在会提前报错。

在统一仓库根目录执行以下任一导入方式（仅使用 Python 标准库，不下载文件）：

```bash
python3 task_a/scripts/setup_assets.py --from-zip '/home/lybm/下载/workspace2.zip'
python3 task_a/scripts/setup_assets.py --from-directory /path/to/DDT_Lab
```

脚本只读取 [外部资源清单](../provenance/external_assets.json) 中的 18 个文件，逐个检查
大小与 SHA256。ZIP 必须包含一份完整的 `DDT_Lab/` 目录；其他内容不会解压。
所有待导入文件先在临时目录完成校验，再原子写入目标；已有文件不符时会报错并保留原文件。
重复导入已验证文件不会覆盖它们。

省略导入选项或使用 `--check` 可检查已有资源：

```bash
python3 task_a/scripts/setup_assets.py --check
```

用 `--destination /path/to/DDT_Lab` 可检查或导入到别处，运行时相应设置
`ATEC_D1G2_ASSET_ROOT=/path/to/DDT_Lab`。原始资料也保留在自己的
[私有 Task A 归档](https://github.com/LYHrmer/atec-taska-d1g2) 的 `assets/DDT_Lab/`，
已有本地检出可直接用 `--from-directory` 导入；脚本不会登录 GitHub 或读取凭据。

启动器会在打开 Isaac Sim 前检查这些资源。清单固定了本次通关使用的文件版本，不能用其他
基础策略或 USD 覆盖后仍把结果当作同一配置。
