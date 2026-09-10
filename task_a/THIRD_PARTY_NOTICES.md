# 来源与许可证

`source/` 和根目录 `LICENSE` 来自 ATEC 官方仓库的固定提交，保留原文件。
官方根许可证为 MIT；其 `setup.py` 的历史 metadata 同时写有 Apache 2.0，未在此替上游解决该元信息差异。

`assets/DDT_Lab/` 是从用户提供的 `workspace2.zip` 本地导入的运行依赖目录，不包含在当前公开 Git 文件中。导入器按 `provenance/external_assets.json` 核验原始文件。
该包没有完整的根 LICENSE/COPYING/NOTICE 文件。`d1_description/package.xml` 与
`rl_controller/package.xml` 的声明仍为 `TODO: License declaration`；`play_g2/package.xml`
声明 Apache-2.0；DDT 的 setup.py 顶部为 BSD-3-Clause、其 metadata 为 Apache-2.0。
这些声明不构成对所有 USD、ONNX 或混合来源文件统一改授 MIT 的依据；这里原样保留线索。
本次公开合并只迁入控制与训练代码、自训残差权重及验收记录；原始 USD、基础 ONNX 和 DDT 配置由本地资源提供，不在此重新授予许可证。

`weights/model_1999_final.pt` 为本次在用户电脑上训练的 D1+G2 残差检查点，依赖上述冻结基础 ONNX。

`provenance/files/isaaclab/` 的文件保留 BSD-3-Clause 原头，完整许可证见 `licenses/`。
RSL-RL 参考代码保留原许可证头，仅用于明确已使用的依赖；运行时仍使用环境安装的 rsl-rl-lib 3.0.1。
Isaac Sim、CUDA、OmniPBR.mdl、Isaac 内置材质与运行环境均为外部依赖，没有随包复制。
`provenance/robot_usd_dependencies.json` 保留实际运行中的全部 11 层 USD 依赖证据；其中临时根层是
原始 combined USDA 重写引用路径的生成副本。导入资源后，由适配器在启动时按本地 assets_root 重定位原始层。
