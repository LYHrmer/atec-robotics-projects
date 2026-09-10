# 来源与使用范围

- 任务定义、动作接口及固定环境参数来自 [ATEC 2026 Simulation Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)。其 MIT 版权声明保留在本仓库 LICENSE 中。
- Piper 机器人、物体网格、纹理与仿真场景依赖官方资产。本仓库仅包含控制代码和运行录像，不重新分发资产文件；使用者应遵循原始来源的许可。
- 相机标定和机器人运动学常量依据公开任务/模型定义导出，随机物体抓取位置由运行时图像估计。
- 视频是本地仿真录像，不是真实机器人实验。彩色展示版的处理参数与原片哈希保存在 `media/provenance.json`。

Task A 的第三方依赖和框架声明见 [Task A 来源说明](task_a/THIRD_PARTY_NOTICES.md)。D1/G2 原始 USD、基础 ONNX 与配置作为外部资源导入；公开的 `model_1999_final.pt` 是本次训练的残差检查点。Task A 录像同样来自本机仿真，未编辑环境终止或成绩。
