# 来源与使用范围

- 任务定义、动作接口及固定环境参数来自 [ATEC 2026 Simulation Challenge](https://github.com/atecup/ATEC2026_Simulation_Challenge)。其 MIT 版权声明保留在本仓库 LICENSE 中。
- Piper 机器人、物体网格、纹理与仿真场景依赖官方资产。本仓库仅包含控制代码和运行录像，不重新分发资产文件；使用者应遵循原始来源的许可。
- 相机标定和机器人运动学常量依据公开任务/模型定义导出，随机物体抓取位置由运行时图像估计。
- 代码与诊断过程使用了 Codex 和 Claude Opus。`task_e_held_motion.py` 的规划实现有 Opus 参与，并经过本地审查、运动学检查与仿真验证。
- 本项目未训练新的 RL 或 ACT 权重，也不包含其他参赛者代码、访问凭证或官方项目 Git 历史。
- 视频是本地仿真录像，不是真实机器人实验。彩色展示版的处理参数与原片哈希保存在 `media/provenance.json`。
