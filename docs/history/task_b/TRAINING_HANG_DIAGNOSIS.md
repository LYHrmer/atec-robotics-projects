# 训练入口死锁：已定位到原因，并有本机已验证的绕行路径

2026-09-14/15。**这是诊断，不是已修好的训练。**

## 现象

`scripts/rsl_rl/train.py --task ATEC-Isaac-Velocity-Flat-Unitree-B2-v0 --headless` 会死锁：
Isaac 到 `app ready`（6.6 s）之后，进程 130% CPU 空转、**日志不再增长**、
**永远不会创建 `logs/rsl_rl/` 目录**。

**关键证据：原版 B2 任务也死锁** ⇒ 不是我们建的 B2W 配置的问题。

已排除：
- X 显示（Xorg `:1` 真实可用，`xdpyinfo` 能连）
- 着色器编译（缓存数分钟未动）
- 环境数量（16 与 256 一样卡）
- hydra 本体（`import hydra` 0.10 s，gymnasium 注册 63 个环境）

**另一个关键证据：整台机器上找不到任何 `logs/rsl_rl/` 目录** ⇒
**这个训练入口在本机从未成功跑出过一次。** 所以不是"最近坏了"。

## 绕行路径（本机已证实可用）

`logs/taskb_f21b_b2w_safe_forward_micro_training/worker_*.py` 里有一份**实际跑起来过**的
训练 worker，它**完全绕开 `train.py` 和 hydra**：

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
import gymnasium as gym
import atec_rl_lab.tasks              # 注册任务
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg(TASK, device=..., num_envs=..., ...)
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
```

**`parse_env_cfg` + `gym.make` 在本机是验证过能工作的。**
`train.py` 用的是 `@hydra_task_config` 装饰器解析配置——**那是死锁所在**。

## 下一步

写一个自定义训练脚本，走上面这条已验证的路径（AppLauncher → `atec_rl_lab.tasks` →
`parse_env_cfg` → `gym.make` → `RslRlVecEnvWrapper` → `OnPolicyRunner`），
**不使用 hydra**。B2W 的环境配置本身已经写好并通过 CPU 核验
（`b2w_locomotion/`，仅新增文件）。

**尚未写、尚未运行。** 本文只是把"卡在哪"和"从哪里绕"写清楚。
