# AME + LSIO（G1 29DoF）

本实现继承 LSIO 的双历史结构，适配 G1 和 AME；只改变 Actor 的本体输入路径。
依据 [LSIO 论文第 V 节](https://arxiv.org/html/2401.16889v2)及
[作者网络代码](https://github.com/HybridRobotics/cassie_rl_walking/blob/master/ppo/policies.py)，
使用长历史时序卷积和短历史直接输入。地形编码、动作 MLP、奖励、动作缩放、可学习标准差与 PPO 参数沿用 AME。
没有加入 Cassie 参考动作、低通滤波或额外的动作输出限制。

## 启动

在已安装本项目扩展和本地 `rsl_rl` 的 Isaac Lab Python 环境中，从项目根目录运行。
原 `AME-G1-29DOF-v0` / `AME-G1-29DOF-Play-v0` 及其检查点继续使用原配置。
LSIO 使用独立目录 `logs/rsl_rl/g1_ame_lsio/`，第一次训练从头开始：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-LSIO-v0 --headless
```

恢复 LSIO 训练（`RUN_DIRECTORY` 替换为该实验目录中的一次运行目录名）：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-LSIO-v0 \
  --resume --load_run RUN_DIRECTORY --checkpoint model_100.pt --headless
```

Play 并显示注意力（现有脚本需要同时开启保存权重开关才绘制注意力）：

```bash
python scripts/rsl_rl/play.py --task AME-G1-29DOF-LSIO-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_lsio/RUN_DIRECTORY/model_100.pt \
  --num_envs 1 --save_attention_weights --vis_attention --video_length 300
```

需要录屏时追加 `--video`；无窗口运行可追加 `--headless`。
模型推理返回 `(actions, attention_weights)`，默认形状分别为 `[B,29]`、`[B,1,187]`。
本次不提供 LSIO ONNX/JIT 导出。

启用全局地形分支：训练和 Play 都追加 `agent.policy.attach_global=True`。
`long_history_length=66`、`short_history_length=4` 在 `LSIOActorCriticCfg` 中公开。
如需改变长历史，必须同时修改 `env.observations.proprio_history.history_length` 和
`agent.policy.long_history_length`；网络启动时校验历史形状，改变长度后通常需要重新训练。
默认关闭观测归一化，LSIO 会明确拒绝开启归一化的配置。

检查点记录历史与地形结构元数据；配置不同会报错。旧 AME 检查点传给 LSIO 时会明确报告架构不匹配，
不会迁移部分权重。恢复训练同时恢复优化器与迭代计数，不保存模拟器状态或上一回合的历史缓冲。

## 输入布局与网络

| 观测组 | 形状 | 顺序 |
|---|---|---|
| `policy` | `[B,2082]` | 当前速度命令 3 + 当前 XYZ 地图 2079 |
| `proprio_history` | `[B,66,93]` | 最旧到最新的 66 帧，每帧如下表 |
| `critic` | `[B,2178]` | 原有特权本体 99 + 当前 XYZ 地图 2079 |

每帧历史固定顺序；索引采用 Python 左闭右开范围：

| 特征 | 索引 | 缩放 | 第二阶段均匀噪声（缩放前） |
|---|---|---|---|
| 基座角速度 | `0:3` | 0.2 | ±0.2 |
| 投影重力 | `3:6` | 1 | ±0.05 |
| 关节相对位置 | `6:35` | 1 | ±0.01 |
| 关节相对速度 | `35:64` | 0.05 | ±2.0 |
| 上一动作 | `64:93` | 1 | 无 |

地图、速度命令、基座线速度不进入历史。每帧为 `(当前状态, 上一动作)`，本帧动作由网络之后产生。
原生 `ObservationManager` 先加噪声、再缩放、再存入历史；已存储帧不重新采样噪声。
每个环境重置时仅清除自身历史，并用重置后的首帧填满窗口，此时上一动作由环境清零。
历史随环境控制步更新，读取观测、调用网络与 PPO 小批次训练均不推进历史。

```text
[B,66,93] -> transpose -> [B,93,66]
  Conv1d(93,32,k=6,s=3,p=0) -> ReLU -> [B,32,21]
  Conv1d(32,16,k=4,s=2,p=0) -> ReLU -> [B,16,9]
  Flatten -> 144

最后 4 帧直接展开 -> 372
当前速度命令 -> 3
concat(144,372,3) -> 519
  Linear(519,64) -> Query
当前 XYZ 地图 -> 现有 CNN -> 187 个 64 维 Key/Value
  16 头 MHA -> 64
concat(注意力 64, 本体 519) -> 583 -> MLP(512,256,128) -> 29
```

`attach_global=True` 时沿用全局点特征 MLP 和最大池化，64 维全局特征参与 Query 融合，
并拼入 Actor，动作 MLP 输入为 647。Critic 使用当前 99 维特权本体的独立投影，
与 Actor 共享地形 CNN/MHA；其结构仍是原 AME 的 Critic（全局开关的行为也相同）。
长、短历史都不进入 Critic。网络 `is_recurrent=False`，PPO 保存每个采样时刻的完整历史快照，
打乱的是样本，历史内部顺序保持不变。

保留现有 50 Hz 控制频率，66 个样本的首尾跨度为 `(66-1)/50 = 1.30` 秒；
原 LSIO 在约 33 Hz 下覆盖约两秒。本实现的时域、G1 的 93 维输入和 AME 动作 MLP 均有适配，
不应视为原 Cassie 控制器的完整复现。

## 两阶段训练

沿用 `velocity_env_cfg_29dof.py` 中的 `FINETUNE` 设置：

- 第一阶段（默认 `False`）：本体与地图均不加观测噪声。
- 第二阶段（`True`）：继承上表本体噪声、地图噪声及原环境随机化；从第一阶段 **LSIO** 检查点恢复。
- Play：本体观测 corruption 关闭，地图噪声关闭，无论 `FINETUNE` 的值。

LSIO 在父任务完成阶段/Play 配置后再拆分观测组，保持这些设置和原有缩放。

## 验证

CPU 测试（网络测试只需要本地 RSL-RL 依赖；原生历史测试还需要已安装 Isaac Lab）：

```bash
PYTHONPATH="$PWD/rsl_rl" python -m pytest -q tests/test_lsio.py tests/test_lsio_history.py
```

本次 CPU 验证结果：**21 项通过**，包括两轮 PPO 更新、保存/恢复后推理完全一致及恢复后继续更新。

覆盖网络维度及全局分支、双历史路径、有限梯度、完整历史存储/打乱、CPU 合成环境 PPO 更新与恢复，
以及真实 `pretrained/ame1.pt`、`pretrained/ame2.pt` 在旧网络中的加载。
历史测试使用安装版本的原生观测管理器与缓冲区，输入为合成传感器值；
为避免启动仿真，隔离了未调用的 `omni.timeline` 导入，并以配置树检查阶段设置。
这些是 CPU 单元/集成测试，不代表机器人仿真通过。

GPU 可用后执行 16 环境、两轮训练、恢复训练和短时 Play 冒烟测试：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-LSIO-v0 \
  --num_envs 16 --max_iterations 2 --run_name lsio_smoke --headless \
  env.episode_length_s=0.2

python scripts/rsl_rl/train.py --task AME-G1-29DOF-LSIO-v0 \
  --num_envs 16 --max_iterations 2 --resume --load_run RUN_DIRECTORY \
  --checkpoint model_1.pt --headless env.episode_length_s=0.2

python scripts/rsl_rl/play.py --task AME-G1-29DOF-LSIO-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_lsio/RESUMED_RUN_DIRECTORY/model_2.pt \
  --num_envs 16 --save_attention_weights --vis_attention --video_length 100 \
  env.episode_length_s=0.2
```

短回合便于实际检查自动重置。运行目录以脚本输出为准；恢复两轮后 `model_2.pt` 的名称遵循现有 Runner 计数规则。
默认 4096 环境、24 步 rollout 时，单历史快照数组约占 **2.25 GiB** float32 内存/显存，另有地图、梯度及小批次中间量；可用 `--num_envs` 调整。

本次环境中 `nvidia-smi` 无法连接驱动，因此 GPU 训练、物理仿真自动重置及注意力实际显示**尚未验证**。
验收关注训练和推理接口；行走效果或性能提升需要后续完整训练与对照实验。
