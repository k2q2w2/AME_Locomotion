# GLAD + LSIO（G1 29DoF）

新增 `AME-G1-29DOF-GLAD-v0` / `AME-G1-29DOF-GLAD-Play-v0`，
在现有 AME2-LSIO 输入基础上使用 GLAD 地形编码。
Actor 与 Critic **共享** CNN、全局注意力评分器、局部显著性评分器、Query 融合层和 MHA；
各自保留独立的本体投影及输出 MLP。LSIO 长短历史仅进入 Actor。

依据 [GLAD v3，III-D 与 IV-A.2](https://arxiv.org/html/2606.00637v3)，
本次增加 attention pooling、可训练的显著性 Top-K 和稀疏 MHA。
这是保留本项目 LSIO 历史和训练环境的网络变体，不是原论文完整训练系统的复现。
原 AME、LSIO 任务及检查点加载路径保持原行为。

## 启动

在已安装本项目与本地 `rsl_rl` 的 Isaac Lab Python 环境中，从项目根目录运行：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-GLAD-v0 --headless
```

GLAD 默认启用全局分支，无需追加 `agent.policy.attach_global=True`。
日志位于 `logs/rsl_rl/g1_ame_glad/`，首次训练从头开始。

恢复训练（将 `RUN_DIRECTORY` 替换成已有 GLAD 运行目录）：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-GLAD-v0 \
  --resume --load_run RUN_DIRECTORY --checkpoint model_100.pt --headless
```

Play 和局部注意力可视化：

```bash
python scripts/rsl_rl/play.py --task AME-G1-29DOF-GLAD-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_glad/RUN_DIRECTORY/model_100.pt \
  --num_envs 1 --save_attention_weights --vis_attention --video_length 300
```

需要录屏时追加 `--video`；无窗口运行可追加 `--headless`。
训练和 Play 必须使用相同网络配置；例如改变保留特征数时，二者均追加 `agent.policy.top_k=16`。
本次不提供 ONNX/JIT 导出或独立的全局注意力可视化。

## 输入与网络

环境直接复用 LSIO Train/Play 配置；观测顺序、历史推进与重置、噪声、地图、奖励、
动作缩放、控制频率、地形课程及 PPO 参数都沿用现有任务。
两阶段训练仍由原环境中的 `FINETUNE` 控制，第二阶段恢复第一阶段的 GLAD 检查点。
历史输入和阶段细节见 [LSIO 说明](lsio.md)。

- `policy`：`[B,2082]`，当前速度命令 3 + XYZ 地图 2079。
- `proprio_history`：`[B,66,93]`，仅供 Actor 使用。
- `critic`：`[B,2178]`，当前特权本体/命令 99 + XYZ 地图 2079。

默认结构如下，`p` 在 Actor 中为 519 维，在 Critic 中为 99 维：

```text
Actor: 66 帧长历史 -> 两层 Conv1d -> 144
       最近 4 帧直接展开 -> 372
       当前速度命令 -> 3
       concat -> p_actor: 519
Critic: 当前特权本体/命令 -> p_critic: 99

XYZ 地图 -> 共享 CNN -> F: [B,187,64]
  全局: Linear(64,1) -> softmax(点维) -> 加权求和 F -> c: 64
  局部: 独立 Linear(64,1) -> 显著性 Top-32 -> K/V: [B,32,64]

p -> 各自独立 Linear -> 64
concat(c, 本体投影) -> 共享 Linear(128,64) -> Q
共享 16 头 MHA(Q, K, V) -> f: 64

Actor:  concat(c, f, p_actor)  -> 647 -> MLP(512,256,128) -> 29
Critic: concat(c, f, p_critic) -> 227 -> MLP(512,256,128) -> 1
```

共享指同一组参数；Actor 和 Critic 使用各自观测分别执行编码。
两者损失均更新共享地形编码器，Critic 不读取 Actor 历史。
GLAD 用线性 attention pooling 替换 AME2 的全局 MLP/max-pool；模型不保留未使用的全局 MLP。
显著性分数只由地形特征产生，本体状态通过后续 Query 影响局部注意力。

## 选择机制与接口

默认 `top_k=32`、`gumbel_temperature=1.0`、`long_history_length=66`、
`short_history_length=4`，保持关闭观测归一化。
`top_k` 必须是 1 到地形 token 数之间的整数，温度必须有限且大于零；
GLAD 不允许关闭 `attach_global`。

训练模式下，对显著性分数加入独立 Gumbel 噪声，选择不重复的 Top-K 索引。
令 `p = softmax((scores + gumbel) / temperature)`，将选中的特征乘以
`1 + (p_selected - p_selected.detach())`：前向保留原始硬选择特征，
反向通过 softmax 将梯度传回显著性评分器。
论文描述了 straight-through softmax weighting，但没有给出完整实现公式；
上述门控是本项目明确采用的具体实现。

选择行为依据 `model.training`，不依据是否启用梯度：
训练 rollout 的 `torch.inference_mode()` 中仍加入 Gumbel 噪声，PPO 重算时也重新采样。
沿用现有 PPO/storage，不额外存储或回放选择噪声。
评估模式使用无噪声、确定性的硬 Top-K，现有 Runner 的 `get_inference_policy()` 会切换至评估模式。
直接调用模型推理时先使用 `model.eval()`。

`act_inference()` 继续返回 `(actions, attention_weights)`。
局部 MHA 在 K 个 token 上计算注意力，再按原空间索引散射到完整网格；
默认输出 `[B,29]` 和 `[B,1,187]`，未选位置权重为零、总权重为一。
可视化显示局部 MHA 权重，不是全局池化权重或显著性分数。

检查点保存架构、历史、地图、Top-K 和温度元数据，配置不匹配会报错。
仅支持同配置 GLAD 检查点恢复，不自动迁移 AME/LSIO 权重。
恢复包含优化器及迭代计数，不包含模拟器、历史缓冲或随机数生成器状态；
评估推理可精确复现，恢复后的训练轨迹不承诺逐步一致。

## 验证

CPU 检查：

```bash
PYTHONPATH="$PWD/rsl_rl" python -m pytest -q \
  tests/test_glad.py tests/test_lsio.py tests/test_lsio_history.py
```

本次结果：**43 项通过**（其中新增 GLAD 22 项）。覆盖共享参数的双分支梯度、
历史隔离、全局池化、Top-K 边界与直通梯度、训练随机选择、评估确定性、空间权重映射、
任务注册与真实 Runner 配置、两轮合成环境 PPO 更新、保存/恢复及继续训练，
并回归旧 AME 预训练检查点与 LSIO 原生历史测试。

当前环境 `torch.cuda.is_available()` 为 `False`，以下机器人仿真训练、恢复和 Play **尚未验证**。
GPU 可用后运行：

```bash
python scripts/rsl_rl/train.py --task AME-G1-29DOF-GLAD-v0 \
  --num_envs 16 --max_iterations 2 --run_name glad_smoke --headless \
  env.episode_length_s=0.2

python scripts/rsl_rl/train.py --task AME-G1-29DOF-GLAD-v0 \
  --num_envs 16 --max_iterations 2 --resume --load_run RUN_DIRECTORY \
  --checkpoint model_1.pt --headless env.episode_length_s=0.2

python scripts/rsl_rl/play.py --task AME-G1-29DOF-GLAD-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_glad/RESUMED_RUN_DIRECTORY/model_2.pt \
  --num_envs 16 --save_attention_weights --vis_attention --video_length 100 \
  env.episode_length_s=0.2
```

运行目录以脚本输出为准。CPU 检查不代表物理仿真通过；行走效果、训练吞吐与性能提升需完整训练评估。
GLAD 不减少 LSIO rollout 历史存储，环境数量仍需根据可用显存设置。
