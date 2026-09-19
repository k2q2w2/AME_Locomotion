# Critic 复用 Actor 地形特征：AME / LSIO / GLAD 消融

新增三组 Train / Play 任务，复用原任务的环境、观测、奖励、网络尺寸和 PPO 配置。训练默认从头开始。

- AME：`AME-G1-29DOF-CriticStopGrad-v0`，日志目录 `logs/rsl_rl/g1_ame_critic_stop_grad/`。
- LSIO：`AME-G1-29DOF-LSIO-CriticStopGrad-v0`，日志目录 `logs/rsl_rl/g1_ame_lsio_critic_stop_grad/`。
- GLAD：`AME-G1-29DOF-GLAD-CriticStopGrad-v0`，日志目录 `logs/rsl_rl/g1_ame_glad_critic_stop_grad/`。

对应 Play 任务将结尾的 `-v0` 替换为 `-Play-v0`。AME、LSIO 默认 `attach_global=False`；GLAD 始终开启全局分支。

## 隔离的范围

网络参数 `critic_encoder_stop_grad` 默认 `False`；上述任务配置为 `True`。新版本 Critic 直接使用 Actor 提取的注意力与全局地形特征，在价值网络入口 `detach()` 后拼接自己的 99 维特权本体观测（含指令和上一动作），只训练价值 MLP。

```text
Actor 地形与本体/历史 → Actor 编码器 → 注意力特征、全局特征 → Actor 动作网络
                                             ↓ detach
                            拼接 Critic 的 99 维特权本体观测
                                             ↓
                                          价值 MLP
```

默认结构按 `[注意力特征, Critic 本体观测]` 拼接，价值 MLP 输入为 163 维；开启全局分支时按 `[全局特征, 注意力特征, Critic 本体观测]` 拼接，输入为 227 维。Actor 的 CNN、本体投影、LSIO 历史编码器、Query 融合层、MHA 和 GLAD 评分器均不接收价值损失梯度。旧 `critic_proprio_embedding` 保留为冻结且不参与消融前向的参数，以保持权重键及参数布局；LSIO / GLAD 架构元数据不变。

**这也改变了 Critic 的信息来源，不是单纯的共享编码器梯度消融。** 环境观测布局保留，但 Critic 的地形尾部不再参与价值计算。微调阶段使用的是 Actor 带噪地形的特征，LSIO / GLAD 的价值预测也会间接依赖 Actor 历史。普通任务仍使用自己的 Critic 地形观测与独立 Query，并联合训练共享编码器。

## 同一次编码与独立价值估计

PPO 在消融任务的 rollout 和 minibatch 更新中调用 `act_and_evaluate(obs, **kwargs)`：Actor 编码一次，动作与价值使用同一组地形特征。GLAD 复用该次随机 Top-K，不为 Critic 再次采样。特征只作为函数参数传递，不设置跨调用缓存，也不保存到 rollout storage；每个 minibatch 都用当前参数重新编码。

`evaluate(obs, actor_terrain_features=None, **kwargs)` 可以显式接收当前观测对应的 Actor 地形特征，并统一在价值网络入口 detach。调用者必须保证特征与观测属于同一批次、同一次编码；普通任务不接受该特征参数。原有 `act()` 和 `act_inference()` 返回接口不变。

末步 bootstrap 或独立 `evaluate(obs)` 没有现成特征时，在 `torch.no_grad()` 内重新编码当前 Actor 观测，再正常调用价值 MLP。此路径不采样动作、不改写策略分布，也不更新 BatchNorm 运行状态。仅 CNN 使用 `torch.func.functional_call` 替换复制的 buffers；不再逐层替换 detached 参数，也不原地恢复 buffers。训练时仍使用批次统计和 GLAD 随机选择，评估时使用运行统计和确定性硬 Top-K。Actor 正常调用时照常更新 BatchNorm；Play 注意力接口不变。

PPO 的优势估计、单一 Adam 优化器、全模型梯度裁剪和多卡梯度同步保持原样。价值损失不能直接改变共享参数或共享状态，但仍可通过优势估计和共同梯度裁剪间接影响策略学习；这不属于本次消融的隔离范围。

## 训练、恢复和 Play

在项目根目录、已安装本项目的 Isaac Lab Python 环境中执行。下列恢复示例中的 `RUN_DIRECTORY` 替换为对应实验目录内的运行文件夹名，`model_100.pt` 替换为实际检查点文件名。

### AME

```bash
# 从头训练
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-CriticStopGrad-v0 \
  --headless --num_envs 2048 --seed 42

# 恢复同配置的消融检查点
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-CriticStopGrad-v0 \
  --headless --num_envs 2048 --resume \
  --load_run RUN_DIRECTORY --checkpoint model_100.pt

# Play
python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-CriticStopGrad-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_ame_critic_stop_grad/RUN_DIRECTORY/model_100.pt

# 两卡，每卡 2048 个环境
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
  bash run_train_multi_gpu.sh --task AME-G1-29DOF-CriticStopGrad-v0 \
  --max_iterations 10000 --seed 42
```

### LSIO

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 \
  --headless --num_envs 2048 --seed 42

python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 \
  --headless --num_envs 2048 --resume \
  --load_run RUN_DIRECTORY --checkpoint model_100.pt

python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_ame_lsio_critic_stop_grad/RUN_DIRECTORY/model_100.pt

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
  bash run_train_multi_gpu.sh --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 \
  --max_iterations 10000 --seed 42
```

### GLAD

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticStopGrad-v0 \
  --headless --num_envs 2048 --seed 42

python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticStopGrad-v0 \
  --headless --num_envs 2048 --resume \
  --load_run RUN_DIRECTORY --checkpoint model_100.pt

python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-GLAD-CriticStopGrad-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_ame_glad_critic_stop_grad/RUN_DIRECTORY/model_100.pt

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
  bash run_train_multi_gpu.sh --task AME-G1-29DOF-GLAD-CriticStopGrad-v0 \
  --max_iterations 10000 --seed 42
```

Runner 默认 `max_iterations=10000`、每轮采集 24 步、每 100 轮保存一次，与各原任务一致。现有多卡脚本自身默认传入 `--max_iterations 15000`，所以上述多卡示例显式指定 10000；可按实验需要更改。恢复训练沿用原 Runner 的迭代语义：请求的迭代次数从已加载的迭代位置继续执行。

多卡恢复时，同样在脚本末尾追加 `--resume --load_run RUN_DIRECTORY --checkpoint model_100.pt`。`NUM_ENVS` 为每卡环境数，比较单卡、多卡实验时需控制总采样量。

## AME2 全局分支开关

AME 和 LSIO 可在命令末尾追加 `agent.policy.attach_global=True`。恢复、Play 和多卡训练必须使用与检查点一致的开关。建议用不同运行名区分实验。

```bash
# AME2 消融
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-CriticStopGrad-v0 --headless \
  --run_name ame2 agent.policy.attach_global=True

# AME2-LSIO 消融
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 --headless \
  --run_name ame2_lsio agent.policy.attach_global=True

# AME2-LSIO 恢复
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 --headless --resume \
  --load_run RUN_DIRECTORY --checkpoint model_100.pt \
  agent.policy.attach_global=True

# AME2-LSIO Play
python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-LSIO-CriticStopGrad-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_ame_lsio_critic_stop_grad/RUN_DIRECTORY/model_100.pt \
  agent.policy.attach_global=True

# AME2-LSIO 两卡
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
  bash run_train_multi_gpu.sh --task AME-G1-29DOF-LSIO-CriticStopGrad-v0 \
  --max_iterations 10000 --run_name ame2_lsio agent.policy.attach_global=True
```

`agent.policy.attach_global=False` 显式选择 AME / LSIO 默认结构。GLAD 依赖全局分支，不能设为 `False`。

## 检查点规则

新消融 Runner 保存顶层 `critic_encoder_stop_grad=True` 和 `critic_feature_source="actor"`。普通任务保存格式保持原样，旧普通检查点缺少 stop-grad 字段时视为 `False`。Runner 在加载任何模型、优化器或迭代状态前检查训练模式和特征来源；普通任务与消融任务之间的加载会报错，Play 或 `load_optimizer=False` 也不会绕过检查。

只有同类、同配置的新版本消融检查点可以恢复训练。旧版 CriticStopGrad 使用独立 Critic Query，即使带有 `critic_encoder_stop_grad=True`，缺少新的来源标记仍会被拒绝，必须从头训练新版本。模型权重键名、LSIO / GLAD 的 `_extra_state` 架构校验保持原样；本次不提供跨模式或跨版本 warm-start，不要手工修改标记绕过校验。该检查由 Runner 负责，直接调用底层网络的 `load_state_dict` 不读取顶层检查点标记。

## 验证

CPU 回归命令：

```bash
PYTHONPATH="$PWD/rsl_rl" python -m pytest -q tests
```

`tests/test_critic_stop_grad.py` 覆盖 AME / LSIO 全局分支开关及 GLAD：只有价值 MLP 获得 Critic 梯度、已有 Adam 动量下参数与状态隔离、Actor 和 LSIO 梯度、裁剪前联合梯度等于单独 Actor 梯度、同一次编码与 Top-K 复用、独立价值估计的状态与策略分布保持、保留 Actor 计算图后的安全反传、Actor 前向与注意力兼容、观测来源依赖、rollout / bootstrap / PPO 编码次数、任务配置继承、检查点模式和来源校验、合成环境保存恢复及继续训练。新旧版本 Critic 使用不同特征，不要求价值输出一致。独立编码参考值使用 dtype 对应的浮点容差；复用特征、受保护参数与 buffers、仅 Critic 更新前后的确定性 Actor 输出要求逐元素完全一致。

当前环境 CUDA 不可用，物理仿真、真实单卡/双卡训练、NCCL 同步与 Play 尚未实测。GPU 可用后，对上述三个训练任务分别执行单卡 `--num_envs 16 --max_iterations 2` 和双卡 `NUM_ENVS=16 ... --max_iterations 2` 冒烟测试，再用生成的检查点运行对应 Play。Play 可追加 `--headless --video --video_length 32` 限定录制步数；AME2 变体需同步追加全局分支开关。
