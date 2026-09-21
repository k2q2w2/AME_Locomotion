# GLAD CriticCleanActorStopGrad

训练任务：`AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-v0`。  
Play 任务：`AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-Play-v0`。  
实验目录：`logs/rsl_rl/g1_ame_glad_critic_clean_actor_stop_grad/`。

本任务将干净本体历史、命令和干净地图送入同一套 Actor 编码器，地形特征 detach 后
交给价值 MLP。没有额外复制可训练编码器，也不使用随机冻结的 Critic 私有投影生成 Query。
原 Critic 私有投影保留注册并冻结，但不参与本任务前向。

```text
clean_proprio_history [B,66,93]
  → Actor LSIO：长历史144维 + 最近4帧372维
  → 拼接 Critic 当前命令3维 → 本体表示519维 → Actor Linear(519,64)
                                                                   ↓
Critic 干净地图 → 同一 CNN → 全局特征 g_C → 同一 Query 融合层 → Query
                       └→ 独立 Top-K → 同一 MHA(Query,K,V) → f_C

[detach(g_C):64, detach(f_C):64, Critic 当前特权本体:99]
  → 价值 MLP 输入227维 → V
```

## 观测与更新边界

Actor 原有的 `policy`、`proprio_history` 不变。新增 `clean_proprio_history`，逐帧顺序为
角速度3、投影重力3、关节相对位置29、关节速度29、上一动作29。它从 Critic 观测项构造，
缩放与 Actor 相同，始终无观测噪声；不包含线速度、命令和地图。这里“干净”仅指传感观测不加噪，
并不移除推力、质量/摩擦随机化或真实动作响应。

命令从现有 `critic` 的 `[9:12]` 取出，地图从其99维本体之后取出；基座线速度等当前特权状态
仍直接进入价值 MLP。两组历史使用原生缓冲，在每个控制步同步推进，局部重置时以该环境首帧
填充历史；网络调用和 PPO minibatch 不推进历史。PPO 保存两份原始历史快照，不保存编码结果。

Critic 的 LSIO、Actor 私有投影和地形编码在 `no_grad` 内执行，输出再显式 detach。
价值损失只更新价值 MLP。Actor 损失仍训练全部原有 Actor 分支，包括它的 LSIO 和私有投影。
Critic CNN 使用复制的 BN buffers：训练使用当前批次统计但不写回运行统计，评估使用已有统计。
独立价值估计不调用动作 MLP、不采样动作、不改写策略分布。

训练时 Actor 和 Critic 分别编码，分别采样随机 Top-K；评估时都是确定性 Top-K。
沿用已有 Gumbel/直通梯度、PPO 概率重算和 storage，不新增选择噪声回放。
PPO 优势估计、损失、优化器、全模型梯度裁剪与多卡梯度同步均保持原样。

| 模式 | Critic 地形 | Critic Query 的本体来源 | 价值损失更新编码器 |
|---|---|---|---|
| 普通 GLAD | 干净地图 | 可训练 Critic 私有投影 | 是 |
| CriticStopGrad | 直接复用 Actor 地形特征 | Actor 观测历史 | 否 |
| CriticCleanStopGrad | 干净地图 | 随机冻结 Critic 私有投影 | 否 |
| **CriticCleanActorStopGrad** | **干净地图** | **干净历史经 Actor LSIO/私有投影** | **否** |

新模式不是纯粹的共享梯度消融：Critic 的 Query 还改为依赖干净历史和 Actor 学到的参数。
这不保证性能提升，需要独立完整训练评估。Play 的动作网络结构与普通 GLAD 相同。

## 从头训练与恢复

以下命令在项目根目录、已安装本项目的 Isaac Lab Python 环境中运行。
默认从头训练；保持原始 `FINETUNE=False` 开始阶段1：

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-v0 \
  --num_envs 2048 --max_iterations 10000 --headless
```

同模式恢复，`--max_iterations` 是追加轮数，替换运行目录和实际存在的检查点文件名：

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-v0 \
  --resume --load_run RUN_NAME --checkpoint model_9999.pt \
  --num_envs 2048 --max_iterations 10000 --headless
```

finetune 阶段续训仍采用本项目现有阶段机制：启动新进程前，将
`source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py`
中的 `FINETUNE` 设为 `True`，然后运行上述恢复命令，从本新任务阶段1检查点继续训练。
此开关同时配置地形、噪声、扰动和奖励，不能只替换地形生成器就当作完整阶段2。
本次实现保留源码默认 `False`，不改变既有实验的阶段。finetune 时 Actor 观测按原配置加噪，
价值路径的地图和历史继续保持干净。

双卡训练，每卡2048环境：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
bash run_train_multi_gpu.sh \
  --task AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-v0 --max_iterations 10000
```

双卡 finetune 恢复同样先设 `FINETUNE=True`，再执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
bash run_train_multi_gpu.sh \
  --task AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-v0 \
  --resume --load_run RUN_NAME --checkpoint model_9999.pt --max_iterations 10000
```

## Play

```bash
python scripts/rsl_rl/play.py \
  --task AME-G1-29DOF-GLAD-CriticCleanActorStopGrad-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_glad_critic_clean_actor_stop_grad/RUN_NAME/model_9999.pt \
  --num_envs 1 --real-time
```

注意力显示可追加 `--save_attention_weights --vis_attention`；无窗口短录制可追加
`--headless --video --video_length 100`。Play 继承原 Play 地形设置，不会自动恢复训练地形。
可以使用 `bash run_play.sh` 代替上面的 `python scripts/rsl_rl/play.py`，其余参数相同。

若 Conda 中遇到 C++ 动态库版本冲突，可在 Python 命令前增加
`LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`；`run_play.sh` 已处理此环境设置。

## 内存和检查点

新增历史 storage 在4096环境、24步、float32下约为2.25 GiB；双卡每卡2048环境约新增
1.12 GiB/进程，另有原 Actor 历史、地图和训练中间量。增加的是观测存储，不是模型参数。
如修改历史长度，需同时设置两组环境历史长度和 `agent.policy.long_history_length`，
否则模型启动时报错；新长度检查点只在同配置下恢复。

顶层保存 `critic_encoder_stop_grad=True`、`critic_feature_source="actor_clean"`，
模型架构元数据为 `glad_lsio_clean_actor_v1`，并检查历史、地图、Top-K 等配置。
仅恢复本模式、同结构检查点；普通 GLAD、两个旧 StopGrad 模式及缺少来源标记的检查点
均拒绝加载，不提供隐式迁移。Play 和 `load_optimizer=False` 也执行校验，校验先于状态修改。
恢复模型、优化器和迭代计数，不恢复模拟器、历史缓冲或随机数生成器的现场。

## 验证

CPU 回归与新测试：

```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
PYTHONPATH=rsl_rl:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q tests scripts/rsl_rl/tests
```

新测试位于 `scripts/rsl_rl/tests/test_critic_clean_actor_stop_grad.py`，覆盖干净历史、Actor 编码
数值一致性、输入依赖、Adam 动量下的隔离、Actor 梯度、原任务行为、rollout/bootstrap、
PPO 更新、检查点保存恢复及跨模式拒绝。测试合成环境不代表机器人已学会行走。

2026-09-22：CPU 全量回归 **184项通过**。本地 RTX 5080 已完成16环境、2轮训练，恢复
`model_1.pt` 后追加1轮，并完成16环境、20步无窗口 Play 和视频录制，均正常退出。
检查点模式/架构标记正确，恢复后 Actor 和价值网络继续更新，未使用的 Critic 私有投影不变。

训练记录：`logs/rsl_rl/g1_ame_glad_critic_clean_actor_stop_grad/2026-09-22_02-51-18_clean_actor_smoke/`；
恢复记录：同目录下的 `2026-09-22_02-53-31_clean_actor_resume_smoke/`。
详细验证记录与进程输出在 `output/critic_clean_actor_smoke_2026-09-22/verification.json` 及同目录日志。
冒烟只验证真实环境与网络接口，不代表策略已学会行走或性能得到提升。
本机只有一张 GPU，**双卡 NCCL 未验证**。本次未部署远端，也未调整正在运行的实验。
