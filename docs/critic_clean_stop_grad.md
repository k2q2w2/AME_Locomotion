# GLAD CriticCleanStopGrad

本任务保留 Critic 自己的干净地形扫描和本体 Query，在 GLAD 编码结果后 detach。
Actor 仍使用 LSIO 历史、自己的地形扫描及共享 GLAD 编码器。

```text
Critic 地形 XYZ 2079维 + Critic 本体投影生成的 Query
                    ↓ GLAD
       [全局特征64维, 注意力特征64维].detach()
                    ↓ 拼接 Critic 本体观测99维
               价值 MLP 输入227维
```

价值损失只训练价值 MLP。Critic 私有本体投影参与前向，但固定在从头训练时的随机初始化；
完整 Query 还经过由 Actor 训练的共享融合层，因此 Query 输出不是恒定向量。
CNN、全局评分器、Top-K 评分器、Query 融合层及 MHA 继续由 Actor 训练。
Critic 编码在 no_grad 内执行，CNN 使用复制的 BatchNorm buffers：训练时读取批次统计，
但不会写回共享运行统计；评估时读取已有运行统计。Actor/Critic 各自计算地形特征，
训练时各自随机 Top-K，评估时使用确定性 Top-K。独立价值估计不访问 Actor 历史或策略分布。

与已有任务的区别：普通 GLAD 使用独立 Critic Query 并联合训练共享编码器；
GLAD-CriticStopGrad 使用 detach 后的 Actor 特征；本任务使用 detach 后的 Critic 特征。
阶段2中 Actor 地形有噪声，Critic 地形仍保持干净。本任务不改变奖励、课程、训练参数或
全模型梯度裁剪，因此仍存在优势估计、共同梯度裁剪等间接影响。

## 启动

以下命令在项目根目录、已安装本项目的 Isaac Lab Python 环境中运行。

从头训练：

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticCleanStopGrad-v0 \
  --num_envs 2048 --max_iterations 10000 --headless
```

双卡、每卡2048环境：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 NUM_ENVS=2048 \
bash run_train_multi_gpu.sh \
  --task AME-G1-29DOF-GLAD-CriticCleanStopGrad-v0 --max_iterations 10000
```

恢复本模式的检查点（替换 RUN_NAME；max_iterations 表示追加轮数）：

```bash
python scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-GLAD-CriticCleanStopGrad-v0 \
  --resume --load_run RUN_NAME --checkpoint model_9900.pt \
  --num_envs 2048 --max_iterations 10000 --headless
```

Play（替换 RUN_NAME 和检查点文件名）：

```bash
bash run_play.sh \
  --task AME-G1-29DOF-GLAD-CriticCleanStopGrad-Play-v0 \
  --checkpoint logs/rsl_rl/g1_ame_glad_critic_clean_stop_grad/RUN_NAME/model_9999.pt \
  --num_envs 1 --real-time
```

本任务继承现有 FINETUNE 阶段选择机制，不增加阶段切换开关。Play 仍继承项目当前的固定
桩阵地形设置。若直接用 Python 训练出现 CXXABI 库冲突，可在该命令前加
`LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`；run_play.sh 已自动处理 Conda C++ 库。

## 检查点边界

新检查点顶层标记为 critic_encoder_stop_grad=True、critic_feature_source="critic"。
仅支持同结构、同模式恢复，不提供从普通 GLAD 或 Actor 特征消融导入的迁移入口。
缺少来源标记的旧消融检查点也被拒绝，包括 Play 和 load_optimizer=False。
校验在模型、优化器与迭代状态被修改之前执行；参数键和 GLAD 架构元数据保持不变。

## 验证

CPU 测试覆盖梯度与 BatchNorm 隔离、已有 Adam 动量、Critic Query 实际参与前向、
输入依赖、普通 Critic 前向数值一致性、联合梯度、PPO rollout/bootstrap、保存恢复与
继续训练、跨模式拒绝及任务配置继承。运行：

```bash
PYTHONPATH=rsl_rl python -m pytest -q tests
```

2026-09-20 验证：157 项 CPU 测试通过；本机 RTX 5080 完成16环境、2轮真实训练，
并用生成的 model_1.pt 完成20步无窗口 Play 与视频录制，均正常退出。
实验位于 logs/rsl_rl/g1_ame_glad_critic_clean_stop_grad/2026-09-20_16-46-35_clean_stop_grad_smoke。
这些冒烟测试验证真实环境启动与接口，不代表策略已经学会行走。
双卡 NCCL 尚未验证；现有远端实验不受本次修改影响。
