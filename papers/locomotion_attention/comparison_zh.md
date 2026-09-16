核对日期：2026-09-17。比较对象为 AME1、AME-2、GLAD、TAGA 四篇论文，以及当前仓库的 `ActorCriticEncoder`。依据论文正文、结构图及本地代码；未进行新的策略训练或性能对比。下载文件的版本、字节数和 SHA-256 见同目录 `sources.json`。

**结论：本项目是修改过的 AME1，并提供 AME 式全局上下文分支。它与 GLAD 论文中的 AME / AME-GC 对照网络具有高度一致的基础配置；尚未实现 GLAD 的核心稀疏选择机制，也不是完整 AME-2 或 TAGA 系统。结构相似不表示性能、训练过程或代码来源相同。**

| 简称 | 已下载版本与原文 | 重点阅读位置 |
|---|---|---|
| AME1 | [Attention-Based Map Encoding for Learning Generalized Legged Locomotion，2506.09588v1](https://arxiv.org/abs/2506.09588v1)；[本地 PDF](/home/kqw/AME_Locomotion/papers/locomotion_attention/AME1_2506.09588.pdf) | 图 8，PDF 第 12 页；第 13–15 页方法与训练 |
| AME-2 | [Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding，2601.08485v3](https://arxiv.org/abs/2601.08485v3)；[本地 PDF](/home/kqw/AME_Locomotion/papers/locomotion_attention/AME2_2601.08485.pdf) | 图 3，PDF 第 5 页；IV-A、IV-B、IV-C |
| GLAD | [Global-Local Attention Decomposition for Terrain Encoding in Humanoid Perceptive Locomotion，2606.00637v3](https://arxiv.org/abs/2606.00637v3)；[本地 PDF](/home/kqw/AME_Locomotion/papers/locomotion_attention/GLAD_2606.00637.pdf) | 图 2，PDF 第 3 页；第 4–5 页编码器、基线和参数 |
| TAGA | [Terrain-aware Active Gaze Learning for Generalizable Agile Humanoid Locomotion，2606.05880v1](https://arxiv.org/abs/2606.05880v1)；[本地 PDF](/home/kqw/AME_Locomotion/papers/locomotion_attention/TAGA_2606.05880.pdf) | 图 3，PDF 第 4 页；3.2、4、附录 D |

以下 `p` 表示本体状态及命令，`F` 表示逐点地形特征，`g` 表示全局地形特征，`a` 表示动作。MHA 是多头注意力；Q 决定“根据什么来查询”，K/V 提供被查询的地形信息。

| 比较项 | AME1 原论文 | AME-2 v3 | GLAD v3 | TAGA |
|---|---|---|---|---|
| 主要输入 | XYZ 地图、当前本体状态 | 教师 XYZ；学生 XYZ+不确定性 | XYZ 地图、当前本体状态 | 深度图、XYZ 地图、5 帧本体历史 |
| 地形编码 | 高度 CNN，再拼接原始 XYZ | 几何 CNN + 位置 MLP，再由 MLP 融合 | XYZ 直接进入带步长的 CNN | 先裁 ROI，再做 CNN 与位置编码融合 |
| 注意力 Q | `Linear(p)` | `MLP(本体嵌入, g)` | `Linear(本体嵌入, g)` | 深度与本体嵌入的联合投影 |
| 全局信息 | 无独立全局分支 | 逐点 MLP 后 max-pool | 对全部特征做 attention pooling | 深度分支提供远处地形预览 |
| MHA 的 K/V | 全部逐点特征 | 全部逐点特征 | 显著性 Top-K 保留特征 | 连续 ROI 内的逐点特征 |
| 关键尺寸 | 64 维、16 头；GR-1 地图 17×11 | 图 3：局部 CNN 48 维、位置 16 维，融合为 96 维；全局 64 维 | 33×21 → 17×11=187 → Top-32；64 维、16 头 | 深度 36×64；地图 21×21 → ROI 11×11；融合 128 维 |
| 动作网络 | MLP | MLP Actor | MLP Actor | 5 专家软路由 MoE Actor |
| Critic / 历史 | 共享编码器、独立输出 MLP | 独立 MoE Critic；学生使用 20 步 LSIO | 共享地形编码器、独立输出 MLP | 此处不补推正文未明确给出的 Critic 细节 |

表中依据分别为 [AME1 方法与图 8](https://arxiv.org/html/2506.09588v1)、[AME-2 IV-A/B 与图 3](https://arxiv.org/html/2601.08485v3)、[GLAD III-D / IV-A](https://arxiv.org/html/2606.00637v3)、[TAGA 3.2](https://arxiv.org/html/2606.05880v1)。AME-2 的图示维度来自已下载 PDF；没有把本项目的 64 维、16 头参数套用到 AME-2 或 TAGA。

**本项目的实际数据流**

[网络实现](/home/kqw/AME_Locomotion/rsl_rl/rsl_rl/modules/actor_critic_encoder.py:119) 的默认 Actor 路径如下。批维省略；地图在代码中按 W×L 排列。

```text
XYZ 地图：3×21×33
  → Conv(3→16, kernel=5, stride=2, padding=2) → ReLU → BatchNorm
  → Conv(16→64, kernel=3, stride=1, padding=1) → ReLU → BatchNorm
  → F：187×64 ─────────────────────→ K/V

本体观测 p：96 → Linear(96→64) ─────→ Q：1×64
  → MHA：16 heads → 地形向量：64
  → 拼接原始 p：160
  → MLP(160→512→256→128→29) → 动作均值
```

Critic 的本体输入为 99 维，额外包含基座线速度；使用独立 Linear 和输出 MLP，但共享 CNN/MHA。地图尾部占 2079 维，故完整 Actor/Critic 输入分别为 2175/2178 维。观测定义见 [环境配置](/home/kqw/AME_Locomotion/source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/29dof/velocity_env_cfg_29dof.py:123)。当前无本体历史编码；Play 中的 RGB 相机也未接入策略输入。

开启 [attach_global](/home/kqw/AME_Locomotion/rsl_rl/rsl_rl/modules/actor_critic_encoder.py:144) 后，增加以下路径：

```text
F：187×64 → 逐点 MLP(64→256→128→64) → 沿点维 max → g：64
Q = Linear(concat(g, Linear(p)))：128→64
MHA 仍读取全部 187 个地形特征
Actor 输入 = concat(g, MHA输出, 原始p)：64+64+96=224
```

两个模式的动作 MLP 都为 `[512,256,128]` 隐藏层。默认配置关闭全局分支；仓库的 `ame1.pt` 对应默认模式，`ame2.pt` 对应开启全局分支的模式。这些文件名表示本仓库的变体，不能据此称为原作者完整系统的检查点。

**与原 AME1 的差别集中在地形前端**

原 AME1 用两层 5×5 CNN 对高度做卷积，输出 61 维后再拼 XYZ，保持逐扫描点对应；本项目把三通道坐标提前混入 CNN，直接输出 64 维，并通过 stride=2 降低空间分辨率。Query、单层交叉注意力和“地形向量拼原始本体状态”的后端思路保持一致。[AME1 方法](https://arxiv.org/html/2506.09588v1)

因此本项目保留了学习局部地形和状态相关聚合的能力，但没有在 CNN 后显式保留每个 token 的原始 XYZ。是否影响稀疏落脚点精度，需要实验判断。另外，本项目从 693 个扫描点缩到 187 个 token；AME1 的 GR-1 配置本来就是 187 个点，不能说本项目比原 GR-1 注意力序列还短。

**与 AME-2 的差别超过一个 max-pool 开关**

论文将位置分支与几何分支分开，再融合为局部特征；学生有不确定性输入和历史编码，训练采用教师—学生流程，且 Critic 是 MoE。位置/朝向目标任务及神经建图也是完整系统的一部分。[AME-2 IV–V](https://arxiv.org/html/2601.08485v3)

本项目只借用了“全局上下文同时进入 Query 和策略输入”的连接关系：仍是 XYZ 卷积、64 维局部特征、单帧状态、线性 Query 融合、共享注意力 Critic、速度命令训练。`attach_global=True` 更适合描述为 AME + global context；不能等同于完整 AME-2 复现。

**与 GLAD 的基础配置最接近，但缺少其两个核心组件**

本项目的 CNN、输入尺寸、激活与 BatchNorm、状态维度、MHA 维度/头数、输出 MLP 均与 GLAD v3 所列基础配置对应；GLAD 的 AME-GC 基线也采用 `[256,128]` 的全局 MLP。但 GLAD 使用 attention pooling，并只对 Top-32 特征做 MHA。训练期以 Gumbel 扰动和 straight-through softmax 处理选择梯度，推理期使用确定性 Top-K。[GLAD III-D / IV-A](https://arxiv.org/html/2606.00637v3)

需按 v3 理解选择过程：显著性分数是 `s_i=wᵀk_i+b`，不直接输入本体状态；后续 MHA 的 Query 才由状态和全局信息共同决定。不能沿用初版“状态直接控制 Top-K”的表述。

从机制看，max-pool 每个通道可选择不同位置的最大值；attention pooling 则形成空间权重分布后加权求和。因此当前项目的 max-pool 并不等于 GLAD 的全局注意力。当前代码也没有显著性评分、Top-K 或相应训练梯度机制。

**与 TAGA 的差别是感知和控制链路的整体扩展**

TAGA 让深度预览与本体历史共同预测连续 ROI，用双线性采样裁出局部地图，再做交叉注意力，最后由 5 个专家加权输出动作。它还采用 AMP、门控对比损失和 ROI 边界约束。[TAGA 3.2 / 4 / 附录 D](https://arxiv.org/html/2606.05880v1)

本项目没有深度策略分支、历史编码、可学习裁剪和 MoE Actor，也未接入上述训练目标。GLAD 可保留空间上分散的候选点；TAGA 的 ROI 是一个连续局部区域。这是两种不同的选择约束，不能把 Top-K 与裁剪互换后仍视为原设计。TAGA 中的 gaze 指地图内的计算关注区域，并不意味着当前架构需要控制真实相机转动。

**计算开销与后续实验的判断边界**

本项目的 Q 长度为 1，注意力分数矩阵是 `1×N`，不是地图 token 两两自注意力的 `N×N`。就固定维度的注意力点积/聚合而言，主要随 N 线性增长；K/V 投影另有约 `N·d²` 的成本。

把 187 个 K/V token 减到 32 个，数量减少约 82.9%，但不代表训练会加速 5.84 倍：CNN、选择模块、环境仿真、输出网络和反向传播仍有成本。TAGA 的 121 个 ROI token 也不能仅凭数量就与本项目比较总计算量，因为其维度和其他分支不同。

如果目标是基于现有项目隔离网络收益，最小改动路线是分别比较：当前默认网络、当前全局分支、仅 attention pooling、仅稀疏选择、完整 GLAD。固定观测、地形、奖励、训练预算，使用多随机种子和相同测试地形，测成功率、落脚误差、训练吞吐和推理延迟。该路线是根据代码改动范围作出的建议，不是已经完成的性能实验。
