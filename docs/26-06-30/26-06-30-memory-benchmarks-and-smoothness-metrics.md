# 记忆类 Benchmark 与动作平滑度指标

日期：2026-06-30

这份笔记比较我们准备用来评估 FastWAM v4 的四个记忆类 manipulation benchmark：MemoryBench、RMBench、RoboMME、ReMem-VLA 扩展版 MemoryBench。同时补一套 action smoothness 指标，因为现有 memory benchmark 大多只看任务成功率，并不直接衡量机械臂抖动、动作连续性或 chunk 之间的跳变。

## 当前 FastWAM 评估目标

v4 现在在标准 LIBERO 类 eval 上已经大致接近原版 FastWAM。因此下一步不应该只问“能不能完成任务”，还要问：

1. 短程记忆是否能在必须依赖近期历史的任务上提升成功率？
2. history action / history video conditioning 是否让机械臂动作更平滑？
3. memory 是否减少 replan 边界处的动作跳变、重复修正和抖动？

建议对比组：

- 原版 FastWAM。
- v4 full：history video + history action。
- v4 去掉 history video。
- v4 去掉 history action。
- v4 current-only fallback。

## 1. MemoryBench

来源：本地 `papers/sam2act.pdf`、本地 `papers/remem.pdf`、SAM2Act arXiv / project page、MemoryBench dataset page。

### 测什么

MemoryBench 随 SAM2Act+ 提出，基于 RLBench。它包含 3 个脚本化 memory-dependent 任务：

| 任务 | 记忆类型 | 核心要求 |
|-|-|-|
| Reopen Drawer | 3D 空间记忆 | 记住一开始哪个抽屉是打开的，关上它，按按钮，然后重新打开同一个抽屉。 |
| Put Block Back | 2D 空间记忆 + 阶段记忆 | 把 block 从原始 patch 移到中心，按按钮，再把 block 放回原始 patch。 |
| Rearrange Block | 对历史动作的反向推理 | 把中心 block 移到空 patch，按按钮，再把原本没有被移动的 block 移到中心。 |

数据规模是每个任务 100 个训练 episode、25 个 held-out evaluation episode。任务很小，但会强制 policy 记住当前视角里已经不再直接可见的早期状态。

### 优点

- 非常直接地测试短程空间记忆。
- 规模小，适合做快速 targeted diagnostic。
- 很适合证明 v4 的 history video 是否能保留初始物体 / 抽屉状态。
- baseline 清楚：SAM2Act+ 在这些任务上相对 SAM2Act / RVT 有明显提升。

### 缺点

- taxonomy 比较窄，主要是空间记忆；对 duration、counting、object identity、procedural memory 覆盖不足。
- 精细接触和低层执行会污染记忆结论。ReMem-VLA 指出原任务中的 button / joint-limit 设置可能让失败看起来像 memory failure，但实际是低层 execution failure。
- horizon 偏短。ReMem-VLA 提到原始任务大约 300 frames，不太能测中程或长程记忆。
- 离散位置和固定布局可能泄漏 cue；如果控制不好，模型可能靠轨迹或场景配置绕过真正的 memory。

### 我们怎么用

把 MemoryBench 作为第一个“短程记忆成功率” sanity check。它最贴 v4 当前短历史设计，但报告时一定要把 memory failure 和 execution/contact failure 分开。

建议报告：

- 每个任务成功率。
- memory failure rate：错抽屉、错 patch、错物体。
- execution failure rate：没按到按钮、碰撞、抓取/释放失败。
- 成功 episode 和失败 episode 分别统计 smoothness。

## 2. RMBench

来源：本地 `papers/rmbench.pdf`、RMBench arXiv / project page。

### 测什么

RMBench 是基于 RoboTwin 2.0 的双臂 benchmark，包含 9 个 memory-centric 任务。它的核心贡献不只是任务集合，还包括 Task Memory Complexity 这个视角：一个任务应该按需要保留多少历史状态、关键证据距离当前有多远来分类。

RMBench 还提出了 Mem-0，一个双系统 memory policy：

- planning module：根据 instruction、当前观测和 key-frame memory 生成高层 subtask。
- execution module：根据当前观测、subtask、anchor memory、sliding memory 生成低层动作。
- subtask end classifier：判断当前阶段是否结束，是否写入 key frame，并进入下一阶段。

### 优点

- 比 MemoryBench 更系统，显式覆盖不同 memory-complexity level。
- 双臂 RoboTwin 设置更接近容易暴露阶段歧义的长程 bimanual 任务。
- Mem-0 的架构有参考价值：显式 task phase、key-frame memory、anchor/sliding memory 拆分。
- 适合测试 policy 是否真的在用历史，而不是只靠当前观测。

### 缺点

- EventVLA 认为 RMBench 中很多任务可以被 initial frame + short-term sliding visual anchors 解决，不一定强制动态记住中途短暂出现的 evidence。
- 基于 RoboTwin 2.0，把 FastWAM eval path 接进去的工程成本会高于 MemoryBench。
- 只看 success rate 不足以隔离 memory use，需要按 phase-memory failure 和 manipulation failure 分类。
- Mem-0 的显式 phase classifier 很有启发，但和 FastWAM 端到端 action denoising 路径并不等价。

### 我们怎么用

把 RMBench 作为 MemoryBench 之后的“中等难度 memory benchmark”。对 v4 来说，它可以测试短历史是否足够，还是必须引入 event / keyframe memory。

建议报告：

- 总成功率和每阶段完成率。
- failure 发生在第一个 memory-relevant transition 之前还是之后。
- short-history sufficiency analysis：current-only、short history、short history + event frames 三种设置对比。

## 3. RoboMME

来源：本地 `papers/robomme.pdf`、RoboMME arXiv / project / Hugging Face 页面。

### 测什么

RoboMME 是 ManiSkill / SAPIEN 上的 memory-augmented generalist policy benchmark。它有 16 个任务、4 个 suite，总共 1600 条 demonstrations。公开文档列出每个任务 100 train、50 validation、50 test demos，总约 768k frames，10 fps。

| Suite | 记忆类型 | 任务 |
|-|-|-|
| Counting | 时间 / 计数记忆 | BinFill, PickXtimes, SwingXtimes, StopCube |
| Permanence | 空间恒常性记忆 | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap |
| Reference | 物体指代记忆 | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder |
| Imitation | 程序 / 过程记忆 | MoveCube, InsertPeg, PatternLock, RouteStick |

论文还在 pi0.5 backbone 上比较了 14 个 memory-augmented VLA variants，分别改变 memory representation 和 integration 方式。

memory representation：

- Symbolic memory：subgoals 或 grounded subgoals。
- Perceptual memory：采样历史帧，或对视觉历史 token 做 token dropping。
- Recurrent memory：固定 latent state，比如 TTT / RMT 类机制。

integration strategy：

- memory-as-context：把 memory tokens 拼进当前 observation tokens。
- memory-as-modulator：用 memory 调制当前特征，例如 cross-attention + AdaLN。
- memory-as-expert：单独的 memory expert，action features block-wise attend memory。

### 优点

- 四个 benchmark 里 taxonomy 最清楚：temporal、spatial、object、procedural memory 被拆开。
- 很适合理解“哪类 memory 对哪类任务有效”。
- 一个重要结论是：没有单一 memory representation 能通吃所有任务。
- memory-as-modulator 对 FastWAM 很有启发，因为它可以加 memory 而不让 action-query attention length 爆炸。

### 缺点

- 比 MemoryBench 更贵。
- ManiSkill / SAPIEN / Linux / Vulkan 依赖可能给当前 server workflow 带来额外成本。
- 它的实验围绕 pi0.5 variants，架构结论迁移到 FastWAM / Wan2.2 需要再翻译一层。
- 主要衡量 memory success，不直接衡量 action smoothness。

### 我们怎么用

如果 infra 能跑，RoboMME 是最完整的 memory capability probe。它最适合支撑“v4 提升了 memory，而不只是某一个空间回忆任务”的论点。

FastWAM 推荐先跑的 suite：

1. Counting：看 history action 是否帮助事件计数和停止时机。
2. Permanence：看 history video 是否帮助遮挡和空间恒常性。
3. Reference：看是否需要 visual keyframe / event memory。
4. Imitation：后续再跑，和当前 v4 的对齐度稍弱。

## 4. ReMem-VLA 扩展版 MemoryBench

来源：本地 `papers/remem.pdf`、ReMem-VLA arXiv。

### 测什么

ReMem-VLA 使用 MemoryBench，并额外加入一个超过 600 frames 的 Long Horizon Task：把 Rearrange Block 和 Put Block Back 组合起来。它还修改了原始 MemoryBench 设置，让 benchmark 更能隔离 memory：

- 降低 button position randomization，避免 joint-limit / contact failure 掩盖 memory failure。
- 在 Rearrange Block 中让轨迹更一致，避免模型靠场景或轨迹 cue 绕过记忆。

simulation tasks：

| 任务 | 要求 |
|-|-|
| Put Block Back | 记住 block 原始位置和任务阶段。 |
| Rearrange Block | 记住哪个 block 原本被移动 / 没被移动。 |
| Reopen Drawer | 记住一开始打开的抽屉。 |
| Long Horizon Task | 组合 Rearrange Block + Put Block Back，超过 600 frames。 |

ReMem-VLA 还设计了 real-world memory tasks：

- 给花浇水约 6 秒：temporal duration memory。
- 舀两勺米：episodic / counting memory。
- 按绿-红-绿，每个按约 3 秒：sequential + temporal memory。
- 把水果放回原盘：visual memory。

### 优点

- 显式修复 MemoryBench 的一些混杂因素。
- 加入更长 horizon 的版本，能测试短窗口之外的 retention。
- real-world task taxonomy 很有参考价值：temporal、episodic/counting、sequential、visual memory。
- 它的 ablation 很支持 dual-timescale memory 这个方向。

### 缺点

- 这里的“benchmark”更像围绕 ReMem-VLA paper 的评估协议，不像 RoboMME 那样标准化。
- 某些细节可能需要 repo-specific reproduction effort。
- ReMem-VLA 自己的架构和 FastWAM 差异很大，因此结果未必能直接隔离同一种 memory 机制。
- real-world tasks 很有启发，但不能直接放进我们当前 simulated eval path。

### 我们怎么用

把扩展版 MemoryBench 当作更严格的 short-to-medium memory test。它尤其适合检查 v4 的短历史是否足够，还是需要 keyframe / event layer。

## Benchmark 横向对比

| Benchmark | 最适合测什么 | 主要短板 | FastWAM 优先级 |
|-|-|-|:-:|
| MemoryBench | 短程空间记忆和动作回忆 | 小而窄；contact precision 会混淆结论 | 高 |
| RMBench | 中等复杂度 memory 和 key-frame 式规划 | 可能被 initial + short sliding window 解决 | 中高 |
| RoboMME | temporal / spatial / object / procedural memory 全 taxonomy | setup 成本高；不直接测 smoothness | infra 就绪后高 |
| ReMem extended MemoryBench | 更长 horizon 的 MemoryBench + 更干净协议 | 作为独立 benchmark 的标准化程度较弱 | 高 |

## 缺失项：动作平滑度指标包

上面这些 benchmark 都回答不了一个问题：v4 是否让机械臂更不抖？所以我们应该给每个 eval 都补一套 action-level smoothness report。

### 需要记录的信号

每个 env step 记录：

- env 实际执行的 action。
- model-normalized space 中的实际执行 action。
- 如果可用，记录 end-effector pose。
- gripper command / state。
- replan boundary index。
- episode success / failure，以及 failure type。

注意：必须使用环境实际执行的 action，而不是模型预测的整个 future chunk。v4 的 online buffer 对 history action 已经遵守了这个原则。

### 指标

设连续执行 action 为 `a_t`，其中 translation 为 `p_t`，rotation 为 `r_t`，gripper 为 `g_t`。

1. Delta magnitude：

```text
M_delta = mean_t ||a_t - a_{t-1}||_2
```

衡量整体动作变化幅度。

2. Acceleration / jerk：

```text
M_accel = mean_t ||a_t - 2a_{t-1} + a_{t-2}||_2
M_jerk  = mean_t ||a_t - 3a_{t-1} + 3a_{t-2} - a_{t-3}||_2
```

jerk 是主要的“机械臂抖动”指标。

3. Replan boundary jump：

```text
M_replan_jump = mean over replan boundaries ||a_t - a_{t-1}||_2
```

直接测试 chunk-to-chunk continuity。

4. High-frequency energy：

```text
对每个 action dimension 做 FFT。
M_hf = cutoff 以上频段能量 / 总能量
```

用于捕捉平均 delta 不一定大的高频振荡。

5. Direction flip rate：

```text
flip_t = 1[dot(a_t - a_{t-1}, a_{t-1} - a_{t-2}) < -epsilon]
M_flip = mean_t flip_t
```

用于衡量重复修正和犹豫。

6. End-effector path inefficiency：

```text
M_path_ratio = EE 总路径长度 / 起点到终点直线距离
```

建议只在成功 episode 或可比较 task phase 上报告。

7. Gripper chatter：

```text
M_grip_switch = 每个 episode 中 gripper open/close switch 次数
```

用于捕捉抓取意图不稳定。

### 建议的综合 Smoothness Score

不要只报综合分，单项指标一定要保留。但 dashboard 排序可以用：

```text
SmoothPenalty =
  w1 * z(M_jerk_translation)
+ w2 * z(M_jerk_rotation)
+ w3 * z(M_replan_jump)
+ w4 * z(M_hf)
+ w5 * z(M_flip)
+ w6 * z(M_grip_switch)

SmoothScore = 100 - clamp(100 * SmoothPenalty, 0, 100)
```

用原版 FastWAM 作为 z-score normalization 的 reference distribution。分别报告：

- `smooth_success_only`：只看成功 episode 的平滑度。
- `smooth_all`：包含 recovery / failure 行为的整体平滑度。

### 推荐评估表

| Model | Success | Memory failure | Execution failure | Jerk ↓ | Replan jump ↓ | HF energy ↓ | Gripper chatter ↓ |
|-|-:|-:|-:|-:|-:|-:|-:|
| FastWAM original | | | | | | | |
| v4 full | | | | | | | |
| v4 no history action | | | | | | | |
| v4 no history video | | | | | | | |

## 参考来源

- MemoryBench / SAM2Act: https://arxiv.org/html/2501.18564v1 和 https://huggingface.co/datasets/hqfang/memorybench
- RMBench: https://arxiv.org/abs/2603.01229 和 https://rmbench.github.io/
- RoboMME: https://arxiv.org/abs/2603.04639 和 https://robomme.github.io/
- ReMem-VLA: https://arxiv.org/html/2603.12942v1
