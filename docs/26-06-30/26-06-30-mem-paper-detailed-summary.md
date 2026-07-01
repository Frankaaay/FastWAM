# MEM 论文超详细总结和问答备战

日期：2026-06-30

论文：`papers/Torne 等 - MEM Multi-Scale Embodied Memory for Vision Language Action Models.pdf`

## 0. 先记住这篇论文的核心

MEM 的主张是：机器人长程控制里的 memory 不能只用一种表示解决。短期 memory 需要密集视觉和动态信息，长期 memory 需要高度压缩的语义状态。MEM 因此把 memory 分成两条路径：

- short-term video memory：用高效 video encoder 压缩最近多帧 observation。
- long-term language memory：用自然语言摘要保存过去几分钟内仍对未来有用的语义事件。

它不是单纯“把历史帧加长”，也不是单纯“让 LLM 写个日志”。它的关键是按时间尺度分配 memory 形态，并让高层 policy 更新语义 memory，让低层 policy 用短窗口视频做动作。

## 1. 要解决什么问题

VLA 常见输入是当前图像、语言目标和当前机器人状态，然后输出 action。如果任务很短，这通常够用。但真实机器人任务经常需要记忆。

短期例子：

- 手臂挡住了目标物，当前帧看不到，但前几秒看到了。
- 刚才用某个抓取高度失败了，现在应该换一个高度。
- 刚才试着从冰箱门左侧打开失败了，现在应换方向。

长期例子：

- 做菜时哪些材料已经拿出来了。
- 清理厨房时哪些盘子洗过、哪些台面擦过。
- 一个十几分钟任务中哪些柜门打开过，最后是否需要关上。

论文指出，这两类 memory 的表示需求完全不同：

- 短期 memory 需要保留图像、空间、运动细节。
- 长期 memory 只需要很少语义 bits，不需要保存原始视频。

如果把 15 分钟多相机高频视频都送进 transformer，延迟和计算不可接受。论文提到真实机器人 dexterous manipulation 有几百毫秒级推理延迟约束，图 3 显示 naive 多帧编码很快超过实时阈值。

## 2. MEM 的系统分解

论文用一个 factorization 把完整长历史问题拆开。

原始目标：

```text
pi(a_{t:t+H}, l_{t+1}, m_{t+1} | o_{t-T:t}, m_t, g)
```

含义：

- `a_{t:t+H}`：未来 action chunk。
- `l_{t+1}`：下一步 subtask instruction。
- `m_t`：当前语言 memory。
- `m_{t+1}`：更新后的语言 memory。
- `o_{t-T:t}`：从很久以前到当前的 observation 历史。
- `g`：总任务目标。

MEM 近似成：

```text
pi_LL(a_{t:t+H} | o_{t-K:t}, l_{t+1}, g)
pi_HL(l_{t+1}, m_{t+1} | o_t, m_t, g)
```

其中 `K << T`。

解释：

- 高层 `pi_HL` 负责读当前 observation、总目标和旧 memory，输出下一步 subtask 和新 memory。
- 低层 `pi_LL` 只需要看短窗口 observation、当前 subtask 和总目标，然后输出连续动作。
- 远期历史不以原始 observation 形式进入低层，而是通过 `m_t` 这种语言摘要进入高层决策。

这套分解把长程任务变成了“高层语义进度管理 + 低层短期动作控制”。

## 3. Long-Term Language Memory 细节

### 3.1 它存什么

Language memory `m_t` 存过去语义事件里对未来仍然有用的部分。它不是完整日志，而是 compressed summary。

例如厨房清理任务中，memory 应该保留：

- 已经把哪些东西放回冰箱。
- 哪些盘子已经洗过。
- 哪些台面已经擦过。
- 某个柜门是否打开、是否需要关。

它不一定要保留：

- 每一次抓取尝试的完整过程。
- 物体所有视觉属性。
- 已经无关的失败细节。

### 3.2 怎么训练

论文的训练数据生成流程：

1. 机器人 episode 里有 subtask language annotations。
2. 每个 subtask 还带有成功/失败信息。
3. 把这些 subtask 序列和状态交给 off-the-shelf LLM。
4. prompt 要求 LLM 总结“对未来任务执行仍然相关”的信息。
5. LLM 输出作为 `m_t -> m_{t+1}` 的监督标签。

这意味着 high-level policy 学的是一个 memory update action：

```text
(current observation, old memory, task goal) -> (next subtask, updated memory)
```

### 3.3 为什么要压缩而不是拼接历史

论文专门 ablate 了 naive text + video memory，也就是直接拼接过去 subtask instructions。它表现明显更差。

原因是训练/推理分布偏移。

训练 demo 通常是接近最优的：

```text
pick up bowl -> place bowl in cabinet
```

推理时可能失败：

```text
pick up bowl -> pick up bowl -> pick up bowl -> place bowl in cabinet
```

如果把所有 instruction 拼起来，高层会看到训练中很少出现的重复 subtask 序列。MEM 的摘要 memory 可以选择在失败没有完成任务时不更新 memory，例如“碗已经拿起”这件事只有真的成功后才写入。

所以 language memory 的压缩不是为了好看，而是为了降低 train-inference distribution shift，并且减少无关信息携带。

### 3.4 它的风险

Language memory 的风险也很明显：

- 依赖 LLM 生成训练标签，pipeline 成本高。
- 如果 memory 写错，后续长任务可能沿着错误摘要继续执行。
- 对低层动作细节无能为力。
- 语言摘要过短会丢信息，过长又会慢且可能引入分布偏移。

## 4. Short-Term Video Memory 细节

### 4.1 为什么不能 naive 多帧输入

VLA 的视觉编码通常是最大计算开销之一。多相机、多帧、448x448 图像会产生大量 patch tokens。

如果每一帧都分别经过 image encoder，再把所有帧 token 送进 VLA backbone：

- token 数随帧数线性增长。
- transformer attention 成本上升。
- 推理延迟很快超过真实机器人控制需要的几百毫秒预算。

论文图 3 用 `pi0.6`、4 camera streams、单 H100 测了 inference time，显示 naive 方式随 total frames 增长很快；MEM video encoder 可以在更多帧下保持在实时阈值附近。

### 4.2 Video encoder 做了什么

MEM 从 ViT image encoder 出发，改成 video encoder。

步骤：

1. 所有输入帧分别 patchify。
2. 常规层做 spatial attention，也就是同一帧内 patch 之间双向注意力。
3. 每第 4 层加入 temporal attention。
4. temporal attention 沿同一空间 patch 在不同 timestep 之间做 causal attention。
5. 空间和时间 attention 分解，而不是所有 patch 和所有 timestep 做 joint attention。
6. 上层只保留当前 timestep 的 patch representations，把 past timestep patch tokens drop 掉。

直观理解：

```text
过去帧 token 不是直接送给 VLA backbone，
而是在 video encoder 内部把有用时序信息写进当前帧 token。
最后只把当前帧 token 交给 VLA。
```

这使得 VLA backbone 看到的 token 数仍接近单帧模型。

### 4.3 复杂度

论文给出的复杂度对比：

- naive spatio-temporal joint attention：`O(n^2 K^2)`。
- factorized spatial + temporal attention：`O(K n^2 + n K^2)`。

这里：

- `n` 是每帧 patch 数。
- `K` 是 observation memory 长度。

这就是为什么它比直接全时空 attention 更适合机器人实时场景。

### 4.4 参数初始化

这是论文里非常重要、也很容易被问到的点。

MEM video encoder 相比标准单图 ViT 不增加新的可学习参数。

它做的是：

- 修改 attention pattern。
- 加固定 sinusoidal temporal positional encoding。
- 复用原 ViT 的 Q/K/V 权重。
- 当前 timestep `t=0` 的 temporal embedding 设为 0。

因此当 `K=1` 时，video encoder 初始化行为和原 image encoder 精确一致。这样模型可以直接从预训练 VLM 权重初始化，不会一开始就破坏单帧能力。

### 4.5 数学定义

附录 C 里定义了 layer `l`、patch `p`、timestep `t` 的输入 embedding：

```text
z^{l-1}_{p,t}
```

先加 temporal position embedding：

```text
z_hat^{l-1}_{p,t} = z^{l-1}_{p,t} + e(t)
```

其中 `e(0)=0`。

然后使用原 ViT 的 Q/K/V：

```text
q = W_Q LN(z_hat)
k = W_K LN(z_hat)
v = W_V LN(z_hat)
```

论文用一个通用 attention 定义表示选定空间集合 `S` 和时间集合 `T` 上的 attention。Video encoder 的层可以理解为把 spatial attention 和 temporal attention 组合起来，其中 temporal 只沿同 patch 的时间维传播信息，并且使用 causal mask。

## 5. pi0.6-MEM 的具体训练设置

论文把 MEM 集成进 `pi0.6`。

模型侧：

- 初始化自 Gemma3-4B VLM。
- 使用预训练 VLM 的 vision-language backbone。
- action 训练同时包含 discrete FAST action token prediction 和 flow-matching action expert。
- action expert 为 860M 参数。
- action expert 梯度不回传到 VLM backbone。
- 输入图像分辨率 `448x448`。
- 最多 4 路 camera streams。

Proprioception 处理：

- 原 `pi0.6` 把机器人状态用 text 表示。
- 但多帧 state history 如果都转成文本，会产生大量 text tokens。
- MEM 改成 continuous state embedding：每个 proprioceptive state 用 linear projection 投到 backbone embedding space。
- 这样长度为 `K` 的 observation memory 只产生 `K` 个 state tokens。

数据侧：

- teleoperated robot demonstrations。
- policy rollout data。
- human corrections。
- vision-language tasks。
- video-language tasks，例如 video captioning。

预训练 memory horizon：

- 6 个 observations。
- 5 个过去 observation + 当前 observation。
- stride 为 1 秒。
- 即预训练 observation-based memory 约 5 秒过去上下文。

Post-training 扩展：

- 可以扩展到更长 observation memory。
- 实验中扩到 18 frames、约 54 秒。

推理：

- 使用 inference-time RTC 或 training-time RTC 做 asynchronous real-time inference。

## 6. 实验 A：长程任务

目标问题：MEM 能不能解决需要最高 15 分钟记忆的任务？

### 6.1 Recipe setup

任务：

- 给机器人一个详细 recipe prompt。
- prompt 指定需要哪些材料和厨具、它们在哪里、最终放到哪里。
- 机器人需要从冰箱、柜子、抽屉、炉灶等位置取出物品。

记忆需求：

- 哪些材料已经拿了。
- 哪些厨具已经放到目标位置。
- 哪些柜门/抽屉/冰箱门打开过，最后是否要关。

数据和评估：

- 训练 42 个 recipes。
- 评估在 unseen kitchens 和 unseen objects。
- 附录说每个 recipe 多数有 6 到 7 个 scoring points。

### 6.2 Clean kitchen

任务：

- 擦台面。
- 用纸巾擦干。
- 把食物放回冰箱。
- 把碗碟从 dish rack 放入柜子。
- 洗 sink 里的盘子并放到 rack。

记忆需求：

- 哪些台面已经擦过。
- 是否已经加 soap。
- 盘子正反面是否都洗了。
- 哪些物品已经收纳。

附录说平均 episode 约 8 个 subtasks。

### 6.3 Ablation 结论

论文比较：

- `pi0.6 (No Memory)`。
- `Only Video Memory`。
- `Only Text Memory`。
- `Naive Text + Video Memory`。
- `pi0.6-MEM`。

结论：

- 无 memory 的强 VLA 在长程任务上仍然困难。
- only video memory 不足以保存远期语义进度。
- only text memory 不足以处理短期视觉、动作和局部动态。
- naive text history 因为重复失败 subtask 的分布偏移表现较差。
- 完整 MEM 最强。

## 7. 实验 B：In-Context Adaptation

目标问题：memory 是否能帮助模型根据最近失败经验改变操作策略？

### 7.1 Chopstick Pick Up

设置：

- 机器人要拿起筷子。
- 数据收集时桌高在较高范围随机。
- 评估时桌子放在最低高度，属于 out-of-distribution。
- policy 容易用错误高度误抓。

训练：

- 当 policy mis-grasp 时，人类干预并演示正确抓法。
- memory 和 non-memory policy 都用同样 correction data 训练。

评分：

- 拿起筷子 +1。
- 放入 bin +1。
- 总分 2 视为 success。

结果：

- 带 memory 的 VLA 比无 memory 高约 `+11%` success。

### 7.2 Open Refrigerator

设置：

- 冰箱门没有明显视觉提示说明 hinge 在哪边。
- 机器人可能先从错误方向尝试打开。

训练：

- 收集人类干预。
- 或收集 demonstrator 初始也不知道开门机制的探索 rollout，自然包含失败尝试和纠正动作。

评估：

- 如果 `<=4` 次 grasp 内打开门，算 success。
- 这个标准是为了看 intentional strategy switching，而不是无限随机试。

结果：

- 带 memory 的 VLA 比无 memory 高约 `+62%` success。

### 7.3 为什么 non-memory 不行

即使 non-memory policy 看过 correction data，它在执行时看不到“刚才自己失败了什么”。因此它没有条件去做策略切换。

MEM 可以在短期 video memory 里看到失败尝试，于是学会：

- 筷子抓取高度不对 -> 改高度。
- 冰箱这个方向打不开 -> 换方向。

## 8. 实验 C：Memory 方法对比

目标问题：MEM 相比已有 memory 压缩方法好在哪里？

### 8.1 对比方法

No Memory：

- `pi0.6` 标准无 memory VLA。

Pool Memory：

- 每个过去 observation 用单帧 ViT 编码。
- 对过去 frame encodings average pooling 成一个 memory token。
- 当前 timestep 单独编码并输入 VLA。
- 优点是便宜，缺点是压缩太粗。

Proprio Memory：

- 只输入历史低维机器人 state。
- 避免高维图像 memory 成本。
- 对记住机器人自己状态有用，但环境状态弱。

Ours：

- MEM video encoder。
- 为公平比较，这里不使用长期 language memory，只比较 observation-based memory。

Ours post-train only：

- 从同一个预训练 `pi0.6` checkpoint 出发。
- 但 video encoder/memory 只在 post-training 加入。
- 用来测试“预训练时有没有培养 memory 能力”的影响。

### 8.2 任务类型

Swap 3 Mugs：

- 三个杯子轮流放到咖啡机下。
- 需要记住哪个杯子已经放过，且不能重复。

Find Object：

- 人把物体放入四个抽屉之一。
- 机器人之后要打开正确抽屉并取出。
- 无 memory 随机猜约 25%。

Unpack Groceries：

- 从购物袋取出所有物品。
- 袋内不完全可见，物品数量随机。
- 需要记住还剩多少。

Scoop Coffee：

- 精确放两勺咖啡豆。
- 是否还要再加一勺本质上接近 50% chance。

Grilled Cheese：

- 组装三明治、等待正确时间、翻面、再等待、装盘。
- 每面 cooking time 要在 30 秒到 3 分钟之间。

Window Cleaning：

- 喷清洁剂、撕纸巾、擦完整窗户、扔纸巾。
- 需要记住哪些步骤完成、哪些区域已擦。

### 8.3 结果趋势

论文报告：

- No Memory 在核心 memory tasks 上普遍困难。
- Pool Memory 对简单 memory 有帮助，但 average pooling 会丢长期和多对象细节。
- Proprio Memory 对自状态有帮助，但环境记忆任务弱。
- MEM 是唯一在 partial observability、counting、visual memory 等能力上都强的方案。

特别注意：图 8/9 是柱状图，PDF 文本没有完整数字表。分享时不要编精确百分比，除非直接展示图。

## 9. 预训练 memory 为什么重要

这是最值得重点讲的发现之一。

论文图 9 比较 `Ours` 和 `Ours (post-train only)`，结论是：

> 在多样化 robot + non-robot video data 上预训练 MEM，会显著提升模型利用 memory 的能力；只在 post-training 加 memory 明显更差。

更细地说：

- `post-train only` 也初始化自同一个已经大规模预训练过的 `pi0.6`。
- 它不是弱 base model。
- 它缺的是：预训练阶段没有用 memory 形式学习如何整合过去 observation。

论文强调，即使 memory horizon 在 post-training 从预训练的 5 秒扩到接近 1 分钟，预训练过 memory 的模型仍然能更好利用过去信息。

可能原因：

- 大规模多样化数据让 video encoder 学会什么过去信息对未来动作有用。
- 数据包含不同 optimality、速度、控制频率，可以减少 spurious correlation。
- 还包含 internet videos，有助于学习一般视频时序结构。
- 小规模同质 robot data 容易让 memory policy 学到错误相关性，比如 causal confusion 或复制过去动作。

回答别人问题时可以这样说：

```text
MEM 的预训练收益不是来自 base VLA 更强，因为 post-train-only 也用同一个强 base checkpoint；
收益来自 memory path 本身在预训练阶段被训练过。模型不是后期才被迫适配多帧，
而是一开始就在多样数据里学习“如何从过去帧提取对当前决策有用的信息”。
```

## 10. 为什么 MEM 不明显损害无 memory 任务

论文还在一组 challenging dexterous manipulation tasks 上比较 `pi0.6` 和 MEM，例如：

- table bussing。
- shirt folding。
- clean up counter。
- make bed。
- dishes to sink。
- batch folding。
- box building。

结论是 MEM 在这些不强依赖 memory 的复杂操作任务上也能 match `pi0.6` 的 performance。

这点重要，因为很多 memory policy 会因为 causal confusion 或额外历史噪声而退化。论文认为 MEM 没明显退化，部分原因是：

- video encoder 初始化保持单帧能力。
- current frame token 数没有暴涨。
- 大规模多样预训练降低 spurious correlations。

## 11. 和其他 memory 思路的差异

### 11.1 vs 直接长上下文

直接长上下文保留信息最多，但成本不可接受。MEM 用 language memory 承担分钟级语义状态，用 video memory 承担短期 dense detail。

### 11.2 vs recurrent memory

Recurrent memory 可以保存状态，但论文关注的是现代 VLA 和 transformer 架构下的可扩展 memory。MEM 不靠单一 recurrent hidden state，而是显式设计视频和语言两种 memory。

### 11.3 vs keyframe memory

Keyframe memory 保存稀疏过去图像。MEM 的短期 memory 更 dense，视频 encoder 会把短窗口信息压缩进当前 frame token；长期则用语言摘要，而不是存无限 keyframes。

### 11.4 vs proprio memory

Proprio memory 便宜，但只能记住机器人自己。Find object、unpack groceries 这类需要环境状态的任务，proprio memory 不够。

### 11.5 vs average pooling memory

Average pooling 太激进，容易丢空间关系、多个对象身份、时序顺序。MEM 在 ViT 内部分层做 temporal attention，能更有选择地抽取过去信息。

## 12. 可被问到的问题

### Q1：MEM 的最大贡献是什么？

不是提出了一个单独的新 attention trick，而是把 VLA memory 系统化地拆成多尺度、多模态：短期 dense video memory + 长期 compressed language memory，并证明这种组合在真实长程机器人任务中有效。

### Q2：为什么长期不用视频，短期不用语言？

长期视频太贵，而且很多长期状态只需要语义摘要，例如“已经拿了牛奶”。短期动作纠错和遮挡处理需要细粒度视觉、空间和动态信息，语言太粗。

### Q3：language memory 是不是只是 prompt engineering？

不是。LLM 用于离线生成训练标签，真正在线运行时是 high-level policy 自己预测 `m_{t+1}` 和 subtask。LLM 不是每一步在线控制的核心。

### Q4：为什么 naive text history 不好？

因为推理时失败会导致重复 subtask，训练 demo 中这种重复少，拼接历史 instruction 会产生分布偏移。摘要 memory 可以只在真正成功完成事件时更新。

### Q5：video encoder 有没有新参数？

论文说相比标准单图 ViT 没有新增可学习参数。它修改 attention pattern，并加入固定 sinusoidal temporal position encoding，复用原 Q/K/V 权重。

### Q6：为什么 `e(0)=0`？

为了保证单帧输入 `K=1` 时，video encoder 的初始化行为精确匹配原 VLM image encoder，最大化迁移并避免破坏单帧能力。

### Q7：为什么只保留当前帧 token？

过去帧的信息已经通过 temporal attention 被写入当前帧 token。丢掉过去帧 token 可以让 VLA backbone 输入 token 数接近单帧模型，从而控制延迟。

### Q8：预训练 memory 为什么比 post-training 加 memory 强？

因为模型需要在大规模、多样数据中学习如何使用过去帧。post-training-only 也有同一个强 base checkpoint，但 memory path 没在预训练中形成能力，所以后期目标任务数据不够时利用历史信息差。

### Q9：MEM 能处理多长记忆？

论文展示长期 language memory 支持 up to 15 minutes 的任务。短期 observation-based memory 预训练约 5 秒，post-training 扩到实验中的 18 frames、约 54 秒。

### Q10：MEM 会不会引入 causal confusion？

论文承认 memory policy 可能有 causal confusion 风险，但实验中 MEM 没明显降低复杂操作任务表现。作者认为大规模多样预训练数据有助于减少 spurious correlations。

### Q11：它和 FastWAM v4 有什么直接启发？

FastWAM v4 的 history video/action K/V 更接近 MEM 的 short-term dense memory。MEM 提醒我们，如果要上分钟级任务，需要额外的 semantic/event-level compression，而不能无限拉长 dense history。另一个启发是 memory 最好在预训练阶段就引入。

### Q12：这篇论文最弱的地方是什么？

复现门槛高：模型、数据、真实机器人评估都很大。长期 language memory 依赖 LLM 标注 pipeline。图表没有完整数字表，很多结论只能从柱状图趋势读。memory 更新错误的长期累积风险也还没有被完全解决。

## 13. 分享时可以强调的三句结论

1. MEM 的核心不是“更长上下文”，而是“不同时间尺度使用不同压缩方式”。
2. 视频 memory 解决短期细节和纠错，语言 memory 解决长期语义进度。
3. memory 能力最好在预训练阶段形成，后期才加 memory 明显更弱。

## 14. 对 FastWAM 的具体启发

### 14.1 当前 v4 更像 MEM 的短期 memory

FastWAM v4 的 history video/action K/V condition 能覆盖短期 observation 和 action 历史，这和 MEM 的 short-term video memory 方向一致。

### 14.2 中长期 memory 不应简单扩 history window

MEM 的结论支持我们不要无限增加 history frames。中长期任务更适合：

- event keyframes。
- semantic state summary。
- task progress memory。
- counting/status tokens。

### 14.3 需要关注 memory pretraining

如果 FastWAM memory 只在小规模 finetune 数据上学，可能会遇到 post-train-only 的问题。更稳的方向是设计 memory-aware pretraining 或至少 memory-rich staged training。

### 14.4 不同 failure mode 对应不同 memory

- 动作平滑 / chunk boundary：history action / motion memory。
- 短期遮挡 / 刚才失败：dense video memory。
- 关键证据早于短窗口：event keyframe memory。
- 长程任务阶段 / counting：language or symbolic memory。

## 15. 页码索引

- page 1-2：问题动机，多尺度 memory，短期视觉和长期语言的直觉。
- page 3：MEM factorization，高低层 policy，language memory 训练。
- page 4-5：video encoder、space-time separable attention、token dropping。
- page 5：pi0.6-MEM 集成、训练数据、6 observation 预训练、54 秒 post-training 扩展。
- page 5-6：Recipe setup、Clean kitchen 长程任务。
- page 7：in-context adaptation，chopstick 和 fridge。
- page 7-9：Pool Memory、Proprio Memory、post-train-only、预训练消融。
- page 13-15：各任务细节和 scoring。
- page 15：video encoder 数学公式。
