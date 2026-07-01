# 面向 FastWAM v4 的中程记忆设计

日期：2026-06-30

这份笔记先从架构和结果角度总结 CronusVLA 与 EventVLA，再给出一版基于当前 FastWAM v4 短历史设计的中程记忆方案。核心原则是：不要推翻 v4，而是在现有 history video / history action K/V condition 路径旁边加一层中程 memory。

## 当前判断

v4 现在在标准 eval 上大致接近原版 FastWAM。下一步问题不是“v4 能不能恢复 baseline”，而是“memory 到底在哪里带来收益”。因此需要同时评估：

- memory-specific 任务上的短程成功率提升。
- history action conditioning 带来的动作平滑度提升。
- 在关键证据早于短窗口出现的任务上，中程 memory 是否有收益。

## FastWAM v4 当前 memory 接口

v4 现在已经有一个很干净的 memory 接入面。

训练侧：

- `_training_loss_v4` 要求同时有 `history_video_latents` 和 `history_action`：`src/fastwam/models/wan22/fastwam.py:729`。
- 它把 clean history/current video latents 和 future noisy video latents 拼起来：`src/fastwam/models/wan22/fastwam.py:756`。
- 训练时对 history video / history action 做 condition dropout：`src/fastwam/models/wan22/fastwam.py:831`。
- 它 prefill video K/V cache，并切出 clean prefix 给 action conditioning：`src/fastwam/models/wan22/fastwam.py:843`、`src/fastwam/models/wan22/fastwam.py:903`。
- 它 prefill clean history action K/V，然后把 video/action condition cache 拼起来：`src/fastwam/models/wan22/fastwam.py:872`、`src/fastwam/models/wan22/fastwam.py:911`、`src/fastwam/models/wan22/fastwam.py:922`。

推理侧：

- v4 memory inference 要求 history video 和 history action 成对出现：`src/fastwam/models/wan22/fastwam.py:1271`。
- 它把 history video 编成 VAE latents，prefill video K/V，再 prefill history action K/V，最后 concat：`src/fastwam/models/wan22/fastwam.py:1330`、`src/fastwam/models/wan22/fastwam.py:1365`、`src/fastwam/models/wan22/fastwam.py:1410`、`src/fastwam/models/wan22/fastwam.py:1421`。
- MoT cache 是逐层 K/V，不是最后一层 embedding：`src/fastwam/models/wan22/mot.py:325`、`src/fastwam/models/wan22/mot.py:347`。
- future action query 通过把 condition keys/values 和当前 action keys/values concat 来读 condition K/V：`src/fastwam/models/wan22/mot.py:581`、`src/fastwam/models/wan22/mot.py:670`。

online buffer：

- 当前 v4 保存近期 observation 和实际执行过的 actions：`experiments/fastwam_online_history.py:8`。
- history video offsets 由 `history_video_past_steps` 和 `action_video_freq_ratio` 决定：`experiments/fastwam_online_history.py:35`。
- 只有实际执行过的 action 进入 history：`experiments/fastwam_online_history.py:56`。
- `build_condition` 返回 `history_video`、`history_action` 和对应 pad masks：`experiments/fastwam_online_history.py:66`。

因此，中程 memory 最好也变成一种 condition source，能被 prefill 到现有 K/V 路径里。

## CronusVLA

来源：本地 `papers/cronusvla.pdf`、arXiv / project page。

### 一句话

CronusVLA 把单帧 VLA features 转成可缓存的多帧 motion features，再用 cross-frame diffusion decoder 让 noisy action queries attend 到 motion-feature K/V。

### 架构

CronusVLA 分三步。

1. 单帧预训练

- 训练一个常规 VLA。
- 输入：单张图像 + language。
- 输出：离散 action tokens，用 autoregressive CE 训练。
- 目的：保留强 single-frame VLM/VLA foundation。

2. 多帧编码

- 不再直接让 VLM 输出 action tokens，而是在 hidden layer 中引入 learnable motion features。
- 每一帧仍然独立经过 VLM/VLA backbone。
- 历史 motion features 组成 feature chunk：

```text
F_t = [f_{t-M+1}, ..., f_{t-1}, f_t]
```

- 推理时用 FIFO queue 保存历史 features，避免重复跑大 VLM backbone。
- post-training 时对 past features 做 stop-gradient regularization，避免多帧训练破坏单帧 backbone，同时省显存。

3. Cross-frame decoding

- modulator 用于平衡 current 和 past motion features。
- diffusion-style action decoder 以 noisy future actions 为 query。
- motion features 作为 cross-attention K/V。
- 输出连续 action chunk。

它还加了 action adaptation：

- 从 demonstrations 建一个 feature-action retrieval bank。
- 根据当前 feature chunk 检索相似片段。
- 把检索到的 action chunk 作为 coarse prior，在 fine-tuning / inference 时和 noisy action 拼接给 decoder。

### 结果

公开报告的主要结果：

- SimplerEnv：平均成功率 70.9%，在其设置下为 SOTA。
- LIBERO：相对 OpenVLA 有明显提升，不同表格 / 对比版本报告约 +12.7% 或 +26.8%。
- Real-world Franka：表现和鲁棒性较强。
- SimplerEnv-OR：他们还提出一个 observational robustness benchmark，测试 temporal / spatial disturbances。

### 对我们的意义

CronusVLA 最适合借鉴的场景是：failure mode 主要是动作连续性、中短程动态和 chunk 之间不稳定。

它的启发是：

- 把历史看成 action-relevant motion latent memory。
- action decoder 直接 cross-attend motion history。
- 历史 features 可以缓存，效率高。

对 FastWAM 来说，可以设计一个 Cronus-lite：

```text
history video/action K/V
  -> learned motion summary tokens
  -> small FIFO motion memory
  -> future action query attends motion memory + v4 condition K/V
```

如果 v4 标准成功率正常，但 smoothness / chunk-boundary 指标不好，再尝试这条线。

## EventVLA

来源：本地 `papers/eventvla.pdf`、EventVLA arXiv / project page。

### 一句话

EventVLA 加的是稀疏 visual evidence memory：initial / short-term visual anchors 加一个 Keyframe Evidence Memory head，预测什么时候把 task-critical visual event 写入 keyframe buffer。

### 架构

EventVLA 有两类 memory。

1. Foundational visual anchors

```text
A_t = initial frame + recent sliding-window frames
```

- initial frame 保存场景布局。
- short-term window 保存近期运动和局部任务阶段。
- 这相当于很多 fixed-history VLA 输入的增强版。

2. Keyframe Evidence Memory，简称 KEM

- KEM 是接在 VLA latent hidden states 上的轻量 head。
- 它预测 future action horizon 中哪些 timestep 会对应 task-critical key evidence frame。
- 如果概率超过 commit threshold，就把对应 observation 写进 event buffer。
- 推理时用 1D NMS 和 cooldown 防止连续重复写入。
- 训练标签由 VLM 离线标注生成，使用 soft labels + BCE。
- 训练 curriculum 从 ground-truth memory construction 逐渐切到模型自己预测 memory construction。

关键点：EventVLA 存的是稀疏 raw visual evidence / keyframes，而不只是压缩 latent summary。它的 ablation 显示，在 RoboTwin-MeM 上 implicit memory bank 明显弱于 visual evidence buffer。

### Benchmark 与结果

EventVLA 评估：

- RMBench。
- RoboTwin-MeM：它们新提出的 benchmark，专门测试 intermediate、transient、non-Markovian visual evidence。
- 4 个真实世界双臂任务。

主要结论：

- 在 RMBench 上，visual anchors only 已经很强，说明 RMBench 很多任务能被 initial + short-term context 解决。
- 在 RoboTwin-MeM 上，visual anchors only 掉到 18.0% 平均成功率。
- 完整 EventVLA，也就是 VA+KEM，在 RoboTwin-MeM 上达到 75.2% 平均成功率。
- implicit memory bank 只有 24.9%。
- hard labels 只有 48.8%，soft-label training 很关键。
- 去掉 NMS 后性能降到 53.4%，说明 memory write 的稀疏性很重要。

### EventVLA 是不是最新？

在我们当前本地 papers 文件夹和这轮讨论的四个 benchmark 里，EventVLA 是最新的，也是最明确针对 dynamic keyframe evidence 的。

不过截至 2026-06-30，还有一篇很接近的新 arXiv：KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies，日期是 2026-06-22。KEMO 不在当前本地 papers 文件夹里，但很相关：它用 robot kinematics + visual filtering 检测 keyframes，把 keyframes 编成 temporally ordered memory tokens，再通过 cross-attention 和 gated residual fusion 融合。它报告在真实双臂任务上比无 memory baseline 提升 +23.6% task success 和 +34.1% stage completion。

所以更准确地说：EventVLA 是当前本地集合里最新；KEMO 是更新的近邻参考，值得后续跟踪。

## 面向 FastWAM v4 的中程记忆计划

方案应该增量化。不要替换 v4，而是在现有 short-history 旁边加一层 medium-term condition。

### 设计原则

FastWAM v4 已经有：

```text
short history video -> video prefill -> layer-wise K/V
short history action -> action prefill -> layer-wise K/V
future action noisy query -> reads condition K/V
```

中程 memory 应该变成：

```text
event visual memory and/or motion memory -> prefill/encode -> layer-wise or projected K/V
future action query -> reads [short history K/V + medium memory K/V]
```

### Phase 0：先评估

加中程 memory 之前，先测：

- MemoryBench / ReMem extended MemoryBench 成功率。
- RMBench 或 RoboMME 选定任务。
- companion benchmark note 中定义的 action smoothness metrics。

如果 v4 提升 smoothness，但 medium-memory task 不行，就加 EventVLA-style keyframes。

如果 v4 成功率正常，但 replan jumps / jerk 仍高，就加 Cronus-style motion memory。

### Phase 1：EventVLA-lite，不先训练 KEM

先做 heuristic event buffer，避免一上来就搭 VLM auto-labeling pipeline。

memory state：

```text
short_history_video: existing V[t-16], V[t-12], V[t-8], V[t-4], V[t]
short_history_action: existing last 20 executed actions
event_video_buffer: up to 2-4 sparse keyframes
event_action_index: action step index for each keyframe
event_type: optional enum/debug string
```

write triggers：

- end-effector 位移很大或 phase 变化明显。
- gripper open / close transition。
- 当前视觉特征与历史 / 当前帧差异很大。
- replan boundary 处 action direction change 很大。
- 如果 benchmark 有 oracle labels，则用 task-specific event。

read path：

Option A，最低代码风险：

```text
event frames + short history frames + current frame
  -> same VAE encode
  -> same video prefill K/V
  -> future action reads expanded video K/V
```

Option B，语义更干净：

- 增加新的 source ids：`event_video`，可选 `anchor_video`。
- 扩展 video source embedding table。
- event frames 使用它们原始 episode step 对应的 canonical temporal positions。
- event video K/V 拼在 short history K/V 前面。

第一版如果赶时间，推荐 Option A；如果想做 paper-clean 设计，推荐 Option B。

### Phase 2：Cronus-lite Motion Memory

只有在 smoothness metrics 显示 history action / video K/V 还不够时再加。

memory state：

```text
motion_summary_buffer: FIFO of K learned tokens per replan
```

motion summary 构造方式：

- 从现有 history action prefill output 或 future action hidden states 中取。
- pool action/history K/V 或 action block hidden states。
- projection 成一小组 tokens。
- 训练时对旧 motion tokens stop-gradient，借鉴 CronusVLA 的 multi-frame regularization。

read path：

```text
future action query
  -> attends [v4 condition K/V + motion_summary K/V]
```

为什么不第一步就做：

- v4 已经有 history action K/V，这是很强的 action-memory path。
- 需要先用 smoothness metrics 证明 motion-summary memory 能带来 history action 之外的增益。

### Phase 3：训练 learned keyframe writer

当 heuristic event buffer 已经证明有用之后，再加：

- 在 FastWAM hidden states 上接 KEM-like head。
- 预测当前 replan horizon 内的 event-write probability。
- 用 simulator oracle state 或 VLM annotation 生成 soft labels。
- 推理时使用 NMS / cooldown。

可能的监督来源：

- MemoryBench / RMBench simulator state transitions。
- gripper / contact / action phase changes。
- VLM-labeled real video event frames。
- failure-driven labels：移除某帧后会导致未来决策错误的关键帧。

### Phase 4：可选 language summary

Language summary 对 counting 和高层 phase state 有用，但不应该是 FastWAM 的第一个中程 memory mechanism。

推荐形式：

```text
memory_text = "drawer opened: middle; pressed button: yes; moved block: red"
```

integration options：

- 追加到 language instruction。
- 作为单独 text context，让 action/video prefill cross-attend。
- 只在 symbolic / counting task 上启用，不用于所有 smooth manipulation task。

风险：

- language summaries 容易出错且会传播错误。
- 它可能提升 benchmark reasoning，但损伤低层动作 fidelity。

## 推荐实验阶梯

1. v4 full vs original FastWAM：
   - standard success。
   - smoothness metric pack。
   - MemoryBench / ReMem extended。

2. v4 ablations：
   - no history action。
   - no history video。
   - 不同 short history length。

3. EventVLA-lite：
   - 加 heuristic event video buffer。
   - 在 MemoryBench long horizon、RMBench、RoboMME Reference / Permanence 子集上评估。

4. Cronus-lite：
   - 加 motion summary tokens。
   - 评估 smoothness、replan boundary jumps、temporal disturbance tasks。

5. Learned KEM：
   - 只有 event buffer 有明确收益后再做。

## 预期应该看到什么

如果 v4 memory 真有帮助，至少应该看到其中一种：

- memory failure rate 降低，同时 execution failure rate 基本不变。
- replan boundary jump 降低，同时任务成功率不掉。
- high-frequency action energy 和 direction flip rate 降低。
- MemoryBench / RMBench / RoboMME memory subsets 成功率提升，同时标准 LIBERO 不下降。

## 参考来源

- CronusVLA: https://arxiv.org/abs/2506.19816 和 https://lihaohn.github.io/CronusVLA.github.io/
- EventVLA: https://arxiv.org/html/2606.20092v1 和 https://github.com/InternRobotics/EventVLA
- KEMO: https://arxiv.org/abs/2606.23589
- 当前 FastWAM v4 代码：`src/fastwam/models/wan22/fastwam.py`、`src/fastwam/models/wan22/mot.py`、`experiments/fastwam_online_history.py`
