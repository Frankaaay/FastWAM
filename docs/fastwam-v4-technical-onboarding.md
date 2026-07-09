# FastWAM v4 架构设计说明



## 1. v4 的核心命题

FastWAM v4 的核心命题是：

```text
用真实历史视觉、当前观测、真实已执行历史动作构造 condition KV cache；
推理时只对 future action 做 diffusion denoise；
不在 policy inference 主路径生成 future video。
```

也就是 **history KV + action-only inference**。

这个选择不是工程捷径，而是 v4 的架构边界。它把“理解历史”放进 condition cache，把“输出控制”限制在 action denoise loop 内。每次 replan 时，模型重新编码真实历史和当前观测，得到一份逐层对齐的 K/V memory；随后 future action token 作为 query 读取这份 memory，并在自己的 action token 序列内部做 self-attention。

因此 v4 推理输出的是 action chunk，不是视频 rollout，也不是 joint video-action trajectory。

## 2. 从 stage2 到 v4：为什么不是继续 prepend

stage2/prepend 类路线的直觉是把历史 token 拼到当前 token 前面，让 DiT 自己学会使用历史。这种方式简单，但有两个结构性风险。

第一，历史 token 和当前/future token 在同一个更新图里反复混合，模型可能把策略能力过度绑定到历史 token 的具体分布。一旦 online rollout 的历史采样、padding 或动作记录方式与训练不同，策略很容易退化。

第二，prepend memory 很难明确约束“哪些信息是 condition，哪些信息是 denoise target”。如果历史、当前、future 在同一条 raw token 序列中一起更新，训练时可见的信息边界很容易和推理时不一致。

v4 的设计反过来做：

```text
condition branch:
  history video / current video / history action
  -> clean prefill
  -> layer-wise K/V cache

target branch:
  future action noisy tokens
  -> denoise loop
  -> read condition K/V + action self K/V
```

condition branch 只负责产出 memory；target branch 才是 diffusion target。这个分工让 v4 可以显式控制训练/推理的信息边界。

## 3. 总体计算图

v4 有三条概念路径，但只有 action 是推理时必须生成的对象：

```text
真实 history/current video
  -> VAE
  -> Video DiT clean prefix
  -> video condition K/V

真实 history action
  -> Action DiT clean prefix
  -> action-history condition K/V

noisy future action
  -> Action DiT denoise
  -> reads condition K/V
  -> action loss / action output
```

训练时还存在 future video suffix，用于保留 video diffusion loss：

```text
history/current clean video prefix + future noisy video suffix
  -> one Video DiT path
  -> future video suffix loss
  -> clean prefix K/V sliced as action condition
```

这点是 v4 的关键：future video 可以作为训练时的辅助监督，但它不能成为 v4 action inference 的必需条件。action branch 读取的 video condition 只来自 clean prefix，也就是真实历史和当前观测。

## 4. Token 角色：source、position、timestep 三者分工

v4 中同一个时间点可能同时有 video token 和 action token。例如当前观测 `V[t]` 与 future action 起点 `A[t]` 都围绕时间 `t`。如果只靠位置编码，模型无法知道 token 的来源和角色；如果只靠 source embedding，模型又不知道 token 在统一时间轴上的相对位置。

所以 v4 同时使用三种语义：

```text
source/type embedding:
  说明 token 是 history video、current video、history action、future action 还是 future video。

RoPE / temporal position:
  说明 token 在统一时间轴上的位置。

diffusion timestep:
  说明 token 当前的噪声强度。
```

`FastWAM` 中固定了 5 类 source：

```text
0: history_video
1: current_video
2: history_action
3: future_action
4: future_video
```

其中 `future_video` 主要服务训练时 video loss，不进入 v4 默认推理 condition。

ActionDiT 的 source embedding 是 4 行，WanVideoDiT 的 source embedding 默认是 5 行。两个 expert 的 source embedding 都带一个零初始化 gate，因此刚加入时可以近似不扰动 base 模型，再逐渐学习 token role bias。这个 gate 是“source bias 的注入强度”，不是位置编码，也不是 memory gate。

需要强调：source embedding 和 positional encoding 是两个正交机制。

```text
source embedding:  这个 token 是什么角色？
RoPE position:     这个 token 在什么时候/哪里？
```

## 5. 统一时间轴

v4 的 action 历史长度固定为 20，当前时间索引记作 20：

```text
history action:
  A[t-20] ... A[t-1]  -> position 0 ... 19

current timeline:
  V[t]                -> position 20
  A[t]                -> position 20, source=future_action

future action:
  A[t] ... A[t+31]    -> position 20 ... 51
```

video latent 时间步因为有视频帧率和动作频率差异，使用 `VIDEO_LATENT_TIMELINE_STRIDE=4` 对齐到同一条 action timeline。典型 LIBERO 配置中 history video 是：

```text
V[t-16], V[t-12], V[t-8], V[t-4], V[t]
```

它们在 action timeline 上分别对应：

```text
4, 8, 12, 16, 20
```

这个统一时间轴让 action branch 可以把 `history_action`、`history_video`、`current_video` 放进同一套相对位置关系里理解，而不是把每段序列各自从 0 开始。

## 6. Video 分支：训练时单路 Video DiT，推理时只 prefill condition

v4 训练时的 video 分支不是“history video path + future video path”两套模型，而是一条合并后的 Video DiT path：

```text
clean prefix:
  history video + current video

noisy suffix:
  future video target

combined video tokens:
  clean prefix + noisy suffix
```

Video DiT 在这条 combined sequence 上运行。loss 只打在 noisy future video suffix 上；clean prefix 的作用是提供真实历史/当前上下文，并通过同一条 self-attention 路径影响 future video 预测。

随后，v4 从这条 video path 中切出 clean prefix 对应的逐层 K/V，作为 action branch 的 video condition。这样做的意义是训练时 action 看见的 video condition 与推理时 action 看见的 video condition 同构：都是由真实历史和当前观测 clean prefill 得到的 K/V。

推理时没有 future video suffix：

```text
history/current video
  -> VAE encode
  -> Video expert pre_dit(timestep=0)
  -> MoT.prefill_video_cache
  -> layer-wise video condition K/V
```

因此 v4 推理速度和接口都保持 action-only。future video imagination 是 v5/IDM 的研究方向，不是 v4 baseline 的默认依赖。

## 7. Action 分支：history action 是 condition，future action 是 target

v4 把 action 分成两类：

```text
history_action:
  已经真实执行过的动作，clean condition。

future_action:
  当前要预测的动作 chunk，diffusion target。
```

history action 进入 ActionDiT 的 prefill 路径：

```text
history_action
  -> action encoder
  -> source=history_action
  -> timestep=0
  -> MoT.prefill_action_cache
  -> layer-wise action-history K/V
```

future action 进入 denoise 路径：

```text
noisy future_action
  -> action encoder
  -> source=future_action
  -> timestep=current denoise step
  -> MoT.forward_action_with_condition_cache
  -> predicts denoised action / noise target
```

这两个 action 序列不能混淆。尤其在 online rollout 中，history action 必须来自环境里实际执行过的动作，而不是上一轮模型预测但尚未执行的完整 action chunk。否则模型读到的是“计划历史”，不是“世界真实发生过的控制历史”，训练/推理语义会错位。

## 8. MoT 的 layer-wise KV cache 语义

v4 的 condition cache 不是把 raw embedding 存起来，也不是只存最后一层 hidden state。MoT 的 prefill 做的是逐层保存 K/V：

```text
for layer l:
  input hidden at layer l
    -> norm / projection / RoPE
    -> K_l, V_l
    -> save into cache[l]
    -> layer update hidden for next layer
```

因此 action 第 `l` 层读取的是 condition 第 `l` 层生成的 K/V：

```text
future action layer l
  Q_l(action)
  attends:
    K_l/V_l(video condition)
    K_l/V_l(history action condition)
    K_l/V_l(future action self)
```

这叫 layer-aligned memory。它比“最后一层 embedding 复用到所有层”更接近 transformer 原生 KV cache 的语义，也避免了不同层表征空间错配。

v4 的 condition cache 通常由两部分拼接：

```text
condition_kv_cache =
  concat(video_condition_kv, history_action_condition_kv)

condition_valid_mask =
  concat(video_visible_mask, history_action_visible_mask)
```

future action denoise 时，MoT 会把 condition K/V 和当前 future action 自身的 K/V 拼在一起，再做 attention。也就是说 future action 既能读历史/当前 condition，也能在 action chunk 内部建模动作间依赖。

## 9. Mask、padding 与 dropout

v4 有三种容易混淆的 mask 语义。

第一是 padding mask。离线数据或 episode 起步时，history video/history action 可能不足。padding token 需要参与 shape 对齐，但不能作为真实历史暴露给 future action。

第二是 condition visible mask。它决定某个 condition token 是否对 future action 可见。video branch 内部为了稳定计算可能仍需要处理 padding token，但暴露给 action target 时必须尊重 visible mask。

第三是 training conditioning dropout。v4 对 history video 和 history action 使用 20% dropout，目的是让模型不要过度依赖某一种历史来源。dropout 改变的是“是否可见”，不是“位置是否重排”。即使某段历史被 drop，对应时间轴定义也不能重新编号。

因此 v4 的不变量是：

```text
dropout / padding 改 mask，不改 position。
```

current observation 不应被 history dropout 随机移除，因为 v4 的 action-only inference 必须始终以当前真实观测为锚点。

## 10. 训练/推理一致性

v4 架构最重要的对齐点是：action branch 在训练和推理时读取的 condition 类型一致。

训练时：

```text
action target reads:
  clean history/current video prefix K/V
  clean history action K/V
  future action self K/V

action target does not read:
  future video noisy/query tokens as condition
```

推理时：

```text
action target reads:
  clean history/current video K/V
  clean history action K/V
  future action self K/V

there is no future video generation in the main policy path
```

这就是为什么 v4 训练里即使保留 future video loss，也必须切出 clean prefix K/V 给 action，而不能把 future video suffix 的 K/V 混进 action condition。否则训练时 action 偷看到了推理时不存在的信息。

