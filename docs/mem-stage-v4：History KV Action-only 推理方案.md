# mem\-stage\-v4：History KV Action\-only 推理方案

**当前结论：**v4 采用 FastWAM 默认 action\-only KV cache 推理路线，并将原本的 first\-frame cache 扩展为 history video \+ history action \+ current observation 的 condition cache。最终采用 A2 source/type embedding（video 5\-way、action 4\-way）、统一时间轴、history video/history action 训练期 20% conditioning dropout、默认保留 condition prefill 梯度。training 中 history/current clean video prefix 与 future noisy video 合并为唯一 Video DiT 路径；action 只读取 clean prefix video K/V \+ history action K/V，不读取 future video query。v4 推理唯一入口是 `infer_action`，不再保留 `infer_joint`、`infer` video rollout、FastWAMJoint 或 FastWAMIDM 路线。H200\-1 服务器已切到 `mem-stage-v4` commit `e6bf164`，真实 LIBERO dataloader 与 forward\-only smoke 在临时 RoPE cache device patch 下通过；正式训练前必须先修复 `WanVideoDiT` / `ActionDiT` 自定义 position id 路径中的 RoPE cache device 问题。

## 一、最终采用路线

- 只参考当前 FastWAM 默认 `infer_action` 的 action\-only 推理路线。

- 不采用 FastWAM Joint/IDM 推理路线，因为它们需要 test\-time future video imagination 或两阶段先生成 future video；代码侧直接删除 FastWAMJoint/FastWAMIDM 及对应配置，避免误用。

- 参考 ImageWAM 的 KV cache conditioning 机制，但不照搬 image editing 语义。v4 的 memory 来自真实 history video、真实 history action 和当前观测。

- training 保留 video loss 作为辅助 world\-model 信号，但不额外开第二条 Video DiT。history/current clean video prefix 和 future noisy video suffix 走同一个 Video DiT；action 路径只切出 clean prefix video cache，不能依赖 inference 时不存在的 future video query。

## 二、当前 FastWAM action\-only 实际做了什么

当前 FastWAM 默认 action\-only 推理不是让 action 直接读 raw current image token，而是先对 current frame 做一次 video branch prefill，生成逐层 K/V cache。

```text
current image
  -> first_frame_latents
  -> video_expert.pre_dit
  -> mot.prefill_video_cache
  -> layer-wise video K/V cache
  -> future action denoise loop reads cache
```

prefill\_video\_cache 的语义是：

```text
x_0 = current visual tokens
for layer l:
  q_l, k_l, v_l = video_block_l.build_attention_io(x_l)
  x_{l+1} = self_attention + text/proprio cross_attention + MLP
  cache[l] = {k_l, v_l}
```

这里保存的不是最后一层 token，而是每一层对应的 K/V。action 第 l 层的 Q 会读 condition 第 l 层的 K/V，这保证了层深对齐。

## 三、为什么 history 不变也仍然需要 prefill forward

history 不在 action denoise loop 中变化，只说明它可以被缓存复用；不说明它可以跳过 transformer 编码。prefill forward 仍然必要，原因有三点：

1. **生成 layer\-aligned memory。** future action 第 l 层应该读 history branch 第 l 层的 K/V，而不是所有层都读原始浅层 history token。

2. **让 history 自己先完成时空整合。** history video 多帧之间、history action 多步之间，需要在 condition branch 内部先做 self\-attention，形成可读的历史状态。

3. **把 task/proprio/context 注入 memory。** current FastWAM 的 prefill 中包含 cross\-attention，因此 cache 是任务调制后的 memory，而不是纯视觉或纯动作 embedding。

结论：v4 不应把 raw history video/action token 直接拼成静态 K/V。正确做法是每次 replan 前对当前 history window 做一次 condition prefill，然后在 action denoise steps 中复用这份 cache。

## 四、v4 推理数据流

推荐 inference condition。本文约定 `A[t]` 是观测 `V[t]` 后立刻执行的 action，因此 history action 不包含 `A[t]`，future action 从 `A[t]` 开始：

- `history_video/current_video`：例如 `V[t-16], V[t-12], V[t-8], V[t-4], V[t]`，保持 5 帧；历史帧使用 `history_video` source id，当前观测使用 `current_video` source id。

- `history_action`：例如 `A[t-20] ... A[t-1]`，clean conditioning，使用 `history_action` source id。

- `future_action`：例如 noisy `A[t] ... A[t+31]`，denoise query，使用 `future_action` source id。

```text
history_video + current_frame
  -> video_expert.pre_dit
  -> + A2 source embedding: history_video / current_video
  -> video/history prefill with timestep 0
  -> layer-wise history/current video K/V cache

history_action
  -> action_expert.pre_dit with clean timestep 0
  -> + A2 source embedding: history_action
  -> clean action-history prefill
  -> layer-wise history action K/V cache

future_action noisy tokens
  -> action_expert.pre_dit at current denoise timestep
  -> + A2 source embedding: future_action
  -> Q attends [history video K/V + current video K/V + history action K/V + future action self K/V]
  -> scheduler step
  -> output future action chunk
```

history cache 每个 replan window 重新计算一次；同一个 action denoise loop 的多个 step 里复用。

### 4\.1 history action buffer 的来源

v4 推理时，`history_action` 必须来自实际执行过的 action stream，而不是上一轮模型生成的完整 future action chunk。原因是 `action_horizon=32` 只表示一次 replan 产出的候选 chunk；真实 rollout 通常只执行 `replan_steps` 个 action，未执行的 pending action 不具备历史观测语义。

- **执行后写入：**只有 action 真正传给 `env.step` / `task_env.take_action` 后，才 append 到 `executed_action_history_buffer`。

- **只取已执行历史：**每次 replan 前取最近 H=20 个已执行 action 构造 `history_action`；不足 20 时左侧 padding \+ invisible mask，position\_id 不重排。

- **action ensemble：**如果启用 action ensemble，history 记录 ensemble 后真正执行的 action，不记录 raw predicted chunk。

- **早停与窗口截断：**如果 episode done 或中途 replan，history 只追加已经执行的部分；pending queue 中剩余但未执行的 action 全部忽略。

- **格式：**建议同时维护 `executed_action_env_buffer` 与 `executed_action_model_buffer`。前者保存环境实际接收的 action；后者保存同一 action 映射回训练时的 normalized/model action 表示，作为 `history_action` condition。

- **warmup / no\-op：**LIBERO 的 `num_steps_wait` dummy action 不进入 policy history；第一轮 policy replan 使用空 history/padding。

```text
replan at time t:
  model predicts A_hat[t:t+31]
  execute selected actions only

then after each real env step:
  executed_action_history.append(action_exec)

next replan:
  history_action = last 20 actually executed actions
  pending or unexecuted predicted actions are ignored
```

当前模型接口已经能消费 `history_video/history_action`，但真实 online rollout 还需要在环境执行侧补 `executed_action_history_buffer` 和 history video window 维护。离线 trainer eval 传入的是 dataset 中的真实 history 字段，只能验证模型接口和离线路径，不能代替 online buffer 语义。

### 4\.2 v4 推理入口约束

生产 v4 action 推理只使用 `infer_action(..., history_video=..., history_action=...)`。不再提供 `infer()` 或 `infer_joint()` 作为兼容入口，也不再在 v4 推理中返回 video。

这意味着：

- v4 inference 输出是 future action chunk，不输出 future video。

- `runtime.run_inference` 应调用 `model.infer_action` 并保存/返回 action tensor。

- trainer eval 只验证 action\-only inference 和 action L1/L2，不再计算 rollout video PSNR/SSIM。

- FastWAMJoint、FastWAMIDM 及其 Hydra 配置不属于 v4 支持范围，已从当前分支删除。

## 五、建议的实现形态

1. 复用 `MoT.prefill_video_cache`：输入 clean history/current video latents，使用 video expert 逐层生成 K/V cache；接口统一返回 `(kv_cache, video_tokens_after_blocks)`，避免暗示存在第二条 Video DiT。

2. 复用 `MoT.prefill_action_cache`：输入 clean history action，使用 action expert 逐层生成 K/V cache。

3. 在 video/action `pre_dit` 中加入 A2 source/type embedding：video token 标记 `history_video/current_video/future_video`，action token 标记 `history_action/future_action`；新增 embedding 通过 zero\-init gate 注入。

4. 使用 `forward_action_with_condition_cache` 跑 future action query。

5. 每层将 condition cache 拼接为 `[history/current_video_kv, history_action_kv]`，再和 future action self K/V 拼接。

6. future action 仍然每个 denoise step 重新计算 Q/K/V，因为它的 noisy sample 和 timestep 都在变化。

```text
history_current_video_tokens = video_pre_dit(..., source_id=history_video/current_video)
history_action_tokens = action_pre_dit(..., source_id=history_action)
future_action_tokens = action_pre_dit(..., source_id=future_action)

condition_kv[layer] = concat(
  history_current_video_kv[layer],
  history_action_kv[layer]
)

action_out = forward_action_with_condition_cache(
  future_action_tokens,
  condition_kv,
  attention_mask
)
```

## 六、history action prefill 的选择

当前更推荐先复用 ActionDiT 的 clean action prefill 路线：

- history action 输入是 clean `A[t-20:t-1]`。

- history action timestep 设为 0，表示 clean condition；同时使用 `history_action` source id，避免和 future action query 混淆。

- history action tokens 经过 action expert blocks，逐层保存 K/V。

- future action 使用同一个 action expert 的 noisy timestep，并使用 `future_action` source id；逐层 Q attend history action K/V。

这样做的好处是层结构和 attention head 结构天然对齐；需要额外处理的是 clean history action 和 noisy future action 在同一 action expert 内同时存在时，必须用 timestep \+ source/type embedding 双重信号区分条件记忆和预测目标。

如果后续发现 history action 过强或计算成本偏高，可以作为 ablation 尝试轻量 history\-action encoder，但 v4 第一版建议先做层对齐的完整 prefill。

## 七、训练设计

training 保持和 inference 一致的 action conditioning 路径，并直接采用 A2 方案：

- history video、history action、current observation 是 clean condition memory；future action noisy tokens 是 denoise query。

- action\-frame 对齐沿用第 4 章约定：`A[t]` 是观测 `V[t]` 后立刻执行的 action，因此 history action 为 `A[t-20] ... A[t-1]`，future action 从 `A[t]` 开始。训练样本中的 `history_action` 使用数据集中真实记录/实际执行的历史 action；推理时使用 `executed_action_history_buffer` 里已经实际送入环境的 action。两者语义必须一致，不能用上一轮未执行的 predicted chunk 充当 history。

- training 和 inference 使用同一套 source id 与统一时间轴：`history_video`、`current_video`、`history_action`、`future_action`；training video loss 额外使用 `future_video` source id 标记 future video suffix。

- 训练时对 `history_action` 和 `history_video` 各自做 20% branch/window conditioning dropout；current observation 不做 dropout。dropout 后不重排 position\_id，被 dropout 的 branch 不作为可见 K/V 参与 attention。

- future action noisy tokens 计算 action loss；future video noisy tokens 保留 video diffusion loss，但必须与 history/current clean prefix 走同一个 Video DiT path，而不是单独再跑一条 history video path。

- loss 设置和 FastWAM 保持一致，不额外设计新的 loss 权重。future action 默认不 attend future video query，避免 inference 没有 future video 时出现 train/infer mismatch。

- clean prefix 不直接计算 video loss，但它在同一个 Video DiT self\-attention 路径中自然影响 future video suffix 的预测；action condition 只从同一条 Video DiT cache 中切出 clean prefix 部分。

- condition prefill 默认保留梯度，使 action loss 可以训练 history video/action memory 接口；inference 才使用 no\_grad cache。

```text
history/current clean video prefix + future noisy video suffix
  -> 20% train-only history branch dropout
  -> one Video DiT path, keep gradients
  -> future video suffix -> video loss
  -> clean prefix cache slice -> action condition

future action noisy + source_id=future_action
  -> action query reads visible history/current memory -> action loss
```

## 八、推荐 token 规模

以 `224x448`、5 帧 history/current video、patch `[1,2,2]` 为例：

|部分|语义|token 数|是否作为 Q 更新|
|---|---|---|---|
|history/current video|clean visual memory|约 196|prefill 阶段更新；action loop 中仅 K/V|
|history action|clean action memory|20|prefill 阶段更新；action loop 中仅 K/V|
|future action|noisy action target|32|是，每个 denoise step 更新|

training video loss 侧还有 future video suffix，因此 Video DiT 训练路径约为：

```text
clean history/current prefix: 2 latent frames x 98 tokens = 196
future video suffix:          2 latent frames x 98 tokens = 196
combined video path:          392 video tokens
```

action denoise 每层近似 attention 规模：

```text
Q length = 32
K/V length = 196 + 20 + 32 = 248
action attention score scale = 32 x 248
```

相比 full mixed attention 中让所有 history/action token 都作为 query 更新，这条路线更省，并且避免 test\-time future video generation。

## 九、和 ImageWAM 的区别

|维度|ImageWAM|v4|
|---|---|---|
|cache 来源|image/editing branch 的中间 K/V 表征|真实 history video、history action、current observation|
|推理模式|condition prefill once，action denoise reads cache|同样采用 condition prefill once，action denoise reads cache|
|核心目的|避免完整 video generation|避免 future video generation，同时补足历史时序信息|
|关键差异|memory 来自 image/editing context|memory 来自真实轨迹历史，因此 history action 需要被显式建模|

## 十、主要风险和约束

- 不能只靠 attention mask 省计算；必须结构上让 history 在 action loop 中只做 K/V，不做 Q。

- history action 的 clean timestep 和 future action 的 noisy timestep 需要明确区分；同时必须使用 source/type embedding 区分 `history_action` 与 `future_action`，不能只依赖 timestep、mask 或序列位置。

- condition cache 是逐层 cache，不是最后一层单个 embedding；实现时需要保证每层 K/V 对齐。

- video expert 和 action expert 的 token hidden dim 可以不同，但 attention K/V 的 head 结构必须一致。当前 FastWAM 已要求 num\_heads 和 attn\_head\_dim 一致。

- 每次 replan 时 history window 会变化，因此 cache 需要重新 prefill；同一个 denoise loop 内才复用。

- online rollout 必须维护实际执行 action 的 history buffer；离线 dataset history 不能自动代表真实部署路径已经正确。

- v4 不再提供 joint/video rollout 推理入口；如果后续需要可视化 future video，应作为单独分析工具重新设计，不能混入 policy inference API。

- padding/window 边界不能依赖 all\-invalid fallback 静默放开 mask；需要显式区分“计算上防 NaN 的 self/key”和“作为 condition 暴露给 future action 的 visible K/V”。

## 十一、当前代码证据索引

- v4 source id 常量：`src/fastwam/models/wan22/fastwam.py:20`

- v4 training loss 入口：`src/fastwam/models/wan22/fastwam.py:729`

- single Video DiT combined latents：`src/fastwam/models/wan22/fastwam.py:763`

- 从同一 video cache 切 clean prefix 给 action：`src/fastwam/models/wan22/fastwam.py:900`

- v4 action condition cache 拼接：`src/fastwam/models/wan22/fastwam.py:919`

- v4 action\-only inference 入口：`src/fastwam/models/wan22/fastwam.py:1175`

- v4 history condition 分支判断：`src/fastwam/models/wan22/fastwam.py:1266`

- video cache prefill 统一返回 cache 和 tokens：`src/fastwam/models/wan22/mot.py:325`

- history action cache prefill：`src/fastwam/models/wan22/mot.py:409`

- future action 读 condition cache：`src/fastwam/models/wan22/mot.py:581`

- A2 action source embedding 注入点：`src/fastwam/models/wan22/action_dit.py:348`，即 `tokens = self.action_encoder(action_tokens)` 之后、生成 Q/K/V 之前。

- A2 video source embedding 注入点：`src/fastwam/models/wan22/wan_video_dit.py:697`，即 video latent patch flatten 成 `x_tokens` 之后、进入 MoT / blocks 之前。

- dataset history video/action 窗口：`src/fastwam/datasets/lerobot/robot_video_dataset.py:63`、`src/fastwam/datasets/lerobot/robot_video_dataset.py:228`

- ImageWAM cache prefill：`/Users/maxliu/MyProjects/AIR/202607/ImageWAM/src/imagewam/models/backbones/mot.py:981`

- ImageWAM action 读 cache：`/Users/maxliu/MyProjects/AIR/202607/ImageWAM/src/imagewam/models/backbones/mot.py:1009`

## 十二、A2 source/type embedding 方案

**确定实现：**v4 直接采用 A2 source/type embedding \+ learnable scalar gate。当前实现为 video 5\-way、action 4\-way：video 侧额外区分 `future_video`，用于 single Video DiT training 中的 future video suffix；action 侧仍只需要区分 history/future action。gate 初始化为 0，使新增来源标识在训练初期不破坏 action\-only 路径。当前不做 A0/A1/A3/A5 ablation，先按 A2 从 scratch 训练。

### 12\.1 需要解决的问题

v4 的 action query 会同时读取 history video、current observation、history action 和 future action self tokens 对应的 K/V。mask 只能限制哪些 token 能互相看见，但不能告诉模型每个 K/V token 的来源和角色。因此需要给每个 token 在进入 Q/K/V 之前加入 source/type embedding。

### 12\.2 source id 设计

当前全局 source id 约定：

```text
0: history_video
1: current_video / current_observation
2: history_action
3: future_action / target_action_query
4: future_video / video-loss suffix
```

action expert 当前使用 4\-row embedding table，实际用到 `history_action=2` 和 `future_action=3`；video expert 使用 5\-row embedding table，实际用到 `history_video=0`、`current_video=1` 和 `future_video=4`。

这些区分比只区分 video/action 更必要，因为 history\_action 和 future\_action 都来自 action encoder，但语义完全不同：前者是条件记忆，后者是正在 denoise 的预测目标；history\_video、current\_video、future\_video 也同理，尤其 single Video DiT training 中 clean prefix 和 noisy future suffix 需要明确区分。

### 12\.3 token 注入位置

```text
token = encoder(raw_input) + alpha * source_embedding[source_id]
alpha: learnable scalar，初始化为 0 或很小值
```

source embedding 应加在 modality\-specific encoder 之后、Transformer / QKV / RoPE 之前。也就是说，video latent patch token 在进入 video expert 前加 video source embedding；action token 在 action\_encoder 之后加 action source embedding。缓存 K/V 时，K/V 已经包含 source 信息。

### 12\.4 和 RoPE、时间标号的关系

source/type embedding 不替代 RoPE 或 temporal position。RoPE 负责位置和相对时序，source embedding 负责来源和角色。history/current/future 仍应保留正确的时间位置或 horizon 标号；source embedding 只是解决“这是哪一类 token”的身份问题。

### 12\.5 参数组织建议

概念上使用同一套 source enum，但参数表建议按 expert 分开，因为 video hidden dim 和 action hidden dim 不同：

```text
video_source_embedding:  [5, video_hidden_dim]
action_source_embedding: [4, action_hidden_dim]
source_gate_video:       scalar, init 0
source_gate_action:      scalar, init 0
```

如果后续扩展更多模态，比如 proprio、goal image、language，可以再把 source embedding 升级为 factorized 形式：modality embedding \+ role embedding。当前 v4 第一版优先保持 video/action 两个 expert 内的简单 learned table，成本最低、歧义最少。

### 12\.6 当前采用方案

- 当前实现和训练直接采用 A2，不把 source embedding、unified position、dropout、detach 等核心开关暴露为配置项。

- A2 对推理速度几乎没有额外影响，只增加逐 token 向量相加；attention 的 Q/K/V 长度和计算量不变。

- 老权重加载时新增 source embedding、source gate、position\_ids 相关 missing keys 允许初始化；但后续 resume v4 checkpoint 时必须严格检查这些新增参数已经从 checkpoint 恢复，不能静默遗漏。

- ablation 暂不进入当前实施范围；文档保留 ablation 作为后续实验记录入口。

## 十三、统一时间轴与 position\_id 方案

**结论：**history、current、future 不应该各自从 0 开始做 RoPE/position 编码。source embedding 解决“token 来自哪里”，时间编码解决“token 在同一条时间线上处于哪里”。两者需要同时存在。

这里需要明确区分三类信号：

- **diffusion timestep：**表示扩散噪声强度。history/current clean condition 使用 clean/noise=0；future noisy action 使用当前 denoise step。

- **RoPE position\_id / temporal\_position\_id：**表示真实时间顺序。history、current、future 应放在同一条相对时间轴上。

- **source/type embedding：**表示来源和角色，例如 history\_video、current\_video、history\_action、future\_action。

以 history action 长度 H=20、future action 长度 F=32 为例，并采用 `A[t]` 是观测 `V[t]` 后立刻执行的 action 这一约定，推荐使用带 offset 的统一时间轴，避免负数 position：

```text
真实相对时间:
history action: A[t-20] ... A[t-1]
current obs:    V[t]
future action:  A[t] ... A[t+31]

RoPE position_id, 使用 offset=20:
A[t-20] ... A[t-1]  -> 0 ... 19
V[t]                -> 20
A[t] ... A[t+31]    -> 20 ... 51
```

`V[t]` 和 `A[t]` 可以共享 position 20，因为它们确实对齐在当前决策时刻；二者由 source/type embedding 区分。真正需要避免的是 `A[t-20]` 和 `A[t]` 都被编码为 position 0。

history/current video 也应放到同一条时间轴。例如使用 5 帧视觉条件：

```text
raw frames:
V[t-16], V[t-12], V[t-8], V[t-4], V[t]

建议 temporal position:
4, 8, 12, 16, 20
```

由于 video VAE temporal compression 后，DiT 看到的 latent temporal slice 不一定和 raw frame 一一对应，当前实现按 latent temporal index 近似标注。以 5 帧 history/current video 被编码成 2 个 latent temporal slice 为例：

```text
history/current clean prefix latent positions:
history latent approx -> 4
current latent        -> 20

training future video suffix latent positions:
future latent 1       -> 24
future latent 2       -> 28
```

最后一个 clean prefix latent 固定标为 current\_video；更早的 clean prefix latent 标为 history\_video。future video suffix 只用于 training video loss，标为 future\_video，不作为 action inference condition。

### 13\.1 dropout 情况下的时间轴

训练时对 `history_action` 和 `history_video` 各自做 20% branch/window conditioning dropout。dropout 只改变某一路 condition 是否可见，不改变 canonical 时间轴。

- **时间轴不重排：**dropout 后不能把剩余 token 重新从 0 编号。所有保留 token 仍使用完整窗口下的 canonical position\_id。

- **推荐实现：**被 dropout 的 branch 优先从 condition K/V 中移除；如果实现上需要固定 shape，则将该 branch token 置零并用 attention mask 完全屏蔽。

- **current observation：**`V[t]` 不做 dropout，始终保留，并继续使用当前时刻 position。

- **source id：**保留 token 的 source id 不变；被 dropout 的 token 不应作为可见 K/V 参与 attention。

- **future video suffix：**training video loss 中 future video suffix 仍可作为 Video DiT query/key 参与 video loss；但 action condition cache 只切 clean prefix，不暴露 future video K/V。

```text
完整时间轴:
A[t-20] ... A[t-1]  -> 0 ... 19
V[t]                -> 20
A[t] ... A[t+31]    -> 20 ... 51

如果 history_action dropout:
history_action K/V 不可见
V[t] 仍是 20
future_action 仍是 20 ... 51

如果 history_video dropout:
history_video K/V 不可见
current V[t] 保留，仍是 20
future_action 仍是 20 ... 51
```

### 13\.2 代码实现建议

- 给 `ActionDiT.pre_dit` 增加 optional `position_ids` 参数。默认 `None` 时保持当前 `self.freqs[:seq_len]` 行为，兼容原 FastWAM 路径。

- 给 `WanVideoDiT.pre_dit` 增加 optional `temporal_position_ids` 参数。默认 `None` 时保持当前 `self.freqs[0][:f]` 行为；v4 路径传入统一时间轴上的 temporal ids。

- source/type embedding 应在 encoder 之后、Q/K/V 之前注入；RoPE position\_id 应用于 Q/K。二者一个是内容身份，一个是时间位置，不互相替代。

- training 和 inference 必须使用同一套时间轴规则。不能训练时 history/current 分开从 0 开始，推理时再改成统一时间轴。

- 需要检查 RoPE cache 长度。当前 action RoPE cache 长度为 1024，H=20、F=32 时最大 position 51，足够；如果未来扩大 history/future horizon，需要同步校验。

## 十四、仍待讨论问题

当前建议采用固定窗口 \+ canonical position \+ visible mask 的边界策略：

- **current observation 不做 padding 开关。**不再引入 `current_video_valid`。`V[t]` 是当前决策必需条件；如果当前观测是 pad 或不可用，应跳过该 sample / 不触发 policy，而不是让模型学习“无当前观测”。

- **history action 不足 20 步时左侧 padding。**保留固定 shape `[20, action_dim]`，position\_id 仍为 `0..19`，padding token 的 `history_action_is_pad=True`，不作为 future action 可见 K/V。第一轮 policy replan 可全部是 padding；warmup/no\-op 不进入 history。

- **history video 不足窗口时左侧 padding。**保留 canonical temporal positions；padding history frame 不作为 future action 可见 K/V。当前观测 frame 始终可见，位置为 20。

- **dropout 与 padding 组合：**最终 visible mask 应为 `valid & ~drop_branch`。dropout 不重排 position，不改变 source id。

- **prefill 内部数值稳定和对外可见性分离。**为了防止 attention 全 false 产生 NaN，condition branch 内部可以保留最小 self/key 兜底；但暴露给 future action 的 `condition_key_valid` 必须严格使用 visible mask，不能因为 all\-invalid fallback 把 padding token 重新暴露出去。

- **全 invalid future target。**如果 future action 全是 padding，loss mask 会把该样本 action loss 置零；训练数据最好通过 sampler/重试尽量避免这种样本长期出现。

当前代码已经有 padding mask 传递和 dropout mask 组合，但 `MoT._apply_key_valid_mask(... ensure_non_empty=True)` 会在某些 all\-invalid 行回退到原 mask。后续如果要严格实现上述语义，应显式区分 branch 内部 compute mask 和 future action 可见 mask，并避免 all\-invalid fallback 影响 condition exposure。

## 十五、代码复查与修复（实现对齐）

对照本设计文档逐链路复查 v4 实现，修复 1 个 High、1 个 Medium、2 个 Low 问题；其余关键点（source embedding 注入、KV 切片复用、dropout、checkpoint、数据\-模型形状）复查通过。代码改动仅涉及 fastwam\.py 与 wan\_video\_dit\.py。

**High — 训练/推理 history\-video self\-attention mask 不一致（已修）：**训练 combined 路径让 history 前缀内部双向可见，但推理 infer\_action 的 history\-video prefill 漏传 clean\_prefix\_latent\_frames（默认取 1），导致前缀第一帧看不到后续帧，history\-video K/V 在 train/infer 下不一致，污染 action condition memory。已让推理处传入完整 history latent 帧数，恢复前缀双向，符合第十三章「train/infer 必须使用同一套规则」。

**Medium — combined video 恢复 video←action 条件（已按原 FastWAM 改回）：**原 v4 combined 路径给 video 传 action=None，相比原 FastWAM 丢失 video 分支的 action 条件（真实原因是 group\-causal mask 的 num\_temporal\_groups=f\-1 在 combined f=4 时与 32 action 不整除会触发 assert）。已把 num\_temporal\_groups 泛化为 f \- clean\_prefix\_latent\_frames（clean prefix 不 attend action，仅 future video 帧按组 attend future action），并改回 action=action；默认 clean\_prefix=1 时与原 FastWAM 完全等价。注意当前 fastwam\.yaml 中 action\_conditioned=false，该条件路径 dormant，本次修复在当前配置下行为不变，价值在于与原 FastWAM 一致并避免将来开启 action\_conditioned 时崩溃。

**Low（已修）：**\(1\) training\_loss 的 v4 触发条件由 OR 改为 AND，与 \_training\_loss\_v4 要求 history\_video 和 history\_action 成对存在保持一致；\(2\) 删除 \_training\_loss\_v4 中对第 0 帧赋值后即被丢弃的死代码，并注释说明当前观测由 history\_video 最后一帧承载。

**LIBERO 数据\-模型一致性确认：**libero\_2cam（num\_frames=33, action\_video\_freq\_ratio=4）产出 history\_video 5 raw→2 latent、video 9 raw→3 latent、history\_action=20、future\_action=32；combined f=4=2 prefix\+2 future，与模型常量（HISTORY\_ACTION\_LEN=20、CURRENT\_TIMELINE\_INDEX=20）一致。LIBERO 用 base FastWAM（v4 开启）\+ libero\_2cam（产出 history 字段），v4 路径会被正确触发。

## 十六、服务器 v4 切换与 smoke 记录（26\-06\-28）

**状态：**本节是服务器实际操作记录。当前只完成 forward\-only smoke；没有启动正式训练，没有 backward、optimizer step 或 checkpoint 保存。

### 16\.1 服务器代码状态

- H200\-1 的 `/data/home/frank/projects/FastWAM` 已切到 `mem-stage-v4`，commit `e6bf164`。这是 conda env 默认 `import fastwam` 实际指向的 repo。

- H200\-1 的 `/data/home/maxliu/projects/FastWAM` 也已切到 `mem-stage-v4`，commit `e6bf164`；其中 `data` 保持为指向 Frank repo 数据目录的未跟踪软链接。

- H200 当前无法解析 `github.com`，因此本次不是在服务器直接 `git fetch origin`，而是从本地 `mem-stage-v4` 生成 `git bundle` 传到服务器后切分支。

- Frank repo 的 `.git` 对 `maxliu` 不可写，直接 fetch 会报 `.git/FETCH_HEAD: Permission denied`；实际用 root 入口执行 `sudo -u frank` 切换，避免污染 repo 权限。

- H200\-2 的 `/data/home/frank/projects/FastWAM` 此前仍在 `feat/mem-vae`，本轮未切。

```bash
cd /data/home/frank/projects/FastWAM
git status --short --branch
git rev-parse --short HEAD

/data/home/frank/.conda/envs/fastwam/bin/python - <<'PY'
import fastwam
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
print('fastwam', fastwam.__file__)
print('has_training_loss_v4', hasattr(FastWAM, '_training_loss_v4'))
print('history_action_len', getattr(FastWAM, 'HISTORY_ACTION_LEN', None), getattr(RobotVideoDataset, 'HISTORY_ACTION_LEN', None))
print('history_video_past_steps', getattr(RobotVideoDataset, 'HISTORY_VIDEO_PAST_STEPS', None))
PY
```

验证输出显示默认 import 已指向 `/data/home/frank/projects/FastWAM/src/fastwam/__init__.py`，且 `has_training_loss_v4=True`、history action 长度为 20、history video past steps 为 16。

### 16\.2 LIBERO 数据与权重状态

- H200\-1 上 LIBERO 数据目录可读：`data/libero_mujoco3.3.2/*_no_noops_lerobot`。

- text embedding cache 可读：`data/text_embeds_cache/libero` 约 9\.9G，缓存文件约 10042 个。

- ActionDiT checkpoint 可读：`checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`，约 2\.0G。

- Wan2\.2 本地缓存存在，但当前服务器缓存是 `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` 与 `diffusion_pytorch_model*.safetensors`；因此 smoke/训练命令需要覆盖 `model.redirect_common_files=false`。默认 `true` 会去找不存在的 `Wan2.2_VAE.safetensors`，再触发 ModelScope 联网下载；服务器当前无法解析 `www.modelscope.cn`。

### 16\.3 forward\-only smoke 命令口径

smoke 使用真实 LIBERO batch 和真实模型权重，但只执行 `model.training_loss(batch)` 前向，不做 backward、不更新参数、不保存 checkpoint。

```bash
cd /data/home/frank/projects/FastWAM
mkdir -p runs
CUDA_VISIBLE_DEVICES=1 \
DIFFSYNTH_SKIP_DOWNLOAD=true \
/data/home/frank/.conda/envs/fastwam/bin/python -u <forward-only smoke script>

# Hydra overrides
task=libero_uncond_2cam224_1e-4
+data.train.pretrained_norm_stats=/data/home/frank/projects/FastWAM/runs/mem_stage2_v2_smoke/dataset_stats.json
batch_size=1
num_workers=0
model.mot_checkpoint_mixed_attn=true
model.redirect_common_files=false
model.load_text_encoder=false
```

### 16\.4 smoke 一手结果

- dataset smoke 通过：`dataset_len=277713`。

- 单 batch shape：`video=(1,3,9,224,448)`、`history_video=(1,3,5,224,448)`、`action=(1,32,7)`、`history_action=(1,20,7)`、`proprio=(1,32,8)`、`context=(1,128,4096)`；浮点 tensor 均 finite。

- video decode 出现非阻塞 warning：TorchCodec 因缺 `libnppicc.so.12` 加载失败，fallback 到 torchvision/pyav。功能可跑，但正式训练吞吐可能受影响。

- 当前代码直接 forward 会失败：`WanVideoDiT.pre_dit` 的自定义 `temporal_position_ids` 路径使用 CUDA position id 索引 CPU `self.freqs[0]`，报 `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`。`ActionDiT.pre_dit` 的自定义 `position_ids` 路径也有同类风险。

- 运行时临时把 `model.video_expert.freqs` 和 `model.action_expert.freqs` 移到 GPU 后，forward\-only smoke 通过：`loss=3.4006836`、`loss_video=1.6419269`、`loss_action=1.7587568`。

### 16\.5 训练前必须处理

- **必须修复：**RoPE cache device 问题。建议在 `WanVideoDiT.pre_dit` 和 `ActionDiT.pre_dit` 中确保 position id 与 RoPE cache 在索引时处于同一 device，或把 RoPE cache 注册为 non\-persistent buffer 随模型迁移。

- **必须重跑：**修复后在 H200\-1 的 `/data/home/frank/projects/FastWAM` 不带 monkey patch 重跑 forward\-only smoke。

- **建议随后再做：**1\-step backward smoke（仍非正式训练）和 LIBERO / RobotWin 各 1 个短 episode，检查 online history buffer、padding mask 与 action queue 行为。

> (Note: The content is generated by AI. Please use with caution.)
