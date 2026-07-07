# mem-stage-v5：IDM 少步 Future Video 推理方案

**当前结论：**v5 建议在 v4 的 history/current/action memory 基础上，重新引入 IDM 式少步 future video imagination。目标不是回到完整 FastWAM-IDM，而是在每次 replan 时先用 1-2 step 生成短程 future video suffix，再把 `history/current video + predicted future video + history action` 做成 condition K/V cache，最后仍用 action denoise loop 输出 future action chunk。第一版优先验证鲁棒性收益，尤其 LIBERO-plus；不把该路线混入 v4 action-only 默认路径。

## 一、动机

v4 的推理路径是 action-only：

```text
history/current video + history action
  -> clean condition prefill
  -> condition K/V cache
  -> future action denoise
```

这条路线完全不生成 future video。优点是快、简单、和 FastWAM 的 action-only 设定一致；缺点是 action 对未来状态没有显式视野，遇到扰动、遮挡、接触不确定性时，可能只能依赖当前/历史 memory 间接推断。

v5 要测试的问题是：**少量 future video denoise 是否能给 action branch 提供更强的短程物理 foresight，从而改善 LIBERO-plus 这类鲁棒性场景。**

## 二、基线路线对照

### 2.1 v4 action-only

当前 v4 推理中，`history_video` 被 VAE encode 后用 clean timestep 0 进入 `video_expert.pre_dit`，再通过 `mot.prefill_video_cache` 生成逐层 video K/V cache。`history_action` 同样以 clean timestep 0 生成 action-history cache，二者拼成 condition cache，future action 每个 denoise step 读取它。

代码参考：

- 当前 v4 history video cache：`src/fastwam/models/wan22/fastwam.py:1451`
- 当前 v4 history action cache：`src/fastwam/models/wan22/fastwam.py:1509`
- 当前 v4 action denoise loop：`src/fastwam/models/wan22/fastwam.py:1576`

### 2.2 旧 FastWAM-IDM

`main` 分支的 IDM 是两阶段：

```text
current frame
  -> initialize noisy video latent chunk
  -> clamp latent frame 0 to current frame
  -> video-only denoise N steps
  -> freeze denoised video as clean condition
  -> video K/V cache
  -> action denoise N steps
```

旧 IDM 每个 video denoise step 后都会把第 0 帧写回真实 current frame，避免 current observation 被 scheduler update 改坏。

代码参考：

- old IDM video denoise stage：`main:src/fastwam/models/wan22/fastwam_idm.py:380`
- old IDM condition cache：`main:src/fastwam/models/wan22/fastwam_idm.py:400`
- old IDM action denoise stage：`main:src/fastwam/models/wan22/fastwam_idm.py:430`

### 2.3 旧 FastWAM-Joint

Joint 不是两阶段，也不是先 video denoise 完再 cache 给 action。它每个 denoise step 同时把 video/action noisy tokens 放进 MoT，输出 `pred_video` 和 `pred_action`，再同时 scheduler step。

代码参考：

- old Joint `_predict_joint_noise`：`main:src/fastwam/models/wan22/fastwam.py:571`
- old Joint synchronized loop：`main:src/fastwam/models/wan22/fastwam.py:854`

因此 v5 更适合从 IDM 形态改，而不是从 Joint 形态改。

## 三、v5 推理数据流

推荐 v5 推理结构：

```text
history/current video
  -> VAE encode clean prefix

future video suffix
  -> initialize Gaussian noisy latent suffix

video imagination stage, 1-2 steps:
  clean prefix + noisy future suffix
  -> video_expert / scheduler step
  -> reset clean prefix to real history/current latents

condition prefill stage:
  clean prefix + predicted future suffix
  -> video_expert.pre_dit with clean/condition timestep
  -> layer-wise video K/V cache

history_action
  -> action_expert.pre_dit with clean timestep 0
  -> layer-wise history action K/V cache

future_action noisy tokens
  -> action denoise loop, e.g. 10 steps
  -> Q attends [video K/V + history action K/V + future action self K/V]
  -> output future action chunk
```

v5 cache 语义和 v4 不同：

- v4 cache：真实 history/current video + 真实 history action。
- v5 cache：真实 history/current video + predicted future video + 真实 history action。

## 四、核心参数

旧 IDM 只有一个 `num_inference_steps`，同时用于 video stage 和 action stage。v5 应拆成两个参数：

```yaml
inference:
  num_video_inference_steps: 1   # or 2
  num_action_inference_steps: 10
```

模型接口建议：

```python
infer_action(
    ...,
    history_video=...,
    history_action=...,
    num_video_inference_steps=1,
    num_action_inference_steps=10,
)
```

代码层面，任意正整数 step 都由 scheduler 支持。`WanContinuousFlowMatchScheduler.build_inference_schedule` 只要求 `num_inference_steps > 0`，然后用 `torch.linspace` 构造 schedule；DiT 本身只是吃当前 timestep 做 forward。也就是说，step 数是外层 scheduler/loop 决定的，不是 Wan DiT 内部固定能力。

## 五、clean prefix reset 设计

旧 IDM 只 reset 第 0 帧，因为旧输入只有 current frame 是 clean condition。v5 的 clean condition 是一个 prefix：

```text
history/current clean prefix + future noisy suffix
```

因此第一版应在每个 video denoise step 后 reset 整个 clean prefix：

```python
latents_video[:, :, :clean_prefix_latent_frames] = clean_prefix_latents
```

这样可以保证 history/current 永远是真实观测，不被 video denoise update 改写。

### 5.1 是否浪费计算

会有一些浪费，但第一版可以接受。原因：

- reset 本身只是一次 tensor slice assignment，成本很小。
- 真正的成本来自 Video DiT forward。clean prefix 作为 query 参与 video forward，会消耗 attention/MLP 计算。
- attention mask 可以限制 clean prefix 不看 future suffix，但不能阻止 clean prefix token 自己被投影成 Q/K/V、经过 block、产生输出。mask 解决信息流，不自动省掉计算。

### 5.2 是否能不碰 history/current

可以，但需要更大的结构改动。更省的版本是：

```text
clean history/current prefix
  -> prefill once as video condition cache

future video suffix denoise loop
  -> only future video suffix as Q
  -> attends clean prefix K/V
  -> scheduler step only future suffix
```

这会避免 clean prefix 在每个 video denoise step 里作为 Q 重复更新，更接近 cache-optimized video imagination。但它不再等价于旧 IDM 的 full-video denoise，也和 v4 training 中 clean prefix + future suffix 走同一 Video DiT path 的实现有差异。

**建议：**v5 第一版采用 reset clean prefix 的正确性优先方案；如果 1-2 step video imagination 显示有收益，再做 prefix-cache video-denoise 优化。

## 六、训练设计

v5 不能只改推理，否则 action branch 可能没有学过读取 predicted/noisy future video condition。推荐训练路径和推理对齐：

```text
history/current clean video prefix + future noisy video suffix
  -> Video DiT
  -> future video loss
  -> clean prefix + future video condition cache

history action clean cache

future action noisy query
  -> reads video condition cache + history action cache
  -> action loss
```

第一阶段可以先做推理 ablation，验证工程可跑和延迟上限；如果结果有信号，再做 v5 正式训练。

训练时要明确区分三类 video token：

- `history_video`：clean condition。
- `current_video`：clean condition，必须保留。
- `future_video`：noisy target / predicted condition。

action 侧仍区分：

- `history_action`：clean condition。
- `future_action`：noisy target query。

source/type embedding、统一 position id、padding mask 规则沿用 v4。

## 七、从哪里开始实现

### 7.1 建议基于 v4 改

建议 v5 从当前 v4 分支发展，而不是从原版 FastWAM/main 开始。原因：

- v4 已经有 history video/history action 数据接口。
- v4 已经有 source/type embedding 和统一时间轴。
- v4 已经有 history action clean cache 与 condition cache 拼接。
- v4 已经处理了 online history action buffer 的语义问题。

原版 FastWAM-IDM 只适合作为参考实现，尤其是两阶段 video-denoise-then-action-cache 的结构。

### 7.2 原版 IDM 仍值得先跑 sanity test

如果已有对应 IDM checkpoint，可以在 `main` 或单独 worktree 上先跑旧 IDM 的 `num_inference_steps=1,2,4,10`，确认少步 video imagination 的延迟和粗略性能曲线。

但这只是 sanity test，不应替代 v5 实现。旧 IDM 没有 v4 history memory，不能回答“history + 少步 future video”是否提升鲁棒性。

## 八、实现计划

### Step 1：恢复 IDM 参考代码

从 `main` 参考以下文件，而不是直接整体回滚：

- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/runtime.py` 中 `create_fastwam_idm`
- `configs/model/fastwam_idm.yaml`

v5 可新建 `fastwam_idm_v5.py` 或在 `FastWAM` 内新增 `infer_action_v5_idm`，避免污染 v4 默认入口。

### Step 2：扩展 video imagination 输入

旧 IDM：

```text
current frame + noisy future video
```

v5：

```text
history/current clean prefix + noisy future video suffix
```

需要复用 v4 的：

- `_history_video_source_and_position_ids`
- `_combined_video_source_and_position_ids`
- `_latent_valid_from_raw_pad`
- source/type embedding
- unified temporal position ids

### Step 3：拆分 step 参数

新增：

```python
num_video_inference_steps: int = 1
num_action_inference_steps: int = 10
```

video scheduler 使用 `num_video_inference_steps`，action scheduler 使用 `num_action_inference_steps`。

### Step 4：video denoise stage

初版直接 full video path：

```text
combined_video_latents = cat(clean_prefix_latents, noisy_future_latents)
for video_step:
    pred_video = video_expert(...)
    combined_video_latents = video_scheduler.step(...)
    combined_video_latents[:, :, :prefix_len] = clean_prefix_latents
```

只把 future suffix 视为可变预测目标；clean prefix 每步强制恢复。

### Step 5：condition cache

video denoise 完后，把 `clean prefix + predicted future suffix` 作为 condition video：

```text
video_pre_cond = video_expert.pre_dit(..., timestep=0 or condition timestep)
video_kv_cache = mot.prefill_video_cache(...)
```

这里是否统一用 timestep 0 需要保留为实现决策。旧 IDM 使用 `timestep_video_cond = 0`，表示 denoised video 已作为 clean condition。v5 第一版建议沿用这个做法。

### Step 6：history action cache

沿用 v4：

```text
history_action -> action_expert.pre_dit(timestep=0, source_id=history_action)
history_action_cache = mot.prefill_action_cache(...)
```

然后：

```text
condition_kv_cache = concat(video_kv_cache, history_action_cache)
```

### Step 7：action denoise stage

沿用 v4 的 `forward_action_with_condition_cache`：

```text
future_action noisy query
  -> action denoise N steps
  -> reads [video condition cache + history action cache]
```

## 九、实验计划

优先做小矩阵，不一上来大训练：

|实验|video steps|action steps|目的|
|---|---:|---:|---|
|v4 baseline|0|10|当前 action-only 基线|
|v5-idm-1v|1|10|最小 future video imagination|
|v5-idm-2v|2|10|稍强 future video imagination|
|v5-idm-4v|4|10|观察收益是否继续增长|
|old-idm sanity|1/2/4/10|同 video|只验证旧实现趋势，非最终结论|

指标：

- LIBERO standard success rate。
- LIBERO-plus full 10030 success rate。
- 单次 replan latency。
- action L1/L2 离线指标。
- 可选：pred future video 可视化，只作为诊断，不作为 policy 输出。

