# MEM IDEA

## 现有 MoT 结构基础

当前 `MoT` 结构并不是两个完全独立的分支。每一层会分别从 video expert 和 action expert 构造：

$Q_v, K_v, V_v$

$Q_a, K_a, V_a$

随后执行 mixed attention。尤其在 inference action 时，代码路径为：

```Python
video_kv_cache = self.mot.prefill_video_cache(...)
action_tokens = self.mot.forward_action_with_video_cache(...)
```

其含义是：video tokens 先被预填充成每一层的 cached $(K_v, V_v)$，然后 action expert 在每个 denoising step、每一层里用 action query 去 attend：

$K = [K_v, K_a]$

$V = [V_v, V_a]$

因此，action 不是只在输入层看到 video，而是每一层都在读 video cache。这个结构适合承载 action\-centric memory。

当前限制在 `_build_mot_attention_mask()` 里：

```Python
# action -> first-frame video only
first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
mask[video_seq_len:, :first_frame_tokens] = True
```

当前 action 仅读取第一帧 video tokens。方案重点不是让 video expert 单独预测得更好，而是将这批 “first\-frame video tokens” 替换为 **short\-term memory tokens**：

$\tilde V_t = \mathrm{MemoryEncoder}(z_{t-K:t})$

随后 action 每一层直接 attend 到 $K_{\tilde V_t}, V_{\tilde V_t}$，从而形成直接的 action 提升路径。

---

## 方案核心：Action\-MEM Cache

核心机制是：用 video expert / temporal memory module 预先把历史视觉压缩成一组 memory K/V cache，然后 action expert 在 denoising 的每一步直接读取这些 cache。

推理流程是：

$x_{t-K:t} \xrightarrow{\text{Frozen VAE}} z_{t-K:t}$

> ★★★需要维护一份 cache，用于复用已完成编码的 $z_{t-K:t-1}$；当前步只需对 $z_t$ 做 encoding。
> 
> 

$z_{t-K:t} \xrightarrow{\text{Video-side Temporal Memory Prefill}} \{K^{\text{mem}}_\ell, V^{\text{mem}}_\ell\}_{\ell=1}^{L}$

每个 action denoising step 执行：

$a^\sigma \xrightarrow{\text{Action Expert}} Q^{\text{action}}_\ell$

$Q^{\text{action}}_\ell \text{ attends to } [K^{\text{mem}}_\ell, K^{\text{action}}_\ell]$

输出为：

$\hat{\epsilon}_a \quad \text{or} \quad \hat{v}_a$

在该方案中，video expert 不以生成 video 为主要目标，而是作为 **action expert 的 memory encoder**。训练时主要优化：

$\mathcal{L}_{\text{action}}$

避免依赖以下间接路径：

$\mathcal{L}_{\text{video}} \downarrow \Rightarrow \mathcal{L}_{\text{action}} \downarrow$

这样可以避免只优化 video loss 却无法稳定提升 action 的问题。

---

## 与 MEM 机制的对应关系

MEM 的关键是：短程视频 memory 通过 temporal attention 把过去 observation 压缩进当前 representation，并且只把当前 timestep representation 传给后面的 VLA backbone。它还特别强调，不直接把多帧全部送入 backbone，因为这样 inference latency 会快速上升；它通过 factorized temporal/spatial attention 和丢弃 past tokens 来保持后端 token 数接近单帧。

该思想可以迁移到 action 目标：

$z_{t-K:t} \rightarrow \tilde z_t^{\text{mem}} \rightarrow \text{action expert}$

因此，方案不要求 video expert 输出完整 future video；它只需要产出适合 action expert 读取的 memory K/V。

---

## Inference控制：在 prefill 阶段压缩历史信息

当前 inference 已经把 video cache prefill 和 action denoising 分开。这个结构有利于控制额外开销。

当前流程为：

```Python
video_kv_cache = self.mot.prefill_video_cache(...)
for denoise step:
    action = self.mot.forward_action_with_video_cache(...)
```

实现上保持这个结构，只把 `prefill_video_cache` 升级成：

```Python
memory_kv_cache = self.mot.prefill_action_memory_cache(...)
```

它做的事情是：

$z_{t-K:t} \rightarrow \text{temporal attention} \rightarrow \tilde z_t \rightarrow \{K^{\text{mem}}_\ell, V^{\text{mem}}_\ell\}_{\ell=1}^{L}$

之后 action denoising 仍然只读取压缩后的 memory cache。

推理成本变成：

$\text{Cost} = \text{memory prefill once} + T_{\text{denoise}}\cdot \text{action denoise}$

而不是：

$T_{\text{denoise}}\cdot \text{full history attention}$

关键差别在于：历史压缩只在 prefill 阶段发生，不在每个 denoising step 重复计算。

---

## Temporal Attention 的放置位置

建议将 temporal attention 放在 **MoT 的 video prefill path**：

```Python
MoT.prefill_video_cache()
```

新增方法：

```Python
MoT.prefill_action_memory_cache()
```

该方法与原有 `prefill_video_cache()` 类似，但增加两个功能。

第一，在前几层允许 video history tokens 做 temporal attention：

$X_{t-K:t} \rightarrow X_{t-K:t}^{\text{mem}}$

第二，到某一层后丢弃 past tokens，只保留 current tokens：

$[B, K N, D] \rightarrow [B, N, D]$

之后每层只继续处理 current memory tokens，并缓存当前层的 $(K, V)$。

这与 MEM 的 “upper layers drop past timestep tokens” 是同一个思路。

---

## 实现路径

### 复用 video self\-attention

> ★★★先采用该版本试试。
> 
> 

不新增复杂 temporal attention module，先复用 video expert 自身的 self\-attention，让 history/current tokens 在前 $L_{\text{mem}}$ 层内混合，然后 crop current tokens。

流程：

$[B, K N, D] \xrightarrow{\text{video blocks }0\ldots L_{\text{mem}}} [B, K N, D]$

$\text{crop current frame} \Rightarrow [B, N, D]$

$[B, N, D] \xrightarrow{\text{video blocks }L_{\text{mem}}\ldots L} \{K^{\text{mem}}_\ell, V^{\text{mem}}_\ell\}$

该版本改动小，可以直接验证“历史视觉进入 action cache 是否提升 action”。缺点是前 $L_{\text{mem}}$ 层如果用 full spatio\-temporal attention，prefill 会变重：

$O((K N)^2)$

但它只发生一次，不在每个 denoise step 重复。第一版可以控制：

$K=3 \text{ or } 4$

$L_{\text{mem}}=2 \text{ or } 4$

用于 feasibility test。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MjgwNzliNjJkMzM2ZjE1ZDMyMjI3ODU0ODg5ZDNmY2ZfYzBlMzE3MjY5N2M1MmE2NGViMzFiYTk5N2EzMjExY2VfSUQ6NzY1NDI1NjIxMDgwMzMxMzYyM18xNzgyMTg3OTA5OjE3ODIyNzQzMDlfVjM)



---

### MEM\-style temporal attention

正式版可增加一个 lightweight temporal branch，而不是使用 full spatio\-temporal attention。

对 video tokens reshape：

$X \in \mathbb{R}^{B\times K\times N\times D}$

对同一个 spatial position $p$ 跨时间做 temporal attention：

$X_{:, :, p, :}$

复杂度是：

$O(NK^2D)$

而不是：

$O(K^2N^2D)$

然后把 temporal 信息写入 current tokens：

$X_t \leftarrow X_t + \alpha \cdot \mathrm{TemporalAttn}(X_{t-K:t})$

最后只保留$X_t$

该版本接近 MEM 的核心思路：temporal attention 融合历史，current token 输出，后端 token 数不变。MEM 也是通过 factorized temporal/spatial attention 避免直接全时空 attention 的高成本。

> (Note: The content is generated by AI. Please use with caution.)

---

# 实施说明

> 分支 `MEM-stage2`。下面只讲**实际改了哪些地方**。核心一句话:**关掉 VAE temporal,
> 冻结 VAE 只做 plain 逐帧 encode,把历史 latent 帧 prepend 到 video 序列最前面**,由
> video expert 自身的 self-attention 混合,action 通过既有 prefill+cache 读 current K/V。
> **零新增 nn.Parameter**(base ckpt strict 加载),trainer 无需改动。

## 0. 帧布局与不变量

沿 latent 时间轴排成:`[history_0 .. history_{K-1}, current, future_1 .. future_{T-1}]`,
其中 `K = num_history_frames`(latent 帧数,**H5 → K_lat=2**)。video self-attention 的帧级规则:

> **历史像素帧必须 4n+1。** 冻结 VAE 的 plain `encode` 把首帧单独成一个 chunk、之后每 4 帧
> 一个 chunk(`iter_ = 1 + (K_px-1)//4`),所以历史**单独编码**时必须 4n+1,否则最近 3 帧落单被
> 丢(H4 只编码到最老一帧)。**不**把 current 拼进历史一起编码——那会让 current 的 conditioning
> latent 被历史污染,破坏"current 与 base 逐位一致"的核心不变量。H5(5 帧=1.0s)→ K_lat=2。

- history(`q < K`)→ 只看 history(`k < K`)
- current(`q == K`)→ 看 history+current(`k <= K`),**不看 future**
- future(`q > K`)→ 看 current+all future(`k >= K`),**不看 history**

**关键不变量:** `current → 不看 future`,使 current 帧 K/V(就是 action 读的 memory)
与 future 是否存在无关 → **训练(有 future)与推理(无 future)产出一致的 memory**,修掉
stage-3 的 train/infer 裂缝。`K=0` 时规则退化为原版 `first_frame_causal`。

## 1. `src/fastwam/models/wan22/wan_video_dit.py`

- 新增静态 helper [`_build_history_causal_mask`](../src/fastwam/models/wan22/wan_video_dit.py#L474-L516):
  实现上面的帧级规则
  `((q<K)&(k<K)) | ((q==K)&(k<=K)) | ((q>K)&(k>=K))`,再 `repeat_interleave(tokens_per_frame)`
  扩到 token 级。`0 < K < num_frames` 越界即 raise。
- [`build_video_to_video_mask`](../src/fastwam/models/wan22/wan_video_dit.py#L518-L556) 新增
  `num_history_frames: int = 0`;`first_frame_causal` 分支在 `num_history_frames>0` 时改派到上面
  的 helper。
- [`pre_dit`](../src/fastwam/models/wan22/wan_video_dit.py#L573-L605) 新增 `num_history_frames`;
  把 `token_timesteps[:, 0, :] = 0` 改成 `token_timesteps[:, : num_history_frames + 1, :] = 0`
  → **history + current 都标记为 clean(t=0)**,future 保留采样噪声 timestep。

## 2. `src/fastwam/models/wan22/fastwam.py`(base FastWAM)

- [`__init__` / `from_wan22_pretrained`](../src/fastwam/models/wan22/fastwam.py#L49-L233) 新增
  `dit_history_memory: bool = False` → `self.dit_history_memory_enabled`;与 `vae_memory_enabled`
  **互斥**(同开即 raise),透传给 `cls(...)`。
- 新增 [`_encode_history_latents`](../src/fastwam/models/wan22/fastwam.py#L363)(`@torch.no_grad`):
  校验 `[B,3,K,H,W]` 且 **`K%4==1`(4n+1,防 VAE 丢最近帧)**,返回 plain `_encode_video_latents`
  → `[B,z,K_lat,h,w]`,`K_lat = 1 + (K-1)//4`。
- [`build_inputs`](../src/fastwam/models/wan22/fastwam.py#L459-L540):新增分支
  `if self.dit_history_memory_enabled:` 编码历史 latent;旧 vae_memory 分支挪到 `else:`;
  返回里带 `"history_latents"`。
- [`_build_mot_attention_mask`](../src/fastwam/models/wan22/fastwam.py#L554-L573):新增
  `num_history_frames`,透传给 `build_video_to_video_mask`;**action→video 改指向 current 帧**
  (current 在时间 index `num_history_frames`,不是 0):
  `cur_start = num_history_frames * video_tokens_per_frame; mask[video_seq_len:, cur_start:cur_end] = True`。
- [`training_loss`](../src/fastwam/models/wan22/fastwam.py#L640-L718):取 `history_latents`,
  `video_in = torch.cat([history_latents, latents], dim=2)`,`num_history_frames = K_lat`;
  pre_dit / mask 调用都带上 `num_history_frames`;post_dit 后**先切掉历史帧**
  `pred_video = pred_video[:, :, num_history_frames:]`,再走既有 `[:,:,1:]` 取 future 算 loss。
- [`infer_action`](../src/fastwam/models/wan22/fastwam.py#L1148-L1226):新增
  `if self.dit_history_memory_enabled and history_images is not None:` 分支,
  `video_in = torch.cat([history_latents, current_latent], dim=2)`,同样透传 `num_history_frames`;
  旧 vae_memory 走 elif,无历史走单帧。(仍要求 `video_attention_mask_mode=='first_frame_causal'`。)

## 3. `src/fastwam/models/wan22/fastwam_joint.py`

- [`_build_mot_attention_mask`](../src/fastwam/models/wan22/fastwam_joint.py#L28-L51) override 新增
  `num_history_frames` 并透传。**必须改**:FastWAMJoint 继承 base 的 `training_loss`,后者现在会
  把 `num_history_frames=` 传进这个 override,不加参数会 TypeError。

## 4. 配置 / 接线

- [`runtime.py`](../src/fastwam/runtime.py#L170):`create_fastwam` 调用里加
  `dit_history_memory=bool(vae_memory.get("dit_prepend", False))`。
- [`configs/model/fastwam.yaml`](../configs/model/fastwam.yaml):`vae_memory:` 下新增
  `dit_prepend: false`(与 `enabled` 互斥)。
- [`scripts/train_mem_stage2_smoke.sh`](../scripts/train_mem_stage2_smoke.sh):1-step smoke,
  `vae_memory.enabled=false vae_memory.dit_prepend=true history_video_frames=5`(H5,4n+1),冷启动 base ckpt。
- [`scripts/train_mem_stage2_prepend.sh`](../scripts/train_mem_stage2_prepend.sh):正式训练(8 卡 × bs32 = 256,
  max_steps=20000,H5,lr=1e-5,wandb offline)。注:旧 `train_mem_stage2.sh` 是被废弃的 patch_embed 路线。

## 5. 为什么 trainer 无需改

`trainer._apply_dit_only_train_mode`:当 `vae_memory_enabled=False`(我们的情形)走默认分支——
`model.eval(); requires_grad_(False); model.dit.train(); model.dit.requires_grad_(True)` + proprio
可训。即 **全 DiT(video+action expert)可训、其余冻结**,正是 option C 想要的,**零改动**。

