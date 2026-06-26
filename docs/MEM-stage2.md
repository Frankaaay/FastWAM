# MEM-stage2 — DiT-side history prepend(全解冻 DiT)

> **重新设计**短期记忆的融合位置:**彻底去掉 VAE 侧 temporal attention**,改为在 **MoT 的 video
> prefill 路径里 prepend 历史 latent 帧**——历史不再融进 latent,而是作为额外的 latent 帧拼在
> video token 序列最前面,由 video expert 自己的 self-attention 去混合,action expert 通过既有的
> prefill+cache 路径读到(被历史增强过的)current-frame K/V。
>
> 分支 `MEM-stage2`(从 `feat/mem-vae` 分出)。目标不变:LIBERO-plus 超过 mem-off 49.83。

---

## 0. 速览 (TL;DR)

- **一句话**:`video_in = [history_0..K_lat-1, current, future...]`,冻结 VAE 只做 plain 逐帧 encode,
  历史当作额外 latent 帧 prepend;**全解冻 DiT** 联合训练(option C / 6.1,v1 不 crop)。
- **训练面**:零新增 `nn.Parameter`(base ckpt **strict 加载**);可训练 = **全 DiT(video+action ~5.9B)+ proprio**。
- **结论**:**失败**。prepend 改变了后端接口(current 的 temporal index 从 0 变成 K),全 DiT
  finetune 后策略**依赖**这条 prepend 路径;一旦回到 mem-off(current 回 index 0)就是 OOD →
  扰动集崩(≈18.5,远低于 mem-off ~50.5)。
- **状态**:路线**已弃**,不再扩大训练。结论直接催生 stage3(fold-current,**不暴露 history 给后端**)。

> 基线口径同 [MEM-stage1.md](MEM-stage1.md#基线口径三篇统一):mem-off 49.83(noise-inclusive 10030),标准 LIBERO base 95.9。

---

## 1. 架构设计

### 1.1 动机:从 VAE 侧搬到 DiT 侧

stage1 把记忆塞进**冻结 VAE**,DiT 读不懂改写过的 latent(接口错配,见 [stage1 §4.1](MEM-stage1.md))。
stage2 换思路:**冻结 VAE 只做 plain encode(逐位等于 base 的 conditioning latent)**,记忆改在
**DiT token 空间**注入,且让 DiT 自己适应——所以这一版**全解冻 DiT**(对照 stage1 的"冻死 DiT"那一极)。

MoT 结构本就适合承载 action-centric memory:每层 action expert 用 action query attend
`K=[K_v,K_a]` / `V=[V_v,V_a]`,即 **action 每一层都在读 video cache**,不是只在输入层。推理路径
`prefill_video_cache(...)` → 循环 `forward_action_with_video_cache(...)`,历史压缩只在 prefill 发生一次。

### 1.2 Pipeline

```
history pixels [B,3,K_px,H,W]
      │ frozen VAE.encode (plain,无 temporal)        K_px 必须 4n+1
      ▼
  K_lat 个历史 latent 帧
      │ prepend 到 [current, future] 最前面
      ▼
  video_in = [history_0..K_lat-1, current, future_1..future_{T-1}]
      │ pre_dit:history+current 标记为 clean t=0
      ▼
  MoT(history-aware mask)            ← video expert self-attn 混合历史→current
      ▼
  video loss 只算 future;action expert attend current-frame K/V
```

代码锚点:历史 plain encode [fastwam.py:394 `_encode_history_latents`](../src/fastwam/models/wan22/fastwam.py#L394);
prepend 拼接 [fastwam.py:689](../src/fastwam/models/wan22/fastwam.py#L689) `video_in = cat([history_latents, latents])`,
`seq_history = num_history_frames`([:702](../src/fastwam/models/wan22/fastwam.py#L702));
prepend 帧标记 clean t=0 [wan_video_dit.py:733](../src/fastwam/models/wan22/wan_video_dit.py#L733);
video loss 切掉历史只算 future [fastwam.py:780](../src/fastwam/models/wan22/fastwam.py#L780)。

### 1.3 核心机制(逐个拆)

**(a) prepend,不 crop(v1)** — 历史 latent 保留到所有层,复用 video 自身 self-attention 混合
(不新增 temporal module),改动最小。

**(b) frozen plain VAE encode + 4n+1 chunk 规则** — [fastwam.py:415](../src/fastwam/models/wan22/fastwam.py#L415)。
冻结 VAE 把**首帧单独成 chunk、之后每 4 帧一 chunk**(`K_lat = 1 + (K_px-1)//4`),喂 4n(如 H4)会
**静默丢掉最近 3 帧**;故 H5 → `[h0][h1..h4]` 全编码、0 丢失,K_lat=2。历史**单独 encode、不拼当前帧**
(拼了会污染 current conditioning latent)。

**(c) current → 不看 future(mask 规则)** — current 帧的 K/V(action 读的那份 memory)**与 future
是否存在无关** → 训练(有 future)与推理(无 future)产出**完全一致**的 memory K/V。这修掉了 stage1-v3
的 train/infer 裂缝。

**(d) 全解冻 DiT** — `enabled=false` 走 trainer 默认分支:`model.dit.requires_grad_(True)`
([trainer.py:391](../src/fastwam/trainer.py#L391)),video+action expert + proprio 全可训。

### 1.4 关键不变量(设计时认为成立)

- **零新增 `nn.Parameter`** → base ckpt **strict 加载**,无 missing/unexpected key。
- **不碰 conditioning latent**:current conditioning = base 原版 plain encode,逐位一致。
- **3D RoPE 相对位置**:prepend 只把绝对 index 平移,current↔future / future↔future 相对关系不变。
- **mask K=0 退化为原版 first_frame_causal**:历史关掉时逐位等于原版。

> ⚠️ 上面第 1/2/3 条都成立,**但仍失败**——见 §4。问题不在"输入是否被扰动",而在"全解冻让策略
> 依赖了 prepend 这条新路径"。

---

## 2. 训练配置

| 项 | 值 |
|---|---|
| 分支 / commit | `MEM-stage2`(HEAD 训练时 69b73d1),node-1 |
| 开关 | `model.vae_memory.enabled=false` + `model.vae_memory.dit_prepend=true`(互斥) |
| 起点 ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt`(base,mem-off 49.83) |
| resume | weights-only 冷启动 |
| 可训练 | **全 DiT(video+action expert ~5.9B)+ proprio**(trainer 默认分支) |
| history | **H5** → 1.0s,K_lat=2。⚠️ 必须 4n+1(见 §1.3b) |
| 卡 / batch | **8 卡 × BS=24 = global 192**(bs32 即便 `expandable_segments:True` 仍差 184MiB OOM:prepend 多 2 帧 + 全 DiT 的真实激活) |
| lr | **1e-5**,cosine + 5% warmup |
| loss 权重 | lambda_video=1.0 / lambda_action=1.0 |
| max_steps | 20000;save_every 1000 |
| output_dir | `runs/mem_stage2_v1` |

**smoke 验证**:
- 2026-06-23 8 卡 1-step smoke PASS:base ckpt strict 加载、ZeRO-1 分片不 OOM、optimizer+ckpt 链路通,
  `loss=0.1178`(≪ stage1-v3 冷启动 2.18 → 印证"贴原版、不扰动"方向)。但用 H4(退化只编码最老一帧),不能当真实代理。
- 2026-06-24 8 卡 **H5** smoke 重跑 PASS(`HEAD=69b73d1`,global 32):`loss=0.1582`,4n+1 修复 + 8 卡路径双双坐实。
- 2026-06-24 正式 20000-step 上线(BS=24,global 192):`step10 loss=2.116 → 30 loss=2.004` 平稳下降,
  ~124GB/卡、~4.3s/step(ETA ≈ 24h)。loss 起点 ~2.1 是正常冷启动值(对得上 stage1-v3 的 2.18;
  smoke 的 0.12/0.16 是小 batch 只抽到低噪声 timestep 的假象)。

---

## 3. 结果

> ⚠️ 来源标注:实验 log 当时停在 "TBD";后续评测结论记录在 stage 复盘里——prepend 路线在扰动集上
> **崩**(perturbed ≈ 18.5,对照 mem-off ≈ 50.5)。若要复核请重跑
> `INCLUDE_NOISE=1` 的 libero-plus 并回填本表。

| step | Total | Camera | Robot | Lang. | Light | BG | Layout | 判读 |
|---|---|---|---|---|---|---|---|---|
| (扰动集) | **≈18.5** | | | | | | | 远低于 mem-off ~50.5,**失败** |

---

## 4. 分析

### 4.1 为什么失败:prepend 改变了后端接口

stage2-v1 的真实问题**不是超参或早停**,而是 history prepend 改了后端接口:

- 原版 FastWAM action 只读 **current-frame** video cache。
- prepend 把 video 序列变成 `[history_0..K-1, current, future...]`,**current 的 temporal index 从 0 变成 K**。
- action 仍只读 current,但这个 current 已经过 history-prepend 路径重塑。
- **全 DiT finetune 后,策略会依赖这条路径**;一旦 mem-off / current 回到 index 0,就是 **OOD** → 扰动集崩。

这与 stage1 的两个失败模式都不同:stage1-v1 是"DiT 读不懂记忆 latent"(接口错配但 rollout 不崩),
stage1-v3 是"全解冻 train/rollout 裂缝"。stage2-v1 这里是**第三种**:接口虽未扰动 conditioning,
但**把记忆变成了一条新的 policy backbone**,mem-off 退不回去。

### 4.2 给 stage3 的结论

- **不能让 history token 暴露给 MoT/action**——否则记忆就会长成一条新 backbone。
- 正确做法:history 只用于**增强 current tokens**,随后**立即丢弃** history tokens,后端 MoT/action
  看到的 token 数、current index、attention mask 全保持 base 形态。
- 这正是 stage3 的 fold-current adapter:见 [MEM-stage3.md](MEM-stage3.md)。

### 5. 目录速查

| 内容 | 路径 |
|---|---|
| 训练脚本 / smoke | `scripts/train_mem_stage2_v1.sh` / `scripts/train_mem_stage2_v1_smoke.sh` |
| run 目录 | `runs/mem_stage2_v1/` |
| 评测脚本 | `scripts/eval_mem_stage2_v1.sh`(锁 VAE_MEM=false + DIT_PREPEND=true + H5 + INCLUDE_NOISE=1) |
| prepend 代码 | `fastwam.py`(`_encode_history_latents` :394,prepend :689);`wan_video_dit.py`(clean 标记 :733) |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
