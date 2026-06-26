# MEM-stage2 实验记录(experiment log)

> 分支 `MEM-stage2`(从 `feat/mem-vae` 分出)。**重新设计**短期记忆的融合方式:
> **彻底去掉 VAE 侧 temporal attention**,改为在 **MoT 的 video prefill 路径里 prepend
> 历史 latent 帧**(option C / 文档 6.1,v1 不 crop)。
> 目标不变:在 LIBERO-plus 上 **超过无记忆基线 49.83**。
> 配套思路见 [MEM-stage2-Idea.md](MEM-stage2-Idea.md);stage1-v1/v2/v3 的旧记录见
> [mem-stage1-expriment-log.md](mem-stage1-expriment-log.md)。
>
> 

---



## 1. 设计:去掉 VAE temporal,改为 DiT-side history prepend(option C / 6.1)

> 思路转向:**冻结的 VAE 只做 plain 逐帧 encode(无 temporal attn)**;历史不再融进
> latent,而是作为**额外的 latent 帧 prepend 到 video token 序列最前面**,由 video expert
> 自己的 self-attention 去混合,action expert 通过既有的 prefill+cache 路径读到
> (被历史增强过的)current-frame K/V。

数据流:

```
history pixels [B,3,K,H,W]  --frozen VAE.encode(plain,无temporal)-->  K_lat 个历史 latent 帧
                                                                            |
                                              prepend 到 [current, future] 最前面
                                                                            v
        video_in = [history_0..K_lat-1, current, future_1..future_{T-1}]
                                                                            v
              pre_dit(history+current 标记为 clean t=0)  -->  MoT(history-aware mask)
                                                                            v
              video loss 只算 future;action expert attend current-frame K/V
```

- **option C** = 最贴原版 FastWAM 联合训练:video loss + action loss 一次 forward;历史
  prepend;video loss 只在 future 帧上算。
- **6.1** = 复用 video 自身 self-attention,**不新增 temporal module**。
- **v1 不 crop**(历史保留到所有层)——更简单,可行性信号一样,且配合下面的 mask 规则即可
  保证 train==infer。

---

## 2. 关键设计点(为什么这样能避开旧坑)

1. **current → 不看 future(mask 规则)**:current 帧的 K/V(就是 action 读的那份 memory)
   **与 future 是否存在无关** → 训练(联合,有 future)和推理(无 future)产出**完全一致**的
   memory K/V。这直接修掉了 stage1-v3 的 train/infer 裂缝(exposure bias)。
2. **不碰 conditioning latent**:current 帧 conditioning = base 原版的 plain encode,**逐位
   一致**;历史只是额外可 attend 的上下文。→ 避开 stage1-v1/v2 的"DiT 读不懂记忆 latent"。
3. **零新增 nn.Parameter** → base ckpt **strict 加载**,无 missing/unexpected key。
4. **3D RoPE 是相对位置**:prepend 只是把绝对 index 平移,current↔future、future↔future 的
   相对关系不变,只多出 history↔current 的相对位置。无需新增 sinusoidal PE。
5. **mask K=0 退化为原版 first_frame_causal**:历史关掉时行为与原版逐位相同。

---

## 3. 配置(待训练)

| 项 | 值 |
|---|---|
| 分支 / commit | `MEM-stage2` / 4f08853(+ 本次实施),node-1 |
| 开关 | `model.vae_memory.enabled=false` + `model.vae_memory.dit_prepend=true`(互斥) |
| 起点 ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt`(base,mem-off 49.83) |
| resume | weights-only 冷启动 |
| 可训练 | **全 DiT(video+action expert ~5.9B)+ proprio**(trainer 默认分支,因 enabled=false) |
| history | **H5** → 1.0s(ratio4/fps20);K_lat=2 个历史 latent 帧。⚠️ **必须 4n+1**:冻结 VAE plain encode 把首帧单独成 chunk、之后每 4 帧一 chunk(`iter_=1+(K-1)//4`),喂 4n(如 H4)会**静默丢掉最近 3 帧**;H5 → `[h0][h1..h4]` 全编码、0 丢失 |
| 卡 / batch | **8 卡 × BS=24 = global 192(实际生效)**。bs32 即便加 `expandable_segments:True` 仍差 184MiB OOM(非碎片,是真实激活:prepend 多 K_lat=2 帧 + 全 DiT),故回退 BS=24,~124GB/卡稳跑 |
| lr | **1e-5**,cosine + 5% warmup |
| loss 权重 | lambda_video=1.0 / lambda_action=1.0(option C 默认贴原版) |
| max_steps | **20000**;save_every **1000** |
| wandb | offline → jump 同步(entity yichx14-uc-irvine,project fastwam-mem) |
| output_dir | `runs/mem_stage2_v1` |

---

## 4. 进展 / 状态

- **2026-06-23 — 8 卡 1-step smoke 通过(干净 PASS)。** 见 `scripts/train_mem_stage2_v1_smoke.sh`。
  - base ckpt **strict 加载**(无 missing/unexpected key)✓
  - 全 DiT 进入训练、**多卡 ZeRO-1 分片不 OOM**(每卡 MA 15.33 GB @bs4)✓
  - **optimizer step 干净跑完**、checkpoint(weights + state)保存链路通 ✓
  - `step=1/1 loss=0.1178 loss_action=0.0262 loss_video=0.0916 lr=1e-7`
  - **初始 loss 0.118 ≪ stage1-v3 冷启动的 2.18**:因为没新增 temporal 参数、current
    conditioning 与 base 逐位一致,base 一上来就在"熟悉的输入 + 额外历史"上工作,起点健康
    (不像 stage1-v1/v2/v3 把 conditioning 改写/扰动了)。**印证"贴原版、不扰动"方向正确。**
  - ⚠️ 该 smoke 用 bs4/卡(非真实 bs32)以与他人 job 共存;真实 bs32 峰值显存待训练时确认。
  - ⚠️ **该 smoke 用 H4(4 帧),只验证了管道连通**:H4 喂进 plain VAE 实际只编码到最老一帧
    (最近 3 帧被丢),记忆内容是退化的,**不能当真实训练代理**。已修(见 §3 的 4n+1 说明),
    smoke 脚本与正式脚本均改用 **H5**。
- **2026-06-24 — 8 卡 H5 smoke 重跑通过(干净 PASS)。** node-1,`HEAD=69b73d1`,8 卡 ×
  bs4 = global 32,1 step,wandb off。
  - `step=1/1 loss=0.1582 loss_action=0.0412 loss_video=0.1170`,无 error/assert/OOM,
    weights+state 保存链路通,8 卡退出干净。
  - loss 0.158(略高于 H4 单卡 0.118):H5 真编码 2 个历史 latent 帧(非 H4 退化单帧),
    仍远低于 stage1-v3 冷启动 2.18,起点健康。**4n+1 修复 + 8 卡路径双双坐实。**
  - smoke 产出的 5.9B throwaway ckpt 已清(保留 `runs/mem_stage2_v1_smoke/smoke8_H5.log`)。
- **2026-06-24 — 正式 20000-step 首次启动 OOM(bs32),已加 alloc-conf 重启。** 真实
  bs32(global 256)在第一步 OOM:GPU0 用 137.4/139.8 GiB,仅差 184 MiB,且 ~214 MiB 是
  碎片化 reserved-unallocated。**根因:prepend 路线给 video 序列多 K_lat=2 历史帧 + 全 DiT,
  激活显存比 smoke 的 bs4 高得多,bs32 踩线。** 修法:脚本加
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(消碎片,保住 global 256);
  `BS` 参数化,仍 OOM 则 `BS=24`。
- **2026-06-24 — 正式训练上线(BS=24,global 192)。** bs32 加 alloc-conf 仍 OOM(碎片
  214→132MiB 但仍差 184MiB,确认是真实激活不够),回退 BS=24 后顺利:
  - `step=10 loss=2.116 → 20 loss=2.042 → 30 loss=2.004`,平稳下降;8 卡 ~124GB/卡、
    94–100% util,余 ~16GB headroom,~4.3s/step(20000 步 ETA ≈ 24h)。
  - wandb offline run `9971x97w`(`runs/mem_stage2_v1/wandb/`),offline→jump 同步到
    wandb.ai/yichx14-uc-irvine/fastwam-mem。
  - ⚠️ **loss 起点 ~2.1 是正常冷启动值**(对得上 stage1-v3 的 2.18);smoke 的 0.12/0.16 是
    小 batch 单批只抽到低噪声 timestep 的假象,bs24 平均后才是真实起点。
- **下一步:** 每 1000 步存档,逐档跑 H5 LIBERO-plus(INCLUDE_NOISE=1),对照 mem-off 49.83。

---

## 5. 成绩

> **训练尚未启动,待填。** 评测口径:LIBERO-plus INCLUDE_NOISE=1(10030),对照 mem-off 49.83。

| step | **Total** | Camera | Robot | Lang. | Light | BG | Layout |
|---|---|---|---|---|---|---|---|
| TBD | | | | | | | |

---


## 6. v2 方案锁定:DiT-side MEM-style current-fold adapter

> 结论:stage2-v1 的 prepend 路线不继续扩大训练。v2 改成 MEM-style factorized temporal
> attention:history 只用于增强 current tokens,随后立即丢弃 history tokens,后端 MoT/action
> 看到的 token 数、current index、attention mask 都保持 base 形态。

### 6.1 为什么不是继续 prepend

stage2-v1 的真实问题不是单纯超参或早停,而是 history prepend 改变了后端接口:

- 原版 FastWAM action 只读 current-frame video cache。
- v1 把 video 序列改成 `[history_0..K-1,current,future...]`,current 的 temporal index 从 0 变成 K。
- action 仍只读 current,但这个 current 已经经过 history-prepend 路径重塑。
- 全 DiT finetune 后策略会依赖该路径;一旦 mem-off/current 回到 index 0,就是 OOD。

因此 v2 不让 history token 暴露给 MoT/action,避免把 memory 变成一条新 policy backbone。

### 6.2 Adapter 是什么

这里的 adapter 不是大模块,而是一个轻量 temporal branch,对应 `docs/MEM-stage2-Idea.md`
里先前写的正式版 "MEM-style temporal attention":

```text
X in R^{B x (K+1) x N x D}
```

其中 `K` 是 history latent 帧数,`+1` 是 current,`N` 是每帧 spatial tokens 数,`D`
是 video token hidden dim。对同一个 spatial position `p` 跨时间做 temporal attention:

```text
X[:, :, p, :] -> TemporalAttn -> delta_current[:, p, :]
current_tokens <- current_tokens + tanh(gate) * delta_current
```

复杂度:

```text
O(N K^2 D)
```

而不是 full spatio-temporal attention 的:

```text
O(K^2 N^2 D)
```

最后只保留 current tokens:

```text
[history,current] -> current_mem
```

所以它不是把 history 压成一个全局 token,而是把时间信息折叠进 **current 的每个
spatial token**。后端仍接收一帧 current token grid,token 数与 base 完全一致。

### 6.3 数据流

```text
history frames -- frozen Wan VAE plain encode --> history_latents
current frame  -- frozen Wan VAE plain encode --> current_latent

[history_latents,current_latent]
  -> video_expert patch/token embedding
  -> TemporalFoldAdapter
  -> current_mem_tokens only
  -> original MoT video prefill
  -> action expert denoise
```

训练时如果有 future video loss,MoT 看到:

```text
[current_mem, future_1, future_2, ...]
```

推理时 MoT 看到:

```text
[current_mem]
```

关键不变量:

- `gate=0` 初始化,step0 严格退化为 base。
- current 仍是 temporal index 0。
- action expert 的 video conditioning token 数不变。
- 不需要 v1 那种 action 指向 `num_history_frames * tokens_per_frame` 的特殊 mask。

### 6.4 与 stage1-v1 / MEM 原文的关系

stage1-v1 的哲学是对的:history 在前端融合,最后只把 current latent 给 DiT。它保住了
rollout:标准 LIBERO 仍 95.9,LIBERO-plus 只小幅掉到 45.64,没有全解冻崩盘。

stage2-v2 继承这个原则,但把融合位置从 VAE latent 压缩器移到 DiT token 空间:

- 比 stage1 更靠近 action conditioning,理论上更容易影响动作。
- 比 stage2-v1 更像 MEM,因为后端只见 current tokens,history 不是 prepend 后端输入。
- 长程 language memory 暂不并入 FastWAM,交给 robo agent / planner;FastWAM 只负责短程视觉 memory。

### 6.5 训练参数 v2 初版

| 项 | 值 |
|---|---|
| 分支 | `MEM-stage2` |
| 新开关 | `model.vae_memory.enabled=false` + `model.vae_memory.dit_prepend=false` + `model.vae_memory.dit_fold_current=true` |
| 互斥 | `enabled` / `dit_prepend` / `dit_fold_current` 三选一 |
| 起点 ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt` |
| VAE | 冻结,plain encode history/current |
| 可训练 | **仅 TemporalFoldAdapter + gate + 可选 adapter norm/proj** |
| 冻结 | action expert、MoT 主体、video expert 原参数、VAE、text encoder |
| history | H5 默认:LIBERO 下 1.0s,plain VAE 单独 encode 满足 4n+1,K_lat=2 |
| lr | `1e-4` 起步(adapter-only);若不稳降到 `3e-5` |
| batch | 8 卡 x BS=32 优先;adapter-only 显存应低于 v1 全 DiT,若 OOM 回退 BS=24 |
| max_steps | 10000 初版;save_every 1000 |
| warmup | 5% cosine |
| loss | action/video 原 loss 不变;可选加 base consistency loss |
| output_dir | `runs/mem_stage2_v2_fold_current` |

### 6.6 History dropout / anti-dependence

目的:避免重演 v1 的 "训练永远有 history -> 策略依赖 history -> mem-off 崩"。训练时对
一部分 batch 故意破坏 history,迫使模型学会 history 有用时用、无用时退回 current。

初版比例建议保守:

| 比例 | 操作 | 作用 |
|---|---|---|
| 70% | 正常 history | 学会使用真实短程视觉记忆 |
| 20% | repeat current | 模拟无有效 history,但视觉分布自然 |
| 10% | shuffle across batch / temporal offset | 模拟错误 history,训练不要盲信 |

实现位置:在 `fastwam.py` 里拿到 `history_video` 后、VAE encode 前做,不要先改 dataset。
`repeat current` 比 zero 黑帧更稳,因为 zero 太 OOD。

伪代码:

```python
if self.training and self.history_dropout_enabled:
    mode = torch.rand(B, device=history_video.device)
    repeat = (mode >= 0.70) & (mode < 0.90)
    shuffle = mode >= 0.90

    history_video[repeat] = input_video[repeat, :, 0:1].repeat(1, 1, K, 1, 1)

    perm = torch.randperm(B, device=history_video.device)
    history_video[shuffle] = history_video[perm][shuffle]
```

### 6.7 验收

先验收“不退化”,再看 memory 增益:

1. `gate=0` smoke:同 seed 下 v2 fold-current 与 base 输出应近似一致。
2. 标准 LIBERO:接近 base 95.9,不能像 v1 prepend 全解冻那样崩。
3. LIBERO-plus:只做鲁棒性门禁,目标不明显低于 mem-off 50.5;永远 `INCLUDE_NOISE=1`。
4. memory benchmark:base vs stage1-v1 vs stage2-v2 三方对比,这里才判定 memory 是否有效。
5. 真机叠衣服:作为最终外部验证,需先定义 fold success / phase accuracy / history-sensitive action accuracy。

---

## 7. 实验目录速查

| 内容 | 路径 |
|---|---|
| 思路文档 | `docs/MEM-stage2-Idea.md`(末尾含实施说明) |
| smoke 脚本 | `scripts/train_mem_stage2_v1_smoke.sh` |
| 训练脚本 | `scripts/train_mem_stage2_v1.sh`(`train_mem_stage1_v2.sh` 是旧 patch_embed 路线,勿混) |
| run 目录(待建) | `runs/mem_stage2_v1/` |
| 评测脚本 | `scripts/eval_mem_stage2_v1.sh`(stage2 包装:锁 VAE_MEM=false+DIT_PREPEND=true+H5+INCLUDE_NOISE=1,自动选最新 ckpt;底层调通用 `scripts/eval_libero_plus.sh`),聚合 `experiments/libero/summarize_libero_plus.py` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
