# MEM-stage2 实验记录(experiment log)

> 分支 `MEM-stage2`(从 `feat/mem-vae` 分出)。**重新设计**短期记忆的融合方式:
> **彻底去掉 VAE 侧 temporal attention**,改为在 **MoT 的 video prefill 路径里 prepend
> 历史 latent 帧**(option C / 文档 6.1,v1 不 crop)。
> 目标不变:在 LIBERO-plus 上 **超过无记忆基线 49.83**。
> 配套思路见 [MEM-stage2-Idea.md](MEM-stage2-Idea.md);stage-1/2/3 的旧记录见
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
   memory K/V。这直接修掉了 stage-3 的 train/infer 裂缝(exposure bias)。
2. **不碰 conditioning latent**:current 帧 conditioning = base 原版的 plain encode,**逐位
   一致**;历史只是额外可 attend 的上下文。→ 避开 stage-1/2 的"DiT 读不懂记忆 latent"。
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
| history | **H4** → 0.8s(ratio4/fps20);K_lat=1 个历史 latent 帧 |
| 卡 / batch | **8 卡 × bs32 = global 256**(节点被占时 **bs24** 回退) |
| lr | **1e-5**,cosine + 5% warmup |
| loss 权重 | lambda_video=1.0 / lambda_action=1.0(option C 默认贴原版) |
| max_steps | **4000**;save_every **500** |
| wandb | offline → jump 同步(entity yichx14-uc-irvine,project fastwam-mem) |
| output_dir | `runs/mem_stage2` |

---

## 4. 进展 / 状态

- **2026-06-23 — 8 卡 1-step smoke 通过(干净 PASS)。** 见 `scripts/train_mem_stage2_smoke.sh`。
  - base ckpt **strict 加载**(无 missing/unexpected key)✓
  - 全 DiT 进入训练、**多卡 ZeRO-1 分片不 OOM**(每卡 MA 15.33 GB @bs4)✓
  - **optimizer step 干净跑完**、checkpoint(weights + state)保存链路通 ✓
  - `step=1/1 loss=0.1178 loss_action=0.0262 loss_video=0.0916 lr=1e-7`
  - **初始 loss 0.118 ≪ stage-3 冷启动的 2.18**:因为没新增 temporal 参数、current
    conditioning 与 base 逐位一致,base 一上来就在"熟悉的输入 + 额外历史"上工作,起点健康
    (不像 stage-1/2/3 把 conditioning 改写/扰动了)。**印证"贴原版、不扰动"方向正确。**
  - ⚠️ 该 smoke 用 bs4/卡(非真实 bs32)以与他人 job 共存;真实 bs32 峰值显存待训练时确认。
- **下一步:** 用户批准 §3 参数 → 写 `scripts/train_mem_stage2.sh` → commit → 同步 → 启动
  4000-step;每 500 步存档,逐档跑 H4 LIBERO-plus(INCLUDE_NOISE=1)。

---

## 5. 成绩

> **训练尚未启动,待填。** 评测口径:LIBERO-plus INCLUDE_NOISE=1(10030),对照 mem-off 49.83。

| step | **Total** | Camera | Robot | Lang. | Light | BG | Layout |
|---|---|---|---|---|---|---|---|
| TBD | | | | | | | |

---

## 6. 实验目录速查

| 内容 | 路径 |
|---|---|
| 思路文档 | `docs/MEM-stage2-Idea.md`(末尾含实施说明) |
| smoke 脚本 | `scripts/train_mem_stage2_smoke.sh` |
| 训练脚本(待建) | `scripts/train_mem_stage2.sh` |
| run 目录(待建) | `runs/mem_stage2/` |
| 评测脚本 | `scripts/eval_libero_plus.sh`,`experiments/libero/summarize_libero_plus.py` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
