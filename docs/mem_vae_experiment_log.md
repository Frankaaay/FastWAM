# Mem-VAE 实验记录(experiment log)

> 分支 `feat/mem-vae`。给 FastWAM 加 **VAE 短期记忆**(temporal memory),
> 目标:在 LIBERO-plus 鲁棒性测试上 **超过无记忆基线**,证明「短期记忆提升鲁棒性」。
> 配套计划见 [mem_vae_eval_and_finetune_plan.md](mem_vae_eval_and_finetune_plan.md)。
> 本文件只记「跑了什么 / 得了多少分 / 下一步」,持续更新。

---

## 0. 评测协议与基线

- **benchmark**:LIBERO-plus(arXiv 2510.13626),8429 个非 noise case,7 因子
  (Camera / Robot / Lang. / Light / BG / Noise / Layout),难度 L1–L5。
- **每 case rollout 1 次**(num_trials=1),Total = 全部 case 整体成功率。
- 评测脚本:`scripts/eval_libero_plus.sh` + `experiments/libero/summarize_libero_plus.py`。

| 基线 | LIBERO-plus Total | 说明 |
|---|---|---|
| Fast-WAM(paper 原版) | 51.5 | 论文 Table 4 报告值 |
| **mem-off(本仓库复现,无记忆)** | **49.83** | 我们的目标:要超过它 |
| mem-on stage-1 | 45.64 | 见 §1,**记忆净负 ~4.2** |

> 现状一句话:**加记忆反而比不加低**。所有微调都是为了把这 4 分赚回来并反超 49.83。

---

## 1. Stage-1 — 只训 VAE temporal 4 参数(DiT 全冻)

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_mem_temporal.sh` |
| run 目录 | `runs/mem_temporal_libero/` |
| ckpt | `runs/mem_temporal_libero/checkpoints/weights/step_021700.pt` |
| 可训练参数 | **4 个**(VAE memory temporal attn / pos / proj) |
| DiT / MoT / base VAE | 全冻结 |
| 训练量 | 10 epoch,batch 16,step 0→21700(已收敛,loss ~step4000 压平) |
| history | `history_video_frames=16` → **3.2s**(16×4÷20) |

**成绩:**
- 标准 LIBERO(无扰动)Avg **95.9**(原版 97.6,-1.7;Long 掉最多 -3.0)。
- LIBERO-plus Total **45.64**(< mem-off 49.83)。

**诊断:** 冻死的 DiT 输入层 `patch_embedding` 是在「无记忆 latent」上训出来的,
读不懂被记忆改写过的 latent → 接口错配、记忆净负(不是记忆本身没价值,
而是 DiT 没机会适应)。Long suite 掉最多正是佐证。

---

## 2. Stage-2 — 额外解冻 DiT 输入接口 patch_embedding(2 参数)

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_mem_stage2.sh` |
| run 目录 | `runs/mem_temporal_libero_stage2/` |
| ckpt | `checkpoints/step_1000.pt` … `step_6000.pt`(每 1000 存一次) |
| resume | weights-only,从 stage-1 `step_021700.pt`(optimizer/step 重置) |
| 可训练参数 | **6 个** = temporal 4 + `video_expert.patch_embedding` 2 |
| 超参 | 4 卡 × bs32 = global 128,lr 3e-5,max_steps 6000,cosine+5% warmup |
| history | `history_video_frames=16` → 3.2s(与 stage-1 一致,可比) |
| wandb | offline→jump→cloud(entity yichx14-uc-irvine / project fastwam-mem) |

**成绩(step_3000,全量 96.8% 8157/8429,准最终):**

| Camera | Robot | Lang. | Light | BG | Layout | **Total** |
|---|---|---|---|---|---|---|
| 8.1 | 38.8 | 64.6 | 79.1 | 38.7 | 58.0 | **46.9** |

- vs mem-on stage-1 45.64 → **+1.3 ✅**(接口对齐方向被证明有效)
- vs mem-off 49.83 → **-2.9 ❌**(还没反超无记忆基线)
- 短板:**Camera 8.1**(视角扰动,和 paper 里所有模型一样崩)。

> step_4000/5000/6000 尚未全量评完;step_3000 不一定是最优点(训练 loss 当时还在降)。

---

## 3. 关键知识点:历史窗口换算

**历史秒数 = history_video_frames × action_video_freq_ratio ÷ fps**

- LIBERO:`action_video_freq_ratio=4`([configs/data/libero_2cam.yaml](../configs/data/libero_2cam.yaml)),`fps=20`。
- 约束:`history_video_frames` 必须是 **4 的倍数**(VAE memory 要求 (K+1)%4==1)。
- **换数据集必须重算**(fps / ratio 都可能不同)。

| history_video_frames | LIBERO 跨秒数 |
|---|---|
| 4 | 0.8s |
| 8 | 1.6s |
| 12 | 2.4s |
| 16(stage-1/2 现状) | 3.2s |

> 领导意见:看过去 **1.2s** 就够。当前喂了 **3.2s**,大概率偏长(无关历史干扰
> 只微调了几张量的 DiT)。1.2s 卡在整 4 倍数之间,实操取 4(0.8s)或 8(1.6s);
> 要精确 1.2s 需把 history 抽帧步长与 action 解耦(stride=2 × 12 帧 = 24 原始帧)。

---

## 4. 未来尝试(按性价比排序)

| # | 思路 | 改动 | 成本 | 预期 |
|---|---|---|---|---|
| **1** | **缩短历史(领导方向)** | `history_video_frames=4`(0.8s)/ `8`(1.6s)重跑 stage-2 | 低 | 先用现有 ckpt eval-only PILOT 验证短历史在推理端是否更好,有信号再重训 |
| **2** | **挑最优 checkpoint** | step_2000/5000/6000 各跑 PILOT 快筛 | 低 | step_3000 未必最优,可能 >46.9 |
| **3** | **解冻更多 DiT** | 再解冻前 N 个 DiT block,或 DiT 上 LoRA(小 LR) | 中 | 接口对齐不够时,给网络更深适应空间;风险:遗忘/过拟合 |
| **4** | **记忆门控 / 历史增强** | learned gate 让模型自决信多少历史;训练时随机丢历史 | 中 | 针对 Camera 8.1(怀疑视角变化时历史在误导);降低对记忆过度依赖 |

**当前推荐顺序:1 + 2 先做(都便宜,且 1 直接对接领导)→ 不够再 3 → 仍不够再 4。**

---

## 5. 实验目录速查

| 内容 | 路径 |
|---|---|
| stage-1 脚本 / run | `scripts/train_mem_temporal.sh` / `runs/mem_temporal_libero/` |
| stage-2 脚本 / run | `scripts/train_mem_stage2.sh` / `runs/mem_temporal_libero_stage2/` |
| 标准 LIBERO eval | `evaluate_results/libero/` |
| LIBERO-plus eval | `evaluate_results/libero_plus/stage2_step*_FULL/` |
| 评测脚本 | `scripts/eval_libero_plus.sh`,`experiments/libero/summarize_libero_plus.py` |
| loss 曲线图 | `docs/mem_loss.png`,`docs/stage2_loss.png` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
