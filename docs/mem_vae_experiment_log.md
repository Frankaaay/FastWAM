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


---

## 5. Stage-3 — 从 base 冷启动,全 DiT + temporal 联合 co-adapt(H4)

> 思路转向:**不再在 stage-1/2 的歪地基上打补丁**。从 base 原版 ckpt 冷启动,
> 整个 DiT 自由适应带记忆的 latent,从一开始就 co-adapt。领导方向:历史砍到 H4。

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_mem_stage3.sh`(`HISTORY` 参数化,默认 4) |
| run 目录 | `runs/mem_temporal_libero_stage3/` |
| 起点 ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt`(base 原版,mem-off 49.83) |
| resume | weights-only 冷启动(不继承 stage-1/2);有 DeepSpeed state 则续(须保持 8 卡) |
| warm_start | **true**(temporal 4 参数从 spatial proj 初始化) |
| 可训练 | **全 DiT(video+action expert ~5.9B)+ proprio + temporal**(`train_temporal_only=false`) |
| history | **4** → 0.8s(ratio4/fps20);参数保留,消融可 8/16 |
| 卡 / batch | **8 卡 × bs32 = global 256** |
| lr | **1e-5**,cosine + 5% warmup(全解冻 blast radius 大,比 stage-2 的 3e-5 更保守) |
| max_steps | **4000**(~3.7 epoch);**故意留长,靠 PILOT 看到掉头就 kill,不跑满** |
| save_every | **500**(8 个存档点,密集抓峰值) |
| wandb | offline→jump→cloud(entity yichx14-uc-irvine / project fastwam-mem) |

**为什么全解冻(vs AdaLN/BitFit):** 仓库只有三档(temporal-only / +patch_embed / 全解冻),
无 LoRA/AdaLN 中间档。病根是「DiT 读不懂记忆 latent」属接口/表征错配,AdaLN 那点
缩放未必修得动,全解冻自由度才够真正 co-adapt。**全解冻不贵**:stage-2 的 backward
本就穿过全网到输入层 patch_embedding,FLOPs ≈ 全解冻;只多优化器状态(~6G/卡)+ 梯度
缓冲(~12G/卡),峰值 ~85-90G/140G,单步 +10-30%,叠加 H4 整体仍 ~6h。OOM 回退 bs24。

**挑 ckpt 的依据 = eval,不是 loss:** stage-1 loss 在 ~step4000 就压平到 ~0.10(到
21700 全平,见 `docs/mem_loss.png`);stage-2 也是 loss 缓降但任务峰值在 step3000、
step6000 过训。所以 stage-3 不看 loss,跑完(或中途)对各存档点 PILOT 快筛挑峰值,
等效任务峰值预计落在 ~1500 步(global256)。再对最优点全量 eval(`INCLUDE_NOISE=1`,10030)。

**成绩:** _(待跑)_

---

## 5. 实验目录速查

| 内容 | 路径 |
|---|---|
| stage-1 脚本 / run | `scripts/train_mem_temporal.sh` / `runs/mem_temporal_libero/` |
| stage-2 脚本 / run | `scripts/train_mem_stage2.sh` / `runs/mem_temporal_libero_stage2/` |
| stage-3 脚本 / run | `scripts/train_mem_stage3.sh` / `runs/mem_temporal_libero_stage3/` |
| 标准 LIBERO eval | `evaluate_results/libero/` |
| LIBERO-plus eval | `evaluate_results/libero_plus/stage2_step*_FULL/` |
| 评测脚本 | `scripts/eval_libero_plus.sh`,`experiments/libero/summarize_libero_plus.py` |
| loss 曲线图 | `docs/mem_loss.png`,`docs/stage2_loss.png` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
