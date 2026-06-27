# MEM-VAE 训练诊断报告

日期：2026-06-21  
本地仓库：`/Users/maxliu/MyProjects/AIR/202606/FastWAM_xyc`  
远端验证机器：`maxliu-h200-qinghua-1`  
远端实际工作区：`/data/home/frank/projects/FastWAM`

配套文件：

- PNG 图册说明：[`26-06-21-mem-vae-figure-guide.md`](26-06-21-mem-vae-figure-guide.md)
- 可视化仪表盘：[`26-06-21-mem-vae-diagnostic-dashboard.html`](26-06-21-mem-vae-diagnostic-dashboard.html)
- 原始数据表：[`26-06-21-mem-vae-diagnostic-raw-data.csv`](26-06-21-mem-vae-diagnostic-raw-data.csv)

## 0. 结论先说

你的判断是对的：**当前结果大幅变差的拐点确实出现在解冻 DiT 的 stage3，而不是 stage1 的 memory-only 阶段。** 我在远端实际 ckpt 和评测结果上验证到了这一点。

但我之前提到的 “memory path 不是 no-op” 也不是凭感觉推断。轻量 VAE latent 实验显示：即使把新增 temporal attention 的 `temporal_gate` 强制设为 0，只要输入从 `current` 变成 `[history + current]`，返回的 current conditioning latent 就已经明显偏离普通 `vae.encode(current)`。K=0 控制组完全相等，K>0 后差异立刻出现。

所以更准确的解释不是单因子，而是两段式：

1. **memory conditioning latent 本身存在分布偏移。** 这个偏移来自 `encode_memory(history+current)` 的整条 VAE encode path，不只来自新增 temporal attention branch。stage1/stage2 主要是在小范围补偿这个偏移，所以 full eval 只小幅变差。
2. **stage3 一旦全 DiT 解冻，模型主体在前 500 step 内就漂得很远。** 真实 ckpt 对比显示，stage3 step500 时 video blocks rel_l2 已经约 0.274，action head 约 1.86，proprio encoder 约 1.43；step500 到 step4000 反而只剩很小变化。这和 stage3 PILOT 从 2% 到最高 17.5% 的崩盘一致。

此外还有一个重要实验设计问题：**stage3 不是从 stage2 继续训练，而是直接从 release checkpoint 启动 full-DiT finetune。** 远端日志明确写着 stage3 加载的是 `checkpoints/fastwam_release/libero_uncond_2cam224.pt`，该 ckpt 没有 `vae_memory`，temporal params 只是构造模型时 warm-start。stage2 则是从 stage1 `step_021700.pt` 继续的。

## 1. 我做了哪些验证

### 1.1 代码路径验证

关键代码：

- `src/fastwam/models/wan22/fastwam.py:340`：把 `history_video` 和 `current_first_frame` 拼成 `[history + current]`。
- `src/fastwam/models/wan22/fastwam.py:342`：调用 `self.vae.encode_memory(...)`。
- `src/fastwam/models/wan22/fastwam.py:415`：target video latent 仍然走普通 `vae.encode(video)`。
- `src/fastwam/models/wan22/fastwam.py:421` 到 `:432`：只有 conditioning first-frame latent 被 memory latent 替换。
- `src/fastwam/models/wan22/wan_video_vae.py:454` 到 `:459`：`temporal_gate` 只 gate 住新增 temporal attention branch。
- `src/fastwam/models/wan22/wan_video_vae.py:1512` 到 `:1567`：`encode_memory` 会对完整 `[history + current]` timeline 做 chunked compression，再把 full latent timeline 送入 middle/head。
- `src/fastwam/trainer.py:327` 到 `:360`：stage1/stage2/stage3 的 trainable 参数集合由 `train_temporal_only` 和 `unfreeze_patch_embed` 控制。

这说明：`temporal_gate=0` 只能保证新增 temporal branch 不贡献 residual，**不能保证 `encode_memory(history+current)` 等价于 `encode(current)`**。

### 1.2 远端评测结果验证

所有结果来自远端 `evaluate_results/libero_plus/**/summary.csv`。

重点数据：

| 实验 | 评测口径 | Total success |
|---|---:|---:|
| MEMOFF release | Full 10030 cases | 49.83% |
| stage1 memory-only | Full 10030 cases | 45.64% |
| stage2 temporal + patch embed | Full subset 8429 cases | 46.43% |
| stage3 step1000 | PILOT 200 cases | 2.00% |
| stage3 step2000 | PILOT 200 cases | 12.50% |
| stage3 step3000 | PILOT 200 cases | 17.50% |
| stage3 step4000 | PILOT 200 cases | 17.00% |

解释：

- stage1/stage2 是小幅下降，和你的观察一致。
- stage3 的 200-case PILOT 不是 full eval，不能和 10030-case full eval 精确横比；但 2% 到 17.5% 已经足以说明 full-DiT 阶段发生了严重退化。
- stage2 full eval 少了 Noise 维度，总 cases 是 8429，不是 10030；文档和 CSV 中保留了这个差异。

### 1.3 远端训练配置和日志验证

stage1：

- 配置：`runs/mem_temporal_libero/config.yaml`
- `learning_rate=1e-4`
- `vae_memory.enabled=true`
- `warm_start=true`
- `train_temporal_only=true`
- 只训练 4 个 VAE temporal params。

stage2：

- 配置：`runs/mem_temporal_libero_stage2/config.yaml`
- 日志：`runs/mem_temporal_libero_stage2.log`
- `learning_rate=3e-5`
- `train_temporal_only=true`
- `unfreeze_patch_embed=true`
- 日志行 173-179 显示加载：`runs/mem_temporal_libero/checkpoints/weights/step_021700.pt`
- 日志行 188-190 显示只解冻 4 个 temporal params + 2 个 patch embedding tensors。

stage3：

- 配置：`runs/mem_temporal_libero_stage3/config.yaml`
- 日志：`runs/mem_temporal_libero_stage3/wandb/offline-run-20260620_133813-at0tme9z/files/output.log`
- `learning_rate=1e-5`
- `train_temporal_only=false`
- `unfreeze_patch_embed=false`
- 日志行 8-14 显示加载的是 `checkpoints/fastwam_release/libero_uncond_2cam224.pt`，并警告该 checkpoint 没有 `vae_memory`。
- 日志行 22-24 显示：`unfroze 4 temporal params + DiT (+ proprio)`。

这点很关键：**stage3 不是 stage2 的 continuation，而是从 release 直接打开 full-DiT 的独立实验。**

### 1.4 ckpt 权重漂移验证

我把实际 ckpt 的 `mot`、`proprio_encoder`、`vae_memory` 和 release ckpt 做了 CPU 上的 L2 范数对比。指标是：

`rel_l2 = ||theta_ckpt - theta_release||_2 / ||theta_release||_2`

结果：

| ckpt | video patch | video blocks | action head | proprio |
|---|---:|---:|---:|---:|
| stage1 step21700 | 0.000 | 0.000 | 0.000 | 0.000 |
| stage2 step3000 | 0.259 | 0.000 | 0.000 | 0.000 |
| stage2 step6000 | 0.271 | 0.000 | 0.000 | 0.000 |
| stage3 step500 | 0.160 | 0.274 | 1.862 | 1.426 |
| stage3 step1000 | 0.159 | 0.274 | 1.861 | 1.426 |
| stage3 step4000 | 0.159 | 0.274 | 1.859 | 1.426 |

额外对比 stage3 step500 -> step4000：

| 区间 | video patch | video blocks | action head | proprio |
|---|---:|---:|---:|---:|
| step500 -> step4000 | 0.0086 | 0.0137 | 0.0079 | 0.0013 |

解释：

- stage1 只改 `vae_memory`，主 DiT 和 proprio 完全没动。
- stage2 只改 `vae_memory` + `video.patch_embedding`，video blocks/action/proprio 没动。
- stage3 在前 500 step 内已经把 DiT/action/proprio 改到和 release 很远的位置。
- stage500 到 stage4000 的继续漂移很小，说明崩坏不是“慢慢训练坏”，而是 full-DiT 打开后早期就进入了很偏的 basin。

### 1.5 VAE latent no-op 验证

实验目标：

验证 `temporal_gate=0` 时，`encode_memory([history + current])` 是否等价于 `encode(current)`。

实验设置：

- 远端 GPU：H200 单卡，`CUDA_VISIBLE_DEVICES=0`
- VAE 权重：`checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`
- synthetic test：随机 current frame + 不同 history 构造
- real-data test：真实训练样本 `idx=1234`
- 对比指标：
  - `rel_l2 = ||z_mem - z_single|| / ||z_single||`
  - `cosine(z_mem, z_single)`
  - `mean_abs`
  - `max_abs`

synthetic 控制组结果：

| 条件 | rel_l2 | cosine |
|---|---:|---:|
| K=0, gate=0 | 0.000 | 1.000 |
| K=4 zero history, gate=0 | 1.379 | 0.127 |
| K=4 repeat history, gate=0 | 0.922 | 0.690 |
| K=4 random history, gate=0 | 0.558 | 0.859 |
| K=16 random history, gate=0 | 0.598 | 0.836 |

真实训练样本结果：

- sample idx：1234
- dataset length：277713
- `video_shape=[3,9,224,448]`
- `history_shape=[3,4,224,448]`

| VAE state | gate | rel_l2 | cosine | mean_abs | max_abs |
|---|---:|---:|---:|---:|---:|
| fresh gate=0 | 0.0000 | 1.306 | 0.572 | 0.455 | 4.086 |
| stage1 step21700 | 0.8320 | 0.666 | 0.752 | 0.238 | 1.938 |
| stage2 step6000 | 0.0801 | 1.304 | 0.579 | 0.444 | 4.273 |
| stage3 step4000 | 0.0017 | 1.306 | 0.572 | 0.455 | 4.086 |

解释：

- K=0 完全相等，证明测试代码没有人为制造差异。
- K>0 且 gate=0 时差异很大，证明 memory path 本身不是 no-op。
- stage1 学到的 `vae_memory` 确实把真实样本的 latent 差异从 1.306 降到 0.666，说明 stage1 不是完全无效。
- stage2/stage3 的 VAE memory latent 又接近 fresh gate=0，stage2 的表现更多可能来自 patch embedding 适配，而不是 VAE memory latent 本身变得更像原始 latent。

## 2. 我现在的判断

### 2.1 你质疑得对：大崩和 DiT 解冻强相关

评测结果和 ckpt 漂移都支持这一点。stage1/stage2 虽有下降，但幅度小；stage3 full-DiT 打开后立刻崩。尤其是 stage3 step500 权重已经大幅偏离 release，后续 3500 step 变化反而小。

### 2.2 我的原始担心也成立：memory path 有结构性分布偏移

这个偏移不等于“最终效果一定崩”，但它解释了为什么：

- stage1 memory-only 会小幅变差；
- stage1 能学出一些补偿；
- patch embedding 适配有帮助；
- full DiT 解冻后，模型可能沿着这个 shifted conditioning latent 过快改写原策略。

### 2.3 stage3 当前设计混入了两个变量

stage3 同时做了：

1. 从 release 直接启动，而不是从 stage2 启动；
2. 打开全 DiT + proprio；
3. 使用 memory conditioning latent；
4. 没有约束 DiT 保持 release 行为。

因此不能只说“解冻 DiT 坏”，更精确地说是：

**当前 stage3 的 full-DiT 解冻方式，在 memory conditioning latent 分布偏移存在的情况下，从 release 直接训练，导致 action/proprio/video 主体很早大幅漂移，最终表现严重退化。**

## 3. 建议下一步实验

优先级从高到低：

1. **stage3 从 stage2 checkpoint 继续，而不是从 release 开始。**  
   这是最直接验证 stage3 实验设计问题的 ablation。

2. **full DiT 前先做 true eval。**  
   确保 eval path 传入真实 history；训练里的普通 `evaluate()` 目前容易走无 history 的 `infer()` 路径，不能代表 memory 模型。

3. **full-DiT 解冻改成更窄范围。**  
   先只解冻 `video.patch_embedding + early video blocks` 或只解冻 video side，暂时冻结 action head、action blocks、proprio encoder。

4. **给 memory conditioning latent 加 identity/consistency regularization。**  
   例如在 early phase 约束 `z_mem` 不要离 `z_single` 太远，或者把 memory 设计成 `z_single + gated_residual(history)`，让 gate=0 时数学上严格 no-op。

5. **stage3 使用分组学习率。**  
   temporal params 和 patch embedding 可以较大学习率；DiT/action/proprio 使用更小 lr，尤其 action head/proprio 需要单独保护。

## 4. 原始数据位置

本报告使用的原始文件：

- 远端评测 summary：`/data/home/frank/projects/FastWAM/evaluate_results/libero_plus/**/summary.csv`
- stage1 ckpt：`/data/home/frank/projects/FastWAM/runs/mem_temporal_libero/checkpoints/weights/step_021700.pt`
- stage2 ckpt：`/data/home/frank/projects/FastWAM/runs/mem_temporal_libero_stage2/checkpoints/weights/step_003000.pt`、`step_006000.pt`
- stage3 ckpt：`/data/home/frank/projects/FastWAM/runs/mem_temporal_libero_stage3/checkpoints/weights/step_000500.pt`、`step_001000.pt`、`step_004000.pt`
- stage2 日志：`/data/home/frank/projects/FastWAM/runs/mem_temporal_libero_stage2.log`
- stage3 日志：`/data/home/frank/projects/FastWAM/runs/mem_temporal_libero_stage3/wandb/offline-run-20260620_133813-at0tme9z/files/output.log`
- 本地整理后的 CSV：`docs/26-06-21/26-06-21-mem-vae-diagnostic-raw-data.csv`

## 5. 复现实验命令摘要

只读读取 summary：

```bash
ssh maxliu-h200-qinghua-1 'cd /data/home/frank/projects/FastWAM && /data/home/frank/.conda/envs/fastwam/bin/python -c "from pathlib import Path
root=Path(\"evaluate_results/libero_plus\")
for p in sorted(root.rglob(\"summary.csv\")):
    print(\"=====\", p.parent.relative_to(root))
    print(p.read_text(errors=\"ignore\").strip())"'
```

ckpt 权重漂移核心逻辑：

```python
rel_l2 = (theta_ckpt.float() - theta_release.float()).norm() / theta_release.float().norm()
```

VAE latent no-op 核心逻辑：

```python
z_single = vae.encode([current_frame], device="cuda:0")
z_mem = vae.encode_memory([torch.cat([history, current_frame], dim=1)], num_current_frames=1, device="cuda:0")
rel_l2 = (z_mem.float() - z_single.float()).norm() / z_single.float().norm()
```

真实样本测试额外设置：

```python
from fastwam.utils import misc
misc.register_work_dir("/tmp/fastwam_mem_diag")
cfg = OmegaConf.load("runs/mem_temporal_libero_stage3/config.yaml")
ds = instantiate(cfg.data.train)
sample = ds[1234]
```

## 6. 一句话版本

**stage3 崩盘不是单纯因为 “MEM 有历史” 或 “DiT 解冻” 中任意一个因素，而是 memory conditioning latent 已经偏离原始 first-frame latent，stage1/stage2 还能局部补偿；stage3 从 release 直接全量解冻 DiT 后，action/proprio/video 主体在前 500 step 内大幅漂移，导致策略行为严重退化。**
