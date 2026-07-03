# 原版 FastWAM 推理/训练加速方法全集：实测、等价性验证与复现指南

日期：2026-07-03 ~ 07-04
实验分支：`experiment/gemm-align-original`（基于 `main` @45d8e14，已推 origin）
远端环境：h200-qinghua-1（8×H200 141G，torch 2.7.1+cu128，DeepSpeed 0.18.5 ZeRO-1，conda `fastwam`），实验 worktree `/data/home/maxliu/projects/FastWAM_orig`
任务口径：fold_cloth 真机数据（865308 样本），原版 FastWAM 训练图（无 v4 history-memory），训练 4 卡（GPU 0,5,6,7）、推理单卡

## 结论总览

**约束**：不增加空间占用（禁磁盘 latent cache）；完全等价性（不改变训练/推理数学）；ckpt 完全兼容（零参数改动）。

| 侧 | 最终成果 | 等价性 |
|---|---|---|
| **推理** | infer_action **544 → 110.3 ms（4.93×）** | **逐位相等**（max_abs=0） |
| **训练** | 默认口径 85.6 → 52.1 ms/sample（**1.64×**，bs40 吞吐）；同 bs16 1.40× | loss 逐 step 同分布；compile 舍入级（fp32 复测证明纯精度舍入） |

训练 2×（严格等价 + 不占空间下）被 GPU 计算下限卡住：优化后 step 的 GPU busy 已占墙钟 ~87%，剩余大头是 DiT GEMM / VAE conv / 梯度计算等真实计算，attention 仅占 11%。天花板约 1.6-1.7×；要 2× 必须减真实计算（fp8 / latent cache / 改模型），三者均违反约束。

---

## 方法一：MoT DiT 区域 torch.compile（训练+推理通用，✅采纳）

**开关**：`model.mot_torch_compile=true`（默认 false），`model.mot_torch_compile_mode`（默认 "default"）。commit `87217e9`（自 v4 分支 fadf199 适配移植）。

**原理**：训练是 launch-bound（~2.7 万 kernel/step，"other" 胶水 kernel 总时间超过 GEMM）。对 MoT 每个 DiT block 的两段计算体（attention I/O：norm+modulate+qkv+RoPE；post block：o-proj+gate+FFN）做 per-block 独立 `torch.compile`（dynamic=False），Inductor fusion 把胶水 kernel 减半。**不编译** attention 本体（保持 SDPA backend 选择）。与 gradient checkpointing 互斥（自动跳过并 warning）；编译失败自动回退 eager。

**实测**：训练 step 1800→1590ms（bs24，−12%）；推理 544→409ms（−25%）。数值等价性：Inductor 融合舍入级（与训练 loss 曲线重合印证）。

**注意**：首个 step 编译预热约 2-4 分钟（60 编译单元），此后稳定。

## 方法二：VAE encode 在线 compile（训练用，✅采纳）

**开关**：`model.vae_torch_compile=true`（默认 false）。commits `35fd6e6` + `d33793d`。

**原理**：VAE encode 是训练 forward 第一大单点（~3 万 kernel/step、墙钟 849ms 中 26% 在等 kernel launch）。对 `vae.model.encoder.forward` 做 `torch.compile`。

**关键坑（已修复）**：不能 `torch.compile(encoder)` 整体替换 module——compile 返回的 OptimizedModule 会让 accelerate `unwrap_model` 抛 `KeyError: '_orig_mod'`。必须**编译 forward 方法**（`encoder.forward = torch.compile(encoder.forward)`），保持模型树里仍是普通 nn.Module。

**实测**：训练 forward 1100→850ms，step 1800→1540ms（1.17×）。

## 方法三：VAE encode 功能化重构 + CUDA graphs（训练用，✅采纳，略优于方法二）

**开关**：`model.vae_encode_functional=true`（默认 false），mode 默认 "reduce-overhead"。commits `c32819b`+`aab5a02`+`6a9048e`。新文件 `src/fastwam/models/wan22/vae_encode_functional.py`。

**原理**：直接对 VAE encoder 上 reduce-overhead（CUDA graphs）会崩：`feat_cache` 是跨时间 chunk 原地变异的 Python list，存的是 graph 输出 tensor，被下次 replay 覆写。重构依据的关键洞察：**encode 的所有"动态"分支按 chunk 位置静态可判**（chunk0=1 帧走初始化分支；chunk1..8=4 帧同 shape 走缓存分支），与数值无关（逐一核查了 CausalConv3d/Resample/ResidualBlock/Encoder3d 等全部 feat_cache 分支）。因此拆成 first/next 两个纯函数，cache 显式传入传出，chunk 输出与 cache tensor **图外 clone**（第二个坑：chunk 输出 `out_i` 也是图输出、也必须 clone）。

**等价性（三级验证，`scripts/verify_vae_encode_equivalence.py`）**：
| 级别 | bf16 max_abs | fp32 max_abs | 判定 |
|---|---:|---:|---|
| A：功能化 eager vs 原版 | **0** | **0** | **逐位相等**（重构零误差） |
| B：compile default | 0.039（rel 1.8%） | 0.0017（rel 0.08%） | 偏差随精度缩小 22×→纯舍入 |
| C：reduce-overhead | 与 B 逐位相同 | 与 B 逐位相同 | CUDA graphs 零额外误差 |

**实测**：单独 1490ms（1.21×）；与方法一组合 **1330ms（1.35×，bs24 同 batch 最优）**。

## 方法四：denoise 循环整步 CUDA graph 持久化（推理用，✅核心成果）

**开关**：`model.infer_denoise_cuda_graph=true`（默认 false）。commits `6b68ee7`+`c763a65`+`7a2ff4c`+`2625e18`。新文件 `src/fastwam/models/wan22/denoise_cuda_graph.py`。

**原理**：推理 90% 时间在 denoise 循环（20 步×45ms），每步 GPU busy 仅 ~8ms——纯 launch-bound。denoise 单步（`_predict_action_noise_with_cache`）输入 shape 固定（latents [1,A,D] + timestep [1]），video_kv_cache/context/attention_mask 在循环内不变——是 CUDA graph 的理想形态。用 `torch.cuda.CUDAGraph` 整步捕获：静态输入 buffer、replay、输出图外 clone。**跨调用持久化**：条件张量（kv k/v、context、mask）也用静态 buffer，后续 infer_action 调用只 `copy_` 更新 + replay，shape 签名变化自动重捕获，失败回退 eager。

**排障记录（三个坑）**：
1. per-block compile reduce-overhead 崩溃（block 输出跨 replay 覆写）→ 放弃 per-block，改整步手动捕获。
2. capture 报 `operation failed due to a previous error`：**根因是 ActionDiT 的 RoPE `freqs` 是普通属性（非 buffer）常驻 CPU，pre_dit 每步做 pageable H2D 拷贝——捕获期禁止操作**。捕获前一次性迁到 device 即修复（常量迁移，数值不变）。
3. 每次调用重新 warmup+capture 的一次性开销 ~270ms 吃掉收益 → 跨调用持久化后收益完整释放。

**实测（bs=1，20 步，稳态）**：
| 配置 | 延迟 | 速比 |
|---|---:|---:|
| eager | 544 ms | 1.00× |
| denoise graph 持久化 | 188.6 ms | 2.88× |
| **+ mot compile** | **110.3 ms** | **4.93×** |

**等价性（最强级别）**：captured 与 reused 两次调用输出均与 eager **逐位相等（max_abs=0）**——graph 只固化 launch 序列，零数值差异。显存增量仅 MB 级静态 buffer。

## 方法五：关闭 gradient checkpointing（训练用，✅显存足够时采纳）

fold_cloth 任务默认 `model.mot_checkpoint_mixed_attn=false`（README 全局默认是 true）。ckpt on 时 backward 重算 forward。bs16 下 ckpt on（默认口径）1370ms vs 全优化（ckpt off+方法一+三）976.6ms。显存允许时应关闭；与 mot compile 互斥（compile 自动跳过）。

## 训练组合矩阵（4×H200 实测汇总）

| 配置 | step | ms/sample | vs README 默认 |
|---|---:|---:|---:|
| README 默认（ckpt on, bs16） | 1370 ms | 85.6 | 1.00× |
| 全优化同 bs16 | 976.6 ms | 61.0 | 1.40× |
| 全优化 bs24 | 1330 ms | 55.4 | 1.55× |
| **全优化 bs40（吞吐推荐）** | 2083 ms | **52.1** | **1.64×** |

全优化 = `model.mot_torch_compile=true model.vae_encode_functional=true`（ckpt off 为任务默认）。VAE 优化省下的显存支撑更大 batch；bs40 下 4 卡显存仍有富余。所有 run loss 逐 step 同分布（step40 loss 0.88-0.99 区间，shuffle 噪声级）。

## 证伪结论（同样重要，避免后人踩坑）

1. **GEMM 16B 对齐修复（v4 的 2.18×）对原版无效**。原版 DiT 1649 参数天然全对齐（探针 `mod16={0:1649}`），两个 trace 均 0 处 `cutlass_75_align1`，GEMM 全走 `nvjet_*`。v4 的问题源于其独有的 `source_embedding_gate` 标量参数（numel=1）推歪 DeepSpeed flatten 指针——原版没有该参数。A/B 实测 GPU kernel 时间完全一致。**勿在原版开启 `align_optimizer_param_order`（无收益且改变 optimizer state 布局）。**
2. **per-block torch.compile reduce-overhead 不可用**（训练 VAE 与推理 MoT 均崩）：多个小图共享内存池时输出被后续 capture/replay 覆写。要 CUDA graphs 必须整步/功能化捕获。
3. **attention 不是原版训练的主要瓶颈**：attn 全部（fwd+bwd）仅 ~108ms/step（11%），flash 拆分收益上限 ~5%，低优先级。
4. **DeepSpeed overlap_comm+contiguous_gradients 是负优化**（v4 实测 backward 917ms→10.8s），保持默认关闭。
5. **VAE latent cache（磁盘预计算）**：2.33×（同 batch）/2.57×（bs40）——效果最好但违反"不占空间"约束，仅作参考线（代码已在分支 `4617e4a`，含 fingerprint 覆盖复用机制）。

## 训练 2× 为何在约束内不可达（经验性证明，基于实测 trace）

用 optim_bs16 run 的 torch profiler trace（15 步 active window）实测：

```text
GPU busy（bwd+vae+fwd 三段 annotation 的 kernel 执行时间和）= 792 ms/step（墙钟 976.6 ms 的 81%）
GPU busy / sample = 792/16 = 49.5 ms   ← 任何 launch 级优化（compile/CUDA graphs/放大 batch/流重叠）的硬下限
训练 2× 目标 = 85.6/2 = 42.8 ms/sample 墙钟
```

**49.5 > 42.8：实测的纯 GPU 计算时间已经超过 2× 目标墙钟。** 等价性约束下的所有优化手段都只作用于「墙钟 − GPU busy」这 19% 的空转，无法低于 GPU 真正执行计算的时间。

对最后一条未实测路径（flash attention mask 拆分）的定量封闭：trace 实测 attention 全部 GPU 时间（fwd+bwd）= 68.6 ms/step = **4.3 ms/sample**。即使 flash 把 attention **整段消除**（物理不可能的收益上限，实际 flash 只加速不消除），下限仍为 49.5 − 4.3 = **45.2 ms/sample > 42.8**。故 flash 路线无论做到多好都无法使训练达到 2×，无需实测即可排除。

剩余 GPU busy 的构成是真实计算：DiT GEMM（~450ms/step）、VAE conv（~320ms）、梯度 elementwise 与 NCCL。要减少它只有三条路，均违反本轮约束：
- fp8 / TF32 降精度 —— 破坏数值等价；
- 磁盘/内存 latent cache —— 破坏"不增加空间占用"（作为参考线实测可达 2.33-2.57×，代码在分支 `4617e4a`）；
- 蒸馏 / 减层 / 改 attention 结构 —— 改变模型。

**结论：在「完全等价 + 不占空间」双约束下，训练 2× 与实测 GPU 计算下限数学冲突；1.64×（实测）~1.7×（理论上限）是该约束下的可达区间。若接受磁盘 latent cache（192G），训练即可达 2.33×+。**

## 复现指南

```bash
# 远端（H200 无法直连 GitHub，用 git bundle 同步；分支已在 origin）
cd /data/home/maxliu/projects/FastWAM_orig   # worktree @ experiment/gemm-align-original

# ── 训练（推荐配置）──
export PYTHONPATH=$PWD/src
bash scripts/train_fold_cloth_orig_2epoch.sh \
  batch_size=40 \
  model.mot_torch_compile=true \
  model.vae_encode_functional=true \
  data.train.pretrained_norm_stats=<dataset_stats.json>

# ── 推理（推荐配置：4.93×）──
python scripts/bench_infer_action.py --device cuda:0 --warmup 6 --iters 30 \
  --override model.infer_denoise_cuda_graph=true \
  --override model.mot_torch_compile=true

# ── 等价性验证 ──
python scripts/verify_vae_encode_equivalence.py --device cuda --dtype bf16   # A 级须 max_abs=0
python scripts/verify_vae_encode_equivalence.py --device cuda --dtype fp32   # 舍入随精度缩小的证明
python scripts/verify_denoise_graph_equivalence.py --device cuda:0           # 两次调用均须 max_abs=0

# ── 参数对齐探针（证明原版无 align1 问题）──
FASTWAM_LOG_PARAM_ALIGNMENT=1 bash scripts/train_fold_cloth_orig_2epoch.sh \
  batch_size=2 max_steps=2 ... # 预期 unaligned_params=2/1847

# ── trace 归因 ──
python scripts/analyze_trace_gpu_breakdown.py <run>/profile/torch/*.trace.json
```

## 分支 commit 清单

| commit | 内容 |
|---|---|
| d5f7b20..2f55bff | profiling 开关/对齐探针/trace 归因/microbench/param_order（自 v4 移植） |
| 2f08bc8 | fold_cloth 原版任务配置 |
| 87217e9 | mot_torch_compile per-block 区域编译 |
| 4617e4a | VAE latent cache（参考线，被空间约束排除） |
| 35fd6e6, d33793d | vae_torch_compile（编译 encoder.forward） |
| c32819b, aab5a02, 6a9048e | VAE encode 功能化 + 三级等价性验证 |
| c84830f, 22ea984 | 原版 infer_action benchmark |
| 6b68ee7, c763a65, 7a2ff4c, 2625e18 | denoise 整步 CUDA graph（捕获修复 + 持久化） |

所有开关默认关闭，eager 路径零改动；零参数/buffer 改动，任意已有 checkpoint 直接可用。
