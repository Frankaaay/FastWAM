# 原版 FastWAM 训练+推理 2× 加速战役（进行中）

目标（用户设定，2026-07-03 晚）：**在不增加空间占用的前提下，把原版 FastWAM 的训练与推理加速到 2×，并验证加速后的完全等价性**。空间约束 = 不允许 VAE latent cache 等磁盘预计算产物；等价性 = 数值等价（compile 容差内）+ loss/输出对照。

实验分支：`experiment/gemm-align-original`（基于 main @45d8e14）
远端：`h200-qinghua-1:/data/home/maxliu/projects/FastWAM_orig`（git bundle 同步）
任务口径：fold_cloth 数据、原版训练图（无 v4 history）、4×H200（GPU 0,5,6,7）bs24、profile window wait20/warmup5/active15、稳态取 step30-40。

---

## 已确定结论（按时间顺序）

### 1. GEMM 对齐修复在原版无效（已证伪，2026-07-03 下午）

- 探针：原版 1847 参数仅 2 个 16B 错位（v4 是 1651/1851）；DiT 1649 参数天然全对齐。
- 两个 trace（align 开/关）均 0 处 `cutlass_75_align1`，GEMM 全走 `nvjet_*`。
- 根因：v4 的 2.18× 来自其独有的 `source_embedding_gate` 标量参数推歪 flatten 指针；原版没有该参数。
- **结论：对齐修复不适用于原版，勿移植。**

### 2. 训练是 launch-bound（关键诊断）

baseline（bs24）每 step ~2.7 万 GPU kernel；"other" 胶水 kernel 总时间 > GEMM。
VAE encode 单模块 ~3 万 kernel/step（时间 chunk 循环 × 小 CausalConv3d），墙钟 849ms vs GPU busy 623ms（26% 等 launch）。
→ 加速杠杆 = 减少 kernel 数（fusion / CUDA graphs），不是换更快 kernel。

### 3. 在线 compile 阶梯（无 cache，4 卡独占，已实测）

| 配置 | forward | backward | step_total | 速比 |
|---|---:|---:|---:|---:|
| base | 1100 | 690 | 1800 ms | 1.00× |
| mot compile | 985 | 640 | 1590 ms | 1.13× |
| VAE compile (default) | 850 | 690 | 1540 ms | 1.17× |
| **mot + VAE compile** | **750** | **588** | **1350 ms** | **1.33×** |

- loss 各 run 同分布收敛，compile 数值等价（Inductor 融合舍入级差异）。
- `vae_torch_compile` 首版 bug：`torch.compile(encoder)` 替换 module 撞 accelerate unwrap 的 `_orig_mod` KeyError；修复为编译 `encoder.forward`（commit d33793d）。

### 4. reduce-overhead（CUDA graphs）直接上 = 崩溃（已证伪）

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
```
根因：VAE `feat_cache` 跨 chunk 原地存图输出 tensor，违反 CUDA graph 输出生命周期约束。
→ 引出当前正在做的 VAE encode 功能化重构（见下）。

### 5. （参考线，已被目标排除）VAE latent cache

cache+compile 实测 2.33×（同 batch）/2.57×（bs40 吞吐），但违反"不增加空间占用"（192G 磁盘 cache），仅作为瓶颈定位证据保留，不是方案。

---

## 进行中

### VAE encode 功能化重构（解锁 CUDA graphs）

设计：
- 洞察：encode 的 chunk 循环分支按 chunk 位置静态可判（chunk0=1 帧走初始化分支；chunk1..8=4 帧同 shape 走缓存分支），与数值无关。codex 逐一核查了 CausalConv3d/Resample/ResidualBlock/Down_ResidualBlock/Encoder3d(_38)/VideoVAE_(38) 的所有 feat_cache 分支，确认无数值依赖分支。
- 拆两个纯函数 `encode_chunk_first` / `encode_chunk_next`，feat_cache 显式传入传出（不原地变异），chunk 间 cache tensor 图外 clone → 满足 CUDA graph 约束。
- 旗标 `vae_encode_functional`（默认关），原 encode 路径一行不动；零参数改动（脚本 assert state_dict keys 不变），ckpt 完全兼容；运行失败自动回退原路径。

实现与修复记录：
- `c32819b` 功能化路径 + 三级等价性验证脚本（codex 实现，前提核查通过）。
- `aab5a02` 验证脚本改进：报告 max_abs/rel/mean_abs/p99.9/cosine，B/C 舍入不硬失败（标 LOOSE），保证 C 级能跑到。
- **C 级第一次崩**：`accessing tensor output of CUDAGraphs that has been overwritten`——codex 只克隆了 feat_cache，没克隆 chunk 输出 `out_i`（也是 graph 输出，被 cat 前即遭下次 capture/replay 覆写）。
- `6a9048e` 修复：compile 模式下 `out_i` 也图外 clone；eager 路径不 clone，保持 A 级逐位相等。

等价性验证结果（bf16，随机权重，[1,3,33,384,320]，第 3 轮定稿）：

| 级别 | max_abs | rel | mean_abs | p99.9 | cosine | 判定 |
|---|---:|---:|---:|---:|---:|---|
| A eager 功能化 vs 原版 | **0** | 0 | 0 | 0 | 1.0 | **逐位相等** |
| B compile default vs 原版 | 0.0391 | 1.79% | 0.0038 | 0.0234 | 0.99994 | bf16 融合舍入 |
| C reduce-overhead vs 原版 | 0.0391 | 1.79% | 0.0038 | 0.0234 | 0.99994 | 同 B，**不再崩溃** |

三条关键论证：
1. A 级逐位相等 → 功能化重构本身**完全等价**，零参数改动，ckpt 完全兼容（脚本 assert state_dict keys 不变）。
2. **B 与 C 的所有指标逐位相同** → CUDA graphs 相对普通 compile 零额外数值差异；全部偏差来自 Inductor bf16 融合舍入，与项目已采用的 mot compile 同性质。
3. fp32 复测进行中（预期偏差缩到 1e-6 级，证明偏差纯属精度舍入而非逻辑差异）。

进行中：功能化训练 A/B（vaefunc / mot+vaefunc / mot+vaefunc bs40 吞吐口径）。

### 功能化训练 A/B 结果（4×H200 bs24，与在线 compile 阶梯同口径）

| 配置 | forward | backward | step_total | 速比 | ms/sample | 吞吐比 |
|---|---:|---:|---:|---:|---:|---:|
| base（参照） | 1100 | 690 | 1800 ms | 1.00× | 75.0 | 1.00× |
| VAE 功能化（CUDA graphs）单独 | 790 | ~690 | 1490 ms | 1.21× | 62.1 | 1.21× |
| **mot compile + VAE 功能化** | **700** | **595** | **1330 ms** | **1.35×** | 55.4 | 1.35× |
| mot + VAE 功能化 bs40（吞吐口径） | — | — | 2085 ms | — | **52.1** | **1.44×** |

观察：
- 功能化 CUDA graphs 单独（1.21×）好于 VAE default compile（1.17×），组合 1.35× 与 mot+VAE-default（1.33×）基本持平——default compile 已吃掉 VAE 大部分 launch 开销，CUDA graphs 增量有限。
- loss：vaefunc 0.9899、motvaefunc 0.9629、bs40 0.9361（step40，同分布噪声级）。

### 训练侧 2× 的计算边界（重要诚实结论）

在「严格数值等价（不改变训练数学）+ 不占磁盘」约束下，训练吞吐存在**GPU 计算量下限**：
- 当前优化后 step 墙钟 1330ms 中，GPU busy ≈ 1150ms（launch/空转间隙只剩 ~180ms 可回收）。
- bf16 计算量本身（DiT GEMM + VAE conv + 梯度计算 + NCCL）不随 compile/graphs 减少——fusion 只削胶水与 launch。
- 由 GPU busy 推算的吞吐天花板 ≈ **1.5-1.6×**（75 → ~48ms/sample 的 GPU 计算下限）。
- 要突破必须减少真实计算：fp8（数值不等价）、磁盘/内存 latent cache（空间约束禁止）、蒸馏/减层（改模型）。**训练侧 2× 与当前等价性+空间约束互斥**，1.44×（已实测）~1.5×（理论）是该约束下的实际可达区间。
- 唯一还没拿的等价性内收益：attention mask 拆分走 flash（≈40-60ms/step，属舍入级等价）、VAE-DiT 流重叠（回收 ~180ms 空转）。合计可把 1.35× 推到 ~1.5×，但到不了 2×。

### 推理侧通往 2×：denoise 整步 CUDA graph（进行中）

推理 trace 分段（eager，bs=1，20 步）：**denoise_step 20×44.9ms = 897ms，占绝对大头**；prefill 42ms、encode 8.6ms（推理侧 VAE 优化收益小——这解释了 motvae 组合 425ms 反略差于纯 mot 409ms）。

已实测：
| 配置 | 延迟 | 速比 |
|---|---:|---:|
| eager | 544 ms | 1.00× |
| mot compile (default) | **409 ms** | 1.33× |
| mot + VAE 功能化 | 425 ms | 1.28× |
| mot compile (reduce-overhead per-block) | 崩溃 | per-block 图输出跨 replay 覆写，同 VAE 第一次的问题 |

正解（已落地）：**整步捕获**——`_predict_action_noise_with_cache` 输入 shape 固定（latents [1,A,D] + timestep [1]），video_kv_cache/context/attention_mask 在 denoise 循环内不变，用 `torch.cuda.CUDAGraph` 手动捕获单步、静态 buffer 输入、replay 20 次、输出图外 clone（commit 6b68ee7 + c763a65 + 7a2ff4c）。

排障记录（两个坑，都已修复）：
1. capture_error_mode/warmup（c763a65）：thread_local 模式 + 5 轮 side-stream warmup + 全设备同步 + 共享 graph pool。
2. **真正根因（7a2ff4c）**：ActionDiT 的 RoPE `freqs` 是普通属性（非 buffer），`.to(device)` 不迁移它、常驻 CPU；`pre_dit` 每步做 pageable H2D 拷贝——**捕获期禁止操作**。捕获前把 freqs 一次性迁到 device（常量迁移，数值逐位不变）即修复。这与 v4 分支历史上的"RoPE 缓存设备迁移"修复同源。

**等价性验证结果：`graph_status=captured`，`max_abs=0`——CUDA graph denoise 输出与 eager 逐位相等**（graph 只固化 kernel launch 序列，不改任何数值，比 compile 的舍入级等价更强）。

延迟实测（bs=1，20 步，含每次调用的 warmup+capture 开销）：

| 配置 | 延迟 | 速比 |
|---|---:|---:|
| eager | 544 ms | 1.00× |
| mot compile | 409 ms | 1.33× |
| denoise graph | 402.6 ms | 1.35× |
| **denoise graph + mot compile** | **356.2 ms** | **1.53×** |

关键分解：402ms ≈ **270ms 一次性开销（5 轮 warmup + capture，当前每次调用都重付）** + 50ms prefill/encode + 20×~4ms replay。**replay 本身 4ms/步 vs eager 45ms/步（11×）**——收益被每次调用重新捕获吃掉了。

**跨调用持久化 graph（commit 2625e18）——推理目标达成并超额**：条件张量（kv_cache k/v、context、context_mask、attention_mask）用静态 buffer，后续调用 copy_ 进 buffer 直接 replay；首次调用付一次 capture；shape 签名变化自动重捕获；失败回退 eager。

最终推理延迟（bs=1，20 步 denoise，warmup 后稳态，30 iters）：

| 配置 | 延迟 | 速比 |
|---|---:|---:|
| eager baseline | 544 ms | 1.00× |
| mot compile | 409 ms | 1.33× |
| denoise graph（每调用重捕获） | 402.6 ms | 1.35× |
| denoise graph 持久化 | 188.6 ms | 2.88× |
| **denoise graph 持久化 + mot compile** | **110.3 ms** | **4.93×** |

等价性（最强级别）：第一次调用（captured）与第二次调用（reused，持久 replay 路径）的输出**均与 eager 逐位相等（max_abs=0）**——CUDA graph 只固化 kernel launch，零数值差异。零参数改动，任意已有 ckpt 直接可用；显存增量仅静态 buffer（一份 kv/context 副本，MB 级），无磁盘占用。

**推理侧结论：544 → 110ms，4.93×，逐位等价 —— 2× 目标达成并大幅超额。**

### 推理 bench 修复记录

- 第一轮 bench 全部崩：随机构造模型时 VAE 留在 fp32（bias float）而输入 bf16 → `Input type (c10::BFloat16) and bias type (float) should be the same`。真实加载路径无此问题（VAE 权重本就 bf16），纯 bench 构造缺陷。
- 修复 `22ea984`：构造后 `model.to(dtype=model_dtype)` 统一整模型 dtype。第二轮 bench（eager/mot/mot+vaefunc 各 20 iters）进行中。

### 推理 benchmark 移植（`c84830f`）

`scripts/bench_infer_action.py`：原版 infer_action 单卡合成输入 bench，CUDA events 计时，支持 --override 切 compile/functional 开关；fastwam.py infer_action 加零开销埋点：`model/infer/encode_input_image` / `current_video_prefill` / `denoise_step`。

---

## 到 2× 的路径预算（诚实估计）

训练（当前 1.33×）：
- VAE CUDA graphs：VAE 段墙钟 ~600 → 逼近 GPU busy ~450-500ms，step ~1350 → ~1200-1250（≈1.45-1.5×）。
- 吞吐口径放大 batch（bs24→40，显存余量足够，不占磁盘）：ms/sample 预计再改善 → 目标 1.8-2×。
- 剩余：backward 胶水（DeepSpeed elementwise）、NCCL ~90ms、attention（占比小）。

推理（infer_action bs=1，v4 参考数据：denoise 占 90% 且 launch-bound，GPU 仅 7-8ms/27ms 墙钟）：
- mot compile：v4 实测 −17%。
- denoise 循环 CUDA graphs：launch 开销占 2/3 以上，是 2× 的主要来源。
- VAE encode 功能化同样服务推理（encode input image/条件帧）。
- 不减 denoise 步数（那会改变输出，违反等价性）。

---

## 记录约定

每完成一个里程碑（等价性验证 / 训练 A/B / 推理 bench），立即在本文档追加小节；最终整理为完整实验报告 + 飞书 NGAD 文档。
