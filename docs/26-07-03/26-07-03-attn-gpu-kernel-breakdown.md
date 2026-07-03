# FastWAM GPU kernel 归因分析：attention 不是主瓶颈

更新时间：2026-07-03 下午

## 目的

按交接文档的第 1 步（先从现有 trace 算 attention 占比，再决定是否投入 attention backend 实验），
对 cached profiling run 的 trace 做 GPU kernel 级归因，回答两个问题：

1. `model/v4/video_prefill_cache` 的 GPU 时间里，attention kernel 占多少。
2. `train/backward` 的 GPU 时间里，attention backward 占多少。

## 数据与方法

- Trace：`profile_trace_latcache_fold_clothv4_v4_20260703_045618`（W&B `d40rx5m9`），
  active window 15 step（`ProfilerStep#25-39`），rank0。
- 方法：`scripts/analyze_trace_gpu_breakdown.py`。
  把 `cat=kernel` 事件按同 GPU 轨道上的 `gpu_user_annotation` 窗口归因（嵌套窗口全部计入），
  kernel 名正则分类为 `attn_fwd` / `attn_bwd` / `gemm` / `other`。
- 局限：窗口归因基于同轨道时间包含关系；NCCL 等独立 stream 的 kernel 不计入任何注解，只出现在全局汇总。

## 结果（per-step，均为 15 step 平均）

| annotation | GPU kernel 总计 | attn_fwd | attn_bwd | gemm | other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model/training_loss_v4`（forward） | 1317.6ms | 32.7ms | 0 | 1129.3ms | 155.7ms |
| `model/v4/video_prefill_cache` | 1231.3ms | 27.0ms | 0 | 1080.4ms | 123.9ms |
| `train/backward` | 1358.6ms | 0 | 101.1ms | 962.2ms | 295.3ms |
| `model/v4/future_action_with_condition_cache` | 34.7ms | 3.5ms | 0 | 15.9ms | 15.2ms |
| `model/v4/history_action_prefill_cache` | 24.6ms | 2.1ms | 0 | 12.7ms | 9.7ms |

全局（per-step）：

| 类别 | GPU 时间 | 占比 |
| --- | ---: | ---: |
| gemm | 2091.5ms | 75.5% |
| other（elementwise/norm/copy/reduce） | 545.7ms | 19.7% |
| attn_bwd | 101.1ms | 3.6% |
| attn_fwd | 32.7ms | 1.2% |

## 结论 1：attention backend 方向收益上限只有 ~3%

- `video_prefill_cache` 1231ms GPU 时间里 attention forward 只有 27ms（2.2%）。
- 全 step attention fwd+bwd 合计约 134ms，加上 bool mask 物化/读取的带宽开销，
  no-mask flash 上限实验的理论收益 ≤ ~150ms/step，占 step_total（约 4.4s）的 ~3%。
- 交接文档里"attention backend / mask 拆分 / varlen attention"整条线降级：
  在 GEMM 问题解决之前不值得投入，mask 拆分重构暂停。
- 已实现的 `attention_debug` 开关保留：`log_sdpa_backend` 观测成本为零，
  `force_video_prefill_no_mask` 可在后续任意 profiling run 里顺带验证上限，不单独跑实验。

## 结论 2：真正的瓶颈是 GEMM 落在 legacy align1 kernel 上

Top kernels（per-step）：

```text
809.5ms  cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1
797.1ms  cutlass_75_tensorop_bf16_s1688gemm_bf16_256x128_nn_align1
290.2ms  cutlass_75_tensorop_bf16_s1688gemm_bf16_256x128_tn_align1
138.4ms  nvjet_*（Hopper 原生 cuBLAS kernel，合计）
```

三个 `cutlass_75_*_align1` kernel 合计约 1897ms/step，占全部 GEMM 时间的 90.7%。

问题所在：H200（SM90）上 bf16 GEMM 正常应主要走 `nvjet_*`（Hopper 优化 kernel），
现在几乎全部落在 SM75 时代的 `s1688gemm` 且是 `align1`（最差对齐）实例上。
`align1` 通常意味着某个操作数指针或 leading dim 不满足 16B 对齐，cuBLAS 被迫走兼容路径。
这类 kernel 相比 nvjet 通常有 1.5-2.5x 的差距，对应 **600-1100ms/step（约 15-25% step time）的潜在收益**，
远大于 attention 方向。

### 根因假设（按可能性排序）

1. **DeepSpeed ZeRO-1 参数 flatten 导致权重指针错位**：
   flatten 后各 param 是 flat buffer 的 view，偏移是前序 param numel 的累计和；
   只要有一个奇数大小的 param（如 odd action_dim 的 bias），其后所有权重都会 2 字节错位，
   系统性触发 align1。这与"几乎所有 GEMM 都是 align1"的现象吻合。
2. 激活侧非连续/错位（rearrange、slice 产生的视图直接进 Linear）。
3. cuBLAS/torch 2.7.1+cu128 在这些 shape 上的启发式选择问题（可能性较低）。

### 判别实验（下一步 profiling 的主内容）

1. **参数对齐探针**（成本最低，先做）：训练启动后（DeepSpeed wrap 之后）遍历
   `model.named_parameters()`，统计 `p.data_ptr() % 16 != 0` 的数量并打印前几个名字。
   若大量未对齐 → 假设 1 成立。
2. **纯 GEMM microbench**：在 H200 上用本模型的 shape
   （M=B*Sv≈24*3000+，K=3072，N∈{3072, 9216, 14336}，bf16，含对齐/错位两组）
   跑 torch.profiler，确认对齐时 cuBLAS 选 nvjet 以及两者吞吐差，量化收益上限。
3. **单卡无 DeepSpeed 短 profile**：同模型 forward-only 几个 step，
   若 GEMM 变回 nvjet → 直接锁定 DeepSpeed flatten。

### 修复方向（待判别实验确认后）

- 若为假设 1：给奇数 numel 的参数做 padding 对齐，或调整 DeepSpeed flatten 行为/参数分组顺序。
- 若为假设 2：定位具体 op 后补 `.contiguous()` 或调整布局。
- 之后再考虑 elementwise/norm 的 fusion（`other` 类还有约 546ms/step，torch.compile 潜在收益）。

## 修正后的优先级

1. align1 GEMM 根因判别 + 修复（潜在 15-25%/step）。
2. `other` 类 elementwise fusion / torch.compile（潜在 ~10%/step，风险中）。
3. attention backend / mask 拆分（≤3%/step，搁置；`attention_debug` 开关已就绪，顺带验证即可）。
4. 推理侧加速沿用同一结论：GEMM 对齐问题同样影响推理，修复后推理 prefill 直接受益。

## 相关产物

- 分析脚本：`scripts/analyze_trace_gpu_breakdown.py`（可复用于修复后的对比 trace）。
- 实验开关：`configs/model/fastwam.yaml` 的 `attention_debug.*`，
  接线见 `src/fastwam/runtime.py`、`src/fastwam/models/wan22/fastwam.py`、`mot.py`、`wan_video_dit.py`。
- MoT prefill 细分 ranges：`mot/prefill/qkv`、`mot/prefill/attn`、`mot/prefill/post_block`
  （下一次 trace 可直接看 prefill 内部构成）。
