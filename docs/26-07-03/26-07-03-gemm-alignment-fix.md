# FastWAM GEMM 对齐修复：align1 kernel 根因、修复与验证

更新时间：2026-07-03 14:45 CST

本文件记录 FastWAM 训练/推理 bf16 GEMM 落到慢速 `cutlass_75_*_align1` kernel 的根因定位、
修复实现（`align_optimizer_param_order`）和端到端验证结果，供后续复现与推理侧迁移参考。

## TL;DR

- **现象**：H200（SM90）上约 90% 的 bf16 GEMM 落到 SM75 时代的 `cutlass_75_tensorop_bf16_s1688gemm_*_align1`
  kernel，而不是 Hopper 原生 `nvjet_*`；microbench 实测同形状慢 7-8 倍。
- **根因**：DeepSpeed ZeRO-1 把 optimizer 参数 flatten 成一个连续 bf16 buffer；参数顺序里有
  **两个 numel 为奇数的标量参数**（`video_expert.source_embedding_gate`、`action_expert.source_embedding_gate`，
  numel=1），把其后所有参数的指针推离 16B 对齐边界，cuBLAS 被迫走 `align1` 兼容路径。
- **修复**：新增 `align_optimizer_param_order`（默认已开启）。在构建 AdamW 前，把 numel 非 16B 对齐倍数
  的参数稳定地移动到参数列表尾部，使前面所有大权重回到 16B 对齐。**不改变训练数学**。
- **实测收益**：同口径 profiling run，`step_total` 从 2889ms 降到 1326ms（**2.18× 吞吐**），
  align1 kernel 完全消失，全部 GEMM 走 `nvjet_*`。
- **副作用**：改变 optimizer state flatten 布局，**不能 resume 由旧顺序（关闭时）训练产生的 checkpoint**。
  全新训练不受影响。

## 背景与定位链路

定位过程按证据逐级收敛，完整分析见 `26-07-03-attn-gpu-kernel-breakdown.md`，此处只记录结论链：

1. VAE latent cache 落地后，训练 forward 的下一个瓶颈是 `model/v4/video_prefill_cache`（~957ms/step）。
   曾怀疑是 attention backend（bool mask 导致走 efficient 而非 flash attention）。
2. 对 cached trace 做 GPU kernel 级归因后发现：attention forward 只占 prefill 的 ~1.5%
   （`mot/prefill/attn` 仅 19ms/step），attention 方向收益上限 ≤3%，**不是瓶颈**。
3. 真正的瓶颈是 GEMM：75% 的 GPU 时间是 GEMM，其中 ~90%（约 1897ms/step）落在三个
   `cutlass_75_*_align1` kernel 上。H200 上 bf16 GEMM 正常应走 `nvjet_*`。
4. `align1` 表示某个操作数指针或 leading dim 未满足 16B 对齐，cuBLAS 退到兼容 kernel。
   trace 中仍有一小部分 GEMM 走 `nvjet_*`，说明不是全局 cuBLAS 启发式问题，而是**部分操作数错位**。

## 判别实验

### 实验 1：GEMM 对齐 microbench（确认机制）

脚本：`scripts/microbench_gemm_alignment.py`（单卡，对齐 vs 2 字节错位 4 变体）。
H200 GPU7 / torch 2.7.1+cu128，用模型代表形状（M=72000，K=3072，N∈{3072,14336}，bf16，`F.linear`）：

| case | proj3072 | ffn14336 | kernel |
| --- | ---: | ---: | --- |
| 对齐（ptr%16=0） | 1.715ms | 9.097ms | `nvjet_tst_*`（Hopper） |
| 错位（任一操作数 ptr%16≠0） | ~14.2ms | ~64.0ms | `cutlass_75_*_s1688gemm_*` |
| **slowdown** | **8.3×** | **7.0×** | |

结论：任一操作数指针 16B 错位即触发训练 trace 中同名 align1 kernel，慢 7-8 倍。

### 实验 2：参数/激活对齐探针（定位错位来源）

探针：`src/fastwam/utils/align_probe.py`，通过环境变量开启，在 `accelerator.prepare()`（即 DeepSpeed
wrap）之后统计：

- `FASTWAM_LOG_PARAM_ALIGNMENT=1`：所有参数 `data_ptr() % 16` 余数分布 + 前若干未对齐参数明细。
- `FASTWAM_LOG_ACT_ALIGNMENT=1`：前 8 个 `nn.Linear` 首次 forward 抽查输入/权重指针对齐与 contiguous。

修复前 run（`align_probe_fold_clothv4_v4_20260703_135354`，8 卡真实训练环境）结果：

```text
param_alignment total_params=1851 checked_params=1851 unaligned_params=1651
ptr_mod16_distribution={0:200, 2:827, 4:824}
act_alignment: 前 8 个 Linear 输入全部 input_ptr_mod16=0（激活侧对齐）
```

判定：错位在**参数**侧（89% 未对齐），激活侧干净 → DeepSpeed flatten 假设成立，只需修参数顺序。

## 根因

`trainer.py` 构建 AdamW 的可训练参数列表：

```python
trainable_params = list(self.model.dit.parameters())
proprio_encoder = getattr(self.model, "proprio_encoder", None)
if proprio_encoder is not None:
    trainable_params.extend(list(proprio_encoder.parameters()))
```

DeepSpeed ZeRO-1 按该列表顺序把参数 flatten 成一个连续 bf16 buffer；每个参数在 buffer 里的字节偏移
= 前序所有参数 numel 的累计和 × 2（bf16）。16B 对齐要求偏移是 8 个 bf16 元素的整数倍。

参数顺序里存在 numel 为奇数（非 8 倍数）的参数，最关键的是两个标量 gate：

- `video_expert.source_embedding_gate`（numel=1）→ 其后 827 个参数被推到 `ptr%16=2`
- `action_expert.source_embedding_gate`（numel=1）→ 之后 824 个参数被推到 `ptr%16=4`

一个奇数 numel 参数会让其后**所有**参数错位，因此 1851 个里有 1651 个未对齐。这些错位参数正是
video/action expert 里的大 Linear 权重（q/k/v/o、FFN），它们进 GEMM 时触发 align1。

## 修复实现

新增开关 `align_optimizer_param_order`（`configs/train.yaml` 根级，已默认 `true`）。

- `src/fastwam/utils/param_order.py`
  - `reorder_params_for_alignment(params, *, alignment_bytes=16)`：纯函数，稳定分区。
    对每个参数计算 `k = alignment_bytes // param.element_size()`（bf16 下 k=8）；
    `numel % k == 0` 的参数保持原相对顺序放前面，其余（tail）保持原相对顺序放后面，
    返回 `(aligned + tail, tail)`，不修改输入 list。
  - `log_param_order_tail(...)`：rank0 打印被移到 tail 的参数数量与 name/numel/shape。
    日志前缀用 `param-order` 而非 `[param-order]`（`RichHandler(markup=True)` 会吞掉方括号标签）。
- `src/fastwam/trainer.py`
  - 读取 `cfg.align_optimizer_param_order`；开启时在创建 AdamW **之前**重排 `trainable_params`，
    rank0 打印 tail 明细并 warning：重排改变 DeepSpeed flatten 顺序，与旧 optimizer state checkpoint
    不兼容，不要在旧 run 上 resume。

### 为什么不影响训练本身

1. AdamW 是逐参数独立更新（每个参数有自己的 momentum/variance），与参数在列表中的顺序无关。
   重排只改变 flatten 后的内存排布，不改变任何一个参数的梯度、更新量或数学结果。
2. `trainer.py` 中对 optimizer 的其他引用（grad-clip 用 `model.parameters()` 原顺序、读 `param_groups[0]["lr"]`）
   都与参数列表顺序无关。
3. 实测两个 run 的 loss 曲线逐 step 重合（见下），差异在数据 shuffle 噪声量级。

### 唯一副作用：checkpoint resume 兼容性

`trainer.py` 用 `accelerator.save_state / load_state` 保存/加载 optimizer state。重排改变 flatten 布局，
所以**由 `align_optimizer_param_order=false` 训练出的 checkpoint，不能在 `true` 下 resume**（反之亦然）。
纯权重加载（`load_checkpoint(..., optimizer=None)`）不受影响。全新训练无此问题。

## 端到端验证结果

同口径 profiling 对比（fold_clothv4_v4，batch 24，8 卡，profile window wait=20/warmup=5/active=15，
latent cache 相同），唯一差异是 `align_optimizer_param_order`：

- baseline（关闭）：`profile_trace_attnmask_fold_clothv4_v4_20260703_132422`
- alignfix（开启）：`profile_trace_alignfix_fold_clothv4_v4_20260703_142741`

### 参数对齐（探针）

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| unaligned_params | 1651 / 1851 | **2 / 1851** |
| ptr_mod16 分布 | `{0:200, 2:827, 4:824}` | `{0:1849, 2:1, 4:1}` |
| tail（被移到尾部的参数） | — | `source_embedding_gate`×2（numel=1）、`action_expert.head.bias`（numel=14） |

修复后仅剩的 2 个错位是标量/14 维 bias 本身，不进 GEMM，无影响。

### Step timing（per-step，profile window 稳定 step）

| 指标 | baseline | alignfix | 变化 |
| --- | ---: | ---: | ---: |
| **step_total** | 2889 ms | **1326 ms** | **−54%（2.18×）** |
| forward | 1356 ms | 412 ms | −70% |
| backward | 1530 ms | 917 ms | −40% |
| data | 1.5 ms | 1.9 ms | ~ |

### GPU kernel 归因（全 active window，15 step）

| 类别 | baseline | alignfix |
| --- | ---: | ---: |
| gemm 总时间 | 31380 ms | **6449 ms（−79%）** |
| `cutlass_75_*_align1` | ~1897 ms/step，占 GEMM 90% | **0（消失）** |
| top GEMM kernel | `cutlass_75_s1688gemm_*_align1` | `nvjet_tst_*`（Hopper 原生） |

### Loss 一致性（训练不受影响的证据）

| step | baseline loss | alignfix loss |
| --- | ---: | ---: |
| 25 | 3.1886 | 3.1820 |
| 30 | 3.1089 | 3.0974 |
| 35 | 2.8402 | 2.8427 |
| 40 | 2.7590 | 2.7485 |
| 45 | 2.5978 | 2.5882 |

逐 step 重合，差异为数据 shuffle 噪声量级，确认重排不改变训练数学。

> 注：forward 降幅（−70%）大于 backward（−40%），因为 backward 里除 GEMM 外还有 NCCL 通信/等待
> 与 elementwise，占比更高；align1 修复只作用于 GEMM 部分。整体 step 仍拿到 2.18×。

## 复现步骤

### 1. 确认代码与配置

```bash
cd /data/home/maxliu/projects/FastWAM
git checkout profiling-mem-stage-v4
git log --oneline -1   # 应包含 align_optimizer_param_order 修复
grep align_optimizer_param_order configs/train.yaml   # 默认 true
```

### 2. 验证参数对齐（快，2 step）

```bash
cd /data/home/maxliu/projects/FastWAM
export RUN_ID=align_verify_$(date +%Y%m%d_%H%M%S)
export FASTWAM_LOG_PARAM_ALIGNMENT=1 FASTWAM_LOG_ACT_ALIGNMENT=1
bash scripts/train_fold_clothv4_v4_2epoch.sh \
  batch_size=24 max_steps=2 save_every=100000 eval_every=0 \
  align_optimizer_param_order=true \
  output_dir=./runs/fold_clothv4_v4_2epoch/${RUN_ID} \
  data.train.pretrained_norm_stats=<dataset_stats.json> \
  +data.train.vae_latent_cache_dir=<vae_latent_cache_dir> \
  +data.train.vae_latent_cache_model_id=Wan-AI/Wan2.2-TI2V-5B \
  wandb.enabled=false
# 预期日志：param_alignment unaligned_params=2；param-order tail_count=3
```

对照实验（关闭修复）把 `align_optimizer_param_order=false`，预期 `unaligned_params=1651`。

### 3. 单卡 GEMM microbench（可选，确认 kernel 差异）

```bash
CUDA_VISIBLE_DEVICES=<空闲卡> python scripts/microbench_gemm_alignment.py --linear
# 对齐行 top_kernel=nvjet_*；错位行 top_kernel=cutlass_75_*_s1688gemm_*，慢 7-8 倍
```

### 4. 同口径 profiling 对比（可选，量化吞吐）

用与 baseline 完全相同的 batch/profile 窗口/latent cache，仅切换 `align_optimizer_param_order`，
跑到 profile window 结束后对 trace 跑 `scripts/analyze_trace_gpu_breakdown.py`，对比 align1 kernel 与
step timing。命令模板见上述两个 run 的 wrapper log。

## 影响面与后续

- **训练**：默认已开启，全新训练直接受益 ~2.18×。旧 `false` checkpoint 不可 resume（改用纯权重加载或从头训）。
- **推理**：推理用同一批权重做 GEMM。只要权重来自对齐顺序训练的 checkpoint，或推理侧构建 optimizer/参数
  时同样保证 16B 对齐，align1 问题一并消除。推理侧 prefill 直接受益。后续接推理加速时应验证此点。
- **attention backend**：收益 ≤3%，继续搁置；`attention_debug` 开关保留，可在任意 profiling run 顺带观测。

## 相关产物

- 修复代码：`src/fastwam/utils/param_order.py`、`src/fastwam/trainer.py`、`configs/train.yaml`。
- 判别探针：`src/fastwam/utils/align_probe.py`（env var 驱动）。
- microbench：`scripts/microbench_gemm_alignment.py`。
- trace 归因：`scripts/analyze_trace_gpu_breakdown.py`。
- 归因分析背景：`docs/26-07-03/26-07-03-attn-gpu-kernel-breakdown.md`。
- 对照 run：baseline `profile_trace_attnmask_fold_clothv4_v4_20260703_132422`、
  alignfix `profile_trace_alignfix_fold_clothv4_v4_20260703_142741`。
