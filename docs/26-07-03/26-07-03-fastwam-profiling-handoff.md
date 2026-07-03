# FastWAM profiling 交接文档

更新时间：2026-07-03 13:52 CST

## 交接范围

当前目标仍是：

1. 在 `mem-stage-v4` 上实现并验证 VAE latent cache，加速训练 forward。
2. 同步到 `profiling-mem-stage-v4`，做远端 profiling 对比。
3. 根据 profiling 证据测试 attention backend。
4. 根据 GPU kernel 归因定位 `video_prefill/MoT` 里的真实瓶颈。
5. 最后补推理侧加速。

当前已经完成第 1、2、3 步。第 3 步的结论是：no-mask 能让部分 SDPA 调用走 flash，
但整体 forward 只提升约 2%，attention backend / mask split 不是当前最大收益点。
随后做了 GPU kernel 归因和 GEMM 对齐 microbench，确认当前最大瓶颈是 bf16 GEMM 大量落到
`cutlass_75_*_align1` 慢 kernel；下一步应优先定位参数或激活的 16B 对齐问题。

## 本地工作区

主 worktree：

```text
/Users/maxliu/MyProjects/AIR/202606/FastWAM_xyc
branch: mem-stage-v4
HEAD: ed32004 docs: 记录 VAE latent cache smoke 结果
```

当前有未提交改动：

```text
M docs/26-06-21/26-06-21-mem-vae-diagnostic-report.md
M docs/26-06-21/26-06-21-mem-vae-stage-case-comparison.md
M docs/26-06-28/26-06-28-mem-stage-v4-implementation.md
M docs/26-07-03/26-07-03-fastwam-profiling-handoff.md
?? docs/26-07-01/
?? docs/26-07-03/26-07-03-fastwam-throughput-gap-calculation.md
```

其中本轮相关的是：

```text
docs/26-07-03/26-07-03-fastwam-profiling-handoff.md
```

其他 `26-06-*` 文档和 `26-07-01/`、`26-07-03-fastwam-throughput-gap-calculation.md` 是既有未提交改动，不要混入本交接文档提交。

Profiling worktree：

```text
/Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/profiling-mem-stage-v4
branch: profiling-mem-stage-v4
latest code commit before this handoff update: 0e533ea feat: 添加可复用 GEMM 对齐 microbench 脚本（对齐/错位 4 变体，供修复后回归对比）
```

当前本地状态：

```text
## profiling-mem-stage-v4...origin/profiling-mem-stage-v4
```

本地 profiling worktree 已经 clean，最新提交也已在 `origin/profiling-mem-stage-v4`。

最新相关提交：

```text
0e533ea feat: 添加可复用 GEMM 对齐 microbench 脚本（对齐/错位 4 变体，供修复后回归对比）
aef2a39 feat: 添加参数与激活 16B 对齐探针定位 align1 GEMM 根因
d27ff3f docs: 补充 GEMM 对齐 microbench 结果，确认 align1 慢 7-8 倍
d6fee05 docs: GPU kernel 归因分析与优先级修正，更新 profiling 交接文档
f4db2f7 feat: 添加 trace GPU kernel 归因分析脚本
bed2a34 feat: attention_debug 实验开关与 MoT prefill profiling ranges
```

## 已实现内容

VAE latent cache 已实现并同步到两个分支：

- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - 支持 `vae_latent_cache_dir`、`vae_latent_cache_keep_video`、`vae_latent_cache_model_id`、`vae_latent_cache_validate_metadata`。
  - cache 路径：

```text
<cache_root>/<fingerprint>/<sample_idx // 1000>/<sample_idx>.pt
```

  - payload 保存 `input_latents` 和 `history_video_latents`，dtype 为 `torch.bfloat16`。

- `src/fastwam/models/wan22/fastwam.py`
  - `build_inputs()` 能读取 cached latents。
  - cached path 会跳过 `_encode_video_latents()`。
  - profiling branch 保留了 `record_function` ranges：

```text
model/build_inputs/current_cached_latents_to_device
model/build_inputs/history_cached_latents_to_device
model/build_inputs/current_video_to_latents
model/build_inputs/history_video_to_latents
```

- `scripts/precompute_vae_latents.py`
  - 支持离线预计算 VAE latents。
  - 支持 `torchrun` 多卡。
  - 新增非分布式手动分片：

```text
+vae_latent_cache.num_shards=<N>
+vae_latent_cache.shard_index=<i>
```

这个手动分片是为规避 `torchrun` 最后阶段 NCCL all-reduce timeout 加的；它已经在 H200 上实测用于补齐全量 cache。

## 验证结果

VAE cache 本地轻量验证：

```bash
python -m py_compile scripts/precompute_vae_latents.py
git diff --check -- scripts/precompute_vae_latents.py docs/26-07-03/26-07-03-vae-latent-cache.md
```

结果：两个 worktree 目标文件均通过。

Attention debug 本地轻量验证：

```bash
git diff --check -- configs/model/fastwam.yaml src/fastwam/runtime.py src/fastwam/models/wan22/wan_video_dit.py src/fastwam/models/wan22/mot.py src/fastwam/models/wan22/fastwam.py docs/26-07-03/26-07-03-fastwam-profiling-handoff.md
python -m py_compile src/fastwam/runtime.py src/fastwam/models/wan22/wan_video_dit.py src/fastwam/models/wan22/mot.py src/fastwam/models/wan22/fastwam.py
```

结果：通过。

尝试做 Hydra 配置解析：

```bash
python scripts/train.py task=fold_clothv4_v4_2epoch model.attention_debug.log_sdpa_backend=true model.attention_debug.force_video_prefill_no_mask=true model.attention_debug.disable_history_condition_dropout=true --cfg job --resolve
```

本地失败原因是当前 Python 环境缺少 `hydra`：

```text
ModuleNotFoundError: No module named 'hydra'
```

这不是代码路径验证失败；后续已在 H200 的 FastWAM conda env 中补跑配置解析。

H200 Hydra 配置解析已补跑通过：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam
python scripts/train.py task=fold_clothv4_v4_2epoch \
  model.attention_debug.log_sdpa_backend=true \
  model.attention_debug.force_video_prefill_no_mask=true \
  model.attention_debug.disable_history_condition_dropout=true \
  --cfg job --resolve >/tmp/fastwam_attention_debug_resolved.yaml
rg -n 'attention_debug|force_video_prefill|disable_history|log_sdpa' /tmp/fastwam_attention_debug_resolved.yaml
```

关键输出：

```text
168:  attention_debug:
169:    log_sdpa_backend: true
170:    force_video_prefill_no_mask: true
171:    disable_history_condition_dropout: true
```

注意：`mem-stage-v4` 根目录全量 `git diff --check` 会因为既有未提交文档里的 trailing whitespace 失败：

```text
docs/26-06-21/26-06-21-mem-vae-diagnostic-report.md:5: trailing whitespace.
```

这不是本轮目标文件。

## 远端状态

远端位置：

```text
h200-qinghua-1:/data/home/maxliu/projects/FastWAM
branch: profiling-mem-stage-v4
remote HEAD at check time: 0e533ea
```

H200 无法解析 GitHub：

```text
fatal: unable to access 'https://github.com/Frankaaay/FastWAM.git/': Could not resolve host: github.com
```

之前用 git bundle 同步过提交。若接手人需要继续同步本地提交到 H200，可以继续用 bundle/scp/fetch 的方式。

远端当前 `git status`：

```text
## profiling-mem-stage-v4...origin/profiling-mem-stage-v4 [ahead 11]
?? data
```

远端 `ahead 11` 是因为 H200 不能直接 fetch GitHub，`origin/*` 追踪引用滞后；本地已经显示
`profiling-mem-stage-v4...origin/profiling-mem-stage-v4` clean。`data` 是远端已有未跟踪目录，不要删除。

## VAE cache 产物

Cache 目录：

```text
/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22
```

Fingerprint：

```text
5ff52f56f7112f87
```

最终状态：

```text
files: 865308 / 865308
size: 192G
validated samples: 0, 432654, 865307
payload dtype: torch.bfloat16
```

第一次全量 `torchrun --nproc_per_node=8` 跑到约 `812883 / 865308` 后失败，根因是 rank6 在最终 all-reduce 统计处 NCCL timeout：

```text
WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=3, NumelOut=3, Timeout(ms)=600000)
scripts/precompute_vae_latents.py FAILED
```

后续用 8 个互不通信的单进程 shard 补齐缺失样本，`overwrite=false`，未删除已有 cache。

## Cached profiling 产物

Run：

```text
profile_trace_latcache_fold_clothv4_v4_20260703_045618
```

路径：

```text
runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618
```

关键文件：

```text
trace: profile/torch/lacy--214-30-239-40_3735196.1783026074570934851.pt.trace.json
summary: profile/trace_summary.tsv
timing lines: profile/profile_timing_lines.txt
log: runs/logs/profile_trace_latcache_fold_clothv4_v4_20260703_045618.log
wandb offline: wandb/offline-run-20260703_045846-d40rx5m9
wandb url: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/d40rx5m9
```

W&B 同步已在跳板机日志中确认：

```text
Syncing: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/d40rx5m9 ... done.
[ok] .../offline-run-20260703_045846-d40rx5m9
```

补充：`profile_timing_lines.txt` 为空，因为等待器在 trace 文件稳定后立即中断训练；trace summary 本身可用于热点对比。

## Baseline vs Cached

Baseline run：

```text
profile_trace_fold_clothv4_v4_20260702_234956
wandb: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/w158oako
```

Cached run：

```text
profile_trace_latcache_fold_clothv4_v4_20260703_045618
wandb: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/d40rx5m9
```

Trace 聚合对比：

| metric | baseline | latent cache | change |
| --- | ---: | ---: | ---: |
| `train/forward_loss` | 2735.19ms | 1402.11ms | -48.7% |
| `train/backward` | 1598.65ms | 1571.49ms | -1.7% |
| `model/build_inputs` | 1326.40ms | 0.42ms | removed |
| `model/vae_encode` | 30 calls, 662.85ms mean | 0 calls | removed |
| `model/build_inputs/current_video_to_latents` | 821.21ms | 0 | removed |
| `model/build_inputs/history_video_to_latents` | 504.72ms | 0 | removed |
| `model/v4/video_prefill_cache` | 957.59ms | 957.61ms | unchanged |

结论：

- VAE latent cache 已达到目标：训练 forward 中的 VAE encode 消失。
- forward 减少约 `1.33s/step`。
- backward 基本不变，符合 VAE 原本 frozen/no-grad 的预期。
- 下一阶段瓶颈表面上是 `model/v4/video_prefill_cache`，进一步拆到 GPU kernel 后，
  真正的大头是 GEMM 对齐问题，不是 SDPA/attention kernel 本身。

## Attention backend 初步结论

当前 attention 调用路径：

```text
src/fastwam/models/wan22/wan_video_dit.py::flash_attention()
  -> torch.nn.functional.scaled_dot_product_attention()
```

函数名叫 `flash_attention`，但实际 backend 由 PyTorch SDPA 自动选择。

Cached trace 里 attention kernel 是：

```text
aten::scaled_dot_product_attention
aten::_scaled_dot_product_efficient_attention
aten::_scaled_dot_product_efficient_attention_backward
```

没有看到 flash attention kernel。

已在 H200 / PyTorch `2.7.1+cu128` 上做过小探针：

- `attn_mask=None, is_causal=False`：Flash backend 可用。
- `attn_mask=None, is_causal=True`：Flash backend 可用。
- 任意 non-null bool `attn_mask`，包括全 True mask、first-frame mask、causal bool mask：Flash backend 不可用。
- PyTorch 明确 warning：

```text
Flash Attention does not support non-null attn_mask.
```

因此当前不能靠“强制 flash backend”解决；只要传 bool mask，PyTorch 会走 efficient attention。

## Attention backend 实验开关已实现并已验证

Profiling worktree 已实现默认关闭的实验开关：

- `configs/model/fastwam.yaml`
  - 新增 `attention_debug.log_sdpa_backend=false`。
  - 新增 `attention_debug.force_video_prefill_no_mask=false`。
  - 新增 `attention_debug.disable_history_condition_dropout=false`。
- `src/fastwam/runtime.py`
  - 将 `model.attention_debug` 透传到 `FastWAM.from_wan22_pretrained()`。
- `src/fastwam/models/wan22/wan_video_dit.py`
  - 在 `flash_attention()` 内增加一次性 SDPA backend support 日志。
  - 打开 `log_sdpa_backend` 后，每种 q/k/v/mask signature 最多打印一次。
  - 日志包含 mask shape/dtype，以及 PyTorch `can_use_flash_attention` / `can_use_efficient_attention` 判断。
- `src/fastwam/models/wan22/mot.py`
  - `prefill_video_cache()` 允许 `video_attention_mask=None`。
  - 当 mask 为 `None` 时禁止同时传 `video_key_valid_mask`，避免语义混乱。
  - 给 video prefill 内部补了 `mot/prefill/qkv`、`mot/prefill/attn`、`mot/prefill/post_block` profiler ranges，便于拆分 `video_prefill_cache`。
- `src/fastwam/models/wan22/fastwam.py`
  - `attention_debug.force_video_prefill_no_mask=true` 时，只对 video prefill self-attention 传 `attn_mask=None`，作为 flash/no-mask 速度上限实验。
  - `attention_debug.disable_history_condition_dropout=true` 时把 history condition dropout 置为 0，作为减少 per-sample key-valid mask 干扰的实验口径。
  - 打开任意 attention debug 开关都会在日志中 warning：这是 profiling-only，可能改变训练语义。

默认配置全为 `false`，因此不改变正常训练语义。

已跑两组远端实验，二者都使用完整 VAE latent cache、同一 task、同一 profile window：

### masked path

Run：

```text
profile_trace_attnmask_fold_clothv4_v4_20260703_132422
wandb: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/hc22c1y2
```

路径：

```text
runs/fold_clothv4_v4_2epoch/profile_trace_attnmask_fold_clothv4_v4_20260703_132422
trace: profile/torch/lacy--214-30-239-40_3802770.1783056579070406557.pt.trace.json
summary: profile/trace_summary.tsv
```

关键 summary：

| metric | calls | mean |
| --- | ---: | ---: |
| `train/forward_loss` | 15 | 1422.73ms |
| `train/backward` | 30 | 1568.87ms |
| `model/v4/video_prefill_cache` | 30 | 339.91ms |
| `aten::_scaled_dot_product_efficient_attention` | 2700 | 0.06ms |
| `aten::_scaled_dot_product_flash_attention` | 0 | 0 |

### no-mask / flash 上限实验

Run：

```text
profile_trace_attn_nomask_fold_clothv4_v4_20260703_134334
wandb: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/s0zvjt86
```

路径：

```text
runs/fold_clothv4_v4_2epoch/profile_trace_attn_nomask_fold_clothv4_v4_20260703_134334
trace: profile/torch/lacy--214-30-239-40_3811298.1783057704065733434.pt.trace.json
summary: profile/trace_summary.tsv
```

关键 summary：

| metric | calls | mean |
| --- | ---: | ---: |
| `train/forward_loss` | 15 | 1394.80ms |
| `train/backward` | 30 | 1575.28ms |
| `model/v4/video_prefill_cache` | 15 | 662.14ms |
| `aten::_scaled_dot_product_efficient_attention` | 2250 | 0.05ms |
| `aten::_scaled_dot_product_flash_attention` | 450 | 0.06ms |

注意：`video_prefill_cache` 两组 calls 不同，不能直接比较 mean；应看 `train/forward_loss` 和
`video_prefill_cache` total。no-mask 上限实验确实让 450 个 SDPA forward call 走 flash，
但 `train/forward_loss` 只从 `1422.73ms` 降到 `1394.80ms`，约 `-27.93ms` / `-2.0%`；
`train/backward` 基本不变。跳板机 W&B sync 日志已确认 `hc22c1y2` 和 `s0zvjt86` 均同步成功。

结论：attention backend 方向收益上限很小，复杂的 `first_frame_causal` mask split 暂停。

## GPU kernel 归因与下一步判断

当前 v4 主要瓶颈：

```text
model/v4/video_prefill_cache: ~957.6ms
```

相关路径：

```text
FastWAM._training_loss_v4()
  -> video_expert.build_video_to_video_mask()
  -> mot.prefill_video_cache()
  -> MoT._apply_key_valid_mask()
  -> MoT._mixed_attention()
  -> F.scaled_dot_product_attention(..., attn_mask=...)
```

当前 `video_attention_mask_mode = first_frame_causal`：

- clean/history prefix tokens 不看 future tokens。
- future tokens 看全部 video tokens。
- 另外还有 `video_key_valid_mask`，用于 history dropout / padding。

因为 `_apply_key_valid_mask()` 会把 mask 规范化为 `[B,1,Q,K]`，所以 flash backend 必然不可用。

但是 `scripts/analyze_trace_gpu_breakdown.py` 对 cached trace 的 GPU kernel 归因显示，
`video_prefill_cache` 内 attention forward 只有约 `27ms/step`，GEMM 约 `1080ms/step`。
全 step 里 GEMM 约 `2091.5ms/step`，占 `75.5%`。

Top kernel 是：

```text
cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1
cutlass_75_tensorop_bf16_s1688gemm_bf16_256x128_nn_align1
cutlass_75_tensorop_bf16_s1688gemm_bf16_256x128_tn_align1
```

这些 `align1` GEMM 合计约 `1897ms/step`。H200 上 bf16 GEMM 正常应更多走 Hopper 原生
`nvjet_*` kernel；落到 SM75 时代 `s1688gemm` 且 `align1`，强烈指向某个操作数指针或
leading dimension 没有满足 16B 对齐。

H200 microbench 已复现机制：

| case | proj3072 | ffn14336 | kernel |
| --- | ---: | ---: | --- |
| 对齐，`ptr%16=0` | 1.715ms | 9.097ms | `nvjet_tst_*` |
| 错位 7 元素，`ptr%16=14` | ~14.2ms | ~64.0ms | `cutlass_75_tensorop_bf16_s1688gemm_*` |
| slowdown | 8.3x | 7.0x | |

当前最可能根因：

1. DeepSpeed ZeRO-1 参数 flatten 后某些 trainable 参数从 flat buffer 的非 16B 对齐偏移开始。
2. MoT/DiT 中间激活经过 slice/rearrange 后 data pointer 错位或 layout 不适合 `F.linear`。
3. cuBLAS/PyTorch 2.7.1+cu128 对这些 shape 的 heuristic 问题；可能性低于前两者。

## 建议接手顺序

1. 先跑参数/激活对齐探针
   - 目标：确认 align1 GEMM 是权重参数错位、激活错位，还是二者都有。
   - 已有开关：`FASTWAM_LOG_PARAM_ALIGNMENT=1` 和 `FASTWAM_LOG_ACT_ALIGNMENT=1`。
   - 日志前缀统一为 `[align-probe]`，可直接 `grep`。

2. 如果参数大量错位，优先排查 DeepSpeed ZeRO-1 flatten
   - 重点看 `accelerator.prepare()` 之后的 `model.named_parameters()`。
   - 若未对齐参数集中出现在 DiT Linear weight，先尝试调整 flatten/padding/param group。
   - 修复后用 `scripts/microbench_gemm_alignment.py` 和完整 short profile 回归。

3. 如果参数对齐而激活错位，继续定位进入 `nn.Linear` 前的 tensor view
   - 先看前 8 个 `nn.Linear` hook 输出。
   - 必要时扩大 `max_layers` 或只 hook MoT/DiT 目标模块。
   - 修复方向通常是局部 `.contiguous()` 或调整 `rearrange/slice` 的顺序；要用 profile 验证收益。

4. 暂停复杂 attention mask split
   - no-mask flash 上限只有约 2% forward 收益。
   - `attention_debug.*` 保留用于后续顺带观测，不作为当前主线。

5. GEMM 对齐修复后再做下一轮 profile
   - 复用 `scripts/analyze_trace_gpu_breakdown.py`。
   - 对比 `cutlass_75_*_align1` 是否下降，`nvjet_*` 是否上升。
   - 再看 `other` 类 elementwise/norm/copy/reduce 是否成为新瓶颈。

## 常用远端命令

查看 cached profiling 汇总：

```bash
cd /data/home/maxliu/projects/FastWAM
/tmp/summarize_latcache_profile_result.sh \
  runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618
```

检查 cache：

```bash
cd /data/home/maxliu/projects/FastWAM
find runs/vae_latent_cache/fold_clothv4_v4_wan22/5ff52f56f7112f87 -name "*.pt" | wc -l
du -sh runs/vae_latent_cache/fold_clothv4_v4_wan22
```

查看 W&B sync：

```bash
ssh h200-qinghua-jump 'tail -120 /home/maxliu/.local/state/wandb-sync/sync.log'
```

查看 attention 实验 summary：

```bash
cd /data/home/maxliu/projects/FastWAM

sed -n '1,80p' \
  runs/fold_clothv4_v4_2epoch/profile_trace_attnmask_fold_clothv4_v4_20260703_132422/profile/trace_summary.tsv

sed -n '1,80p' \
  runs/fold_clothv4_v4_2epoch/profile_trace_attn_nomask_fold_clothv4_v4_20260703_134334/profile/trace_summary.tsv
```

运行参数/激活对齐探针时，沿用当前 cached profiling 命令，只追加环境变量：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

# 按实际训练启动方式替换下面的 python scripts/train.py；重点是两个环境变量。
FASTWAM_LOG_PARAM_ALIGNMENT=1 \
FASTWAM_LOG_ACT_ALIGNMENT=1 \
python scripts/train.py task=fold_clothv4_v4_2epoch \
  profile.torch_enabled=false \
  profile.timing_enabled=false \
  max_steps=2
```

查看探针日志：

```bash
cd /data/home/maxliu/projects/FastWAM
grep -R "\\[align-probe\\]" runs/logs/*.log | tail -80
```

运行 GEMM 对齐 microbench：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

CUDA_VISIBLE_DEVICES=7 python scripts/microbench_gemm_alignment.py \
  --linear \
  --shape 72000,3072,3072 \
  --shape 72000,3072,14336 \
  --no-dgrad \
  --warmup 10 \
  --iters 50
```

对完整 trace 做 GPU kernel 归因：

```bash
cd /data/home/maxliu/projects/FastWAM
python scripts/analyze_trace_gpu_breakdown.py \
  runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618/profile/torch/lacy--214-30-239-40_3735196.1783026074570934851.pt.trace.json
```

## 当前不要做的事

- 不要删除或重建 `runs/vae_latent_cache/fold_clothv4_v4_wan22`，现在已经完整。
- 不要把 v4 worktree 里的旧 dirty docs 混入 VAE cache 提交。
- 不要把 `profile_timing_lines.txt` 为空误判为 profiling 失败；trace summary 是有效的。
- 不要强制 PyTorch flash backend 直接跑当前 mask，已验证会失败。
- 不要继续优先做复杂 attention mask split；当前证据显示收益远小于 GEMM 对齐。

## 后续实验 checklist

接手人继续时建议按这个顺序做，避免被表层 `video_prefill_cache` 名字带偏：

1. 保留当前 cached baseline
   - 固定对照 run：`profile_trace_latcache_fold_clothv4_v4_20260703_045618`。
   - 固定核心指标：`train/forward_loss`、`train/backward`、`model/v4/video_prefill_cache`、top GEMM kernel rows。
   - 不要换 batch size、profile window、latent cache 或 dataset stats。

2. 先看 `[align-probe] param_alignment`
   - 如果 `unaligned_params` 很大，直接做 DeepSpeed/参数 flatten 修复。
   - 如果参数基本对齐，再看 activation hook。

3. 再看 `[align-probe] act_alignment`
   - 关注 `input_ptr_mod16`、`input_contiguous`、`weight_ptr_mod16`。
   - 第一批只 hook 前 8 个 Linear；不够再扩大。

4. 每次修复都产出同格式 summary
   - `profile/trace_summary.tsv`
   - W&B offline run URL
   - 一张 baseline vs experiment 表：
     - `train/forward_loss`
     - `train/backward`
     - `model/v4/video_prefill_cache`
     - `cutlass_75_*_align1`
     - `nvjet_*`
     - `gemm/other/attn` GPU kernel breakdown

建议判断标准：

- 如果 align1 kernel 大幅减少，同时 `train/forward_loss` 和 `train/backward` 都下降，继续完善对齐修复。
- 如果 forward 降了但 backward 不降，下一步看 backward GEMM 和 activation checkpoint。
- 如果 align1 已消失但 step time 仍高，再转向 `other` 类 elementwise/norm/copy/reduce fusion。
