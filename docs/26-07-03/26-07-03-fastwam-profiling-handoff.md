# FastWAM profiling 交接文档

更新时间：2026-07-03 13:04 CST

## 交接范围

当前目标仍是：

1. 在 `mem-stage-v4` 上实现并验证 VAE latent cache，加速训练 forward。
2. 同步到 `profiling-mem-stage-v4`，做远端 profiling 对比。
3. 根据 profiling 证据继续测试 attention backend。
4. 之后再看 `video_prefill/MoT` 结构优化。
5. 最后补推理侧加速。

当前已经完成第 1、2 步；第 3 步只做了初步 backend 探针，还没有提交代码或启动正式 attention 实验。

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
HEAD: 8bba6e8 docs: 记录 VAE latent cache smoke 结果
```

当前有未提交改动：

```text
M configs/model/fastwam.yaml
M docs/26-07-03/26-07-03-fastwam-profiling-handoff.md
M src/fastwam/models/wan22/fastwam.py
M src/fastwam/models/wan22/mot.py
M src/fastwam/models/wan22/wan_video_dit.py
M src/fastwam/runtime.py
```

其中代码/config 改动是 attention backend 最小实验开关；交接文档记录当前状态和运行口径。

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

这不是代码路径验证失败；需要在 H200 的 FastWAM conda env 中补跑配置解析或直接跑短 profiling。

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
remote HEAD at check time: 8bba6e8, plus local hotfix changes to scripts/precompute_vae_latents.py
```

H200 无法解析 GitHub：

```text
fatal: unable to access 'https://github.com/Frankaaay/FastWAM.git/': Could not resolve host: github.com
```

之前用 git bundle 同步过提交。若接手人需要继续同步本地提交到 H200，可以继续用 bundle/scp/fetch 的方式。

远端当前 `git status`：

```text
## profiling-mem-stage-v4...origin/profiling-mem-stage-v4 [ahead 4]
 M scripts/precompute_vae_latents.py
?? data
```

`scripts/precompute_vae_latents.py` 是手动 scp 过去的未提交热修版本；`data` 是远端已有未跟踪目录，不要删除。

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
- 下一阶段瓶颈是 `model/v4/video_prefill_cache` 和 attention backend。

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

## Attention backend 实验开关已实现

Profiling worktree 当前已实现默认关闭的实验开关：

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

建议第一轮远端实验只打开：

```text
model.attention_debug.log_sdpa_backend=true
```

用于确认当前 masked path 仍不可走 flash。第二轮再跑上限实验：

```text
model.attention_debug.log_sdpa_backend=true
model.attention_debug.force_video_prefill_no_mask=true
model.attention_debug.disable_history_condition_dropout=true
```

这会改变训练语义，只能作为速度上限，不可直接当最终训练方案。

## Mask 结构与下一步判断

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

可行方向：

1. 做一个 no-mask / flash-only 对照实验
   - 目的：测理论上限，不作为最终训练语义。
   - 方法：临时将 `video_attention_mask_mode=bidirectional`，并且避免 key-valid mask 进入 `prefill_video_cache`。
   - 风险：改变训练语义，只能看速度上限。

2. 拆分 `first_frame_causal` video attention
   - prefix query 部分只 attend prefix keys。
   - future query 部分 attend full keys。
   - 两次 `scaled_dot_product_attention(..., attn_mask=None)`，理论上可走 flash。
   - 需要把输出按 query 拼回去。
   - 还要处理 padding/history dropout；如果 per-sample key-valid 继续存在，仍会需要 mask 或分组。

3. 优先禁用 history condition dropout 做速度实验
   - 当前 `FastWAM.HISTORY_CONDITION_DROPOUT = 0.2`，训练时会产生 batch-level key-valid mask。
   - 如果设为 0，且数据 padding 对当前 batch 不影响，video prefill 更容易走无 mask 拆分。
   - 这是实验口径，不一定是最终训练策略。

4. 接入外部 varlen/block-sparse attention
   - 可能更贴合 first-frame mask 和 padding。
   - 风险和改动量更大，应在 1/2 的证据出来后再做。

## 建议接手顺序

1. 先提交 profiling worktree 的 attention debug 实验开关
   - VAE cache 的核心代码和 smoke 文档已经在两个本地分支 HEAD 中。
   - 当前 profiling worktree 本轮相关文件：

```bash
git add \
  configs/model/fastwam.yaml \
  src/fastwam/runtime.py \
  src/fastwam/models/wan22/wan_video_dit.py \
  src/fastwam/models/wan22/mot.py \
  src/fastwam/models/wan22/fastwam.py \
  docs/26-07-03/26-07-03-fastwam-profiling-handoff.md
```

2. 同步 H200 前先处理远端热修状态
   - H200 不能直接 fetch GitHub，继续用 bundle 或 scp。
   - 注意远端现在有 `scripts/precompute_vae_latents.py` 热修未提交，正式同步时不要误回退。

3. Attention backend 最小实验
   - 先不要重构 MoT。
   - 第一轮只开 `model.attention_debug.log_sdpa_backend=true`，确认 masked path 的 SDPA backend 判断。
   - 第二轮开 `force_video_prefill_no_mask=true` 和 `disable_history_condition_dropout=true`，跑 no-mask 上限 profiling。

4. 如果 no-mask flash 对 `video_prefill_cache` 有明显收益，再做 mask 拆分版本。

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

Attention backend 诊断 run 建议 overrides：

```bash
cd /data/home/maxliu/projects/FastWAM

# masked path，只观察 SDPA backend
python scripts/train.py \
  profile.torch_enabled=true \
  profile.timing_enabled=true \
  model.attention_debug.log_sdpa_backend=true

# no-mask 上限实验；会改变训练语义，只看速度上限
python scripts/train.py \
  profile.torch_enabled=true \
  profile.timing_enabled=true \
  model.attention_debug.log_sdpa_backend=true \
  model.attention_debug.force_video_prefill_no_mask=true \
  model.attention_debug.disable_history_condition_dropout=true
```

实际运行时沿用之前 cached profiling run 的完整训练命令、latent cache 配置、batch size、profile window 和 output naming，只追加上面的 `model.attention_debug.*` overrides。

## 当前不要做的事

- 不要删除或重建 `runs/vae_latent_cache/fold_clothv4_v4_wan22`，现在已经完整。
- 不要把 v4 worktree 里的旧 dirty docs 混入 VAE cache 提交。
- 不要把 `profile_timing_lines.txt` 为空误判为 profiling 失败；trace summary 是有效的。
- 不要强制 PyTorch flash backend 直接跑当前 mask，已验证会失败。

## Attention backend 最小实验 checklist

接手人如果继续第 3 步，建议按这个顺序做，避免一上来重构 MoT：

1. 保留当前 cached baseline
   - 固定对照 run：`profile_trace_latcache_fold_clothv4_v4_20260703_045618`。
   - 固定核心指标：`train/forward_loss`、`train/backward`、`model/v4/video_prefill_cache`、SDPA kernel rows。
   - 不要换 batch size、profile window、latent cache 或 dataset stats。

2. 增加 SDPA backend 观测开关
   - 目标：训练日志里明确打印当前 PyTorch 是否能用 flash/efficient。
   - 位置建议：`src/fastwam/models/wan22/wan_video_dit.py::flash_attention()`。
   - 只在 rank0 / 前几个 step 打印，避免污染日志。
   - 用 `torch.nn.attention.SDPAParams`、`can_use_flash_attention`、`can_use_efficient_attention` 做判断。

3. 做 no-mask 上限实验
   - 目标：测 `video_prefill_cache` 如果能走 flash 的理论收益。
   - 方法 A：临时把 `model.video_dit_config.video_attention_mask_mode=bidirectional`。
   - 方法 B：额外临时跳过 `video_key_valid_mask`，否则 `_apply_key_valid_mask()` 仍会创建 `[B,1,Q,K]` mask。
   - 判断：trace 中应出现 flash attention kernel；`aten::_scaled_dot_product_efficient_attention` 应减少或消失。
   - 注意：这会改变训练语义，只作为速度上限，不可直接作为最终训练方案。

4. 如果 no-mask 明显更快，再拆 `first_frame_causal`
   - prefix queries: `q[:prefix]` attend `k[:prefix]`，无 mask。
   - future queries: `q[prefix:]` attend `k[:]`，无 mask。
   - 两次 SDPA 输出拼回 `[B,S,H*Dh]`。
   - 这个拆法只覆盖结构 mask；padding/history dropout 仍需单独处理。

5. 处理 padding / history dropout
   - 当前训练有 `FastWAM.HISTORY_CONDITION_DROPOUT = 0.2`，会引入 per-sample key-valid mask。
   - 第一版实验可以把 dropout 设为 0，看拆 mask 是否能跑通并加速。
   - 若必须保留 dropout，需要按样本分组、varlen attention 或 block-sparse attention，而不是直接传 bool mask。

6. 每次实验都产出同格式 summary
   - `profile/trace_summary.tsv`
   - W&B offline run URL
   - 一张 baseline vs experiment 表：
     - `train/forward_loss`
     - `train/backward`
     - `model/v4/video_prefill_cache`
     - `aten::_scaled_dot_product_flash_attention*`
     - `aten::_scaled_dot_product_efficient_attention*`

建议判断标准：

- 如果 no-mask flash 只带来很小收益，先不要做复杂 mask 拆分。
- 如果 `video_prefill_cache` 明显下降，再实现语义等价的 `first_frame_causal` 拆分。
- 如果 forward 降了但 backward 不降，下一步要看 activation checkpoint 和 attention backward，而不是继续只优化 forward。
