# VAE cache 预计算加速测试分支

## 目的

当前训练路径里的 VAE encode 加速只接在 `FastWAM._encode_video_latents()` 上；
离线 cache 预计算脚本直接加载裸 `wan_video_vae` 并调用 `vae.encode(...)`，
因此训练侧 VAE 加速不会自动作用到 `scripts/precompute_vae_latents.py`。

本分支目标是把 VAE cache 预计算路径变成可测试、可切换、易接入的形态：

- 保留旧 `wrapper` encode backend 作为默认行为；
- 新增 `model_batch` backend，直接调用 `vae.model.encode([B,C,T,H,W])`，避免 wrapper 内部逐样本循环；
- 新增预计算专用 `compile_encoder`、`inference_mode`、`cudnn_benchmark` 和 timing 日志；
- 新增等价性脚本，先确认 batch-native/compile 输出与旧 wrapper 输出一致或在容差内。

## 分支与位置

- 本地 worktree：`/Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/codex-vae-cache-speedtest`
- 分支：`codex-vae-cache-speedtest`
- base：`mem-stage-v4` 的干净 HEAD `442644b`
- 主工作树 `/Users/maxliu/MyProjects/AIR/202606/FastWAM_xyc` 有未提交 speedup 改动，本分支不依赖也不修改它们。

## 改动范围

1. `scripts/precompute_vae_latents.py`
   - 新增 `+vae_latent_cache.encode_backend=wrapper|model_batch`。
   - 新增 `+vae_latent_cache.compile_encoder=true` 和 `+vae_latent_cache.compile_mode=default|reduce-overhead`。
   - 新增 `+vae_latent_cache.inference_mode=true`。
   - 新增 `+vae_latent_cache.cudnn_benchmark=true`。
   - 新增 `+vae_latent_cache.pin_memory=true|false`、`prefetch_factor`、`persistent_workers`、`in_order`。
   - 新增 `+vae_latent_cache.warmup_batches=N`，用于把 `torch.compile`/cuDNN 首批成本放在主循环计时外。
   - 新增 `+vae_latent_cache.log_timing=true` 与 `+vae_latent_cache.sync_timing=true`，输出 data_wait / h2d / encode / save / wall 时间。
   - 新增实验参数 `+vae_latent_cache.ram_preload=true`，用于把当前 bounded shard 的 video/history 先转成 CPU bf16 tensor 放入 RAM，再单独测 VAE encode 阶段。
   - 明确把仓库根目录和 `src/` 放入 `sys.path`，避免远端运行时误用环境里已安装的旧 `fastwam` 包。

2. `scripts/verify_vae_cache_encode_equivalence.py`
   - 比较旧 wrapper 与新 `model_batch` backend。
   - 可选比较 compiled `model_batch`。
   - 支持 `verify_source=random|dataset`。

3. `src/fastwam/datasets/lerobot/robot_video_dataset.py`
   - 新增 `vae_latent_cache_precompute_only`，VAE cache 预计算时只返回 `sample_idx`、`video`、`history_video`。
   - 训练路径默认值为 `false`，不改变正常训练 dataset 输出。

## 训练阶段默认接入策略

当前分支把加速和 profiling 开关全部收在 `scripts/precompute_vae_latents.py` 的 `vae_latent_cache.*` 参数内；
仓库现有 `configs/` 中没有默认配置 `data.train.vae_latent_cache_dir`，因此正常 training 默认仍走原始视频输入和在线 VAE encode，不会因为本分支自动启用 cache。

训练阶段是否默认接入 VAE cache 仍需商榷，原因如下：

- cache fingerprint 与 `dataset_dirs`、video/history frame index、video size、VAE model id 等强绑定；正式训练必须使用与 cache 生成完全一致的 combined dataset 配置。
- `model_batch` 与 `compile_encoder` 在 bf16 下与旧 wrapper 不是 bitwise identical，宽容差通过但仍需 downstream 短训/eval 接受。
- cache 命中后训练侧 I/O 模式会改变：会减少 VAE compute，但增加小文件读取与 metadata 校验，是否净收益需要用同一训练配置实测。

建议接入门槛：

1. 先完成 5 个目标 fold-cloth dataset 的 combined cache 生成。
2. 用 `data.train.vae_latent_cache_dir=<cache_dir>` 显式开启一段短训/eval，对比 loss 曲线、吞吐和显存。
3. 确认收益稳定且数值行为可接受后，再考虑把某个 pretrain 配置默认指向 cache；在此之前不建议默认开启。

## 远端验证位置

运行位置：

```bash
cd /data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
```

### 1. 随机输入等价性

严格容差用于发现非 bitwise/近似不一致，当前 `model_batch` 和 `compile_encoder` 在 bf16 下会触发严格失败。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/verify_vae_cache_encode_equivalence.py \
  task=fold_clothv4_v4_2epoch \
  model.redirect_common_files=false \
  +vae_latent_cache.verify_source=random \
  +vae_latent_cache.verify_batch_size=4 \
  +vae_latent_cache.compile_encoder=true \
  +vae_latent_cache.compile_mode=default
```

当前严格容差结论：

```text
wrapper_vs_model_batch max_abs ~= 0.03125, rel ~= 0.007
wrapper_vs_compiled_wrapper max_abs ~= 0.046875, rel ~= 0.0105
```

因此 `model_batch`/`compile_encoder` 应视为加速候选，需要用 downstream eval 接受数值偏差；默认 backend 保持 `wrapper`。

### 2. 真实 dataset batch 等价性

复用已有 stats，避免验证脚本重新全量扫描 norm stats：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/verify_vae_cache_encode_equivalence.py \
  task=fold_clothv4_v4_2epoch \
  model.redirect_common_files=false \
  data.train.pretrained_norm_stats=/data/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/dataset_stats.json \
  +vae_latent_cache.verify_source=dataset \
  +vae_latent_cache.verify_batch_size=4 \
  +vae_latent_cache.verify_num_workers=4 \
  +vae_latent_cache.verify_abs_tol=0.1 \
  +vae_latent_cache.verify_rel_tol=0.02 \
  +vae_latent_cache.compile_encoder=true \
  +vae_latent_cache.compile_mode=default
```

宽容差通过，主要结果：

| 对比 | max_abs | rel | mean_abs | cosine |
| --- | ---: | ---: | ---: | ---: |
| current wrapper vs model_batch | 0.03125 | 0.00699 | 0.00219 | 0.99999291 |
| history wrapper vs model_batch | 0.03125 | 0.00714 | 0.00269 | 0.99999052 |
| current wrapper vs compiled wrapper | 0.046875 | 0.01049 | 0.00498 | 0.99997824 |
| history wrapper vs compiled wrapper | 0.046875 | 0.01071 | 0.00493 | 0.99997765 |

### 3. 预计算吞吐 benchmark

使用临时输出目录，不碰已有完整 cache：

```bash
export CACHE_BENCH=/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/$(date +%Y%m%d_%H%M%S)
export STATS=/data/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/dataset_stats.json

CUDA_VISIBLE_DEVICES=1 python scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  model.redirect_common_files=false \
  data.train.pretrained_norm_stats=${STATS} \
  +vae_latent_cache.output_dir=${CACHE_BENCH}/wrapper \
  +vae_latent_cache.work_dir=${CACHE_BENCH}/work_wrapper \
  +vae_latent_cache.max_samples=512 \
  +vae_latent_cache.batch_size=8 \
  +vae_latent_cache.num_workers=16 \
  +vae_latent_cache.prefetch_factor=2 \
  +vae_latent_cache.persistent_workers=true \
  +vae_latent_cache.encode_backend=wrapper \
  +vae_latent_cache.inference_mode=true \
  +vae_latent_cache.log_timing=true \
  +vae_latent_cache.sync_timing=true \
  +vae_latent_cache.overwrite=false

CUDA_VISIBLE_DEVICES=1 python scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  model.redirect_common_files=false \
  data.train.pretrained_norm_stats=${STATS} \
  +vae_latent_cache.output_dir=${CACHE_BENCH}/model_batch_compile_warm \
  +vae_latent_cache.work_dir=${CACHE_BENCH}/work_model_batch_compile_warm \
  +vae_latent_cache.max_samples=512 \
  +vae_latent_cache.batch_size=8 \
  +vae_latent_cache.num_workers=16 \
  +vae_latent_cache.prefetch_factor=4 \
  +vae_latent_cache.persistent_workers=true \
  +vae_latent_cache.encode_backend=model_batch \
  +vae_latent_cache.compile_encoder=true \
  +vae_latent_cache.compile_mode=default \
  +vae_latent_cache.warmup_batches=2 \
  +vae_latent_cache.inference_mode=true \
  +vae_latent_cache.log_timing=true \
  +vae_latent_cache.sync_timing=true \
  +vae_latent_cache.overwrite=false
```

#### 实测结果

H200-1，`CUDA_VISIBLE_DEVICES=1`，临时输出目录：

- 旧基准：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/20260704_152355`
- worker grid：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/worker_grid_20260704_153238`
- precompute-only grid：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/precompute_only_20260704_154518`

| case | batch | workers | prefetch | compile | warmup | data_wait | encode | save | wall | samples/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wrapper old | 4 | 4 | default | no | 0 | - | 24.13s | 0.40s | 61.00s | 8.39 |
| wrapper worker grid | 8 | 16 | default | no | 0 | - | 24.57s | 0.40s | 32.26s | 15.87 |
| model_batch worker grid | 8 | 16 | default | no | 0 | - | 14.56s | 0.44s | 22.64s | 22.61 |
| wrapper precompute-only | 8 | 16 | 2 | no | 0 | 6.48s | 24.31s | 0.37s | 31.95s | 16.03 |
| model_batch precompute-only | 8 | 16 | 2 | no | 0 | 7.24s | 14.07s | 0.43s | 22.43s | 22.82 |
| model_batch precompute-only | 8 | 16 | 4 | no | 0 | 7.50s | 14.08s | 0.38s | 22.60s | 22.65 |
| model_batch compile warm | 8 | 16 | 4 | yes | 2 | 10.77s | 9.47s | 0.36s | 21.24s | 24.10 |

### 4. 2048 samples worker 与 RAM preload 实验

这组实验使用 `model_batch + compile_encoder + warmup_batches=2`，并设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

目的不是只看单卡峰值，而是判断 loader producer 和 VAE consumer 的速度关系。

运行目录：

- worker scale：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/worker_scale_2048_20260704_175056`
- RAM preload nw16：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/ram_preload_2048_20260704_175214`
- RAM preload nw24：`/data/home/maxliu/projects/FastWAM_worktrees/codex-vae-cache-speedtest/runs/vae_latent_cache_bench/ram_preload_nw24_2048_20260704_175947`

#### Streaming DataLoader

| workers | data_wait | encode | active_total | wall | samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 87.41s | 38.48s | 40.65s | 128.34s | 15.96 |
| 16 | 29.16s | 38.63s | 40.96s | 70.60s | 29.01 |
| 24 | 10.45s | 38.72s | 41.05s | 52.15s | 39.27 |
| 32 | 9.99s | 38.84s | 41.23s | 52.04s | 39.36 |

#### RAM preload

| workers | preload | preload samples/s | encode phase wall | encode phase samples/s | ideal double-buffer samples/s | serial total samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 70.78s | 28.94 | 40.50s | 50.57 | 28.94 | 18.41 |
| 24 | 52.37s | 39.11 | 40.74s | 50.27 | 39.11 | 22.01 |

关键结论：

- `num_workers=16` 是第一层大收益；`batch_size=8` 主要服务 `model_batch`，对 wrapper 不明显。
- `model_batch` 将 VAE encode 从约 `24s/512` 降到约 `14s/512`。
- `compile_encoder + warmup` 将 encode 进一步降到约 `9.5s/512`，但 data wait 上升，端到端只从 `22.43s/512` 到 `21.24s/512`。
- `prefetch_factor=4` 没有收益，当前建议保持 `prefetch_factor=2`。
- `save` 只有约 `0.36-0.43s/512`，当前不是瓶颈；异步 writer 可以后置。
- `torch.compile` warmup 约 `41.6s`。百万级 cache 可以摊平，但必须使用长生命周期进程，避免按小 shard 频繁重启。
- 2048 samples 下，`num_workers=8` 明显 loader-bound；`16` 接近平衡；`24` 把瓶颈推回 VAE active；`32` 相比 `24` 几乎没有收益。
- RAM 完全够放 bounded chunk，例如 2048 samples 的 current+history bf16 约 `19.69GiB`。但串行 RAM preload 会更慢；即使理想双缓冲，上界也等于 producer 速度，`24 workers` 时约 `39.1 samples/s`，和 streaming DataLoader 的 `39.3 samples/s` 基本一致。
- 因此真正有价值的 chunk queue 不是“先全量塞进 RAM”，而是 bounded async producer-consumer：用于平滑 I/O 抖动、限制 pinned memory、暴露 queue depth/backpressure，并在多 GPU 时做 NUMA/CPU 预算。

## 时长估计

按图中总量 `8,361,052` 近似为本脚本 sample 数估算：

| 配置 | 单 GPU | 8 GPU shard | 备注 |
| --- | ---: | ---: | --- |
| wrapper + 16 workers | 约 145h | 约 18.1h | 严格最稳，但慢 |
| model_batch + 16 workers | 约 101.8h | 约 12.7h | 当前最推荐的非 compile 方案 |
| model_batch + compile warm + 16 workers | 约 80.1h | 约 10.0h | 2048 samples 复测，较 512 估计更接近长跑 |
| model_batch + compile warm + 24 workers | 约 59.1h | 约 7.4h | 单卡最优附近；8 卡需 NUMA/CPU 绑定后再验证 |

考虑共享盘抖动、尾批、失败重试和多 GPU 争抢，正式跑 5 个 fold-cloth dataset 建议按 `8 GPU 约 8-12 小时` 预留。若先用保守 `16 workers/GPU`，按 `约 10-14 小时` 预留更稳。

## 正式生成建议

建议分两档：

1. 稳妥档：`encode_backend=model_batch`、`batch_size=8`、`num_workers=16`、`prefetch_factor=2`、`persistent_workers=true`、`inference_mode=true`。
2. 激进档：`encode_backend=model_batch`、`compile_encoder=true`、`warmup_batches=2`、`num_workers=24`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，先跑一小段训练/eval 确认 latent 偏差可接受。

多 GPU 时避免 loader 和 encode 互相抢资源：

- 每张 GPU 一个长生命周期 rank，不按小 episode 或小文件夹频繁重启。
- GPU0-3 绑 NUMA node0，GPU4-7 绑 NUMA node1；`24 workers/GPU` 在 8 卡时已经接近 192 CPU 的上限，不应继续加到 `32 workers/GPU`。
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，防止每个 DataLoader worker 再开内部线程。
- 大 RAM chunk 只作为 bounded queue 使用，不要全量常驻；queue depth 建议先从 `2` 开始。
- pinned memory 只 pin 小的 batch/ring buffer，不 pin 巨型全量 RAM chunk。
- loader 保持 CPU-only decode/resize/normalize，VAE encode 独占 GPU；除非后续 profile 证明 GPU transform 有净收益，否则不要把预处理也搬到 GPU。

激进档正式命令示例：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  model.redirect_common_files=false \
  data.train.pretrained_norm_stats=/data/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/dataset_stats.json \
  +vae_latent_cache.output_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_cloth_combined_5sets_wan22 \
  +vae_latent_cache.batch_size=8 \
  +vae_latent_cache.num_workers=24 \
  +vae_latent_cache.prefetch_factor=2 \
  +vae_latent_cache.persistent_workers=true \
  +vae_latent_cache.encode_backend=model_batch \
  +vae_latent_cache.compile_encoder=true \
  +vae_latent_cache.compile_mode=default \
  +vae_latent_cache.warmup_batches=2 \
  +vae_latent_cache.inference_mode=true \
  +vae_latent_cache.log_timing=true \
  +vae_latent_cache.sync_timing=false \
  +vae_latent_cache.overwrite=false
```

正式 benchmark 可保留 `sync_timing=true`；正式全量生成建议 `sync_timing=false`，减少人为同步。

## 后续接 combined dataset cache

当前 cache fingerprint 包含完整 `dataset_dirs` 列表。若后续训练使用 5 个 fold-cloth dataset 的 combined list，
必须用最终 combined 配置生成 cache；单独 v4 cache 的 fingerprint/sample_idx 空间不能直接复用。

本分支的预计算加速保持在 `scripts/precompute_vae_latents.py` 内部，因此后续只需要把最终 `data.train.dataset_dirs`
覆盖成 5 个目标目录，再复用同一组 `vae_latent_cache.*` 参数即可。

## 当前状态

- 本地已改代码和文档，尚未提交。
- 本地验证通过：`python -m py_compile scripts/precompute_vae_latents.py scripts/verify_vae_cache_encode_equivalence.py src/fastwam/datasets/lerobot/robot_video_dataset.py`。
- 本地验证通过：`git diff --check -- scripts/precompute_vae_latents.py scripts/verify_vae_cache_encode_equivalence.py src/fastwam/datasets/lerobot/robot_video_dataset.py docs/26-07-04/26-07-04-vae-cache-speedtest-branch.md`。
- 远端验证通过：同三文件 `py_compile`。
- 远端 H200 已完成等价性与吞吐 benchmark，详见上方结果。
