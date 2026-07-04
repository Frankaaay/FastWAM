# VAE Cache RAM Chunk Prefetch

## 目的

在 `mem-stage-v4` 基础上试验 VAE latent cache 预计算/训练读取时的数据等待优化：不改变 VAE encode 数值路径，不改变模型结构，只在数据侧增加 opt-in 的 RAM batch 预取队列，让 CPU/DataLoader 提前把后续 chunk/batch 放入内存，主线程在下一步 VAE encode 前尽量少等 I/O/解码。

## 分支与位置

- 分支：`feature/vae-ram-chunk-prefetch`
- 基线：`mem-stage-v4` commit `442644b`
- 本地 worktree：`/Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/feature-vae-ram-chunk-prefetch`
- 远端建议路径：`/data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch`

## 改动

- 新增 `src/fastwam/utils/async_prefetch.py`：通用有界后台线程预取器 `AsyncBatchPrefetcher`，用 RAM 队列缓存已取出的 batch。
- `scripts/precompute_vae_latents.py`：
  - 支持 `vae_latent_cache.prefetch_factor`、`persistent_workers`、`pin_memory`。
  - 支持 `vae_latent_cache.ram_prefetch_batches`，在 VAE load 前启动 DataLoader iterator 和 RAM 队列，尽量让 chunk/batch 在进入 VAE encode loop 前已经进入 RAM。
  - 支持 `vae_latent_cache.ram_prefetch_to_dtype`，后台线程可先把 `video/history_video` 转成训练 dtype，减少主线程 CPU 转换。
  - 支持 `vae_latent_cache.log_timing` / `sync_timing` 记录 `data_wait/h2d/encode/save`，用于判断 RAM 预取是否真的压低等待。
- `RobotVideoDataset` 增加 `vae_latent_cache_precompute_only`：VAE cache 预计算只返回 `sample_idx/video/history_video`，跳过 action/proprio/text context 构造。
- `Wan22Trainer`：
  - 增加训练 DataLoader 参数 `pin_memory/prefetch_factor/persistent_workers/ram_prefetch_batches`。
  - `ram_prefetch_batches > 0` 时用同一个 RAM 队列异步拉取训练 batch。
- `configs/train.yaml` 增加默认配置，默认 `ram_prefetch_batches: 0`，旧训练行为保持关闭。

## 本地验证

运行位置：

```bash
cd /Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/feature-vae-ram-chunk-prefetch
```

命令：

```bash
python -m py_compile \
  src/fastwam/utils/async_prefetch.py \
  src/fastwam/datasets/lerobot/robot_video_dataset.py \
  scripts/precompute_vae_latents.py \
  src/fastwam/trainer.py
```

结果：通过，无语法错误。

## 远端 smoke 记录

运行位置：

```bash
cd /data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch
```

第一次远端运行在进入预计算前失败：

```text
ModuleNotFoundError: No module named 'fastwam.utils.async_prefetch'
```

原因：远端 `fastwam` conda 环境中已有安装版 `fastwam` 包，`scripts/precompute_vae_latents.py` 没有优先插入当前 checkout 的 `src/`，导致新加的 `fastwam.utils.async_prefetch` 不可见。

修复：在脚本入口加入 `REPO_ROOT` / `SRC_ROOT` 到 `sys.path`，与旧测速分支的脚本入口保持一致。

第二次远端运行失败在 VAE 权重解析：

```text
ValueError: Cannot detect model type for wan_video_vae. File: [].
```

原因：远端 conda 环境里的已安装 `fastwam` 指向 `/data/home/maxliu/projects/FastWAM/src/fastwam`，旧任务实际使用主 checkout 的 `./checkpoints`。feature worktree 优先导入本 checkout 后，默认 `./checkpoints` 为空。后续远端测试显式设置：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/data/home/maxliu/projects/FastWAM/checkpoints
```

## 远端 A/B 结果

运行机器：`h200-qinghua-1`

运行分支/路径：

- 分支：`feature/vae-ram-chunk-prefetch`
- 远端 worktree：`/data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch`
- 代码 commit：`16c5f4c` 加本地未提交 early prefetch 调整

资源与约束：

- 使用 `CUDA_VISIBLE_DEVICES=0,5,6,7`，`torchrun --nproc_per_node=4`。
- GPU0 有 root 的 idle `picpp server` 占约 12GB，当前用户无法免密 sudo 清理；H200 显存余量充足，本次测试共享 GPU0。
- 旧 VAE cache 全量任务仍在 GPU1-4 上运行，本次没有中断。

共同参数：

- dataset：`/data/shared/datasets/fold_cloth_fastwam_lerobot/fold_clothv3_240x320_fastwam`
- `max_samples=512`，4 rank 后 rank0 处理 128 samples / 16 batches
- `batch_size=8`，`num_workers=4`，`prefetch_factor=2`，`persistent_workers=true`
- `compile_encoder=false`，`warmup_batches=1`，`encode_backend=model_batch`
- `log_timing=true`，`sync_timing=true`，`overwrite=true`

结果摘要（rank0 日志）：

| 配置 | data_wait | encode | wall | samples/s | data_wait/batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ram_prefetch_batches=0` | 24.062s | 7.407s | 32.339s | 3.958 | 1503.846ms |
| loop 内启动 `ram_prefetch_batches=4` | 26.149s | 7.373s | 34.409s | 3.720 | 1634.301ms |
| VAE load 前启动 `ram_prefetch_batches=16` | 6.987s | 6.487s | 13.648s | 9.379 | 436.685ms |

结论：

- 只在 encode loop 内启动 RAM prefetch 没有解决首批 batch 等待，甚至略慢；这个版本不是有效方向。
- 把 RAM prefetch 提前到 VAE load 之前启动后，dataset/DataLoader 的等待可以和 VAE 权重加载、warmup 重叠。当前 512-sample smoke 中，rank0 loop wall 从 32.339s 降到 13.648s，吞吐约提升 2.37x。
- 这验证了“异构地先把 chunk 放 RAM，再让 VAE encode 读取”的方向是可行的；真正全量收益还需要用更长 shard、不同 `ram_prefetch_batches` 和主机 RAM 监控确认。

## 远端 A/B 建议

预计算先做小样本 A/B，不要一上来跑全量：

```bash
cd /data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch

git fetch origin
git checkout feature/vae-ram-chunk-prefetch
git pull origin feature/vae-ram-chunk-prefetch

python scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_ram_prefetch_smoke \
  +vae_latent_cache.work_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/ram_prefetch_smoke \
  +vae_latent_cache.max_samples=128 \
  +vae_latent_cache.batch_size=8 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.prefetch_factor=2 \
  +vae_latent_cache.persistent_workers=true \
  +vae_latent_cache.ram_prefetch_batches=0 \
  +vae_latent_cache.log_timing=true \
  +vae_latent_cache.overwrite=true

python scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_ram_prefetch_smoke \
  +vae_latent_cache.work_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/ram_prefetch_smoke_on \
  +vae_latent_cache.max_samples=128 \
  +vae_latent_cache.batch_size=8 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.prefetch_factor=2 \
  +vae_latent_cache.persistent_workers=true \
  +vae_latent_cache.ram_prefetch_batches=16 \
  +vae_latent_cache.ram_prefetch_to_dtype=true \
  +vae_latent_cache.log_timing=true \
  +vae_latent_cache.overwrite=true
```

预期输出：

- 两组都完成 `Finished VAE latent precompute`。
- 第二组日志中的 `data_wait_ms_per_batch` 应低于第一组；如果不下降，说明瓶颈不在主线程等待，而可能在单样本视频 decode、保存 `.pt` 或 VAE encode。

训练侧小步 smoke：

```bash
cd /data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch

python scripts/train.py \
  task=fold_clothv4_v4_2epoch \
  max_steps=20 \
  eval_every=0 \
  save_every=20 \
  num_workers=4 \
  prefetch_factor=2 \
  persistent_workers=true \
  ram_prefetch_batches=4
```

## 当前结论

这是一个安全的第一层试验：它复用 DataLoader/worker 的读取机制，只增加有界 RAM 队列来重叠 CPU 数据准备与 VAE load/encode。远端 smoke 已确认早启动 RAM prefetch 对当前卡顿有效，但它仍是 batch 级缓存，不保证解决所有卡顿；如果全量运行中 `data_wait_ms_per_batch` 仍高，下一步应改成 episode/chunk 级缓存，避免相邻窗口反复 decode 同一段视频。

## 下一步

1. 在远端用更长 shard 做 `ram_prefetch_batches=0/8/16/32` A/B。
2. 对比 `data_wait_ms_per_batch`、wall samples/s、显存和主机 RAM。
3. 如果收益有限，再把缓存粒度从 batch 提升到 episode/chunk，优先复用相邻窗口共享的视频帧。
