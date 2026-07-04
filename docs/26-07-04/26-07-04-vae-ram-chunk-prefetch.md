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

补充长一点的生产参数 smoke：

- 命令差异：`max_samples=8192`，`compile_encoder=true`，`compile_mode=default`，`sync_timing=false`，其它保持 `batch_size=8`、4 rank、`ram_prefetch_batches=16`。
- 运行环境：仍与旧全量 VAE cache 任务并发，旧任务占用 GPU1-4 且持续读写 cache；本次使用 GPU0/5/6/7，CPU/DataLoader/I/O 会和旧任务竞争。
- rank0 结果：`samples=2048`，`batches=256`，`data_wait=328.544s`，`encode=104.727s`，`save=2.470s`，`wall=435.862s`，`wall_samples_per_s=4.699`，`data_wait_ms_per_batch=1283.376`。
- 4 rank 全局折算：约 `18.8 samples/s`，`0.587 steps/s`（这里 step 指每个 rank 各处理一个 batch 的 VAE cache 预计算 step，`batch_size=8`）。

预计算路径结论：

- 只在 encode loop 内启动 RAM prefetch 没有解决首批 batch 等待，甚至略慢；这个版本不是有效方向。
- 把 RAM prefetch 提前到 VAE load 之前启动后，dataset/DataLoader 的等待可以和 VAE 权重加载、warmup 重叠。当前 512-sample smoke 中，rank0 loop wall 从 32.339s 降到 13.648s，吞吐约提升 2.37x。
- 这只验证了 VAE cache 预计算路径里“异构地先把 chunk 放 RAM，再让 VAE encode 读取”的方向是可行的；它不是训练吞吐结论。8192-sample smoke 显示，在同机已有全量 cache 任务并发时，长期速度主要被 `data_wait` 压住，batch 级 RAM prefetch 仍不够，需要继续提升到 episode/chunk 级缓存或减少保存/读取竞争。

## 远端训练内 8 卡 A/B 结果

运行机器：`h200-qinghua-1`

运行分支/路径：

- 分支：`feature/vae-ram-chunk-prefetch`
- 远端测试 worktree：`/data/home/maxliu/projects/FastWAM_worktrees/feature-vae-ram-chunk-prefetch-train8`
- 代码 commit：`9f208d7`
- 测试产物目录：`/data/home/maxliu/projects/FastWAM_train_bench_20260704`

共同参数：

- 入口：`bash scripts/train_fold_clothv4_v4_2epoch.sh`
- 8 卡：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `max_steps=80`，`log_every=10`，`eval_every=0`，`save_every=0`，`wandb.enabled=false`
- `batch_size=16` 每卡，global batch size = 128 samples/step
- `num_workers=8`，`pin_memory=true`，`persistent_workers=true`，`prefetch_factor=4`
- 显式设置 `PYTHONPATH=$PWD/src:$PWD`，避免远端 conda 环境导入主仓库旧版 `fastwam`
- 显式设置：
  - `DIFFSYNTH_MODEL_BASE_PATH=/data/home/maxliu/projects/FastWAM/checkpoints`
  - `model.action_dit_pretrained_path=/data/home/maxliu/projects/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
  - `data.train.pretrained_norm_stats=/data/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/dataset_stats.json`

sanity 记录：

- 直接在 detached feature worktree 里跑训练会找不到相对路径 `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`；已用主仓库 checkpoint 绝对路径覆盖。
- `max_steps=2` 的 8 卡 sanity 完成真实 forward/backward/optimizer step，并确认日志来自 feature worktree 的 `src/fastwam/trainer.py`。
- `accelerate`/`tee` 在 `max_steps reached` 后存在父进程退出不干净的问题，但 GPU 已释放；测试后按 run id 清理父进程。

结果摘要：

| 配置 | step80 累计 step/s | step80 累计 samples/s | step60-80 区间 step/s | step60-80 区间 samples/s | 日志 |
| --- | ---: | ---: | ---: | ---: | --- |
| tuned DataLoader，`ram_prefetch_batches=0` | 0.31 | 40.32 | 0.351 | 44.9 | `logs/bench_train8_noram_tuned_20260704_80.log` |
| tuned DataLoader，`ram_prefetch_batches=16` | 0.30 | 38.98 | 0.345 | 44.1 | `logs/bench_train8_ram16_tuned_20260704_80.log` |

关键日志：

```text
# ram_prefetch_batches=0
07/04 [22:53:30] step=60/80 speed=0.31 step/s, 39.04 samples/s
07/04 [22:54:27] step=80/80 speed=0.31 step/s, 40.32 samples/s

# ram_prefetch_batches=16
07/04 [23:17:11] step=60/80 speed=0.29 step/s, 37.55 samples/s
07/04 [23:18:09] step=80/80 speed=0.30 step/s, 38.98 samples/s
```

训练内结论：

- 在真实训练 forward/backward 路径中，`ram_prefetch_batches=16` 没有带来吞吐提升；80 step 累计吞吐比 no-RAM tuned DataLoader 低约 3.3%，step60-80 稳定区间低约 1.7%，基本可视为无收益或轻微负收益。
- 稳定后速度约 `44-45 samples/s`，折合约 `0.35 steps/s`；如果使用 trainer 累计日志口径，到 step80 是 `39-40 samples/s`，约 `0.30-0.31 steps/s`。
- 当前实现只把下一批 raw chunk/batch 从 DataLoader 异步拉进 RAM，能重叠的是 CPU/I/O/解码等待；训练中的 VAE encode 本身仍在当前 GPU forward 内执行，不能被这个 RAM 队列加速。
- 对这组 8 卡训练，tuned DataLoader 已经基本把 host-side batch 准备隐藏在 GPU forward/backward 后面。继续单纯加深 RAM batch queue 不是主要加速方向；下一步应考虑训练直接读取已经生成的 VAE latent cache，或把异步粒度提升到 episode/chunk 级，减少相邻窗口重复 decode/encode。

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

这是一个安全的第一层试验：它复用 DataLoader/worker 的读取机制，只增加有界 RAM 队列来重叠 CPU 数据准备与 VAE load/encode。远端 precompute-only smoke 已确认早启动 RAM prefetch 对 VAE cache 预计算的前期等待有效，但 8 卡真实训练 A/B 没有看到吞吐提升。训练加速不能只看预计算脚本，必须以训练 loop 的 forward/backward speed 为准。

## 下一步

1. 如果继续优化训练吞吐，优先接入训练读取 VAE latent cache，绕过训练内 VAE encode，而不是继续增大 batch RAM prefetch queue。
2. 如果仍要走在线 encode，下一步需要 profile `training_loss` 内 VAE encode、DiT forward/backward、DataLoader wait 的真实占比，确认瓶颈是否已经从 host-side 等待转到 GPU compute。
3. 如果 host-side 等待在更大规模训练中重新出现，再把缓存粒度从 batch 提升到 episode/chunk，优先复用相邻窗口共享的视频帧。
