# 训练 profiling 接入记录

## 目的

在 `profiling-mem-stage-v4` 分支上为当前 FastWAM 训练加入可开关的 profiling 能力，用于定位 fold-cloth v4 训练中的 dataloader、forward、backward、optimizer、eval、checkpoint 以及模型内部 VAE / v4 MoT 阶段耗时。

## 基线

- 分支：`profiling-mem-stage-v4`
- 基线 commit：`2da32c3`
- 本地工作树：`/Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/profiling-mem-stage-v4`
- 远端项目主目录：`/data/home/maxliu/projects/FastWAM`
- 本轮状态：代码已修改，尚未提交；本地只做轻量静态验证，不运行 GPU 训练。

## 改动内容

- `configs/train.yaml`
  - 新增 `profile.*` 配置块，默认全部关闭。
  - `profile.timing_enabled`：记录阶段平均耗时。
  - `profile.torch_enabled`：启用 `torch.profiler` 并导出 TensorBoard trace。
  - `profile.rank0_only`：默认只在 rank0 写 trace，避免 8 卡同时写大量文件。
- `src/fastwam/trainer.py`
  - 在训练循环中记录 `data / forward / backward / optimizer / metrics / eval / checkpoint / step_total`。
  - profiling timing 日志输出为 `[profile-timing] step=... data=...ms forward=...ms ...`。
  - timing 指标同步写入 W&B 的 `profile_timing_ms/<stage>`。
  - 可选启用 PyTorch profiler schedule，trace 默认写到 `${output_dir}/profile/torch`。
- `src/fastwam/models/wan22/fastwam.py`
  - 增加 `torch.profiler.record_function` range：
    - `model/build_inputs`
    - `model/vae_encode`
    - `model/build_inputs/current_video_to_latents`
    - `model/build_inputs/history_video_to_latents`
    - `model/training_loss_v4`
    - `model/v4/video_pre_dit`
    - `model/v4/video_prefill_cache`
    - `model/v4/video_post_dit`
    - `model/v4/history_action_pre_dit`
    - `model/v4/history_action_prefill_cache`
    - `model/v4/future_action_pre_dit`
    - `model/v4/future_action_with_condition_cache`
    - `model/v4/action_post_and_loss`

## 建议远端验证命令

### 1. 同步 profiling 分支

```bash
cd /data/home/maxliu/projects/FastWAM

git fetch origin
git checkout profiling-mem-stage-v4
git pull origin profiling-mem-stage-v4
```

### 2. 低侵入阶段计时

用于先判断瓶颈归属。建议先跑 30 step，不保存 checkpoint，不 eval。

```bash
cd /data/home/maxliu/projects/FastWAM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29651
export RUN_ID=profile_timing_fold_clothv4_v4_$(date +%Y%m%d_%H%M%S)
export NPROC_PER_NODE=8

bash scripts/train_fold_clothv4_v4_2epoch.sh \
  batch_size=24 \
  max_steps=30 \
  save_every=100000 \
  eval_every=0 \
  wandb.enabled=false \
  profile.timing_enabled=true \
  profile.timing_log_every=5 \
  profile.timing_sync_cuda=true \
  profile.torch_enabled=false
```

预期输出：

```text
[profile-timing] step=5 data=...ms forward=...ms backward=...ms optimizer=...ms metrics=...ms step_total=...ms
```

判断口径：

- `data` 明显高：优先查 dataset/video decode、`num_workers`、存储吞吐。
- `forward` 明显高：继续看 PyTorch trace 中 VAE encode、video prefill、action condition cache。
- `backward` 明显高：重点看 activation checkpoint、attention backward、通信等待。
- `optimizer` 明显高：重点看 DeepSpeed/ZeRO optimizer、梯度裁剪或调度器开销。

### 3. PyTorch trace 短窗口

用于在阶段计时已经指向 forward/backward 后查看 operator 级热点。建议只导出 rank0 trace。

```bash
cd /data/home/maxliu/projects/FastWAM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29652
export RUN_ID=profile_trace_fold_clothv4_v4_$(date +%Y%m%d_%H%M%S)
export NPROC_PER_NODE=8

bash scripts/train_fold_clothv4_v4_2epoch.sh \
  batch_size=24 \
  max_steps=16 \
  save_every=100000 \
  eval_every=0 \
  wandb.enabled=false \
  profile.timing_enabled=true \
  profile.timing_log_every=4 \
  profile.torch_enabled=true \
  profile.wait_steps=2 \
  profile.warmup_steps=2 \
  profile.active_steps=6 \
  profile.repeat=1 \
  profile.rank0_only=true \
  profile.record_shapes=false \
  profile.profile_memory=false \
  profile.with_stack=false
```

预期 trace 位置：

```text
runs/fold_clothv4_v4_2epoch/<RUN_ID>/profile/torch/
```

查看方式：

```bash
cd /data/home/maxliu/projects/FastWAM
tensorboard --logdir runs/fold_clothv4_v4_2epoch/<RUN_ID>/profile/torch --host 0.0.0.0 --port 6006
```

如需下载到本地分析，可只复制 `profile/torch/` 目录，不要复制 checkpoint 或大日志。

## 当前结论

先跑 `profile.timing_enabled=true` 的短任务。只有当 `forward/backward` 已经明确是瓶颈时，再打开 `profile.torch_enabled=true` 导出 trace。这样 profiling 自身对训练吞吐和磁盘 I/O 的扰动最小。

## 实际 profiling 记录（26-07-02，H200-1）

目标：按 fold-cloth v4 当前训练口径直接采集细粒度 PyTorch profiler trace，并把阶段耗时同步到 W&B。

运行位置：

```text
h200-qinghua-1:/data/home/maxliu/projects/FastWAM
```

分支与 commit：

```text
profiling-mem-stage-v4 / 7f17fda
```

启动方式：

```bash
cd /data/home/maxliu/projects/FastWAM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29652
export RUN_ID=profile_trace_fold_clothv4_v4_20260702_234956
export NPROC_PER_NODE=8

bash scripts/train_fold_clothv4_v4_2epoch.sh \
  batch_size=24 \
  max_steps=45 \
  save_every=100000 \
  eval_every=0 \
  wandb.enabled=true \
  wandb.mode=offline \
  wandb.name=profile_trace_fold_clothv4_v4_20260702_234956 \
  wandb.group=fold-cloth-real-profile \
  profile.timing_enabled=true \
  profile.timing_log_every=5 \
  profile.timing_sync_cuda=true \
  profile.torch_enabled=true \
  profile.wait_steps=20 \
  profile.warmup_steps=5 \
  profile.active_steps=15 \
  profile.repeat=1 \
  profile.rank0_only=true \
  profile.record_shapes=false \
  profile.profile_memory=false \
  profile.with_stack=false
```

本次 run 因远端无法解析 `github.com`，先从本地 `git bundle` 同步 `profiling-mem-stage-v4` 到远端。H200 本机没有 W&B API key，W&B offline run 后续通过 `h200-qinghua-jump` 的 token/spool 机制手动同步。

W&B 同步脚本补充：

- 跳板机脚本：`h200-qinghua-jump:/home/maxliu/.local/bin/wandb-sync-fastwam-offline`
- 备份：`/home/maxliu/.local/bin/wandb-sync-fastwam-offline.bak-20260702-235613`
- 已把 `SCAN_ROOTS` 从只扫描 frank 路径扩展为同时扫描：

```text
/data-214-30-239-40/home/frank/projects/FastWAM
/data-214-30-239-40/home/maxliu/projects/FastWAM
```

- `bash -n /home/maxliu/.local/bin/wandb-sync-fastwam-offline` 通过。
- 手动验证命令使用 `WANDB_SYNC_MIN_STAMP=20260702_230000`，避免先扫到 frank 的旧 run；验证输出已出现 maxliu 路径并成功同步：

```text
[sync] source=/data-214-30-239-40/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/wandb/offline-run-20260702_235338-w158oako
Syncing: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/w158oako ... done.
[ok] /data-214-30-239-40/home/maxliu/projects/FastWAM/runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/wandb/offline-run-20260702_235338-w158oako
```

说明：手动验证前发现 23:35 的旧 maxliu 同步进程卡住并持有 `flock`，已只终止该 `maxliu` 同步进程以释放锁；未触碰 frank 用户自己的同步进程。

关键产物：

```text
log: runs/logs/profile_trace_fold_clothv4_v4_20260702_234956.log
output_dir: runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956
torch trace: runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/profile/torch/lacy--214-30-239-40_2742079.1783007860966418201.pt.trace.json
wandb offline: runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/wandb/offline-run-20260702_235338-w158oako
wandb url: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/w158oako
```

产物大小：

```text
profile/torch: 2.9G
wandb: 96K
checkpoints: 0
```

说明：trace 在 step 40 生成后，为避免 `max_steps=45` 结束时写入大 ZeRO checkpoint，已手动中断 run。GPU 进程已释放，未写 checkpoint。

阶段耗时摘要：

| step | data ms | forward ms | backward ms | optimizer ms | metrics ms | step_total ms | samples/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5632.02 | 2909.41 | 1743.76 | 0.30 | 1.03 | 10287.14 | - |
| 10 | 16.80 | 2474.06 | 1578.17 | 0.16 | 0.84 | 4071.10 | 26.74 |
| 15 | 17.86 | 2467.84 | 1535.46 | 0.15 | 0.77 | 4023.10 | - |
| 20 | 17.74 | 2468.26 | 1527.95 | 0.15 | 1.02 | 4016.18 | 34.28 |
| 25 | 15.65 | 2605.71 | 1577.95 | 0.22 | 1.01 | 4201.52 | - |
| 30 | 17.65 | 2762.50 | 1651.91 | 0.52 | 1.13 | 4434.25 | 37.10 |
| 35 | 17.06 | 2726.67 | 1654.47 | 0.56 | 1.13 | 4400.39 | - |
| 40 | 16.85 | 2716.83 | 1630.47 | 0.66 | 1.17 | 4366.46 | 38.57 |

初步结论：

- 稳定后 dataloader 不是主要瓶颈：`data` 约 16-18ms。
- 主要耗时集中在 forward + backward：稳定后 forward 约 2.47-2.76s，backward 约 1.53-1.65s。
- optimizer / metrics 很小：均远低于 2ms。
- PyTorch profiler active window 覆盖 step 26-40，trace 已生成，可用于继续看 VAE encode、v4 video prefill、history action prefill、future action condition cache 等内部阶段。

## 下一步

- 用 TensorBoard 或 Chrome trace 打开 `profile/torch/*.pt.trace.json`，重点查看 `model/vae_encode`、`model/v4/video_prefill_cache`、`model/v4/future_action_with_condition_cache` 和 backward 对应 kernel。
- 如需后续对照，建议固定 `batch_size=24`，复用已有 `dataset_stats.json`，避免每次 profiling 前重复计算 norm stats。
