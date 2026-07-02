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

## 下一步

- 本地通过 `py_compile` 和 `git diff --check` 后，提交并推送到 `origin/profiling-mem-stage-v4`。
- 远端按上方命令跑 30-step timing，再根据 `[profile-timing]` 结果决定是否跑 PyTorch trace。
