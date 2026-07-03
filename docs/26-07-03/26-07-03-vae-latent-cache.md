# VAE latent cache 实现记录

## 目的

profiling trace 显示当前 fold-cloth v4 训练的稳定窗口里，forward 主要开销之一是 frozen VAE encode：每 step 对 current video 和 history video 各 encode 一次。VAE 不参与训练、`_encode_video_latents()` 也在 `torch.no_grad()` 下运行，因此第一阶段先把 video latents 预计算到磁盘，训练时直接读取 latent，跳过 VAE encode。

## 当前分支与范围

- 分支：`mem-stage-v4`
- 运行位置：本地 `/Users/maxliu/MyProjects/AIR/202606/FastWAM_xyc`
- 远端目标位置：`h200-qinghua-1:/data/home/maxliu/projects/FastWAM`
- 本次只实现训练 forward 的 VAE latent cache，不修改 attention backend、MoT 结构和推理路径。

## 代码改动

1. `RobotVideoDataset`
   - 新增可选参数：
     - `vae_latent_cache_dir`
     - `vae_latent_cache_keep_video`
     - `vae_latent_cache_model_id`
     - `vae_latent_cache_validate_metadata`
   - 默认不启用 cache，行为与原来一致。
   - 启用 cache 后按 dataset/preprocess metadata 生成 fingerprint，缓存路径为：

```text
<cache_root>/<fingerprint>/<sample_idx // 1000>/<sample_idx>.pt
```

   - 每个 cache payload 保存：
     - `sample_idx`
     - `fingerprint`
     - `metadata`
     - `model_id`
     - `vae_path`
     - `input_latents`
     - `history_video_latents`
   - dataset 现在会返回 `sample_idx`，用于预计算脚本按真实样本编号写文件。

2. `FastWAM.build_inputs()`
   - 支持 `sample["input_latents"]` 和 `sample["history_video_latents"]`。
   - 如果 cached latents 存在，直接搬到模型 device/dtype，不再调用 `_encode_video_latents()`。
   - 如果 cached latents 不存在，保留原来的 raw video VAE encode 路径。
   - raw video 缺失时，使用 `image_is_pad` 恢复原始 video timeline 长度，用于 action/video transition 校验。

3. `scripts/precompute_vae_latents.py`
   - Hydra 入口，复用训练配置。
   - 只加载 VAE，不加载 video DiT / ActionDiT。
   - 支持单卡或 `torchrun` 多卡分片。
   - 支持 `overwrite=false` 断点续跑。
   - 支持 `max_samples` 做小样本 smoke。

## 远端使用命令草案

cache 是提前生成的离线文件，不是在训练过程中边算边写。这样训练 profiling 里 VAE encode 是否消失会更清楚，训练进程也不会混入 cache write I/O。

空间估算：

- fold-cloth v4 当前 video 采样：current video 9 帧、history video 5 帧。
- 按 Wan VAE latent 估算：current latent `[48,3,24,20]`，history latent `[48,2,24,20]`。
- bf16 tensor payload：`(48*3*24*20 + 48*2*24*20) * 2 = 230400 bytes`，约 `225 KiB/样本`。
- H200 smoke 实测当前 fold-cloth v4 dataset size 为 `865308`，64 条 cache 占用 `15M`，约 `234 KiB/样本`。
- 按实测外推，全量 cache 约 `198 GiB`；考虑文件系统和后续配置变体，正式跑前按 `200-220 GiB` 预留更稳。

先做 64 条样本 smoke：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export CACHE_DIR=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22

torchrun --standalone --nproc_per_node=1 scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=${CACHE_DIR} \
  +vae_latent_cache.work_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/fold_clothv4_v4_wan22_smoke \
  +vae_latent_cache.batch_size=2 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.max_samples=64 \
  +vae_latent_cache.overwrite=false

du -sh ${CACHE_DIR}
```

smoke 通过后再跑全量 cache：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export CACHE_DIR=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22

torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=${CACHE_DIR} \
  +vae_latent_cache.work_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/fold_clothv4_v4_wan22_full \
  +vae_latent_cache.batch_size=4 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.overwrite=false

du -sh ${CACHE_DIR}
```

训练时启用 cache：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

export CACHE_DIR=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22
export RUN_ID=fold_clothv4_v4_latcache_bs24_5epoch_$(date +%Y%m%d_%H%M%S)

bash scripts/train_fold_clothv4_v4_2epoch.sh \
  batch_size=24 \
  num_epochs=5 \
  wandb.name=fold_clothv4_v4_latcache_bs24_5epoch \
  wandb.group=fold-cloth-real-bs24-latcache \
  +data.train.vae_latent_cache_dir=${CACHE_DIR} \
  +data.train.vae_latent_cache_model_id=Wan-AI/Wan2.2-TI2V-5B
```

## profiling 对比标准

同步到 profiling 分支后，沿用之前的 `profile.torch_enabled=true` 方式抓 trace。预期变化：

- `model/vae_encode` 调用数应从每 step 约 2 次降到 0。
- `model/build_inputs/current_video_to_latents` 和 `model/build_inputs/history_video_to_latents` 应消失或接近 0。
- `train/forward_loss` 应明显下降。
- `train/backward` 预期变化不大，因为 VAE 原本就是 frozen/no_grad。

## 风险与注意

- cache 文件绑定 dataset/preprocess fingerprint；修改 `video_size`、`action_video_freq_ratio`、`concat_multi_camera`、dataset split 等参数后需要重新预计算。
- 当前第一阶段没有重写底层 LeRobot image decode。训练时 dataset 仍读取样本元信息和 image payload，但不会把 raw video tensor 返回给训练 batch。此前 profiling 中 dataloader 稳定窗口约 17ms，不是主瓶颈。
- 全量 cache 按 smoke 实测外推约 `198 GiB`；正式跑前需要确认 `runs/vae_latent_cache` 所在磁盘空间，并建议预留 `200-220 GiB`。
- `precompute_vae_latents.py` 不写 WandB；后续训练 run 仍按训练配置写 WandB offline run，现有跳板机同步脚本会继续同步。

## 远端 smoke（26-07-03，H200-1）

运行位置：

```text
h200-qinghua-1:/data/home/maxliu/projects/FastWAM
branch: profiling-mem-stage-v4
commit: 44204a3
```

同步方式：

- H200 无法解析 `github.com`，`git pull` 失败：

```text
fatal: unable to access 'https://github.com/Frankaaay/FastWAM.git/': Could not resolve host: github.com
```

- 本地生成 bundle：`/private/tmp/fastwam_profiling_latcache_44204a3.bundle`。
- 远端从 `/tmp/fastwam_profiling_latcache_44204a3.bundle` fetch 并 fast-forward 到 `44204a3`。

smoke 命令：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

export CUDA_VISIBLE_DEVICES=0
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export CACHE_DIR=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22
export WORK_DIR=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/fold_clothv4_v4_wan22_smoke

torchrun --standalone --nproc_per_node=1 scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=${CACHE_DIR} \
  +vae_latent_cache.work_dir=${WORK_DIR} \
  +vae_latent_cache.batch_size=2 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.max_samples=64 \
  +vae_latent_cache.overwrite=false

du -sh ${CACHE_DIR}
```

结果：

```text
Dataset size=865308
fingerprint=5ff52f56f7112f87
to_encode=64
new=64 overwrite=0 skip=0
VAE latents rank 0/1: 64/64, 8.65 sample/s
cache size: 15M
```

结论：

- 预计算脚本可在 H200 的 `maxliu` 环境中正常解析 Hydra、加载 dataset、加载 VAE、写入 cache。
- 当前配置下每样本 cache 实测约 `15M / 64 = 0.234M`，与 bf16 tensor payload 估算一致。
- 全量 cache 约 `198 GiB`，当前 `/data` 盘仍有约 `20T` 可用，可以承载全量 cache。
- 非阻塞 warning：torchrun/单进程退出时出现 `destroy_process_group() was not called before program exit`，未影响 cache 写入；后续可在脚本收尾处按需显式 destroy。

## 远端全量预计算与 cached profiling 等待器（进行中）

运行位置：

```text
h200-qinghua-1:/data/home/maxliu/projects/FastWAM
branch: profiling-mem-stage-v4
commit: 8bba6e8
```

全量预计算 tmux：

```text
session: vae_latent_cache_full_8bba6e8_20260703_0045
script: /tmp/vae_latent_cache_full_8bba6e8_20260703_0045.sh
log: runs/logs/vae_latent_cache_full_8bba6e8_20260703_0045.log
cache_dir: runs/vae_latent_cache/fold_clothv4_v4_wan22
work_dir: runs/vae_latent_cache_precompute/fold_clothv4_v4_wan22_full
```

关键命令：

```bash
cd /data/home/maxliu/projects/FastWAM
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /data/home/maxliu/.conda/envs/fastwam

torchrun --standalone --nproc_per_node=8 scripts/precompute_vae_latents.py \
  task=fold_clothv4_v4_2epoch \
  +vae_latent_cache.output_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache/fold_clothv4_v4_wan22 \
  +vae_latent_cache.work_dir=/data/home/maxliu/projects/FastWAM/runs/vae_latent_cache_precompute/fold_clothv4_v4_wan22_full \
  +vae_latent_cache.batch_size=4 \
  +vae_latent_cache.num_workers=4 \
  +vae_latent_cache.overwrite=false
```

2026-07-03 01:45 进度快照：

```text
cache files: 199856 / 865308
cache size: 45G
status: tmux 仍在运行，cache 文件数持续增长
```

2026-07-03 01:57 进度快照：

```text
cache files: 251356 / 865308
cache size: 56G
status: full precompute 与 latcache_profile_waiter 两个 tmux 均仍在运行
```

2026-07-03 02:00 进度快照：

```text
cache files: 263668 / 865308
cache size: 59G
estimated rate from waiter log: 69.87 files/s
estimated remaining time: 2.41h
status: full precompute 与 latcache_profile_waiter 两个 tmux 均仍在运行，cached profiling 尚未启动
```

2026-07-03 02:03 ongoing payload 抽查：

```text
cache files seen: 277544
checked files:
- 000000/000000000.pt
- 000138/000138772.pt
- 000301/000301342.pt
fingerprint: 5ff52f56f7112f87
model_id: Wan-AI/Wan2.2-TI2V-5B
input_latents: (48, 3, 24, 20), torch.bfloat16
history_video_latents: (48, 2, 24, 20), torch.bfloat16
```

2026-07-03 02:05 日志健康检查：

```text
cache files: 285124 / 865308
cache size: 64G
grep error/exception/traceback/failed/missing/oom/no space: no matches
rank0 progress: about 35376 / 108156, 33%
status: full precompute 仍在继续，cached profiling 尚未启动
```

2026-07-03 02:11 资源健康检查：

```text
cache files: 310688 / 865308
cache size: 69G
/data disk: 28T total, 8.7T used, 20T available, 31% used
precompute procs: 41
grep error/exception/traceback/failed/missing/oom/no space: no matches
status: full precompute 仍在继续，cached profiling 尚未启动
```

cached profiling 等待器：

```text
session: latcache_profile_waiter_20260703
script: /tmp/wait_and_run_latcache_profile_20260703.sh
log: runs/logs/wait_and_run_latcache_profile_20260703.log
```

等待器逻辑：

- 每 5 分钟检查 cache 文件数、cache 目录大小和 `precompute_vae_latents.py` 进程数。
- 只有当 cache 文件数达到 `865308` 且预计算进程退出后，才抽查 sample `0`、`432654`、`865307` 的 schema、fingerprint、model id 和 latent shape。
- 抽查通过后启动 cached profiling，复用 `runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/dataset_stats.json`，避免再次全量计算 norm stats。
- profiling 命令使用 `profile.torch_enabled=true`、`wait_steps=20`、`warmup_steps=5`、`active_steps=15`、`rank0_only=true`，并写 W&B offline run 到 `maxliuyy_thu/fastwam-mem`。
- 为避免 trainer 在 `max_steps` 正常结束时写最终 checkpoint，等待器把 profiling 进程放到独立 process group 中运行；检测到 `profile/torch/*.pt.trace.json` 生成后等待 20 秒，然后对该 process group 发送 `INT`/`TERM` 停止 run。
- 2026-07-03 01:49 热修等待器：`pgrep` 使用 `[s]cripts/precompute_vae_latents.py` pattern，避免进程检测误匹配自身；`bash -n /tmp/wait_and_run_latcache_profile_20260703.sh` 通过。
- 2026-07-03 01:53 增强等待器：trace 截停后自动运行 `/tmp/aggregate_fastwam_trace.py`，写入 `runs/fold_clothv4_v4_2epoch/<RUN_ID>/profile/trace_summary.tsv`；同时从 `runs/logs/<RUN_ID>.log` 抽取 `[profile-timing]` 行到 `profile/profile_timing_lines.txt`，并在等待器日志中记录 W&B offline run 路径。
- 2026-07-03 01:59 增强等待器：检测到 `*.pt.trace.json` 后不立即停止训练，而是每 10 秒检查文件大小；连续 3 次大小稳定后再发送 `INT`/`TERM`，降低 3GB 级 trace 尚未写完时被截断的风险。`bash -n /tmp/wait_and_run_latcache_profile_20260703.sh` 通过。
- 2026-07-03 02:02 热修等待器：cache 计数从 `CACHE_DIR` 改为 `CACHE_DIR/5ff52f56f7112f87`，避免未来同一 cache root 下出现其他 fingerprint 时 overcount；当前 cache root 只有目标 fingerprint，等待器重启后计数正常。

当时结论：截至 2026-07-03 02:11，全量 cache 尚未完成，cached profiling 尚未启动。后续最终结果见文末“远端全量结果与 profiling 对比”。

## W&B 同步链路检查（26-07-03）

检查位置：

```text
h200-qinghua-jump:/home/maxliu/.local/bin/wandb-sync-fastwam-offline
```

发现问题：

- cron 仍为每 5 分钟运行一次：`*/5 * * * * /home/maxliu/.local/bin/wandb-sync-fastwam-offline`。
- 但 2026-07-03 00:50 启动的一次旧 frank run 同步卡住，占用 `sync.lock`，导致后续 cron 一直输出 `[skip] previous sync is still running`。
- 卡住的源 run 是 2026-07-01 的旧训练：`/data-214-30-239-40/home/frank/projects/FastWAM/runs/fold_clothv4_v4_2epoch/fold_clothv4_v4_bs24_5epoch_20260701_202646/wandb/offline-run-20260701_203004-gq2t66kb`。

处理：

```text
backup: /home/maxliu/.local/bin/wandb-sync-fastwam-offline.bak-20260703-0156
MIN_STAMP: 20260701_200000 -> 20260702_230000
follow-up backup: /home/maxliu/.local/bin/wandb-sync-fastwam-offline.bak-20260703-0159
follow-up MIN_STAMP: 20260702_230000 -> 20260703_015000
```

- 新默认时间线会跳过 2026-07-01 的旧 frank run，但保留 2026-07-02 profiling run 和后续 latcache profiling run。
- 终止 stale 的 `wandb sync` / wrapper 进程释放 lock。
- `bash -n /home/maxliu/.local/bin/wandb-sync-fastwam-offline` 通过。
- 手动运行同步脚本通过，结果：`[done] candidates=1 failed=0`。
- 进一步收窄默认 `MIN_STAMP` 到 `20260703_015000` 后，手动运行同步脚本结果为 `[done] candidates=0 failed=0`；后续只扫描 latcache profiling 之后的新 run，避免每 5 分钟重复同步旧 baseline run。

当前结论：latcache profiling 产出的 W&B offline run 后续应能被 cron 同步到 `maxliuyy_thu/fastwam-mem`。如果新的 run 产生后仍未出现在 W&B，应优先检查 `h200-qinghua-jump:/home/maxliu/.local/state/wandb-sync/sync.log`。

## trace 聚合准备

为避免 cached trace 产出后手工翻大型 JSON，已在远端准备临时聚合脚本：

```text
h200-qinghua-1:/tmp/aggregate_fastwam_trace.py
h200-qinghua-1:/tmp/summarize_latcache_profile_result.sh
```

验证命令：

```bash
cd /data/home/maxliu/projects/FastWAM
TRACE=$(find runs/fold_clothv4_v4_2epoch/profile_trace_fold_clothv4_v4_20260702_234956/profile/torch -name "*.pt.trace.json" -type f | head -1)
/tmp/aggregate_fastwam_trace.py "$TRACE" | sed -n "1,120p"
```

baseline trace 聚合关键结果：

```text
trace size: 3032318957 bytes
events_total: 10722569
train/forward_loss: count=15 mean=2735.19ms
train/backward: mean=1599.87ms
model/vae_encode: mean=662.49ms
model/build_inputs/current_video_to_latents: mean=410.66ms
model/build_inputs/history_video_to_latents: mean=252.39ms
model/v4/video_prefill_cache: mean=957.59ms
aten::scaled_dot_product_attention: count=4500
aten::_scaled_dot_product_efficient_attention: count=4500
aten::_scaled_dot_product_efficient_attention_backward: count=2670
```

说明：该临时脚本用于快速排序热点和检查 `model/vae_encode` 是否消失。模型内部 annotation 在 Chrome trace 中可能同时包含 CPU/GPU 口径，严格耗时对比仍以 `train/forward_loss`、`train/backward`、`ProfilerStep#*` 和同一脚本的前后相对变化为准。

cached profiling 结束后的汇总命令模板：

```bash
cd /data/home/maxliu/projects/FastWAM
/tmp/summarize_latcache_profile_result.sh runs/fold_clothv4_v4_2epoch/<profile_trace_latcache_run_id>
```

该脚本会输出 run id、trace 路径、`trace_summary.tsv`、`profile_timing_lines.txt`、W&B offline run 路径，并打印 `train/forward_loss`、`train/backward`、`model/vae_encode`、`model/build_inputs*`、`model/v4/video_prefill_cache` 和 attention kernel 的关键行。

## 本地验证

本地只做轻量语法检查，不运行 GPU 训练或 VAE encode：

```bash
python -c "from pathlib import Path; files=['src/fastwam/datasets/lerobot/robot_video_dataset.py','src/fastwam/models/wan22/fastwam.py','scripts/precompute_vae_latents.py']; [compile(Path(p).read_text(), p, 'exec') for p in files]; print('compile ok:', ', '.join(files))"
```

结果：通过。

## 远端全量结果与 profiling 对比（26-07-03）

全量预计算第一次使用 `torchrun --nproc_per_node=8` 跑到约 `812883 / 865308`
后退出。日志根因是 rank6 在最终统计汇总附近触发 NCCL watchdog：

```text
WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=3, NumelOut=3, Timeout(ms)=600000)
scripts/precompute_vae_latents.py FAILED
```

为了保留已生成的 cache，没有删除重跑；改为给 `scripts/precompute_vae_latents.py`
增加非分布式手动分片参数：

```text
+vae_latent_cache.num_shards=8
+vae_latent_cache.shard_index=<0..7>
```

然后用 8 个互不通信的单进程 shard，分别设置 `CUDA_VISIBLE_DEVICES=<gpu>`，
继续以 `overwrite=false` 补齐缺失样本。该模式不使用 `torchrun`，因此没有 NCCL
all-reduce/barrier 风险。

最终 cache：

```text
cache_dir: runs/vae_latent_cache/fold_clothv4_v4_wan22
fingerprint: 5ff52f56f7112f87
files: 865308 / 865308
size: 192G
validated samples: 0, 432654, 865307
payload dtype: torch.bfloat16
```

cached profiling run：

```text
run id: profile_trace_latcache_fold_clothv4_v4_20260703_045618
trace: runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618/profile/torch/lacy--214-30-239-40_3735196.1783026074570934851.pt.trace.json
trace size: 1.1G
summary: runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618/profile/trace_summary.tsv
wandb offline: runs/fold_clothv4_v4_2epoch/profile_trace_latcache_fold_clothv4_v4_20260703_045618/wandb/offline-run-20260703_045846-d40rx5m9
wandb url: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/d40rx5m9
```

W&B 同步已在跳板机日志中确认：

```text
Syncing: https://wandb.ai/maxliuyy_thu/fastwam-mem/runs/d40rx5m9 ... done.
[ok] .../offline-run-20260703_045846-d40rx5m9
```

trace 对比：

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

- VAE latent cache 达到预期：训练 forward 中的 VAE encode 已消失。
- 主要收益在 forward，`train/forward_loss` 从约 `2.74s` 降到约 `1.40s`。
- backward 基本不变，符合 VAE 原本 frozen/no-grad 的预期。
- 下一阶段瓶颈转移到 `model/v4/video_prefill_cache` 与 attention backend。
