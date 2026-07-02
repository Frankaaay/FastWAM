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

## 本地验证

本地只做轻量语法检查，不运行 GPU 训练或 VAE encode：

```bash
python -c "from pathlib import Path; files=['src/fastwam/datasets/lerobot/robot_video_dataset.py','src/fastwam/models/wan22/fastwam.py','scripts/precompute_vae_latents.py']; [compile(Path(p).read_text(), p, 'exec') for p in files]; print('compile ok:', ', '.join(files))"
```

结果：通过。
