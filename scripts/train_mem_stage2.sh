#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage2.sh — VAE-memory 微调 stage-2:输入接口对齐
#
# 背景:stage-1(train_mem_temporal.sh)只训了 4 个 VAE temporal 张量,DiT 全程冻结。
# LIBERO-plus 上 mem-on(45.6)反而低于 mem-off(49.8)—— 冻死的 DiT 输入层
# patch_embedding 是在「无记忆 latent」上训出来的,读不懂被记忆改写过的 latent。
#
# stage-2 在 stage-1 权重基础上,额外只解冻 DiT 输入接口 video_expert.patch_embedding
# (2 个张量),和 temporal 一起继续微调(co-adapt),让 DiT 学会消化带记忆的 latent。
#
# 关键设计:
#   - weights-only resume(指向 .pt,不是 state 目录):新增了 patch_embedding 进
#     可训练集合,旧的 ZeRO optimizer 分片对不上,必须重置 optimizer。重置后
#     world_size 不再受锁 —— 但我们仍固定 4 卡 × 32 = global 128,和 stage-1 可比。
#   - warm_start=false:temporal 已训好,绝不能再用 spatial proj 覆盖它。
#   - lr=3e-5(低于 stage-1 的 1e-4):finetune-on-finetune,且 patch_embedding 是
#     预训练层,轻轻动,别把干净 latent 的读法带偏。trainer 自动加 5% warmup + cosine。
#   - max_steps=6000(~2.75 epoch,1 epoch≈2170 步),save_every=1000:存多个点,
#     离线分别 eval(先 PILOT 快筛,再全量),挑最好的,避免过训 patch_embedding。
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

# stage-1 产出的权重(只取权重,optimizer/step 重置)
CKPT=${CKPT:-runs/mem_temporal_libero/checkpoints/weights/step_021700.pt}
[ -f "$CKPT" ] || { echo "[FATAL] 找不到 stage-1 ckpt: $CKPT"; exit 1; }
echo "[stage-2] weights-only resume from $CKPT (optimizer/step 重置,卡数自由)"

# 4 卡 × per-GPU 32 × grad_accum 1 = 全局 batch 128(与 stage-1 一致,可比)
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 4 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  batch_size=32 \
  learning_rate=3e-5 \
  max_steps=6000 \
  save_every=1000 \
  eval_every=100000 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=true \
  model.vae_memory.warm_start=false \
  model.vae_memory.train_temporal_only=true \
  model.vae_memory.unfreeze_patch_embed=true \
  data.train.history_video_frames=16 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=10 \
  resume="$CKPT" \
  output_dir=./runs/mem_temporal_libero_stage2
