#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage2_smoke.sh — MEM-stage2 (option C / 6.1) 1-step smoke test.
#
# Goal: catch shape / runtime errors in the DiT-side history-prepend path BEFORE
# committing to a multi-GPU run. NOT a real training run.
#
# What it exercises (the new path):
#   - model.vae_memory.enabled=false  -> NO VAE temporal params (old path off)
#   - model.vae_memory.dit_prepend=true -> frozen plain-encode history latents are
#       PREPENDED to the video token sequence; current frame absorbs history via
#       the video expert self-attention; action reads the (history-enriched)
#       current-frame K/V. Full DiT trainable (trainer default branch, since
#       vae_memory.enabled=false), cold-started from the base ckpt.
#   - data.train.history_video_frames=5 -> H5 (1.0s), K_lat=2 history latent frames.
#       (MUST be 4n+1: the frozen plain VAE encodes frame0 as its own chunk then every
#        4 frames as one chunk, so a 4n input like H4 silently drops the most-recent 3
#        frames; H5 -> [h0][h1..h4] encodes all 5.)
#
# Single GPU, batch 2, max_steps=1, wandb off. Expect: it loads the base ckpt
# strict (no new params), runs one forward+backward, prints loss_video/loss_action,
# and exits. Any TypeError / shape mismatch surfaces here.
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

BASE_CKPT="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
echo "[smoke] cold-start from $BASE_CKPT ; dit_prepend=true ; H5 ; 1 step ; 1 GPU"

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 1 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  batch_size=2 \
  learning_rate=1e-5 \
  max_steps=1 \
  save_every=100000 \
  eval_every=100000 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=false \
  model.vae_memory.dit_prepend=true \
  data.train.history_video_frames=5 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=1 \
  wandb.enabled=false \
  resume="$BASE_CKPT" \
  output_dir=./runs/mem_stage2_smoke
