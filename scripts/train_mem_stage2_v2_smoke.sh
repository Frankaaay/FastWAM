#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage2_v2_smoke.sh — MEM-stage2-v2 (fold-current) 1-step smoke test.
#
# Goal: catch shape / runtime errors in the fold-current path BEFORE a multi-GPU run.
#
# What it exercises (the new path):
#   - model.vae_memory.enabled=false + dit_prepend=false + dit_fold_current=true
#       -> frozen VAE plain-encodes history; the video expert's TemporalFoldAdapter
#          folds history into the CURRENT frame's tokens and DROPS the history frames,
#          so MoT / action expert / masks see the exact base layout. Additive +
#          zero-init gate -> step-0 output == base. ONLY the adapter is trainable
#          (whole base DiT/VAE/action frozen; trainer fold-current branch).
#   - data.train.history_video_frames=5 -> H5 (1.0s), K_lat=2 (MUST be 4n+1, see v1 doc).
#   - fold_history_dropout=0.4 -> per-sample mem-off regularisation.
#
# Single GPU, batch 2, max_steps=1, wandb off. Expect: base ckpt loads strict=False
# (adapter params fresh), one forward+backward, prints loss_video/loss_action, exits.
#
# TIP: also run the weightless invariant unit test (no data, fast):
#   python scripts/test_fold_current.py
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

BASE_CKPT="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
echo "[smoke] cold-start from $BASE_CKPT ; dit_fold_current=true ; H5 ; 1 step ; 1 GPU"

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
  model.vae_memory.dit_prepend=false \
  model.vae_memory.dit_fold_current=true \
  model.vae_memory.fold_history_dropout=0.4 \
  data.train.history_video_frames=5 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=1 \
  wandb.enabled=false \
  resume="$BASE_CKPT" \
  output_dir=./runs/mem_stage2_v2_smoke
