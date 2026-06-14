#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

# Auto-resume: if a saved DeepSpeed training state exists, continue from the latest;
# otherwise cold-start (warm-start temporal params from the released checkpoint).
LATEST_STATE=$(ls -d runs/mem_temporal_libero/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_STATE" ]; then
  RESUME="$LATEST_STATE"
  echo "[resume] continuing full training state from $RESUME"
else
  RESUME="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  echo "[cold-start] warm-starting weights from $RESUME"
fi

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 8 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=true \
  model.vae_memory.warm_start=true \
  model.vae_memory.train_temporal_only=true \
  data.train.history_video_frames=16 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  resume="$RESUME" \
  output_dir=./runs/mem_temporal_libero \
  eval_every=100000
