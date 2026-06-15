#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

# 8 卡全空，用全部 8 张(0-7)。注意：step_008000 断点是按 8 卡 world_size 存的，
# DeepSpeed ZeRO 不支持改卡数续训，所以续训必须保持 8 卡。

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

# 4 卡 × per-GPU 32 × grad_accum 1 = 全局 batch 128(与断点前一致，续训干净)
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 4 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  batch_size=32 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=true \
  model.vae_memory.warm_start=true \
  model.vae_memory.train_temporal_only=true \
  data.train.history_video_frames=16 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  resume="$RESUME" \
  output_dir=./runs/mem_temporal_libero \
  eval_every=100000
