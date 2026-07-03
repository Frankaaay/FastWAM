#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-fold_cloth_orig_2epoch_$(date +%Y%m%d_%H%M%S)}"
TASK_NAME="fold_cloth_orig_2epoch"
LOG_DIR="${LOG_DIR:-./runs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"

export RUN_ID
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

echo "=========================================================="
echo " FastWAM fold-cloth original training"
echo "   RUN_ID=$RUN_ID"
echo "   NPROC_PER_NODE=$NPROC_PER_NODE"
echo "   LOG_FILE=$LOG_FILE"
echo "   DATA=/data/shared/datasets/fold_cloth_fastwam_lerobot/fold_clothv4_240x320_fastwam"
echo "=========================================================="

stdbuf -oL -eL bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task="$TASK_NAME" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
