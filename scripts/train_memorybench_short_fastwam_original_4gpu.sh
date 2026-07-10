#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TASK_NAME="${TASK_NAME:-memorybench_short_fastwam_original_4gpu_1e-5}"
RUN_ID="${RUN_ID:-${TASK_NAME}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-./runs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
AUTO_PRECOMPUTE_TEXT="${AUTO_PRECOMPUTE_TEXT:-1}"

DATA_ROOT="${MEMORYBENCH_DATA_ROOT:-/data/shared/offline/datasets/memorybench}"
TRAIN_DATA="${DATA_ROOT}/lerobot/memorybench_short_train"
TEXT_CACHE="${DATA_ROOT}/text_embeds_cache"
TEXT_CACHE_GLOB="*.t5_len128.wan22ti2v5b.pt"

export RUN_ID
export MEMORYBENCH_DATA_ROOT="$DATA_ROOT"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "[FATAL] MemoryBench train data not found: $TRAIN_DATA" >&2
  echo "Expected converted data under MEMORYBENCH_DATA_ROOT/lerobot/memorybench_short_train." >&2
  exit 1
fi

if ! compgen -G "${TEXT_CACHE}/${TEXT_CACHE_GLOB}" >/dev/null; then
  if [[ "$AUTO_PRECOMPUTE_TEXT" == "1" ]]; then
    echo "[precompute] MemoryBench text embedding cache is missing; generating it under: $TEXT_CACHE"
    PYTHONPATH=src python scripts/precompute_text_embeds.py task="${TASK_NAME}" +overwrite=false
  else
    echo "[FATAL] MemoryBench text embedding cache is missing under: $TEXT_CACHE" >&2
    echo "Run this once on the H200 repo before training:" >&2
    echo "  PYTHONPATH=src python scripts/precompute_text_embeds.py task=${TASK_NAME} +overwrite=false" >&2
    exit 1
  fi
fi

echo "=========================================================="
echo " FastWAM original MemoryBench short training"
echo "   TASK_NAME=$TASK_NAME"
echo "   RUN_ID=$RUN_ID"
echo "   NPROC_PER_NODE=$NPROC_PER_NODE"
echo "   AUTO_PRECOMPUTE_TEXT=$AUTO_PRECOMPUTE_TEXT"
echo "   TRAIN_DATA=$TRAIN_DATA"
echo "   TEXT_CACHE=$TEXT_CACHE"
echo "   LOG_FILE=$LOG_FILE"
echo "=========================================================="

stdbuf -oL -eL bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task="${TASK_NAME}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
