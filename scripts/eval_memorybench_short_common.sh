#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TASK_NAME="${TASK_NAME:?TASK_NAME is required}"
RUN_LABEL="${RUN_LABEL:-MemoryBench open-loop eval}"
DATA_ROOT="${MEMORYBENCH_DATA_ROOT:-/data/shared/offline/datasets/memorybench}"
TEST_DATA="${MEMORYBENCH_TEST_DATA:-${DATA_ROOT}/lerobot/memorybench_short_test}"
LOG_DIR="${LOG_DIR:-./runs/logs}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${TASK_NAME}_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluate_results/memorybench_open_loop/${EVAL_RUN_ID}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EVAL_RUN_ID}.log}"
EVAL_SAMPLE_STRIDE="${EVAL_SAMPLE_STRIDE:-32}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-10}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

export MEMORYBENCH_DATA_ROOT="$DATA_ROOT"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

if [[ ! -d "$TEST_DATA" ]]; then
  echo "[FATAL] MemoryBench test data not found: $TEST_DATA" >&2
  exit 1
fi

if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(find "./runs/${TASK_NAME}" -path "*/checkpoints/weights/step_*.pt" -type f 2>/dev/null | sort | tail -1 || true)"
fi
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "[FATAL] checkpoint not found. Pass CKPT=/path/to/step_xxxxxx.pt" >&2
  exit 1
fi

if [[ -z "${DATASET_STATS_PATH:-}" ]]; then
  DATASET_STATS_PATH="$(python - "$CKPT" <<'PY'
import sys
from pathlib import Path

ckpt = Path(sys.argv[1]).expanduser()
for parent in ckpt.parents[:6]:
    candidate = parent / "dataset_stats.json"
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
)"
fi
if [[ -z "${DATASET_STATS_PATH:-}" || ! -f "$DATASET_STATS_PATH" ]]; then
  echo "[FATAL] dataset_stats.json not found. Pass DATASET_STATS_PATH=/path/to/dataset_stats.json" >&2
  exit 1
fi

echo "=========================================================="
echo " $RUN_LABEL"
echo "   TASK_NAME=$TASK_NAME"
echo "   CKPT=$CKPT"
echo "   DATASET_STATS_PATH=$DATASET_STATS_PATH"
echo "   TEST_DATA=$TEST_DATA"
echo "   OUTPUT_DIR=$OUTPUT_DIR"
echo "   EVAL_SAMPLE_STRIDE=$EVAL_SAMPLE_STRIDE"
echo "   EVAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES"
echo "   EVAL_NUM_INFERENCE_STEPS=$EVAL_NUM_INFERENCE_STEPS"
echo "   EVAL_DEVICE=$EVAL_DEVICE"
echo "   LOG_FILE=$LOG_FILE"
echo "=========================================================="

stdbuf -oL -eL python experiments/memorybench/eval_open_loop.py \
  task="${TASK_NAME}" \
  "++memorybench_eval.ckpt=${CKPT}" \
  "++memorybench_eval.dataset_dir=${TEST_DATA}" \
  "++memorybench_eval.dataset_stats_path=${DATASET_STATS_PATH}" \
  "++memorybench_eval.output_dir=${OUTPUT_DIR}" \
  "++memorybench_eval.sample_stride=${EVAL_SAMPLE_STRIDE}" \
  "++memorybench_eval.max_samples=${EVAL_MAX_SAMPLES}" \
  "++memorybench_eval.num_inference_steps=${EVAL_NUM_INFERENCE_STEPS}" \
  "++memorybench_eval.device=${EVAL_DEVICE}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
