#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-fold_clothv4_v4_2epoch_20260701}"
TASK_NAME="${TASK_NAME:-fold_clothv4_v4_2epoch}"
RUN_DIR="${RUN_DIR:-./runs/${TASK_NAME}/${RUN_ID}}"
LOG_FILE="${LOG_FILE:-./runs/logs/${RUN_ID}.log}"

echo "=========================================================="
echo " FastWAM fold-cloth v4 monitor"
echo "   RUN_DIR=$RUN_DIR"
echo "   LOG_FILE=$LOG_FILE"
echo "=========================================================="

echo "--- latest loss/progress lines ---"
grep -E "step=.*loss=|loss|max_steps reached|epoch=" "$LOG_FILE" 2>/dev/null | tail -30 || true

echo "--- checkpoints ---"
find "$RUN_DIR/checkpoints" -maxdepth 3 -type f 2>/dev/null | sort | tail -30 || true

echo "--- gpu ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
