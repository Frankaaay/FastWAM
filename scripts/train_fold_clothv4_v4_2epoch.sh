#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-fold_clothv4_v4_2epoch_$(date +%Y%m%d_%H%M%S)}"

export RUN_ID
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

conda run -n fastwam bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=fold_clothv4_v4_2epoch \
  "$@"
