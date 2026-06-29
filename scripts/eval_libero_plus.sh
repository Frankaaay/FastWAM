#!/bin/bash
# Backward-compatible entrypoint. The unified launcher supports both:
#   BENCH=libero bash scripts/eval_libero_plus.sh
#   BENCH=libero_plus bash scripts/eval_libero_plus.sh
set -u
cd "$(dirname "$0")/.." || exit 1
BENCH=${BENCH:-libero_plus}
export BENCH
exec bash scripts/eval.sh "$@"
