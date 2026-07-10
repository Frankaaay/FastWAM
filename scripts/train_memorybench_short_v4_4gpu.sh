#!/usr/bin/env bash
set -euo pipefail

export TASK_NAME="${TASK_NAME:-memorybench_short_v4_1e-5}"
export RUN_LABEL="${RUN_LABEL:-FastWAM v4 MemoryBench short training}"

exec bash "$(dirname "$0")/train_memorybench_short_common.sh" "$@"
