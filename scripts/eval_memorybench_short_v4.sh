#!/usr/bin/env bash
set -euo pipefail

export TASK_NAME="${TASK_NAME:-memorybench_short_v4_1e-5}"
export RUN_LABEL="${RUN_LABEL:-FastWAM v4 MemoryBench open-loop eval}"

exec bash "$(dirname "$0")/eval_memorybench_short_common.sh" "$@"
