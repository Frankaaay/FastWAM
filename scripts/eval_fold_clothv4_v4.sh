#!/usr/bin/env bash
# Fold-cloth v4 checkpoint helper. This keeps evaluation/deployment on the
# mem-stage-v4 history path by using the RobotWin/FastWAM policy wrapper, which
# instantiates FastWAMOnlineHistoryBuffer and passes history_video/history_action
# into model inference.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:?Set CKPT=/path/to/checkpoints/weights/step_xxxxxx.pt}"
STATS="${STATS:-}"
if [[ -z "$STATS" ]]; then
  RUN_DIR="$(dirname "$(dirname "$(dirname "$CKPT")")")"
  STATS="$RUN_DIR/dataset_stats.json"
fi

[[ -f "$CKPT" ]] || { echo "[FATAL] checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "[FATAL] dataset stats not found: $STATS" >&2; exit 1; }

export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

echo "Fold-cloth v4 eval/deploy parameters:"
echo "  CKPT=$CKPT"
echo "  STATS=$STATS"
echo "  task=fold_clothv4_v4_2epoch"
echo "  replan_steps=${REPLAN_STEPS:-8}"
echo
echo "Use these args for the real-robot/RoboTwin policy entrypoint:"
echo "  ckpt_setting=$CKPT"
echo "  dataset_stats_path=$STATS"
echo "  sim_task=fold_clothv4_v4_2epoch"
echo "  replan_steps=${REPLAN_STEPS:-8}"
echo
echo "The policy path is experiments/robotwin/fastwam_policy/deploy_policy.py,"
echo "and it uses experiments/fastwam_online_history.py for v4 history."
