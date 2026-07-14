#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# v3:动作空间从关节速度(v2)改为绝对关节位置 + 时移 1 帧。
# 依据 26-07-14 真值回放诊断:速度经 JointVelocity 回放漂移,连真值都无法完成任务;
# JointPosition(absolute) + pos[t+1] 三任务真值回放全部成功。
DATA_ROOT="${MEMORYBENCH_DATA_ROOT:-/data/shared/offline/datasets/memorybench}"
RAW_ROOT="${MEMORYBENCH_RAW_ROOT:-${DATA_ROOT}/raw_hf}"
WORKERS="${MEMORYBENCH_CONVERT_WORKERS:-4}"
LIMIT_EPISODES="${MEMORYBENCH_CONVERT_LIMIT_EPISODES:-}"

convert_split() {
  local split="$1"
  local target="${DATA_ROOT}/lerobot/memorybench_short_${split}_v3"
  local staging="${target}.incomplete.$(date +%Y%m%d_%H%M%S)"

  if [[ -e "$target" ]]; then
    echo "[FATAL] Refusing to replace existing dataset: $target" >&2
    exit 1
  fi

  local args=(
    scripts/convert_memorybench_to_lerobot.py
    --raw-root "$RAW_ROOT"
    --out-root "$staging"
    --split "$split"
    --action-source joint_positions+gripper_open
    --action-shift 1
    --action-dim 8
    --state-source gripper_pose+gripper_open
    --state-dim 8
    --workers "$WORKERS"
  )
  if [[ -n "$LIMIT_EPISODES" ]]; then
    args+=(--limit-episodes "$LIMIT_EPISODES")
  fi

  python "${args[@]}"
  mv "$staging" "$target"
  echo "[done] $target"
}

echo "MemoryBench v3 conversion (joint_positions, shift=1): CPU workers=$WORKERS, CUDA is not used"
convert_split train
convert_split test
