#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_GROUPS="${MEMORYBENCH_GPU_GROUPS:-0,1,2,3 4,5,6,7}"
POLL_SECONDS="${MEMORYBENCH_GPU_POLL_SECONDS:-60}"
STABLE_CHECKS="${MEMORYBENCH_GPU_STABLE_CHECKS:-3}"
MAX_USED_MIB="${MEMORYBENCH_GPU_MAX_USED_MIB:-1024}"
MAX_UTIL="${MEMORYBENCH_GPU_MAX_UTIL:-5}"
WATCH_LOCK="${MEMORYBENCH_WATCH_LOCK:-/tmp/fastwam_memorybench_vae_watcher.lock}"

exec 8>"$WATCH_LOCK"
if ! flock -n 8; then
  echo "[FATAL] Another MemoryBench VAE watcher holds $WATCH_LOCK" >&2
  exit 1
fi

gpu_is_free() {
  local gpu="$1"
  local pids used util
  pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "$pids" && "$used" -le "$MAX_USED_MIB" && "$util" -le "$MAX_UTIL" ]]
}

group_is_free() {
  local group="$1"
  local gpu
  IFS=',' read -r -a group_gpus <<< "$group"
  [[ "${#group_gpus[@]}" -eq 4 ]] || return 1
  for gpu in "${group_gpus[@]}"; do
    gpu_is_free "$gpu" || return 1
  done
}

echo "[watch] groups='$GPU_GROUPS' poll=${POLL_SECONDS}s stable_checks=$STABLE_CHECKS max_used=${MAX_USED_MIB}MiB max_util=${MAX_UTIL}%"
stable_group=""
stable_count=0

while true; do
  selected=""
  for group in $GPU_GROUPS; do
    if group_is_free "$group"; then
      selected="$group"
      break
    fi
  done

  if [[ -n "$selected" ]]; then
    if [[ "$selected" == "$stable_group" ]]; then
      stable_count=$((stable_count + 1))
    else
      stable_group="$selected"
      stable_count=1
    fi
    echo "[watch] $(date '+%F %T') candidate=$selected stable=$stable_count/$STABLE_CHECKS"
    if [[ "$stable_count" -ge "$STABLE_CHECKS" ]]; then
      sleep 5
      if group_is_free "$selected"; then
        echo "[watch] $(date '+%F %T') launching on GPUs=$selected"
        export MEMORYBENCH_PRECOMPUTE_GPUS="$selected"
        exec bash scripts/precompute_memorybench_vae_cache_4gpu.sh
      fi
      stable_group=""
      stable_count=0
    fi
  else
    stable_group=""
    stable_count=0
    snapshot="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    echo "[watch] $(date '+%F %T') no-free-group $snapshot"
  fi
  sleep "$POLL_SECONDS"
done
