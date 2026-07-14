#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${MEMORYBENCH_DATA_ROOT:-/data/shared/offline/datasets/memorybench}"
CACHE_ROOT="${MEMORYBENCH_VAE_CACHE_DIR:-${DATA_ROOT}/vae_latent_cache/memorybench_short_wan22}"
DATASET_STATS_PATH="${MEMORYBENCH_DATASET_STATS_PATH:?MEMORYBENCH_DATASET_STATS_PATH is required}"
POLL_SECONDS="${MEMORYBENCH_TRAIN_POLL_SECONDS:-60}"
STABLE_CHECKS="${MEMORYBENCH_TRAIN_STABLE_CHECKS:-3}"
MAX_USED_MIB="${MEMORYBENCH_GPU_MAX_USED_MIB:-1024}"
MAX_UTIL="${MEMORYBENCH_GPU_MAX_UTIL:-5}"
TRAIN_LOCK="${MEMORYBENCH_TRAIN_LOCK:-/tmp/fastwam_memorybench_dual_training.lock}"

exec 7>"$TRAIN_LOCK"
if ! flock -n 7; then
  echo "[FATAL] Another MemoryBench dual-training orchestrator holds $TRAIN_LOCK" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
export MEMORYBENCH_DATA_ROOT="$DATA_ROOT"
export MEMORYBENCH_VAE_CACHE_DIR="$CACHE_ROOT"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$DATASET_STATS_PATH" ]] || { echo "[FATAL] Missing dataset stats: $DATASET_STATS_PATH" >&2; exit 1; }
python scripts/validate_memorybench_vae_cache.py \
  --task memorybench_short_v4_1e-5 \
  --cache-root "$CACHE_ROOT" \
  --require-complete
python scripts/validate_memorybench_training_configs.py

gpu_is_free() {
  local gpu="$1"
  local pids used util
  pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "$pids" && "$used" -le "$MAX_USED_MIB" && "$util" -le "$MAX_UTIL" ]]
}

all_gpus_free() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    gpu_is_free "$gpu" || return 1
  done
}

wait_for_all_gpus() {
  local stable=0
  while true; do
    if all_gpus_free; then
      stable=$((stable + 1))
      echo "[train-watch] $(date '+%F %T') all-8-free stable=$stable/$STABLE_CHECKS"
      if [[ "$stable" -ge "$STABLE_CHECKS" ]]; then
        sleep 5
        all_gpus_free && return 0
        stable=0
      fi
    else
      stable=0
      snapshot="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
      echo "[train-watch] $(date '+%F %T') waiting-all-8 $snapshot"
    fi
    sleep "$POLL_SECONDS"
  done
}

allocate_ports() {
  python - <<'PY'
import socket

sockets = []
ports = []
for _ in range(2):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sockets.append(sock)
    ports.append(sock.getsockname()[1])
print(*ports)
for sock in sockets:
    sock.close()
PY
}

run_pair() {
  local mode="$1"
  shift
  local timestamp original_run v4_run original_log v4_log original_port v4_port
  local original_pid v4_pid original_status v4_status status
  timestamp="$(date +%Y%m%d_%H%M%S)"
  original_run="memorybench_original_v3_${mode}_${timestamp}"
  v4_run="memorybench_v4_v3_${mode}_${timestamp}"
  original_log="$PWD/runs/logs/${original_run}.log"
  v4_log="$PWD/runs/logs/${v4_run}.log"
  read -r original_port v4_port < <(allocate_ports)

  echo "[train-launch] mode=$mode original_run=$original_run GPUs=0,1,2,3 port=$original_port"
  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT="$original_port" \
    RUN_ID="$original_run" LOG_FILE="$original_log" AUTO_PRECOMPUTE_TEXT=0 \
    FASTWAM_LOG_PARAM_ALIGNMENT=1 \
    bash scripts/train_memorybench_short_fastwam_original_4gpu.sh \
      +data.train.pretrained_norm_stats="$DATASET_STATS_PATH" \
      align_optimizer_param_order=true \
      "$@" &
  original_pid="$!"

  echo "[train-launch] mode=$mode v4_run=$v4_run GPUs=4,5,6,7 port=$v4_port"
  CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT="$v4_port" \
    RUN_ID="$v4_run" LOG_FILE="$v4_log" AUTO_PRECOMPUTE_TEXT=0 \
    FASTWAM_LOG_PARAM_ALIGNMENT=1 \
    bash scripts/train_memorybench_short_v4_4gpu.sh \
      +data.train.pretrained_norm_stats="$DATASET_STATS_PATH" \
      align_optimizer_param_order=true \
      "$@" &
  v4_pid="$!"

  status=0
  set +e
  wait "$original_pid"
  original_status="$?"
  wait "$v4_pid"
  v4_status="$?"
  set -e
  echo "[train-exit] mode=$mode original_status=$original_status log=$original_log"
  echo "[train-exit] mode=$mode v4_status=$v4_status log=$v4_log"
  [[ "$original_status" -eq 0 && "$v4_status" -eq 0 ]] || status=1
  return "$status"
}

mkdir -p runs/logs
echo "[train-watch] cache validated; waiting for all 8 GPUs before dual smoke"
wait_for_all_gpus
run_pair smoke \
  max_steps=1 \
  num_epochs=1 \
  save_every=0 \
  save_final_checkpoint=false \
  eval_every=100000 \
  wandb.enabled=false

echo "[train-watch] dual smoke passed; waiting for all 8 GPUs before full training"
wait_for_all_gpus
run_pair full \
  batch_size=24 \
  learning_rate=1e-5 \
  max_steps=10000 \
  num_epochs=10 \
  save_every=0 \
  save_final_checkpoint=true \
  eval_every=100000 \
  wandb.enabled=true

echo "[train-done] original and v4 training both completed"
