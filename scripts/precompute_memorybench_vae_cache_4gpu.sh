#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_CSV="${MEMORYBENCH_PRECOMPUTE_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "[FATAL] MEMORYBENCH_PRECOMPUTE_GPUS must contain exactly 4 GPU indices: $GPU_CSV" >&2
  exit 1
fi

DATA_ROOT="${MEMORYBENCH_DATA_ROOT:-/data/shared/offline/datasets/memorybench}"
CACHE_ROOT="${MEMORYBENCH_VAE_CACHE_DIR:-${DATA_ROOT}/vae_latent_cache/memorybench_short_wan22}"
TASK_NAME="${MEMORYBENCH_PRECOMPUTE_TASK:-memorybench_short_v4_1e-5}"
RUN_ID="${MEMORYBENCH_PRECOMPUTE_RUN_ID:-memorybench_vae_v2_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${MEMORYBENCH_PRECOMPUTE_RUN_ROOT:-$PWD/runs/vae_latent_cache_precompute/memorybench_short_v2}/${RUN_ID}"
LOCK_FILE="${MEMORYBENCH_PRECOMPUTE_LOCK:-/tmp/fastwam_memorybench_vae_precompute.lock}"
MAX_USED_MIB="${MEMORYBENCH_GPU_MAX_USED_MIB:-1024}"
MAX_UTIL="${MEMORYBENCH_GPU_MAX_UTIL:-5}"

mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[FATAL] Another MemoryBench VAE precompute holds $LOCK_FILE" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
export MEMORYBENCH_DATA_ROOT="$DATA_ROOT"
export MEMORYBENCH_VAE_CACHE_DIR="$CACHE_ROOT"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PWD/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

gpu_is_free() {
  local gpu="$1"
  local pids used util
  pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "$pids" && "$used" -le "$MAX_USED_MIB" && "$util" -le "$MAX_UTIL" ]]
}

for gpu in "${GPUS[@]}"; do
  if ! gpu_is_free "$gpu"; then
    echo "[FATAL] GPU $gpu is no longer free; refusing to launch." >&2
    nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >&2
    exit 1
  fi
done

TRAIN_DATA="${DATA_ROOT}/lerobot/memorybench_short_train_v2"
TEXT_CACHE="${DATA_ROOT}/text_embeds_cache"
VAE_PATH="$PWD/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
[[ -d "$TRAIN_DATA" ]] || { echo "[FATAL] Missing train data: $TRAIN_DATA" >&2; exit 1; }
[[ -f "$VAE_PATH" ]] || { echo "[FATAL] Missing VAE checkpoint: $VAE_PATH" >&2; exit 1; }
text_count="$(find "$TEXT_CACHE" -maxdepth 1 -type f -name '*.t5_len128.wan22ti2v5b.pt' | wc -l)"
[[ "$text_count" -ge 5 ]] || { echo "[FATAL] Expected at least 5 text caches, found $text_count" >&2; exit 1; }
available_kib="$(df -Pk "$CACHE_ROOT" | awk 'NR==2 {print $4}')"
[[ "$available_kib" -ge 52428800 ]] || { echo "[FATAL] Less than 50 GiB free under $CACHE_ROOT" >&2; exit 1; }
if pgrep -af '[s]cripts/precompute_vae_latents.py' >/dev/null; then
  echo "[FATAL] Another precompute_vae_latents.py process is already running." >&2
  pgrep -af '[s]cripts/precompute_vae_latents.py' >&2
  exit 1
fi

echo "[preflight] run_id=$RUN_ID GPUs=$GPU_CSV data=$TRAIN_DATA cache=$CACHE_ROOT text_cache_files=$text_count"

SMOKE_WORK="$RUN_ROOT/smoke"
SMOKE_LOG="$RUN_ROOT/smoke.log"
mkdir -p "$SMOKE_WORK"
echo "[smoke] GPU=${GPUS[0]} samples=4 log=$SMOKE_LOG"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python scripts/precompute_vae_latents.py \
  task="$TASK_NAME" \
  +vae_latent_cache.output_dir="$CACHE_ROOT" \
  +vae_latent_cache.work_dir="$SMOKE_WORK" \
  +vae_latent_cache.batch_size=2 \
  +vae_latent_cache.num_workers=2 \
  +vae_latent_cache.max_samples=4 \
  +vae_latent_cache.num_shards=1 \
  +vae_latent_cache.shard_index=0 \
  +vae_latent_cache.overwrite=false \
  >"$SMOKE_LOG" 2>&1

STATS_PATH="$SMOKE_WORK/dataset_stats.json"
[[ -f "$STATS_PATH" ]] || { echo "[FATAL] Smoke did not create dataset stats: $STATS_PATH" >&2; exit 1; }
python scripts/validate_memorybench_vae_cache.py \
  --task "$TASK_NAME" \
  --cache-root "$CACHE_ROOT" \
  --minimum-files 4 \
  --sample-indices 0,1,2,3 \
  | tee "$RUN_ROOT/smoke_validation.json"

declare -a PIDS=()
declare -a LOGS=()
for shard in 0 1 2 3; do
  gpu="${GPUS[$shard]}"
  shard_work="$RUN_ROOT/shard_${shard}"
  shard_log="$RUN_ROOT/shard_${shard}.log"
  mkdir -p "$shard_work"
  echo "[launch] shard=$shard/4 physical_gpu=$gpu log=$shard_log"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/precompute_vae_latents.py \
    task="$TASK_NAME" \
    +data.train.pretrained_norm_stats="$STATS_PATH" \
    +vae_latent_cache.output_dir="$CACHE_ROOT" \
    +vae_latent_cache.work_dir="$shard_work" \
    +vae_latent_cache.batch_size=2 \
    +vae_latent_cache.num_workers=4 \
    +vae_latent_cache.num_shards=4 \
    +vae_latent_cache.shard_index="$shard" \
    +vae_latent_cache.overwrite=false \
    >"$shard_log" 2>&1 &
  PIDS+=("$!")
  LOGS+=("$shard_log")
done

status=0
set +e
for shard in 0 1 2 3; do
  wait "${PIDS[$shard]}"
  shard_status="$?"
  echo "[exit] shard=$shard status=$shard_status log=${LOGS[$shard]}"
  if [[ "$shard_status" -ne 0 ]]; then
    status=1
    tail -80 "${LOGS[$shard]}" >&2
  fi
done
set -e
[[ "$status" -eq 0 ]] || { echo "[FATAL] One or more VAE cache shards failed." >&2; exit 1; }

python scripts/validate_memorybench_vae_cache.py \
  --task "$TASK_NAME" \
  --cache-root "$CACHE_ROOT" \
  --require-complete \
  | tee "$RUN_ROOT/full_validation.json"

echo "[done] run_id=$RUN_ID cache=$CACHE_ROOT logs=$RUN_ROOT"
