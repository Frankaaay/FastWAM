#!/bin/bash
# Unified LIBERO / LIBERO-plus eval launcher for mem-stage-v4.
#
# Defaults:
#   bash eval.sh                         # standard LIBERO, 4 suites, 50 trials/task
#   BENCH=libero_plus bash eval.sh       # LIBERO-plus, noise-inclusive, 1 trial/case
#   BENCH=libero_plus PILOT=200 bash eval.sh
#
# Common overrides:
#   CKPT=/path/to/step_xxxxxx.pt
#   STATS=/path/to/libero_uncond_2cam224_dataset_stats.json
#   NUM_GPUS=8 GPU_OFFSET=0 MAX_PER_GPU=1 SAVE_VIDEO=false
#   OUT=/path/to/output_dir
#   REDIRECT_COMMON_FILES=false          # required on the H200 local checkpoint layout
#   EXTRA_OVERRIDES='EVALUATION.replan_steps=10 model.foo=bar'

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

BENCH=${BENCH:-libero}
if [ "$BENCH" != "libero" ] && [ "$BENCH" != "libero_plus" ]; then
    echo "[FATAL] BENCH must be 'libero' or 'libero_plus', got: $BENCH"
    exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] failed to activate conda env: fastwam"; exit 1; }

if [ "$BENCH" = "libero" ]; then
    export PYTHONPATH=/data/home/frank/projects/LIBERO${PYTHONPATH:+:$PYTHONPATH}
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero_orig}
    echo "[BENCH=libero] PYTHONPATH=$PYTHONPATH LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
fi

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$ROOT/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export TOKENIZERS_PARALLELISM=false

MAGICK_HOME=${MAGICK_HOME:-/data/shared/offline/noise_deps/imagemagick}
if [ -d "$MAGICK_HOME/lib" ]; then
    export MAGICK_HOME
    export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

find_latest_ckpt() {
    python - <<'PY'
from pathlib import Path

roots = [
    Path("runs/libero_uncond_2cam224_1e-4"),
    Path("/data/home/frank/projects/FastWAM/runs/libero_uncond_2cam224_1e-4"),
    Path("/data/home/maxliu/projects/FastWAM/runs/libero_uncond_2cam224_1e-4"),
]
candidates = []
for root in roots:
    if root.exists():
        candidates.extend(root.glob("*/checkpoints/weights/step_*.pt"))
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
print(candidates[-1])
PY
}

CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
    CKPT=$(find_latest_ckpt) || {
        echo "[FATAL] CKPT is not set and no step_*.pt checkpoint was found under runs/libero_uncond_2cam224_1e-4."
        exit 1
    }
fi

STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
NUM_GPUS=${NUM_GPUS:-8}
GPU_OFFSET=${GPU_OFFSET:-0}
MAX_PER_GPU=${MAX_PER_GPU:-1}
PILOT=${PILOT:-0}
SAVE_VIDEO=${SAVE_VIDEO:-false}
REDIRECT_COMMON_FILES=${REDIRECT_COMMON_FILES:-false}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

if [ "$BENCH" = "libero_plus" ]; then
    INCLUDE_NOISE=${INCLUDE_NOISE:-1}
    TRIALS=${TRIALS:-1}
else
    INCLUDE_NOISE=${INCLUDE_NOISE:-1}
    TRIALS=${TRIALS:-50}
fi

OUT=${OUT:-./evaluate_results/$BENCH/libero_uncond_2cam224_1e-4/$(date +%Y%m%d_%H%M%S)}

if [ "$BENCH" = "libero_plus" ] && [ "$INCLUDE_NOISE" != "1" ]; then
    echo "[FATAL] LIBERO-plus eval must use INCLUDE_NOISE=1 for paper-aligned comparison."
    exit 1
fi

[ -f "$CKPT" ] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 1; }
[ -f "$STATS" ] || { echo "[FATAL] dataset stats not found: $STATS"; exit 1; }

NWORKERS=$((NUM_GPUS * MAX_PER_GPU))
mkdir -p "$OUT/shards" "$OUT/worker_logs"

echo "=========================================================="
echo " FastWAM eval"
echo "   ROOT=$ROOT"
echo "   BENCH=$BENCH TRIALS=$TRIALS PILOT=$PILOT INCLUDE_NOISE=$INCLUDE_NOISE"
echo "   CKPT=$CKPT"
echo "   STATS=$STATS"
echo "   NUM_GPUS=$NUM_GPUS GPU_OFFSET=$GPU_OFFSET MAX_PER_GPU=$MAX_PER_GPU NWORKERS=$NWORKERS"
echo "   SAVE_VIDEO=$SAVE_VIDEO REDIRECT_COMMON_FILES=$REDIRECT_COMMON_FILES"
echo "   OUT=$OUT"
[ -n "$EXTRA_OVERRIDES" ] && echo "   EXTRA_OVERRIDES=$EXTRA_OVERRIDES"
echo "=========================================================="

python - "$OUT/shards" "$NWORKERS" "$PILOT" "$INCLUDE_NOISE" "$BENCH" <<'PY'
import json
import os
import sys
from collections import Counter

shard_dir, nworkers, pilot, include_noise, bench = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
)

cases = []
if bench == "libero":
    import libero.libero.benchmark as B

    bench_dict = B.get_benchmark_dict()
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        task_suite = bench_dict[suite]()
        for task_id in range(task_suite.n_tasks):
            cases.append({
                "suite": suite,
                "task_id": task_id,
                "name": None,
                "category": suite,
                "difficulty_level": None,
            })
    cases.sort(key=lambda c: (c["category"], c["task_id"]))
else:
    import libero.libero as L

    cls_path = os.path.join(os.path.dirname(L.__file__), "benchmark", "task_classification.json")
    with open(cls_path, encoding="utf-8") as f:
        data = json.load(f)
    noise = "Sensor Noise"
    for suite, tasks in data.items():
        for task in tasks:
            if (not include_noise) and task.get("category") == noise:
                continue
            cases.append({
                "suite": suite,
                "task_id": int(task["id"]) - 1,
                "name": task.get("name"),
                "category": task.get("category"),
                "difficulty_level": task.get("difficulty_level"),
            })
    cases.sort(key=lambda c: (c["category"] or "", c["suite"], c["task_id"]))

if pilot > 0:
    step = max(1, len(cases) // pilot)
    cases = cases[::step][:pilot]

shards = [[] for _ in range(nworkers)]
for i, case in enumerate(cases):
    shards[i % nworkers].append(case)

os.makedirs(shard_dir, exist_ok=True)
for worker_id, shard in enumerate(shards):
    with open(os.path.join(shard_dir, f"shard_{worker_id}.json"), "w", encoding="utf-8") as f:
        json.dump(shard, f)

out_dir = os.path.dirname(os.path.normpath(shard_dir))
with open(os.path.join(out_dir, "total_cases.txt"), "w", encoding="utf-8") as f:
    f.write(str(len(cases)))
print(f"[shards] total_cases={len(cases)} nworkers={nworkers} per_worker~{len(cases)//max(nworkers, 1)}")
print("[shards] by category:", dict(Counter(c["category"] for c in cases)))
PY
[ $? -eq 0 ] || { echo "[FATAL] failed to generate shards"; exit 1; }
TOTAL=$(cat "$OUT/total_cases.txt" 2>/dev/null || echo 0)

monitor_progress() {
    local out="$1" total="$2" start done now elapsed pct rate eta
    start=$(date +%s)
    while true; do
        sleep 60
        done=$(ls "$out"/*/gpu*_task*_results.json 2>/dev/null | wc -l | tr -d ' ')
        now=$(date +%s)
        elapsed=$((now - start))
        pct=$(awk "BEGIN{printf \"%.1f\", $total?100.0*$done/$total:0}")
        rate=$(awk "BEGIN{printf \"%.1f\", $elapsed?60.0*$done/$elapsed:0}")
        if [ "$done" -gt 0 ]; then
            eta=$(awk "BEGIN{printf \"%.0f\", ($total-$done)*$elapsed/$done/60}")
        else
            eta="?"
        fi
        printf "[%s] done=%s/%s %s%% elapsed=%dm rate=%s/min ETA=%smin\n" \
            "$(date '+%m-%d %H:%M:%S')" "$done" "$total" "$pct" "$((elapsed/60))" "$rate" "$eta" \
            | tee -a "$out/progress.log"
        [ "$total" -gt 0 ] && [ "$done" -ge "$total" ] && break
    done
}

read -r -a EXTRA_ARGS <<< "$EXTRA_OVERRIDES"

PIDS=()
for ((w=0; w<NWORKERS; w++)); do
    SHARD="$OUT/shards/shard_${w}.json"
    [ -s "$SHARD" ] || { echo "shard_$w is empty, skip"; continue; }
    PHYS=$(( (w % NUM_GPUS) + GPU_OFFSET ))
    LOG="$OUT/worker_logs/gpu${PHYS}_w${w}.log"
    CUDA_VISIBLE_DEVICES=$PHYS nohup python experiments/libero/eval_libero_multi.py \
        ckpt="$CKPT" \
        task=libero_uncond_2cam224_1e-4 \
        EVALUATION.num_trials=$TRIALS \
        +EVALUATION.save_video=$SAVE_VIDEO \
        +EVALUATION.task_list_file="$SHARD" \
        EVALUATION.dataset_stats_path="$STATS" \
        EVALUATION.output_dir="$OUT" \
        model.redirect_common_files=$REDIRECT_COMMON_FILES \
        gpu_id=$w \
        "${EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  worker w=$w -> phys_gpu=$PHYS pid=$! log=$LOG"
    sleep 2
done

echo "Started ${#PIDS[@]} workers."
echo "  progress: tail -f $OUT/progress.log"
echo "  worker log: tail -f $OUT/worker_logs/gpu${GPU_OFFSET}_w0.log"

monitor_progress "$OUT" "$TOTAL" &
MON_PID=$!

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL+1))
done
kill "$MON_PID" 2>/dev/null

echo "=========================================================="
echo "All workers exited (failed workers=$FAIL). Summary:"
if [ "$BENCH" = "libero" ]; then
    python experiments/libero/summarize_results.py --output_dir "$OUT"
else
    python experiments/libero/summarize_libero_plus.py --output_dir "$OUT"
fi
echo "=========================================================="
echo "Result dir: $OUT"
exit "$FAIL"
