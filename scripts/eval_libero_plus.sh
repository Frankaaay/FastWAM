#!/bin/bash
# LIBERO-plus robustness eval for mem-stage-v4.
#
# Default protocol is noise-inclusive (`INCLUDE_NOISE=1`, 10030 cases). For a
# quick 200-case pilot, run `PILOT=200 bash scripts/eval_libero_plus.sh`.

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

BENCH=${BENCH:-libero_plus}

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] failed to activate conda env: fastwam"; exit 1; }

if [ "$BENCH" = "libero" ]; then
    export PYTHONPATH=/data/home/frank/projects/LIBERO${PYTHONPATH:+:$PYTHONPATH}
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero_orig}
    echo "[BENCH=libero] PYTHONPATH=$PYTHONPATH LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
fi

export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export TOKENIZERS_PARALLELISM=false

MAGICK_HOME=${MAGICK_HOME:-/data/shared/offline/noise_deps/imagemagick}
if [ -d "$MAGICK_HOME/lib" ]; then
    export MAGICK_HOME
    export LD_LIBRARY_PATH="$MAGICK_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

CKPT=${CKPT:-}
STATS=${STATS:-}
NUM_GPUS=${NUM_GPUS:-8}
GPU_OFFSET=${GPU_OFFSET:-0}
MAX_PER_GPU=${MAX_PER_GPU:-1}
PILOT=${PILOT:-0}
INCLUDE_NOISE=${INCLUDE_NOISE:-1}
TRIALS=${TRIALS:-1}
SAVE_VIDEO=${SAVE_VIDEO:-false}
OUT=${OUT:-./evaluate_results/$BENCH/libero_uncond_2cam224_1e-4/$(date +%Y%m%d_%H%M%S)}

[ "$INCLUDE_NOISE" = "1" ] || echo "[WARN] INCLUDE_NOISE=$INCLUDE_NOISE; standard FastWAM comparison uses INCLUDE_NOISE=1."
[ -n "$CKPT" ] || { echo "[FATAL] CKPT is required"; exit 1; }
[ -n "$STATS" ] || { echo "[FATAL] STATS is required"; exit 1; }
[ -f "$CKPT" ] || { echo "[FATAL] checkpoint not found: $CKPT"; exit 1; }
[ -f "$STATS" ] || { echo "[FATAL] dataset stats not found: $STATS"; exit 1; }

NWORKERS=$((NUM_GPUS * MAX_PER_GPU))
mkdir -p "$OUT/shards" "$OUT/worker_logs"

echo "=========================================================="
echo " LIBERO-plus eval"
echo "   CKPT=$CKPT"
echo "   STATS=$STATS"
echo "   NUM_GPUS=$NUM_GPUS GPU_OFFSET=$GPU_OFFSET MAX_PER_GPU=$MAX_PER_GPU NWORKERS=$NWORKERS"
echo "   BENCH=$BENCH TRIALS=$TRIALS PILOT=$PILOT INCLUDE_NOISE=$INCLUDE_NOISE SAVE_VIDEO=$SAVE_VIDEO"
echo "   OUT=$OUT"
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
    data = json.load(open(cls_path, encoding="utf-8"))
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
open(os.path.join(out_dir, "total_cases.txt"), "w", encoding="utf-8").write(str(len(cases)))
print(f"[shards] total_cases={len(cases)} nworkers={nworkers} per_worker~{len(cases)//max(nworkers, 1)}")
print("[shards] by factor:", dict(Counter(c["category"] for c in cases)))
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
        gpu_id=$w \
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
python experiments/libero/summarize_libero_plus.py --output_dir "$OUT"
echo "=========================================================="
echo "Result dir: $OUT"
