#!/bin/bash
# Unified LIBERO / LIBERO-plus eval launcher for mem-stage-v4.
#
# Presets. Use EVAL to switch mode, then override only the variables you mean to change:
#   bash eval.sh                         # custom mode; BENCH defaults to standard LIBERO
#   EVAL=libero_full bash eval.sh        # standard LIBERO, 4 suites x 10 tasks, 50 trials/task
#   EVAL=libero_pilot bash eval.sh       # standard LIBERO smoke pilot, default PILOT=8, TRIALS=1
#   EVAL=plus_full bash eval.sh          # LIBERO-plus full 10030 cases, INCLUDE_NOISE=1, TRIALS=1
#   EVAL=plus_pilot bash eval.sh         # LIBERO-plus pilot, default PILOT=200, INCLUDE_NOISE=1
#   EVAL=case200 bash eval.sh            # alias of plus_pilot
#
# Low-level overrides:
#   BENCH=libero|libero_plus             # custom bench selection when EVAL=custom
#   PILOT=0                              # 0 means full set; >0 means evenly sampled cases
#   TRIALS=50                            # standard LIBERO default; LIBERO-plus default is 1
#   CKPT=/path/to/step_xxxxxx.pt
#   STATS=/path/to/libero_uncond_2cam224_dataset_stats.json
#   NUM_GPUS=8 GPU_OFFSET=0 MAX_PER_GPU=1 SAVE_VIDEO=false
#   OUT=/path/to/output_dir
#   REDIRECT_COMMON_FILES=false          # required on the H200 local checkpoint layout
#   LIBERO_ORIG_PATH=/data/home/frank/projects/LIBERO
#   LIBERO_CONFIG_PATH=$HOME/.libero_orig
#   EXTRA_OVERRIDES='EVALUATION.replan_steps=10 model.foo=bar'
#   DRY_RUN=1                            # generate shards and print config, but do not launch workers

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

EVAL=${EVAL:-custom}
case "$EVAL" in
    custom)
        BENCH=${BENCH:-libero}
        ;;
    libero_full)
        BENCH=${BENCH:-libero}
        PILOT=${PILOT:-0}
        TRIALS=${TRIALS:-50}
        ;;
    libero_pilot)
        BENCH=${BENCH:-libero}
        PILOT=${PILOT:-8}
        TRIALS=${TRIALS:-1}
        ;;
    plus_full|libero_plus_full)
        BENCH=${BENCH:-libero_plus}
        PILOT=${PILOT:-0}
        TRIALS=${TRIALS:-1}
        INCLUDE_NOISE=${INCLUDE_NOISE:-1}
        ;;
    plus_pilot|libero_plus_pilot|case200)
        BENCH=${BENCH:-libero_plus}
        PILOT=${PILOT:-200}
        TRIALS=${TRIALS:-1}
        INCLUDE_NOISE=${INCLUDE_NOISE:-1}
        ;;
    *)
        echo "[FATAL] EVAL must be one of: custom, libero_full, libero_pilot, plus_full, plus_pilot, case200; got: $EVAL"
        exit 1
        ;;
esac

if [ "$BENCH" != "libero" ] && [ "$BENCH" != "libero_plus" ]; then
    echo "[FATAL] BENCH must be 'libero' or 'libero_plus', got: $BENCH"
    exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] failed to activate conda env: fastwam"; exit 1; }

if [ "$BENCH" = "libero" ]; then
    LIBERO_ORIG_PATH=${LIBERO_ORIG_PATH:-/data/home/frank/projects/LIBERO}
    export PYTHONPATH=$LIBERO_ORIG_PATH${PYTHONPATH:+:$PYTHONPATH}
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero_orig}
    echo "[BENCH=libero] LIBERO_ORIG_PATH=$LIBERO_ORIG_PATH PYTHONPATH=$PYTHONPATH LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
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
DRY_RUN=${DRY_RUN:-0}

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
echo "   EVAL=$EVAL BENCH=$BENCH TRIALS=$TRIALS PILOT=$PILOT INCLUDE_NOISE=$INCLUDE_NOISE"
echo "   CKPT=$CKPT"
echo "   STATS=$STATS"
echo "   NUM_GPUS=$NUM_GPUS GPU_OFFSET=$GPU_OFFSET MAX_PER_GPU=$MAX_PER_GPU NWORKERS=$NWORKERS"
echo "   SAVE_VIDEO=$SAVE_VIDEO REDIRECT_COMMON_FILES=$REDIRECT_COMMON_FILES"
echo "   OUT=$OUT"
[ -n "$EXTRA_OVERRIDES" ] && echo "   EXTRA_OVERRIDES=$EXTRA_OVERRIDES"
echo "   DRY_RUN=$DRY_RUN"
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

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] Shards generated under $OUT/shards; workers were not launched."
    echo "[DRY_RUN] First worker command would run experiments/libero/eval_libero_multi.py with:"
    echo "  ckpt=$CKPT task=libero_uncond_2cam224_1e-4 EVALUATION.num_trials=$TRIALS EVALUATION.dataset_stats_path=$STATS EVALUATION.output_dir=$OUT model.redirect_common_files=$REDIRECT_COMMON_FILES"
    exit 0
fi

# GPU_LIST（空格/逗号分隔的物理 GPU 编号）可覆盖默认的连续映射：worker 按序轮转落在
# 列表内的 GPU 上，可实现「8 个 shard 只用 4 张卡、每卡 2 worker」这类布局。
# 场景：h200-1 上的 GPU 进程会被外部反复静默杀掉，worker 改压指定卡。
GPU_ARR=()
if [ -n "${GPU_LIST:-}" ]; then
    read -r -a GPU_ARR <<< "${GPU_LIST//,/ }"
    echo "GPU_LIST override: workers rotate over physical GPUs [${GPU_ARR[*]}]"
fi

phys_for_worker() {
    local w=$1
    if [ ${#GPU_ARR[@]} -gt 0 ]; then
        echo "${GPU_ARR[$(( w % ${#GPU_ARR[@]} ))]}"
    else
        echo "$(( (w % NUM_GPUS) + GPU_OFFSET ))"
    fi
}

# 启动 shard worker；worker 正常跑完（rc=0）时落 done_w 标记，供 KEEPALIVE 判断完成。
launch_worker() {
    local w=$1 attempt=$2
    local SHARD="$OUT/shards/shard_${w}.json"
    local PHYS; PHYS=$(phys_for_worker "$w")
    local LOG="$OUT/worker_logs/gpu${PHYS}_w${w}$( [ "$attempt" -gt 0 ] && echo "_r${attempt}" ).log"
    CUDA_VISIBLE_DEVICES=$PHYS nohup bash -c "python experiments/libero/eval_libero_multi.py \
        ckpt='$CKPT' \
        task=libero_uncond_2cam224_1e-4 \
        EVALUATION.num_trials=$TRIALS \
        +EVALUATION.save_video=$SAVE_VIDEO \
        +EVALUATION.task_list_file='$SHARD' \
        EVALUATION.dataset_stats_path='$STATS' \
        EVALUATION.output_dir='$OUT' \
        model.redirect_common_files=$REDIRECT_COMMON_FILES \
        gpu_id=$w \
        $EXTRA_OVERRIDES \
        && touch '$OUT/shards/done_w${w}'" \
        > "$LOG" 2>&1 &
    LAST_PID=$!
    echo "  worker w=$w attempt=$attempt -> phys_gpu=$PHYS pid=$LAST_PID log=$LOG"
}

# KEEPALIVE=1：worker 被外部杀掉后自动重启（断点续跑，已有结果自动跳过），直到该
# shard 正常退出（done_w 标记）或达到重试上限。用于会被静默清卡的机器。
KEEPALIVE=${KEEPALIVE:-0}
MAX_RETRY=${MAX_RETRY:-30}

monitor_progress "$OUT" "$TOTAL" &
MON_PID=$!

FAIL=0
if [ "$KEEPALIVE" = "1" ]; then
    declare -A WPID WTRY
    while :; do
        ALL_DONE=1
        for ((w=0; w<NWORKERS; w++)); do
            SHARD="$OUT/shards/shard_${w}.json"
            [ -s "$SHARD" ] || continue
            [ -f "$OUT/shards/done_w${w}" ] && continue
            if [ "${WTRY[$w]:-0}" -gt "$MAX_RETRY" ]; then
                continue    # 已放弃的 shard 不再算入 ALL_DONE 阻塞（计入 FAIL）
            fi
            ALL_DONE=0
            pid=${WPID[$w]:-}
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                continue
            fi
            attempt=${WTRY[$w]:-0}
            if [ -n "$pid" ]; then
                echo "[keepalive $(date '+%m-%d %H:%M:%S')] worker w=$w (pid=$pid) died, relaunching (attempt $((attempt+1))/$MAX_RETRY)"
            fi
            if [ "$attempt" -ge "$MAX_RETRY" ]; then
                echo "[keepalive] worker w=$w exceeded MAX_RETRY=$MAX_RETRY, giving up"
                WTRY[$w]=$((attempt+1)); FAIL=$((FAIL+1))
                continue
            fi
            launch_worker "$w" "$attempt"
            WPID[$w]=$LAST_PID
            WTRY[$w]=$((attempt+1))
            sleep 2
        done
        [ "$ALL_DONE" = 1 ] && break
        sleep 60
    done
else
    PIDS=()
    for ((w=0; w<NWORKERS; w++)); do
        SHARD="$OUT/shards/shard_${w}.json"
        [ -s "$SHARD" ] || { echo "shard_$w is empty, skip"; continue; }
        launch_worker "$w" 0
        PIDS+=("$LAST_PID")
        sleep 2
    done
    echo "Started ${#PIDS[@]} workers."
    echo "  progress: tail -f $OUT/progress.log"
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL+1))
    done
fi
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
