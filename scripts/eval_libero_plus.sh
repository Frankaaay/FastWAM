#!/bin/bash
# ----------------------------------------------------------------------------
# eval_libero_plus.sh — LIBERO-plus 鲁棒性 eval(10,030 个 case,每个 1 trial)
#
# 对齐 robustness paper "Do World Action Models Generalize Better than VLAs?"
# 的协议(num_trials_per_task = 1),跑完按 7 个扰动 factor 聚合成绩,可直接和
# paper Table 4 的 Fast-WAM 行逐列比较。
#
# 与全量 LIBERO eval 的区别:这里用 eval_libero_multi.py —— 每张卡只加载一次
# 5B 模型,然后循环跑分配给它的那一片 case(否则一进程一 case 光加载就十几小时)。
# worker 用 nohup 后台起,绕开 tmux;每个 case 独立 results.json,可断点续跑。
#
# 前置(只做一次):
#   - LIBERO-plus 已 `pip install -e . --no-deps`(libero 包指向 LIBERO-plus)
#   - assets.zip 已解压到 LIBERO-plus/libero/libero/assets/
#   - ~/.libero/config.yaml 指向 LIBERO-plus(assets/bddl/benchmark_root/init_states)
#
# 用法:
#   bash scripts/eval_libero_plus.sh                  # 6 factor + Original(跳过 Noise)
#   PILOT=16 bash scripts/eval_libero_plus.sh         # 小样验证(只跑 16 个 case)
#   INCLUDE_NOISE=1 bash scripts/eval_libero_plus.sh  # 含 Noise(需先装 wand+libMagickWand+skimage)
#
# 可调环境变量:
#   CKPT / STATS / NUM_GPUS=8 / MAX_PER_GPU=2 / HISTORY=16
#   PILOT=0(>0 则只跑这么多 case,跨 factor 均匀取样,用于验证全链路)
#   INCLUDE_NOISE=0(1 则包含 Sensor Noise 这 1601 个 case)
#   OUT(默认带时间戳,绝不覆盖旧结果)
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] 无法激活 conda env: fastwam"; exit 1; }

export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export TOKENIZERS_PARALLELISM=false

CKPT=${CKPT:-runs/mem_temporal_libero/checkpoints/weights/step_021700.pt}
STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
NUM_GPUS=${NUM_GPUS:-8}
MAX_PER_GPU=${MAX_PER_GPU:-2}
HISTORY=${HISTORY:-16}
PILOT=${PILOT:-0}
INCLUDE_NOISE=${INCLUDE_NOISE:-0}
OUT=${OUT:-./evaluate_results/libero_plus/libero_uncond_2cam224_1e-4/$(date +%Y%m%d_%H%M%S)}

[ -f "$CKPT" ]  || { echo "[FATAL] 找不到 ckpt: $CKPT"; exit 1; }
[ -f "$STATS" ] || { echo "[FATAL] 找不到 dataset_stats: $STATS"; exit 1; }

NWORKERS=$((NUM_GPUS * MAX_PER_GPU))
mkdir -p "$OUT/shards" "$OUT/worker_logs"

echo "=========================================================="
echo " LIBERO-plus eval"
echo "   CKPT=$CKPT"
echo "   NUM_GPUS=$NUM_GPUS  MAX_PER_GPU=$MAX_PER_GPU  NWORKERS=$NWORKERS"
echo "   PILOT=$PILOT  INCLUDE_NOISE=$INCLUDE_NOISE"
echo "   OUT=$OUT"
echo "=========================================================="

# ---- 生成分片:读 task_classification.json,展平成 case 列表,round-robin 切到 NWORKERS 个 shard ----
python - "$OUT/shards" "$NWORKERS" "$PILOT" "$INCLUDE_NOISE" <<'PY'
import json, os, sys
import libero.libero as L

shard_dir, nworkers, pilot, include_noise = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cls_path = os.path.join(os.path.dirname(L.__file__), "benchmark", "task_classification.json")
data = json.load(open(cls_path))
NOISE = "Sensor Noise"

# 展平:每项 {suite, task_id(0-based = id-1), name, category, difficulty_level}
cases = []
for suite, lst in data.items():
    for t in lst:
        if (not include_noise) and t.get("category") == NOISE:
            continue
        cases.append({
            "suite": suite,
            "task_id": int(t["id"]) - 1,
            "name": t.get("name"),
            "category": t.get("category"),
            "difficulty_level": t.get("difficulty_level"),
        })

# 按 (category, suite) 排序后 round-robin,让每个 factor/suite 均匀分到各 worker
cases.sort(key=lambda c: (c["category"] or "", c["suite"], c["task_id"]))

if pilot > 0:
    # 跨 factor 均匀取 pilot 个:按排序后等距抽样
    step = max(1, len(cases) // pilot)
    cases = cases[::step][:pilot]

shards = [[] for _ in range(nworkers)]
for i, c in enumerate(cases):
    shards[i % nworkers].append(c)

for w, sc in enumerate(shards):
    json.dump(sc, open(os.path.join(shard_dir, f"shard_{w}.json"), "w"))

from collections import Counter
cat = Counter(c["category"] for c in cases)
print(f"[shards] total_cases={len(cases)} nworkers={nworkers} "
      f"per_worker~{len(cases)//nworkers}")
print("[shards] by factor:", dict(cat))
PY
[ $? -eq 0 ] || { echo "[FATAL] 生成分片失败"; exit 1; }

# ---- 起 worker:第 w 个 worker 用物理卡 (w % NUM_GPUS),gpu_id=w(只用于文件名) ----
PIDS=()
for ((w=0; w<NWORKERS; w++)); do
    SHARD="$OUT/shards/shard_${w}.json"
    [ -s "$SHARD" ] || { echo "shard_$w 为空,跳过"; continue; }
    PHYS=$((w % NUM_GPUS))
    LOG="$OUT/worker_logs/gpu${PHYS}_w${w}.log"
    CUDA_VISIBLE_DEVICES=$PHYS nohup python experiments/libero/eval_libero_multi.py \
        ckpt="$CKPT" \
        task=libero_uncond_2cam224_1e-4 \
        model.vae_memory.enabled=true \
        data.train.history_video_frames="$HISTORY" \
        EVALUATION.num_trials=1 \
        +EVALUATION.save_video=false \
        +EVALUATION.task_list_file="$SHARD" \
        EVALUATION.dataset_stats_path="$STATS" \
        EVALUATION.output_dir="$OUT" \
        gpu_id=$w \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  worker w=$w -> phys_gpu=$PHYS pid=$! log=$LOG"
    sleep 2
done

echo "已起 ${#PIDS[@]} 个 worker,等待全部完成..."
echo "  看进度:  ls $OUT/*/ | grep results.json | wc -l"
echo "  看某 worker: tail -f $OUT/worker_logs/gpu0_w0.log"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL+1))
done

echo "=========================================================="
echo " 所有 worker 退出(失败 worker 数=$FAIL)。聚合成绩:"
python experiments/libero/summarize_libero_plus.py --output_dir "$OUT"
echo "=========================================================="
echo " 结果目录:$OUT"
