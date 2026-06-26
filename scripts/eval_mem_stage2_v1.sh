#!/bin/bash
# ----------------------------------------------------------------------------
# eval_mem_stage2_v1.sh — MEM-stage2(prepend 路线)v1 的 LIBERO-plus eval 包装。
#
# 只是把 stage2 必需的配置锁死,再 exec 通用的 scripts/eval_libero_plus.sh:
#   - VAE_MEM=false + DIT_PREPEND=true   -> 走「冻结 VAE plain-encode 历史 -> prepend
#       到 video 序列」的推理路径(runtime.py: dit_history_memory),与训练一致;
#   - HISTORY=5                          -> H5(4n+1,K_lat=2),与 train_mem_stage2_v1.sh 对齐;
#   - INCLUDE_NOISE=1                    -> 全 10030(含 Sensor Noise),对齐 paper(项目硬规则)。
#
# 用法:
#   bash scripts/eval_mem_stage2_v1.sh                 # 自动选 runs/mem_stage2_v1 最新 weights ckpt
#   STEP=4000 bash scripts/eval_mem_stage2_v1.sh       # 指定档位(weights/step_004000.pt)
#   CKPT=/path/to.pt bash scripts/eval_mem_stage2_v1.sh# 显式指定 ckpt
#   PILOT=16 bash scripts/eval_mem_stage2_v1.sh        # ⚠️ 全量前先跑一次:验证 prepend 推理链路通
#   NUM_GPUS=4 GPU_OFFSET=4 ...                         # 训练还占着 0-3 时,把 eval 钉在 4-7
#   BENCH=libero bash scripts/eval_mem_stage2_v1.sh    # 原版未扰动标准 LIBERO(基线 95.9,默认 TRIALS=5)
#   BENCH=libero TRIALS=3 bash scripts/eval_mem_stage2_v1.sh
#
# ⚠️ 第一次评 stage2 强烈建议先 PILOT=16:smoke 只验证了「训练 forward」,推理 rollout
#    的 prepend 路径(prefill_video_cache 喂 buffered 历史)是另一条代码路径,先小样确认
#    不报错、成绩不是 0,再上全 10030。
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

RUN_DIR=${RUN_DIR:-runs/mem_stage2_v1}
WEIGHTS_DIR="$RUN_DIR/checkpoints/weights"

# CKPT 优先级:显式 CKPT > STEP 指定 > 最新一档
if [ -z "${CKPT:-}" ]; then
  if [ -n "${STEP:-}" ]; then
    CKPT=$(printf "%s/step_%06d.pt" "$WEIGHTS_DIR" "$STEP")
  else
    CKPT=$(ls -1 "$WEIGHTS_DIR"/step_*.pt 2>/dev/null | sort -V | tail -1)
  fi
fi
[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || {
  echo "[FATAL] 找不到 stage2-v1 ckpt(CKPT=${CKPT:-未解析})。"
  echo "        现有档位:"; ls -1 "$WEIGHTS_DIR"/step_*.pt 2>/dev/null || echo "        (weights 目录为空,训练还没存档?)"
  exit 1
}

STEP_TAG=$(basename "$CKPT" .pt)   # e.g. step_004000
export CKPT
export VAE_MEM=false
export DIT_PREPEND=true
export HISTORY=${HISTORY:-5}
export STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

# BENCH=libero_plus(默认,扰动鲁棒性 + INCLUDE_NOISE=1 硬规则)
#       libero(原版未扰动标准集,对照基线 95.9;eval_libero_plus.sh 内部用环境变量切回原版)
BENCH=${BENCH:-libero_plus}
export BENCH
if [ "$BENCH" = "libero" ]; then
  export TRIALS=${TRIALS:-5}       # 标准 libero 习惯多 trial 取均值(40 task ×5 = 200 ep)
  export OUT=${OUT:-./evaluate_results/libero_std/mem_stage2_v1/${STEP_TAG}_$(date +%Y%m%d_%H%M%S)}
  BASELINE="对照基线 标准 libero 95.9"
else
  export INCLUDE_NOISE=1           # 硬规则:libero-plus stage2 eval 也必须全 10030,不可关
  export OUT=${OUT:-./evaluate_results/libero_plus/mem_stage2_v1/${STEP_TAG}_$(date +%Y%m%d_%H%M%S)}
  BASELINE="对照基线 mem-off 50.5"
fi

echo "[stage2-v1 eval] BENCH=$BENCH CKPT=$CKPT"
echo "[stage2-v1 eval] VAE_MEM=false DIT_PREPEND=true HISTORY=$HISTORY TRIALS=${TRIALS:-1} INCLUDE_NOISE=${INCLUDE_NOISE:-0} PILOT=${PILOT:-0}"
echo "[stage2-v1 eval] OUT=$OUT  $BASELINE"

exec bash scripts/eval_libero_plus.sh
