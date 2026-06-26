#!/bin/bash
# ----------------------------------------------------------------------------
# eval_mem_stage2_v2.sh — MEM-stage2-v2(fold-current 路线)eval 包装。
#
# 把 fold-current 必需的配置锁死,再 exec 通用的 scripts/eval_libero_plus.sh:
#   - VAE_MEM=false + DIT_PREPEND=false + DIT_FOLD_CURRENT=true
#       -> 走「冻结 VAE plain-encode 历史 -> TemporalFoldAdapter 折进 current 帧并丢弃
#          历史帧」的推理路径(runtime.py: dit_fold_current),与训练一致;
#   - HISTORY=5   -> H5(4n+1,K_lat=2),与 train_mem_stage2_v2.sh 对齐;
#   - INCLUDE_NOISE=1(libero_plus 时)-> 全 10030,对齐 paper(项目硬规则)。
#
# 用法:
#   bash scripts/eval_mem_stage2_v2.sh                 # 自动选 runs/mem_stage2_v2 最新 weights ckpt
#   STEP=4000 bash scripts/eval_mem_stage2_v2.sh       # 指定档位(weights/step_004000.pt)
#   CKPT=/path/to.pt bash scripts/eval_mem_stage2_v2.sh# 显式指定 ckpt
#   PILOT=16 bash scripts/eval_mem_stage2_v2.sh        # ⚠️ 全量前先跑:验证 fold 推理链路通
#   NUM_GPUS=4 GPU_OFFSET=4 ...                         # 训练还占着 0-3 时,把 eval 钉在 4-7
#   BENCH=libero bash scripts/eval_mem_stage2_v2.sh    # 原版未扰动标准 LIBERO(基线 95.9,默认 TRIALS=5)
#
# ⚠️ 第一次评 stage2-v2 先 PILOT=16:smoke 只验证训练 forward,推理 rollout 的 fold 路径
#    (prefill_video_cache 喂 buffered 历史)是另一条代码路径,先小样确认不报错、成绩不是 0。
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

RUN_DIR=${RUN_DIR:-runs/mem_stage2_v2}
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
  echo "[FATAL] 找不到 stage2-v2 ckpt(CKPT=${CKPT:-未解析})。"
  echo "        现有档位:"; ls -1 "$WEIGHTS_DIR"/step_*.pt 2>/dev/null || echo "        (weights 目录为空,训练还没存档?)"
  exit 1
}

STEP_TAG=$(basename "$CKPT" .pt)   # e.g. step_004000
export CKPT
export VAE_MEM=false
export DIT_PREPEND=false
export DIT_FOLD_CURRENT=true
export HISTORY=${HISTORY:-5}
export STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}

# BENCH=libero_plus(默认,扰动鲁棒性 + INCLUDE_NOISE=1 硬规则)
#       libero(原版未扰动标准集,对照基线 95.9)
BENCH=${BENCH:-libero_plus}
export BENCH
if [ "$BENCH" = "libero" ]; then
  export TRIALS=${TRIALS:-5}
  export OUT=${OUT:-./evaluate_results/libero_std/mem_stage2_v2/${STEP_TAG}_$(date +%Y%m%d_%H%M%S)}
  BASELINE="对照基线 标准 libero 95.9"
else
  export INCLUDE_NOISE=1           # 硬规则:libero-plus eval 必须全 10030
  export OUT=${OUT:-./evaluate_results/libero_plus/mem_stage2_v2/${STEP_TAG}_$(date +%Y%m%d_%H%M%S)}
  BASELINE="对照基线 mem-off 50.5"
fi

echo "[stage2-v2 eval] BENCH=$BENCH CKPT=$CKPT"
echo "[stage2-v2 eval] VAE_MEM=false DIT_PREPEND=false DIT_FOLD_CURRENT=true HISTORY=$HISTORY TRIALS=${TRIALS:-1} INCLUDE_NOISE=${INCLUDE_NOISE:-0} PILOT=${PILOT:-0}"
echo "[stage2-v2 eval] OUT=$OUT  $BASELINE"

exec bash scripts/eval_libero_plus.sh
