#!/bin/bash
# ----------------------------------------------------------------------------
# probe_batch_size.sh — 探测当前显卡能吃多大的 per-GPU batch_size
#
# 原理:用和正式训练完全一致的模型/memory 配置，冷启动只跑几步（max_steps），
#       关掉存档和 eval，从小到大逐个试候选 batch；某个值 OOM 就停，
#       最后一个不 OOM 的就是上限。运行期间采样每卡显存峰值，方便留安全余量。
#
# 用法（在服务器上、激活 conda env 之前都行，脚本会自己激活）:
#   bash scripts/probe_batch_size.sh
#   GPUS=0,1,2,3 CANDIDATES="16 24 32" bash scripts/probe_batch_size.sh
#
# 可调环境变量（都有默认值）:
#   GPUS         参与探测的卡，逗号分隔        默认 0,1,2,3
#   CANDIDATES   候选 per-GPU batch（从小到大） 默认 "16 24 32 40"
#   STEPS        每个候选跑多少步               默认 6
#   COLD_CKPT    冷启动权重                     默认 checkpoints/fastwam_release/libero_uncond_2cam224.pt
#   NORM_STATS   pretrained_norm_stats 路径     默认从 scripts/train_mem_temporal.sh 自动提取
#   CONDA_SH / ENV  conda 初始化脚本 / 环境名
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# ---- conda 环境 ----
CONDA_SH=${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
ENV=${ENV:-fastwam}
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$ENV" || { echo "[FATAL] 无法激活 conda env: $ENV"; exit 1; }

# ---- 参数 ----
GPUS=${GPUS:-0,1,2,3}
CANDIDATES=${CANDIDATES:-"16 24 32 40"}
STEPS=${STEPS:-6}
COLD_CKPT=${COLD_CKPT:-checkpoints/fastwam_release/libero_uncond_2cam224.pt}
ACCEL_CFG=${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}
NUM_PROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# pretrained_norm_stats：默认从正式训练脚本里抠出来，保证和训练一致
TRAIN_SH=scripts/train_mem_temporal.sh
if [ -z "${NORM_STATS:-}" ] && [ -f "$TRAIN_SH" ]; then
    NORM_STATS=$(grep -oE 'pretrained_norm_stats=[^ ]+' "$TRAIN_SH" | head -1 | cut -d= -f2-)
fi

echo "=========================================================="
echo " batch-size 探测"
echo "   GPUS=$GPUS  (num_processes=$NUM_PROC)"
echo "   CANDIDATES=$CANDIDATES   STEPS/候选=$STEPS"
echo "   COLD_CKPT=$COLD_CKPT"
echo "   NORM_STATS=${NORM_STATS:-<未设置>}"
echo "=========================================================="
[ -z "${NORM_STATS:-}" ] && { echo "[FATAL] 找不到 pretrained_norm_stats，请用 NORM_STATS=... 显式传入"; exit 1; }

OUT_ROOT="runs/batch_probe"
mkdir -p "$OUT_ROOT"
BEST=0

for BS in $CANDIDATES; do
    echo ""
    echo "---------- 试 per-GPU batch_size=$BS （全局=$((BS*NUM_PROC))）----------"
    LOG="$OUT_ROOT/probe_bs${BS}.log"
    OUTDIR="$OUT_ROOT/bs${BS}"
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

    # 后台采样显存峰值（只看参与的卡）
    PEAK_FILE="$OUT_ROOT/peak_bs${BS}.txt"; echo 0 > "$PEAK_FILE"
    ( while true; do
        used=$(CUDA_VISIBLE_DEVICES="$GPUS" nvidia-smi --query-gpu=memory.used \
               --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | sort -n | tail -1)
        cur=$(cat "$PEAK_FILE" 2>/dev/null || echo 0)
        [ -n "$used" ] && [ "$used" -gt "$cur" ] 2>/dev/null && echo "$used" > "$PEAK_FILE"
        sleep 2
      done ) &
    SAMPLER=$!

    CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
        --config_file "$ACCEL_CFG" --num_processes "$NUM_PROC" \
        scripts/train.py \
        data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
        model.vae_memory.enabled=true \
        model.vae_memory.warm_start=true \
        model.vae_memory.train_temporal_only=true \
        data.train.history_video_frames=16 \
        +data.train.pretrained_norm_stats="$NORM_STATS" \
        batch_size="$BS" max_steps="$STEPS" \
        save_every=999999 eval_every=999999 resume=null \
        output_dir="$OUTDIR" \
        > "$LOG" 2>&1
    RC=$?

    kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
    PEAK=$(cat "$PEAK_FILE" 2>/dev/null || echo "?")

    if grep -qiE "out of memory|CUDA out of memory|OutOfMemoryError" "$LOG"; then
        echo ">> batch=$BS  ❌ OOM（峰值 ${PEAK} MiB），停止往上试。日志: $LOG"
        break
    elif [ "$RC" -ne 0 ]; then
        echo ">> batch=$BS  ⚠️ 非 OOM 报错(rc=$RC)，请看日志: $LOG"
        break
    else
        echo ">> batch=$BS  ✅ 跑通，单卡峰值 ${PEAK} MiB / 143771 MiB"
        BEST=$BS
    fi
    rm -rf "$OUTDIR"   # 通过的清掉，省磁盘
done

echo ""
echo "=========================================================="
if [ "$BEST" -gt 0 ]; then
    echo " 能跑通的最大 per-GPU batch_size = $BEST  （全局 = $((BEST*NUM_PROC))）"
    echo " 建议正式训练取比它略小一档、显存峰值离 143GB 留 ~10% 余量。"
else
    echo " 最小候选都没跑通，检查日志: $OUT_ROOT/probe_bs*.log"
fi
echo "=========================================================="
