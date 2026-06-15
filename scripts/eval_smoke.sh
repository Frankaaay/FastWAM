#!/bin/bash
# ----------------------------------------------------------------------------
# eval_smoke.sh — LIBERO eval 冒烟测试:单任务单卡,先验证整条链路能跑通
#
# 跑通后再上 8 卡全量(experiments/libero/run_libero_manager.py)。
# 脚本会自己激活 conda、设好 DiffSynth 离线 env、打开 MEM 记忆。
#
# 用法(服务器上):
#   bash scripts/eval_smoke.sh
#   GPU=1 SUITE=libero_object TASK_ID=0 TRIALS=10 bash scripts/eval_smoke.sh
#
# 可调环境变量(都有默认值):
#   CKPT     权重 ckpt                默认 runs/mem_temporal_libero/checkpoints/weights/step_021700.pt
#   STATS    dataset_stats.json       默认 checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
#   GPU      用哪张卡                  默认 0
#   SUITE    task_suite_name          默认 libero_spatial
#   TASK_ID  任务号                    默认 0
#   TRIALS   跑几个 trial(冒烟少点)   默认 5
#   HISTORY  history_video_frames     默认 16(要和训练一致)
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] 无法激活 conda env: fastwam"; exit 1; }

# DiffSynth 离线(air-gapped H200 必须),缺了会回落联网下载直接挂
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true

CKPT=${CKPT:-runs/mem_temporal_libero/checkpoints/weights/step_021700.pt}
STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
GPU=${GPU:-0}
SUITE=${SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-0}
TRIALS=${TRIALS:-5}
HISTORY=${HISTORY:-16}

[ -f "$CKPT" ]  || { echo "[FATAL] 找不到 ckpt: $CKPT"; exit 1; }
[ -f "$STATS" ] || { echo "[FATAL] 找不到 dataset_stats: $STATS"; exit 1; }

echo "=========================================================="
echo " LIBERO 冒烟测试"
echo "   CKPT=$CKPT"
echo "   GPU=$GPU  SUITE=$SUITE  TASK_ID=$TASK_ID  TRIALS=$TRIALS  HISTORY=$HISTORY"
echo "=========================================================="

CUDA_VISIBLE_DEVICES="$GPU" python experiments/libero/eval_libero_single.py \
    ckpt="$CKPT" \
    task=libero_uncond_2cam224_1e-4 \
    model.vae_memory.enabled=true \
    data.train.history_video_frames="$HISTORY" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.task_suite_name="$SUITE" \
    EVALUATION.task_id="$TASK_ID" \
    EVALUATION.num_trials="$TRIALS" \
    gpu_id="$GPU"
RC=$?

echo "=========================================================="
if [ "$RC" -eq 0 ]; then
    echo " 冒烟测试结束(rc=0)。看上面有没有 'successes' 行,以及 evaluate_results/ 下的视频。"
    echo " 通了就上全量:experiments/libero/run_libero_manager.py(见对话里的命令)"
else
    echo " 冒烟测试失败 rc=$RC,先排依赖(LIBERO 模拟器/资产、DiffSynth 离线)。"
fi
echo "=========================================================="
exit $RC
