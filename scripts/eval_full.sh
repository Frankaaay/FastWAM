#!/bin/bash
# ----------------------------------------------------------------------------
# eval_full.sh — LIBERO 全量 eval:8 卡并行跑完四个 suite,出 paper Table 2 的分数
#
# 冒烟(scripts/eval_smoke.sh)通过后再上这个。manager 会把任务分发到各卡的
# tmux pane 里跑(session 名 libero_test_v3),全部跑完自动调 summarize_results.py。
#
# 用法(服务器上):
#   bash scripts/eval_full.sh
#   TRIALS=50 NUM_GPUS=8 MAX_PER_GPU=2 bash scripts/eval_full.sh
#
# 看进度:tmux attach -t libero_test_v3   (Ctrl-b d 退出不打断)
#
# 可调环境变量(都有默认值):
#   CKPT        权重 ckpt        默认 runs/mem_temporal_libero/checkpoints/weights/step_021700.pt
#   STATS       dataset_stats    默认 checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
#   TRIALS      每任务 trial 数  默认 50(对齐 paper)
#   NUM_GPUS    用几张卡         默认 8
#   MAX_PER_GPU 每卡并行任务数   默认 2
#   HISTORY     history_video_frames 默认 16(要和训练一致)
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam \
  || { echo "[FATAL] 无法激活 conda env: fastwam"; exit 1; }

# DiffSynth 离线 + 无头渲染:导出后会被 manager 启的 tmux pane 继承,
# worker 脚本(run_libero_parallel_test.sh)里也会再设一遍兜底。
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}

CKPT=${CKPT:-runs/mem_temporal_libero/checkpoints/weights/step_021700.pt}
STATS=${STATS:-checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}
TRIALS=${TRIALS:-50}
NUM_GPUS=${NUM_GPUS:-8}
MAX_PER_GPU=${MAX_PER_GPU:-2}
HISTORY=${HISTORY:-16}

[ -f "$CKPT" ]  || { echo "[FATAL] 找不到 ckpt: $CKPT"; exit 1; }
[ -f "$STATS" ] || { echo "[FATAL] 找不到 dataset_stats: $STATS"; exit 1; }

echo "=========================================================="
echo " LIBERO 全量 eval(8 卡并行)"
echo "   CKPT=$CKPT"
echo "   NUM_GPUS=$NUM_GPUS  MAX_PER_GPU=$MAX_PER_GPU  TRIALS=$TRIALS  HISTORY=$HISTORY"
echo "   进度:tmux attach -t libero_test_v3"
echo "=========================================================="

python experiments/libero/run_libero_manager.py \
    ckpt="$CKPT" \
    task=libero_uncond_2cam224_1e-4 \
    model.vae_memory.enabled=true \
    data.train.history_video_frames="$HISTORY" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.num_trials="$TRIALS" \
    MULTIRUN.num_gpus="$NUM_GPUS" \
    MULTIRUN.max_tasks_per_gpu="$MAX_PER_GPU"
RC=$?

echo "=========================================================="
if [ "$RC" -eq 0 ]; then
    echo " 全量 eval 结束(rc=0)。汇总分数见上面 summarize_results.py 的输出,"
    echo " 以及 evaluate_results/ 下对应时间戳目录里的 json/视频。"
else
    echo " 全量 eval 失败 rc=$RC。看 evaluate_results/.../failed_tasks.txt 和 task_logs/。"
fi
echo "=========================================================="
exit $RC
