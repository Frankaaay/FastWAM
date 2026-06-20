#!/bin/bash
# ----------------------------------------------------------------------------
# eval_std_confirm.sh — 标准 LIBERO(无扰动)确认实验:串行跑两个 ckpt,
#   判别 stage-3 全解冻是否把底座的「闭环 rollout 稳定性」搞坏了。
#
#   base 原版标准 LIBERO = 95.9。判读:
#     - 若某 ckpt 崩到 ~10-40% → 该模型 rollout 整体不稳(训练/rollout 裂缝坐实)
#     - 若仍接近 95%          → 底座没坏,问题只在 libero-plus 的扰动/OOD
#
#   两个跑共用同一 tmux session(libero_test_v3)且都占满 8 卡,只能【串行】。
#   单次几小时,务必 detached 起:
#     cd ~/projects/FastWAM
#     setsid bash -c 'exec > std_confirm.log 2>&1; bash scripts/eval_std_confirm.sh' < /dev/null &
#   看进度:tail -f std_confirm.log   或   tmux attach -t libero_test_v3
#
#   可调:TRIALS(默认 50,对齐 base 95.9 的口径;想快可设 20 仍足以区分崩/不崩)
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1

TRIALS=${TRIALS:-50}

S2_CKPT=runs/mem_temporal_libero_stage2/checkpoints/weights/step_003000.pt
S3_CKPT=runs/mem_temporal_libero_stage3/checkpoints/weights/step_004000.pt

run_one () {
  local tag="$1" ckpt="$2" hist="$3"
  echo "############################################################"
  echo "##  标准 LIBERO 确认  ::  $tag"
  echo "##    CKPT=$ckpt"
  echo "##    HISTORY=$hist  TRIALS=$TRIALS"
  echo "############################################################"
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] 缺 ckpt: $ckpt"
    return
  fi
  CKPT="$ckpt" HISTORY="$hist" TRIALS="$TRIALS" NUM_GPUS=8 MAX_PER_GPU=2 \
    bash scripts/eval_full.sh
  echo "########## DONE $tag (rc=$?) ##########"
  echo
}

run_one "stage2_step3000_H16" "$S2_CKPT" 16
run_one "stage3_step4000_H4"  "$S3_CKPT" 4

echo "########## ALL std-LIBERO confirm DONE ##########"
echo "分数在各自时间戳目录的 summarize_results.py 输出里;按结束时间对应上面两个 tag。"
