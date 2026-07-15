#!/bin/bash
# mem-stage-v4 history 消融批量评测(推理开关,无需重训)。
#
# 默认:LIBERO-plus 全量(10030 case,INCLUDE_NOISE=1,TRIALS=1),只跑 B/C/D 三个消融:
#   B_video_only  v4 ckpt  history_ablate=video_only  只保留 video history(屏蔽 action history)
#   C_action_only v4 ckpt  history_ablate=action_only 只保留 action history(屏蔽 video history 过去帧)
#   D_no_history  v4 ckpt  history_ablate=no_history  双路屏蔽(仍走 v4 condition cache 路径)
# 可选(通过 CONFIGS 追加):
#   A_full        v4 ckpt  history_ablate=none        完整 v4(已有历史结果时可跳)
#   E_base        base ckpt history_ablate=off        原版 first-frame KV 路径
#
# 用法(在项目主目录,fastwam env 由 eval.sh 内部激活):
#   V4_CKPT=/path/to/step_014470.pt \
#   setsid nohup bash scripts/run_v4_history_ablation.sh > runs/logs/v4_ablation_launcher.log 2>&1 < /dev/null &
#
# 常用覆盖:
#   EVAL=plus_full|libero_full   评测预设(默认 plus_full;INCLUDE_NOISE 由 eval.sh 强制=1)
#   CONFIGS="B_video_only ..."   config 子集与顺序
#   NUM_GPUS/GPU_OFFSET/TRIALS/PILOT 透传 eval.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

# 强制优先使用本 worktree 的 src：节点上 fastwam 包是主仓 editable 安装（可能停在
# 其它分支，如 mem-v5），不含本分支的 drop_history_* 结构性消融参数。
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

V4_CKPT=${V4_CKPT:?"必须提供 V4_CKPT(mem-stage-v4 的 step_014470.pt 路径)"}
BASE_CKPT=${BASE_CKPT:-$ROOT/checkpoints/fastwam_release/libero_uncond_2cam224.pt}
EVAL=${EVAL:-plus_full}
NUM_GPUS=${NUM_GPUS:-8}
GPU_OFFSET=${GPU_OFFSET:-0}
TRIALS=${TRIALS:-}
PILOT=${PILOT:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$ROOT/evaluate_results/v4_history_ablation/${EVAL}_${STAMP}}
CONFIGS=${CONFIGS:-"B_video_only C_action_only D_no_history"}

[ -f "$V4_CKPT" ] || { echo "[FATAL] V4_CKPT not found: $V4_CKPT"; exit 1; }
case " $CONFIGS " in
    *" E_base "*)
        [ -f "$BASE_CKPT" ] || { echo "[FATAL] BASE_CKPT not found: $BASE_CKPT"; exit 1; }
        ;;
esac
mkdir -p "$OUT_ROOT"

run_one() {
    local tag=$1 ckpt=$2 mode=$3
    local out="$OUT_ROOT/$tag"
    if [ -f "$out/DONE" ]; then
        echo "[skip] $tag 已完成"
        return 0
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] ===== START $tag (EVAL=$EVAL history_ablate=$mode ckpt=$(basename "$ckpt")) ====="
    # TRIALS/PILOT 传空串时 eval.sh 内 ${VAR:-default} 会回落预设默认值
    EVAL=$EVAL CKPT="$ckpt" \
    TRIALS="$TRIALS" PILOT="$PILOT" \
    NUM_GPUS=$NUM_GPUS GPU_OFFSET=$GPU_OFFSET MAX_PER_GPU=1 \
    OUT="$out" \
    EXTRA_OVERRIDES="+EVALUATION.history_ablate=$mode" \
    bash scripts/eval.sh
    local rc=$?
    if [ $rc -eq 0 ]; then
        touch "$out/DONE"
    else
        echo "[WARN] $tag 以 rc=$rc 结束(worker 有失败),继续后续 config;结果目录: $out"
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] ===== END $tag rc=$rc ====="
    return 0
}

echo "OUT_ROOT=$OUT_ROOT"
echo "EVAL=$EVAL CONFIGS=$CONFIGS NUM_GPUS=$NUM_GPUS GPU_OFFSET=$GPU_OFFSET TRIALS=${TRIALS:-preset} PILOT=${PILOT:-preset}"
for c in $CONFIGS; do
    case "$c" in
        A_full)        run_one A_full        "$V4_CKPT"   none ;;
        B_video_only)  run_one B_video_only  "$V4_CKPT"   video_only ;;
        C_action_only) run_one C_action_only "$V4_CKPT"   action_only ;;
        D_no_history)  run_one D_no_history  "$V4_CKPT"   no_history ;;
        E_base)        run_one E_base        "$BASE_CKPT" off ;;
        *) echo "[FATAL] unknown config: $c"; exit 1 ;;
    esac
done

echo "=========================================================="
echo "ALL CONFIGS FINISHED. 汇总:"
for c in $CONFIGS; do
    s="$OUT_ROOT/$c"
    if [ -d "$s" ]; then
        echo "--- $c ---"
        if [[ "$EVAL" == plus* ]]; then
            python experiments/libero/summarize_libero_plus.py --output_dir "$s" 2>/dev/null | tail -15
        else
            python experiments/libero/summarize_results.py --output_dir "$s" 2>/dev/null | tail -8
        fi
    fi
done
echo "Result root: $OUT_ROOT"
