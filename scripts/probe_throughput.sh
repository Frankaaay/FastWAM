#!/bin/bash
# ----------------------------------------------------------------------------
# probe_throughput.sh — 扫不同 batch 的「真实吞吐」并判断瓶颈在 IO 还是算力
#
# 和 probe_batch_size.sh 的区别:
#   probe_batch_size.sh 只回答「能装下多大 batch」(看 OOM + 显存峰值)。
#   本脚本回答「哪个 batch 吞吐(samples/s)最高」+「GPU 在等数据还是在算」。
#   ——能装下的最大 batch ≠ 吞吐最高的 batch,逼近显存上限反而会变慢。
#
# 原理:
#   1) 每个候选 batch 冷启动跑 STEPS 步,log_every=1,关存档/eval。
#      trainer 每步打印 "... speed=X step/s, Y samples/s ...",直接解析 Y。
#      丢掉前 WARMUP 步(含编译/数据预热),取稳态步的平均 samples/s。
#   2) 跑的同时后台每秒采样 GPU 利用率(utilization.gpu)和显存:
#        - 利用率稳定 ~100%        → 算力瓶颈(加 batch 不会更快)
#        - 利用率平均偏低/常掉档    → 数据/IO 瓶颈(GPU 在等 dataloader)
#   3) 选 samples/s 最高的 batch 作为推荐(留显存余量)。
#
# 用法(服务器上,脚本会自己激活 conda):
#   bash scripts/probe_throughput.sh
#   GPUS=0,1,2,3 CANDIDATES="16 24 32 48 64" bash scripts/probe_throughput.sh
#
# 判 IO 专项(固定 batch 只扫 num_workers,涨则说明 IO 瓶颈):
#   WORKER_SWEEP="2 4 8 16" WS_BS=16 bash scripts/probe_throughput.sh
#
# 可调环境变量(都有默认值):
#   GPUS         参与的卡,逗号分隔            默认 0,1,2,3
#   CANDIDATES   候选 per-GPU batch(从小到大) 默认 "16 24 32 48 64"
#   STEPS        每个候选跑多少步              默认 24
#   WARMUP       丢弃前多少步再算吞吐          默认 8
#   WORKERS      覆盖 num_workers(可选)
#   WORKER_SWEEP 若设置则只做 num_workers 扫描(空格分隔),不扫 batch
#   WS_BS        WORKER_SWEEP 时用的固定 batch  默认 16
#   COLD_CKPT / NORM_STATS / ACCEL_CFG / CONDA_SH / ENV  同 probe_batch_size.sh

'''mkdir -p runs/throughput_probe
nohup bash scripts/probe_throughput.sh > runs/throughput_probe/probe_summary.log 2>&1 &
echo $! > runs/throughput_probe/probe.pid     # 记下 PID 备查
# 实时看:
tail -f runs/throughput_probe/probe_summary.log
# 想停:kill $(cat runs/throughput_probe/probe.pid)'''
# ----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

CONDA_SH=${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}
ENV=${ENV:-fastwam}
# conda 的 activate.d 脚本(zz-fastwam-libs.sh)在 set -u 下会因引用未定义的
# LD_LIBRARY_PATH 直接报 "unbound variable" 退出 → 激活期间临时关掉 set -u。
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$ENV" || { echo "[FATAL] 无法激活 conda env: $ENV"; exit 1; }
set -u

# DiffSynth 离线(air-gapped H200 必须),缺了会回落联网下载直接挂
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true

GPUS=${GPUS:-0,1,2,3}
STEPS=${STEPS:-24}
WARMUP=${WARMUP:-8}
# 历史窗口帧数(必须 4 的倍数)。stage-1/2 用 16(3.2s);若准备转 H4 训练,
# 用 HISTORY=4 探吞吐才能反映真实显存/吞吐(历史越短显存越省、bs 可更大)。
HISTORY=${HISTORY:-16}
COLD_CKPT=${COLD_CKPT:-checkpoints/fastwam_release/libero_uncond_2cam224.pt}
ACCEL_CFG=${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}
NUM_PROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

# pretrained_norm_stats:默认从正式训练脚本里抠出来,保证和训练一致
TRAIN_SH=scripts/train_mem_stage1_v1.sh
if [ -z "${NORM_STATS:-}" ] && [ -f "$TRAIN_SH" ]; then
    NORM_STATS=$(grep -oE 'pretrained_norm_stats=[^ ]+' "$TRAIN_SH" | head -1 | cut -d= -f2-)
fi
[ -z "${NORM_STATS:-}" ] && { echo "[FATAL] 找不到 pretrained_norm_stats,请用 NORM_STATS=... 显式传入"; exit 1; }

OUT_ROOT="runs/throughput_probe"
mkdir -p "$OUT_ROOT"

# ---- 跑一次,回填全局变量:SPS(稳态平均 samples/s) UMEAN UMIN UPCT(利用率) PEAK(显存MiB) RC ----
run_once () {  # 参数: BS WORKERS_OR_EMPTY LOGTAG
    local BS=$1; local WK=$2; local TAG=$3
    local LOG="$OUT_ROOT/${TAG}.log"
    local OUTDIR="$OUT_ROOT/${TAG}"
    local UFILE="$OUT_ROOT/${TAG}.util"
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"; : > "$UFILE"

    # 后台每秒采样:每卡利用率取最大、显存取最大,各记一行
    ( while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null \
          | awk -F',' '{u=$1+0; m=$2+0; if(u>mu)mu=u; if(m>mm)mm=m} END{print mu, mm}' >> "$UFILE"
        sleep 1
      done ) &
    local SAMPLER=$!

    local WK_ARG=""
    [ -n "$WK" ] && WK_ARG="num_workers=$WK"

    CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
        --config_file "$ACCEL_CFG" --num_processes "$NUM_PROC" \
        scripts/train.py \
        data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
        model.redirect_common_files=false \
        model.vae_memory.enabled=true \
        model.vae_memory.warm_start=true \
        model.vae_memory.train_temporal_only=true \
        data.train.history_video_frames="$HISTORY" \
        +data.train.pretrained_norm_stats="$NORM_STATS" \
        batch_size="$BS" max_steps="$STEPS" log_every=1 $WK_ARG \
        save_every=999999 eval_every=999999 resume=null \
        output_dir="$OUTDIR" \
        > "$LOG" 2>&1
    RC=$?
    kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null

    if grep -qiE "out of memory|CUDA out of memory|OutOfMemoryError" "$LOG"; then
        RC=137  # 标记为 OOM
    fi

    # 解析稳态 samples/s:取 step>WARMUP 的行求平均
    SPS=$(awk -v w="$WARMUP" '
        /\[train\]/ {
            step=0; sps=0;
            for(i=1;i<=NF;i++){
                if($i ~ /^step=/){split($i,a,"=");split(a[2],b,"/");step=b[1]}
                if($i=="samples/s"){sps=$(i-1)+0}
            }
            if(step>w && sps>0){s+=sps;n++}
        }
        END{ if(n>0) printf "%.1f", s/n; else printf "NA" }' "$LOG")

    # 利用率统计:只看 util>=20 的活跃采样 → 平均 / 最低 / 满载占比(>=90)
    read -r UMEAN UMIN UPCT < <(awk '
        {u=$1+0; if(u>=20){s+=u;n++; if(n==1||u<mn)mn=u; if(u>=90)hi++}}
        END{ if(n==0) print "NA NA NA"; else printf "%.0f %.0f %.0f", s/n, mn, 100*hi/n }' "$UFILE")
    PEAK=$(awk '{m=$2+0; if(m>p)p=m} END{print p+0}' "$UFILE")

    rm -rf "$OUTDIR"
}

echo "=========================================================="
echo " 吞吐 / 瓶颈探测   GPUS=$GPUS (num_processes=$NUM_PROC)  STEPS=$STEPS WARMUP=$WARMUP  HISTORY=$HISTORY"
echo " NORM_STATS=$NORM_STATS"
echo "=========================================================="

# ============ 模式 A:固定 batch,扫 num_workers(专门判 IO 瓶颈) ============
if [ -n "${WORKER_SWEEP:-}" ]; then
    WS_BS=${WS_BS:-16}
    echo " [模式] num_workers 扫描  固定 batch=$WS_BS   workers=$WORKER_SWEEP"
    printf "%-10s %-14s %-10s %-10s %-12s %-12s\n" workers samples/s util均值 util最低 满载占比% 显存峰值MiB
    PREV=""
    for WK in $WORKER_SWEEP; do
        run_once "$WS_BS" "$WK" "wk${WK}"
        printf "%-10s %-14s %-10s %-10s %-12s %-12s\n" "$WK" "$SPS" "$UMEAN" "$UMIN" "$UPCT" "$PEAK"
    done
    echo "----------------------------------------------------------"
    echo " 解读:加 worker 吞吐还在涨 → 数据/IO 瓶颈,继续加 worker / prefetch_factor / pin_memory。"
    echo "       吞吐已平 + util≈100% → 算力瓶颈,加 worker 没用。"
    exit 0
fi

# ============ 模式 B:扫 batch,找吞吐最优,并判断瓶颈 ============
CANDIDATES=${CANDIDATES:-"16 24 32 48 64"}
echo " [模式] batch 扫描   candidates=$CANDIDATES"
printf "%-8s %-12s %-14s %-10s %-10s %-12s %-12s\n" batch 全局batch samples/s util均值 util最低 满载占比% 显存峰值MiB
BEST_BS=0; BEST_SPS=0; BEST_UTIL="NA"
for BS in $CANDIDATES; do
    run_once "$BS" "${WORKERS:-}" "bs${BS}"
    if [ "$RC" -eq 137 ]; then
        printf "%-8s %-12s %-14s %-10s %-10s %-12s %-12s\n" "$BS" "$((BS*NUM_PROC))" "OOM" "-" "-" "-" "$PEAK"
        echo "  >> bs=$BS OOM,停止往上扫。"
        break
    elif [ "$RC" -ne 0 ]; then
        printf "%-8s %-12s %-14s\n" "$BS" "$((BS*NUM_PROC))" "ERR(rc=$RC)"
        echo "  >> bs=$BS 非 OOM 报错,看日志 $OUT_ROOT/bs${BS}.log"
        break
    fi
    printf "%-8s %-12s %-14s %-10s %-10s %-12s %-12s\n" "$BS" "$((BS*NUM_PROC))" "$SPS" "$UMEAN" "$UMIN" "$UPCT" "$PEAK"
    # 记录吞吐最优(SPS 是浮点,用 awk 比较)
    if [ "$SPS" != "NA" ] && awk "BEGIN{exit !($SPS>$BEST_SPS)}"; then
        BEST_SPS=$SPS; BEST_BS=$BS; BEST_UTIL=$UMEAN
    fi
done

echo "=========================================================="
if [ "$BEST_BS" -gt 0 ]; then
    echo " 吞吐最优 per-GPU batch = $BEST_BS  (全局 $((BEST_BS*NUM_PROC)))  ≈ $BEST_SPS samples/s"
    echo " ——注意这是单卡吞吐最高点,不一定是能装下的最大 batch;正式训练取它即可。"
    echo ""
    if [ "$BEST_UTIL" != "NA" ] && [ "$BEST_UTIL" -ge 90 ]; then
        echo " 瓶颈判定:算力瓶颈(util≈${BEST_UTIL}%,GPU 一直在算)。"
        echo "   → 加 batch / 加卡都不会提高单位吞吐;想更快只能减 epoch、降分辨率、或换更快的 kernel/精度。"
    else
        echo " 瓶颈判定:疑似数据/IO 瓶颈(util 仅 ${BEST_UTIL}%,GPU 在等数据)。"
        echo "   → 先跑:WORKER_SWEEP=\"4 8 16 24\" WS_BS=$BEST_BS bash scripts/probe_throughput.sh"
        echo "     若吞吐随 worker 上涨,就调大 num_workers / prefetch_factor / persistent_workers / pin_memory,"
        echo "     或把数据预处理成更快的格式(webdataset / 预解码 latent)。"
    fi
else
    echo " 没有跑通的候选,检查日志:$OUT_ROOT/*.log"
fi
echo "=========================================================="
