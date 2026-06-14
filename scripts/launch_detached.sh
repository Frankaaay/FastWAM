#!/bin/bash
cd ~/projects/FastWAM
mkdir -p runs/mem_temporal_libero
LOG=runs/mem_temporal_libero/train_$(date +%Y%m%d_%H%M%S).log
echo "$LOG" > runs/mem_temporal_libero/latest_log_path.txt
setsid bash scripts/train_mem_temporal.sh > "$LOG" 2>&1 < /dev/null &
echo $! > runs/mem_temporal_libero/train.pid
echo "LAUNCHED pid=$(cat runs/mem_temporal_libero/train.pid) log=$LOG"
