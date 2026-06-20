#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage3.sh — VAE-memory 微调 stage-3:全 DiT + temporal 联合 co-adapt
#
# 背景 / 改进点:
#   stage-1(只训 temporal 4 参数,DiT 全冻)→ 45.6,记忆净负。
#   stage-2(额外松 patch_embedding 2 参数)→ 46.9,还是不如 mem-off 49.83。
#   病根:DiT 输入层在「无记忆 latent」上训死,单边解冻输入层 = 接口打架。
#
#   stage-3 换思路 —— 不在 stage-1/2 的歪地基上打补丁:
#     - 从 base 原版 ckpt(已全量 LIBERO 训过、mem-off 49.83 那个)**冷启动**;
#     - memory warm_start=true(temporal 4 参数从 spatial proj 初始化);
#     - **整个 DiT(video+action expert ~5.9B)+ proprio + temporal 一起联合微调**,
#       让网络从一开始就 co-adapt 带记忆的 latent,没有 stage-1→2 的顺序错配;
#     - 历史砍到 H4(0.8s,领导方向),省算力 + 减无关历史干扰;
#     - 8 卡全开,global batch 256。
#
#   仓库只有三档(temporal-only / +patch_embed / 全解冻),没有 LoRA/AdaLN 中间档,
#   所以 train_temporal_only=false 即「全解冻」。全解冻 ≈「在 base 上继续 LIBERO 训、
#   只是把 memory+H4 加进来」,well-posed;鲁棒先验在 Wan2.2 深层特征里,低 lr+少
#   epoch 不会抹掉。靠 save_every=500 密集存档 + PILOT 扫描挑峰值,防过训遗忘。
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

# 历史窗口帧数(必须 4 的倍数;VAE memory 要求 (K+1)%4==1)。这一版集中算力先做 H4;
# 参数保留,以后消融可 HISTORY=8 / 16。H4=0.8s, H8=1.6s, H16=3.2s(ratio4/fps20)。
HISTORY=${HISTORY:-4}

# Auto-resume:有 DeepSpeed 训练 state 就续(注意 ZeRO 不支持改卡数,续训必须保持 8 卡);
# 否则冷启动 —— warm-start 从 base 原版 ckpt 拉权重,temporal 从 spatial proj 初始化。
LATEST_STATE=$(ls -d runs/mem_temporal_libero_stage3/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_STATE" ]; then
  RESUME="$LATEST_STATE"
  echo "[resume] continuing full training state from $RESUME (须保持 8 卡)"
else
  RESUME="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  echo "[cold-start] warm-starting weights from $RESUME (full-DiT + temporal joint)"
fi

# 8 卡 × per-GPU 32 × grad_accum 1 = 全局 batch 256
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 8 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  batch_size=32 \
  learning_rate=1e-5 \
  max_steps=4000 \
  save_every=500 \
  eval_every=100000 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=true \
  model.vae_memory.warm_start=true \
  model.vae_memory.train_temporal_only=false \
  data.train.history_video_frames="$HISTORY" \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=10 \
  wandb.enabled=true \
  wandb.mode=offline \
  wandb.workspace=yichx14-uc-irvine \
  wandb.project=fastwam-mem \
  wandb.name=mem_temporal_libero_stage3 \
  resume="$RESUME" \
  output_dir=./runs/mem_temporal_libero_stage3
