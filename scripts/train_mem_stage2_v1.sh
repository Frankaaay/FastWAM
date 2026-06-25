#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage2_v1.sh — MEM-stage2(prepend 路线)v1 正式训练(option C / 文档 6.1)
#
# 命名:stage2 = DiT-side history prepend 路线;v1 = 该路线第一版。
# 与旧"改VAE"路线区别(勿与 train_mem_stage1_v1/v2/v3.sh 混淆,那是被废弃的 VAE-temporal 路线):
#   - 彻底关掉 VAE temporal(vae_memory.enabled=false,零新增 nn.Parameter);
#   - 冻结 VAE 只做 plain 逐帧 encode,历史 latent 帧 PREPEND 到 video token 序列最前面
#     (vae_memory.dit_prepend=true),由 video expert 自身 self-attention 混合,
#     action 通过既有 prefill+cache 读到被历史增强过的 current-frame K/V;
#   - vae_memory.enabled=false → trainer 默认分支:全 DiT(video+action expert ~5.9B)
#     + proprio 可训,其余冻结(正是 option C 想要的)。
#
# 设计要点(为什么避开 stage1-v1/v2/v3 的坑):
#   - current → 不看 future(mask):current 的 K/V(action 读的 memory)与 future 是否
#     存在无关 → 训练(有 future)与推理(无 future)产出一致,修掉 stage1-v3 的 train/infer 裂缝;
#   - 不碰 conditioning latent:current 帧 = base 原版 plain encode,逐位一致;
#   - base ckpt strict 加载(无 missing/unexpected key)。
#
# ⚠️ 历史窗口必须 4n+1。冻结 VAE 的 plain encode 把首帧单独成一个 chunk、之后每 4 帧一个
#    chunk(iter_ = 1 + (K-1)//4),历史单独编码时若用 4n(如 H4)会静默丢掉最近 3 帧。
#    H5(5 帧=1.0s)→ [h0][h1..h4] 全编码、0 丢失、K_lat=2。HISTORY 可改 9/13 做消融。
#    seconds = frames * action_video_freq_ratio / fps = frames * 4 / 20。
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
# prepend 路线给 video 序列多了 K_lat 历史帧,bs32 激活显存踩线(首步 OOM,仅超 ~184MiB,
# 且有 ~214MiB 碎片化 reserved)。expandable_segments 消碎片,保住 global 256。仍 OOM 则降 BS=24。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 历史像素帧数,必须 4n+1(H5=1.0s, H9=1.8s, H13=2.6s)。
HISTORY=${HISTORY:-5}
# per-GPU batch(8 卡 → global = BS×8)。正式训练用 24=global192(bs32 会 OOM,且步数/epoch
# 与 step_006000 的 resume sampler 偏移对不上)。如需改回先确认显存与 resume 一致性。
BS=${BS:-24}

# Auto-resume:有 DeepSpeed 训练 state 就续(ZeRO 不支持改卡数,续训必须保持 8 卡);
# 否则冷启动 —— 从 base 原版 ckpt(mem-off 49.83)weights-only 加载,全 DiT 联合微调。
LATEST_STATE=$(ls -d runs/mem_stage2_v1/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_STATE" ]; then
  RESUME="$LATEST_STATE"
  echo "[resume] continuing full training state from $RESUME (须保持 8 卡)"
else
  RESUME="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  echo "[cold-start] weights-only from $RESUME ; dit_prepend=true ; H${HISTORY}"
fi

# 8 卡 × per-GPU 32 × grad_accum 1 = 全局 batch 256(节点被占时把 batch_size 降到 24)
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 8 \
  scripts/train.py \
  data=libero_2cam model=fastwam task=libero_uncond_2cam224_1e-4 \
  batch_size="$BS" \
  learning_rate=1e-5 \
  num_epochs=10 \
  max_steps=null \
  save_every=1000 \
  eval_every=100000 \
  model.redirect_common_files=false \
  model.vae_memory.enabled=false \
  model.vae_memory.dit_prepend=true \
  data.train.history_video_frames="$HISTORY" \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=10 \
  wandb.enabled=true \
  wandb.mode=offline \
  wandb.workspace=yichx14-uc-irvine \
  wandb.project=fastwam-mem \
  wandb.name=mem_stage2_v1 \
  resume="$RESUME" \
  output_dir=./runs/mem_stage2_v1
