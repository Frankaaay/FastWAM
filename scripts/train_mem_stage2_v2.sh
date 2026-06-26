#!/bin/bash
# ----------------------------------------------------------------------------
# train_mem_stage2_v2.sh — MEM-stage2-v2(fold-current 路线)正式训练。
#
# 命名:stage2 = DiT-side 短时记忆路线;v2 = fold-current(在 stage2-v1 prepend 之上的改进)。
#
# 与 stage2-v1(prepend + 全解冻 DiT)的关键区别:
#   - 不再 prepend:历史 latent 帧在 video expert 内由 TemporalFoldAdapter「折」进
#     current 帧的 token,然后丢弃历史帧 → MoT / action expert / mask 看到的序列与 base
#     逐位同构(token 数 / current index / action 接口全不变);
#   - 不再全解冻:整个 base DiT(video+action ~5.9B)+ VAE + proprio 全冻,只训 fold adapter
#     (~百 M 量级,zero-init 门控);trainer 走 fold-current 专属分支。
#
# 为什么能保住 baseline(同时具备 stage1/stage2 各自的优点):
#   - 加法 + 零初始化门控:step0 输出逐位 == base(K=1 / mem-off ≡ base 不变性,构造保证);
#   - 冻结 base → 继承 stage1 的 rollout 稳定性(不会重蹈 stage1-v3 / stage2-v1 全解冻翻车);
#   - current conditioning latent = base 原版 plain encode,逐位一致(不碰 latent,避开 stage1 错配);
#   - history dropout(fold_history_dropout=0.4)→ 模型也见过「无历史」,mem-off 永远在分布内。
#
# ⚠️ 历史窗口必须 4n+1(冻结 VAE plain encode 的 chunk 规则,见 train_mem_stage2_v1.sh)。
#    H5(5 帧=1.0s)→ K_lat=2。HISTORY 可改 9/13 做消融。seconds = frames * 4 / 20。
#
# ⚠️ 先跑 smoke 再上多卡:
#    python scripts/test_fold_current.py            # 无数据、秒级:验证 gate=0≡base 等不变性
#    bash scripts/train_mem_stage2_v2_smoke.sh       # 1 step、真数据:验证训练 forward/backward
# ----------------------------------------------------------------------------
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd ~/projects/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 历史像素帧数,必须 4n+1(H5=1.0s, H9=1.8s, H13=2.6s)。
HISTORY=${HISTORY:-5}
# per-GPU batch(8 卡 → global = BS×8)。fold 只训 adapter、优化器 state 极小,显存余量比
# stage2-v1(全 DiT 可训)大得多,理论上可比 v1 的 24 更高;保守默认 24,OOM 再降,显存富裕可升。
BS=${BS:-24}
# 每样本丢历史概率(mem-off 在分布内)。
FOLD_DROPOUT=${FOLD_DROPOUT:-0.4}

# Auto-resume:有 DeepSpeed state 就续(ZeRO 续训必须保持同卡数);否则从 base 冷启动
# (base mem-off 49.83;strict=False 加载,fold adapter fresh-init + 零门控 → 起点==base)。
LATEST_STATE=$(ls -d runs/mem_stage2_v2/checkpoints/state/step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_STATE" ]; then
  RESUME="$LATEST_STATE"
  echo "[resume] continuing training state from $RESUME (须保持同卡数)"
else
  RESUME="checkpoints/fastwam_release/libero_uncond_2cam224.pt"
  echo "[cold-start] weights-only from $RESUME ; dit_fold_current=true ; H${HISTORY} ; dropout=${FOLD_DROPOUT}"
fi

# 8 卡 × per-GPU BS × grad_accum 1 = 全局 batch BS*8
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
  model.vae_memory.dit_prepend=false \
  model.vae_memory.dit_fold_current=true \
  model.vae_memory.fold_history_dropout="$FOLD_DROPOUT" \
  data.train.history_video_frames="$HISTORY" \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  log_every=10 \
  wandb.enabled=true \
  wandb.mode=offline \
  wandb.workspace=yichx14-uc-irvine \
  wandb.project=fastwam-mem \
  wandb.name=mem_stage2_v2 \
  resume="$RESUME" \
  output_dir=./runs/mem_stage2_v2
