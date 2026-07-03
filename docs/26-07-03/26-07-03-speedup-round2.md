# FastWAM 训练/推理加速第二轮：torch.compile、DS overlap、推理 benchmark

更新时间：2026-07-03 16:50 CST

本文件记录 GEMM 对齐修复（见 `26-07-03-gemm-alignment-fix.md`）之后第二轮加速的三条实验线
及最终结果，供复现与后续决策参考。

## TL;DR

| 实验线 | 结论 | 收益 |
| --- | --- | ---: |
| ① MoT torch.compile 区域编译 | **成功，建议开启** | 训练 step −14%；推理延迟 −17% |
| ② DeepSpeed overlap_comm | **负优化，弃用** | backward 917ms → 10.8s（10×劣化） |
| ③ 推理 benchmark 与减步数 | **denoise 占 90%，launch-bound** | 20→10 步：598→346ms（质量需另行验证） |

累计（自本轮优化起点）：训练 step `~4.4s → 1.14s`（**≈3.9×**）；
推理 `infer_action` `598ms → 495ms`（compile），组合减步数可到 ~300ms 量级。

## 前置状态（本轮起点）

- VAE latent cache 已落地（forward −48.7%）。
- GEMM 16B 对齐修复已默认开启（step 2889→1326ms，2.18×）。
- alignfix 后 kernel 归因：elementwise/copy/norm（"other"）~550ms/step 成为第一大类（53%），
  每 step 发射 ~1.74 万 kernel；GEMM 已全部走 `nvjet_*`。

## 实验 ①：MoT torch.compile 区域编译（成功）

### 实现

开关：`model.mot_torch_compile`（默认 false）、`model.mot_torch_compile_mode`（默认 "default"）。
commit：`fadf199`。

- 编译粒度：**per-block 独立 compiled callable**（每个 DiT block 一对），避免同一 code object
  被 30 层复用导致 dynamo cache 冲突；`dynamic=False`；`cache_size_limit` 自动提升。
- 编译范围（`src/fastwam/models/wan22/mot.py`）：
  - `_build_expert_attention_io` 计算体：norm1 + modulate + q/k/v 投影 + q/k RMSNorm + RoPE。
  - `_apply_expert_post_block` 计算体：o-proj + gate + norm3/cross_attn + norm2/modulate + FFN + gate。
  - **不编译** `_mixed_attention`（保持 SDPA backend 选择不受干扰）。
- 安全边界：`mot_checkpoint_mixed_attn=true` 时自动跳过编译并 warning；编译构建或首次调用
  失败时整体回退 eager，训练不会失败。
- 推理同样受益：`prefill_*` / `forward_action_with_condition_cache` 调用点都接了 compiled key。

### 训练验证（同口径 profiling run）

baseline = alignfix run（`profile_trace_alignfix_..._142741`），
compile run = `profile_trace_motcompile_..._160229`，仅加 `model.mot_torch_compile=true`：

| 指标 | alignfix | + compile | 变化 |
| --- | ---: | ---: | ---: |
| step_total | 1326 ms | **1139 ms** | **−14%** |
| forward | 412 ms | 351 ms | −15% |
| backward | 917 ms | 784 ms | −15% |
| other 类 kernel（全 step GPU） | ~550 ms | **~328 ms** | −40% |
| kernel 数 / step | ~17.4 万* | ~9 万* | 减半 |
| 吞吐 | — | 96.55 samples/s（bs24×8 卡） | |

*15-step window 总数换算。

一次性成本：首个 step 编译预热（~120 个编译单元，数分钟）；此后稳定（实测跑到 step 480 无异常）。

**Loss 一致性**：compile run `3.197→2.582` vs alignfix `3.182→2.588`，逐 step 重合于噪声量级，
数值等价性成立（Inductor 融合仅有极小舍入差异）。

### 使用方式

```bash
# 训练（profiling 分支）
bash scripts/train_fold_clothv4_v4_2epoch.sh ... model.mot_torch_compile=true
```

注意：当前任务配置 `mot_checkpoint_mixed_attn=false`（fold_clothv4 任务），compile 可用；
若某任务开了 checkpoint，会自动跳过 compile 并提示。

## 实验 ②：DeepSpeed overlap_comm（负优化，弃用）

变体：`scripts/ds_configs/ds_zero1_overlap_config.json`（`overlap_comm=true` +
`contiguous_gradients=true`），经 `ACCELERATE_CONFIG_FILE` 环境变量启用（commit `f45c245`，
默认路径不变）。

结果（run `profile_trace_dsoverlap_..._154816`）：backward 从 917ms 劣化到 **10.8-14.4s**，
step_total 11.3-15s，从 step 5 起稳定复现，非噪声。已终止 run。

结论：**保持现有默认配置**（`overlap_comm=false, contiguous_gradients=false`）。
若后续重试，应先做只开 `overlap_comm`、不开 `contiguous_gradients` 的隔离实验定位病因；
优先级低（NCCL 仅 ~93ms/step）。

## 实验 ③：推理 benchmark（denoise 占 90%，launch-bound）

### 工具

- `infer_action` 全路径 `model/infer/*` profiler 埋点（`encode_input_image` /
  `encode_history_video` / `history_video_prefill` / `history_action_prefill` /
  `current_video_prefill` / `denoise_step`），profiler 未激活时零开销。
- `scripts/bench_infer_action.py`：单卡合成输入 benchmark（无需 text encoder / 数据集），
  CUDA events 计时 + chrome trace 导出。commit `fb30321`。

注意：合成输入分辨率必须匹配训练管线的 **384×320**（`video_size`，非 Resize 的 240×320；
240 高度的 latent 15 不能被 patch 2 整除会直接报错）。

### 基线剖析（GPU7，bs=1，384×320，history 9 帧，20 步 denoise，随机权重）

| 段 | 耗时 | 占比 |
| --- | ---: | ---: |
| infer_action 全程 | **598 ms**（±1.3） | 100% |
| denoise 循环（20 步 × ~27ms） | ~535 ms | ~90% |
| history prefill + VAE encode | ~60 ms | ~10% |

关键结论：denoise 每步墙钟 ~27ms 但 GPU kernel 仅 ~7-8ms —— **CPU/launch-bound**
（bs=1、32 action token、30 层、海量小 kernel）。因此 compile（减少 launch）与
CUDA graphs（单次 graph launch）是正确方向，堆 GPU 算力无用。

### 延迟矩阵

| 配置 | 延迟 | 相对基线 |
| --- | ---: | ---: |
| eager，20 步 | 598 ms | — |
| + mot_torch_compile，20 步 | **495 ms** | **−17%** |
| eager，10 步 | 346 ms | −42% |
| + mot_torch_compile，10 步 | **290 ms**（±0.6） | **−51%** |

compile 后 denoise 每步 CPU 时间 42.6 → 31.1ms，验证收益来自 launch 开销削减。

**重要告警**：`num_inference_steps` 20→10 的收益是纯速度口径；对 action 质量的影响
必须在真实任务（RoboTwin/真机）上验证后才能采纳。compile 则是数值等价的，可直接采纳。

### 复现命令

```bash
cd /data/home/maxliu/projects/FastWAM
conda activate fastwam && export DIFFSYNTH_SKIP_DOWNLOAD=true
# eager 基线
python scripts/bench_infer_action.py --device cuda:7 --history --height 384 --width 320 \
  --num-inference-steps 20 --out runs/bench_infer_action/eager
# compile 对照（--warmup 5 覆盖编译预热）
python scripts/bench_infer_action.py --device cuda:7 --history --height 384 --width 320 \
  --num-inference-steps 20 --warmup 5 --override model.mot_torch_compile=true \
  --out runs/bench_infer_action/compiled
# 真实权重：加 --ckpt <weights/step_xxxxxx.pt>
```

## 累计成果与建议

训练 step（fold_clothv4_v4，bs24×8 卡 H200）：

```text
~4.4s  (起点)
→ 2.89s (VAE latent cache)
→ 1.33s (GEMM 16B 对齐修复, 默认开启)
→ 1.14s (mot_torch_compile=true)          ≈ 3.9× 累计
```

推理 `infer_action`（bs=1）：`598 → 495ms`（compile，数值等价）；再减步数可到 ~300ms（需质量验证）。

建议落地顺序：

1. 正式训练直接带 `model.mot_torch_compile=true`（首 step 有编译预热；与 checkpoint 互斥自动处理）。
   跑一段完整训练确认 loss 曲线与 eager 全程一致后，再考虑改默认值。
2. 推理部署侧开启 compile；减步数实验放到任务评测里做质量-速度权衡。
3. 后续可选：denoise 循环 CUDA graphs（预期再砍大部分 launch 开销）；NCCL/overlap 线保持关闭。

## 相关产物

- commit：`fadf199`（compile）、`f45c245`（DS overlap 变体）、`fb30321`（推理埋点 + bench）。
- 训练 run：`profile_trace_alignfix_..._142741`（baseline）、`profile_trace_motcompile_..._160229`、
  `profile_trace_dsoverlap_..._154816`（负结果）。
- 推理 bench 输出：H200 `runs/bench_infer_action/{eager,compiled,eager_10step,compiled_10step}/`。
- 归因脚本：`scripts/analyze_trace_gpu_breakdown.py`（trace 对比复用）。
