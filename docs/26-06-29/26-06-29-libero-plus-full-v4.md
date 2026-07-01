# 26-06-29 v4 全量 LIBERO-plus 评测（进行中）

## 目的
在 mem-stage-v4（History KV Action-only inference）的 step_014470 checkpoint 上，跑全量
LIBERO-plus（10030 case，INCLUDE_NOISE=1），与论文口径对齐，确认 v4 记忆机制在带扰动场景下
是否真正有效。前置标准 LIBERO（STD, TRIALS=50）已跑完，成功率 **96.5%**，基本无掉点。

## 分支 / commit
- 分支：mem-stage-v4
- eval 脚本：`scripts/eval.sh` @ `ba77a03`（统一 EVAL 预设接口）

## 运行位置
- node-1（ssh h200-qinghua-1）
- 项目主目录：`/data/home/frank/projects/FastWAM`
- 用 8 卡（GPU 0–7）。GPU 1/2 上有他人小显存进程（pi.cpp 12GB / supernova 9GB），
  H200 143GB 显存充足，共存运行，未触碰他人进程。

## 关键命令
```bash
EVAL=plus_full \
CKPT=/data/home/maxliu/projects/FastWAM/runs/libero_uncond_2cam224_1e-4/mem_stage_v4_libero_fastcfg_b24_redirectfix_e43c832_20260629_011015/checkpoints/weights/step_014470.pt \
NUM_GPUS=8 GPU_OFFSET=0 MAX_PER_GPU=1 \
OUT=<见下> \
setsid bash scripts/eval.sh
```
- `EVAL=plus_full` 预设：BENCH=libero_plus、PILOT=0（全量）、TRIALS=1、INCLUDE_NOISE=1。
- REDIRECT_COMMON_FILES 默认 false（redirectfix 已内置，离线用本地 .pth）。

## 路径
- checkpoint：`/data/home/maxliu/.../mem_stage_v4_libero_fastcfg_b24_redirectfix_e43c832_20260629_011015/checkpoints/weights/step_014470.pt`
- 输出目录：`evaluate_results/libero_plus/libero_uncond_2cam224_1e-4/mem_stage_v4_step014470_PLUS_FULL_noise_20260629_232052/`
- 进度日志：上述目录下 `progress.log`
- worker 日志：上述目录下 `worker_logs/gpu*_w*.log`

## 配置确认
- total_cases = 10030（含 Sensor Noise，符合论文口径）
- shards = 8，8 worker 各跑 ~1254 case
- 预计耗时 ~9–10 小时（按 STD 实测 ~0.46 min/rollout 推算）

## 当前结论
- 状态：**进行中**（2026-06-29 23:20 启动）。
- STD（无扰动）96.5%，作为基线对照。
- 全量 plus 结果待 `summarize_libero_plus.py` 汇总。

## 下一步
- 跑完后读取 OUT 目录下汇总（summary.json / summarize_libero_plus.py 输出），
  按 7 类扰动因子拆分成功率，与论文及 base ckpt 对比，判断 v4 记忆是否在扰动场景带来增益。
- 注意：progress.log 的 ETA 在 done < 8 时不可信（8 worker 并行，首个 case 完成前全局 done 偏低）。
