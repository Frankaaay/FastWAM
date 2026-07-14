# 26-07-14 MemoryBench v4 迁移至 h200-2 并启动 VAE cache precompute

## 目的

h200-2 8 卡空闲,把 `run/memorybench-v4-smoke`(= 本地 `feat/memorybench-v4-eval-adapter`,commit `3c7bfb4`)所需的全部内容迁移到 h200-2,启动 v2 数据的 VAE cache precompute,并经内置 watcher 自动接续原版/v4 双模型训练。

## 分支/commit

- 分支:`run/memorybench-v4-smoke` @ `3c7bfb4`(fix:训练watcher)
- 迁移方式:h200-1 `git bundle create` → jump host NFS 复制到 `/data-214-30-239-42/home/frank/tmp/memorybench-v4-3c7bfb4.bundle` → h200-2 `git fetch <bundle>` + `git worktree add`(不经外网,不影响 h200-2 现有 mem-v5 主 worktree)

## 运行位置

- 节点:h200-qinghua-2
- worktree:`/data/home/frank/projects/FastWAM-memorybench-v4`(FastWAM 主仓的 worktree,`checkpoints` 软链到主仓 checkpoints)

## 迁移前校验(全部通过)

- v2 数据已在 `/data/shared/offline/datasets/memorybench/lerobot/{memorybench_short_train_v2,memorybench_short_test_v2}`(15G,此前经 rsync 搬运)
- text_embeds_cache:5 个 `*.t5_len128.wan22ti2v5b.pt`,满足脚本 ≥5 要求
- checkpoints:h200-1 与 h200-2 `fastwam_release/` + `Wan-AI/` 逐文件大小完全一致(含 release ckpt 12G、Wan2.2_VAE.pth 2.7G、T5 11G)
- 数据集构建校验:两个 task(`memorybench_short_v4_1e-5`、`memorybench_short_fastwam_original_4gpu_1e-5`)均 size=93,027、fingerprint=`d20e3b119e06ad7f`,与预期一致(两任务共用同一份 cache)
- 磁盘:h200-2 `/data` 剩余 6.0T
- 可加载性:precompute 内置 smoke(4 样本)已实际加载 VAE 并通过 `validate_memorybench_vae_cache.py` 抽样校验

## 关键命令

```bash
# h200-2
cd /data/home/frank/projects/FastWAM-memorybench-v4
setsid nohup bash scripts/precompute_memorybench_vae_cache_4gpu.sh > runs/precompute_launcher.log 2>&1 < /dev/null &
```

## 状态:双训练进行中(13:25 起)

- precompute run_id:`memorybench_vae_v2_20260714_125814`,GPU 0-3,4 shard × 23,256 样本,约 30 分钟完成
- 全量校验通过:恰好 **93,027** 个 cache,fingerprint `d20e3b119e06ad7f`,`full_validation.json` status=ok
- cache 输出:`/data/shared/offline/datasets/memorybench/vae_latent_cache/memorybench_short_wan22/d20e3b119e06ad7f/`
- 双模型 1-step smoke 通过(13:19),正式训练 13:25 启动:
  - 原版:`memorybench_original_v2_full_20260714_132519`,GPU 0-3,约 1.12 step/s,ETA ~2.5h
  - v4:`memorybench_v4_v2_full_20260714_132519`,GPU 4-7,约 0.79 step/s,ETA ~3.5h
  - 参数一致:BS 24/卡、LR 1e-5、10,000 steps、GEMM 对齐、只存最终 ckpt
- 训练日志:`runs/logs/memorybench_{original,v4}_v2_full_20260714_132519.log`
- wandb offline 目录已生成(`e6n28py1` v4 / `gtflbzka` original),由 jump 上 node-2 daemon 每 600s 同步至云端 `fastwam-mem`

## 监控命令

```bash
ssh h200-qinghua-2
cd /data/home/frank/projects/FastWAM-memorybench-v4
tail -f runs/precompute_launcher.log      # precompute → 校验 → watcher 交接
ls runs/logs/                              # 训练开始后的双模型日志
nvidia-smi
```

## 备注 / 下一步

- h200-1 上的 MemoryBench watcher 此前已全部停止,本次未在 h200-1 启动任何任务;bundle 临时文件在两节点 `~/tmp/` 下,可后续清理。
- wandb:两个 memorybench task 配置均为 `enabled=true, mode=offline`(entity `yichx14-uc-irvine` / project `fastwam-mem` / group `memorybench-short-v2-comparison`)。已在 jump 上为 h200-2 补第二个同步 daemon(复用同一 `sync_daemon.sh`/venv/API key,`SCAN_ROOT=/data-214-30-239-42/home/frank/projects/FastWAM-memorybench-v4/runs`,每 600s 增量 sync,日志 `/data-214-30-239-40/home/frank/wandb_sync/daemon_node2.log`)。原 node-1 daemon 不受影响,两个并存。
- 训练完成后按计划分别跑各自的闭环 MemoryBench rollout(`eval_memorybench_short_fastwam_original.sh` / `eval_memorybench_short_v4.sh`)比较 benchmark 分数。
