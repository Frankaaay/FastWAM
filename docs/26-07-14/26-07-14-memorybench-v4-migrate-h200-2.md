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

## 状态:双训练已全部完成(16:38)

- precompute run_id:`memorybench_vae_v2_20260714_125814`,GPU 0-3,4 shard × 23,256 样本,约 30 分钟完成
- 全量校验通过:恰好 **93,027** 个 cache,fingerprint `d20e3b119e06ad7f`,`full_validation.json` status=ok
- cache 输出:`/data/shared/offline/datasets/memorybench/vae_latent_cache/memorybench_short_wan22/d20e3b119e06ad7f/`
- 双模型 1-step smoke 通过(13:19),正式训练 13:25 启动,参数一致:BS 24/卡、LR 1e-5、10,000 steps、GEMM 对齐、只存最终 ckpt
- **原版**:`memorybench_original_v2_full_20260714_132519`,GPU 0-3,15:45 跑满 10,000 步,exit status=0
  - 最终 ckpt:`runs/memorybench_short_fastwam_original_4gpu_1e-5/memorybench_original_v2_full_20260714_132519/checkpoints/weights/step_010000.pt`(12G)
- **v4**:`memorybench_v4_v2_full_20260714_132519`,GPU 4-7,16:38 跑满 10,000 步,exit status=0
  - 最终 ckpt:`runs/memorybench_short_v4_1e-5/memorybench_v4_v2_full_20260714_132519/checkpoints/weights/step_010000.pt`(12G)
- watcher 打出 `[train-done] original and v4 training both completed`,8 卡全部释放(0 MiB / 0%)
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

## Open-loop eval 结果(17:23-17:31,h200-2)

服务器扩容后(`/data` 56T 总 / 34T 可用)启动双 open-loop eval:原版 GPU0、v4 GPU4,单卡单进程,test_v2 数据(stride=32 → 727 样本,22,051 值/维),teacher-forced history,10 inference steps,两者均 0 failures。

| 指标 | 原版 | v4 | v4 相对变化 |
|---|---|---|---|
| norm_mse | 0.011382 | 0.011295 | -0.8% |
| norm_mae | 0.032223 | 0.029667 | **-7.9%** |
| raw_mse | 0.007806 | 0.007123 | **-8.7%** |
| raw_mae | 0.028641 | 0.026079 | **-8.9%** |
| gripper_norm_mse | 0.025135 | 0.033969 | +35%(v4 更差) |

- 逐维看:7 个关节维度 v4 全面更低(norm_mse dim0-6 均优),但 gripper 维(dim7)v4 更差,拉平了整体 norm_mse。
- v4 `use_history=true` 生效,原版 `use_history=false`,符合预期。
- 结果:`evaluate_results/memorybench_open_loop/memorybench_{original,v4}_v2_openloop_eval/results.json`(h200-2 worktree 内)
- 日志:`runs/logs/memorybench_{original,v4}_v2_openloop_eval.log`

## 闭环 eval 基础设施缺口(未解决)

`experiments/memorybench/eval_closed_loop.py` 需要 RLBench 仿真(自定义任务 `put_block_back`/`rearrange_block`/`reopen_drawer`),当前:

- rlbench/pyrep 在两台节点任何 conda env 里都不存在(h200-1 的 `RMBench` env 是另一套 Sapien 风格 benchmark,不是 RLBench)
- CoppeliaSim Player 4.1 只在 h200-1 `/data/shared/offline/sim/`,h200-2 没有
- 自定义任务源码在本地 untracked `third_party/SAM2Act/sam2act/libs/{RLBench,PyRep}`(947M),两台服务器都未部署
- 测试 episodes 在 h200-2 只有未解压 zip:`raw_hf/data/test/{put_block_back,rearrange_block,reopen_drawer}.zip`

搭建方案(待确认):jump 克隆/中转 SAM2Act → NFS 到 h200-2;复制 CoppeliaSim 到 h200-2;clone fastwam env 装 PyRep+RLBench(避免污染训练 env);解压 test zips;之后原版/v4 各占 4 卡按任务并行跑闭环 rollout。

## 备注 / 下一步

- h200-1 上的 MemoryBench watcher 此前已全部停止,本次未在 h200-1 启动任何任务;bundle 临时文件在两节点 `~/tmp/` 下,可后续清理。
- wandb:两个 memorybench task 配置均为 `enabled=true, mode=offline`(entity `yichx14-uc-irvine` / project `fastwam-mem` / group `memorybench-short-v2-comparison`)。已在 jump 上为 h200-2 补第二个同步 daemon(复用同一 `sync_daemon.sh`/venv/API key,`SCAN_ROOT=/data-214-30-239-42/home/frank/projects/FastWAM-memorybench-v4/runs`,每 600s 增量 sync,日志 `/data-214-30-239-40/home/frank/wandb_sync/daemon_node2.log`)。原 node-1 daemon 不受影响,两个并存。
- 下一步:分别跑各自的闭环 MemoryBench rollout(`eval_memorybench_short_fastwam_original.sh` / `eval_memorybench_short_v4.sh`)比较 benchmark 分数(尚未开始;服务器可能先停机维护)。
