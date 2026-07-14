# 26-07-14 node-1 home 目录清理(4.3T → 1.6T)

## 目的

h200-qinghua-1 上 `/data/home/frank` 占用达 4.3T,几乎全部为历史训练 checkpoint。经用户逐项确认后删除无用 run,释放约 2.7T。

## 运行位置

- 节点:h200-qinghua-1(node-1)
- 操作方式:本地 Claude Code 经 `ssh h200-qinghua-1` 远程执行
- 清理时确认:8×H200 当前全部被其他用户(Jiapeilin)的训练占用,frank 名下无在跑任务,删除不影响任何进行中的作业

## 占用分析结论

- 每个 checkpoint 由 `state/`(约 80G,DeepSpeed 优化器状态,仅续训需要)+ `weights/*.pt`(约 12G,评估加载)组成,state 约为 weights 的 7 倍。
- 大头分布:`FastWAM/runs` 3.7T、`FastWAM-memorybench-v4/runs` 456G、`.cache/huggingface/datasets` 103G、`~/tmp/debug-cli.frank.log` 38G。

## 已删除(经用户确认)

| 路径(相对 `/data/home/frank`) | 大小 | 原因 |
|---|---|---|
| `projects/FastWAM/runs/mem_stage2_v1` | 1.1T | 实验已确认失败(perturbed 18.5 vs 基线 50.5) |
| `projects/FastWAM/runs/stage1-v3` | 729G | 用户确认删除 |
| `projects/FastWAM/runs/stage1-v1` | 262G | 旧迭代,已被替代 |
| `projects/FastWAM/runs/stage1-v2` | 143G | 旧迭代,已被替代 |
| `projects/FastWAM/runs/robotwin_uncond_3cam_384_1e-4`(整个,含 smoke/sanity100/2epoch 三个子 run) | 274G | smoke/sanity + 用户确认删 2epoch |
| `projects/FastWAM-memorybench-v4/runs/.../memorybench_original_smoke_20260710_103448` | 92G | smoke,仅 step_000001 |
| `projects/FastWAM-memorybench-v4/runs/.../memorybench_original_cache_smoke_20260710_115035` | 92G | smoke,仅 step_000001 |
| `projects/FastWAM/runs/libero_uncond_2cam224_v5_idm_1e-4`(整个,仅含 mem_v5_train_smoke_) | 93G | smoke |
| `projects/FastWAM/runs/mem_stage2_v2_smoke` | 25G | smoke,仅 step_000001 |
| `projects/FastWAM/runs/mem_stage2_v1_smoke` | 76K | smoke 日志 |
| `tmp/debug-cli.frank.log` | 38G | 调试日志 |

## 保留(经用户确认)

- `projects/FastWAM/runs/fold_clothv4_v4_2epoch`(1.1T,含 5epoch 911G + 2epoch 183G 两个子 run)——用户明确要求保留,含全部 state。
- `projects/FastWAM-memorybench-v4/runs` 剩余部分(约 272G):`memorybench_original_bs24_4gpu_cache_noeval_20260710_141549`(183G)与 `memorybench_short_v4_1e-5`(92G),为当前 memorybench-v4 工作产物。
- `.cache/huggingface/datasets`(103G)——机器离线、重建需经 jump host,暂留。
- `.conda`(18G)、`wandb-venv` 等环境。

## 结果

- `/data/home/frank`:4.3T → **1.6T**(释放约 2.7T)
- `/data` 整盘:15T/28T(54%)→ **13T/28T(44%)**

## 备注 / 下一步

- home 下有一个名为 `\`(反斜杠)的 223M 目录,是某次命令转义出错产生的 FastWAM 仓库残缺副本(内含 .git、configs、docs 等),本次未删,可后续确认后清理。
- 若后续确认 fold_clothv4 训练不再续训,可删其 `state/` 再省约 880G。
