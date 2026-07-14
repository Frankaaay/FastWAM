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

## 闭环 eval 设施:h200-1 已有,复制到 h200-2(18:07 完成)

更正早前"设施缺失"的判断:闭环设施一直在 **h200-1** 上(26-07-13 曾跑过一次闭环,双模型 0/75),只是位置在 `/data/shared/offline/repos/SAM2Act`,当时未搜索到。h200-1 的构成:

- RLBench fork + 自定义任务:`/data/shared/offline/repos/SAM2Act/sam2act/libs/{RLBench,PyRep}`,以 pip editable 装进 fastwam env
- CoppeliaSim Player 4.1:`/data/shared/offline/sim/CoppeliaSim_Player_V4_1_0_Ubuntu20_04`
- 环境脚本 `/tmp/fastwam_memorybench_env.sh`:conda activate fastwam + `COPPELIASIM_ROOT`/`LD_LIBRARY_PATH`/`QT_QPA_PLATFORM_PLUGIN_PATH` 三变量;运行必须套 `xvfb-run -a`
- 测试 demos:`raw_hf/data/test_unzipped/<task>/`(必须传 `++memorybench_eval.rlbench_dataset_root` 指向它,默认 `test/` 只有 zip)

26-07-13 的 0/75(`h200-1:.../evaluate_results/memorybench_closed_loop/full_20260713_124951`)用的是 v1(7D 无 gripper)ckpt + `gripper_strategy=keep`,episodes 全部跑满 400 步无报错——系统性抓取失败,不代表 memory 能力。

**h200-2 复制过程**(节点直连 ssh 双向被拒,全部经 jump 跨 NFS,不走本地):

1. node-1 打 tar(SAM2Act 954M + CoppeliaSim 272M = 1.2G)→ jump `cp` 跨挂载点(86s)→ node-2 解包。教训:双层 NFS rsync 小文件极慢(~5MB/min),大目录树必须 tar 单文件传
2. `test_unzipped`(5.1G)node-2 上此前已有
3. 依赖 cffi/pycparser/pyquaternion/natsort:从 node-1 fastwam env 的 site-packages 直接 rsync 到 node-2 同路径(同 py3.12 同架构)
4. node-2 fastwam env:`pip install -e .../PyRep --no-deps --no-build-isolation`,RLBench 同理;`import rlbench, pyrep` 通过
5. 仿真 smoke:`xvfb-run -a` 启动 CoppeliaSim headless,三个任务各 reset 成功

## 闭环 eval v2 运行(18:07 起,h200-2)

- launcher:node-2 `/tmp/run_closed_loop_dual.sh`(全文即本节参数),run_dir `evaluate_results/memorybench_closed_loop/full_20260714_180732`
- 切分与 26-07-13 完全一致:原版 GPU0-3 / v4 GPU4-7;put_block_back 0+25、rearrange_block 0+25、reopen_drawer 0+13 与 13+12
- 与上次的关键差别:v2 ckpt(8D 动作含真实 gripper 维)+ `gripper_strategy=last_dim`(上次 keep);其余同(action_horizon=32, replan_steps=10, max_steps=400)
- 每 worker 独立 `hydra.run.dir` 防 8 进程冲突;~49 s/episode,25-ep 分片约 21 分钟
- 踩坑记录:① launcher `set -u` 需先 `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"` 再 source conda;② 漏传 `rlbench_dataset_root` 会退到 `test/` 报 "Can't find the demos"

### 闭环 v2 结果(18:07-18:33):仍然双 0/75

| 分片 | 原版 | v4 |
|---|---|---|
| put_block_back 0+25 | 0/25 | 0/25 |
| rearrange_block 0+25 | 0/25 | 0/25 |
| reopen_drawer 0+13 / 13+12 | 0/13, 0/12 | 0/13, 0/12 |

v2 8D ckpt + `gripper_strategy=last_dim` 依旧全 0;episodes 全部跑满 400 步、`errors=[]`。gripper 策略假设被排除,问题更深。

### 真值回放诊断(18:40-19:10):动作空间错了

用测试集 demo 的真值动作直接开环回放(`/tmp/memorybench_gt_replay*.py`,put_block_back ep0,demo 长 311 步):

| 动作定义 | 动作模式 | 结果 |
|---|---|---|
| `joint_velocities+gripper`(= v2 训练动作) | JointVelocity | **失败**(offset 0/1 都失败) |
| `joint_positions(t+1)+gripper` | JointPosition(absolute) | **成功** reward=1.0 @308 |

rearrange_block、reopen_drawer 的 JointPosition 真值回放同样成功(reward=1.0 @301/@284)。

**结论**:MemoryBench demos 是 waypoint 规划采集,观测到的 joint_velocities 经 JointVelocity 模式回放会漂移,连真值都无法完成任务——**v2 的"速度当动作"路线在闭环里根本不可行,两次 0/75 均为系统性失败,不反映 memory 能力差异**。正确路线是绝对关节位置动作。

### 根因分析(为什么速度动作必然失败)

1. **观测≠指令**:MemoryBench demos 由 waypoint 规划器 + 位置控制采集,原始数据没有 `obs.action`;转换 fallback 用的 `joint_velocities` 是传感器读数(运动的结果),不是控制命令(运动的原因),含 PID 瞬态/接触扰动/采样混叠。
2. **开环积分误差累积**:JointVelocity 回放 = 对含噪导数做无反馈数值积分,位置误差按步累加,300 步后漂移厘米级;抓取需毫米级精度 → proximity sensor 永远不触发。offset 只改对齐,救不了积分发散。
3. **BC 上界 = 真值回放**:模型最好也就完美复现训练动作,而真值回放成功率为 0,所以速度路线训练的策略天花板就是 0。动作分块(32 步 chunk / 10 步 replan)让 chunk 内是纯开环,复合误差(covariate shift)进一步放大。
4. **绝对位置自稳定**:每步命令"到 pos[t+1] 去",误差不积累、每 50ms 重新锚定,真值回放 3/3 成功 → BC 上界回到 100%。
5. open-loop MSE 好看与闭环 0 分不矛盾:teacher-forced 只量单步误差,不暴露复合;v4 关节维比原版准 ~9% 的相对结论仍有效。

## v3 实施(动作空间→绝对关节位置)

代码改动(本地,commit 后部署两节点):

| 文件 | 改动 |
|---|---|
| `scripts/convert_memorybench_to_lerobot.py` | 新增 `--action-shift N`:action[t]=extract(obs[t+N]),末帧重复;v3 用 shift=1(pos[t] 是"原地不动") |
| `scripts/convert_memorybench_short_v3.sh` | 新建:`--action-source joint_positions+gripper_open --action-shift 1`,输出 `lerobot/memorybench_short_{train,test}_v3` |
| `experiments/memorybench/eval_closed_loop.py` | 动作模式可配置 `++memorybench_eval.arm_action_mode`,默认 `joint_position`(absolute,带 ignore_collisions wrapper);results json 记录实际模式 |
| `configs/data/memorybench_short.yaml` | train 数据目录 → `memorybench_short_train_v3` |
| `scripts/{eval,precompute,train,watch_and_train}` 4 个脚本 | v2 → v3 目录/run 名(watcher run 名 `memorybench_{original,v4}_v3_*`) |
| 2 个 task config | wandb group → `memorybench-short-v3-comparison` |

处理器侧确认无需改:`delta_action_dim_mask` 只对 padding 步置零(速度语义下 0=静止合理;位置语义下 padding 步 loss 有 `action_is_pad` 掩码,不影响),`action_state_transforms: null` 无绝对→相对变换,闭环 `normalizer.backward` 与训练对称。

流水线:raw 训练数据只在 node-1 → **node-1 转换 v3**(CPU)→ tar 经 jump 到 node-2 → node-2 重新 precompute(fingerprint 含 dataset_dirs 路径,v3 新目录必然新 fingerprint,~30 分钟)→ 自动串双模型训练(~3.5h)→ 闭环 eval(JointPosition)。

## 备注 / 下一步

- h200-1 上的 MemoryBench watcher 此前已全部停止,本次未在 h200-1 启动任何任务;bundle 临时文件在两节点 `~/tmp/` 下,可后续清理。
- wandb:两个 memorybench task 配置均为 `enabled=true, mode=offline`(entity `yichx14-uc-irvine` / project `fastwam-mem` / group `memorybench-short-v2-comparison`)。已在 jump 上为 h200-2 补第二个同步 daemon(复用同一 `sync_daemon.sh`/venv/API key,`SCAN_ROOT=/data-214-30-239-42/home/frank/projects/FastWAM-memorybench-v4/runs`,每 600s 增量 sync,日志 `/data-214-30-239-40/home/frank/wandb_sync/daemon_node2.log`)。原 node-1 daemon 不受影响,两个并存。
- 下一步:分别跑各自的闭环 MemoryBench rollout(`eval_memorybench_short_fastwam_original.sh` / `eval_memorybench_short_v4.sh`)比较 benchmark 分数(尚未开始;服务器可能先停机维护)。
