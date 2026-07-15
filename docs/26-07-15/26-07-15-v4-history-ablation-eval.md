# 26-07-15 mem-stage-v4 history 消融评测(LIBERO-plus 全量,推理开关)

## 目的

对 mem-stage-v4(History KV Action-only)做 history 分支消融,回答 video history 和 action history 各自的贡献。**不重新训练**:利用 v4 训练期对两路 history 各 20% 独立 branch dropout 的事实,推理时屏蔽对应分支属于训练分布内状态(双路保留 64% / 仅 video 16% / 仅 action 16% / 全无 4%)。

本轮只跑三个消融 configuration(A/E 已有历史结果作参照):

| tag | ckpt | history_ablate | 含义 |
|---|---|---|---|
| B_video_only | v4 step_014470 | video_only | 只保留 video history(屏蔽 action history) |
| C_action_only | v4 step_014470 | action_only | 只保留 action history(屏蔽 video history 过去帧) |
| D_no_history | v4 step_014470 | no_history | 双路屏蔽(仍走 v4 condition cache 路径) |

参照:A(full v4)plus_full 见 26-06-29 评测;STD 96.5。E(base)STD 95.9。

## 分支 / commit

- 分支:`mem-stage-v4`(本文档与代码同 commit)
- 改动文件:
  - `experiments/libero/eval_libero_single.py`:新增 `_apply_history_ablation`,由 `EVALUATION.history_ablate`(hydra,经 eval.sh `EXTRA_OVERRIDES` 透传)控制,五种模式 `none / video_only / action_only / no_history / off`。
  - `scripts/run_v4_history_ablation.sh`:B/C/D 串行批量脚本,默认 `EVAL=plus_full`(10030 case,INCLUDE_NOISE=1,TRIALS=1),支持断点续跑(每 config 目录 `DONE` 标记),单 config 失败不阻塞后续。

## 屏蔽语义(关键设计,26-07-15 下午改为结构性丢弃)

初版用 key-visibility mask(复刻 episode 起步 padding 状态);应用户要求改为**结构性丢弃**:被消融分支的 token 完全不进 DiT(`infer_action` 新增 `drop_history_video/drop_history_action`),两者对分数**数学等价**(mask 掉的 key 对 attention 输出贡献恒为零,v4 实现中被 drop 分支对 condition 内部同样不可见),结构版额外省去被丢弃分支的 prefill 计算并缩短 denoise 阶段 condition K/V。等价性已在 H200 上实测:三种模式 mask vs 结构版同 seed 输出动作 max|Δ| = 7.8e-3(bf16 量化精度级)。**因此 mask 版已跑出的结果与结构版可以直接混用续跑。**

- `video_only`(丢 action):不做 action prefill,condition cache 只含 video 分支;
- `action_only`(丢 video):窗口内过去 raw 帧仍替换为当前帧(因果 VAE 会把历史帧像素混入 current latent,替换等价训练 episode 起步 index-clamp 补帧),VAE 编码后只保留 current latent 帧进 DiT;
- `no_history`:两者同时(仍走 v4 condition cache 路径,current 帧保留);
- `off`:完全不传 history,退回原版 first-frame KV 路径(留给 base ckpt 的 E 配置)。

判读注意:推理时消融衡量"双记忆模型在推理时对该路信息的依赖",给出的单路成绩是偏保守下界,不等价"从头单路训练"。若出现反直觉结果(如 C≈A),再考虑重训单路变体。

## 推理延迟基准(26-07-15,H200 单卡,bf16,20 denoise steps,端到端每次 replan 含 T5+VAE)

脚本 `experiments/libero/bench_v4_history_ablation.py`,warmup 3 + 计时 20 次,结果 JSON 在 h200-1 `~/tmp/bench_v4/bench_v4_history_ablation.json`:

| mode | mean (ms) | Hz | vs A |
|---|---|---|---|
| none(A 完整 v4) | 375.7 | 2.66 | — |
| video_only(B,丢 action hist) | 351.6 | 2.84 | −6.4% |
| action_only(C,丢 video hist) | 381.0 | 2.62 | +1.4%(反而略慢) |
| no_history(D) | 353.9 | 2.83 | −5.8% |
| off(原版路径,≈base 开销) | 339.5 | 2.95 | −9.6% |

结论:
- **v4 双路 history 的全部推理开销仅 ~36 ms/replan(+10.7% vs 原版路径)**,瓶颈在 20 步 denoise 循环本身,condition K/V 长短影响很小;
- 丢 action history 省 ~24 ms——主要省的是 action prefill 那次完整 30 层 action expert 前向,不是 K/V 长度;
- 丢 video history latent **几乎不省时间**(甚至测得 +5ms,应为 kernel 尺寸/dispatch 噪声):video prefill 少一个 latent 帧的收益被淹没;
- 部署上"lite 版砍分支提速"的空间很小,v4 的速度代价本来就低。

顺带修复:history 路径此前每次 replan 都白算一次单帧 VAE 编码(`first_frame_latents` 只被原版路径使用),已门控跳过,上表 A/B/C/D 均已受益。

## 实现与运行环境备注

- `infer_action` 校验放宽:`drop_history_action=True` 时可不传 `history_action`;`history_video` 始终必传(末帧承载 current observation)。
- **节点上 `import fastwam` 解析到主仓 editable 安装(`/data/home/frank/projects/FastWAM`,在 mem-v5 分支)**,不含本分支新参数;`run_v4_history_ablation.sh` 已强制 `PYTHONPATH=$ROOT/src` 前置。此前 v4 eval 一直跑的是 mem-v5 分支的模型代码(其 v4 路径语义兼容,26-06-29 A 结果同此环境)。
- `eval.sh` 新增 `GPU_LIST`(空格/逗号分隔)覆盖 worker→物理卡映射,支持"8 shard 压 4 卡、每卡 2 worker"。

## 运行位置与设施迁移(node-1 → h200-2,经 jump)

node-1 8 卡已被占满(100%),评测目标为 h200-2。h200-2 原缺 LIBERO/plus 全部设施,已迁移:

- LIBERO 原版 repo(424M)→ `/data/home/frank/projects/LIBERO`;`~/.libero_orig` 同步(留作后续 STD 复跑)
- v4 ckpt → `~/ckpts/mem_stage_v4_libero_step_014470.pt`(12G,源:node-1 maxliu run `mem_stage_v4_libero_fastcfg_b24_redirectfix_e43c832_20260629_011015`)
- LIBERO-plus 实体(16G,含 assets)→ `/data/shared/offline/datasets/LIBERO-plus`,`/data/home/frank/projects/LIBERO-plus` 建同名符号链接;`~/.libero` 配置同步
- imagemagick noise 依赖(104M)→ `/data/shared/offline/noise_deps/imagemagick`
- base ckpt、dataset_stats、Wan-AI/T5/VAE h200-2 原本已有
- 待做:fastwam env `pip install -e /data/home/frank/projects/LIBERO-plus --no-deps` + import 冒烟;代码经 git bundle 送达(h200-2 无外网 git)+ worktree

## 关键命令(h200-2,等卡空后执行)

```bash
cd /data/home/frank/projects/FastWAM-v4-ablation   # mem-stage-v4 worktree

# 冒烟(200 case pilot,单模式)
V4_CKPT=$HOME/ckpts/mem_stage_v4_libero_step_014470.pt \
EVAL=plus_pilot CONFIGS="B_video_only" bash scripts/run_v4_history_ablation.sh

# 全量(B/C/D 串行,每个 ~9-10h × 3 ≈ 28-30h,8 卡)
V4_CKPT=$HOME/ckpts/mem_stage_v4_libero_step_014470.pt \
setsid nohup bash scripts/run_v4_history_ablation.sh > runs/logs/v4_ablation_launcher.log 2>&1 < /dev/null &
```

## 状态

- **进行中(结构版)**:2026-07-15 16:23 起以结构性丢弃重启 B/C/D 全量队列(h200-1,commit `55ef514`),断点续跑同一输出目录(B 已有 ~300 个 mask 版结果,两版等价可混用)。
- **h200-1 奇数卡进程会被外部静默杀掉(已发生两次)**:14:38-14:47 与 16:0x 两轮,奇数卡(1/3/5/7)上的 worker 均无 traceback/无 OOM 死亡,同期其他用户进程也消失,偶数卡不受影响,原因不明(疑似外部按卡清理)。对策:`GPU_LIST="0 2 4 6"`,8 worker 轮转压 4 张偶数卡(每卡 2 worker,~50G/141G 显存),已验证映射正确。
- 中途插曲:16:1x 一次误杀(pkill 模式匹配到自身 shell)导致 B 的 eval.sh 先死、launcher 串到 C 提前启动,已全部清理后按 GPU_LIST 重启,无结果污染(结果文件按 task 命名幂等)。
- 运行 commit 时间线:`1955e9f`(mask 版首启)→ `37320f2`(结构性丢弃 + bench)→ `c7405b1`(PYTHONPATH 修复)→ `55ef514`(GPU_LIST,当前运行)。
- 输出目录:`evaluate_results/v4_history_ablation/plus_full_20260715_143258/{B_video_only,C_action_only,D_no_history}/`
- launcher 日志:`runs/logs/v4_ablation_launcher_structural.log`;每 config 内 `progress.log` / `worker_logs/`。
- 预计:4 卡×2 worker 吞吐接近原 8 卡(rollout 部分受 CPU/sim 限制),单 config 约 10-14h,三个串行预计 7-16 深夜至 7-17 白天完成。
- 冒烟验证:结构版 worker 日志确认 `v4 history ablation mode: video_only (structural drop)`;bench 等价自检 max|Δaction|=7.8e-3 通过。

## 监控命令

```bash
ssh h200-qinghua-1
tail -f /data/home/frank/projects/FastWAM-v4-ablation/runs/logs/v4_ablation_launcher.log
tail -f /data/home/frank/projects/FastWAM-v4-ablation/evaluate_results/v4_history_ablation/plus_full_20260715_143258/B_video_only/progress.log
```

## 下一步

1. 跑完后 `summarize_libero_plus.py` 汇总 B/C/D,与 A(26-06-29 同 ckpt plus_full)按 7 类扰动因子对比;
2. 结果回填本文档;若出现反直觉结果(如 C≈A)再评估重训单路变体;
3. 卡充足时补 A/E 同机基线(`CONFIGS="A_full E_base"`,STD 用 `EVAL=libero_full`)。
