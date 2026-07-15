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

## 屏蔽语义(关键设计)

复刻 **episode 起步 padding 状态**(在线 buffer 首次 replan 的合法输入),而非仅翻 is_pad:

- `video_only`(屏蔽 action):`history_action` 置零 + `history_action_is_pad` 全 True;
- `action_only`(屏蔽 video):`history_video` 过去 4 帧替换为当前帧 + is_pad `[1,1,1,1,0]`,当前帧永远可见;
- `no_history`:两者同时;
- `off`:完全不传 history,退回原版 first-frame KV 路径(留给 base ckpt 的 E 配置)。

正确性依据:

1. `_latent_valid_from_raw_pad` 把 raw `[1,1,1,1,0]` 映射为 latent `[屏蔽, 可见]`,与训练 drop_video 只屏蔽 history latent、保留 current latent 完全一致;position_id 不重排。
2. all-pad action history 是每个 episode 首次 replan 的既有路径,此前 STD/plus 评测已反复经过,无 `ensure_non_empty` 掩码回退风险(future action 的 condition mask 行内始终有有效 video key)。
3. 过去帧替换为当前帧,避免 VAE 时间压缩把被屏蔽帧像素混进 current latent 的信息泄漏(纯 is_pad 方案做不到),且与训练数据 episode 开头 index-clamp 的补帧行为一致。

判读注意:推理时屏蔽衡量"双记忆模型在推理时对该路信息的依赖",给出的单路成绩是偏保守下界,不等价"从头单路训练"。若出现反直觉结果(如 C≈A),再考虑重训单路变体。

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

- **进行中**(2026-07-15):代码已提交;LIBERO-plus 16G 迁移后台进行;等 h200-2 卡空启动。
- 输出目录约定:`evaluate_results/v4_history_ablation/plus_full_<stamp>/{B_video_only,C_action_only,D_no_history}/`
- 每个 config 目录内 `progress.log` / `worker_logs/` 同 eval.sh 惯例;汇总用 `summarize_libero_plus.py`。

## 下一步

1. h200-2 完成 pip install + DRY_RUN/pilot 冒烟;
2. 卡空后启动全量,按 7 类扰动因子拆分对比 A(26-06-29)/B/C/D;
3. 结果回填本文档;若需严格结论再评估重训单路变体;
4. 卡充足时补 A/E 同机 STD 基线(脚本已支持 `CONFIGS="A_full E_base" EVAL=libero_full`)。
