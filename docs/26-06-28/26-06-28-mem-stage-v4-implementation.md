# 26-06-28 mem-stage-v4 实现记录

## 目的

在 `mem-stage-v4` 分支实现 v4 方案：在 FastWAM 默认 action-only/KV 推理基础上，加入 history video、history action、current observation 条件缓存；training 中 history/current clean video prefix 与 future noisy video 合并进唯一 Video DiT 路径；保留 FastWAM video/action diffusion loss 口径；移除 `current_video_valid` 作为显式输入或配置。

## 分支与位置

- 分支：`mem-stage-v4`
- 运行位置：本地 `/Users/maxliu/MyProjects/AIR/202606/FastWAM_xyc`
- 远端 GPU 验证：尚未运行，本地不执行大规模训练或推理。

## 主要改动

1. 数据集窗口
   - `RobotVideoDataset` 底层一次读取更长窗口：video delta `-16..32`，action delta `-20..31`。
   - 对外保留 FastWAM 原字段：
     - `video`: 当前及未来 9 帧，raw index `16,20,...,48`
     - `action`: future action 32 步，index `20..51`
     - `proprio`: 从当前状态开始对齐 action，index `16..47`
   - 新增 v4 条件字段：
     - `history_video`: 5 帧，raw index `0,4,8,12,16`
     - `history_action`: 20 步，index `0..19`
     - `history_video_is_pad` / `history_action_is_pad`

2. source 与时间编码
   - `ActionDiT` 新增 4 类 source embedding，并支持显式 action RoPE `position_ids`。
   - `WanVideoDiT` 新增 5 类 source embedding，并支持显式 video temporal RoPE `temporal_position_ids`。
   - source embedding gate 初始为 0，旧 checkpoint/预训练初始化不会立刻改变行为。
   - history action 使用位置 `0..19`，future action 使用 `20..51`。
   - combined video latent 使用统一 timeline：history/current prefix 约为 `4,20`，future video latent 约为 `24,28`。
   - diffusion timestep 与 temporal position 分开：history/current clean prefix 的 diffusion timestep 为 0，表示 clean conditioning；时间顺序仍由 RoPE position/source embedding 表达，不会同时从 0 重新开始。

3. KV condition cache
   - `MoT.prefill_video_cache` 统一返回 `(kv_cache, video_tokens_after_blocks)`；不保留 cache-only 可选路径，避免接口层暗示存在第二条 Video DiT。
   - v4 video loss 路径：
     - `history_video_latents [B,48,2,14,28]` 与 `future_noisy_latents [B,48,2,14,28]` 拼成 `combined_video_latents [B,48,4,14,28]`；
     - 只走一次 Video DiT / video expert block；
     - video loss 只监督 future suffix，即 `pred[:, :, 2:]` 对齐原 `target_video[:, :, 1:]`；
     - clean prefix 不算 video loss，但通过同一路 Video DiT 自然影响 future video 预测。
   - v4 action loss 路径：
     - 从同一个 combined Video DiT path 的 KV cache 中切出 clean history/current video prefix；
     - clean history action 进入 action KV cache；
     - noisy future action 作为 query，attend 到 condition KV 与自身 action KV；
     - action loss mask 仍按 `action_is_pad`。
   - history video/action 训练时分别 20% dropout；dropout 不重排 position id，只通过 key visibility mask 改变可见性。
   - history video dropout 同时作用在 combined Video DiT self-attention 的 key visibility 上；被 drop 的 history video 不允许通过 current/future video self-attention 间接影响 video loss 或 action condition K/V。

4. 训练与推理
   - `FastWAM.training_loss` 检测到 history 字段时走 v4；否则保留原 FastWAM 路径。
   - 当前分支只保留 base `FastWAM` 路线；Joint/IDM model configs、task configs 和 model wrapper 文件已移除。
   - video loss 保持 diffusion MSE 口径，但输入路径改为 single combined Video DiT path。video 分支的 action conditioning 走 `video_expert.pre_dit` 的 context-concat（group-causal mask），与原 FastWAM 一致：当 `action_conditioned=True` 时，future video latent 帧按组 attend future action，clean prefix（history/current）不 attend action；当前 `fastwam.yaml` 里 `action_conditioned=false`，该路径 dormant，video 仅靠 text/proprio 与 video self-attention 预测。（详见下文「代码复查与修复」MEDIUM 项。）
   - `infer_action` 默认仍是 FastWAM 当前帧 action-only KV 推理；传入 `history_video/history_action` 时切到 v4 condition cache。
   - base `FastWAM.infer_joint` / `FastWAM.infer` video rollout 入口已删除；v4 推理唯一入口是 `infer_action`。
   - trainer eval 会把 history 字段传入 action-only inference，避免训练/评估 action 路径错位；eval 不再生成 rollout mp4 或 video PSNR/SSIM。
   - `runtime.run_inference` 切到 action-only inference，输出 action tensor 或保存 `output_action_path`，不再默认生成 mp4 video。

5. online rollout history
   - 新增 `experiments/fastwam_online_history.py`，统一维护 online `history_video/history_action/*_is_pad`。
   - history video window 与训练集一致：每个 policy step 记录当前观测，replan 时取 `V[t-16], V[t-12], V[t-8], V[t-4], V[t]`，不足窗口左侧 padding，当前帧必须可见。
   - history action 只记录实际送入环境后的 action；RobotWin 在 `task_env.take_action` 后写入，LIBERO 在 `env.step` 后写入。
   - history action 保存模型 normalized/action-token 空间的动作，不保存未执行的完整 predicted chunk；LIBERO action ensemble 打开时，同时对 normalized action 做同 timestamp averaging，再把实际执行的 ensemble action 写入 history。
   - RobotWin policy 的 `should_request_observation()` 固定返回 `True`，保证 replan 内也能维护逐步 history video；推理仍只在 pending queue 为空时触发。
   - LIBERO eval 删除 `visualize_future_video` / `infer_joint` / future-video PSNR 路径，当前在线评测只走 `infer_action`。

6. checkpoint
   - v4 checkpoint 保存 `mem_stage_v4=True`。
   - 旧 FastWAM 权重初始化允许新增参数 missing。
   - v4 checkpoint resume 时 MoT 权重必须严格恢复，避免 source embedding / cache 相关参数静默缺失。

## 代码复查与修复（26-06-28，基于 KV history 设计文档）

对照设计文档逐链路复查 v4 实现，修复以下问题（仅改动 `fastwam.py` / `wan_video_dit.py`）：

- **High — 训练/推理 history-video self-attention mask 不一致（已修）：** 训练 combined 路径用 `clean_prefix_latent_frames=全部 history latent 帧`，在 `first_frame_causal` 下前缀内部双向可见；但推理 `infer_action` 的 history-video prefill 调用 `build_video_to_video_mask` 时漏传该参数（默认 1），导致第 0 个 history 帧看不到后续 history 帧，history-video K/V 在 train/infer 下不一致，污染 action condition memory。修复：推理处传入 `clean_prefix_latent_frames=history_video_latents.shape[2]`，恢复前缀双向，与训练对齐。
- **Medium — combined video 丢失 video←action 条件（已按原 FastWAM 改回）：** 原 v4 combined 路径给 video `pre_dit` 传 `action=None`，相比原 FastWAM 丢掉了 video 分支的 action conditioning（真实原因是 group-causal mask 的 `num_temporal_groups=f-1` 在 combined `f=4` 时与 32 action 不整除会触发 assert）。修复：(1) `pre_dit` 把 `num_temporal_groups` 泛化为 `f - clean_prefix_latent_frames`，clean prefix 帧不 attend action，只有 future video 帧按组 attend future action（默认 `clean_prefix=1` 时与原 FastWAM 完全等价）；(2) combined 路径改回 `action=action`。注意当前 `fastwam.yaml` 中 `action_conditioned=false`，该条件路径 dormant，本次修复在当前配置下行为不变（no-op），价值在于与原 FastWAM 一致并避免将来开启 `action_conditioned` 时崩溃。原 point 4「不会丢失 video<-action 信息」的论证只覆盖了 MoT self-attention mask、漏了 pre_dit 的 context-concat 路径，已在上文订正。
- **Low — v4 触发条件统一为 AND（已修）：** `training_loss` dispatcher 原用 `history_video_latents is not None or history_action is not None`，但 `_training_loss_v4` 内部要求两者同时存在。改为 AND，缺任一字段则回退原 FastWAM，避免只给其一时直接崩溃。
- **Low — 删除 v4 死代码（已修）：** `_training_loss_v4` 中 `latents[:, :, 0:1] = first_frame_latents` 随后即被 `latents[:, :, 1:]` 丢弃，是 no-op，已删除并加注释说明当前观测由 `history_video` 最后一帧承载。

复查确认无问题的关键点：source embedding 注入点（encoder 后、QKV/RoPE 前，zero-init gate）、source/position id 边界检查、RoPE cache 长度（1024，足够 max position 51）、训练保留梯度 / 推理 `no_grad`、clean prefix K/V 通过 `first_frame_causal` 与 noisy future 隔离（切片复用成立）、dropout 仅作用 history（不动 current）且不重排 position、train/infer condition 拼接顺序一致、checkpoint 中 experts 经 `MoT.ModuleDict` 共享因此 source embedding 会被 `mot.state_dict()` 保存/恢复、`fuse_vae_embedding_in_latents` 仅影响 per-token timestep 置零（combined concat 合法）、`ensure_non_empty` 兜底避免全屏蔽行 NaN。

数据-模型链路一致性（LIBERO）：`libero_2cam` 用 `num_frames=33, action_video_freq_ratio=4` → history_video 5 raw→2 latent、video 9 raw→3 latent、history_action=20、future_action=32；combined `f=4 = 2 prefix + 2 future`，与模型常量（`HISTORY_ACTION_LEN=20`、`CURRENT_TIMELINE_INDEX=20`）一致。LIBERO 用 `override /model: fastwam`（base FastWAM，`enable_mem_stage_v4=True`）、`override /data: libero_2cam`（产出 history 字段），v4 路径会被正确触发。

## 本地验证

已执行：

```bash
python -m py_compile src/fastwam/models/wan22/wan_video_dit.py src/fastwam/models/wan22/mot.py src/fastwam/models/wan22/fastwam.py src/fastwam/runtime.py src/fastwam/trainer.py
python -m py_compile experiments/fastwam_online_history.py experiments/robotwin/fastwam_policy/deploy_policy.py experiments/libero/eval_libero_single.py
python -m py_compile src/fastwam/models/wan22/fastwam.py src/fastwam/models/wan22/wan_video_dit.py
git diff --check
rg -n "action_pre_for_video|clean_timestep = torch.zeros\(\(batch_size,\).*history_video|history_video_pre = self\.video_expert\.pre_dit" src/fastwam/models/wan22/fastwam.py -S
rg -n "return_tokens|prefill_video_cache\(" src/fastwam/models/wan22 docs/26-06-28 -S
rg -n "FastWAMJoint|FastWAMIDM|fastwam_joint|fastwam_idm|create_fastwam_joint|create_fastwam_idm" src configs -S
rg -n "infer_joint|FastWAMJoint|FastWAMIDM|fastwam_joint|fastwam_idm|visualize_future_video|predicted_future" src configs experiments -S --glob '!third_party/**'
```

结果：

- `py_compile` 通过。
- `git diff --check` 通过。
- training v4 中不再存在 `action_pre_for_video` 或第二条 `history_video_pre`；`history_video_pre` 只剩 v4 action-only inference path。
- `return_tokens` 无残留；`prefill_video_cache` 调用点均按统一 `(kv_cache, video_tokens_after_blocks)` 返回值处理。
- 尝试运行最小 `WanVideoDiT.build_video_to_video_mask(clean_prefix_latent_frames=2)` runtime sanity 时，本地 Python 环境缺少 `torch`，报错 `ModuleNotFoundError: No module named 'torch'`，未完成该项 runtime 校验。
- 尝试运行 `FastWAMOnlineHistoryBuffer` CPU tensor shape sanity 时，同样因本地环境缺少 `torch`，报错 `ModuleNotFoundError: No module named 'torch'`，未完成 runtime 校验。
- `runtime.py` 语法检查通过；未运行真实 action-only inference。
- `src/` 和 `configs/` 中已无 Joint/IDM 入口引用；`third_party/FastWAM/` 仍保留上游参考文件，不属于当前分支代码路径。
- online history buffer、RobotWin policy、LIBERO eval 语法检查通过。
- 当前 `src/configs/experiments` 中已无 `infer_joint`、`visualize_future_video`、`FastWAMJoint/FastWAMIDM` 等 action-only 外路径残留；`src/fastwam/models/wan22/wan22.py` 的 `Wan22Core.infer` 是底层通用接口，不属于 FastWAM policy route。

## 未验证项目

- 未运行 GPU 训练、真实 dataloader batch、真实 inference rollout。
- 未运行 torch runtime 单测，因为当前本地 Python 环境缺少 `torch`。
- 未运行 LIBERO / RobotWin 真实 online rollout，因为需要仿真环境、真实 checkpoint 和 GPU。
- 未运行 runtime action-only 推理，因为需要真实模型权重和 GPU。
- 未运行远端验证，因为当前 `AGENTS.md` 未填写真实远端项目主目录。

## 当前结论

v4 主路径已经完整接入数据、模型、训练 loss、action-only 推理、离线 eval 参数传递、online history buffer 和 checkpoint resume 约束。下一步应在远端 GPU 上做一个小 batch dataloader + forward/backward smoke test，并用 LIBERO / RobotWin 各跑 1 个短 episode，确认 online history shape、padding mask 和 action queue 行为正常。
