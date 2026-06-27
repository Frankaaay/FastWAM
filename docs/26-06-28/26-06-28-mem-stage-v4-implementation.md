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
   - video loss 保持 diffusion MSE 口径，但输入路径改为 single combined Video DiT path；FastWAM 原 mixed mask 中 video row 本身只看 video，不看 action，因此不会丢失 video<-action 信息。
   - `infer_action` 默认仍是 FastWAM 当前帧 action-only KV 推理；传入 `history_video/history_action` 时切到 v4 condition cache。
   - base `FastWAM.infer_joint` / `FastWAM.infer` video rollout 入口已删除；v4 推理唯一入口是 `infer_action`。
   - trainer eval 会把 history 字段传入 action-only inference，避免训练/评估 action 路径错位；eval 不再生成 rollout mp4 或 video PSNR/SSIM。
   - `runtime.run_inference` 切到 action-only inference，输出 action tensor 或保存 `output_action_path`，不再默认生成 mp4 video。

5. checkpoint
   - v4 checkpoint 保存 `mem_stage_v4=True`。
   - 旧 FastWAM 权重初始化允许新增参数 missing。
   - v4 checkpoint resume 时 MoT 权重必须严格恢复，避免 source embedding / cache 相关参数静默缺失。

## 本地验证

已执行：

```bash
python -m py_compile src/fastwam/models/wan22/wan_video_dit.py src/fastwam/models/wan22/mot.py src/fastwam/models/wan22/fastwam.py src/fastwam/runtime.py src/fastwam/trainer.py
git diff --check
rg -n "action_pre_for_video|clean_timestep = torch.zeros\(\(batch_size,\).*history_video|history_video_pre = self\.video_expert\.pre_dit" src/fastwam/models/wan22/fastwam.py -S
rg -n "return_tokens|prefill_video_cache\(" src/fastwam/models/wan22 docs/26-06-28 -S
rg -n "FastWAMJoint|FastWAMIDM|fastwam_joint|fastwam_idm|create_fastwam_joint|create_fastwam_idm" src configs -S
```

结果：

- `py_compile` 通过。
- `git diff --check` 通过。
- training v4 中不再存在 `action_pre_for_video` 或第二条 `history_video_pre`；`history_video_pre` 只剩 v4 action-only inference path。
- `return_tokens` 无残留；`prefill_video_cache` 调用点均按统一 `(kv_cache, video_tokens_after_blocks)` 返回值处理。
- 尝试运行最小 `WanVideoDiT.build_video_to_video_mask(clean_prefix_latent_frames=2)` runtime sanity 时，本地 Python 环境缺少 `torch`，报错 `ModuleNotFoundError: No module named 'torch'`，未完成该项 runtime 校验。
- `runtime.py` 语法检查通过；未运行真实 action-only inference。
- `src/` 和 `configs/` 中已无 Joint/IDM 入口引用；`third_party/FastWAM/` 仍保留上游参考文件，不属于当前分支代码路径。

## 未验证项目

- 未运行 GPU 训练、真实 dataloader batch、真实 inference rollout。
- 未运行 torch runtime 单测，因为当前本地 Python 环境缺少 `torch`。
- 未运行 runtime action-only 推理，因为需要真实模型权重和 GPU。
- 未运行远端验证，因为当前 `AGENTS.md` 未填写真实远端项目主目录。

## 当前结论

v4 主路径已经完整接入数据、模型、训练 loss、action-only 推理、eval 参数传递和 checkpoint resume 约束。下一步应在远端 GPU 上做一个小 batch dataloader + forward/backward smoke test，再启动短训确认 loss 正常下降、history dropout 不触发 NaN。
