# 26-06-28 mem-stage-v4 实现记录

## 目的

在 `mem-stage-v4` 分支实现 v4 方案：在 FastWAM 默认 action-only/KV 推理基础上，加入 history video、history action、current observation 条件缓存；保留 FastWAM 原有 video/action loss 口径；移除 `current_video_valid` 作为显式输入或配置。

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
   - `WanVideoDiT` 新增 4 类 source embedding，并支持显式 video temporal RoPE `temporal_position_ids`。
   - source embedding gate 初始为 0，旧 checkpoint/预训练初始化不会立刻改变行为。
   - history action 使用位置 `0..19`，future action 使用 `20..51`。
   - history/current video latent 使用统一 timeline，最后一个 latent 固定为 current position `20`。

3. KV condition cache
   - `MoT` 新增 action cache prefill 和 generic condition cache action forward。
   - v4 action loss 路径：
     - clean history video/current video 进入 video KV cache；
     - clean history action 进入 action KV cache；
     - noisy future action 作为 query，attend 到 condition KV 与自身 action KV；
     - action loss mask 仍按 `action_is_pad`。
   - history video/action 训练时分别 20% dropout；dropout 不重排 position id，只通过 key visibility mask 改变可见性。
   - 修复 history video dropout 的间接泄漏：被 drop 的 history video 不允许通过 current video prefill self-attention 影响 current K/V。

4. 训练与推理
   - `FastWAM.training_loss` 检测到 history 字段时走 v4；否则保留原 FastWAM 路径。
   - v4 自动触发仅限 base `FastWAM`，避免 `FastWAMJoint` / `FastWAMIDM` 被 dataset 新字段误触发。
   - video loss 保持原 FastWAM mixed video/action denoise 口径。
   - `infer_action` 默认仍是 FastWAM 当前帧 action-only KV 推理；传入 `history_video/history_action` 时切到 v4 condition cache。
   - `infer/infer_joint` 接受 history 字段；有 history 时返回 v4 action-only 的 action，video 仍按原 joint 路径生成。
   - trainer eval 会把 history 字段传入 inference，避免训练/评估 action 路径错位。

5. checkpoint
   - v4 checkpoint 保存 `mem_stage_v4=True`。
   - 旧 FastWAM 权重初始化允许新增参数 missing。
   - v4 checkpoint resume 时 MoT 权重必须严格恢复，避免 source embedding / cache 相关参数静默缺失。

## 本地验证

已执行：

```bash
python -m py_compile src/fastwam/models/wan22/action_dit.py src/fastwam/models/wan22/wan_video_dit.py src/fastwam/models/wan22/mot.py src/fastwam/models/wan22/fastwam.py src/fastwam/datasets/lerobot/base_lerobot_dataset.py src/fastwam/datasets/lerobot/processors/base_processor.py src/fastwam/datasets/lerobot/processors/fastwam_processor.py src/fastwam/datasets/lerobot/robot_video_dataset.py src/fastwam/trainer.py
git diff --check
rg "current_video_valid" -n .
```

结果：

- `py_compile` 通过。
- `git diff --check` 通过。
- `current_video_valid` 无残留匹配。

## 未验证项目

- 未运行 GPU 训练、真实 dataloader batch、真实 inference rollout。
- 未运行远端验证，因为当前 `AGENTS.md` 未填写真实远端项目主目录。

## 当前结论

v4 主路径已经完整接入数据、模型、训练 loss、action-only 推理、eval 参数传递和 checkpoint resume 约束。下一步应在远端 GPU 上做一个小 batch dataloader + forward/backward smoke test，再启动短训确认 loss 正常下降、history dropout 不触发 NaN。
