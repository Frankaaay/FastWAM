# 26-07-13 v4 博客流程图（Spatial-Temporal Memory / KV-cache History Prepend）

## 目的

为 blog 制作 mem-stage-v4 的方法流程图，风格对齐此前两张图（"Sequenced Spatial-Temporal Encoding" PPT 风格）。
本次为纯文档/资产产出，无代码改动。

## 分支 / 位置

- 分支：`feat/memorybench-v4-eval-adapter`（`mem-stage-v4` 为其祖先，图内容以 v4 实现为准）
- 产物：
  - `docs/26-07-13/v4-spatial-temporal-memory-flow.svg`（源文件，可继续编辑）
  - `docs/26-07-13/v4-spatial-temporal-memory-flow.png`（2200×1320 渲染稿，可直接插入 blog）

## 生成方式（本地，Windows）

SVG 手工绘制；PNG 用 Edge headless 渲染：

```bash
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless --disable-gpu \
  --hide-scrollbars --screenshot="docs/26-07-13/v4-spatial-temporal-memory-flow.png" \
  --window-size=2200,1320 "file:///C:/Project/FastWAM/docs/26-07-13/v4-spatial-temporal-memory-flow.svg"
```

## 图中信息与代码对应（核对过的事实）

- 输入与常量：`RobotVideoDataset`（`src/fastwam/datasets/lerobot/robot_video_dataset.py`）
  `HISTORY_ACTION_LEN=20`、`HISTORY_VIDEO_PAST_STEPS=16`；配置 `configs/data/memorybench_short.yaml`
  `num_frames=33`、`action_video_freq_ratio=4` → 历史视频 16 步窗口采 5 帧（含当前）→ VAE 后 2 个 clean latent 帧，
  未来 32 步动作 + 未来视频 latent。
- 时间轴 / source-type：`FastWAM`（`src/fastwam/models/wan22/fastwam.py`）5 类 `SOURCE_*`、
  `CURRENT_TIMELINE_INDEX=20`（history action 0–19，future action 20–51，video 位置 4–20 / 24+），
  `HISTORY_CONDITION_DROPOUT=0.2`（视频/动作独立 dropout）。
- KV-cache 机制：`MoT.prefill_video_cache` / `prefill_action_cache`（逐层缓存 K/V，history 以 clean、t=0 前向一次）、
  `forward_action_with_condition_cache`（future action 的 Q attend [history K/V ‖ 自身 K/V]）；
  推理路径中 condition cache 在整个去噪循环外 prefill 一次、N 步复用（`fastwam.py:1603-1723`）。
- 注意力可见性：`build_video_to_video_mask`（`first_frame_causal` + clean prefix 不看 future）；
  future video 读 history video，future action 读 history video + history action。

## 结论 / 下一步

- 流程图完成并渲染检查通过（「逐层 K/V」标签初版被面板遮挡，已修复为最后绘制 + 底色 chip）。
- 如 blog 需要英文版或深色版，可在 SVG 上直接改配色/文案再渲染。
