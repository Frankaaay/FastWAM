# ReMem / MemoryBench Setup Notes

日期：2026-07-09

## 结论

先跑 ReMem extended MemoryBench 的短程部分，也就是 MemoryBench 三个原始任务：

- `reopen_drawer`
- `put_block_back`
- `rearrange_block`

官方 MemoryBench 数据集已经有固定 split：每个任务 `100` 条 train demos 和 `25` 条 held-out test demos。三短任务合计：

| split | 每任务 | 三任务合计 |
|-|-:|-:|
| train | 100 | 300 |
| test | 25 | 75 |

ReMem extended MemoryBench 额外加的 `Long Horizon Task` 单独作为压力测试，不并入第一版主分数。若四个任务都按同规模计，短程部分是 `3/4 = 75%`。

## 官方资源

- SAM2Act repo: `https://github.com/sam2act/sam2act`
- MemoryBench dataset: `https://huggingface.co/datasets/hqfang/memorybench`

MemoryBench 原版任务在 SAM2Act 论文/README 中对应：

- `put_block_back`
- `rearrange_block`
- `reopen_drawer`

## 路径约定

不要把大数据下载到仓库根目录的 `data/`，也不要下载到 B300 的 `/DATA/disk0` 或 `/DATA/disk1`。

B300 上优先使用：

```text
/DATA/kpfs/performance2/Lyle/Data/FastWAM_data/data/memorybench
```

建议结构：

```text
memorybench/
  raw_hf/
    data/
      train/
      test/
  lerobot/
    memorybench_short_train/
    memorybench_short_test/
    memorybench_long_train/
    memorybench_long_test/
```

`raw_hf` 保留官方原始数据；`lerobot` 是转换后给 FastWAM 读取的格式。

## 下载命令

优先在 jump host / KPFS 可见位置执行，不要在 air-gapped compute node 直接拉。

如果有 `huggingface-cli`：

```powershell
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
huggingface-cli download hqfang/memorybench --repo-type dataset --local-dir E:\memorybench\raw_hf --local-dir-use-symlinks False
```

如果没有 `huggingface-cli`，可以用 git-lfs：

```powershell
git lfs install
git clone https://huggingface.co/datasets/hqfang/memorybench E:\memorybench\raw_hf
```

本机预检结果：GitHub 可达；Hugging Face git 探测超时；`git` 和 `git-lfs` 可用，`gh` / `huggingface-cli` 未安装。

## FastWAM 接入状态

当前 FastWAM 训练数据入口是 LeRobot 格式：

- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `configs/data/libero_2cam.yaml`

MemoryBench 原始数据不能直接训练，需要先转换成 LeRobot 格式，至少提供：

- 两路或一路 RGB 图像，映射到 `image` / `wrist_image` 或改成单相机 config。
- action，FastWAM 当前默认期望 7D action：EEF delta pose 6D + gripper 1D。
- state/proprio，当前 LIBERO config 是 8D：EEF pose 6D + gripper 2D。
- task instruction。
- episode metadata / stats。

转换时必须保留 v4 需要的时间关系：

- history video window: `V[t-16], V[t-12], V[t-8], V[t-4], V[t]`
- history action: 最近 20 个实际执行 action
- future action horizon: 32
- `action_video_freq_ratio = 4`

## 第一版实验安排

先做三短任务多任务 finetune：

| model | 初始化 | 数据 | history |
|-|-|-|-|
| FastWAM original | release ckpt | short train 300 demos | current-only |
| v4 full | 同一个 release ckpt | short train 300 demos | history video + history action |
| v4 no history video | 同一个 release ckpt | short train 300 demos | action-only history ablation |
| v4 no history action | 同一个 release ckpt | short train 300 demos | video-only history ablation |

训练预算：

- smoke: 1k steps，确认 dataloader / loss / rollout。
- first official: 10k steps。
- extended: 20k steps，如果 10k 仍明显上涨。
- eval checkpoints: 2k, 5k, 10k, 20k。

主结果：

| model | reopen_drawer | put_block_back | rearrange_block | short avg | memory fail | execution fail |
|-|-:|-:|-:|-:|-:|-:|
| FastWAM original | | | | | | |
| v4 full | | | | | | |
| v4 no history video | | | | | | |
| v4 no history action | | | | | | |

## 下一步 TODO

1. 下载 `raw_hf` 到 KPFS。
2. 检查原始 episode 文件结构，确认图像、动作、状态字段。
3. 写 `scripts/convert_memorybench_to_lerobot.py`。
4. 加 `configs/data/memorybench_short_*.yaml`。
5. 跑 dataloader smoke：取 2 个 batch，检查 `history_video/history_action/video/action` shape。
6. 跑 1k-step training smoke。

