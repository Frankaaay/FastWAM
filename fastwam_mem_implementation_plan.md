# FastWAM Memory-VAE 落地与训练计划

本文档分两部分：
- **Part A — 代码接线**：把已实现的 `encode_memory` / `SpatioTemporalAttentionBlock` 等真正接进训练流程（目前只有积木，没有调用处）。
- **Part B — 操作手册**：下载数据集 → 配 config → 启动训练的具体命令。

---

## 训练是怎么串起来的（背景）

```
bash scripts/train_zero2.sh <nproc> task=task/robotwin_joint_3cam_384_1e-4
  └─ accelerate launch scripts/train.py task=...
       └─ runtime.run_training(cfg)
            ├─ model = instantiate(cfg.model)         # create_fastwam_joint(...)
            │     └─ FastWAMJoint.from_wan22_pretrained(...)
            │           └─ load_wan22_ti2v_5b_components(...)   # ← VAE 在这里加载
            ├─ train_ds, val_ds = build_datasets(cfg.data)      # RobotVideoDataset
            └─ Wan22Trainer(...).train()
                 ├─ optimizer = AdamW(model.dit.parameters())   # ← 只训 DiT！
                 └─ loss = model.training_loss(sample)
                       └─ build_inputs(sample)                  # ← VAE encode 在这里
```

要让 memory 生效，需要在 **5 个地方**接线（下面 A1–A5）。每处都用一个 config 开关 `enable_memory_vae`/`num_history_frames` 控制，默认关 → 不影响现有训练。

---

## Part A — 代码接线

### A1. 加 config 开关

`configs/model/fastwam_joint.yaml`（及 fastwam.yaml / fastwam_idm.yaml）加：
```yaml
enable_memory_vae: true        # 打开 temporal attention VAE
```

`configs/data/robotwin.yaml` 的 `train:` 和 `val:` 各加：
```yaml
num_history_frames: 16         # 历史帧数，必须是 4 的倍数
```
并把任务 config（如 `robotwin_joint_3cam_384_1e-4.yaml`）加：
```yaml
model:
  enable_memory_vae: true
```

### A2. loader.py：打开 temporal attention + 暖启动

`src/fastwam/models/wan22/helpers/loader.py`
- `load_wan22_ti2v_5b_components(...)` 增加参数 `enable_memory_vae: bool = False`。
- VAE 加载处（约 line 210）改成：
```python
from fastwam.models.wan22.wan_video_vae import init_temporal_from_spatial

vae = _load_registered_model(
    vae_config.path, "wan_video_vae",
    torch_dtype=torch_dtype, device=device,
    model_kwargs_override={"use_temporal_attention": enable_memory_vae},
)
if enable_memory_vae:
    init_temporal_from_spatial(vae)   # 必须在 strict=False 加载之后
```

### A3. 把开关从 config 串到 loader

`runtime.create_fastwam_joint(...)`（及 create_fastwam / create_fastwam_idm）增加 `enable_memory_vae: bool = False` 形参，透传给 `FastWAMJoint.from_wan22_pretrained(..., enable_memory_vae=...)`；
`fastwam_joint.py` 的 `from_wan22_pretrained` 再透传给 `load_wan22_ti2v_5b_components(..., enable_memory_vae=...)`，并把 `num_history_frames` 存到 model 上（`self.num_history_frames`，供 build_inputs 用）。

> 注：fastwam_joint.py 的 `from_wan22_pretrained` 是 `**kwargs` 转发（line 16），多数情况只要在 base `FastWAM.from_wan22_pretrained` 接住即可。

### A4. dataset：产出 history_video

`src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `__init__` 增加 `num_history_frames: int = 0`，保存为 `self.num_history_frames`。
- `_get`（返回 dict 处，约 line 223）增加一个 `history_video` key。

**v1（先验证管线，0 新数据，最省事，推荐第一步就用这个）**：用当前 clip 自己的首帧重复当历史——纯粹验证端到端能跑、gate 能训、step 0 与 baseline 对齐：
```python
# video: [C, T_cur, H, W]
if self.num_history_frames > 0:
    first = video[:, :1]                                  # [C,1,H,W]
    history_video = first.repeat(1, self.num_history_frames, 1, 1)
    data["history_video"] = history_video                 # [C, T_hist, H, W]
```

**v2（真历史，验证通过后再做）**：让底层 lerobot 多取 `num_history_frames` 帧（按 ~1s stride）放进 `history_video`；episode 开头不足时用首帧 padding 并记 mask。这一步要动 `BaseLerobotDataset` 的取帧窗口，单独提一个改动做。

### A5. fastwam.build_inputs：用 encode_memory（且不要 no_grad）

`src/fastwam/models/wan22/fastwam.py`，`build_inputs`（line 337-338）改为：
```python
input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

history_video = sample.get("history_video", None)
if getattr(self, "enable_memory_vae", False) and history_video is not None:
    history_video = history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
    mem_video = torch.cat([history_video, input_video], dim=2)   # [B,3,T_hist+T_cur,H,W]
    # 不能 no_grad：temporal 参数要拿梯度
    input_latents = self.vae.encode_memory(
        mem_video, num_current_frames=input_video.shape[2], device=self.device
    )
else:
    input_latents = self._encode_video_latents(input_video, tiled=tiled)  # 原路径
```
下游 `first_frame_latents = input_latents[:, :, 0:1]`、DiT 调用、loss 全不变（`input_latents` 的 shape 与原来一致）。

### A6. trainer：解冻 temporal 参数 + 加进 optimizer

`src/fastwam/trainer.py`
- freeze 段（约 line 289-295，`model.requires_grad_(False); model.dit.requires_grad_(True)`）后面加：
```python
from fastwam.models.wan22.wan_video_vae import MEMORY_PARAM_KEYS
if getattr(model, "enable_memory_vae", False):
    for name, p in model.vae.named_parameters():
        if any(k in name for k in MEMORY_PARAM_KEYS):
            p.requires_grad_(True)
```
- optimizer 段（line 85，`trainable_params = list(self.model.dit.parameters())`）后面加：
```python
if getattr(self.model, "enable_memory_vae", False):
    trainable_params += [
        p for n, p in self.model.vae.named_parameters()
        if p.requires_grad and any(k in n for k in MEMORY_PARAM_KEYS)
    ]
```
> 这一步是**第一阶段**：DiT + VAE temporal 参数可训，其余冻结。
> **第二阶段**（如表达力不够）：把上面的过滤条件放宽，解冻 `vae.model.encoder` 更多层并加进 optimizer 即可。

---

## Part B — 操作手册（服务器上按顺序执行）

### B0. 环境
```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .          # 装本项目
pip install -U huggingface_hub
```

### B1. 下载数据集（RobotWin 为例；数据不在仓库里，要自己下）
```bash
mkdir -p data/robotwin2.0 && cd data/robotwin2.0
huggingface-cli download yuanty/robotwin2.0-fastwam --repo-type dataset --local-dir .
# 合并分卷并解压
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
# 解压后应有：data/robotwin2.0/robotwin2.0/...  和  data/robotwin2.0/dataset_stats.json
```
LIBERO 同理：`huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset ...` 到 `data/libero_mujoco3.3.2/`。

### B2. 下载已发布 ckpt 作为 warm-start
```bash
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints
```

### B3. 预计算文本 embedding（dataset 需要缓存）
```bash
python scripts/precompute_text_embeds.py   # 产物到 ./data/text_embeds_cache/robotwin
```
（具体参数看脚本顶部；它会把 instruction 编码缓存，训练时 dataset 直接读。）

### B4. 改 config 里的路径（指向你服务器上的真实位置）
`configs/data/robotwin.yaml`（train 和 val 都要改）：
```yaml
dataset_dirs:
  - <你的绝对路径>/data/robotwin2.0/robotwin2.0
pretrained_norm_stats: <你的绝对路径>/data/robotwin2.0/dataset_stats.json
text_embedding_cache_dir: <你的绝对路径>/data/text_embeds_cache/robotwin
num_history_frames: 16        # A1 加的
```

### B5. 先跑单元测试确认 VAE 改动没坏
```bash
python scripts/test_memory_vae.py
# 期望输出最后一行: All memory-VAE sanity checks passed.
```

### B6. 第一阶段训练命令
```bash
# <nproc> = 用几张卡，例如 8
bash scripts/train_zero2.sh 8 \
  task=task/robotwin_joint_3cam_384_1e-4 \
  resume=./checkpoints/robotwin_uncond_3cam_384.pt \
  model.enable_memory_vae=true \
  learning_rate=5e-5 \
  output_dir=./runs/robotwin_mem_stage1/$(date +%F_%H-%M-%S)
```
说明：
- `resume=...released.pt` → 从已发布 DiT ckpt 暖启动（新增 temporal 参数 ckpt 里没有，靠 `strict=False` 跳过，再由 `init_temporal_from_spatial` 暖启动）。
- `model.enable_memory_vae=true` → 打开整条 memory 链路。
- 任何 `configs/train.yaml` / task 里的字段都能用 `key=value` 在命令行覆盖（Hydra）。
- 单机调试可先用 `bash scripts/train_zero2.sh 1 ... batch_size=2`。

### B7. 看训练 & 产物
- 日志：`./runs/.../`；ckpt 每 `save_every` 步存一次（task 里是 2500）。
- 关注：`temporal_gate` 是否从 0 逐渐变大（说明 history 在起作用）；loss 不应在 step 0 突然变差（zero-gate 保证起步等价 baseline）。
- 开 wandb：命令行加 `wandb.enabled=true wandb.project=fast-wam`。

### B8. 第二阶段（可选，第一阶段收益不够再做）
- 按 A6 的"第二阶段"放宽解冻范围。
- 命令同 B6，把 `resume` 换成第一阶段产出的 ckpt。

---

## 执行顺序建议
1. 先做 **A1–A6 接线** + `python scripts/test_memory_vae.py` 过。
2. 用 **v1 的 history（首帧重复）** 跑通 B6，确认端到端不报错、loss 正常、gate 能更新。
3. 再做 **A4 的 v2 真历史**，重训，才是真正有记忆的版本。
4. 不够再上 **第二阶段解冻**。
```
```
