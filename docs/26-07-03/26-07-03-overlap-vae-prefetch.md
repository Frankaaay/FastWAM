# overlap VAE encode prefetch 实现记录

## 目的

无 latent cache 训练时，frozen VAE 在线 encode 仍在每个 step 内占用明显时间。本次新增默认关闭的 `overlap_vae_encode` 开关，用于在训练 batch N 时，提前在独立 CUDA stream 上编码 batch N+1 的 raw video，并复用现有 cached-latents 注入字段让 `FastWAM.build_inputs()` 跳过在线 encode。

## 当前分支与范围

- 分支：`profiling-mem-stage-v4`
- 基线 commit：`91170d0`
- 运行位置：本地 `/Users/maxliu/MyProjects/AIR/202606/FastWAM_worktrees/profiling-mem-stage-v4`
- 远端目标位置：`/data/home/maxliu/projects/FastWAM`
- 本次只改训练侧 prefetch 与配置开关，不修改 `mot.py`，也不修改 `fastwam.py` 的 cached-latents 判断。

## 代码改动

1. `configs/train.yaml`
   - 根级新增 `overlap_vae_encode: false`。
   - 默认关闭，默认训练路径不进入 lookahead/prefetch 分支。

2. `src/fastwam/trainer.py`
   - 新增 `overlap-vae` rank0 日志，打印 requested/enabled 状态。
   - 启用守卫：
     - 若 train dataset 已配置 `vae_latent_cache_dir`，warning 后禁用，磁盘 cache 优先。
     - 若 `gradient_accumulation_steps != 1`，warning 后禁用。
     - 非 CUDA device 或模型无 `_encode_video_latents` helper 时禁用。
   - 启用后训练循环改为一步 lookahead：
     - 当前 batch N 训练前先取到 batch N+1，但不提前增加 `batch_in_epoch`。
     - 在当前 forward 前启动 batch N+1 的 side-stream VAE encode。
     - batch N+1 真正成为当前 batch 时，主 stream 等待对应 CUDA event，并注入 latents。
   - epoch 末尾取不到 N+1 时只设置下一轮重置 epoch 的标记，当前最后一个 batch 正常训练；新 epoch 第一个 batch 没有已有 prefetch 时走在线 encode。

## 注入字段一致性

当前 dataset latent cache 的字段是：

- `input_latents`
- `history_video_latents`

单样本 cache payload 中二者均为 4D `[C,T,H,W]` CPU contiguous tensor；DataLoader collate 后进入 model 为 5D `[B,C,T,H,W]`。`FastWAM.build_inputs()` 检测到这两个字段后走 `_prepare_cached_video_latents()`，校验 5D、floating point、batch size，再搬到 `self.device` 与 `self.torch_dtype`。

overlap prefetch 注入同名字段：

- `sample["input_latents"] = <GPU tensor [B,C,T,H,W]>`
- 若存在 `history_video`，则 `sample["history_video_latents"] = <GPU tensor [B,C,T,H,W]>`

`build_inputs()` 对 GPU tensor 仍复用 `_prepare_cached_video_latents()`，因此 dtype/device/contiguous 处理与磁盘 cache 路径一致。

## 数值语义核对

在线路径为：

```python
input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
input_latents = self._encode_video_latents(input_video, tiled=tiled)
```

overlap 路径复用同一个 `_encode_video_latents()` helper，且在 `torch.no_grad()` 与 `accelerator.autocast()` 下执行，输入搬运同样使用 `device=model.device`、`dtype=model.torch_dtype`、`non_blocking=True`。当前训练调用 `training_loss(sample)` 时不传 `tiled`，因此 prefetch 固定使用 `tiled=False`，与默认在线训练一致。

## 同步与生命周期

- side stream 启动前执行 `stream.wait_stream(current_stream)`，避免与主 stream 上已有相关工作乱序。
- side stream 结束时记录 CUDA event。
- 主 stream 使用 prefetched latents 前执行 `current_stream.wait_event(event)`。
- side stream 产出的 `input_latents` / `history_video_latents` 在注入前对主 stream 调用 `record_stream(current_stream)`，防止主 stream 仍在使用时缓存分配器提前复用内存。
- max_steps 提前结束且存在未消费 prefetch 时，对 event 做 CPU 侧 `synchronize()` 后再保存最终 checkpoint。

## 验证记录

- 本地已执行：

```bash
python -m py_compile src/fastwam/trainer.py
```

结果：通过。

- 本地已执行：

```bash
git diff --check
```

结果：通过。

## 当前结论与下一步

当前实现满足默认关闭和 cache 优先原则，适合后续在无 latent cache 的短训 run 中用 `overlap_vae_encode=true` 做吞吐对比。建议远端验证时先跑小步数 smoke，重点看日志中 `overlap-vae enabled=True`，以及 profiler 中 `model/build_inputs/current_video_to_latents` 是否从主 forward 路径消失或明显缩短。
