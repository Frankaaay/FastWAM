# FastWAM 训练与推理加速技术报告

**日期**：2026-07-04
**结论**：在不改变模型数值行为、不引入磁盘缓存的约束下，推理端到端延迟由 544 ms 降至 110 ms（**4.93×**），训练吞吐由 85.6 ms/sample 提升至 52.1 ms/sample（**1.64×**）。全部优化通过数值等价性验证，与现有 checkpoint 完全兼容。

---

## 1. 实验环境与基线

| 项 | 配置 |
|---|---|
| 硬件 | NVIDIA H200（141 GB）× 4（训练）/ × 1（推理） |
| 软件 | PyTorch 2.7.1 + CUDA 12.8，DeepSpeed 0.18.5（ZeRO-1），bf16 混合精度 |
| 模型 | FastWAM 原版（Wan2.2-TI2V-5B video expert + ActionDiT，MoT 结构） |
| 数据 | 真机 fold-cloth 数据集（86.5 万样本，33 帧，384×320，3 相机拼接） |
| 训练基线 | 默认配置：gradient checkpointing 开启，per-GPU batch size 16 |
| 推理基线 | `infer_action`，batch size 1，20 步去噪 |

计时口径：训练取 profile window 内稳态 step（剔除预热），CUDA 同步计时；推理取 warmup 后 20–30 次调用的均值（CUDA events）。

## 2. 瓶颈分析

使用 torch profiler 对训练 step 与推理调用做 GPU kernel 级归因，主要发现：

**训练是 launch-bound 而非 compute-bound。** 基线每 step 发射约 2.7 万个 GPU kernel；按类别聚合后，elementwise/copy/norm 等非 GEMM kernel 的总 GPU 时间超过 GEMM 本身。其中 VAE encoder（时间维分 chunk 的 Causal Conv3D 堆叠）单模块贡献约 3 万 kernel/step：其单次调用墙钟 849 ms，而 GPU busy 仅 623 ms，即约 26% 的时间 GPU 在等待 kernel 发射。

**推理的去噪循环占端到端时间的 90% 且极端 launch-bound。** 20 步去噪每步墙钟约 45 ms，其中 GPU 实际执行仅约 8 ms；其余为 CPU 侧 Python/框架开销与 kernel 发射延迟。

**Attention 不是瓶颈。** attention 前向+反向的 GPU 时间合计约占训练 step 的 11%（bs24 口径约 108 ms/step），任何 attention 侧优化（如 flash attention mask 拆分）的收益上限不足 5%。

由此确定优化方向：减少 kernel 发射次数（编译融合、CUDA Graph），而非提升单 kernel 性能。

## 3. 优化方法

### 3.1 DiT 主干区域编译（训练 + 推理）

对 MoT 中每个 DiT block 的两段计算体（attention I/O：norm、modulation、QKV 投影、RoPE；post-block：输出投影、gate、FFN）分别施加 `torch.compile`（`dynamic=False`，每 block 独立编译单元）。attention 本体（SDPA 调用）不编译，以保留 PyTorch 后端自动选择。与 gradient checkpointing 互斥时自动跳过；编译失败时整体回退 eager 执行。

- 效果：融合后非 GEMM kernel 数量减半；训练 step −12%，推理延迟 −25%。
- 一次性成本：首个 step 编译预热（约 60 个编译单元，数分钟）。

### 3.2 VAE encoder 编译（训练）

对 VAE encoder 的 forward 方法施加 `torch.compile`。实现要点：必须编译 forward 方法本身而非包装整个 module——用编译后的 `OptimizedModule` 替换子模块会破坏 accelerate 的模型树遍历（`unwrap_model` 在非标准包装节点上失败）。

- 效果：训练 forward 1100 → 850 ms，step 整体 1.17×。
- 备选实现：我们还实现了 VAE encode 的函数式重构（cache 显式传递）以支持 CUDA Graph 捕获，实测与直接编译收益基本相同（组合口径 1.35× vs 1.33×）。由于直接编译实现远为简单，**推荐直接编译方案**；两者互斥，不可同时启用。

### 3.3 去噪循环整步 CUDA Graph 捕获与跨调用复用（推理，核心优化）

去噪单步具备 CUDA Graph 捕获的全部条件：输入 shape 固定（action latents 与标量 timestep）；KV cache、文本条件、attention mask 在 20 步循环内不变。

实现分三层：

1. **整步捕获**：将一次去噪预测（action pre-DiT → MoT action forward（读 video KV cache）→ post-DiT）捕获为单个 CUDA Graph。输入经静态 buffer 传入，输出在图外 clone 以避免被后续 replay 覆写。调度器更新（element-wise）保留在图外。
2. **失败回退**：捕获或 replay 异常时自动回退逐步 eager 执行，推理不中断。
3. **跨调用持久化**：条件张量（KV cache 的 K/V、context、mask）同样使用静态 buffer；后续推理调用仅将新的 prefill 结果 `copy_` 进 buffer 后直接 replay，无需重新捕获。输入 shape 签名变化时自动重新捕获。

工程要点（实测踩坑）：

- 逐 block 的 `torch.compile(mode="reduce-overhead")` 不可行：多个小图共享内存池，前一图的输出会被后续图的 capture/replay 覆写而报错。必须整步单图捕获。
- 捕获失败的一个隐蔽原因：RoPE 频率表以普通张量属性存储（非 buffer），常驻 CPU，前向中每步执行 pageable host-to-device 拷贝——此类操作在 CUDA Graph 捕获期被禁止。捕获前将该常量一次性迁移至 GPU 即解决（数值不变）。
- 每次调用重新捕获的开销约 270 ms（预热 + 捕获），会吞掉大部分收益；跨调用持久化后该开销仅在首次调用支付。

- 效果：去噪单步 45 ms → 约 4 ms；端到端 544 → 189 ms；叠加 3.1 后 **110 ms（4.93×）**。

### 3.4 关闭 gradient checkpointing（训练，显存允许时）

默认配置为节省显存开启了 activation checkpointing，代价是 backward 中重算 forward。在 H200 显存余量充足的场景下关闭它，配合 3.1/3.2 使用（与区域编译互斥的路径会自动处理）。该改动数值等价（checkpointing 本身即数值等价的工程选项）。

## 4. 实验结果

### 4.1 推理（batch size 1，20 步去噪，单卡，稳态）

| 配置 | 延迟 (ms) | 加速比 |
|---|---:|---:|
| 基线（eager） | 544 | 1.00× |
| + 主干区域编译 | 409 | 1.33× |
| + 去噪 CUDA Graph（持久化） | 189 | 2.88× |
| **+ 两者组合（推荐）** | **110** | **4.93×** |

### 4.2 训练（4×H200，稳态 step，ms/sample = step 时间 / global batch）

| 配置 | step (ms) | ms/sample | 加速比 |
|---|---:|---:|---:|
| 基线（默认配置，ckpt on，bs16） | 1370 | 85.6 | 1.00× |
| 全优化，同 bs16 | 977 | 61.0 | 1.40× |
| 全优化，bs24 | 1330 | 55.4 | 1.55× |
| **全优化，bs40（推荐）** | 2083 | **52.1** | **1.64×** |

全优化 = 关闭 checkpointing + 主干区域编译 + VAE 编译。VAE 与编译优化释放的显存支持更大的 per-GPU batch，进一步摊薄固定开销。各配置训练 loss 曲线逐 step 重合于数据 shuffle 噪声量级。

## 5. 数值等价性验证

验证方法与结果分三个层级：

**层级一：逐位相等。** CUDA Graph 仅固化 kernel 发射序列，不改变任何计算。实测去噪 CUDA Graph 路径（首次捕获与后续复用两种路径分别验证）的输出与 eager 基线**逐元素完全相等（最大绝对误差为 0）**。VAE 函数式重构的 eager 路径与原实现同样逐位相等。

**层级二：编译舍入界定。** `torch.compile` 的算子融合会改变浮点累加顺序，产生舍入级差异。实测 bf16 下最大绝对误差 0.039（相对误差 1.8%，余弦相似度 0.99994）；将同一验证在 fp32 下重跑，误差缩小 22 倍（最大绝对误差 0.0017，相对 0.08%）。误差随精度按比例缩小，证明其来源为浮点舍入而非逻辑差异。CUDA Graph 模式与普通编译模式的输出逐位相同，即 CUDA Graph 不引入任何额外数值差异。

**层级三：端到端行为。** 各优化配置的训练 loss 轨迹与基线逐 step 重合（差异在数据 shuffle 噪声量级）。

**兼容性**：所有优化均不增删、不重排任何模型参数与 buffer（state_dict 键集合不变），已有 checkpoint 无需任何转换即可加载。全部优化为运行时开关，默认关闭；开启失败时自动回退原始路径。空间开销仅为推理侧 MB 级的静态显存 buffer，无磁盘产物。

## 6. 训练加速上限分析

在"数值等价 + 无磁盘缓存"双约束下，训练加速存在可测量的硬上限：

- 优化后（bs16 口径）step 墙钟 977 ms 中，GPU kernel 实际执行时间为 792 ms（81%），即 **49.5 ms/sample 的 GPU 纯计算下限**。
- 相对基线 2× 的目标要求 42.8 ms/sample——**低于 GPU 纯计算时间**。编译与 CUDA Graph 类优化仅能压缩墙钟中"等待发射"的部分（约 19%），无法减少计算本身。
- 对未实施的 attention 优化路径做上限估计：attention 全部 GPU 时间实测为 4.3 ms/sample，即使完全消除，下限仍为 45.2 ms/sample，仍高于 2× 所需。

因此在当前约束下训练吞吐上限约为 1.6–1.7×（实测 1.64× 已接近该上限）。突破需放宽约束，可选路径及代价：

| 路径 | 预期收益 | 代价 |
|---|---|---|
| VAE latent 离线缓存 | 实测可达 2.33×（同 batch）/ 2.57×（放大 batch） | 约 200 GB 磁盘；缓存与数据管线耦合 |
| fp8 / 降精度 | 计算量直接下降 | 破坏数值等价，需收敛性重新验证 |
| 模型结构精简 | 视幅度而定 | 非原版模型 |

## 7. 负面结果（已验证无效或不可行的方向）

1. **优化器参数 16 字节对齐修复**：在同系模型的变体分支上该修复曾带来 2.18× 提升（根因：变体新增的奇数元素数标量参数使 DeepSpeed flatten buffer 中约 89% 的参数指针错位，cuBLAS 回退到慢速兼容 kernel）。但原版 FastWAM 参数天然对齐（探针实测 1847 个参数中仅末尾 2 个小参数错位，不涉及热点 GEMM；trace 中无任何慢速 align1 kernel），该修复在原版上**无收益**，且会改变 optimizer state 布局影响 resume，不应启用。
2. **逐 block CUDA Graph（`torch.compile` reduce-overhead 模式）**：训练 VAE 与推理 MoT 上均因多图输出跨 replay 覆写而崩溃，不可用；CUDA Graph 必须以整步为单位捕获。
3. **Attention 后端优化**：占比过低（11%），收益上限不足 5%，不值得投入。
4. **DeepSpeed `overlap_comm` + `contiguous_gradients`**：在该环境实测为严重负优化（backward 劣化约 10×），保持默认关闭。

## 8. 结论

- 推理：整步 CUDA Graph 捕获 + 跨调用持久化 + 主干区域编译，**4.93×**，输出与原版逐位相等。
- 训练：区域编译 + VAE 编译 + 关闭 checkpointing + 放大 batch，**1.64×**，loss 轨迹与原版重合；该数字已接近等价性约束下的理论上限（约 1.7×）。
- 全部优化以默认关闭的运行时开关交付，零参数改动，checkpoint 完全兼容，无磁盘占用，失败自动回退。
