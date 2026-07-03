# 数值无损的 Transformer/Diffusion 训练与推理加速技术调研

日期：2026-07-03

目标系统：FastWAM，5B video DiT（Wan2.2，30 层，hidden 3072，FFN 14336，bf16）+ 1B action DiT 的 MoT 结构；H200 8 卡；PyTorch 2.7.1+cu128；accelerate + DeepSpeed ZeRO-1；视频序列约 480 token。

约束：只考虑不改变训练数学语义和模型效果的技术。排除 FP8、量化、蒸馏、减少采样步数、稀疏化、改变 token/分辨率/模型结构等方案；接受 kernel fusion、CUDA Graph、精确 attention backend 切换等浮点重排序级变化。

当前基线：训练 step 约 1139ms，其中 GEMM 约 430ms（接近 roofline）、elementwise 约 328ms、attention forward+backward 约 148ms（当前走 memory-efficient backend，bool mask 阻断 flash）、launch/通信缝隙约 230ms（约 9000 kernel launch/step）。推理 bs=1 denoise loop 明显 launch-bound，墙钟约 27ms/步而 GPU 时间约 8ms/步。

## 优先级汇总

| 优先级 | 技术方向 | 结论 | 预期收益量级 | 集成风险 | 主要证据 URL |
| --- | --- | --- | --- | --- | --- |
| P0 | 推理 denoise loop CUDA Graph / 静态缓冲整图捕获 | 最适合当前 bs=1 launch-bound 症状；如果 schedule、shape、mask 静态，收益上限最高。 | 推理单步墙钟理论上可从 27ms 向 8ms 靠近；实际先按 1.5-3.0x 估计。 | 中：需要固定张量地址、静态 shape、去掉 CPU sync；采样器状态要改成静态 buffer。 | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ ，https://github.com/chengzeyi/stable-fast |
| P1 | 训练侧分段 CUDA Graph / CUDAGraph Trees | 可减少 9000 launch/step 和 230ms 缝隙，但 DeepSpeed、动态 mask、optimizer/comm 使整步捕获风险高；建议先分段。 | 若消掉 30-60% launch/comm 缝隙，约 70-140ms/step，整步约 6-12%。 | 高：DeepSpeed ZeRO、NCCL、autograd hooks、动态 shape 都会增加跳图或错误风险。 | https://docs.pytorch.org/docs/2.7/torch.compiler_cudagraph_trees.html ，https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ |
| P1 | attention backend：cuDNN SDPA / FlexAttention / FA3 | 必须先解决 `first_frame_causal + per-sample key padding` 的表达；FlexAttention 表达力最好，FA3/Hopper 性能最好，cuDNN PyTorch 路径要实测 mask 是否命中。 | attention-only 1.2-2.0x；折到整步约 2-7%，上限受 148ms attention 占比约束。 | 中高：mask 兼容性、dropout/backward、数值差异回归、PyTorch backend fallback 都要验证。 | https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html ，https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html ，https://tridao.me/blog/2024/flash3/ ，https://docs.pytorch.org/docs/2.7/nn.attention.flex_attention.html |
| P2 | DeepSpeed ZeRO-1 `overlap_comm` 与 bucket A/B | 公开 issue 没找到明确“病态变慢”的可复用案例；找到的是 `overlap_comm + torch.compile` 多 stream gradient bucket correctness/race 风险。 | bucket/overlap 调优先按 0-5% 估计；更可能是避免退化和抖动，而不是大幅提速。 | 中：不同 bucket size 会改变 overlap、显存和 NCCL 排队；`torch.compile` 下 overlap_comm 有已知竞态讨论。 | https://www.deepspeed.ai/docs/config-json/ ，https://github.com/deepspeedai/DeepSpeed/pull/8080 |
| P2 | `torch.compile(mode="max-autotune-no-cudagraphs")` | 官方说明它是 max-autotune 但不启用 CUDA Graph；未查到相对 `default` 的公开稳定增益数据。 | 对本系统先按 0-3% 或不确定处理；GEMM 已近 roofline，收益主要看 epilogue fusion/模板 matmul 是否还能命中。 | 中低：编译时间和 cache 成本更高；可能引入不同 kernel selection，需要数值回归。 | https://docs.pytorch.org/docs/2.7/generated/torch.compile.html |
| P3 | FSDP2 替代 ZeRO-1 | 数学等价的分片训练路线，PyTorch 文档强调 per-parameter sharding、prefetch 和 collective scheduling 控制；不应作为短期小改。 | 单机 8 卡上收益不确定，先按 0-10% 探索项；主要价值是更可控的 overlap/compile 组合。 | 高：需要替换 accelerate+DeepSpeed 集成、checkpoint/state dict、optimizer 和 launch 脚本。 | https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html ，https://github.com/pytorch/torchtitan |
| P3 | 视频生成系统中的无损系统优化 | Open-Sora、HunyuanVideo、FastVideo、TorchTitan 公开材料支持 FA/FA3、sequence parallel、FSDP2、selective checkpointing、torch.compile 等；MovieGen 公开页未给出可直接复用的 kernel/graph 细节。 | 对当前 480 token、单节点训练的直接收益不如 P0/P1；对更长视频、多机或多 GPU 推理更有价值。 | 中高：SP/TP/USP 会重写并行拓扑和通信路径；activation checkpointing 省显存但通常增加算力。 | https://github.com/hpcaitech/Open-Sora ，https://github.com/Tencent-Hunyuan/HunyuanVideo ，https://github.com/hao-ai-lab/FastVideo ，https://github.com/pytorch/torchtitan ，https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/ |

## 1. cuDNN SDPA、FlashAttention-3、FlexAttention

结论：注意力后端有可做空间，但不是当前最大整步瓶颈；先做 mask 表达与 backend 命中验证，再谈替换。PyTorch 2.7 的 SDPA 文档说明 `attn_mask` 支持 bool mask，且可通过 `sdpa_kernel` 选择 fused backend；`SDPBackend` 中确实有 `CUDNN_ATTENTION`，但文档同时提醒 fused backend 对输入有局限，不支持时会 fallback 或报警，证据见 https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html 和 https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.SDPBackend.html 。cuDNN attention 文档显示 Hopper 支持 fp16/bf16 SDPA fprop/bprop，支持 padded/ragged、causal、sliding window、additive bias、arbitrary masking 等 mask 类别；但这不能自动等同于 PyTorch `CUDNN_ATTENTION` 对任意 bool `attn_mask` 都会命中，证据见 https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html 。FlashAttention-3 在 H100 上公开称 FP16 比 FA2 快 1.5-2.0x，但 FA3 Python API 主要暴露 causal/window/varlen 等结构化参数，不是直接吃任意 dense bool mask；per-sample padding 可以走 varlen/`cu_seqlens`，`first_frame_causal` 若不能映射成 causal/window/偏置模式，就要自定义或改用 FlexAttention，证据见 https://tridao.me/blog/2024/flash3/ 和 https://github.com/Dao-AILab/flash-attention 。FlexAttention 的 `mask_mod(b,h,q_idx,kv_idx)->bool` 与 `and_masks/or_masks` 能表达 `first_frame_causal + per-sample key padding`，但 PyTorch 2.7 标为 prototype，BlockMask 构造也要缓存；预期收益按 attention-only 1.2-2.0x、整步 2-7% 估算，证据见 https://docs.pytorch.org/docs/2.7/nn.attention.flex_attention.html 。

## 2. CUDA Graphs：训练侧与推理侧

结论：推理侧优先级最高，训练侧只建议先分段验证。PyTorch CUDA Graphs 博文说明 replay 会用单次 `cudaGraphLaunch` 提交整段 GPU work，主要收益来自消除 Python/C++/driver launch overhead；它还给出 full forward/backward/optimizer step 捕获示例，以及 MLPerf BERT 1.12x、Mask R-CNN 最高约 1.7x 的公开结果，证据见 https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ 。这与 FastWAM 推理 bs=1 的 27ms 墙钟 vs 8ms GPU 时间高度吻合，若固定 denoise step 数、scheduler 分支、shape 和 buffer 地址，手动捕获一个或多个采样 step 是最值得先做的 POC。公开 diffusion 实现中，stable-fast 明确支持把 UNet/VAE/TextEncoder 捕获为 CUDA Graph，并说明小 batch 可降 CPU overhead，且 StableVideoDiffusionPipeline 有 2x speedup 描述；它不是完整 denoise loop 整图，但证明 diffusion pipeline 上 CUDA Graph 是公开实践，证据见 https://github.com/chengzeyi/stable-fast 。训练侧方面，Megatron Core 2026 报告把 CUDA Graphs 列为 computation 优化之一，但我未查到 DeepSpeed ZeRO-1 + full-iteration graph 的直接公开可套实现；因此训练侧应先用 `make_graphed_callables` 或 CUDAGraph Trees 捕获稳定 DiT 子段，收益按整步 6-12% 上限估算，证据见 https://arxiv.org/abs/2603.07685 和 https://docs.pytorch.org/docs/2.7/torch.compiler_cudagraph_trees.html 。

## 3. `torch.compile(mode="max-autotune-no-cudagraphs")`

结论：可以做 A/B，但不能把它当确定收益项。PyTorch 2.7 文档说明 `default` 是性能与开销的平衡，`max-autotune` 会启用 Triton/template matmul 和 GPU convolution autotune 并默认启用 CUDA Graph，而 `max-autotune-no-cudagraphs` 是不带 CUDA Graph 的 max-autotune，证据见 https://docs.pytorch.org/docs/2.7/generated/torch.compile.html 。文档列出的相关 options 包括 `epilogue_fusion`、`max_autotune`、`shape_padding`、`triton.cudagraphs`，其中最可能对 FastWAM 有用的是 epilogue/pointwise 融合和少量 matmul template 选择。公开检索未找到 `max-autotune-no-cudagraphs` 相对 `default` 在大型 DiT/Transformer 训练上的稳定增益数据；因此本备忘录不给确定加速结论。考虑当前 GEMM 已接近 roofline、区域 compile default 已启用，预期收益先按 0-3% 或不确定处理；风险是编译时间增加、kernel selection 变化以及局部数值回归需要重新比较。

## 4. DeepSpeed ZeRO-1 `overlap_comm`、bucket 调优、FSDP2 迁移

结论：DeepSpeed bucket/overlap 值得做小矩阵 A/B，但不是优先大改；FSDP2 是中期替代路线。DeepSpeed 配置文档说明 `overlap_comm` 试图将梯度 reduction 与 backward 重叠，`reduce_bucket_size` 和 `allgather_bucket_size` 默认都是 5e8 elements，`contiguous_gradients` 默认开启，证据见 https://www.deepspeed.ai/docs/config-json/ 。我没有找到一个可直接引用的 DeepSpeed GitHub issue 证明 ZeRO-1 `overlap_comm` 会稳定导致“病态变慢”；找到的强证据是 PR #8080/#8061 指出 ZeRO 1/2 在 `overlap_comm + torch.compile` 下，IPG bucket 可能由多个 autograd stream 写入，而 `average_tensor` 只等待 current stream，可能导致 all-reduce 读到未完成 bucket 并出现 NaN，证据见 https://github.com/deepspeedai/DeepSpeed/pull/8080 。这意味着如果继续组合区域 compile 与 overlap_comm，正确性与 stream 同步要作为风险项，不应只看速度。bucket 调优建议只做具名配置 A/B：`overlap_comm=false/true`，`reduce_bucket_size` 取 1e8、2e8、5e8，观察 backward 尾部 NCCL 排队、step p50/p95、显存峰值和数值一致性；预期收益先按 0-5%。FSDP2 文档说明它使用 DTensor per-parameter sharding，支持手动 prefetch 和 collective scheduling，且每个 shard group 的参数 all-gather/梯度 reduce-scatter 可用于通信计算 overlap；这是数学等价迁移，但集成风险高，证据见 https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html 。

## 5. 视频生成训练系统报告中的无损优化

结论：这些系统能佐证方向，但要过滤掉不符合硬约束的技术。Open-Sora 2.0 README 明确要求或推荐 xFormers、flash-attn，并提到可选 FlashAttention-3；也展示 ColossalAI sequence/tensor parallel 相关效率测试，这些属于可保持数学语义的系统优化，但 shift-window attention、VAE/patch/token 结构变化不适合作为 FastWAM 的无损 retrofit，证据见 https://github.com/hpcaitech/Open-Sora 。HunyuanVideo README 发布了由 xDiT 驱动的 parallel inference，支持 sequence parallel/USP 参数；这更适合长序列或多 GPU 推理，对当前 bs=1 launch-bound 先级低于 CUDA Graph，证据见 https://github.com/Tencent-Hunyuan/HunyuanVideo 。FastVideo README 把 FSDP2、sequence parallelism、selective activation checkpointing、multiple attention backends 列为训练/推理基础设施；其中 sparse attention、distillation、FastWan/DMD 等应按本任务硬约束排除，证据见 https://github.com/hao-ai-lab/FastVideo 。TorchTitan README 支持 FSDP2、TP/PP/CP、selective/full activation checkpointing、torch.compile、profiling 等 PyTorch-native 训练栈能力；其中 FP8/MXFP8 属于排除项，证据见 https://github.com/pytorch/torchtitan 。MovieGen 公开页面只说明模型族和生成能力，没有查到足够细的无损 kernel、CUDA Graph、DeepSpeed/FSDP 配置细节可直接用于 FastWAM，证据见 https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/ 。

## 6. 其他 kernel launch 开销消减途径

结论：除手写 CUDA Graph 外，`torch.compile` 的 CUDAGraph Trees 是最贴近现有 PyTorch 栈的 launch 消减机制，但要按静态子图分段。PyTorch CUDAGraph Trees 文档说明 CUDA Graphs 将多个 GPU ops 作为一个 CPU launch 提交，适合 CPU overhead 高或小计算模型；同时要求相同 kernel、相同参数、相同内存地址、静态 shape，并列出 input mutation、CPU ops、多设备 ops、动态 shape、incompatible ops 等跳过原因，证据见 https://docs.pytorch.org/docs/2.7/torch.compiler_cudagraph_trees.html 。该文档还说明 CUDAGraph Trees 可为不同路径建立 graph tree，共享 memory pool，并支持含 NCCL operators 的函数，但动态 shape 会对每个唯一 shape 重录，许多重录会带来内存和收益风险。PyTorch CUDA Graphs 博文给出的 CPU-bound 训练案例说明，小 batch/短 kernel 中 launch 间隙可被 graph replay 大幅压缩；这与 FastWAM 当前约 9000 launch/step 的症状相关，证据见 https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ 。对训练，不建议一开始 fullgraph 包住 accelerate+DeepSpeed step；更稳的方式是先捕获不含 CPU 分支和跨设备逻辑的 DiT block/attention/MLP 子段，再扩大到 forward/backward 稳定区域。

## 针对本系统的建议执行顺序

1. 先做推理侧 CUDA Graph POC：固定 batch=1、固定 denoise steps、固定 latent/text/action 条件 shape，预分配 `static_latent/static_t/static_cond/static_out`，捕获一个完整 denoise step 或一小段连续 steps；目标是把单步墙钟从 27ms 明显拉向 GPU 8ms，验证指标为同 seed 输出一致、每步 wall/GPU time、显存峰值和 graph replay 成功率。
2. 并行做 attention mask backend 微基准：抽取真实 Q/K/V shape、dropout、`first_frame_causal` 和 per-sample padding mask，分别强制 `SDPBackend.CUDNN_ATTENTION`、当前 memory-efficient、FlexAttention BlockMask、FA3 varlen/causal 可表达版本；目标是确认哪个 backend 真命中、数值误差范围、attention fwd+bwd ms 和整步折算收益。
3. 训练侧先做分段 CUDAGraph Trees/`make_graphed_callables`，不要先捕获整个 DeepSpeed step；优先选择 shape 固定、无 CPU sync、无 NCCL 的 repeated DiT block 子段，记录跳图原因和显存变化，再决定是否扩大到更多 forward/backward 区域。
4. 做 DeepSpeed 小矩阵 A/B：在同一短 run 上测试 `overlap_comm` 开关和 1e8/2e8/5e8 bucket，必须同时记录速度、p95、NCCL 排队、NaN/grad norm、显存峰值；若与 compile 同时开启，特别关注 PR #8080 描述的多 stream bucket 风险。
5. 最后再评估 `max-autotune-no-cudagraphs` 和 FSDP2：前者作为低成本 A/B，不预设收益；后者只有在 ZeRO-1 overlap/compile/graph 组合持续受限，且愿意改训练栈和 checkpoint 流程时才进入设计。

## 本次未采用或需谨慎的方向

- FP8、量化、蒸馏、减少 denoise steps、sparse attention、VSA、FastWan/DMD、VAE/token 结构压缩均不符合本任务硬约束。
- activation checkpointing 数学上可保持训练目标，但通常以更多 recompute 换显存，不是当前 GEMM/launch-bound 的首要加速项。
- sequence/tensor/context parallel 数学上可等价，但对当前单节点 8 卡、约 480 token 的问题，集成成本可能高于收益；更适合长视频、更长上下文或多机规模化。
- 未查到 DeepSpeed ZeRO-1 full-iteration CUDA Graph 的直接公开可复用实现；未查到 MovieGen 公开材料中足够细的无损 kernel/graph 配置。
