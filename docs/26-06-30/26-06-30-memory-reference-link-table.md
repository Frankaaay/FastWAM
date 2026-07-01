# Memory / 中程记忆参考链接表

日期：2026-06-30

这张表只做快速跳转索引。四个 benchmark 和三个架构设计分开列；如果某项没有找到稳定官方 project/code，就显式标注“未确认”。

## 四个 Benchmark

| 类别 | 名称 | 主要用途 | Paper / arXiv | Project / Website | Code | Dataset / Docs | 本地 PDF |
|-|-|-|-|-|-|-|-|
| Benchmark | MemoryBench / SAM2Act+ MemoryBench | 短程空间记忆、动作回忆；适合 v4 短历史 sanity check | [SAM2Act arXiv](https://arxiv.org/abs/2501.18564) / [HF paper](https://huggingface.co/papers/2501.18564) | [SAM2Act project](https://sam2act.github.io/) | [sam2act/sam2act](https://github.com/sam2act/sam2act) | [hqfang/memorybench](https://huggingface.co/datasets/hqfang/memorybench), [sam2act-datasets](https://huggingface.co/datasets/hqfang/sam2act-datasets) | `papers/sam2act.pdf` |
| Benchmark | RMBench | RoboTwin 2.0 上的 9 个 memory-complexity manipulation tasks；适合中程 memory 能力评估 | [arXiv](https://arxiv.org/abs/2603.01229) / [HTML](https://arxiv.org/html/2603.01229v1) | [rmbench.github.io](https://rmbench.github.io/) | [RoboTwin-Platform/RMBench](https://github.com/RoboTwin-Platform/RMBench) | 同 project/code；基于 RoboTwin 2.0 | `papers/rmbench.pdf` |
| Benchmark | RoboMME | 16 个任务、4 类 memory taxonomy：Counting / Permanence / Reference / Imitation | [arXiv](https://arxiv.org/abs/2603.04639) / [HTML](https://arxiv.org/html/2603.04639v1) / [HF paper](https://huggingface.co/papers/2603.04639) | [robomme.github.io](https://robomme.github.io/) | [RoboMME/robomme_benchmark](https://github.com/RoboMME/robomme_benchmark) | [LeRobot RoboMME docs](https://huggingface.co/docs/lerobot/main/robomme), [sample data](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data_sample) | `papers/robomme.pdf` |
| Benchmark / Eval protocol | ReMem-VLA Extended MemoryBench | MemoryBench 改进版 + 600+ frames long-horizon task；适合更严格的 short-to-medium memory 测试 | [arXiv](https://arxiv.org/abs/2603.12942) / [HTML](https://arxiv.org/html/2603.12942v1) | 未找到稳定官方 project page | 未找到稳定官方 code | 评估协议在 paper 中；不是独立标准 benchmark repo | `papers/remem.pdf` |

## 三个架构设计参考

| 类别 | 名称 | 主要借鉴点 | Paper / arXiv | Project / Website | Code | 备注 | 本地 PDF |
|-|-|-|-|-|-|-|-|
| 架构设计 | CronusVLA | multi-frame motion feature cache；action decoder cross-attend motion history；适合动作连续性 / motion memory | [arXiv](https://arxiv.org/abs/2506.19816) / [PDF](https://arxiv.org/pdf/2506.19816) | [CronusVLA project](https://lihaohn.github.io/CronusVLA.github.io/) | [InternRobotics/CronusVLA](https://github.com/InternRobotics/CronusVLA) | 可作为 FastWAM 的 Cronus-lite motion summary 参考 | `papers/cronusvla.pdf` |
| 架构设计 | EventVLA | visual anchors + Keyframe Evidence Memory；动态写入 task-critical raw keyframes；适合中程 transient evidence | [arXiv HTML](https://arxiv.org/html/2606.20092v1) / [HF paper](https://huggingface.co/papers/2606.20092) | 未单独确认 project page；GitHub 为主入口 | [InternRobotics/EventVLA](https://github.com/InternRobotics/EventVLA) | repo 同时发布 RoboTwin-MeM benchmark | `papers/eventvla.pdf` |
| 架构设计 | KEMO | robot kinematics + visual filtering 检测 event keyframes；cross-attention + gated residual fusion；keyframe-aligned loss weighting | [arXiv](https://arxiv.org/abs/2606.23589) / [HTML](https://arxiv.org/html/2606.23589v1) | [Hatty-z.github.io/KEMO](https://Hatty-z.github.io/KEMO) | [cytoplastm/KC-VLA](https://github.com/cytoplastm/KC-VLA)（外部页面提到，需后续确认） | 比 EventVLA 更新；可作为 EventVLA-lite 后续参考 | 暂无 |

## 快速选择

| 目标 | 优先看 |
|-|-|
| 证明 v4 短程 memory 有用 | MemoryBench、ReMem-VLA Extended MemoryBench |
| 系统评估 memory 类型 | RoboMME |
| 评估中程 / keyframe memory | RMBench、EventVLA / RoboTwin-MeM、KEMO |
| 评估动作平滑和 action memory | CronusVLA + 自定义 smoothness metrics |
