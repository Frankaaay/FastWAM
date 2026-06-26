# MEM-stage1 — VAE 侧 spatio-temporal 短时记忆(只训 4 参数,冻 DiT)

> 给 FastWAM 加**短期视觉记忆**的第一条路线:在**冻结的 Wan VAE encoder 内部**,把历史帧
> 通过 MEM-style temporal attention 压进 current latent,VAE 输出后切掉历史 latent,只把
> current latent 送进原 FastWAM DiT。后端接口逐位不变。
>
> 目标(三个 stage 共用):在 LIBERO-plus 鲁棒性测试上 **超过无记忆基线**,证明「短期记忆提升鲁棒性」。

---

## 0. 速览 (TL;DR)

- **一句话**:VAE encoder 中所有 spatial `AttentionBlock` → `SpatioTemporalAttentionBlock`,
  历史在 VAE 内逐层注入 current 表征,只把 current latent 给 DiT。
- **训练面**:只训 temporal 4 参数(`temporal_proj` / `temporal_gate` / `temporal_pos`),
  base VAE + 整个 DiT/MoT 全冻。
- **结论**:**保住了 rollout**(标准 LIBERO 95.9,没崩),但**记忆净负**(libero-plus 45.64 < mem-off 49.83)。
  根因 = **接口错配**:冻死的 DiT `patch_embedding` 是在「无记忆 latent」上训的,读不懂被记忆改写过的 latent。
- **状态**:已评测,作为 stage2/stage3 的对照与教训来源。**v2/v3(解冻 patch_embed / 全解冻冷启动)见 §4 教训,不再单列。**

### 基线口径(三篇统一)

| 基线 | 数值 | 口径 |
|---|---|---|
| Fast-WAM(paper Table 4) | 51.5 | — |
| **mem-off(本仓库复现,无记忆)** | **49.83** | LIBERO-plus noise-inclusive(10030) |
| 标准 LIBERO base(无扰动) | **95.9**(原版 97.6) | 40 task,无扰动 |

> ⚠️ libero-plus 所有 eval 一律 `INCLUDE_NOISE=1`(10030),否则与 49.83 不可比(见 §4 v2 教训)。

---

## 1. 架构设计

### 1.1 动机

MEM 原文 video encoder 做三件事:(1) 输入过去多帧 + 当前 observation;(2) 在中间层交替
spatial / causal temporal attention;(3) 丢弃过去 timestep tokens,只把 current 表征给 VLA backbone。
迁移到 FastWAM 的关键不是把 VAE 改成 ViT,而是把 **space-time separable attention** 移植进
Wan VAE 已有的 attention block。

| MEM | FastWAM-VAE 改造 |
|---|---|
| ViT image encoder | Wan/FastWAM 卷积 Video VAE encoder |
| intermediate ViT layers | VAE encoder 里的 `AttentionBlock` 所在层 |
| spatial attention | 原 VAE 的 spatial self-attention(保留) |
| causal temporal attention | **新增** temporal branch |
| drop past tokens | VAE 输出后切掉 history latent |
| pass current to VLA | 只把 current latent 给 DiT |

### 1.2 Pipeline

```
原版:   current_video ── VAE ──> z_cur ──> Video DiT / Action DiT

stage1:  history_video + current_video
              │ concat along time (T = 4n+1)
              ▼
         Spatio-Temporal VAE Encoder         ← 历史在此逐层融进 current
              │
              ▼
         z_mem = [z_hist.., z_cur]
              │ 切片:只保留 current latent slot
              ▼
         z_cur ──> 原 FastWAM Video DiT / Action DiT   ← 接口逐位不变
```

DiT **不接收 history latent**;history 只在 VAE encoder 内部经 temporal attention 压进 current latent。

### 1.3 核心机制(逐个拆,勿混)

**(a) SpatioTemporalAttentionBlock —— 空间分支 + 时间分支**
[wan_video_vae.py:345](../src/fastwam/models/wan22/wan_video_vae.py#L345)。原 `AttentionBlock`
([:304](../src/fastwam/models/wan22/wan_video_vae.py#L304))把 `[B,C,T,H,W]→[BT,HW,C]` 做**帧内空间** self-attn。
新 block 保留它,再加一条**时间分支**:reshape 成 `[BHW,T,C]`,同一空间 patch 沿时间 attend
(causal,current 只看 past/current)。两条分支组织方式互补,不是同一种 attention。

**(b) QKV 共享、temporal_proj 独立**
时间分支**复用** spatial 的 `to_qkv`(让 history/current token 在同一表征坐标系里比较),
但**新增独立** `temporal_proj`([:389](../src/fastwam/models/wan22/wan_video_vae.py#L389)):
空间上下文与时间上下文写回 current 的目的不同,输出投影要分开。

**(c) temporal_gate —— 零初始化标量融合(≠ 位置编码)**
[wan_video_vae.py:392](../src/fastwam/models/wan22/wan_video_vae.py#L392) `temporal_gate=nn.Parameter(torch.zeros(()))`。
forward 融合 [:458](../src/fastwam/models/wan22/wan_video_vae.py#L458):
`x = x + tanh(temporal_gate) * temp`。`tanh(0)=0` → 初始严格等价原 VAE,gate 训练中逐渐打开。
**注意不 zero-init `temporal_proj`**:identity-at-start 由 gate 一个就够;若 proj 也置零会卡住梯度
([代码注释 :380-386](../src/fastwam/models/wan22/wan_video_vae.py#L380))。

**(d) temporal_pos —— 时间位置编码(≠ gate)**
给时间分支注入帧序信息,与 gate 是两回事(一个管融合强度,一个管谁先谁后)。

**(e) copy-init `temporal_proj` ← spatial `proj`**
[init_temporal_from_spatial :470](../src/fastwam/models/wan22/wan_video_vae.py#L470)。
调用顺序必须 `load_state_dict(strict=False)` → `init_temporal_from_spatial`,因为 `proj` 预训练权重要先加载。

### 1.4 关键不变量

- `temporal_gate=0` → VAE 输出逐位 == 原版 → 训练起点 == base。
- DiT 输入仍是单帧 `z_cur`,token 数 / 接口不变。
- 可训练参数键白名单 `MEMORY_PARAM_KEYS=("temporal_proj","temporal_gate","temporal_pos")`
  ([:478](../src/fastwam/models/wan22/wan_video_vae.py#L478)),其余全冻。

---

## 2. 训练配置

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_mem_stage1_v1.sh` |
| run 目录 | `runs/mem_temporal_libero/` |
| ckpt | `runs/mem_temporal_libero/checkpoints/weights/step_021700.pt` |
| 可训练参数 | **4 个**(VAE temporal attn / pos / proj / gate) |
| 冻结 | DiT / MoT / base VAE 全冻 |
| history | `history_video_frames=16` → **3.2s**(16×4÷20) |
| 训练量 | 10 epoch,batch 16,step 0→21700(loss ~step4000 压平,已收敛) |

![stage1-v1 loss](stage1_v1_loss.png)

**历史帧换秒**:`秒数 = history_video_frames × action_video_freq_ratio ÷ fps`。
LIBERO `ratio=4`([configs/data/libero_2cam.yaml](../configs/data/libero_2cam.yaml))、`fps=20`。
约束:VAE memory 要求 `(K+1)%4==1`,换数据集必须重算。

---

## 3. 结果

| 评测 | 数值 | 对照 | 判读 |
|---|---|---|---|
| 标准 LIBERO Avg | **95.9** | 原版 97.6(-1.7;Long 掉最多 -3.0) | rollout **没崩** |
| LIBERO-plus Total | **45.64**(noise-inclusive) | mem-off 49.83 | **记忆净负 ~4.2** |

---

## 4. 分析

### 4.1 为什么净负:接口错配(不是崩)

冻死的 DiT 输入层 `patch_embedding` 是在「无记忆 latent」上训出来的,**读不懂被记忆改写过的 latent**
→ 接口错配、记忆净负。这**不是记忆本身没价值**,而是 DiT 没机会适应。Long suite 掉最多正是佐证。
关键是:**rollout 本身不崩**(标准 LIBERO 仍 95.9),只是小幅净负。

### 4.2 v2 / v3 教训(已弃,压缩保留)

沿 stage1 试过两个补丁,均失败,直接催生了后两个 stage 的方向:

- **v2(解冻 DiT 输入接口 `patch_embedding`,6 参数)**:Total 46.9,但**口径事故**——算在**不含 noise 的
  8429 集**上,与 49.83/45.64(10030)**不可比**,"+1.3"是假象;真正可比仍是 v1 的 45.64。
  即便只看接口对齐方向,6 参数自由度仍不够修透表征错配。
  → **教训:以后 libero-plus 一律 `INCLUDE_NOISE=1`(10030)。**
- **v3(从 base 冷启动、全 DiT + temporal 联合 co-adapt,H4)**:训练 loss 健康(2.18→0.178,与 v1 同量级),
  但 eval **崩**到 2–17.5%(峰值 step3000 仅 17.5,Camera/BG 全程 0)。**loss 健康 + eval 崩 =
  train/rollout 泛化裂缝**(exposure bias / 误差累积):全解冻让 DiT 走上一条"只在 teacher-forcing
  单步最优、闭环自回归下不稳"的路;zero-init 门控只挡第 0 步,挡不住几千步全解冻的漂移。

### 4.3 给后续 stage 的结论

1. **「前端融合、只把 current 给 DiT」的哲学是对的** → 保住了 rollout。stage3 继承此原则。
2. **冻死 DiT 会接口错配** → 需要让 DiT 有机会适应记忆(stage2 用全解冻,但翻车)。
3. **全解冻冷启动会撕裂 train/rollout** → stage3 改为**冻结 base、只训轻量 adapter**,正是规避 v3。

### 5. 目录速查

| 内容 | 路径 |
|---|---|
| 脚本 / run | `scripts/train_mem_stage1_v1.sh` / `runs/mem_temporal_libero/` |
| VAE temporal 代码 | `src/fastwam/models/wan22/wan_video_vae.py`(`SpatioTemporalAttentionBlock` :345) |
| 标准 LIBERO eval | `evaluate_results/libero/` |
| LIBERO-plus eval | `scripts/eval_libero_plus.sh` + `experiments/libero/summarize_libero_plus.py` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem |
