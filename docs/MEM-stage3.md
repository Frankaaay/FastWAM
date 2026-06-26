# MEM-stage3 — DiT token 空间 fold-current adapter(冻 base,只训 adapter)

> MEM-style factorized temporal attention:历史只用于**增强 current tokens**,随后**立即丢弃**
> history tokens,后端 MoT / action 看到的 token 数、current index、attention mask 全保持 base 形态。
> 把融合位置从 stage1 的 VAE latent 压缩器,搬到 **DiT token 空间**;同时**冻结整个 base**,只训一个
> 轻量 `TemporalFoldAdapter`(加性 + 零初始化门控)。
>
> 分支 `MEM-stage2`。脚本 / run 目录 / wandb 均已统一为 `*stage3*`;功能开关是描述性的
> `model.vae_memory.dit_fold_current=true`(见 §2)。

---

## 0. 速览 (TL;DR)

- **一句话**:patch-embed `[history, current, future]` → `TemporalFoldAdapter` 把历史按**每个空间
  token**折进 current 帧 → **丢历史帧** → 后端见到与 base **逐位同构**的布局。加性 + 零初始化门控
  → step0 严格 == base。
- **训练面**:**冻结整个 base DiT(~5.9B)+ VAE + proprio**,只训 adapter(14 个 param tensor)。
- **设计目标**:同时拿到 stage1(保 rollout)与 stage2(靠近 action conditioning)的优点,规避两者的坑——
  既不接口错配(不碰 conditioning),又不让记忆长成新 backbone(history 不暴露给后端),也不全解冻(不撕裂 rollout)。
- **状态**:已实现 + 双层 smoke 通过(见 §2);**正式训练未启动**,结果待填。

> 基线口径同 [MEM-stage1.md](MEM-stage1.md#基线口径三篇统一):mem-off 49.83/50.5(noise-inclusive 10030),标准 LIBERO base 95.9。

---

## 1. 架构设计

### 1.1 动机:同时绕开前两个 stage 的坑

| stage | 失败模式 | stage3 怎么规避 |
|---|---|---|
| stage1-v1 | 冻死 DiT 读不懂记忆 latent(接口错配) | **不碰 conditioning latent**,current = base plain encode 逐位一致 |
| stage1-v3 | 全解冻 → train/rollout 裂缝 | **冻结整个 base**,只训零初始化 adapter |
| stage2-v1 | prepend 改接口 → 策略依赖新路径,mem-off OOD | **history 不暴露给后端**:折进 current 后立即丢弃,token 数/current index 不变 |

继承 stage1 哲学(前端融合、只把 current 给后端),但融合点更靠近 action conditioning(DiT token 空间)。

### 1.2 Pipeline

```
history pixels [B,3,K_px,H,W]   current/future pixels
      │ frozen VAE plain encode (4n+1)       │ frozen VAE plain encode
      ▼                                       ▼
  history_latents [B,z,K_lat,h,w]        current/future latents
      └───────────────┬───────────────────────┘
                      ▼  cat along time
      video_in = [history_0..K_lat-1, current, future_1..future_{T-1}]
                      │ patch_embedding (Conv3d, patch_size=(1,2,2))  → token grid [B,C,F,H',W']
                      ▼
              TemporalFoldAdapter            ← 每个空间 token:current attend [history..current]
                      │   cur += tanh(gate)*delta ;  丢掉 history 帧
                      ▼
              [current_mem, future_1, ...]   ← T = F - K,与 base 逐位同构
                      ▼
              原 MoT video prefill → action expert denoise
```

训练时 MoT 见 `[current_mem, future_1, ...]`;推理时只见 `[current_mem]`。
代码:adapter 挂载与折叠在 [pre_dit wan_video_dit.py:714-718](../src/fastwam/models/wan22/wan_video_dit.py#L714)
(`fold_adapter(x, num_history_frames, ...)` 后 `num_history_frames=0`)。

### 1.3 核心机制(逐个拆,严格不混)

#### (a) 压缩 / fold —— TemporalFoldAdapter
[wan_video_dit.py:310-389](../src/fastwam/models/wan22/wan_video_dit.py#L310)。在 DiT token 空间(patch_embedding 之后)。
patch 时间维=1,latent 帧 1:1 映射到 token 帧,时间轴布局 `[history_0..history_{K-1}, current, future...]`。

**对每个空间 token 位置独立做一次时间注意力**([:363-378](../src/fastwam/models/wan22/wan_video_dit.py#L363)):
- `ctx = xt[:, :K+1]`(history..current,K+1 帧)= key/value;`cur = xt[:, K]`(current)= query。
- 把空间 `S=H'*W'` **折进 batch**(`B*S`),让 `K+1` 个时间帧成为**唯一被 attend 的轴**:
  `q:[B*S,1,A]`,`k/v:[B*S,K+1,A]`,`flash_attention` → `[B*S,1,A]`。
- **这就是"压缩"**:`(K+1)` 帧 → 每个空间 token **1 个 delta**([:379](../src/fastwam/models/wan22/wan_video_dit.py#L379)),
  压缩比 `(K+1)→1` 但 **per spatial token**,空间分辨率不丢。
- 复杂度 `O(N·K²·D)`,而非 full spatio-temporal 的 `O(K²·N²·D)`。

> ⚠️ 它**不是**把 history 压成一个全局 token,而是把时间信息**折进 current 的每个 spatial token**。

#### (b) gate —— 零初始化标量加性融合(≠ 位置编码)
定义 [:350](../src/fastwam/models/wan22/wan_video_dit.py#L350) `self.gate = nn.Parameter(torch.zeros(()))`;
融合 [:380-383](../src/fastwam/models/wan22/wan_video_dit.py#L380):
```python
gate = torch.tanh(self.gate)          # tanh(0)=0 → step0 无操作
cur = cur + gate * delta              # 加性;gate=0 时 cur 原样不变
```
零初始化 gate 是 **baseline 保命符**:训练 step0 时 current 帧逐位等于无记忆 base。

#### (c) temporal_pos —— per-frame 位置编码(≠ gate)
定义 [:348](../src/fastwam/models/wan22/wan_video_dit.py#L348)
`temporal_pos = nn.Parameter(torch.zeros(1, max_history_frames+1, 1, hidden_dim))`;
加在 ctx 上 [:368](../src/fastwam/models/wan22/wan_video_dit.py#L368)。作用 = 给 K+1 帧"谁先谁后"的顺序感。
**与 gate 是两回事**:gate 管融合强度(一个标量),temporal_pos 管帧序(per-frame 向量)。

#### (d) history dropout —— per-sample 丢历史(反依赖)
训练侧 [fastwam.py:696-700](../src/fastwam/models/wan22/fastwam.py#L696):
```python
if torch.is_grad_enabled() and self.fold_history_dropout > 0.0:
    keep = torch.rand(batch_size) >= self.fold_history_dropout   # 每样本 0/1
    history_keep_mask = keep.to(...)                             # [B]
```
adapter 侧 [:381-382](../src/fastwam/models/wan22/wan_video_dit.py#L381):`gate = tanh(gate) * keep_mask`。
某样本 keep=0 → gate=0 → current 不被历史改写 → 等价 mem-off。三个设计点:
- **per-sample 不 per-batch**:整 batch 丢会让 adapter 脱离计算图、backward 报错;逐样本保证 loss
  永远有一条路径穿过 adapter。
- **用 `torch.is_grad_enabled()` 而非 `self.training`**:memory 模式下 trainer 强制 `model.eval()`
  (见 §1.3e),`self.training=False` 不可靠;`is_grad_enabled()` 训练 True / 推理 False。
- **作用**:让模型也见过"无历史"输入 → **mem-off 永远在分布内**,保 baseline(直接针对 stage2-v1
  的"策略依赖记忆路径"病根)。

#### (e) 冻结策略 —— 只训 adapter
[trainer.py:366-387](../src/fastwam/trainer.py#L366):fold-current 模式下 `model.eval()` +
`requires_grad_(False)` 冻结整个 base,只 `adapter.requires_grad_(True)`(14 个 param tensor);
再 `model.dit.train()` 仅为**重开 MoT 的 gradient checkpointing**(MoT/DiT 无 Dropout/BN,train()==eval()
除 checkpointing);VAE 保持 eval。

### 1.4 关键不变量

- `gate=0` 初始化 → step0 严格退化为 base(同 seed 下 fold 输出与 base 逐位一致)。
- current 仍是 temporal index 0;action expert 的 video conditioning token 数不变。
- 不需要 stage2-v1 那种 action 指向 `K*tokens_per_frame` 的特殊 mask。
- validate 已 history-aware:[wan_video_dit.py:534](../src/fastwam/models/wan22/wan_video_dit.py#L534)
  `num_current_frames = num_latent_frames - num_history_frames`,action 整除性/单帧豁免都按
  `num_current_frames` 校验(num_history_frames=0 时逐位等于 base)。

---

## 2. 训练配置

### 功能开关

| 本文档 | 代码 / 脚本 / 开关 |
|---|---|
| stage3 | 开关 `model.vae_memory.dit_fold_current=true`;脚本 `*_mem_stage3*`;run `runs/mem_stage3`;wandb `mem_stage3` |

> `dit_fold_current` 是**描述性功能名**(fold current),不随 stage 编号变动;stage 标签已全部统一为 stage3。

### 配置表

| 项 | 值 |
|---|---|
| 分支 | `MEM-stage2`,node-1 |
| 开关 | `enabled=false` + `dit_prepend=false` + `dit_fold_current=true`(三选一互斥) |
| 起点 ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt`(base,mem-off 49.83;strict=False 加载,adapter fresh-init + 零门控 → 起点==base) |
| 可训练 | **仅 TemporalFoldAdapter**(14 param tensor) |
| 冻结 | base DiT(video+action ~5.9B)、VAE、proprio、text encoder |
| history | **H5** → 1.0s,K_lat=2;真实配置 `num_frames=33` / `action_horizon=32` / T_lat=9 → fold 堆栈 11 latent 帧 |
| dropout | `fold_history_dropout=0.4`(per-sample) |
| 卡 / batch | 8 卡 × BS=24 = global 192(adapter-only 显存远低于全 DiT,OOM 再降) |
| lr | **1e-5**,cosine + 5% warmup |
| max_steps | num_epochs=10;save_every 1000 |
| output_dir | `runs/mem_stage3` |

脚本:`scripts/train_mem_stage3.sh`(正式) / `scripts/train_mem_stage3_smoke.sh`(1-step) /
`scripts/test_fold_current.py`(无数据 CPU 不变性单测)。

### smoke 验证(双层,均 PASS)

1. **无权重不变性**(`test_fold_current.py`):gate=0 ≡ base **逐位一致**;接口不变;mem-off ≡ base;
   gate>0 激活时 max|Δ|=0.6957。
2. **1-step 真数据**(`train_mem_stage3_smoke.sh`):`loss=0.1235`(action 0.0009 / video 0.1226),
   14 adapter tensor 解冻,base ckpt strict=False 加载,checkpoint 保存通。

---

## 3. 结果

> **正式训练未启动(用户决定先不训)。** 评测口径:LIBERO-plus `INCLUDE_NOISE=1`(10030),对照 mem-off 49.83/50.5。

| step | Total | Camera | Robot | Lang. | Light | BG | Layout |
|---|---|---|---|---|---|---|---|
| TBD | | | | | | | |

**验收顺序**(先"不退化",再看 memory 增益):
1. gate=0 smoke:fold-current 与 base 输出近似一致 ✅(已过)。
2. 标准 LIBERO:接近 base 95.9,不能像 stage2-v1 那样崩。
3. LIBERO-plus:鲁棒性门禁,目标不明显低于 mem-off ~50;永远 `INCLUDE_NOISE=1`。
4. **memory benchmark**:base vs stage1 vs stage3 三方对比——**这里才判定 memory 是否有效**。
5. 真机叠衣服:最终外部验证(需先定义 fold success / phase accuracy / history-sensitive action accuracy)。

---

## 4. 分析

### 4.1 设计自洽性(为什么预期能保住 baseline)

- **加法 + 零初始化门控** → step0 逐位 == base(构造保证)。
- **冻结 base** → 继承 stage1 的 rollout 稳定性,不重蹈 stage1-v3 / stage2-v1 全解冻翻车。
- **current conditioning = base plain encode 逐位一致** → 避开 stage1 的接口错配。
- **history dropout** → 模型也见过"无历史",mem-off 永远在分布内 → 避开 stage2-v1 的"策略依赖记忆路径"。

### 4.2 与 MEM 原文 / stage1 的关系

stage1-v1 哲学正确(前端融合、只给 current),保住了 rollout(标准 95.9,plus 仅小掉到 45.64)。
stage3 继承此原则,但把融合从 **VAE latent 压缩器** 移到 **DiT token 空间**:
- 比 stage1 更靠近 action conditioning,理论上更易影响动作。
- 比 stage2-v1 更像 MEM:后端只见 current tokens,history 不 prepend 进后端输入。
- 长程 language memory 暂不并入 FastWAM(交给 robo agent / planner);FastWAM 只负责短程视觉 memory。

### 4.3 风险 / 待验证

- adapter-only 自由度是否足够把"历史有用"学进 current(stage1 净负的另一面:容量太小可能学不动)。
- fold 推理 rollout 路径(`prefill_video_cache` 喂 buffered 历史)是另一条代码路径,**首次 eval 先
  `PILOT=16`** 确认不报错、成绩非 0,再全量。

### 5. 目录速查

| 内容 | 路径 |
|---|---|
| adapter 代码 | `wan_video_dit.py`(`TemporalFoldAdapter` :310,`enable_history_fold` :484,fold :714) |
| dropout / encode | `fastwam.py`(`_encode_history_latents` :394,dropout :696,infer fold :1224) |
| 冻结逻辑 | `trainer.py`(fold-current 分支 :366) |
| 脚本 | `scripts/train_mem_stage3.sh` / `_smoke.sh` / `test_fold_current.py` / `eval_mem_stage3.sh` |
| run 目录 | `runs/mem_stage3/` |
| wandb 看板 | wandb.ai/yichx14-uc-irvine/fastwam-mem(offline → jump 同步) |
