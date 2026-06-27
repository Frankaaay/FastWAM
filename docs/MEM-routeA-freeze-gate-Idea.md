# Route A:冻结 base + 零初始化门控(prepend 记忆的止血方案)

> 分支基线:`MEM-stage2`(prepend 路线)。本方案**不改记忆注入的总体结构(仍 prepend)**,
> 只解决 Ablation A 暴露的头号病因:**全 DiT 冷启动全量重训把策略重写成"离开历史即崩"的
> 历史依赖体**(step12000 记忆 OFF = 1.5%,base = 50.5%)。
> 目标:**让"加记忆"在最坏情况下也 == base,不再可能退化**;在此前提下再争取记忆增益。

---

## 1. 设计不变量(必须满足,可一键验证)

**init 时(门控关闭),整模型逐位等于 base —— 无论喂不喂历史。**

- 无历史:走原版 `first_frame_causal`,本来就 == base(已验证 `wan_video_dit.py:489`)。
- 有历史:门控关闭时,current/future **完全不 attend 历史** → current K/V 与 base 逐位相同
  → action 输出 == base。

只要这条成立,clean 在 step 0 就 == 95.9,且训练只能在 base 之上**做加法**,
**结构上不可能再出现 Ablation A 的崩塌**。

---

## 2. 两处代码改动

### 2.1 `wan_video_dit.py`:把历史注意力改成"零初始化门控"

现状:`_build_history_causal_mask`(474-516)返回**布尔** mask,`(q==K)&(k<=K)` 让 current
硬性 attend 历史。

改法:历史列从"硬允许"改成"加一个可学习偏置 `history_gate`":
- 对**任意 query → history-key** 的注意力 logits 加 `history_gate`(标量,**建议 per-block**)。
- `history_gate` 初值取强负(如 `-20`,或参数化为 `g` 经 `−softplus(−g)` 之类保证起点≈`-∞`);
  init 时历史列 ≈ `-inf` → 没有任何 token attend 历史 → **forward 严格退化为原版**。
- 训练让 `history_gate` 上升,历史逐步可被读取。

实现位置:`build_video_to_video_mask`(547-560)在 `num_history_frames>0` 分支产出的 mask,
由"布尔"改为"加性 float bias"(allowed=0/`history_gate`,disallowed=`-inf`),传入自注意力。
> 注:`future→history` 本就不允许(mask `q>K→k>=K`),保持 `-inf` 不变;只门控 `current/history → history` 这些原本 allowed 的历史列。

### 2.2 `trainer.py`:冻结 base,只训记忆相关参数

现状:prepend 路线(`vae_memory.enabled=false`)走默认分支(362-369)**全 DiT + proprio 全解冻**。

改法:加一个 prepend-freeze 分支(开关 `vae_memory.freeze_base=true`):
```
model.requires_grad_(False)          # 冻结 video expert / action expert / proprio 全部 base 权重
# 仅以下可训(均为零初始化 → init no-op):
#   - history_gate(2.1 的门控,每个 block 一个标量)
#   - 历史输入接口:video_expert.patch_embedding(让 DiT 读懂历史 latent 帧)
#   - [Tier-2] 零初始化 LoRA(见 §3)
```
保持 `model.eval()`(VAE/Dropout 关闭、编码确定),仅打开上述子集的 `requires_grad`。

---

## 3. 可训练容量:两档(先 Tier-1,弱了再上 Tier-2)

纯门控 + frozen heads 可能太弱(frozen 注意力没学过历史 → 类似 stage1 的 45.64)。分档:

- **Tier-1(最小、最安全)**:可训 = `history_gate` + `video.patch_embedding`(2 张量)。
  改动最小;若有增益,clean 被结构性保证。先验证不可能退化,再看上不上限。
- **Tier-2(若 Tier-1 增益不足)**:再加**零初始化 LoRA**(rank 8–16)到 video expert 自注意力
  Q/K/V(B 矩阵置 0 → init no-op)。给冻结 base 一点"学会读历史"的容量,**base 权重不动、
  init 仍严格 == base**。

> 关键:**Tier-1/2 的所有新增项都零初始化** → §1 不变量恒成立。

---

## 4. 配置开关(`configs/model/fastwam.yaml` 的 `vae_memory` 段)

```yaml
vae_memory:
  enabled: false
  dit_prepend: true
  freeze_base: true        # 新增:冻结 base,只训记忆参数
  history_gate: true       # 新增:启用零初始化门控
  mem_lora_rank: 0         # 新增:0=Tier-1;>0=Tier-2(如 8/16)
```
`freeze_base/history_gate/mem_lora_rank` 缺省与旧行为兼容(false/false/0)。

---

## 5. 训练配方

- **起点**:仍冷启动 base ckpt `libero_uncond_2cam224.pt`,但 base **冻结**。
- **lr**:base 冻结后可给记忆参数较大 lr(如 `1e-4`);warmup 同旧。
- **数据/loss**:`lambda_video=lambda_action=1.0` **原样保留**(video loss 是 FastWAM 原生、有益,
  已排除为病因);历史仍 H5(4n+1)。
- **可选叠加 Route B**:训练期对历史帧做 camera/noise/bg 增强 + 历史 dropout(治"扰动历史脆"),
  作为后续增量,先把 A 的不变量跑通。

---

## 6. 验证(端到端,先证不变量再看增益)

1. **零步不变量验证(必须先过)**:加载未训练的 step0(门控关、LoRA=0),在 **200 pilot** 上
   - 记忆 ON → 必须 == base **50.5**;
   - 记忆 OFF → 必须 == base **50.5**;
   - 标准 LIBERO clean → 必须 == **95.9**。
   若任一 ≠ base,**门控/冻结没做到 no-op,先停下修**(这是整套方案的地基)。
2. **逐档 eval**:每 1000 步同 200 pilot,**clean 必须全程 ≈95.9 不掉**(Route A 的核心承诺);
   看 perturbed Total 是否爬过 base 50.5。
3. **对照**:与 stage2-v1(18.0 / clean 81)和 base(50.5 / 95.9)三方比。

---

## 7. 与论文/旧路线的关系

- 本质是把 **stage1 成功的那半**(冻结 base + 零初始化门控 → clean 稳 95.9)搬到 prepend 路线上,
  补掉 stage2 丢掉的"保护底座"。
- 仍非完整 MEM:MEM 的 encoder 压缩 + drop past tokens(`MEM-stage2-Idea.md:152` 已知但 v1 未做)
  是更彻底的下一步;Route A 先用最小代价**结构性消除退化**,再谈增益与 encoder 化。

---

## 8. 改动文件清单(实施时)

| 文件 | 改动 |
|---|---|
| `src/fastwam/models/wan22/wan_video_dit.py` | 历史 mask 布尔→加性 bias;新增 per-block `history_gate`(零初始化);[Tier-2] 自注意力零初始化 LoRA |
| `src/fastwam/trainer.py` | 新增 `freeze_base` 分支:冻 base,只开 `history_gate`/`patch_embedding`/LoRA |
| `src/fastwam/models/wan22/fastwam.py` | 透传 `history_gate`/`freeze_base`/`mem_lora_rank` 到 DiT 构造;`infer_action`/`training_loss` 路径不变 |
| `src/fastwam/runtime.py` | 解析新增 `vae_memory.*` 开关 |
| `configs/model/fastwam.yaml` | 新增 3 个开关(缺省兼容旧行为) |
| `scripts/train_mem_routeA.sh`(新) | 训练脚本:`freeze_base=true history_gate=true` |
