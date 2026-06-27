# Mem-VAE 版本:对照实验 + 微调计划

> 分支 `feat/mem-vae`。本版给 FastWAM 加了 **VAE 短期记忆**(temporal memory):
> 冻结 base VAE / DiT / MoT,只 warm-start 并训练 VAE memory 的 4 个 temporal
> 参数张量(temporal attn / pos / proj)。checkpoint = `step_021700.pt`。

---

## 0. 当前 LIBERO 复现成绩(Table 2)

ckpt `runs/mem_temporal_libero/checkpoints/weights/step_021700.pt`,50 trials/task,
run dir `evaluate_results/libero/libero_uncond_2cam224_1e-4/20260617_100251`:

| Suite | 成功/总数 | Ours (mem) | Paper Fast-WAM | Δ |
|---|---|---|---|---|
| Spatial | 482/500 | 96.4 | 98.2 | -1.8 |
| Object  | 498/500 | 99.6 | 100.0 | -0.4 |
| Goal    | 477/500 | 95.4 | 97.0 | -1.6 |
| Long    | 461/500 | 92.2 | 95.2 | **-3.0** |
| **Avg** | 1918/2000 | **95.9** | 97.6 | -1.7 |

**加了记忆反而比原版低 ~1.7 个点,Long(长时序)掉得最多。** Long 恰恰是短期
记忆最该发力的场景,却伤得最重 → 强烈提示当前 memory 是"接口错配 / 净负",
而不是"记忆本身没价值"。

---

## 1. 诊断:大概率是"冻结的 DiT 没见过带记忆的 latent"

- 原版 DiT 是在**原始 VAE latent 分布**上训出来的。插入 temporal memory 改变了
  latent 表示,但 DiT 一个梯度都没吃 → 它看到的是轻微 OOD 的输入。
- temporal_proj 即便从 spatial proj warm-start,一旦偏离,**冻死的 DiT 无法补偿**。
- 证据 A:Long suite 掉最多(见上)。
- 证据 B:loss 已收敛(见 §3),不是训得不够 → 天花板是容量/接口,不是步数。

---

## 2. 先做对照实验(便宜,决定后续是否烧卡)

> 这两个几乎零成本,但能直接定责:差距里多少是 memory、多少是环境/seed。
> **动 DiT 之前必须先跑完这两个。**

### 2.1 同一 ckpt,eval 时关掉 memory
```bash
# 在 eval_full.sh 基础上加 override
MUJOCO_GL=egl bash scripts/eval_full.sh   # 改 manager 传参 model.vae_memory.enabled=false
```
- 若关掉 memory → 回到 ~97.6:**铁证 memory 在拖后腿**。
- 若关掉仍 ~95.9:差距来自 harness/env/seed,不是 memory。

### 2.2 原版 Fast-WAM ckpt(无 memory)在当前 harness 上 eval
- 确认 97.6 在我们这套环境能复现,排除环境差异。
- 需要原版 release ckpt(`checkpoints/fastwam_release/` 下确认有无)。

---

## 3. Loss 曲线(回答"是否欠拟合")

训练**没有** tensorboard/wandb,loss 只在 train log 里。从两段 log 抽出:

| 阶段 | loss 轨迹(时间序) |
|---|---|
| `train_20260613_225204.log`(step 0→~8000) | 0.42 → 0.43 → **0.17 → 0.14 → 0.12** → 0.10 → 0.09 |
| `train_20260614_221018.log`(resume 8000→21700) | 0.115 → 0.112 → 0.107 → 0.10 → 0.10 → **0.11**(min 0.077) |

**结论:loss 在 ~step 4000 就基本压平,后面 ~17000 步在 0.09–0.11 震荡,没有进一步下降。**
→ **不是欠拟合(步数意义上)。再训也压不动。** 天花板来自冻结 DiT 的容量/接口,
这反而**加强了"该微调 DiT"的判断**,而不是"再多训几个 epoch"。

> 注:这是 flow-matching 去噪 loss,绝对值跟任务成功率不直接挂钩;关键信号是
> **17k 步几乎不降的平台**。

---

## 4. 关于 step_021700 的由来(回答"是不是数据跑完就这么多步")

- config:`num_epochs: 10`,`max_steps: null`,`batch_size: 16`。
- train log 末尾:`>> max_steps reached step=21700`,训练进程已退出(pid 已死)。
- 即 **21700 = 跑满计划的 10 个 epoch**(trainer 由 num_epochs 换算出的 max_steps),
  数据被完整看了 10 遍,**不是中途断点**。
- 所以"没跑完 epoch 导致欠拟合"这条**被排除**。结合 §3 的 loss 平台,
  当前 ckpt 是**已收敛**状态,不是训得不够。

> ⚠️ 注意:训练已经停了(不是之前以为的"已 resume 在跑")。要继续训需重新拉起。

---

## 5. 若确认是 memory/接口问题 → 微调按 ROI 排序

| 方案 | 改动 | 成本 | 风险 | 备注 |
|---|---|---|---|---|
| **A. 解冻 DiT 输入接口** | patch-embed / 第一层 proj(可加 action expert) | 低 | 低 | **首选**,让 DiT 学会消化带记忆的 latent |
| **B. DiT 上 LoRA** | DiT attn/mlp LoRA,小 LR | 中 | 低 | 保留预训练能力同时给适应空间,性价比高 |
| **C. 部分/全量微调 DiT** | 解冻 DiT,小 LR + 短 schedule | 高 | catastrophic forgetting | 盯曲线,容量最大 |

顺序:先 §2 对照 → 若是 memory 锅 → A → 不够再 B → 仍不够再 C。

---

## 6. LIBERO-Plus 鲁棒性测试(对齐 robustness paper)

目标 paper:**"Do World Action Models Generalize Better than VLAs? A Robustness Study"**
(arXiv 2603.22078)。其 Table 4 报了 **Fast-WAM(原版,无 memory)** 的 LIBERO-Plus:

| Model | Original | Camera | Robot | Lang. | Light | BG | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| **Fast-WAM(paper, 原版)** | 97.6 | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | **51.5** |
| ABot-M0(best) | 98.6 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| Cosmos-Policy(WAM) | 98.5 | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 82.2 |
| VLA-JEPA(VLA+WM) | 97.2 | 64.2 | 67.7 | 88.1 | 91.8 | 93.4 | 65.8 | 83.9 | 77.9 |

**核心卖点验证**:原版 Fast-WAM 鲁棒性极差(Camera 16.4 / Noise 37.7 / Total 51.5)。
**若短期记忆能提升鲁棒性,这里就是直接证据** → 把我们 mem 版同协议跑一遍,
直接和上面 Fast-WAM 行比 Total 和各 factor。

### LIBERO-Plus 协议(对齐 paper,便于直接比较)

- benchmark:`github.com/sylvestf/LIBERO-plus`(arXiv 2510.13626)。
- 规模:**10,030 个 case**,7 个扰动因子(Camera/Robot/Lang/Light/BG/Noise/Layout)
  + 21 个 component,5 个难度等级 L1–L5。
- **每个 case 只 rollout 1 次**(`num_trials=1`)—— case 数太多,这也是 paper 的做法。
- Total = 全部 10,030 个 case 的整体成功率(各 factor 的 column 是该 factor 子集的成功率)。
- task↔factor↔难度映射:`.libero/libero/benchmark/task_classification.json`。

### 落地步骤(air-gapped,需经 jump host 暂存)

1. jump host 上 clone `sylvestf/LIBERO-plus`,下载 assets.zip。
2. 解压 assets 到 `LIBERO-plus/libero/libero/assets/`,经 NFS 同步到 H200 节点。
3. 更新 `~/.libero/config.yaml` 指向 LIBERO-plus。
4. eval 设 `num_trials=1`,遍历全部 case;按 factor 汇总出上表同款 8 列 + Total。
5. 和 paper 的 Fast-WAM 行逐列对比,看 memory 是否提升鲁棒性(尤其 Camera/Noise)。

---

## 附:参考链接
- Robustness paper: arXiv 2603.22078
- LIBERO-Plus benchmark: github.com/sylvestf/LIBERO-plus,arXiv 2510.13626
