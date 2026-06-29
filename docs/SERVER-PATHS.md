# 服务器路径总结(NGAD H200)

> 实测于 **node-1**(`ssh h200-qinghua-1`),2026-06-28 只读核实。两节点 `/data`、`runs/`、
> `evaluate_results/` **各自独立不互通**;默认主工作节点 = node-1。基础设施/SSH 细节见
> memory `ngad-server-guide` / `ssh-server-playbook`。

## 0. 根

| 名称 | 路径 | 说明 |
|---|---|---|
| 仓库 ROOT | `/home/frank/projects/FastWAM` | = `~/projects/FastWAM`。`/data/home/frank` 与 `/home/frank` 等价(同一份) |
| conda env | `fastwam` | `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam` |

> ⚠️ 所有 `configs/data/*.yaml` 里的 `./data/...` 展开为 `~/projects/FastWAM/data/...`。
> **2026-06-28 起,大数据集物理搬到 `/data/shared/offline/datasets/`,仓库内 `data/<集>` 改为软链**
> 指过去(见 §1 / §5);config 路径不变、训练/eval 照常。`du` 4.5G 的 LIBERO 等都已外移。

## 1. 数据集(`$ROOT/data/`)

| 数据 | 绝对路径 | 状态 |
|---|---|---|
> **物理位置**:大数据集都在 `/data/shared/offline/datasets/<集>/`,仓库 `data/<集>` 是软链。
> 下面写仓库相对路径,实际经软链落到 offline。LIBERO=`libero_mujoco3.3.2/`、RobotWin=`robotwin2.0/`。
> RobotWin 压缩包(8 分卷 tar.gz,74G)另存 `/data/shared/offline/archives/datasets/robotwin2.0-fastwam/`。

| LIBERO spatial | `~/projects/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot` | ✅(软链) |
| LIBERO object | `~/projects/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot` | ✅(软链) |
| LIBERO goal | `~/projects/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot` | ✅(软链) |
| LIBERO 10/long | `~/projects/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot` | ✅(软链) |
| 文本嵌入缓存(libero) | `~/projects/FastWAM/data/text_embeds_cache/libero` | ✅ |
| **RoboTwin 2.0**(27,500 ep / 6M frames) | `~/projects/FastWAM/data/robotwin2.0/robotwin2.0`(stats: `…/robotwin2.0/dataset_stats.json`) | ✅(软链)**2026-06-28 从 node-2 直传解压**;LeRobot v2.1,videos 74G |
| 文本嵌入缓存(robotwin) | `~/projects/FastWAM/data/text_embeds_cache/robotwin` | ❌ **仍需 precompute**(数据已就位,文本嵌入未生成→跑 robotwin 训练前必须先生成) |
| **RMBench demonstrations** | `~/projects/RMBench/data/<task>/demo_clean/` **->** `/data/shared/offline/datasets/RMBench/data/<task>/demo_clean/` | ✅ 已下载并迁入 shared;repo 内 `data/<task>` 为软链 |
| **RMBench assets** | `~/projects/RMBench/assets/{embodiments,objects}` **->** `/data/shared/offline/datasets/RMBench/assets/{embodiments,objects}` | ✅ 已下载并迁入 shared;repo 根目录 `embodiments/objects` 也软链到同一位置 |

配置出处:[configs/data/libero_2cam.yaml](../configs/data/libero_2cam.yaml)、
[configs/data/robotwin.yaml](../configs/data/robotwin.yaml)。

## 2. 权重 / checkpoint

| 名称 | 路径 |
|---|---|
| base ckpt(mem-off 起点) | `~/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` |
| base dataset_stats | `~/projects/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json` |
| DiffSynth 模型根(离线) | `DIFFSYNTH_MODEL_BASE_PATH=$ROOT/checkpoints`(配 `DIFFSYNTH_SKIP_DOWNLOAD=true`) |
| stage1-v1 训练 ckpt | `~/projects/FastWAM/runs/mem_temporal_libero/checkpoints/weights/step_021700.pt` |

## 3. 训练输出(`$ROOT/runs/`)

实测现存目录(**仍是改名前的旧命名**,服务器尚未 pull 分支重构):
`runs/mem_stage2_v1/`、`runs/mem_stage2_v1_smoke/`、`runs/mem_stage2_v2_smoke/`、
`runs/stage1-v1|v2|v3/`、`runs/mem_temporal_libero/`。

> 新命名(本地已统一):stage3 训练 → `runs/mem_stage3/`(`_smoke` 同)。服务器 pull 新脚本后,
> **新跑的** run 才会落到 `runs/mem_stage3/`;旧目录名不会自动改。

## 4. 评测输出(`$ROOT/evaluate_results/`)

| BENCH | 路径 |
|---|---|
| 标准 LIBERO | `~/projects/FastWAM/evaluate_results/libero/` |
| LIBERO-plus(扰动) | `~/projects/FastWAM/evaluate_results/libero_plus/` |
| 标准 LIBERO(plus 环境内跑) | `~/projects/FastWAM/evaluate_results/libero_std/` |
| 默认 OUT 模板 | `./evaluate_results/$BENCH/libero_uncond_2cam224_1e-4/<时间戳>/` |

每次 eval 写 `$OUT/progress.log`(done/total+ETA)与 per-case `*_results.json`;汇总:
`python experiments/libero/summarize_libero_plus.py --output_dir "$OUT"`。

## 5. LIBERO 仿真 / LIBERO-plus(两套仓库并存)

**plus 与标准是两个独立仓库**,eval 通过切 libero config 跳转,代码不变:

| 口径 | 仓库根 | config 文件 | benchmark_root |
|---|---|---|---|
| **LIBERO-plus**(扰动,默认) | `/data/home/frank/projects/LIBERO-plus` **(软链)→** `/data/shared/offline/datasets/LIBERO-plus` | `~/.libero/config.yaml` | `…/LIBERO-plus/libero/libero` |
| **标准 LIBERO**(无扰动,base 95.9) | `/home/frank/projects/LIBERO`(=`/data/home/frank/…`,**未移动**) | `~/.libero_orig/config.yaml`(`LIBERO_CONFIG_PATH`) | `…/LIBERO/libero/libero` |

> ⚠️ LIBERO-plus 是 **`pip install -e` 可编辑安装**的包(`libero.egg-link` 指 `/data/home/frank/projects/LIBERO-plus`)。
> 2026-06-28 物理搬到 offline/datasets,旧路径留软链 → egg-link、`~/.libero/config.yaml`、PYTHONPATH **全部不动**仍解析。
> 切勿删旧软链,否则 `import libero` 与 plus eval 全断。

LIBERO-plus 关键资产(均在 `…/LIBERO-plus/libero/libero/` 下):

| 内容 | 路径 |
|---|---|
| 7-factor 扰动分类表 | `…/LIBERO-plus/libero/libero/benchmark/task_classification.json` |
| bddl / init_states / assets | `…/LIBERO-plus/libero/libero/{bddl_files, init_files, assets}` |
| Noise 因子依赖(ImageMagick,离线) | `/data/shared/offline/noise_deps/imagemagick`(`MAGICK_HOME`) |

> `eval_libero_plus.sh` 读 `os.path.dirname(libero.__file__)/benchmark/task_classification.json`
> 定位扰动集([eval_libero_plus.sh:138](../scripts/eval_libero_plus.sh#L138))——默认 `~/.libero` 把
> `libero` 包解析到 LIBERO-plus 仓库。
> LIBERO-plus 一律 `INCLUDE_NOISE=1`(全 10030),否则与 mem-off 49.83 口径不可比。
> 标准/扰动切换细节见 memory `libero-std-vs-plus-switch`。

## 6. 离线 staging(跨互联网内容必经)

| 区域 | jump host 路径 | H200 路径 |
|---|---|---|
| 离线总目录 | `/data-214-30-239-40/shared/offline`(node-1) | `/data/shared/offline` |
| 子目录 | — | `wheels/ conda/ npm/ docker/ apt/ models/ datasets/ source/ noise_deps/` |

H200 不能联网;任何 github/HF/pip 公网内容先在 jump host 下载,经 NFS 落到对应节点 `/data/shared/offline`。

## 7. wandb(离线)

offline run 落在共享盘,jump host 跑 sync daemon 推云端。entity `yichx14-uc-irvine`、
project `fastwam-mem`。详见 memory `wandb-offline-jump-pipeline`。

## 8. Dataset / Benchmark inventory

> 这里区分两个概念:
> **dataset** = 训练/finetune 用的 demonstration / trajectory / HDF5 / LeRobot 数据;
> **benchmark** = 跑策略、reset 仿真环境、执行 action、统计 success 的评测协议与代码。
> 同一个项目常同时包含 assets、dataset、benchmark runner 和 policy adapter。

### 8.1 已在用 / 当前主线

| 名称 | 类型 | 当前用途 | 数据 / 代码路径 | 状态 |
|---|---|---|---|---|
| **LIBERO** | dataset + benchmark | FastWAM 主训练集与标准无扰动 eval;base 口径约 95.9 | 数据:`~/projects/FastWAM/data/libero_mujoco3.3.2/*_no_noops_lerobot`;仿真仓库:`/home/frank/projects/LIBERO`;结果:`~/projects/FastWAM/evaluate_results/libero/` / `libero_std/` | ✅ 已用 |
| **LIBERO-plus** | benchmark extension | 7-factor 扰动鲁棒性评测;MEM 当前主门禁;一律 `INCLUDE_NOISE=1` 全 10030 | 仓库软链:`/data/home/frank/projects/LIBERO-plus -> /data/shared/offline/datasets/LIBERO-plus`;结果:`~/projects/FastWAM/evaluate_results/libero_plus/` | ✅ 已用 |
| **RoboTwin 2.0** | dataset + benchmark | FastWAM 官方支持的第二个模拟 benchmark;适合训练/评估 RoboTwin policy | 本仓库 vendored eval:`~/projects/FastWAM/third_party/RoboTwin`;FastWAM adapter:`~/projects/FastWAM/experiments/robotwin/fastwam_policy`;数据 config 期望:`~/projects/FastWAM/data/robotwin2.0/robotwin2.0` | ⚠️ 代码在,数据 node-1 未 staging |
| **RMBench** | memory benchmark + dataset + assets | 准备用来测短程/任务相关 memory;比 EventVLA 更贴近当前 stage3 的 sliding visual memory | 仓库:`~/projects/RMBench`;env:`conda activate RMBench`;物理数据:`/data/shared/offline/datasets/RMBench`;repo 软链:`~/projects/RMBench/{data/<task>,assets/embodiments,assets/objects}` | ✅ 已下载/待 policy 适配 |

### 8.2 RMBench 目录语义

| 项 | 路径 | 含义 |
|---|---|---|
| RMBench repo | `~/projects/RMBench` | benchmark 代码、任务环境、policy 模板、下载脚本 |
| assets | `~/projects/RMBench/assets/{embodiments,objects}` | 仿真用机器人/物体/mesh/纹理/碰撞模型;没有它环境起不来 |
| data | `~/projects/RMBench/data/<task>/demo_clean/` | 训练/finetune demonstration;通常含 HDF5 trajectory、instructions 等 |
| policy adapters | `~/projects/RMBench/policy/<PolicyName>/` | benchmark 调用模型的包装层,定义如何 load ckpt、处理 observation、输出 action |
| FastWAM policy(待建) | `~/projects/RMBench/policy/FastWAM/` 或 symlink 到 FastWAM repo | 把 RMBench observation 转成 FastWAM `infer_action(...)` 输入;mem-on 还需接 history buffer |

RMBench README 里的 `Run Policies` 指的是可运行的策略接口/算法族,例如 `Mem-0`、`DP`、`ACT`、
`Pi0.5`、`X-VLA`。如果我们用 FastWAM 在 RMBench 数据上训练/finetune 出 ckpt,并提供
`policy/FastWAM/{deploy_policy.py,deploy_policy.yml,eval.sh}`,它同样就是一个 RMBench policy。
之后可比较:

| 对照 | 目的 |
|---|---|
| `FastWAM-mem-off on RMBench` | 同架构无 memory baseline |
| `FastWAM-stage3-mem on RMBench` | 测 stage3 短程视觉 memory 是否提升连续性/鲁棒性 |

### 8.3 未来可用 benchmark

| 名称 | 链接 | 主要测什么 | 对 FastWAM-MEM 的意义 | 当前状态 |
|---|---|---|---|---|
| **RoboMME** | https://robomme.github.io/ | memory-augmented manipulation;Counting / Permanence / Reference / Imitation 四类任务 | taxonomy 完整,适合后续诊断 memory 到底帮 temporal、spatial、object 还是 procedural;可作为 RMBench 之后的外部验证 | 未下载/未适配 |
| **EventVLA / RoboTwin-MeM** | https://ganlin-yang.github.io/EventVLA.github.io/ | event-driven visual evidence memory;存关键帧/keyframe evidence 做长程证据检索 | 更偏 long-horizon keyframe memory,不完全贴合当前 stage3 的“过去几帧让动作更连续”;适合作为未来加 keyframe/event memory 后的 benchmark | 未下载/未适配 |

### 8.4 推荐评测路线

1. **保 baseline**:标准 LIBERO + LIBERO-plus,确认 stage3 不明显低于 mem-off。
2. **测短程 memory**:RMBench 上训练/finetune `FastWAM-mem-off` 与 `FastWAM-stage3-mem`,同 benchmark 对比。
3. **扩展到 RoboTwin 2.0**:补齐 `data/robotwin2.0` 后跑官方 RoboTwin eval,验证通用模拟操作能力。
4. **外部 memory taxonomy**:RoboMME 子集验证 memory 类型;EventVLA/RoboTwin-MeM 等有 keyframe/event memory 后再上。

