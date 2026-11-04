# FastWAM 单卡 Clean / Random 评估优化

## 最终结论

Stage3-u8000 mono 最终 step 10725，阿里云单张 L20Y / Hopper 9.0 / 81559 MiB。通过冻结常驻模型的 Python GC 扫描集合，避免每场景 40 次 CUDA Graph 初始化 GC 重复遍历全部常驻对象；模型推理、规划、仿真和渲染未修改。优化已推送个人分支，未合并、未切换既有正式评估。

| 样本 | 动作数 | 基线 episode 秒 | 优化 episode 秒 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| Clean click_bell 0（最终实现复测） | 60 | 56.0996 | 26.8389 | 2.0902x |
| Random 静态 click_bell 0 | 54 | 60.1828 | 30.0218 | 2.0046x |
| Random 动态 click_bell 19 | 400 | 127.9063 | 98.1900 | 1.3026x |
| Clean open_microwave 99 | 1500 | 332.3975 | 303.3256 | 1.0958x |
| Random open_microwave 1 | 1500 | 329.6941 | 298.3967 | 1.1049x |

以上沿用原 evaluator 的 episode 计时，不含模型首次加载及外层 GC 作用域进入/退出。最终生产作用域另外记录了完整 wall：Clean 短例 56.1340 → 27.6679 秒；Clean 长例 332.4223 → 304.0617 秒；Random 长例 329.7236 → 299.1338 秒，包含作用域进入/退出和场景清理后仍稳定省约 28–30 秒。

5 个不同冻结样本共 3514 动作、113 次完整观测、111 次模型推理；完整模型输入/输出、观测哈希、动作、双臂 qpos/qvel、Python/NumPy 环境 RNG 精确一致，成功/失败结果一致。两条 1500 步样本均保持原生失败结局，没有缩短上限或提前结束。所有配对结束 CUDA allocated 都为 30,357,744,128 字节。

单个 episode 固定节省约 28 秒，按同配置 5000 局外推约省 39 GPU 小时；不是全量重跑结果，也不把短例 2x 外推到全部任务。未单独验证 Wan-RGB / 双目 checkpoint 的全量收益。

最终长测 `dlc12fbqizolonmn` 为 Succeeded；短测 `dlcdq7nrfgh67e6q` 为 Succeeded。所有本轮 GPU 作业均已结束（早期诊断一个主动停止、一个退出 139，见下）。CPU 管理实例确认无额外 Python/tmux/rsync/scp 工作进程后请求停止。产物均保留于 CPFS。

## 实验记录

- 范围：用户授权独立分支、诊断脚本和记录、commit/push 到个人 fork，不合并；阿里云单张闲时 GPU（priority 4，ForceQuotaOverSold）。
- 分支 `frank/fastwam-eval-speed`，原始基线 `f6ca41b828a835fc0b6b535e0d9d16d4b987e499`。原工作区和正式评估保持原样。
- RAM Frank `205455090823393499`，workspace `350607`，quota `quotan2i9e2t9tqr`；CPU 管理 DSW `dsw-1a0plk4ipgycdyvy40` 恢复后 Running，SSH 当前 47.107.184.36:1016，命令已到远端。
- 模型从既有 mono clean manifest 读取，训练目录 `/mnt/workdir/frank/runs/stage3-robotwin-mono-repair-20261030/train-fastwam-mono-left`，checkpoint `checkpoints/weights/step_010725.pt`。
- 不改变模型、推理 10 steps / CFG 1 / horizon 32、冻结种子与指令、原生步数上限、Random 光照或观测时序。
- 初步代码事实：已有动作队列和提示词缓存。仿真控制每个物理步内多个 setter 重复调用两臂被动力计算；是否重要及可优化须通过 profiling 和逐帧精确验证。
- 诊断入口 `tests/h800_fastwam_eval_speed.py`，复用冻结 evaluator，记录完整观测哈希、动作、双臂 qpos/qvel、环境 RNG，并输出 cProfile；profile 耗时不当作最终加速比。
- 样本：Clean click_bell 0、Random click_bell 0 和 19；后续长轨迹 Clean open_microwave 99、Random open_microwave 1。
- 首轮 profile 作业 `dlchbwrgrolc42p3`，代码 SHA `3489570d8c5aaad3424e2393407f0ca6057096de`，已 Running 且日志出现 SAPIEN 初始化。单 GPU / 12 CPU / 128 GiB / 32 GiB SHM，最长 180 分钟。
- 请求：本机 `C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/fastwam-eval-speed-20261104-profile01.json`。远端 clean clone `/mnt/workdir/frank/projects/fastwam-eval-speed-20261104`；复用既有资产和 task_config 符号链接，未更改源资产。
- 命令：既有 `fastwam-stereotok-eval-cu128-20261009/bin/python -u tests/h800_fastwam_eval_speed.py /mnt/workdir/frank/runs/fastwam-eval-speed-20261104/profile01`。日志 `/mnt/workdir/frank/runs/fastwam-eval-speed-20261104/profile01.log`，结果同名目录。尚无本轮速度提升证据。
- profile01 持续初始化且只有 SAPIEN 提示，没有 episode；为定位停滞，停止本次诊断，给实验脚本添加标准 logging 和每 120 秒栈转储，再用同样的单卡配置重试。未改变生产推理逻辑。
- profile02 `dlc1eclukr82122d`，SHA `c9027d5f03263d35274b38dc48945ecc43b73156`。模型组件加载 112.79 秒，栈定位 StereoTok checkpoint 读取/校验。Clean click_bell 60 动作成功；Random 静态 54 动作成功。profile 含首次编译和额外开销，不作为加速比分母。
- 观察到 Random 静态场景初始化中的 40 次 `gc.collect` 合计约 26.9 秒；推理仅 0.965 秒。先验证 `gc.freeze()` 将长期驻留模型对象移出重复 GC 遍历的候选方案，保留每次原生 GC 调用，新场景对象仍正常回收；配对结束 unfreeze。仅实验脚本实现，不修改冻结 RoboTwin 树。
- 配对新增模型输入/输出哈希、目标函数定时、CUDA 分配内存记录，恢复相同 Torch/环境 RNG，先做同进程 warmup；最终速度以无 cProfile 的 baseline/candidate 为准。
- profile02 前两例保存完整结果，第三例退出 139，尚不能确定根因；其 cProfile 部分调用计数也有异常。因此后续配对不启用 cProfile，使用显式函数计时，并仅在初始化期间定时转储，保留致命错误栈。失败样本不作为有效性能/一致性证据。
- paired01 `dlcdq7nrfgh67e6q` / SHA `ce15cc28f05db121bc1386b08dabb6d8a0c1b914`：Clean 60 动作 55.2627 → 27.3928 秒（2.0174x），完整输入/输出、观测、动作、qpos/qvel、环境 RNG 精确一致。40 次 GC 28.477 → 0.392 秒；被动力重复计算仅约 0.476 秒，因此不改仿真控制逻辑。
- Random 静态 54 动作初步配对也通过：60.1828 → 30.0218 秒。继续验证动态光照和长轨迹。
- 将同一候选实现为 evaluator 内 `fastwam_model_gc` 作用域，仅 FastWAM 的 rollout chunk 使用；开始 collect/freeze，结束或异常时 unfreeze。新场景循环引用仍可回收。GPU 配对直接调用此生产作用域，同时记录包含开始 GC 和结束清理的完整 wall 秒数；冻结 RoboTwin 树保持不变。
- paired01 全部通过且作业 Succeeded：Random 动态 400 动作、13 次观测，127.9063 → 98.1900 秒（1.30264x），全部轨迹与随机状态一致。Clean/Random 静态结束后的 CUDA 实际分配内存均为 30,357,744,128 字节，基线/候选完全相同。
- 本地 unittest：19 项，18 通过、1 个 POSIX 进程组测试因 Windows 跳过；`git diff --check` 通过。新增真实 GC 生命周期检查覆盖新场景循环引用回收、异常退出 unfreeze、常驻对象之后可回收。
- 最终实现 SHA `e9dbd1367b335ca72012949eb6c01aa6f72e41e2`；长测 `dlc12fbqizolonmn`，请求 `C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/fastwam-eval-speed-20261104-long01.json`，同单卡配置。日志 `/mnt/workdir/frank/runs/fastwam-eval-speed-20261104/long01.log`，结果 `long01/`；先短例预热和配对，再原生上限 Clean open_microwave 99 / Random open_microwave 1 配对。
- 远端 Linux 的 19 项 unittest 全部通过（包括 POSIX 进程组测试）。
- 最终作用域短例复测：episode 56.0996 → 26.8389 秒，包含 GC 进入/退出及清理的 wall 56.1340 → 27.6679 秒。全部精确检查通过。
- Clean 长轨迹 1500 动作、47 次模型观测，原生失败结果不变：episode 332.3975 → 303.3256 秒（1.09584x），wall 332.4223 → 304.0617 秒。完整输入/输出、观测、动作、状态、环境 RNG 全部一致。
- Random 长轨迹 1500 动作、47 次模型观测，原生失败结果不变：episode 329.6941 → 298.3967 秒（1.10489x），wall 329.7236 → 299.1338 秒；全部比较项通过，long01 最终 Succeeded。生产代码验证 SHA 为 `e9dbd1367b335ca72012949eb6c01aa6f72e41e2`，此后仅更新本文结果记录。

请求 SHA256（均位于上述本机 `h800-requests/`）：

| 请求后缀 | SHA256 |
| --- | --- |
| profile01 | `489254BEFC867D316C6CDC14F687E55BEE715945289E633A171241A1382B2174` |
| profile02 | `50BACCE232745BBB9CDFC26427B7E4B76BCBC9BC770B174632D71901E38E53C4` |
| paired01 | `1E33F52550ED4848A835E0CA7AB3698CEE3DCB75E5637C82A13AA7CF85B24D72` |
| long01 | `B2329D787014E5E6438F1FC5A7FE196DDA4378E11C3481E4230932C380DB51E6` |
