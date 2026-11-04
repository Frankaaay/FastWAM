# FastWAM 单卡 Clean / Random 评估优化

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
