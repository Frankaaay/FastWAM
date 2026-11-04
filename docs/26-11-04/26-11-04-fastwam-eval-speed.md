# FastWAM 单卡 Clean / Random 评估优化

- 范围：用户授权独立分支、诊断脚本和记录、commit/push 到个人 fork，不合并；阿里云单张闲时 GPU（priority 4，ForceQuotaOverSold）。
- 分支 `frank/fastwam-eval-speed`，原始基线 `f6ca41b828a835fc0b6b535e0d9d16d4b987e499`。原工作区和正式评估保持原样。
- RAM Frank `205455090823393499`，workspace `350607`，quota `quotan2i9e2t9tqr`；CPU 管理 DSW `dsw-1a0plk4ipgycdyvy40` 恢复后 Running，SSH 当前 47.107.184.36:1016，命令已到远端。
- 模型从既有 mono clean manifest 读取，训练目录 `/mnt/workdir/frank/runs/stage3-robotwin-mono-repair-20261030/train-fastwam-mono-left`，checkpoint `checkpoints/weights/step_010725.pt`。
- 不改变模型、推理 10 steps / CFG 1 / horizon 32、冻结种子与指令、原生步数上限、Random 光照或观测时序。
- 初步代码事实：已有动作队列和提示词缓存。仿真控制每个物理步内多个 setter 重复调用两臂被动力计算；是否重要及可优化须通过 profiling 和逐帧精确验证。
- 诊断入口 `tests/h800_fastwam_eval_speed.py`，复用冻结 evaluator，记录完整观测哈希、动作、双臂 qpos/qvel、环境 RNG，并输出 cProfile；profile 耗时不当作最终加速比。
- 样本：Clean click_bell 0、Random click_bell 0 和 19；后续长轨迹 Clean open_microwave 99、Random open_microwave 1。
- 当前尚未提交 GPU 作业，尚无本轮速度提升证据。
