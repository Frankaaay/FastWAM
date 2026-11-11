# 旧Stage3-u8000 midtrain加载准入

用户要求在PPU1.5排一个新的StereoTok双目midtrain，只将冻结VAE换成之前旧Stage3-u8000，其余沿用已完成clean Stage3路线；另继续查看PPU1.1 smoke。本记录对应必要的加载适配，不代表正式任务已提交。

## 基线与范围

FastWAM分支frank/fastwam-stereotok-robotwin，HEAD及origin均491e518fac6e04646ecfbd464027dd575ed99bde，fetch origin成功。原有fastwam.py、test_robotwin_stereotok.py、旧文档和未跟踪资产保留。本任务仅改checkpoints.py、stereotok_vae.py、已有test_stereotok_stage3_loading.py及本记录；未commit/push。

旧权重历史来源 `/mnt/workdir/frank/runs/stage3-u8000-wan-eval-20261028/input/stage3-u8000.ckpt`（深圳CPFS），13,799,251,677bytes，SHA256 `5388b175591e0045f06b52fb0cfa5f769f1c03c79086c0d71661808e6b8df5e1`。此前RoboTwin四项Stage3运行使用此权重；与本次clean Stage3导出 `4c024b1d...` 区分。PPU15只读定位作业dlc1qzwxe3esows8 Succeeded：已知4个候选路径均不存在，from-sz仅datasets，不能声称权重已就绪。深圳0GPU身份审计dlcy54k5n7h328su已提交，尚待实际回执。

## 局部改动

复用已发布734fe840中的训练态加载规则，但只准入上述精确SHA；核验format、architecture、stage3和generator_updates8000。仅取vae前缀，排除原RGB路径不调用的semantic_head和latent_geometry_head，保留所有推理张量并strict=True加载。clean Stage3与原Stage2导出准入保持不变。binding对旧权重如实记录training格式/stage/update/hash，不伪标为clean版或Stage2源码。

未改encode/decode、数据、训练器、优化器或调度器。新midtrain应仍由Alpha154000初始化非codec参数，使用原双目DROID+ABC 3,206,442窗口、global128、32卡batch2/accum2、50,101microsteps，独立输出与W&B run，不从已完成Wan或clean midtrain权重续训。

## 验证与剩余门禁

本地命令 `C:/Project/StereoTok/.venv/Scripts/python.exe C:/Project/FastWAM/tests/test_stereotok_stage3_loading.py`：2项CPU合同测试通过。覆盖clean metric head严格重载、旧Stage3推理权重逐tensor一致/冻结、未知SHA拒绝、错误update拒绝和缺失推理tensor严格失败。测试使用小型模块替代，尚非实际13.8GB权重/PPU前向证据。具名文件diff检查无问题。

当前任务没有明确commit/push授权，遵守仓库发布规则，不能将本地未提交生产代码嵌入DLC绕过发布。正式排队仍需发布这个最小补丁、旧权重迁移与严格加载验收。PPU1.1 smoke仍是原clean版本独立任务dlc1p1yd57spnmsa，未修改其运行来源或配置。

深圳只读作业 `dlcy54k5n7h328su` 最终Succeeded，重新完整计算13,799,251,677-byte源文件SHA256与5388b175...一致，实际format=stereotok-wan-training-v1、architecture=stereo-v3、stage=3、generator_updates=8000；排除原训练readout后769个推理tensor共4,605,565,724bytes。源文件身份已实时确认。尚未向PPU15迁移，也未创建依赖未发布代码的正式作业。比较734fe840与491e518的student/codec：student仅新增MetricHead类，encode/decode路径未变，旧geometry不启用该head；补丁沿用旧RGB张量选择语义。下一步待明确发布许可后，迁移已核实旧权重、验证完整加载及Alpha非codec初始化，然后以独立输出/run排PPU15固定32卡。
