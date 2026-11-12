# C2R 显式 offline 模式校验

用户已授权 W&B offline，并要求 FastWAM RoboTwin Stage2 u15853 mono/stereo 各8卡L20Y固定资源训练，沿用原配方。当前入口硬编码要求 online，会拒绝正确初始化的 offline run。

当前分支 frank/c2r-stereotok-2461，HEAD及 fetch 后 origin 均 f6ca41b828a835fc0b6b535e0d9d16d4b987e499。原有修改仅两份10月13日文档，保留；未触碰原有 .worktree/。

仅修改 scripts/c2r/run_training.py：实际模式须等于 cfg.wandb.mode，且只能为 online 或 offline；无 run、disabled、模式不符均拒绝。默认配置仍为 online，不改变训练参数或损失，不静默降级。

验证：AST解析、git diff --check 通过；从实际AST提取检查块验证6例，online/online和offline/offline通过，交叉模式、disabled、无run拒绝。未做GPU训练验证。

尚未 commit/push。正式训练要求自己的 clean clone 与已推送精确SHA；等待用户授权发布这一小补丁，之后准备显式offline配置与新盘启动请求并提交两组固定8卡。当前没有新FastWAM Job ID，不声称预约成功。

用户随后明确同意提交、推送补丁并继续两组固定8卡训练预约。
