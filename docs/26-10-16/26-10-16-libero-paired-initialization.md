# LIBERO paired initialization 修复

最终训练验收发现同seed42但Wan/StereoTok随机动作头、proprio头指纹不同：不同冻结VAE构造消耗CPU RNG。实际两次完整模型初始化证据在H800 twelve-ready-u15853/fastwam-initialization-audit.json，equal=false。

仅在helpers/loader.py冻结VAE构造周围用torch.random.fork_rng(devices=[])恢复CPU RNG，保证后续trainable heads一致；不改加载权重、数据、训练预算或接口。C2R独立分支已有显式共享初始化，不修改C2R。发布前diff检查；发布后复用实际factory做两组完整初始化指纹复核。

发布后真实factory检查三个分支指纹全部相同；使用全量LIBERO reader及全量统计再次完整前后向通过，Wan loss0.5904445052、StereoTok loss0.5749142766，无optimizer更新。证据为H800 twelve-ready-u15853/fastwam-initialization-audit-fixed.json与fastwam-initialization-fixed.log。

用户确认保留原版历史权重，完整恢复状态仅最新一份。LIBERO保存频率2000不变，新状态保存并写trainer_state后，逐文件清理旧step恢复负载，保留权重和目录。尚待远端定向验收。
最终验证通过：c0fc784与C2R e6de34e生产保存方法AST一致；两次保存历史权重保留、旧恢复状态移除、新保存失败保留旧恢复状态。已更新正式启动SHA，未启动正式训练。
用户再次明确按原repo保存策略：撤回上一轮给FastWAM新增的旧恢复状态清理；原版历史模型权重和完整state均保留，LIBERO每2000 optimizer steps及最终保存。此次仅删除6行清理逻辑，恢复原save_checkpoint，语法和diff检查通过。
