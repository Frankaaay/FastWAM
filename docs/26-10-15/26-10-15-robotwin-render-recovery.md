# RoboTwin H100 渲染中断恢复

- 现场：Job16110在2273 move_stapler_pad超时后退出，单条exit124触发srun kill-on-bad-exit。部分HDF5和7路视频存在，说明已进入回放，不能仅凭没有FRAMES30断言初始化卡死。
- 保留全部已验收和partial。新增120秒重复调用栈及setup/replay阶段日志，复用已有基准入口单独重放2273；不先扩大超时或跳过样本。
- 本地分支frank/fastwam-stereotok-robotwin，9e5f82c，fetch后与origin一致；保留原有untracked。目标脚本/相机自5510750没有其他改动。
