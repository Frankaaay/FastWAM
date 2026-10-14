# H100 RoboTwin 2500 六目 RGB 与原生 GT

- 授权：沿 H800 最终 2500 条、50任务各50条合同，在H100最多8卡并行；允许最小修改、测试、commit/push及个人持久输出。320x240、15Hz、基线6/2/2cm、32SPP/pathdepth8/OIDN不变。
- 初始本地分支 frank/fastwam-stereotok-robotwin，HEAD cc95c0482671a0fec2553ea17f299e22b618b44b；fetch成功，与origin一致。原有untracked保留。
- H100 Frank，xn01-gpu1-0058实核8卡空闲。本人clean FastWAM-stereotok-replay从855907a经fetch/pull --ff-only同步cc95c04。没有操作其他作业。
- H800只读实核最终ready.json和reader-accepted.json：2500 episodes，551534 frames，549034 windows，manifest SHA256 ea55afb8f616b860633d97185fc2eb390d82bbbfdb458bfeefa4cc2fb22ee90e，39替代示范已完成。
- 旧H100渲染13646已保存140帧，但后处理因ffmpeg缺失失败，非渲染成功验收。现有SAPIEN3.0.0b1/render-extras及本人train runtime可复用，首先通过Slurm验证明确分配设备上的RT与Position/Segmentation缓冲。
- 尚未启动全量；计划动态逐条领取、长轨迹优先，避免固定分片长尾。GT必须六目同帧数值标签，不能用旧三目深度或着色分割图冒充。
- RT探针16078在4秒内退出1：Position含背景非有限值，原探针错误要求所有像素有效。定向修正为要求可见几何存在有限正Z，背景无效值由mask表达；Color/Segmentation仍要求全部有限。未改渲染参数。
