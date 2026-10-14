# H100 RoboTwin 2500 六目 RGB 与原生 GT

- 授权：沿 H800 最终 2500 条、50任务各50条合同，在H100最多8卡并行；允许最小修改、测试、commit/push及个人持久输出。320x240、15Hz、基线6/2/2cm、32SPP/pathdepth8/OIDN不变。
- 初始本地分支 frank/fastwam-stereotok-robotwin，HEAD cc95c0482671a0fec2553ea17f299e22b618b44b；fetch成功，与origin一致。原有untracked保留。
- H100 Frank，xn01-gpu1-0058实核8卡空闲。本人clean FastWAM-stereotok-replay从855907a经fetch/pull --ff-only同步cc95c04。没有操作其他作业。
- H800只读实核最终ready.json和reader-accepted.json：2500 episodes，551534 frames，549034 windows，manifest SHA256 ea55afb8f616b860633d97185fc2eb390d82bbbfdb458bfeefa4cc2fb22ee90e，39替代示范已完成。
- 旧H100渲染13646已保存140帧，但后处理因ffmpeg缺失失败，非渲染成功验收。现有SAPIEN3.0.0b1/render-extras及本人train runtime可复用，首先通过Slurm验证明确分配设备上的RT与Position/Segmentation缓冲。
- 尚未启动全量；计划动态逐条领取、长轨迹优先，避免固定分片长尾。GT必须六目同帧数值标签，不能用旧三目深度或着色分割图冒充。
- RT探针16078在4秒内退出1：Position含背景非有限值，原探针错误要求所有像素有效。定向修正为要求可见几何存在有限正Z，背景无效值由mask表达；Color/Segmentation仍要求全部有限。未改渲染参数。
- 修正后Job16081在3秒完成，batch和srun退出0。明确cuda:0设备RT32SPP/path8/OIDN，Color/Segmentation全部有限，Position有限元素比例0.8594；存在正深度。
- 最小实现scripts/h100_robotwin_gt.py：只读导出H800冻结manifest的轨迹/种子/场景/qpos/指令；H100同轨迹回放，逐帧qpos atol1e-5。流式保存六目无损libx264/yuv444p/15Hz视频与压缩数值GT，避免逐帧pickle缓存和二次merge。动态逐条flock领取、长轨迹优先、单条30分钟有界时限，任一失败停止并保留现场。
- 六目GT：float32米制Z、有效mask、float32 fx*b/Z视差、原始uint32 mesh/actor ID；遮挡是nearest-pixel对目深度检验，容差max(2mm,0.5%Z)，视野外单独标记，不冒充精确光线级可见性。保持RGB渲染与相机合同。
- _base_task.py仅增加显式render_device scene创建，避免SAPIEN旧wrapper忽略选卡。其他调用保留原行为。
- 静态检查发现_init_task_env_原先无参数调用setup_scene；同步传递render_device，确保选卡实际到达RenderSystem。两个本地GT边界测试通过。每条完成后全量解码六路MP4并核帧数，2500条/50任务各50条/551534帧均齐全才发布ready.json。
- 每次失败保留独立attempt目录；修复后重新提交只跳过accepted条目，绝不因失败目录存在而漏样本，也不覆盖旧尝试。H800导出经只读stdin读取已有文件，不在H800写入或申请新资源。
- H800导出2500条成功，273688767 bytes，SHA256 1eff4cc50f0c28828c42bed6bf59b016e2d20599d10a71752bae26aef8c62bbb；仅传轨迹/qpos/场景/指令，未复制旧视频。原始资产归档精确中央索引：Aloha 779877448 bytes、objects 4636111726 bytes。每个renderer要求torch只见一张卡，记录SAPIEN实际设备名称/PCI/CUDA_VISIBLE_DEVICES。
