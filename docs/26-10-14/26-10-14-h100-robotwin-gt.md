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

- H100输入包SHA256复核一致。准备Job16084 CPU/8CPU/64G，43秒完成，所有step 0:0。单卡两条回放Job16086（canonical0及补采16），8卡全量Job16088依赖afterok:16086；正式资源128CPU/768G/48h，动态8workers。日志 /gpfs/jiuquyun/home/Frank/logs/robotwin-gt-{JobID}.out；逐条日志在输出根logs/。首个实际renderer设备H10080GB、PCI0000:19:00.0、CUDA_VISIBLE_DEVICES=0。单条验收和全量启动尚待结果。

## 全量启动与一次健康检查

- Job16086 COMPLETED，4m14s，batch/srun均0:0。canonical0：140帧/117.90995秒；补采canonical16：144帧/110.08886秒。两条均逐帧qpos atol1e-5、原生任务成功、六目JPEG完整解码、逐帧标定、六目MP4全量解码及帧数通过。GT为同一次camera capture；这只是两条smoke，不是2500验收完成。
- 全量Job16088于2026-10-14 09:52:10 Asia/Shanghai开始，节点xn01-gpu1-0058，8GPU/128CPU/768GiB/48小时。实际冻结源码26b3533d662bb0881022f1071d0986f3dfcd1d5e，运行clone保持clean；后续仅本地提交记录，不更新正在运行代码。
- 一次启动健康检查：八个renderer分别记录不同PCI 19/3b/4c/5d/9b/bb/cb/db，每卡6305MiB、GPU利用率33%-100%；首批动态领取IDs466/467/481/484/492/498/2106/2128（最长轨迹优先）。没有新增traceback。旧Warp驱动入口警告在成功smoke和正式日志均存在，保存轨迹回放不依赖新规划；不声称该Warp环境已通过规划验收。
- 单条140帧实际产物193046721 bytes。粗推2500最终约0.8TB，占当时19TB空闲约4%；其中深度/视差占主要空间，压缩分割与遮挡小得多。保留float32数值，不有损量化GT。
- 按两条约1.19-1.31观测帧/秒/卡外推，主体及每条解码校验估16-24小时；预计10月15日02:00-10:00（北京）。这不是8卡完整稳定吞吐实测，复杂任务/共享IO/失败可延长。最终元数据汇总仅在2500条/50任务各50条/551534帧全部验收后生成ready.json，预计另需数分钟。
- 输出 /gpfs/jiuquyun/projects/Frank/robotwin-stereo-gt-20261014/；正式日志 /gpfs/jiuquyun/home/Frank/logs/robotwin-gt-16088.out；逐条logs/*-16088.log；每条episodes/ID/accepted.json绑定实际attempt HDF5和源码。
- 当前已验收2条，八卡正式运行中；没有训练或更改其他任务的数据入口，没有删除文件。按规则启动后只做此次健康检查，不循环轮询。

## 用户要求暂停全量并测试单卡多进程

- 用户明确要求停止当前全量，测试提速后恢复。实时核对Frank/Job16088/源码26b3533后执行scancel；sacct确认CANCELLED，运行7m29s。已验收0/16共284帧保留；八条部分尝试466/467/481/484/492/498/2106/2128保留，停止前已保存约570-600帧/条；没有删除或覆盖。
- 本地HEAD0e5ef8c、fetch成功且origin差异0/0，原有untracked不动。新增匹配吞吐实验：三GPU分别单卡1/2/4进程、每卡16CPU，相同IDs0/16/1/2，完整渲染/GT/写盘/解码验收。实验输出独立benchmarks/JobID，不进入正式数据。
- 生产采集只增加阶段计时与实验输出位置参数，不改变相机/渲染精度/轨迹/数据验收标准。依据批次总帧/墙钟时间选择并发；不按单卡显存直接推断加速。
- 并发实验Job16091正在三卡运行，冻结1276de5；首条单进程117.12秒中observation79.38、GT4.20、write11.05、setup21.20、validation0.67秒。依据SAPIEN本地类型定义新增独立六相机RenderCameraGroup能力/数值探针；复用另一未运行的本人clean FastWAM-stereotok-robotwin clone，避免更新活跃基准源码。总测试GPU不超过4，低于已授权8卡。
- CameraGroup探针16093退出0，但serial0.381s/group0.377s且Position/Segmentation读回不同，不能用于生产。单卡4进程批次578帧/445.10s=1.299帧每秒；等待1/2进程完整结果。进一步使用同一基准改变OMP线程数，核实单线程CPU去噪限制，不改变OIDN/32SPP等质量参数。
- Job16091全部通过：1/2/4进程分别491.509/487.303/445.102秒、1.176/1.186/1.299帧每秒；4进程相对1进程+10.43%。OMP8单进程同批次Job16095，冻结078a872，运行在第二clone。原replay clone已空闲，可同步后续异步写入实验。
- 定向优化Capture：单个后台writer，最多两帧在途，渲染与JPEG/GT压缩/视频管道写入重叠；按已知帧数预分配HDF5，省去逐帧resize。保持所有数值dtype/压缩参数和六目视频编码不变，验收前等待所有写入并传播异常。增加顺序/dtype和后台错误传播测试；需真实轨迹吞吐及数据对比后才能正式启用。
- 本地编译通过；本地环境缺h5py，新增writer测试改在既有H100运行环境、Slurm基准开始前执行，未为测试安装新依赖。
- CameraGroup追加接口核实：此前逐camera读取产生stale-buffer警告，下一探针直接读取group.get_picture_cuda张量再CPU回读，判断批量接口本身是否可用；仍只在tests实验，不进入生产。
- 拟采用已验证的每GPU四个领取lane（八卡共32），沿用全局flock、最长轨迹优先、仅跳过accepted。单条时限按max(1800,6*帧数+300)秒，避免同卡竞争让长轨迹误触旧30分钟上限；最终并发仍以异步版实测为准。
- Slurm共享GPU绑定实核：srun --ntasks=12 --gpu-bind=single:4，12个rank实际PCI为9b/bb/cb各4个。正式使用32个独立Slurm rank、每rank4CPU、GPU总数8，保留srun kill-on-bad-exit，不在pool内部叠线程。异步版首条与16091基线84个HDF5 dataset逐帧完全一致；比较经分配中的Slurm step执行。首次检查缺PYTHONPATH失败，补齐既有extras后通过，非数据失败。
- 异步writer首两条123.74/118.16秒，慢于基线117.12/111.27；后台线程wall计时大幅拉长，未实现预期重叠收益。恢复此前已验收的同步Capture（包括原resize方式），不把无收益实现留在生产；最终保留32个Slurm rank调度与长轨迹时限优化。本地原生GT两项测试通过。
