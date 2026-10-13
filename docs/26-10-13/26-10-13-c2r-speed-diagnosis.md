# Wan 训练速度与通信诊断

最新用户约束：在H800得出结论前保留H100排队，不取消。15516在纠正消息到达前已取消，已原样重提为15518（同clean b77a7ba、ruhu-env.sbatch、4GPU/6更新）。H800持续调查已交给新任务01a13571-e6f2-7900-8f50-0a27053506a8；H10015518与H800调查并行保留，本条覆盖下文早期撤销H100的计划。

用户授权调查 H800/L20Y 的卡间通信、挂载盘读取和计算耗时，并在 H100 上用相同 Wan 代码短测；不涉及停止或热改正在启动的正式训练。

基线：已完成 Wan DLC dlc594fm3bvfmi8x，连续稳态间隔中位40.681秒；Stereo dlc1miyh3f363oyc为44.032秒。ruhu H100参考2.698秒。读取原Wan日志确认通信通道via P2P/CUMEM，单节点8GPU、NVLS支持；不能从这些初始化日志推断实际通信带宽，也不能仅按卡名归因15倍差距。

新增 tests/profile_c2r_speed.py，仅用于短测：原训练入口外包装DataLoader next、VAE encode、forward和Accelerator backward（包含DeepSpeed step）的同步墙钟计时；各rank分别记录，前8个microbatch不计入中位值。嵌套forward项不能相加。额外GPU驻留256MiB all-reduce，5次预热20次计时，取最慢rank，记录algorithm和bus GB/s、GPU拓扑。此同步诊断有扰动，不能冒充未插桩训练吞吐；跳过checkpoint写入，避免诊断产出重复大权重。生产训练代码和参数不改。

H100 own clone：/gpfs/jiuquyun/projects/Frank/FastWAM-C2R-20261013，初始clean f8e563f，已fetch/pull成功。复用自己的Torch2.7.1+cu126运行时，独立extras补齐Accelerate1.12/DeepSpeed0.18.5/Transformers4.49等依赖，不改原环境。H800是Torch2.7.1+cu128，因此短测不是完全相同软件环境；H100用现有原始RoboTwin和ruhu授权只读root内的manifest/text cache/模型，H800是fresh2500，数据差异也须保留。两处均BS8/GA4/8GPU、原模型/精度/训练核心。

本地AST和git diff检查通过；实际8卡诊断尚待提交与验证。现有正式训练DLC dlc1xq5u3hveuc4w和dlc3lnq4rrwptiap由其他并行操作提交，本任务不变更其源代码或生命周期。

已发布诊断SHA 07f8cdd924455b46d185ea873026ca01ea62bc9e；H100自己的clean clone fetch/pull通过。独立extras导入C2RTrainer/FastWAM/DeepSpeed0.18.5成功；原runtime没有pip，改用既有uv向独立target安装，无原环境改动。W&B既有netrc条目存在，实际在线初始化仍由运行入口验证。

H100 Slurm Job15479：8GPU/96CPU/1TiB、debug QoS、45分钟上限、6更新。bash -n和sbatch --test-only通过，后者估计23:03:43可启动；实际提交后单次健康查询为PENDING(Resources)、START_TIME N/A，预测不是保证。脚本/配置/prepared与输出在/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/c2r-speed-20261013，日志/gpfs/jiuquyun/home/Frank/logs/c2r-wan-profile-15479.out；尚无模型执行、带宽或phase结果。

H800独立诊断clone尝试已到远端，但有界90秒命令未成功，退出1且无profile脚本落地。没有修改正式训练clone，也没有在该失败同步基础上运行新诊断。因此H800实测通信带宽仍缺失，P2P/CUMEM日志仅证明该次训练采用直接GPU通信路径，不能据此排除NVLink带宽或通信参数问题。

用户改为H100四卡诊断。实时确认15479仍PENDING后取消，sacct确认CANCELLED；替换任务保持每卡BS8，GA4改8，全局256不变，仍6更新。入口仅放宽为4或8进程且仍强制全局256，run.json记录实际world size；测试拓扑、bus带宽因子和预热丢弃改用实际进程数/GA。四卡ZeRO2每卡优化器分片增加且累积翻倍，不能直接用其秒/更新与ruhu八卡比较；重点比较每microbatch阶段时间和实测通信带宽。

四卡代码5aef19e533f6fa7383999f43d5f1438e05717dd9已commit/push，H100 fetch/ff后clean。AST、global batch/bus因子断言、diff、bash语法及sbatch --test-only通过。替代Job15483，4GPU/48CPU/512Gi/45分钟；独立配置同时将训练器和DeepSpeed GA设8。路径/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/c2r-speed-4gpu-20261013，日志/gpfs/jiuquyun/home/Frank/logs/c2r-wan-4gpu-profile-15483.out。预检预计20:46:25在0046启动；提交后单次健康检查为PENDING(Priority)，日志尚未创建，不能称已运行或测试通过。

用户询问崩溃后实时确认15483 FAILED/exit1，运行6秒。首异常为Transformers实际模型模块导入缺regex；尚未模型加载/计算，非OOM或通信失败。先前仅导入runtime.create_fastwam没有触发其lazy模型导入，因此预检覆盖不足。在独立c2r-20261013-extras补regex2025.11.3；通过runpy导入完整profile脚本、DeepSpeed导入、三个包的非可选依赖版本核验，以及实际资产T5TokenizerFast加载。一次-c命令因SSH引号转义导致SyntaxError，改stdin后检查成功，无训练参数变化。

按相同clean SHA与4卡BS8/GA8配置重提15486；sbatch语法和test-only通过，单次启动健康检查RUNNING于xn01-gpu1-0046，日志c2r-wan-4gpu-profile-15486.out当时尚无输出。不能把RUNNING称为训练或带宽测量通过，仍待首次模型/phase结果。

用户请求进度复核：15486运行5分14秒，RUNNING，无Traceback/OOM/RuntimeError/NCCL WARN匹配。Wan加载约105秒，ZeRO2/world4/GA8初始化成功，W&B online zy0fjn72。20:55:05完成step1/6，loss1.925726、action1.430495/video0.495231、grad58.9047，峰值51.3163GiB。首步约121秒含启动/编译，尚无连续稳态间隔，不能据此判断吞吐。

H100四卡GPU-only 256MiB all-reduce：5次预热20次计时、最慢rank平均1.189816ms，algorithm225.61085GB/s、bus338.41627GB/s。所有peer access=true，四卡两两NV18，各链路状态26.562GB/s。仅本次H100四卡测量；H800实际带宽与训练phase结果仍待补齐，不据此判定整体瓶颈。

15486的6个更新均完成，step2..6间隔89.1440/89.5859/89.3631/89.6860/89.3907秒，中位89.3907秒，全局256对应2.8638样本/s。四rank丢弃前16个microbatch后：backward/DeepSpeed step中位10.8446至10.8512秒，forward_total0.2858至0.2869秒，其中VAE0.1631至0.1633秒、denoise0.1210至0.1218秒；数据next等待0.224至0.265毫秒。反向阶段约占每microbatch总时长97%，尚未分离梯度计算、ZeRO和细粒度通信。ruhu八卡GA4中位2.6982秒/更新，相当于0.6746秒/microbatch；本次四卡GA8约11.1738秒/microbatch，归一化仍约16.56倍。诊断同步有扰动，不能把归一化当实测八卡吞吐。H100复现慢速，H800独有硬件/CPFS不足以解释共同现象；H100当前数据等待可排除为主耗时，H800读取仍未直接计时。

作业最终FAILED/exit1：诊断包装save_checkpoint返回None，父trainer在第6步完成后读取weights_path时报TypeError。4rank phase文件均已在finally写出，计时有效，但completed.json缺失，不称成功退出。定向修复仅测试包装返回显式SKIPPED_DIAGNOSTIC路径标签，不伪造checkpoint文件；增加第17次反向的rank0 CPU/CUDA operator profile，以分离慢反向具体原因。正式训练代码、数据和优化器参数不改，下一次只跑3更新取一个已预热反向样本。AST及diff检查通过。

追加Job15506（4GPU/48CPU/512Gi/45分钟上限/3更新），clean发布源码b77a7ba，H100 fetch/ff及bash/test-only通过；沿用已编译cache、独立wan-backward-profile输出，日志/gpfs/jiuquyun/home/Frank/logs/c2r-wan-4gpu-backward-15506.out。反向operator表尚未产出，不提前断言具体慢算子。

按用户要求直接对照ruhu run：trainer.py、wan_video_dit.py、action_dit.py、mot.py、pytorch_utils.py文本完全一致；FastWAM差异是Stereo分支和绑定，Wan仍走原路径。参考run已保存trainability为video4,999,787,712/action1,020,912,660/proprio86,016，VAE冻结；DS相同ZeRO2、无offload、200M buckets、不overlap、不contiguous。W&B差异仅scalar log，无watch/梯度hook注册。

使用ruhu同一manifest/data/text_cache，原reader与当前Wan reader分别在固定Python/NumPy/Torch seed42下读取index0/12345/549786：所有字段（含video/action/proprio/context/mask）逐tensor完全相等。单样本video[3,9,384,320]float32、action[32,20]float32、proprio[1,20]float32、context[512,4096]bf16及bool masks相同。这是三样本证据，不冒充全量扫描；同时本次H100慢速使用的就是该原始数据，不是H800 fresh数据。

发现实质运行环境差异：自己的extras Accelerate1.12.0/DeepSpeed0.18.5；ruhu授权root内runtime/overlays/openwam/site-packages为1.15.0/0.18.9。prepare_private_runtime.py及报告表明其从train8.sbatch使用的runtime27及两个overlay复制并逐文件验证（这是训练后保存的副本，不能绝对排除源环境此前变化）。两个版本accelerate/utils/deepspeed.py文本相同；DS stage_1_and_2.py存在关键差异：0.18.5每个参数grad_handling_hook均调用count_used_parameters_in_backward(all_params_requiring_grad)，helper遍历全部参数并查询autograd节点；0.18.9通过should_refresh_expected_hook_count判定，复用_max_expected_hooks_seen，避免每hook重复全参数遍历。此为与慢反向高度吻合的候选根因，尚需profile或隔离版本A/B确认，不称已修复；未修改正在正式训练的环境。

用户明确要求使用ruhu环境进行对照，授权只读执行/gpfs/jiuquyun/projects/ruhu/fastwam-c2r-20261011/runtime/model27/bin/python。实际元数据确认Accelerate1.15.0/DeepSpeed0.18.9/Transformers4.53.2/einops0.8.2/h5py3.16.0/W&B0.30.0/Hydra1.3.6，完整profile入口导入成功。使用自己的同一clean源码b77a7ba、同一ruhu数据/manifest/cache/初始化、4卡BS8/GA8/全局256；单独train-ruhu-env.json、输出wan-ruhu-env与个人inductor/triton/extensions缓存，不写ruhu环境。

15506旧环境算子profile在第2更新后、第17次反向阶段长时间未产出operator表，按替换环境诊断目标停止并保留已有日志（sacct CANCELLED），不把该段停留当成训练崩溃。提交新Job15516：4GPU/48CPU/512Gi、6更新、45分钟上限，脚本ruhu-env.sbatch，日志/gpfs/jiuquyun/home/Frank/logs/c2r-wan-4gpu-ruhu-env-15516.out。bash和sbatch --test-only通过，预检估计21:59:40在0057启动，预测非保证。未修改H800正式训练或其运行环境。

用户要求继续分析差异，先H100复现ruhu速度，再回H800继续训练。执行顺序固定为：15516验证整体环境差异；若恢复速度，再隔离必要依赖差异而不直接归因单一包；H800保持fresh2500正式数据/8卡/BS8/GA4/global256/W&B/5epoch合同，使用验证后的独立环境先短测再切正式任务。四卡只能按microbatch和阶段耗时对照ruhu八卡，不能宣称已直接复现八卡2.698秒。

## 独立调查任务：H800 DeepSpeed 隔离 A/B

用户授权交接后在 H800 用最多四卡继续因果诊断，保持 BS8/GA8/global256，不再申请 H100 GPU。本地 HEAD827906f 与 fetch 后 origin 一致；原有正式训练记录 dirty 保留。RAM frank/205455090823393499、research 角色、training quota Ready、DLC 优先级4已实时复核。正式 Wan/Stereo 作业仍 Running，不修改其源码和环境。复用个人准备 DSW，仅为同步及独立 overlay 准备。

诊断 test 的 operator profiler 改为仅在显式 C2R_PROFILE_OPERATORS=1 时开启，默认保留同步阶段计时，避免旧 DS 的海量 profiler 事件扰动 A/B。仅修改测试包装，不改训练实现。计划独立 target 安装 DS0.18.9，其他依赖维持原样，用相同 H800 fresh2500 数据、配置、初始化比较旧/新 DS；当前仍为候选机制，尚无因果验收。

再次逐字段对比参考configs/train.yaml与train-ruhu-env.json：除输出目录、W&B online、GA4→8外一致；实际入口另将max_steps限制6、save_every改epoch间隔，诊断跳过checkpoint写入。这些差异不修改模型、监督或数据reader。H800两任务GetJob均Running，最新日志Wan step63/10725（约40秒/更新）、Stereo step52/10725（约43秒/更新），loss有限；15516仍PENDING(Resources)。首个正式保存点2145尚未到达，切换前必须实读完整checkpoint，不能把已有smoke权重当正式恢复点或静默丢弃正式进度。
