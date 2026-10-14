# 三模型九项 baseline 准备与验证记录

## 当前有效合同（用户最新纠正）

- FastWAM LIBERO必须继承官方`libero_uncond_2cam224_1e-4`：原生reader、224x448双相机横拼、action7/proprio8、文本128；只有FastWAM RoboTwin clean-only和RoboDojo对齐OpenWAM对应实验。此前OpenWAM-reader LIBERO smoke是适配实验，撤销其“原版baseline通过”的计数，保留日志溯源并重跑原版。
- OpenWAM三组保持其官方配置。LingBot-VA的LIBERO/RoboTwin采用原作者配置与官方lingbot-va-base初始化；RoboDojo采用RoboDojo团队XPolicyLab适配，不能称为原作者官方实验。不套用FastWAM初始化或数据表示。包括FastWAM历史RoboTwin配置复核在内，总范围为九项。
- 全部只做训练smoke、checkpoint与真实仿真闭环；无正式训练、无StereoTok接入。历史段落保留当时操作事实，冲突之处以本节为准。

### 当前验收快照（2026-10-14 18:10，按用户最新口径）

| 模型 | LIBERO | RoboDojo | RoboTwin |
| --- | --- | --- | --- |
| FastWAM | 原版四套训练/权重/短闭环通过 | 旧适配不计；OpenWAM对应FastWAM复现配置来源待核实 | 按用户最新指示保留此前已完成结果，不再以配置来源复核阻塞 |
| OpenWAM | 四套短闭环通过 | 训练/权重/8个真实动作通过 | clean及randomized动作闭环通过 |
| LingBot-VA | 官方Long训练/权重/8个真实动作通过 | RoboDojo团队适配训练/权重/8个真实动作通过 | 环境及所选发布文件完成；两份latent缺项按原生reader过滤，后续继续548/550窗口；训练和闭环尚未通过 |

按用户保留FastWAM RoboTwin历史结果的最新口径，完成7/9（本轮六项闭环验收，加此前FastWAM RoboTwin）。LingBot RoboDojo保留上游训练8动作/latent与推理12动作/frame的已知差异，证明发布适配代码跑通，不宣称时序一致或论文分数复现。每项证据路径与具体覆盖范围见下面的运行记录；不得把历史适配结果加入当前计数。

日期：2026-10-14，Asia/Shanghai。目标：先在阿里云 H800 平台跑通原版 FastWAM 的两个仿真，再开展 StereoTok 对照。本轮尚未安装、下载数据、启动实例或训练。

> 上述为初轮状态；最新状态及初始化合同以文末“Wan初始化纠正”节为准。公开策略checkpoint调研仅保留溯源，不作为本任务初始化来源。

## 本地与云端事实

- 初始 cwd：C:\Project\StereoTok；分析仓库 C:\Project\FastWAM。
- 分支 frank/fastwam-stereotok-robotwin，HEAD/origin 同为 0cee262d2ff8a15e15513cb104c6ec61e247436b；git fetch origin 成功。tracked diff/cached diff 均为空；既有未跟踪 .agents/、AGENTS.md、docs/26-10-08/、docs/26-10-09/、third_party/SAM2Act/ 保留，未 pull。
- 已完整读取 FastWAM AGENTS.md 和 StereoTok/doc/frank/H800服务器使用指南.md。
- sts GetCallerIdentity 实核 RAM frank / 205455090823393499；工作空间 350607，research 角色 role-ruehjev23bey9is0wh。
- 实时 CommonQuotaConfig 已允许该角色使用 training 配额 quota1bybv9m0ixu，DSW/DLC 可用；ListQuotas 返回 Ready / L20Y。旧指南 research 子配额不可照搬；优先级上限仍为4。
- ListInstances 带/不带 CreateUserId 均返回 TotalCount=0，但 GetInstance 精确查询可读旧实例，不能用空列表断言旧实例不存在。
- dsw-j3tbe3ui1kjny2qq8x：Stopped，1GPU/24CPU/64GiB/shared16GiB；dsw-korgjd2dl8182z6qpi：Stopped，1GPU/24CPU/256GiB/shared32GiB。
- 未 SSH，尚未核验真实 GPU、CPFS 挂载、LIBERO 资产、环境或写盘。后续按用户要求独占创建小测试文件、fsync 并重读；不以 df 的零剩余容量直接判不可写，也不以小文件成功证明大数据容量足够。不删除或覆盖已有资产。

## LIBERO

本地 configs/task/libero_uncond_2cam224_1e-4.yaml：batch16，GA1，workers8，LR1e-4，10 epochs。configs/data/libero_2cam.yaml：spatial/object/goal/10 四套，双相机224，action7/state8，32 action/9 video frames。官方数据源 https://huggingface.co/datasets/yuanty/LIBERO-fastwam ，需先检查现有数据版本再选择确切下载文件。官方模型源 https://huggingface.co/yuanty/fastwam 。现有 experiments/libero/ 已提供评测入口。

## RoboDojo 已有适配与分数证据

- RoboDojo Team 已提供 FastWAM 适配：https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/FastWAM 。无需从零实现集成。
- OpenWAM 官方仓库公开表记录 Fast-WAM Avg SR=2.03%、Avg Sc=3.48：https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/robodojo/README.md 。这是该表版本的数值，本轮未成功读取实时榜单正文，不能称为实时榜单最新结果。
- 当前 README 示例为 RoboDojo cotrain / arx_x5 / joint；LeRobot v2.1，配套 stats、T5 cache、ActionDiT 权重。
- 当前 train.sh 默认每卡batch8、GA1、workers8、epochs16，复用 robotwin_uncond_3cam_384_1e-4；传入seed<=0时实际训练seed改为1。该默认不证明历史榜单训练配置。
- 当前 deploy.yml 默认 BF16、replan_steps24、num_inference_steps10、text_cfg_scale1、action_horizon=null（上游决定）、allow_dummy_policy=false。
- INSTALLATION.md 下载的是 robotwin_uncond_3cam_384.pt 与其 stats，不能据此认定是 RoboDojo cotrain 榜单 checkpoint。榜单 checkpoint/hash、训练数据清单、实际预算、评测种子与每任务回合数尚待核验。
- RoboDojo README 标明 Python3.11 / Isaac Sim5.1 / Isaac Lab2.3。2026-09 RGB/观测修正要求配套 XPolicyLab bb9a0b5f5136a74503b679af830bfd0a3a837d5c 或更新版本。安装前须冻结相容的代码与资产版本。

源码证据：https://raw.githubusercontent.com/XPolicyLab/XPolicyLab/main/policy/FastWAM/README.md 、INSTALLATION.md、train.sh、deploy.yml；https://github.com/RoboDojo-Benchmark/RoboDojo 。本轮读取 main，仅作调研；正式运行必须固定完整SHA。

## 初轮提出的执行合同（后续授权见下节）

建议本轮仅环境/数据准备和最小闭环 smoke，使用1GPU/24CPU/128GiB/shared32GiB、priority4、最长6小时的专用DSW；当前可用training配额，闲时策略需确认。不启动正式训练或全量评测。LIBERO按官方四套原版基线；RoboDojo按官方42-task模拟评测及cotrain数据路径核对。需要用户确认资源与执行分支后继续；尚未完成环境、数据或仿真验收。

## 用户扩展授权与已执行准备

用户已同意沿用 FastWAM 当前分支、隔离仿真代码与参数，并允许增加GPU并行smoke；新范围为 FastWAM/{libero,robodojo}、OpenWAM/{libero,robodojo,robotwin} 五组。正式训练不得启动；未来训练参数要求以OpenWAM为准。

已从官方克隆以下本地源码，未修改生产代码、未commit/push：

| 目录 | SHA | 用途 |
| --- | --- | --- |
| C:\Project\OpenWAM | 4d7f642b841913ea3c176010b692b1de3d5ef180 | 官方策略及LIBERO/RoboTwin客户端 |
| C:\Project\RoboDojo | 09bd5f5e762b949f5858ece31347ce62057b3a5e | Isaac仿真入口 |
| C:\Project\XPolicyLab | f6e309e8ece20bfa33c0e5d7b05a5f3287522757 | RoboDojo模型适配 |

OpenWAM main clean，fetch/pull --ff-only通过。RoboDojo与XPolicyLab clone后status clean。XPolicyLab Windows checkout提示无关InternVLA配置a1-Franka/a1-franka大小写冲突；未修改该模型文件，H800须独立Linux clone，不能拷贝此工作树作为运行来源。已阅读XPolicyLab根AGENTS.md。

### 资源回执

专用DSW dsw-t7b951pdehd9to9kgb，4GPU/96CPU/512GiB/shared64GiB，priority4/ForceQuotaOverSold；本人当前training配额quota1bybv9m0ixu。CreateInstance RequestId 01A13877-5BA8-5B66-9909-F46F30FA81B9，CLI dry-run先通过。请求目录 C:\Users\Frank\AppData\Local\AliyunCLI\h800-requests\frank-five-baselines-20261014；dsw.json SHA256 b08d009e258b379fe59156f02b290f6a89f7e9113119a5df50f323c6ca1d7fdb。

最后实例状态Queuing，SSH Allocating，无地址，事件列表为空；尚未到达远端，不能声明环境就绪。已设置并读取验证云端自动关机：2026-10-14 17:34:17 +08:00（DueTime 09:34:17Z），从设置时起6小时，包含排队时间。不自动续期，不重复申请排队实例。调度ETA未知；安装/资产传输ETA须测实际网络后报告。

### 隔离设计与验收要求（尚未落地远端）

- 五个模型/benchmark组合分别拥有运行配置与输出目录；模型服务与三个仿真客户端分离环境，绑定明确解释器、源码SHA、checkpoint、stats及端口。
- RoboDojo的FastWAM与OpenWAM使用各自适配目录和策略环境；OpenWAM的XPolicyLab vendored实现带batch支持，不可用官方单请求源码无声替代。
- 共享资产只按固定版本只读引用，不共享可写配置、stats生成位置或运行输出。禁用dummy policy；缺少权重、维度/动作类型/相机顺序不匹配时应终止，不能静默回退。
- 验收须包含实际checkpoint加载、真实仿真reset/render、obs→policy→action→step闭环与结果文件。import/debug hold-position不是闭环通过，也不把smoke分数当榜单复现。

### 新证据与尚待对齐的技术路线

RoboDojo官方HF数据仓库 revision 35efbc7dedfdbeeb6e95fb749bd885d73d483e41 的 ckpt/RoboDojo/FastWAM/RoboDojo-sim-arx_x5-joint-0/checkpoints/weights/step_020000.pt 已查到：12041813433字节，LFS SHA256 38d38afac480e151c4df079dbf4e869d3b45f3a023f862215a7baaa689af4db0。配套dataset_stats.json：10942字节，SHA256 af119307e4fdd0cc0c052e2017d1aa2afc270a07e7d0dedf1f325b86f3c8c4ef。未下载权重；文件名20k不替代直接checkpoint计数，也尚未证明该权重与公开分数一一绑定。

发布配置实查：https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-LIBERO/raw/main/config.yaml 为10epochs/per-rankB4/GA1/LR1e-4；RoboDojo对应发布配置为60000steps/per-rankB4，均从foundation微调。不能拿仓库通用train.yaml的5epochs/B16冒充发布配方；全局batch还须核验world size。

动作和输入存在实质差异：FastWAM LIBERO为7D action/8D state、双相机224；OpenWAM LIBERO为native10D→unified80D、384x320。FastWAM RoboDojo发布权重joint，而OpenWAM RoboDojo只支持EEF20→80D。立即把FastWAM所有配置改成OpenWAM会改变模型/数据合同并使原版checkpoint不兼容。需确认本轮先按各自发布ABI跑通、后续训练再做语义对齐，还是本轮就实现统一动作/图像合同；后者不再是原版checkpoint直接复现。

RoboDojo下载脚本列出的HDF5约523GB、joint LeRobot v2.1约64GB、demo约1.5GB；先用对应demo做数据smoke，不能混用joint导出替代EEF训练数据；大规模下载尚未开始。所有五项GPU闭环目前均未验收。

## Wan初始化纠正及本地代码准备

用户明确纠正：五组均从Wan初始化开始，不使用已训练FastWAM策略、OpenWAM-Alpha foundation或下游策略checkpoint。本轮只做smoke，不启动正式训练。原先以joint发布checkpoint兼容性为由暂停是不适用的路线判断，已撤回。

### 云端阻塞

完整GetInstance返回4GPU实例Failed，首因：InternalError / Resource queue timeout, lack of sufficient resources。未进入服务器，不是磁盘写入失败。将同一实例缩为2GPU/48CPU/256GiB/shared32GiB，UpdateInstance RequestId 01A13896-89B0-5F0E-A348-9804C603D8F3；StartInstance RequestId 01A13896-C29D-57A5-A2E0-41C82565877A。最后状态Queuing/SSH Allocating，无地址。17:34:17 +08:00自动关机仍经API验证有效。未创建第二个实例。

### 已完成的本地改动（均未提交/推送）

- FastWAM新增 configs/wan_init_smoke_{libero,robodojo}.yaml，configs/data/openwam_{libero,robodojo}.yaml：两个独立入口，输出按fastwam/benchmark/timestamp隔离；原有RoboTwin配置不变。
- src/fastwam/datasets/openwam_dataset.py直接构建官方OpenWAM reader，固定SHA 4d7f642b841913ea3c176010b692b1de3d5ef180并拒绝reader/基准配置dirty。复用EEF归一化、映射、画布、采样与mask，不自己实现这些数据语义。只将PIL视频转为FastWAM的C,T,H,W张量，复用既有T5 cache读取方法；无随机样本恢复。512长度T5、80维动作/状态，32动作/9视频帧，384x320。
- src/fastwam/models/wan22/fastwam.py支持[B,T,D]动作padding mask，分母仅计有效单元；保留原有[B,T]时间mask语义。对应tests/test_openwam_action_mask.py覆盖未使用维度零梯度、两种mask等价、全padding与错误宽度。
- OpenWAM新增 configs/wan_init_smoke_{libero,robodojo,robotwin}.yaml：继承官方train及对应dataloader；video_backbone.from_scratch=false（保留Wan预训练）、finetune/resume=null。
- 五组smoke均batch2/GA1/max_steps2/save@2；这是集成检查资源配置，不是正式预算或最终训练配方。OpenWAM其他训练默认继承官方train.yaml；FastWAM匹配LR1e-4、AdamW betas(0.9,0.95)、weight_decay0.01、cosine/5%warmup/1%minimum、clip1、BF16、seed42、5epochs上限。使用既有accelerate_zero2_ds.yaml启动，不用train_zero1.sh默认的ZeRO1。

### 后续执行命令合同

两侧均使用 `--config-name wan_init_smoke_<benchmark>`。OpenWAM通过原有scripts/train.sh；FastWAM通过 `accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes 2 scripts/train.py --config-name wan_init_smoke_<benchmark>`。正式运行只能使用已推送SHA的本人clean clone；目前不具备这个发布门禁。

必须显式提供各基准的BASELINE_LIBERO_DATA/BASELINE_ROBODOJO_DATA/BASELINE_ROBOTWIN_DATA，BASELINE_RUN_ROOT，OpenWAM使用BASELINE_WAN_ROOT；FastWAM另外指定BASELINE_ACTION_DIT（仅Wan派生ActionDiT骨干）、BASELINE_TEXT_CACHE_ROOT及DIFFSYNTH_MODEL_BASE_PATH。T5缓存需在对应基准目录按512长度生成；未授权使用现成策略权重替代这些资产。FastWAM的processor专用视频验证关闭（eval_every=0），真实仿真客户端集成仍待实现/验收。

### 验证与缺口

- 已执行：3个Python文件AST解析、5组smoke YAML及2个data YAML解析；两仓库git diff --check通过。
- 未执行：Hydra组合/配置解析、PyTorch单元测试（本机base、download、media-tools三个环境均无torch；base无hydra）、真实数据reader、Wan载入/反向/保存、checkpoint计数、模拟器render/reset、五组动作闭环。没有把静态检查称为smoke通过。
- OpenWAM代码结果表确实列出FastWAM LIBERO/RoboDojo结果，但当前仓库configs与scripts检索未找到同名独立重训配方。不能称本次新对齐配置为论文原始FastWAM配置。
- 仿真环境、实际数据格式/库存/缓存、FastWAM的EEF80闭环解码与两框架各自smoke checkpoint部署均未完成。后续先实查数据并运行掩码测试，不能据5份YAML宣称五个baseline跑通。

## 12:40 左右的远端实测更新

实例 dsw-t7b951pdehd9to9kgb 已 Running；SSH root@120.77.254.228:1020，hostname dsw-388227-f5d9bd45-6bdwj。2 张 NVIDIA L20Y，各 81559 MiB，driver550.127.08，torch2.7.1+cu128 实际双卡BF16矩阵乘通过。Vulkan枚举均支持VK_KHR_ray_tracing_pipeline及acceleration_structure；尚不能据此声称IsaacSim已工作，也不能仅据CUDA能力9.0判定不支持。

五组本地配置通过远端Hydra内存ConfigStore组合检查；FastWAM真实loss helper AST提取后4项测试通过（0.417s）。这些是临时stdin定向检查，未将本地未提交生产代码部署远端，尚无五组训练/闭环成功证据。

CPFS显示0可用但实际3840字节独占创建、fsync、读取成功，随后只删除该明确probe文件。OpenWAM干净官方clone位于/mnt/workdir/frank/projects/OpenWAM-baselines-20261014，SHA4d7f642b841913ea3c176010b692b1de3d5ef180。RoboDojo独立clone位于同projects/RoboDojo-baselines-20261014，SHA09bd5f5e762b949f5858ece31347ce62057b3a5e。

独立策略环境/mnt/workdir/frank/envs/openwam-baselines-20261014已装torch2.7.1+cu128/torchvision0.22.1，其余官方pyproject依赖安装进行中。LIBERO使用官方benchmarks/libero/setup_env.sh，固定LIBERO8f1084e3132a39270c3a13ebe37270a43ece2a01及官方补丁，独立env libero-baselines-20261014安装进行中。RoboDojo独立Python3.11.15 env robodojo-baselines-20261014已创建，IsaacSim5.1安装进行中。现有FastWAM和RoboTwin环境未修改。

LIBERO旧/mnt/data/data/h100-jiuquyun/PRETRAIN_DATA/LIBERO目录为空。OpenWAM/LIBERO revision bcb2eaf1121ae4cbd324f8862807abce282500e8，1944569454bytes，目标/mnt/workdir/frank/datasets/OpenWAM-LIBERO-bcb2eaf。HF直连Network unreachable；hf-mirror可用，首次xet-bridge读取10s超时，保留断点以120s超时重试。数据未完成，不宣称ready。

仅为现有Wan初始化文件建立独立symlink集合：/mnt/workdir/frank/checkpoints/baselines-20261014/Wan2.2-TI2V-5B，包括3个DiT shard、Wan2.2 VAE、UMT5、tokenizer；源是frank已有fastwam-c2r-shared-20261013下的官方Wan及DiffSynth converted权重，无机器人策略checkpoint。尚未完整模型加载验收。

仍保留17:34:17 +08云端关机，不启动正式训练。FastWAM本地变更尚未commit/push，受AGENTS“Local edits → user pushes → server pulls”约束；远端目前可运行干净官方OpenWAM及临时检查，不能假称FastWAM新适配已上线。

## 远端数据验收与已提交测试链

- LIBERO snapshot已完整下载，459个文件逐项大小匹配官方固定revision，合计1944569454bytes；首个视频经PyAV实际解码RGB uint8 256x256，范围0..244、std41.063。conversion.json记录1693episodes/273465frames/40tasks/native_delta_eef10。
- RoboDojo smoke精确下载revision35efbc7dedfdbeeb6e95fb749bd885d73d483e41的pour_liquid_into_cup/arx_x5 episode0000000/1。episode0 SHA256 72b5eff7a627c76917f3af0c501e7c2daa8259f6ea2734cfefade69b35dccd57匹配；HDF5实际读取176帧，左右pose[176,7]、gripper[176,1]、三路压缩RGB及instruction存在。目标/mnt/workdir/frank/datasets/RoboDojo-smoke-35efbc7/data/RoboDojo，仅smoke子集，不能报完整benchmark覆盖。
- RoboTwin从已有官方adjust_bottle的aloha-agilex_clean_50/randomized_500 zip各解出episode0/1及对应instruction，共8个文件，目标/mnt/workdir/frank/datasets/OpenWAM-RoboTwin-smoke-20261014。未改共享原始数据；烟测统计只落此独立根。
- RoboDojo源码固定submodule：XPolicyLab53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2，IsaacLabafca7b09d60d8beb9c1cb28b43066499940b969b，curobod17b54ce32cba095c0b000c4c58777075d11de0e。不是--remote追踪漂移。Assets/**共15365files/41269513111bytes，已按上述数据revision下载到/mnt/workdir/frank/assets/RoboDojo-35efbc7；尚未完成和链接。
- 安装缓存切换到/tmp/frank-baselines-openwam-cache与/tmp/frank-baselines-uv-cache，最终环境仍在CPFS。原uv进程SIGINT后有序结束，未删除缓存或改其他环境。现OpenWAM安装PID31185，Isaac安装PID30321；LIBERO官方setup父timeout PID10635，pip PID16856。可能随安装结束自然消失。
- OpenWAM三组测试链PID35953：/mnt/workdir/frank/runs/five-baselines-20261014/openwam-training-chain.log。等待安装，严格检查imports、SHA/clean/upstream，然后真实reader样本shape/finite验证，依次libero/robodojo/robotwin两卡、B2/GA1、max_steps2/save2、Wan init、finetune/resume null。每组独立reader.log、train.log、training-started.json及checkpoint目录；退出0仍需检查checkpoint内容，不自动认定闭环成功。顶层timeout6300s。仅调用已发布官方入口及CLI覆盖，未部署未提交生产代码。
- LIBERO渲染链PID36999：同根/libero-render-chain.log，等待官方setup和训练链释放GPU，避免CUDA/EGL并发。spatial/object/goal/libero_10各task0、256分辨率、3步，直接调用官方smoke_libero.py；不是策略rollout。日志libero-<suite>-env.log，顶层timeout8100s。
- /mnt/workdir/frank/runs/five-baselines-20261014/preparation-status.json记录已验证事实。两条链已提交但最后处于安装等待，不代表训练或仿真通过。FastWAM新适配仍未提交发布，实际训练链尚未启动；完整五组baseline未完成。

## 持续目标与发布授权

用户明确确认本任务commit/push，并要求持续处理五组baseline直到通过后停止。持续目标active；不启动正式训练。FastWAM首批实现已推送0c106f2（原始0cee262之后），远端独立clone为/mnt/workdir/frank/projects/FastWAM-baselines-20261014。新独立策略env fastwam-baselines-20261014复用wheelhouse安装，不修改已有环境。

继续核查发现文本条件差异：OpenWAM LIBERO的LeRobotV3Reader._resolve_prompt直接返回strip后的任务文本；RoboDojo会调用format_prompt_for_inference添加固定前缀。旧precompute_text_embeds只读tasks.jsonl且始终包装DEFAULT_PROMPT，不能复用于canonical LIBERO。适配器新增unique_prompts读取固定SHA reader已有元数据；原缓存入口对OpenWAMDataset直接调用该方法，禁止override_instruction覆盖canonical提示词。两项真实方法AST回归（原始LIBERO文本/空文本拒绝）远端通过；动作mask此前4项通过。端到端reader/cache/model尚待独立环境安装后验证。

RoboDojo资产首次下载已因hf-mirror HTTP429失败（Assets/Eval_Layout/.../cover_blocks_35.json）；保留下载断点，不删除文件。需退避并降低请求频率续传。未成功暂停活跃资产进程，因为检查时其已退出；没有误停其他进程。

## 13:18 镜像修复与真实 reader 启动

- FastWAM代码已发布63d6c78，独立远端clone通过fetch、clean检查及ff-only更新；原有实验目录未改。新env的torch/hydra/cv2/h5py/pyarrow/av、OpenWAM reader与FastWAM适配器真实import通过，torch2.7.1+cu128。
- PyPI慢下载经实测定位：同一sentencepiece wheel首1MiB，腾讯镜像0.14s、阿里云1.686s；清华wheel返回403。只切换索引，不改变版本约束。原四个本任务安装进程定向SIGINT后已退出，断点/缓存保留。腾讯镜像新PID：OpenWAM63078、FastWAM63079（已完成）、Isaac63080、LIBERO官方setup63082。日志在本任务runs根下 <name>-install-mirror.log。
- OpenWAM训练链35953整个进程组暂时SIGSTOP，防止旧安装PID退出造成过早放行。需在新依赖和FastWAM T5缓存任务完成后SIGCONT；LIBERO渲染链36999仍等待该训练链，不会并发占用CUDA/EGL。
- FastWAM T5生成链66530使用已发布源码，两个benchmark分别生成512长度缓存。LIBERO真实reader已成功构造：1693eps、273465windows、10fps、multiview、min-max。tokenizer_model_id显式指定Wan2.2-TI2V-5B内相同google/umt5-xxl本地文件；模型仍是Wan初始化。输出cache/five-baselines-20261014/text/<benchmark>，日志fastwam-<benchmark>-text-cache.log；尚未宣称生成完毕。
- RoboDojo Assets续传PID65528，固定同一HF revision、max_workers=1，失败后60至600秒退避，最多8次；约20%文件位置恢复正常推进。此百分比是文件数而非字节完成度，无完整资产成功结论。
- 两张GPU尚未开始正式模型反向；五组baseline均未满足最终闭环验收。持续目标保持active，无正式训练。

## 13:23 真实样本与训练进程

- 远端已发布测试文件通过unittest discover：6 tests，0.469s。不是仅AST临时检查。
- 两个FastWAM真实样本读取通过：video[3,9,384,320]、action及action_is_pad[32,80]、proprio[1,80]、context[512,4096]bf16，所有tensor finite。证据fastwam-real-readers.log。
- T5缓存完成：LIBERO40条、RoboDojo当前smoke子集1条，均无512长度截断；TEXT_CACHE_READY两组均出现。
- FastWAM双卡训练链PID70448已启动，使用63d6c78、原accelerate_zero2_ds入口、B2、max_steps2/save2、Wan预训练及Wan派生ActionDiT，未加载策略权重。日志fastwam-training-chain.log与fastwam/<benchmark>/train.log；13:23仍在Wan组件装载，尚无反向/loss/checkpoint通过结论。
- OpenWAM官方依赖安装完成；初次import因libGLX.so.0不在搜索路径失败。个人已有system-libs/prefix/usr/lib/x86_64-linux-gnu包含此库，显式LD_LIBRARY_PATH后torch/hydra/cv2/h5py/pyarrow/av/deepspeed/transformers/diffusers全import通过，torch2.7.1+cu128、transformers5.19.0。不替换官方opencv版本。
- 原35953/36999等待链在尚未开始训练/仿真时定向结束，重新提交携带LD_LIBRARY_PATH且绑定新PID的链：OpenWAM72159等待FastWAM70448及LIBERO setup63082；LIBERO环境smoke72160等待63082和72159。日志分别openwam-training-chain-v2.log与libero-render-chain-v2.log。不存在继续恢复35953的待办。
- LIBERO官方pip安装和IsaacSim包写入CPFS仍进行中；RoboDojo资产低并发续传仍运行。5组闭环验收均未完成，持续处理目标未结束。

## 13:28 首个训练成功与闭环服务适配

FastWAM LIBERO训练进程exit0；step1 loss1.6486（action1.0798/video0.5688），step2 loss4.4030（action3.7078/video0.6952），梯度累计1、world_size2、ZeRO2、bf16。运行目录fastwam/libero/20261014_132144，权重checkpoints/weights/step_000002.pt及完整state已写出。仍需直接读取checkpoint计数/张量及策略仿真。RoboDojo同链已启动Wan载入。

新增tests/serve_openwam_baseline.py，专用于本任务闭环smoke，不接入生产路径：保存config与请求benchmark必须匹配；dataset适配器继续验证OpenWAM SHA；mot/proprio严格加载训练checkpoint，step至少2；各run内保存归一化统计副本和hash绑定。只复用官方dataloader的Normalizer、统一维度scatter/gather、multiview拼图；FastWAM仍调用infer_action。单客户端JSON WebSocket与官方LIBERO客户端协议相同，拒绝缺相机、错维度、非有限数、未知提示词缓存。默认10去噪步/full32chunk对齐OpenWAM deploy默认。RoboDojo还需XPolicyLab原生坐标/动作桥接。

没有直接复用OpenWAM整套PolicyServer，因为其包导入会拉起全部模型架构；FastWAM独立环境实测首先缺ftfy，继续安装会造成无关模型依赖耦合。当前test server仅需独立websockets运行依赖，待FastWAM训练退出后安装，避免修改正在运行的环境。新增脚本AST解析通过；尚未进行真实checkpoint加载和客户端测试。

## 13:40 左右 checkpoint 与离线动作验收

- FastWAM两组训练链70448均exit0。RoboDojo step1 loss1.6447（action1.2185/video0.4262），step2 loss3.5958（action2.9290/video0.6668），运行目录fastwam/robodojo/20261014_132808。
- checkpoint审计88437已完成：两组weights各12042624505bytes；payload.step=2，trainer_state.global_step=2、batch_in_epoch=2；mot1649keys/6020823312params、proprio2keys/331776params。逐张量finite检查全通过，报告fastwam-checkpoint-audit.json。
- 训练结束后远端clean clone ff-only到9d53118，FastWAM独立env安装websockets16.1.1。离线推理审计94246分别严格重载mot与proprio，复用真实数据第一帧和恢复的raw state；两组都生成32步finite chunk，LIBERO raw10D、RoboDojo raw20D。证据fastwam-inference-audit.log及每个run/canonical_deployment/offline-inference-audit.json。只证明重载/推理，不冒充真实仿真闭环。
- 推理审计期间OpenWAM链72159定向SIGSTOP防GPU冲突，finally已SIGCONT恢复；它仍等待LIBERO安装63082完成（官方conda在OpenWAM目录生成临时requirements，当前未隐藏/删除这个活跃文件）。
- IsaacSim5.1.0.0安装63080已完成；实查torch2.7.0为CUDA12.6轮子，官方IsaacLab/RoboDojo要求cu128。新链99194按官方精确torch2.7.0+cu128/vision0.22.0+cu128/audio2.7.0+cu128补齐，之后执行固定submodule的isaaclab.sh --install none。日志robodojo-isaaclab-setup.log；后续仍需官方runtime repin、CuRobo和渲染验收，不能据安装metadata声称仿真成功。
- RoboTwin独立准备链96530：clone到projects/RoboTwin-openwam-baselines-20261014，固定OpenWAM验证SHA0aeea2d669c0f8516f4d5785f0aa33ba812c14b4；从frank自己的robotwin-render-py310只读克隆独立env robotwin-openwam-baselines-20261014。源版本torch2.4.1+cu124/sapien3.0.0b1/mplib0.2.1/curobo0.7.8符合；warp1.4.1、缺websockets需要在新env修正为官方warp1.13.0及客户端依赖，不修改已有FastWAM RoboTwin环境。clone尚未完成，后续需独立curobo源码绑定和资产检查。

LIBERO安装63082现已正常完成：官方setup打印No broken requirements found以及LIBERO evaluation environment ready with official requirements and MuJoCo3.3.2。OpenWAM72159恢复后import gate通过，当前进入git fetch门禁，尚无OpenWAM真实训练loss。RoboDojo固定XPolicyLab submodule实际包含policy/OpenWAM/model.py和client_server/ws/model_client.py，可复用其EE坐标桥接；并非假设集成不存在。后续验收仍须真实仿真，当前五组最终baseline通过数为0。

## 13:57 首个 baseline 闭环通过与 OpenWAM 显存修复

- FastWAM LIBERO闭环链104250正常结束。四套spatial/object/goal/libero_10各task0、1trial、settle30、policy_steps8；四份results.json均实际记录8步，策略服务严格加载step2并输出四个finite [32,10] chunk。日志位于five-baselines-20261014/fastwam/libero-closed-loop。2步训练权重成功率0/1符合smoke预期，不表示达到论文分数。结合之前训练、checkpoint审计，FastWAM LIBERO的smoke闭环通过；仍非40任务完整评测。
- OpenWAM训练链106363在LIBERO第一个AdamW更新失败：torch._foreach_sqrt临时请求11.22GiB，当时GPU仅剩9.32GiB；不是数据、Wan加载或前向初始化失败。保留openwam/libero/train.log，未覆盖旧输出。
- 使用官方现有training.offload_optimizer_device=cpu重试，保持B2、GA1、lr/seed/数据/模型不变；CPU offload是本次smoke运行差异，正式训练配置尚未启动。新链139735，日志openwam-training-cpu-offload-chain.log，三个benchmark分别写openwam/<benchmark>/cpu-offload-attempt，max_steps2/save2；当前Wan加载中，尚未通过。
- RoboTwin独立conda clone96530已结束，新env仅安装warp-lang1.13.0/websockets13.1/pyyaml，PID140324，日志robotwin-openwam-runtime-deps.log。检测clone保留editable curobo指向Frank自己的projects/curobo-v0.7.8（clean d64c4b005459db10c5dd867d8b30a87d5bda9bdb），后续需单独绑定源码避免与旧运行相互影响。
- RoboDojo CUDA12.8精确torch栈安装完成；IsaacLab扩展安装99194仍运行。Assets下载65528仍正常推进，约40%文件数，不能当作40%字节完成。持续目标active；当前最终smoke通过1/5，无正式训练。

RoboTwin runtime依赖140324已成功结束：warp1.4.1→1.13.0、websockets13.1。独立cuRobo构建链167338使用官方v0.7.8/d64c4b0，目标RoboTwin-openwam-baselines-20261014/envs/curobo，编译arch9.0、MAX_JOBS8；随后仅创建三个资产目录symlink到Frank已有assets/RoboTwin-bf44be5，不覆盖原资产。日志robotwin-openwam-curobo.log，尚待构建完成。OpenWAM CPU-offload链139735已完成CPU Adam初始化并进入Training 0/2，暂无通过结论。

## RoboDojo 单环境策略协议桥接

新增tests/serve_robodojo_baseline_bridge.py，仅用于已授权smoke：固定XPolicyLab53c5da5/OpenWAM4d7f642源码合同，复用XPolicyLab官方state→EEF20及EEF20→world动作方法，不调用其模型初始化或dummy路径。通过官方JSON WS predict_once接收完整32步chunk，再由XPolicyLab官方binary WS server提供给真实仿真；拒绝空instruction、多环境以及未reset切换env_idx，检查raw20D和转换动作finite。模型仍由各自独立服务加载本次Wan初始化smoke权重。之所以新增薄桥，是官方OpenWAM4d7f642没有XPolicyLab vendored版本需要的generate_batch，不能混用vendored模型实现。当前本地AST及diff检查通过，远端import/坐标回归和仿真仍待验证。

13:57 OpenWAM LIBERO CPU-offload重试完成两步：loss2.5384（video0.4230/action2.1154）、11.5109（video0.3098/action11.2011）；正在保存checkpoint_step_2.safetensors。未以日志进度代替checkpoint验收或仿真通过。

## 14:02 桥接回归与后续准备

- e2dd655已推送并同步到远端clean FastWAM clone。桥接首次import缺msgpack_numpy，在空闲FastWAM独立env安装msgpack-numpy0.4.8后通过。robodojo-bridge-contract.log记录官方EEF20→world→EEF20往返atol1e-6、三相机payload以及空instruction/未reset切换env/多env拒绝检查，exit0；这是合同测试而非仿真通过。
- OpenWAM LIBERO训练链已exit0，完整checkpoint_step_2.safetensors为24813767464bytes、2089tensor keys，未包含metadata独立step计数。训练链139735继续RoboDojo，之后RoboTwin。审计链177433等待训练链结束后逐benchmark验证exit0、保存配置Wan初始化及全部tensor finite，日志openwam-checkpoint-audit.log；尚未产出最终报告。
- IsaacLab99194已完成。RoboDojo运行依赖链176682按官方install.sh的版本列表repin，之后安装base requirements及固定submodule cuRobo[cu12]，不执行官方脚本的submodule --remote。约束保存在runs根robodojo-official-runtime-constraints.txt，日志robodojo-runtime-curobo.log；仍需处理真实import/render问题。XPolicyLab bridge运行在独立FastWAMenv的websockets16，不与Isaac要求的websockets12混装。

## LingBot-VA 并行准备范围

用户新增LingBot-VA训练准备，明确全部先做Wan baseline、全部通过后再考虑StereoTok。原五组继续，未暂停。已查询官方Robbyant/lingbot-va README：官方提供RoboTwin和LIBERO流程；XPolicyLab已有LingBot_VA RoboDojo适配，使用预编码Wan2.2 latent、独立单env cache。官方post-training入口加载lingbot-va-base，不能将其默认当作本任务要求的原始Wan初始化；需检查权重映射和新增action参数初始化，暂未下载LingBot策略checkpoint。H800独立源码clone PID181361，目标projects/LingBot-VA-baselines-20261014，日志lingbot-va-source-clone.log；当前仍clone中。待用户确认LingBot-VA是否同样覆盖三个benchmark及训练smoke/闭环验收，独立代码检查可先推进。

原五组进展：OpenWAM三组CPU-offload训练均已exit0；审计177433已核验LIBERO全部2089张量finite，其余继续。LIBERO闭环链180042等待训练及checkpoint审计成功后使用官方deploy默认compile/cache、10步去噪，四套各8动作；输出openwam/libero-closed-loop，日志openwam-libero-closed-loop-chain.log。RoboTwin独立cuRobo167338构建完成，已绑定新源码并连接三个既有资产目录。RoboDojo运行依赖176682继续安装。

## 用户确认 LingBot-VA 范围与模型配置独立原则

用户明确确认LingBot-VA覆盖RoboTwin/RoboDojo/LIBERO及训练smoke、checkpoint、仿真闭环，具体初始化对齐LingBot-VA官方，不套用FastWAM的原始Wan初始化要求。当前总范围为原五组加LingBot-VA三组；目标统一为先通过各自Wan baseline，再考虑StereoTok，不启动正式训练。LingBot-VA官方post-training使用lingbot-va-base为合法初始化来源；OpenWAM采用自身官方配置；FastWAM采用OpenWAM报告里对应FastWAM实验口径，保留FastWAM模型实现。所有环境、数据处理、参数与输出按模型和benchmark隔离。

LingBot-VA远端官方clone clean main，SHA7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb；不使用本机C:/Project/lingbot-va这个无Git元数据目录作为运行来源。准备独立env lingbot-va-baselines-20261014，按官方Python3.10.16/torch2.9.0+cu126/vision0.24/audio2.9；实际安装结果待核验。

原五组OpenWAM三个checkpoint审计已结束：每组2089tensor keys、24813767464bytes，全部finite，包含video/action/proprio三组。报告openwam-checkpoint-audit.json；计数证据为训练完成日志和checkpoint文件名，safetensors自身无独立step metadata。LIBERO闭环服务链180042已进入启动，尚未闭环通过。

LingBot-VA独立环境链184552已启动，源码fetch/pull确认up-to-date；官方base模型元数据固定HF robbyant/lingbot-va-base revision68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c，总仓库文件24382314341bytes，实际只下载text_encoder/tokenizer/transformer/vae四组。下载PID186492，独立checkpoint目录lingbot-va-base-68b7bc1，max_workers1、HF mirror、失败退避；不使用benchmark posttrain策略权重作为初始权重。

OpenWAM LIBERO首轮服务成功加载本次checkpoint和官方compile/cache，但客户端因启动环境漏传LIBERO_PATH而退出，尚未执行动作；保留原日志，不是模型推理失败。补齐独立LIBERO_PATH与LIBERO_CONFIG_ROOT后重启闭环链186140，输出openwam/libero-closed-loop-v2，日志openwam-libero-closed-loop-v2.log。RoboDojo依赖链176682已完成cuRobo v2安装；import验证184769已提交，日志robodojo-runtime-import.log，仍待结果及真实渲染。

FastWAM原版LIBERO纠正：wan_init_smoke_libero.yaml改为继承官方task，删除512文本、80D/OpenWAM reader与5epoch覆盖；继承10epoch等原版参数，只保留smoke B2/max2/save2/eval0和明确路径覆盖。四套原始tar数据源yuanty/LIBERO-fastwam revisionee018b997c430bb12b5bf3c892d744798c5a2f91，不能误取同repo的lerobot_v30派生版。当前Frank个人datasets只有OpenWAM版本，需独立下载官方四个tar并使用原生reader验证。原版baseline目前尚未重跑通过。

## 14:29 最终配置归属确认与当前验收

用户最终确认：FastWAM LIBERO对齐FastWAM原版；FastWAM RoboTwin clean-only/RoboDojo只对齐OpenWAM实际复现FastWAM的设置，不能套OpenWAM自身训练配置。OpenWAM三个benchmark遵循自身来源；LingBot-VA LIBERO/RoboTwin遵循原作者，RoboDojo遵循RoboDojo团队XPolicyLab适配并单独核验。未公开参数保留待核实，不推定相同。

- OpenWAM LIBERO闭环v2通过：四套各task0/trial1/8 policy actions，results均有实际动作计数。结合三组训练checkpoint审计，目前八组目标只有OpenWAM LIBERO完成smoke验收（1/8），不是完整论文评测或成功率复现。证据runs/five-baselines-20261014/openwam/libero-closed-loop-v2。
- FastWAM官方LIBERO四个tar已下载解包到datasets/FastWAM-LIBERO-native-ee018b9，日志fastwam-native-libero-download.log包含四个NATIVE_LIBERO_SUITE_EXTRACTED。尚需原生reader/128长度T5缓存及训练闭环。纠正配置已推送bae58fa；远端fetch连续TLS/GnuTLS -110失败，remote仍e2dd655，尚未通过新配置Hydra验证，不绕过Git门禁复制源码。
- RoboTwin独立env已装官方stable PyTorch3D0.7.8（75ebeea），安装日志robotwin-pytorch3d-install.log。将独立env中imageio_ffmpeg自带7.0.2二进制链接为该env/bin/ffmpeg，客户端PATH明确优先独立env/bin。重试链PID208717，输出openwam/robotwin-closed-loop-v2，分别demo_clean/demo_randomized各1episode，未启用planner fallback。启动健康检查时尚在策略服务装载，无闭环通过结论。
- RoboDojo IsaacSim首轮启动在0.85s左右崩溃，尚无Kit场景日志、渲染或policy动作。日志robodojo-isaac-startup.log及/root/.nvidia-omniverse/logs/omni.processlifetime.log；不能据此确定驱动是根因。CuRobo/IsaacLab imports此前通过不等于渲染通过。资产下载65528继续，约56%文件数而非字节进度。
- LingBot官方torch环境链184552及base下载186492仍活跃；尚未完成整体依赖或训练。各自日志lingbot-va-environment.log、lingbot-va-base-download.log。
- OpenWAM论文https://arxiv.org/pdf/2609.07398的附录B/Table9/Table10明示OpenWAM-alpha预训练/SFT，不能用它们填充FastWAM复现参数；表内FastWAM成绩引用也不能独立证明重训方法。FastWAM RoboDojo/RoboTwin复现配置来源仍需核实，已有OpenWAM-reader FastWAM结果只保留为适配实验。

本轮RAM重新核验frank/205455090823393499；DSW dsw-t7b951pdehd9to9kgb Running，SSH实际到达120.77.254.228:1020。持续目标保持active，未启动正式训练或StereoTok。
## 14:36 原版 LIBERO 门禁恢复与后续队列

- 远端Git重试成功，clean clone ff-only到f1f385c。真实Hydra解析wan_init_smoke_libero验证原生RobotVideoDataset、video224x448、context/tokenizer128、action7/proprio8、10epochs、mot_checkpoint_mixed_attn=false；日志fastwam-native-libero-config-check.log。数据四套metadata为原生LeRobot v2.1、20fps，分别434/457/433/388episodes，53229/67309/52895/104280frames；不是OpenWAM的数据版本。
- 原版FastWAM LIBERO链215003等待Isaac诊断链213730退出后执行官方precompute_text_embeds及accelerate ZeRO2双卡2步训练。新缓存cache/five-baselines-20261014/fastwam-native-libero-text；输出runs/five-baselines-20261014/fastwam/libero-native；链日志fastwam-native-libero-chain.log。仅tokenizer本地文件定位使用Wan2.2同源UMT5目录，不变更128文本合同。链内固定f1f385c且要求clean，不在任务运行中同步新源码。尚未开始GPU训练，不能算通过。
- Isaac诊断链213730等待RoboTwin208717退出；仅开启Python faulthandler、显式Kit日志路径并关闭崩溃报告器，保留原渲染和驱动设置，收集真实根因；日志robodojo-isaac-diagnostic-chain.log、robodojo-isaac-startup-diagnostic.log。
- OpenWAM RoboTwin clean完成400/400 policy steps且client exit0，success0/1（两步训练smoke不要求任务成功）；randomized已运行29/400。链208717仍活跃，完整两模式验收仍待终态和动作审计。
- LingBot后续requirements链215946等待torch链184552退出后验证torch2.9/cu126，再按官方requirements安装其它依赖、单独构建flash_attn、pip check及版本断言；日志lingbot-va-official-requirements.log。没有在torch安装过程中并发改同一env。
- LingBot官方LIBERO发布数据为LIBERO-Long，来源robbyant/libero-long-lerobot revision8c0313b1c7cd9fa3798798479cbf59b11af8979d，API usedStorage4693296349bytes。启动完整原生data/meta/videos/latents下载PID220180到datasets/LingBot-LIBERO-Long-8c0313b，日志lingbot-va-libero-native-download.log，2507目标文件；不以FastWAM/OpenWAM数据加工替代。官方dataset缺empty_emb.pt，后续须按其T5 encoder生成，不使用其它模型缓存；尚未验证reader。

本轮实现进展和已核验活跃等待均有PID依据；八组目标未完成，持续目标保持active，无正式训练。
## 14:50 用户明确最短 smoke 验收与九项计数

用户明确优先尽快证明代码跑通：训练只需1至2个更新、checkpoint检查、少量真实仿真动作，配置按已确认来源表核对；无需完整episode或分数复现。计数统一九项，包含FastWAM RoboTwin历史完成但配置仍需复核。

- OpenWAM RoboTwin接受为smoke通过：clean400步自然exit0，randomized在已实际执行346步后按用户最新指示SIGINT停止（退出130，不是自然exit0），服务随父链finally退出。checkpoint报告、真实推理日志与动作计数已核对，证据openwam/robotwin-closed-loop-v2/smoke-acceptance.json；当前2/9明确通过，另1/9历史完成待复核。
- FastWAM原版LIBERO训练215003已exit0，loss1.7995/5.2131；运行fastwam/libero-native/20261014_144406。审计240346通过：payload.step=2，weights12041735545bytes，mot1649keys、proprio_encoder2keys全部finite，证据checkpoint-smoke-audit.json。原版文本缓存40条/128，无截断；仿真尚未通过。
- FastWAM官方评测import实测缺robosuite。训练结束后才启动独立FastWAMenv评测依赖链241144，保持numpy1.26.4及既有模型版本，添加robosuite1.4/mujoco3.3.2/bddl1.0.1等仿真依赖；日志fastwam-native-libero-eval-dependencies.log。没有改动独立OpenWAM LIBERO模拟器环境。
- 新增tests/run_native_libero_smoke.py，复用官方eval_libero_single的全部预处理、模型推理、动作转换和episode执行，仅在test进程限制每套8个policy动作；计数实际env.step返回，断言7D有限动作及原生输入合同，写smoke-action-evidence.json。生产评测步数不变。本地AST及diff检查通过，远端真实运行待依赖完成及源码同步。
- Isaac诊断213730已exit0，完成启动和10次update；新增显式日志/关闭crash reporter后不再首秒崩溃，不能据此证明根因或相机正确。明确缺libGLU.so.1导致材质库无法加载；229786已将libglu1-mesa9.0.2/libopengl0包解到独立envs/robodojo-system-libs-20261014，下一次渲染将显式加此LD_LIBRARY_PATH，未更换驱动或渲染配置。
- LingBot官方torch栈已完成。官方requirements因LeRobot0.3.3声明torch<2.8/torchvision<0.23而与LingBot显式2.9/.24冲突，安装器确切报错已保留。新链239461用仅torch2.9/torchvision.24的显式resolver overrides保留LingBot原版版本，继续其余官方依赖；记录metadata冲突，pip check不声称全绿，后续须真实reader/训练验收。日志lingbot-va-official-requirements-v2.log。
原版LIBERO短闭环首轮在import阶段因CUDA_VISIBLE_DEVICES只含0而EGL指定1被robosuite拒绝，未执行动作；第二轮改为可见0,1、模型cuda:0、EGL1后import通过，但test wrapper误用了main函数名。现已按官方真实入口eval_single_process修正并静态核验符号；这两轮不计验收。旧日志保留，下一轮使用新输出目录。评测依赖另补matplotlib后已完成。LingBot官方依赖主体安装完成，flash-attn2.8.3.post2正构建。

## 15:08 原版 FastWAM LIBERO 闭环通过与用户补充

用户澄清“缩短测试”不是指定固定1–2训练步或8动作；实际要求是用足够短的测试证明代码跑通并搭好环境、资产，配置来源不变。当前已完成的步数是本轮具体smoke取值，不是所有模型的强制标准，不为凑步数重复测试。

- FastWAM LIBERO v3因Hydra从导入模块寻找configs包而失败，显式--config-path到已推送repo/configs后解决；dry配置解析成功输出，测试末尾无episode断言仅因dry模式而触发，不是配置解析失败。
- v4链257972已自然结束：四套Spatial/Object/Goal/Long各真实8步，原版native evaluator，全部client exit0。结合原生训练与checkpoint审计，fastwam/libero-native-closed-loop-v4/smoke-acceptance.json已验收。当前3/9通过：OpenWAM LIBERO、RoboTwin、FastWAM LIBERO；另FastWAM RoboTwin历史完成待配置复核。退出阶段EGL析构有ignored警告，动作和渲染在执行阶段成功，保留此限制，不额外扩大修改。
- LingBot base下载186492已完成，固定68b7bc1；官方LIBERO下载220180因latents/.DS_Store返回403反复退避。确认进程后仅SIGINT它，保留文件，使用相同revision及目标目录、ignore_patterns仅**/.DS_Store续传257971。非数据文件排除不修改样本范围或数据内容。
- RoboDojo全量Assets下载65528仍在Material阶段，尚未到机器人与物体。确认并停止该进程，优先下载当前pour_liquid_into_cup所需x5/Room/Sensor/Geometry及wuliangye/mug/goblet/Fluid，再继续原全量Assets；新PID270126，日志robodojo-assets-prioritized.log，复用缓存且无删除。优先阶段2297文件，尚未完成。
- RoboDojo真实相机诊断链275772等待FastWAM闭环释放GPU后运行，使用已补GLU私有prefix、原生Isaac Camera及红色立方体场景，检查128x128 RGB有限/非零方差，输出robodojo-camera-smoke.log/json/png；此项只证明渲染，不代替任务policy闭环。当前正在启动，未宣称通过。
- LingBot flash-attn编译239461经ps确认ninja/nvcc/cicc实际活跃，非下载假死，仍待完成；数据和模型环境相互独立。
## 15:20 渲染验证与 LingBot 原生资产推进

- 用户最新确认继续测试；测试长度由足够证明代码跑通决定，不固定1–2步，不启动正式训练或StereoTok。保持三模型独立配置归属。
- RoboDojo首个相机probe先报空图像shape(0,)，随后清理阶段segfault；不能把后续segfault误作首个根因。私有系统库prefix补齐libXt/libSM/libICE，不修改系统驱动。第二次原生Camera通过world.play及rep.orchestrator.step(rt_subframes=4)显式推进渲染，得到128x128x4、值域112–228、std55.1835的真实图像；PID283109已结束，日志robodojo-camera-smoke-v2.log、证据robodojo-camera-smoke-v2.json/png。此项仅证明渲染，不计任务策略闭环通过。
- LingBot LIBERO下载257971停在HfApi.repo_info未指定timeout的metadata请求，确认后定向停止。新288304仅将metadata timeout设为30秒，其余revision/文件范围不变，继续原目录断点下载；15:17已推进856/2506文件。日志lingbot-va-libero-native-download-v3.log，尚未宣称数据完整。
- LingBot flash-attn239461继续实际编译，15:19 ninja日志已有70条目标完成记录。RoboDojo优先资产270126继续下载，15:17为841/2297文件；文件比例不代表字节或完整准备比例。
- 本轮RAM frank205455090823393499与DSW Running/SSH公网映射再次核验，命令实际到达远端；当前完整验收仍3/9，其余继续推进。
新增tests/run_lingbot_libero_smoke.py：只导入固定SHA的官方LingBot原生训练入口；复制其LIBERO配置，仅覆盖本地路径、测试时长、保存频率、worker数和smoke W&B开关。保留B1/GA10、原始优化器与动作/latent合同，训练前逐episode检查所有发布的双相机latent窗口，缺失立即失败，不容许reader静默筛样本。默认3个optimizer更新是本次短测试选择，不是通用验收要求；双卡与官方8卡的全局batch差异只用于运行门禁，不作性能复现。AST与git diff检查通过，真实训练待下载和环境终态。

15:24：LingBot flash-attn构建/安装完成，官方模型依赖imports及原生train import均通过。pip check仍明确记录LeRobot对torch/vision的两条metadata冲突，未宣称全绿。官方VA_Server._get_t5_prompt_embeds方法以AST直接提取执行（未改其实现），使用LingBot base自己的T5生成empty_emb.pt，shape512x4096/bfloat16/全部finite；日志lingbot-official-empty-embedding.log。

测试包装78a55c5已推送且远端clean clone同步。启动原生LingBot LIBERO链PID298690，日志lingbot-native-libero-chain-v1.log；先等待288304完整下载成功标记，再使用2卡、原版B1/GA10、3次optimizer更新和末次checkpoint保存。输出lingbot/libero-native-v1。一次健康检查确认链处于WAITING_FOR_OFFICIAL_DATA_DOWNLOAD，尚未开始训练，不算验收通过。下载进程与RoboDojo资产进程均活跃。首次本地push遇TLS握手失败，幂等重试成功；未旁路复制源码。

## 15:31 LingBot RoboTwin 原生任务数据与 RoboDojo 动作计数准备

- LingBot RoboTwin官方dataset robbyant/robotwin-clean-and-aug-lerobot固定595d4acc7e026aa57afba218f4170c6d79e39103，全量API usedStorage443290849720bytes；不为smoke下载全量。选adjust_bottle官方完整clean50子集355文件695733602bytes、aug500子集3503文件7504776917bytes，保留clean+aug语义、原生latent和元数据。它们是代码smoke输入，不宣称全任务数据齐备。
- HuggingFace repo_info siblings返回100000文件上限，未涵盖clean路径；不把截断列表当完整manifest。改用本机官方HF tree API对两个精确任务前缀分页读取，保存lingbot-robotwin-adjust-bottle-595d4ac-manifest.json；远端mirror树查询403/超时未造成数据变更。按固定manifest逐文件hf_hub_download并核对size，PID303771，日志lingbot-robotwin-adjust-bottle-download.log，目标datasets/LingBot-RoboTwin-adjust-bottle-595d4ac。一次健康检查确认已完成101/3858文件，仍在下载。
- 新增tests/run_robodojo_policy_smoke.py，仅包装固定09bd5f5的官方main/create_eval_env；保持真实任务、观测、原生策略调用与take_action_batch，临时缩短step_lim，只有原生动作调用返回后才记录实际计数。证据明确区分短动作闭环与benchmark分数，生产代码不改。AST检查通过；尚未远端执行，需等待LingBot双卡训练释放和RoboDojo资产就绪。
- OpenWAM对FastWAM两项复现的精确配置来源仍未找到；搜索结果中另有同名Stanford OpenWAM，不能混用。只认OpenWAM-Official/OpenWAM固定4d7f642及对应2609.07398论文。已有XPolicyLab FastWAM joint训练脚本本身并不能证明OpenWAM采用了相同参数。

## 15:36 LingBot 原生 reader 故障定向修复

- 官方LIBERO数据已完整下载2506个文件（只排除非数据.DS_Store），训练门禁确认500episodes/500完整双相机窗口。首轮298690已结束exit1，真实模型加载和FSDP初始化后，native reader遇ValueError: Feature type List not found。原始parquet的action/state Arrow metadata确为List；LeRobot0.3.3要求datasets<=3.6.0，与官方发布数据不兼容。
- 已确认首轮及GPU进程退出后，定向安装datasets4.0.0，保留数据文件、模型/优化器/动作/latent配置不变；同时补RoboDojo官方转换器必需h5py3.14.0，不改numpy/torch。日志lingbot-datasets4-reader-repair.log：Dataset.from_parquet读出388x7动作，与PyArrow原始列逐元素完全相等。pip check明确保留3条LeRobot metadata冲突（datasets、torch、torchvision），不能声称依赖声明全绿。仍需native训练验收。
- 新增tests/serve_lingbot_baseline.py，按显式benchmark选择官方配置，仅注入已审计权重组件装配路径与本机监听地址。新增tests/run_lingbot_libero_policy_smoke.py，复用官方Long客户端，计数真实7D有限policy动作，达到短测试上限后正常终止测试，不伪造done或成功率。两者尚未实际执行；AST检查通过。
- 原生训练wrapper补齐官方main调用的init_logger，确保加载、训练和checkpoint日志完整；下一轮使用新输出目录。基础权重中的patch_embedding两项未使用与官方当前实现使用patch_embedding_mlp相符，不将其静默替换为FastWAM结构。

## 15:42 原生 reader 通过，修正历史服务入口歧义

- LingBot native reader真实读取500窗口/138090frames，样本latents(48,97,8,16)、text(512,4096)、actions/mask(30,97,4,1)均finite，日志lingbot-native-reader-v2.log。独立LingBot环境已补齐LIBERO仿真依赖，torch2.9/numpy1.26.4/datasets4版本断言保持；日志lingbot-libero-eval-dependencies.log。下一轮训练等当前OpenWAM RoboDojo链释放GPU后开始。
- RoboDojo优先Assets阶段完整结束，继续原全量资产；repo/Assets仅链接到自己固定revision资产目录。OpenWAM首轮315256、二轮317105已终止，都在import阶段失败，没有加载checkpoint、没有动作证据。检查发现历史tests/serve_openwam_baseline.py本质上是FastWAM旧reader适配服务，不能用作OpenWAM模型。为防入口混淆，定向改名tests/serve_fastwam_openwam_reader_baseline.py并在说明中标明不计原生对齐验收；无生产逻辑改动。
- 当前第三轮319155使用真正的OpenWAM官方scripts/deploy.py与该模型自己的safetensors checkpoint，单独bridge连接原生RoboDojo客户端。输出openwam/robodojo-closed-loop-v3，主日志openwam-robodojo-closed-loop-v3.log；启动脚本openwam-robodojo-chain-v3.py保存到run目录。按XPolicyLab OpenWAM适配约定关闭compile和DiT cache，保持原始10步同步denoise与32动作chunk。尚待服务/场景/动作检查，不能计入通过。

15:44：OpenWAM RoboDojo第三轮319155/服务319156仍实际活跃，官方loader已输出Architecture dual_system_self_attn并加载对应safetensors；尚未ready、尚未启动simulator，不把模型装载等同动作闭环。LingBot重试链321035已启动并健康检查，日志lingbot-native-libero-chain-v2.log，明确等待319155及其finally清理释放GPU，再用同一官方配置运行3个optimizer更新，输出lingbot/libero-native-v2。远端源码暂保持8c5cf603fac54ec066b042d693a321ffefd8af69，不在活跃任务期间pull最新仅入口改名/文档提交。当前完整验收仍3/9。

## 15:47 真实训练与部署阻塞修复

- OpenWAM RoboDojo v3使用官方模型入口，加载架构后因cv2缺libGLX.so.0退出；不是checkpoint结构错误。v4将既有system-libs私有prefix加入policy/bridge启动LD_LIBRARY_PATH，复用原模型和配置，不安装或改系统驱动。新链PID见本轮工具回执，日志openwam-robodojo-closed-loop-v4.log。
- LingBot v2在FSDP后构建dataset Pool时，两个子进程收到SIGSEGV，栈位于torch libgomp/GOMP_parallel；父进程Pool仍等结果，不能把它当训练继续。只终止321035及经ps父子关系确认的259个本作业后代。CPU独立reader已通过，故定向将测试进程的multiprocessing改为spawn、native dataset初始化worker由默认128降到2；DataLoader worker仍2，模型/数据/损失/优化器/GA10不改，测试合同明确记录这项运行资源覆盖。原版源码不修改。

## 15:59 LingBot 原生训练与 checkpoint 通过、后续真实闭环

- LingBot LIBERO原生v3在spawn/2初始化worker下完成3次optimizer更新，保持B1/GA10和官方优化参数。video/action losses依次0.1667/0.4615、0.1591/0.4570、0.1540/0.4972，grad_norm3.17/3.26/3.33；15:51:55保存step3并exit0。输出lingbot/libero-native-v3；checkpoint audit对官方模型meta实例逐key/shape核对839 tensors，全部finite，10177831668bytes。官方保存为weights-only，无optimizer state；证据checkpoint-smoke-audit.json及lingbot-libero-checkpoint-audit.log。还未通过仿真，不能计入第四项完成。
- LingBot RoboTwin官方clean任务原生reader通过50episodes/50窗口，样本latents(48,9,24,20)、text(512,4096)、actions(30,9,16,1)有限；aug仍下载。代码另建RoboTwin-lingbot-baselines-20261014，按LingBot README固定2eeec322d95799f537cbfe5f291a8220d965ccb8，clean detached HEAD，不复用OpenWAM的0aeea2d源码版本。
- LingBot RoboDojo来源转换325312通过：官方XPolicyLab converter从两条H5生成375frames/14D joint。新目标datasets/LingBot-RoboDojo-source-v21-35efbc7，未覆盖已有数据。递归删除在实验进程内被禁止，保留生成的images；退出时TemporaryDirectory清理因此有ignored异常，未执行删除。官方process_data.py链330634随后exit0，输出datasets/LingBot-RoboDojo-latents-35efbc7：两条轨迹176/199frames，三相机latent分别22/25frames，独立empty_emb.pt与30D映射。其官方默认target_fps10在原始25fps下使用round stride2，实际采样12.5fps；须继续核对与robotwin30配置action_per_frame12的训练/推理关系，不能静默修改采样或宣称完整验收。
- OpenWAM RoboDojo v4服务和bridge已成功ready，simulator在argparse阶段把--device误作--device_id的缩写而退出；v5移除此多余参数后进入真实场景，但缺生成的Assets/Robots/x5/curobo.yml。虽然Isaac退出码为0，异常栈及无action-evidence证明不通过。按官方utils/update_embodiment_config_path.py规则，仅从x5/curobo_tmp.yml替换本机资产根生成该文件（独占新建），保留原模板和SHA256，证据robodojo-x5-curobo-path-binding.json。
- OpenWAM v6链355736再次运行，日志openwam-robodojo-closed-loop-v6.log，除exit0外明确要求action-evidence中真实动作>0以及最终短闭环通过标记，避免Isaac吞退出码误计通过。尚未验收。
- LingBot LIBERO闭环v1的官方服务成功ready，客户端缺官方LIBERO依赖future而失败；已补future0.18.2，原生client import实测通过。v2链359086等待OpenWAM v6释放GPU后执行，输出lingbot/libero-native-closed-loop-v2，使用审计后的step3 transformer与官方base VAE/text/tokenizer装配。v1 server的SIGTERM是失败后父链清理，不是模型自身失败。当前完整验收仍3/9。

## 16:03 LingBot LIBERO 验收通过（4/9）

- LingBot LIBERO闭环v2实际通过：官方client task0执行8个7D有限policy动作，client exit0，LINGBOT_NATIVE_LIBERO_POLICY_SMOKE_PASS 8。已核对inference-assembly的transformer真实指向审计过的step3，VAE/text/tokenizer来自同一官方base。验收文件lingbot/libero-native-closed-loop-v2/smoke-acceptance.json，非论文分数。当前通过项为OpenWAM LIBERO/RoboTwin、FastWAM原版LIBERO、LingBot官方LIBERO-Long，共4/9；FastWAM RoboTwin仍另列历史完成待来源复核。
- OpenWAM RoboDojo v6缺Assets/Eval_Layout/RoboDojo/arx_x5/42，原启动错误地将训练seed42用于场景索引。官方SeedManager默认0，发布布局目录只有0/1/2；v7明确选布局seed0，训练seed42不改，并写launch-contract.json分别记录。没有复制布局伪造seed42。v6退出0但缺action-evidence被门禁拒绝。v7链361002在LingBot闭环释放GPU后继续，日志openwam-robodojo-closed-loop-v7.log，仍待实际动作验收。

## 16:10 继续最小代码闭环验收

- 用户要求缩短测试而非固定 1–2 步；当前已验收仍为 4/9，不重复已通过项。
- RoboDojo 团队 XPolicyLab 53c5da5 的 LingBot 原生 reader 已读取两条完整轨迹：latents 分别 [48,22,16,48] / [48,25,16,48]，actions 为 [30,22,8,1] / [30,25,8,1]，所有 tensor 有限。证据：H800 runs/five-baselines-20261014/lingbot-robodojo-native-reader.json。
- 新增 tests/run_lingbot_robodojo_smoke.py 仅作为该适配的训练测试入口，固定来源 SHA，保留原配置 B1/GA8、初始化和动作映射，仅覆盖路径、测试长度、保存周期、worker 和 W&B。沿用已验证 spawn/2 个数据初始化 worker。
- 上游转换 target_fps=10 在原始 25fps 下取整 stride=2，训练实际每 latent 8 个动作；发布推理配置 action_per_frame=12。原值分别保留并显式写入 contract，不宣称时序一致或分数复现。
- OpenWAM RoboDojo v7 仍运行，GPU1 有真实负载，场景完成设置并重置；没有动作证据前不计通过。官方 Dockerfile 的单 Vulkan ICD 建议已阅读，当前已有显式 VK_ICD_FILENAMES；未为等待场景启动擅自更改物理配置。
- 本地 schannel fetch 握手失败，通过单次 git -c http.sslBackend=openssl fetch origin 成功；未关闭证书校验，未改全局 Git 配置。目标远端仍 91a33f0。
- 新训练入口仅完成语法与 diff 检查，尚未运行；下一步待当前 RoboDojo 闭环释放 GPU 后启动。

- 16:12 已推送 eb15e88；H800 LingBot RoboDojo 短训练链 PID 371228 已启动，等待 OpenWAM v7 PID 361002 释放 GPU 后核验 clean、fetch/ff-only 并要求 runner=eb15e88，再以两卡运行原生训练入口。日志 /mnt/workdir/frank/runs/five-baselines-20261014/lingbot-robodojo-native-train-v1.log，输出 lingbot/robodojo-native-v1；当前是等待状态，不计训练通过。

## 16:13 RoboDojo ffmpeg 根因修复
- 全日志定位 v7 反复重置直接根因 FileNotFoundError: ffmpeg，发生在 get_obs_batch 的录像入口，尚无策略动作。官方循环捕获异常继续布局种子，因此不能用尾部场景日志判断推进成功。向核验过命令行的 simulator PID 363280 发送 SIGTERM，保留日志并让原链清理服务。
- 既有私有 system-libs/prefix/usr/bin/ffmpeg 已存在，需同时加入 bin PATH 和 usr/lib/x86_64-linux-gnu/pulseaudio 的 LD_LIBRARY_PATH；完整环境下 ffmpeg -version exit0，6.1.1-3ubuntu5。没有安装替代模型依赖或更改物理参数。
- 新运行链 openwam-robodojo-chain-v8.py PID 372981 等待 LingBot RoboDojo 训练 PID 371228 后再启动，输出 openwam/robodojo-closed-loop-v8，日志 openwam-robodojo-closed-loop-v8.log；同时按官方 Dockerfile 将 VK_DRIVER_FILES 与既有 VK_ICD_FILENAMES 绑定到同一驱动 ICD。未获得真实动作前不计通过。

## 16:15 LingBot RoboDojo 训练启动及 RoboTwin 入口
- 原 RoboDojo v7 在自身 600 秒超时及 finally 清理后结束，保留其 ffmpeg 首因，不将 timeout 当新根因。LingBot RoboDojo 链已完成 clean fetch/ff-only，实际 runner eb15e88735bb2e7b497c60ee38e20ba2a00d92cc，打印 NATIVE_ROBODOJO_TRAIN_START，两卡执行中。
- 新增独立 tests/run_lingbot_robotwin_smoke.py，使用原作者 robotwin_train（B1/GA1、30D、action_per_frame16、robotwin_tshape），仅路径与测试运行长度/保存/worker/W&B 覆盖。启动前逐项检查已选 adjust_bottle clean50+aug500 的 metadata、Parquet 和三个相机所有窗口，避免 reader 静默跳过；记录固定 HF revision 和任务子集范围。现仅 AST 检查通过，数据仍下载中，尚未启动 RoboTwin 训练。

## 16:17 LingBot RoboTwin 仿真资产和环境隔离
- 原作者 README 指定的 RoboTwin 2eeec322 clean clone 已 fetch 核验；按其 assets/_download.py 指向的 TianxingChen/RoboTwin2.0，绑定 Frank 既有 RoboTwin-bf44be5 的 background_texture/embodiments/objects，只新增三个链接，未改资产。绑定证据 runs/five-baselines-20261014/lingbot-robotwin-asset-bindings.json。
- 仿真既有环境核心版本与 LingBot README 一致：sapien3.0.0b1/scipy1.10.1/mplib0.2.1/gymnasium0.29.1/trimesh4.4.3/open3d0.18/imageio2.34.2。单独 conda clone 到 robotwin-lingbot-baselines-20261014，补 huggingface_hub0.36.2/msgpack1.1.1/websockets15.0.1（原客户端使用 additional_headers/ping_interval，旧13.1不可直接复用）。准备进程 PID378820，日志 lingbot-robotwin-sim-env-setup.log，尚待 import 验证。不变更 OpenWAM 环境。
- LingBot RoboDojo 首个更新前正在 TorchInductor 编译，已观察到加载生成 .so 的活跃子进程，不依据 GPU 瞬时0%将其判为停滞。

## 16:19 LingBot 首个有效更新及 RoboTwin 原生客户端
- LingBot RoboDojo 两卡训练完成第一个 optimizer update，video loss0.2004/action loss0.3017/grad norm2.74，首次更新耗时181.91秒含编译；未保存最终 checkpoint 前不计训练门禁通过。
- 添加 tests/run_lingbot_robotwin_policy_smoke.py，固定作者模型/仿真 SHA，只替换官方客户端的安装路径占位符，保留专家筛种、prompt、动作转换和 take_action；只在原生服务返回 action 后统计已完成16D有限动作，达到短测试界限退出并清理环境，不产生虚假任务成功率。AST通过，尚未实跑。

## 16:20 LingBot RoboDojo 训练门禁进展
- 原生短训练 NATIVE_ROBODOJO_TRAIN_EXIT 0；两个 optimizer 更新分别 video/action loss0.2004/0.3017、0.1841/0.2651，grad norm2.74/2.65。保存路径 lingbot/robodojo-native-v1/checkpoints/checkpoint_step_2/transformer，正在 CPU 上对固定 XPolicyLab 原生模型逐键/形状/有限值审计。没有 optimizer 状态，不宣称完整续训。
- OpenWAM RoboDojo v8 自动开始，policy PID381447；后续真实动作未通过前不增加验收数。

## 16:22 LingBot RoboDojo checkpoint 审计通过
- checkpoint-smoke-audit.json 已生成：step2 transformer 10,177,831,668 bytes，839 keys，与固定 XPolicyLab 原生模型逐键/形状一致，所有 tensor 有限。训练和 checkpoint 两项通过，仿真尚未通过。
- 添加单独 serve_lingbot_robodojo_baseline.py，固定 XPolicyLab53c5da5与 robotwin30_train，保留30D/action_per_frame12；仅绑定组装 checkpoint 路径及监听端口，不借用作者 LIBERO/RoboTwin 配置。官方单环境 take_action 内部调用 take_action_batch，现有 RoboDojo 真实动作计数包装可直接覆盖，无需修改仿真步进逻辑。

- LingBot XPolicyLab 转发服务导入定位缺少 msgpack_numpy；原 env 不带 pip（命令真实到远端失败），使用既有 uv 向该已结束训练的独立环境安装 msgpack-numpy0.4.8 --no-deps 成功，未改变模型依赖。下一步复核转发服务导入后启动该项仿真。

## 16:25 当前验收更新为 5/9：OpenWAM RoboDojo 通过
- v8 已由真实策略驱动仿真8个动作，action-evidence.json计数8，原生任务正常完成、录像9帧、SIMULATOR_EXIT0，完整 simulator.log 无 Traceback/unhandled exception。两步训练不代表任务成功率；本次0/1只是流程结果。
- 外层最后成功打印未出现导致 AssertionError：上游 main 最后 simulation_app.close() 可直接终止进程，包装器返回后的打印不会执行。独立核验动作、原生完成标记和退出码后生成 openwam/robodojo-closed-loop-v8/smoke-acceptance.json，保留原始错误日志，不修改证据。
- 测试包装将成功动作标记放到实际 take_action_batch 返回且计数达标后输出，防止关闭阶段吞掉标记；无动作仍不通过。不为此重跑已充分验证的 OpenWAM。
- 当前完整通过：OpenWAM LIBERO/RoboTwin/RoboDojo、FastWAM原版LIBERO、LingBot原版LIBERO-Long。LingBot RoboDojo仅训练+checkpoint通过。闭环队列PID387816，日志 lingbot-robodojo-closed-loop-v1.log，输出 lingbot/robodojo-closed-loop-v1；固定RoboDojo适配与joint动作，组装训练step2权重，推理独立config副本attn_mode=torch不修改训练checkpoint。

## 16:29 LingBot RoboDojo 闭环重启及 RoboTwin 训练队列
- LingBot闭环v1在git fetch阶段120秒超时退出，未启动模型。只读确认进程已结束后，单次重试fetch成功，clean clone fast-forward到8eefc5b5637e84541f286a5f42e78ecdc7c94f37。v2在新输出目录启动PID390828，后端390831、转发391396；保留v1现场。
- RoboTwin训练链PID391559，日志 lingbot-robotwin-native-train-v1.log，等待数据下载303771完成且LingBot闭环390828释放GPU，要求下载完成标记和manifest所有3858文件大小一致、clean runner严格8eefc5b；之后才执行官方robotwin_train短训练，不将排队视为通过。
- 环境复制PID378821仍有磁盘I/O，已写约5.9GB；未因日志安静擅自重启。

## 16:30 当前验收 6/9：LingBot RoboDojo 完整通过
- v2真实执行8个策略动作，原生仿真exit0，外层REAL_POLICY_ACTION_EVIDENCE_PRESENT8，simulator.log无异常回溯；训练exit0、839键全有限审计以及实际assembly权重指向均验证。证据 lingbot/robodojo-closed-loop-v2/smoke-acceptance.json。服务的SIGTERM来自链finally正常清理，不是推理崩溃。上游train8/infer12时序差异仍如实记录，本项仅证明已发布适配代码链路，不宣称论文成绩或时序一致。
- RoboTwin conda clone offline已在拷贝untracked文件后、安装Conda包阶段失败，缺包需访问repo.anaconda.com。检查conda clone_env源码确认执行阶段；目标尚无conda-meta。使用源环境PrefixData/PrefixGraph精确URL列表继续explicit安装阶段，避免重复复制约5.9GB或删除不完整目录。恢复PID395956，日志lingbot-robotwin-sim-env-resume.log，仍需完成后import检查。
- 再次搜索固定OpenWAM仓库完整tracked源码及官方训练文档，仍无明确FastWAM clean-only/RoboDojo复现实验recipe，不能用OpenWAM模型训练配置代替。该来源门禁未通过。

## 16:33 RoboTwin reader 和客户端导入前置核验
- 实际安装的 LeRobot get_episodes_file_paths 同时要求 Parquet 和各 camera video 存在，不能仅凭 latent 下载完就提前通过原生reader；保持3858文件完整性要求，继续原下载303771。
- 独立环境恢复395956仍活跃并下载源环境精确Conda包；原生客户端导入检查排队PID397665，等待恢复明确成功后执行，仅绑定作者安装路径、验证import，不启动额外仿真/GPU。日志 lingbot-robotwin-native-client-import.log。

## 16:37 LingBot RoboTwin 环境安装完成与闭环队列
- 精确Conda包恢复和客户端依赖安装完成；首次Sapien import因未传已有EGL/Vulkan路径失败，未修改驱动，使用私有LD_LIBRARY_PATH/VK_ICD_FILENAMES/__EGL_VENDOR_LIBRARY_FILENAMES后，原生客户端import通过。渲染检查PID400803，日志lingbot-robotwin-native-client-import-v2.log，检查真实Sapien scene.step，不等同策略闭环。
- 训练后checkpoint审计与原生闭环队列PID401897，日志lingbot-robotwin-closed-loop-v1.log，等待训练391559 exit0及原生import/render标记后，逐键/形状/有限值审计step3，再用原作者robotwin配置部署并运行原生客户端；固定runner8eefc5b，独立输出lingbot/robotwin-native-closed-loop-v1。仍未计验收通过。

## 16:37 实时等待与未公开配置来源
- 当前下载303771、训练队列391559、闭环队列401897均live；RoboTwin已完成约2251/3858个文件，网络失败按原脚本重试，未替换数据或减少任务覆盖。RoboDojo全资产270126仍live，优先任务资产已满足两项闭环，全量约10706/15365文件，网络退避中。
- 实核DSW仍Running，关机DueTime09:34:17Z（17:34:17+08），剩余约57分钟；本轮未续期。
- FastWAM两项需要OpenWAM实际复现recipe来源；公开通用训练配置、榜单数值和RoboDojo团队默认脚本不足以证明该来源。请求用户提供来源或明确变更为独立有据的工程baseline前，不擅自替代配置。其他已授权LingBot RoboTwin工作继续。

## 16:39 精确清单视频预取加速
- 原下载303771确认仍live，进度2301/3858。2251之后清单约1.43GB，其中视频约0.992GB、latent约0.435GB；无数据范围变更。
- 新增运行期预取PID404373，仅固定595d4acc revision、既有manifest的aug视频1500文件，4线程hf_hub_download到同一授权数据目录；使用Hub自带文件锁/原子落盘，每文件校验发布size，保留原串行下载和最终全3858文件门禁。不自行替换缺失视频、跳过样本或提前启动训练。日志lingbot-robotwin-video-prefetch.log。

- 预取首轮遗漏原下载环境HF_ENDPOINT，huggingface.co网络不可达；读取303771环境确认其使用hf-mirror.com及HF_HUB_DISABLE_XET=1。停止仅此错误入口预取404373，以相同精确revision/manifest恢复镜像入口（v2日志），原下载未中断。已有文件保留且仍做size校验。

## 16:47 下载限流与退避
- 原RoboTwin下载已进入最后cam_right_wrist视频（约episode217），服务返回HTTP429，提示共享resolver五分钟配额。停止额外预取406225；非关键全量RoboDojo资产270126暂SIGSTOP，恢复进程412221将在6分钟退避后SIGCONT（asset-rate-backoff.log）。原RoboTwin下载303771保留其重试退避，不换身份/绕过限制、不跳过文件。训练391559和闭环401897继续等待真实完成。

### 16:59 LingBot RoboTwin 发布 latent 完整性阻断
- 实时确认 DSW 仍 Running，RAM frank；训练链与闭环链均已退出，GPU 空闲。
- 官方所选 clean+aug 清单 3858 个文件已全部下载并通过文件大小检查。但 episode metadata 要求的两份 latent 不在发布清单中：aug episode 000003 左腕（0–149）、episode 000498 右腕（0–133）。其余相机及原始视频存在。此为发布数据缺项，不是下载未完成。
- 原版 `wan_va/dataset/lerobot_latent_dataset.py::_check_meta/parse_meta` 会过滤缺少任何相机 latent 的窗口；因此原版实际会读 clean 50 + aug 498。本任务严格检查在训练前报错，保留了此差异，未放宽检查、未删样本、未自行重编码。
- 日志：`/mnt/workdir/frank/runs/five-baselines-20261014/lingbot-robotwin-native-train-v1.log`，首个根因是左腕 latent 缺失；train exit 1。closed-loop-v1 正确因无 train exit 0 而拒绝继续。
- 待确认：是否采用原版 reader 的过滤行为并显式记录 548/550 窗口，或要求重新生成缺失 latent（原作者当前源码未提供相应离线预处理入口，需额外核对处理合同）。当前通过仍 6/9。
- RoboDojo 全量资产下载也已因 HF mirror HTTP 429 退出；优先 smoke 所需资产已齐备，已通过验收不受影响。完整资产不得标记完成。DSW 原自动关机时间仍为 17:34，未延长。

- 17:00 在确认原下载 PID270126 已终止后，以相同 RoboDojo revision、Assets/** 范围、原目标和单 worker 恢复 snapshot_download，PID424133，日志 robodojo-assets-resume-v2.log；复用缓存，无覆盖原资产、无删除。LingBot RoboTwin 仍等待数据过滤确认，未重启训练。

- 17:02 续传 PID424133 已终止，日志再次记录 HTTP429 和 LocalEntryNotFoundError，无全量完成标记。停止立即重复请求，保留缓存。仍为6/9；LingBot RoboTwin 等待原生548/550窗口筛选确认，FastWAM RoboTwin clean-only/RoboDojo 等待实际OpenWAM复现配置来源。未继续训练、未改合同。

### 18:10 用户纠正与实时资源查询
- 用户明确 FastWAM RoboTwin 沿用已完成、已对齐 ruhu 的既有实验；本轮未重新审计该训练，将其计为历史完成，合计7/9。
- LingBot RoboTwin 按原论文/仓库执行已是用户明确授权。此前额外要求确认原生reader过滤属于过严门禁，撤销该人为阻塞；后续应显式报告官方reader的548/550有效窗口，不伪造缺项、不补造latent。尚未修改或重跑测试，不计通过。
- RAM身份再次验证frank；GetInstance实时返回Stopped，InstanceShutdownTimer到期09:34:30Z（北京时间17:34:30）。此刻无可用SSH映射，未启动实例；无训练或下载正在继续。
- RoboDojo全量资产最后已核验续传进度11174/15365，约72.7%文件数；之后HTTP429退出。此数不是字节进度或本轮实时目录统计。资产总清单41,269,513,111 bytes；不能据文件比例推算已下载GB。优先smoke资产2297文件已完成。
- 发布数据准备范围仍为smoke：OpenWAM LIBERO全发布版本、FastWAM LIBERO四套原生版本、LingBot LIBERO-Long所选完整版本、LingBot RoboTwin adjust_bottle clean50+aug500发布文件已齐；OpenWAM RoboTwin/RoboDojo及LingBot RoboDojo只准备既定少量轨迹和派生缓存，不能声称全任务训练数据已齐。
- 重新读取OpenWAM官方RoboDojo README，Fast-WAM平均SR2.03/Sc3.48；XPolicyLab公开train.sh存在默认B8/GA1/epochs16及原task配置，故不是没有任何配置，而是尚无证据把该默认配置绑定到OpenWAM报告分数。

### 18:37 恢复授权与配置来源纠正
- 用户明确同意LingBot RoboTwin按原生548/550有效窗口smoke，全量数据准备留到后续。测试入口仅允许固定revision已核实的两个缺项，记录被原生reader排除的窗口；任何其他缺项仍报错。不修改原版reader或数据。
- FastWAM RoboDojo配置来源改为RoboDojo论文附录K和RoboDojo团队XPolicyLab适配。已直接读取论文确认Wan2.2-TI2V-5B初始化、batch256、20K训练预算；仅短测试，不启动正式20K训练。撤销此前缺OpenWAM复现配置的阻塞。
- 用户授权恢复全量Assets下载，不把Assets包当作训练轨迹。RoboTwin既有2500条优先复用，后续另核各模型转换/latent缓存，不重复下载原始数据。
- 实核RAM frank和当前角色可用quota1bybv9m0ixu；复用原2GPU/48CPU/256Gi/priority4实例，StartInstance返回成功，RequestId01A139FC-58B9-5F48-9228-AE61474154CA。等待实际启动/SSH；尚未恢复训练或下载。

### 18:42 FastWAM RoboDojo 原生测试入口
- 从本地XPolicyLab git对象直接读取固定53c5da5版本train.sh/model.py/deploy.py及vendored训练入口；没有修改该仓库，运行仍使用远端已固定同一SHA。
- 新增tests/run_fastwam_robodojo_smoke.py：调用该版本原生run_training，joint14、arx_x5、384x320三相机、text128、lr1e-4、weight_decay1e-2，Wan初始化，不使用OpenWAM reader或策略checkpoint。
- 记录论文batch256/20K与脚本默认每卡B8/GA1/epochs16。smoke明确B2/GA1、短steps/save、workers2/eval0/W&B禁用，正式预算未启动。要求相应独立数据、stats、T5缓存、Wan派生ActionDiT路径，拒绝源码tracked dirty或不同SHA。
- AST通过；本机无GPU模型依赖，尚无真实Hydra/reader/训练证据。DSW持续Queuing，无SSH地址；没有重复申请实例。
