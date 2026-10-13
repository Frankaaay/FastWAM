# ruhu C2R baseline + matched StereoTok：H800迁移准备

## 授权与当前状态

2026-10-13用户授权：复用H800/L20Y已有2461条fresh左右目；Wan对齐ruhu run，StereoTok仅更换VAE及必需双目输入；共同EEF20/minmax/seen文本/增强/5epoch/global256/每卡BS8/GA4，各8卡；W&B yichx14-uc-irvine/StereoTok；每epoch一个checkpoint共5个，不删除旧产物。本轮目标为代码适配、预处理和smoke，未启动正式训练。

本地cwd最初C:/Project/StereoTok。保留C:/Project/FastWAM原f86b842分支及原未跟踪文件。新独立目录C:/Project/FastWAM-C2R，origin官方yuantianyuan01/FastWAM，detached HEAD=7faa71108368fbb3b6885649f112af607427a2d4，fetch成功。未创建分支、commit/push。H800个人基础clone /mnt/workdir/frank/projects/FastWAM-C2R-20261013 同SHA且clean，尚不含本地适配。

## 来源与修改

只读导入 /gpfs/jiuquyun/projects/ruhu/fastwam-c2r-20261011 内fastwam_c2r及准备/训练/缓存脚本与配置；原文件哈希见configs/c2r/source_hashes.json（记录导入前内容，manifest原文件存本地请求目录和H800 prep目录，不入Git）。StereoTok冻结student实现来自我方f86b842，原student-v3/Stage2-u8000合同保持。

- runtime/helpers loader/FastWAM：添加codec选择、训练video_right、推理input_image_right、权重tokenizer绑定。Wan保留整batch CUDA Graph编码和tensor joint core，compile_training_denoise=false。
- C2RDataset：相同fresh左图/EEF/text，Stereo增加右图；把两眼拼入一次VideoColorJitter调用，保留相同随机数消耗和左目结果。固定2461，不重新筛选。
- 预处理：共同EEF minmax、旋转固定[-1,1]、指令池、窗口计数和同算法LR表。映射须区分episode_map的原始canonical ID与replay_verification的compact ID；部分目录名为canonical-0105，不能用目录名推断episode。前两次定位失败保留日志，未跳样本或改源数据；第三次完整成功。
- Trainer：online W&B接入父类；取消递归checkpoint清理；每epoch保存，最终重复调用幂等；fullstate保存/加载tokenizer绑定。
- 初始化：Wan发布其proprio权重，Stereo读取；两组视频/动作原始预训练相同，避免codec构造消耗不同RNG使随机头不同。运行时另保存初始化抽样fingerprint；尚未GPU执行验证。
- H800 launcher：单节点8卡、BS8/GA4/global256；text/wan/stereotok准备与smoke入口，必须提供实际已发布clean SHA，不接受占位SHA。默认2更新smoke，完整5epoch LR表保持；smoke输出独立，不能用于正式初始化。

## H800资产及预处理结果

RAM205455090823393499/workspace350607/research优先级4。当前ListQuotas/allowedRoles变为training quota1bybv9m0ixu，Ready，允许当前角色DLC/DSW；没有沿用失效research配额。准备DSW dsw-dqg0h974kfthja8hs9，1L20Y/24CPU/256Gi，Create Request01A13494-2B32-5535-8285-E0528F599B8D。SSH真实可达120.77.254.228:1020，/mnt/workdir实际CPFS挂载。没有提交DLC训练。

数据来源 /mnt/workdir/frank/datasets/RoboTwin2.0/stereo-clean-fresh-v21-parallel-20261009-99f7b8c/meta/stereo_provenance.json，复用其对应原始HDF5六路RGB/endpose。第一条139个训练观测对应140个原始模拟状态；沿用ruhu N-1窗口合同。

输出 /mnt/workdir/frank/runs/c2r-prep-20261013/prepared/：manifest.json、prompts.json、lr.json、prepared.json。全部2461条/50tasks；543053模拟状态；540592窗口；每epoch补齐540672；2112更新/epoch；总10560更新。所有2461用作训练，无新train/val筛分，保持ruhu all-train。保存步数2112/4224/6336/8448/10560。157332唯一格式化prompt；T5缓存尚未生成。

manifest SHA256=9f277c833fe31d420d8e19eefb387de5865d49023513eb36f82a9ed612eaa8bd；原provenance SHA256=20ff630a19fcd4b8719cdc3171cccd74a7891effe22f827c49968f969c210f66；来源ruhu manifest SHA256=67f1fd8a405da718825c76f49b413bc2ab42627fc06bfe02bddb92b5b929da6f。

H800现有ActionDiT文件与ruhu文件整体SHA不同，但ZIP内data.pkl与820个tensor storage的名称/大小/SHA256全部一致；聚合digest=76c0fe16a19e25833769245f781329344a6bbb9fe66f9b7b4e66d2402c793232。因此可直接复用 /mnt/workdir/frank/checkpoints/fastwam-shared-clean15fps/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt，不传输重复大权重。

个人资产链接 /mnt/workdir/frank/checkpoints/fastwam-c2r-shared-20261013，引用既有Wan分片、UMT5 tokenizer及DiffSynth转换VAE/T5 safetensors。配置redirect_common_files=true为现有资产封装差异；其真实strict load与GPU数值仍待smoke，不宣称本轮完成原始pth和转换safetensors逐tensor对比。未下载模型。

使用既有fastwam-original-cu128解释器，加只读fastwam-stereotok-overlay-20261009中的h5py/timm/headless cv2；新隔离overlay fastwam-c2r-overlay-20261013仅装scipy1.15.3、ftfy6.3.1、wcwidth0.2.13。不修改旧共享环境。早先fastwam-train-torch271不含所需依赖，不作为本次运行解释器。

## 验证与未完成项

- AST：29文件通过；git diff --check通过。
- CPU unittest：4/4通过，涵盖两组左目/动作/proprio/prompt/mask/RNG一致、同图左右增强一致、t+1目标/尾padding、缺右目失败、sampler补齐/恢复。通过stdin临时加载本地模块，在H800既有renderer解释器执行；不是正式模型smoke。
- 真实窗口0/270296/540591：上述两组共同字段逐tensor完全一致；右图同shape且有限。
- 原10741项LR表使用原vendor scheduler独立生成，与ruhu参考逐浮点完全相等。
- 完整2461 EEF与manifest CPU预处理exit0；无数据重渲染、无新排除。
- 尚未运行8卡forward/backward、save/resume或策略推理；未生成全部T5缓存、未验证本轮online loss上传。不要称smoke通过。

本机证据 C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/frank-c2r-stereotok-20261013/：cpu-tests.log、real-data-parity.log、prepare-data-r3.log、lr-parity.log、local-source-hashes.json、DSW请求回执。测试临时文件保留，没有递归删除。

准备DSW已请求停止以免等待发布授权时占卡：Stop Request01A134A4-0476-562F-8DD2-29343DCE9274；最终状态另在聊天核验。下一步需要用户明确授权在Frankaaay/FastWAM创建frank/c2r-stereotok-2461分支并commit/push本次具名源码与记录，以满足AGENTS的显式Git授权和已推送精确SHA运行规则；随后远端同步、8卡文本缓存、Wan8卡smoke、Stereo8卡smoke与完整保存/恢复验证。正式5epoch训练未自动授权启动。

## 发布授权与服务器验证续作

用户明确批准将本地适配作为自己的repo分支并commit + push，然后到服务器测试。当前分支frank/c2r-stereotok-2461，origin已改为https://github.com/Frankaaay/FastWAM.git；fetch成功，远端尚无同名分支。保留ruhu精确7faa711基线，不合入用户fork的其他main改动。准备提交必要源码、配置、CPU测试与本记录，服务器只运行发布后的clean SHA。


已发布475e770f5d35855746875ed8c8e029f4b601417c到Frankaaay/FastWAM的frank/c2r-stereotok-2461。H800个人clone已fetch、切换该分支、pull --ff-only，HEAD一致且clean。真实运行环境torch2.7.1+cu128、einops0.8.1、CUDA可用，C2RTrainer/StereoTokVAE导入通过；该已发布源码的CPU测试4/4通过（0.441s）。暂存检查发现原导入文件尾部空行，保留来源格式；排除blank-at-eof后的git diff --cached --check通过。

文本缓存DLC dlc1v3jymjmf2j86，Request01A134AD-D37D-5140-AD0C-C8C01F9BD3D4；单节点8GPU/96CPU/512Gi/64Gi共享内存、优先级4、training quota1bybv9m0ixu、ForbiddenQuotaOverSold、240分钟上限、不自动重启。请求文件本机frank-c2r-stereotok-20261013/text-r1.json。状态EnvPreparing，尚无模型执行结果。运行入口对应475e770；本条记录更新不改变运行源码。


启动健康检查：DLC于2026-10-13 09:53:54 UTC进入Running，master-0 Running，挂载准备成功，训练日志文件已创建但尚无文本编码进度；不能视为模型smoke通过。文本缓存157332条的主体ETA尚无吞吐依据，4小时为作业时限而非预计耗时。后续Wan/Stereo各8卡smoke和save/resume尚未提交，须等待缓存成功。DSW仅用于Git同步与CPU验证，已请求停止以释放1卡；DLC独立继续。


## 文本完成与训练smoke存储阻塞

文本DLC dlc1v3jymjmf2j86 Succeeded，master退出0，2026-10-13 10:25:48 UTC完成；text_cache.json实读status=passed、157332 prompts、context_len512，exit.json=0。

随后提交Wan单节点8卡smoke dlcn9or9nk2ipnsj（Request01A134CE-9080-523B-AA51-7052D6F1EAD9），BS8/GA4/2更新/1024Gi主存/90分钟上限，拟运行633981a；容器Running但尚无训练输出，远端HEAD仍475e770，不宣称已同步或进入训练。

当前DSW实时df -B1 /mnt/workdir：总10995116277760、已用10995116277760、可用0字节；inode使用25%，为容量满。无可靠日志/完整checkpoint空间，已请求停止本轮Wan smoke和准备DSW，不提交Stereo smoke，不删除任何文件。需用户释放共享盘空间或指定另一个持久输出位置后继续。此前1至2小时ETA失效。


## 切换完整2500数据

用户明确指定2500条用于smoke与正式训练。已读取published/ready.json：2500条/50任务各50条，551534帧、549034窗口；补采39新增8481帧/8442窗口，为新示范而非原39轨迹恢复。manifest SHA256 ea55afb8f616b860633d97185fc2eb390d82bbbfdb458bfeefa4cc2fb22ee90e，84个reader窗口验收通过。复用已发布归一化统计、manifest与prompts，不重复扫描旧数据；prepare_data入口改为published-root，独立prep目录c2r-prep-2500-20261013，复用文本缓存仅补缺。全局256，每epoch2145更新，5epoch10725；保存2145/4290/6435/8580/10725。AST与diff检查通过。

纠正前轮存储判断：df/statvfs的0不能据此断言CPFS无法写入；用户提供实际fsync/readback成功证据，按授权继续写入并以实际训练checkpoint结果验收，不清理数据。


2500准备在9950d1e执行成功，manifest符号链接到补采published目录，LR与prepared写入独立prep成功；159480 prompts中仅2148缺失。服务器4项CPU测试再次通过（0.481s）。增量文本8GPU任务dlc11pc2krvfo0pw输出text_cache.json=passed/159480/512，exit.json=0。Wan2500 smoke dlc1kam9sbs1ygeh已提交（Request01A134EB-B87D-5328-B3C8-D8C24FB76CA4），8GPU/96CPU/1024Gi/64Gi shared/90min/priority4，源码9950d1e，2次更新；GPU训练结果待验。


Wan2500首次smoke两个更新loss=1.7409/1.9815，W&B online176xxbn2；保存后exit0。Stereo首次dlc1eqvsp7t751xe在训练前正确拒绝初始化不一致：ActionDiT.from_pretrained明确保留action_encoder/head随机权重，原共同初始化只包含proprio遗漏这两部分。定向修复：三部分共同初始化均保存并strict恢复，三部分全tensor hash比对，原video/action backbone仍使用严格相同预训练；独立common-initialization-v2保留旧证据。AST/diff检查通过，下一步Wan从step2恢复到step3验证fullstate并发布完整共同初始化，Stereo重跑。未改BS/LR/数据语义。


Wan服务端W&B API验证176xxbn2=finished，step/engine_steps=2，loss1.9814679622650144、action1.42553973197937、video0.5559281706809998、grad_norm29.05506706237793、lr3.76038563015148e-7，peak52.6919GiB。c2r_complete确认world8/BS8/GA4/batch_in_epoch8，completed step2/target2，exit0。

修复0e360ad已push，但远端同步受网络阻塞：首个fetch长时间无输出，确认自身git进程后终止；有界IPv4重试报Failure when receiving data from the peer；GitHub网页curl301可达，最后HTTP/1.1定向fetch仍120秒超时。服务器HEAD保持9950d1e；按Git规则停止依赖新代码的执行，没有启动Wan恢复或Stereo修复后重试。prepared/cache/首次Wan完整checkpoint均保留，未热改或旁路源码。待Git同步成功后续跑Wan恢复和Stereo smoke。


## 网络恢复后的实际验证

用户要求重新尝试。远端fetch/ff成功，clean SHA8377fc76b8783f5daa996ded95c15bf2fa0c955a。Wan恢复任务dlc1j6xq8bclwgll：模型/optimizer/scheduler/random/dataloader状态均加载成功，epoch0 batch8 sample_offset512，从step2实际推进到step3，loss2.2558、lr5.63e-7，完整保存后completed=3/3且exit0；W&B uxpn4cxb。

修复后Stereo任务dlc1j6z4sr9lqtdc，W&B rxnl46bh，成功进入训练。所有8 rank共同初始化合同和fingerprints均与Wan rank0相同；proprio/action_encoder/action_head为全tensor哈希，video/action为首参数抽样指纹。配置差异仅tokenizer/right输入、output/wandb.name及本次resume测试的step限制/resume路径。正式配置不引入这些测试差异。


## 同口径速度对比

用户要求比较ruhu速度。只读H100授权root内training/fastwam-c2r/metrics.jsonl：10740行，连续步wall_time差中位2.6982207298s、均值2.83408675s、P90 2.70790577s；完整1至10740耗时8.454h。sacct14685确认8GPU/96CPU/1TiB，训练step耗时8:31:01，source trainer.py和mot.py去CRLF后SHA与本仓库相同。

H800 WAN两步wall_time差40.24736643s=6.36066样本/s；Stereo44.81613779s=5.71223样本/s，分别约ruhu稳态14.916x/16.610x。仅一个非首步间隔，非充分稳态测量；不能把日志累计speed当瞬时速度。按这一短样本外推10725步约119.9h/133.5h，非正式ETA。Stereo首步包含首次编译约318s，不用于稳态对比。硬件DSW实测L20Y/SM9.0/132SM/80GiB、torch2.7.1+cu128/cuDNN90701，尚未证明差距根因。DLC GPU metric API返回null不冒充利用率。追加有界Wan恢复至step10作速度测量，Job dlc594fm3bvfmi8x，不启动正式训练。


Stereo修复后两步smoke完成：dlc1j6z4sr9lqtdc exit0，c2r_complete=step2/batch8/world8/BS8/GA4，完整checkpoint存在；loss1.7733/2.0047。继续Stereo恢复并测至step10：dlc1miyh3f363oyc，入口SHA8377fc7，共同初始化v2，未修改正式训练配置；启动健康检查已到ZeRO/W&B初始化。Wan速度任务dlc594fm3bvfmi8x已恢复step3并产生step4 loss2.0757，等待后续连续计时。两任务仅最多step10，不是正式10725步训练。预计还需约10至15分钟主体、1至3分钟保存（粗估，编译/存储可延长）。

DSW隔离诊断：BF16 [2880,3072]@[3072,14336]，5次预热20次CUDA Event计时，0.333528ms/760.57TFLOP/s；单进程torch线程1、8个随机Wan完整reader样本0.0826至0.1382s。仅准备实例证据，不能替代训练节点profile，不能认定计算/并发IO/通信瓶颈已排除。未修改生产算子、通信参数或数值精度。准备DSW诊断结束，停止以释放空闲1GPU，两个DLC独立继续。

