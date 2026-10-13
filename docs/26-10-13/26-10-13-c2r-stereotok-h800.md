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

