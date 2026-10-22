# 三模型共享 RoboTwin 冻结 plan

用户确认以已验收FastWAM的100份冻结plan为基准。保留原文档修改，当前分支已快进336019e；本轮修改未commit/push，未部署。

正式入口统一为 `python scripts/c2r/evaluate.py rollout --root OUTPUT`，汇总使用summary。prepare/pipeline不再可选。固定清单索引是`configs/robotwin_shared_plan.json`，文件哈希锁定在evaluate.py，CPFS目录为`/mnt/workdir/frank/runs/c2r-stereotok-mono-eval-final-20261017/plans`。Clean/Randomized各5000回合，不允许缺失后补采样。

OUTPUT/manifest.json包含：model（必填fastwam/openwam/lingbot）、episodes=100、chunk_size=5、cases（直接从锁定索引选择完整phase）。FastWAM继续提供checkpoint/training_root。OpenWAM policy字段为host/port/stereo；LingBot policy字段为host/port/stereo/video_guidance_scale/action_guidance_scale，实际推理参数仍须核对原模型服务配置。不同模型/checkpoint使用不同OUTPUT目录，不能复用历史结果。

执行环境需要FastWAM src、OpenWAM仓库、XPolicyLab父目录可导入；模型服务分别使用原有环境和启动方式。不要把FastWAM模型推理超参数复制到其他模型。现有H800 launcher仍配置FastWAM解释器，Open/Ling正式请求的客户端依赖与PYTHONPATH需在发布新请求时实际验收，不能仅修改model字段便宣称可运行。

结果绑定plan哈希与完整manifest哈希。缺失/篡改plan、预算变化、模型混用、seed/instruction不一致直接失败。环境Python/NumPy RNG独立于策略；结果保留初始场景与RGB指纹。

本机9项共享测试、既有12项通过；服务器CPU验证真实100份/10000回合hash及Open4项/Ling5项测试。尚未进行真实GPU/SAPIEN三模型联调，不能作为正式上线验收。完整证据与作业ID见 `C:/Project/StereoTok/docs/26-10-21/26-10-21-robotwin-shared-plan.md`。

## 授权发布与单卡验收

2026-10-21 用户明确允许提交/推送本次变更和1张闲时GPU串行验收。具名提交保留其他dirty文件。Fast测试入口tests/check_shared_plan_runtime.py仅用于真实Clean/Randomized初始化和两次原生策略调用，不生成正式分数；固定索引LF换行，LingBot空动作步维度拒绝。实际远端结果另行追加。

## 2026-10-21 授权发布后的实际状态

- 用户已明确授权本次提交/推送及1张闲时GPU串行验收。
- 已推送代码：FastWAM `a02945527cde731ecf312798372778087b4ab910`；OpenWAM `11abfa5bf382c04457ae9824c48fb472ec502527`；XPolicyLab `93178a8af456f216f410b09d47bf156ee4ab67bb`。仅暂存本次具名代码、测试、记录；原有dirty文件保留。
- 最新本机重跑Fast共享9项、既有12项通过，diff检查通过。LingBot空动作步维度检查本轮补充并静态解析通过，更新后的该边界仍待真实依赖环境执行。索引以.gitattributes固定LF，避免Windows自动换行改变锁定哈希。
- 已提交 `dlc13hptsbyitwmn`，1GPU/16CPU/120Gi、ForceQuotaOverSold、priority4、30分钟上限。RAM实时仍为frank，research配额Ready；允许DLC优先级上限为6，本次保持4。最后查询为 **Queuing / JobEnqueued，无StartTime**；尚无实际GPU型号、仿真或模型运行证据。
- 首阶段将建立3个已推送SHA的独立clean clone于 `/mnt/workdir/frank/projects/shared-robotwin-20261021/`，复用旧冻结测评实际使用的assets/task_config只读输入，核验100份计划；在同一Fast仿真解释器下以3个模型标签分别初始化adjust_bottle的Clean/Randomized ordinal0，逐项比较scene、六路RGB、seed、instruction和step limit。此初始化检查不加载模型，不能代替原生策略联调。
- 输出预定为 `/mnt/workdir/frank/runs/shared-robotwin-acceptance-20261021/`；仅在作业实际运行后才会存在。当前未改动已有训练/正式测评。
- 本机请求/回执：`C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/frank-rolling-two-20261018/shared-runtime-init-20261021.json` 与同名前缀 `-receipt.json`。远端准备脚本保存在同目录 `shared-runtime-prepare.py`。
- 后续串行原生模型联调草稿 `shared-runtime-models.py` 已准备但**未提交**，必须先读首阶段真实日志和runtime-inputs.json，修正实际依赖/配置差异后执行。拟使用Fast Wan既有最终checkpoint、Open StereoTok 30K、LingBot StereoTok最新完整checkpoint，保留各自原生推理参数；每个模型两种phase各两次策略调用，不生成分数。单卡顺序运行，前一模型进程释放后才加载下一模型。
- 剩余门禁：真实场景及像素一致、客户端依赖可导入、原生模型checkpoint/encoder/服务端配置绑定、有限16D EEF动作和动作执行；全部仍待验收。没有宣称三模型已完成GPU验证，也没有发布新的全量测评作业。

## 用户询问结果后的实时复查

首作业 `dlc13hptsbyitwmn` 已 Failed。首个根因：从GitHub拉取FastWAM已推送SHA时443连接超时（128847ms），git fetch返回128；只创建了目标空Git仓库，未运行真实仿真，也未加载任何模型。该失败不能解释为策略、plan或场景一致性测试失败。

已定向修正本机启动脚本：存在目录但没有HEAD时，核验origin和clean状态后继续fetch/checkout；已有HEAD则必须等于锁定SHA，不覆盖其他checkout。没有删除目录、改模型参数或改资源合同。脚本AST解析及CLI dry-run通过，按原授权1GPU/16CPU/120Gi、ForceQuotaOverSold、priority4提交一次重试 `dlc9km2vfuwaksej`。请求/回执为既有请求目录下 `shared-runtime-init-v2-20261021.json` / `shared-runtime-init-v2-20261021-receipt.json`，脚本 `shared-runtime-prepare-v2.py`。没有重复启动仍在运行的作业。

三模型真实GPU验收仍未完成；模型联调阶段尚未提交。

## 第二次结果复查及网络依赖修复

`dlc9km2vfuwaksej` 已Failed：GitHub443再次连接超时（131106ms），git fetch返回128。仍未进入仿真或模型测试，不能声称通过。

已改为传输已推送提交的Git增量bundle：FastWAM目标a029455/基线336019e；OpenWAM目标11abfa5/基线160f300；LingBot目标93178a8/基线1a96f3d。先fetch各自Frank既有仓库的基线Git对象，再校验bundle SHA256及Git前置提交，导入并checkout精确已推送SHA，最后检查clean。只读既有Git对象，不复制dirty工作树、不使用alternate、不修改原仓库。三个bundle本机git bundle verify通过。目标分支fetch成功，目标提交不变。

新作业 `dlcn743yjn7fefha`，请求 `shared-runtime-init-v3-20261021.json`，回执同前缀-receipt.json，脚本shared-runtime-prepare-v3.py，均位于既有本机请求目录。保持1GPU/16CPU/120Gi、priority4、ForceQuotaOverSold、30分钟上限。运行期间无需访问GitHub。提交成功不代表验收通过，模型联调阶段仍未执行。

## 连续推进验收：v4

v3 `dlcn743yjn7fefha` 已Failed。FastWAM已成功导入/checkout a029455并通过clean校验；OpenWAM既有来源为shallow clone，普通fetch未接入shallow边界，导致bundle verify报告基线对象虽存在但未连接历史，仿真尚未开始。定向改为fetch --update-shallow并将已知基线绑定refs/remotes/runtime/base，再执行bundle verify和精确SHA checkout。没有修改源仓库或丢弃dirty文件。

确认v3终止后提交 `dlc1krqz9nj1az0t`，请求shared-runtime-v4-20261021.json；仍1GPU/16CPU/120Gi、ForceQuotaOverSold、priority4、30分钟上限。该作业将场景验收和三个原生模型串行联调串在同一次执行中；只有前一检查通过才继续，原生客户端先做真实解释器导入检查，LingBot服务使用原生logger初始化。模型/计划/环境/训练参数未改变。等待实际日志，未宣称通过。

## 真实场景相机检查修复

v4已成功进入真实SAPIEN，首个Clean场景返回7路RGB：6路策略双目加front_camera。测试误断言6路导致停止；改为核验精确7个相机键，保留全部RGB指纹。没有改环境、相机、模型输入或验收一致性要求。修复仅tests/check_shared_plan_runtime.py，AST解析通过。

## 场景验收通过；模型客户端依赖修复

v5 `dlcpfj4ejh7plz1z` 实际输出 REAL_SCENE_IDENTITY_PASSED_3_ROUTES_2_PHASES：adjust_bottle ordinal0的Clean/Randomized在3次独立进程初始化中，seed、instruction、scene、全部7路RGB及step limit逐项完全一致。Clean scene=2eb93b3fd4fa26a8eb141acfefbf84a5cca0261e2676628905b2f00d195df110，RGB=1c01a1de49c7b2050ce1c7689531e1ac068859043aad8c432ca8dfbf1946946a；Randomized scene=6be639c8ebf1f9d704d9fd9c6f55580b64251da4ccef1e4b2dfb2eefa1c36c97，RGB=947e749180b639339ce2d76bb9831366b20f872788a7a25bd3ad031979143cee。三个init.json已写入远端任务输出目录。此部分尚未加载模型。

后续真实客户端导入在LingBot缺websockets时报错。v6仅配置独立client-deps目录，链接既有OpenWAM CPython3.12环境的websockets/msgpack两个传输包；不将整个其他site-packages加入PYTHONPATH，不改Torch/SAPIEN和正在使用的环境。导入后打印实际Torch/SAPIEN路径及传输包版本。v6 `dlcem1rm391wybl6` 已提交；直接复用已通过的初始化证据，继续三个真实checkpoint串行调用，保持1卡闲时。请求/脚本前缀shared-runtime-models-v6。模型联调尚未通过，不得把初始化通过写成全部完成。

## 最终状态更新（2026-10-22）

共享plan真实联调已通过：FastWAM Wan、OpenWAM StereoTok、LingBot StereoTok在Clean/Randomized代表场景完成真实策略执行，三模型初始场景/七路RGB/seed/指令/step_limit完全一致。LingBot额外修复KV dtype边界和完整多组流式反馈，9项回归通过、每phase实际48动作；最后作业dlckkotu3pv3ga54 Succeeded。汇总产物/mnt/workdir/frank/runs/shared-robotwin-acceptance-20261021/accepted-models.json。实际代码FastWAM a8a62f8、OpenWAM 11abfa5、XPolicyLab b8164454。完整证据见StereoTok仓库docs/26-10-22/26-10-22-robotwin-shared-plan-acceptance.md。此前待运行/失败段落是过程历史；本结论仅代表场景与连续接口验收，不是全量分数。
