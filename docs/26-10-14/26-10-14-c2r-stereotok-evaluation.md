# C2R StereoTok 全量闭环评估与优化

用户授权优化、必要入口/测试、commit/push、H800 多卡短测并直接启动全量评估；不改训练、不删除历史产物。目标 final step10725，50任务×clean/randomized×100回合，unseen、EEF20/minmax、horizon/replan32、10去噪、shift5、CFG1、模型seed0、专家候选seed100000起、三组双目6/2/2cm。动作合同来自本次C2R训练；不能直接使用旧qpos/replan24入口。

基线：本地frank/c2r-stereotok-2461，origin Frankaaay/FastWAM，9a7c95cf7b7a7ba580ee55c6c90596d470540dbc，fetch成功且HEAD与upstream一致。原有两份26-10-13文档dirty保留，不纳入本次提交。StereoTok DLC dlc1ife1hnwmbq9s与Wan均Succeeded。Stereo completed.json=10725/10725、exit.json=0、final c2r_complete=world8/BS8/GA4、lr1e-6；五个weights文件存在。最终权重全tensor有限及完整state进一步核验中，不能把文件存在当完整载入验收。W&B thob0ldr日志确认step10725与同步完成。

旧H800评估记录：44GPU/88环境，Stereo全量6h31m，Wan5h13m；task-condition100回合粒度仍存在长尾可能，尚未完成阶段计时证明占比。

实现：复用现有Camera双目扩展及render_device定向绑定；policy复用指令embedding，动作队列非空时不请求新观察；新evaluate入口先做官方专家筛选和第二次scene setup，记录有效seed/指令/Python和NumPy随机状态，再动态领取小批回合、持久复用模型。规划仅对原UnStableError跳过种子，意外运行错误失败，不静默缩小覆盖。禁用仅用于视频产物的录像，不改RGB策略观察、仿真步数、成功判定或推理步数。原专家筛选成功判断保留。计划固定Python RNG0以使语言选择可复现（上游未固定Python RNG）；后续Wan必须复用同一plan。

最小CPU验证：独占claim、完整覆盖才聚合、重复episode拒绝，共2测试通过；git diff --check通过。实际GPU吞吐、并发显存、输出数值一致性和闭环smoke尚待验证。DSW dsw-dqg0h974kfthja8hs9已按授权启动，独立8/多卡正式评估未提交。

远端准备：独立clone FastWAM-C2R-eval-20261014，已有Frank assets与task_config仅符号链接引用并精确写入.git/info/exclude；未修改原配置或训练clone。直连fetch首次停滞，核对本任务PID后定向结束，有界重试成功。初次仿真启动发现Device在sapien顶层，已按已有h100_robotwin_gt实现定向修复；未更换依赖。该次尚未领取任务，重试不重复episode。

实测checkpoint验收：全部10725条metrics连续且有限，final权重1651 tensor有限；五份state内8个random_states及所有已列文件非空，c2r_complete匹配2145/4290/6435/8580/10725。真实GPU strict载入通过。缓存对照4个稳态样本：prompt median .322300701秒，cached .2915683365秒，减少9.53%；动作逐元素一致(maxabs0)，peak37.20597GiB。初始模型加载耗时约3分钟，完整ETA必须含冷启动。证据/mnt/workdir/frank/runs/c2r-stereotok-eval-smoke-20261014/{checkpoint-acceptance,inference-benchmark}.json。

4GPU/96CPU/512Gi pilot prepare dlc1v2nnrfkynu57，6个task-condition各2回合，已完成5组；open_microwave clean专家在无抓取pose时抛固定AssertionError。官方eval会将其作为专家无解候选跳过；本入口补充仅该已定位异常的拒绝记录，其余AssertionError仍失败，不减少有效回合目标。另为rollout增加setup/observation/action分阶段计时。

首轮闭环4/4成功（仅接口smoke，不作为全量分数），进程总290.59秒含模型/仿真初始化。单回合61/69/61/56动作，steady第二/第四回合19.58/16.67秒，缓存推理约.65-.70秒/回合，仿真占主导。定位take_action每个内部physics step调用scene.update_render。新增定向候选：无视频/UI时推迟camera/scene同步到get_obs，原随机灯光更新调用次数保留。tests/benchmark_c2r_render.py要求clean和randomized同seed下像素hash、每步动作、每步EEF状态全部逐元素一致；通过前不用于全量。

延迟渲染GPU对照通过：clean/randomized各一个固定episode，pixels/actions/EEF states全部相等。扣除首次模型编译后仅小幅阶段加速，不能用78→47秒宣称40%渲染加速。阶段计时揭示本轮入口每episode clear_cache=True导致反复加载资产；恢复官方(ordinal+2)%clear_cache_freq清理节奏。专家筛选最终按官方对expert/第二次setup的Exception拒绝候选并记录全trace，rollout异常绝不跳过；最多10000候选后失败，不减少100有效episode目标。

并行对照：2GPU×1进程 dlc5r9lvvdp7t6c4；2GPU×2进程 dlc6b8snff1a60ww；共同4个task-condition×2episode。峰值/完成时间后续补齐。改进调度为pipeline：每worker完成已领取专家plan后立即消费其他已发布plan的小批rollout，不设全局准备屏障；全量每chunk10回合，持久模型复用。聚合校验ordinal/seed/instruction及覆盖，文件锁避免多job聚合覆盖，新CPU ready-case测试覆盖首个慢plan不阻塞已完成plan。

## 优化验收及正式启动（北京时间2026-10-14）

- CPU定向测试3/3通过；真实GPU缓存动作bitwise一致、渲染clean/randomized像素/动作/EEF状态逐元素一致。双进程与单进程闭环步数存在小幅波动，不宣称跨并发/调度逐步确定性；固定计划和成功判定保持一致。
- 并发实测：单进程每卡峰值35220MiB、8回合572.0597s；双进程每卡峰值70076MiB、8回合502.1072s，端到端缩短12.23%，两组均exit0、6/8成功（pilot，不是正式成功率）。包含首次模型加载/编译及当时每回合清缓存，不将此时长当优化后全量稳态。
- 最终pipeline单卡双进程smoke：/mnt/workdir/frank/runs/c2r-eval-pipeline-smoke-20261014，PID41402已结束，exit全0、summary complete、4/4回合，无错误；进程总260.5357秒。后续回合setup .26396s，分别22.16/16.66s；首次setup23-24秒和首次编译49秒另列。修复清缓存频率后不再每回合重载场景资源。
- 正式代码固定d91d45b67b4759f27cdc4e0bd31bc3004cd34a36。独立clean clone /mnt/workdir/frank/projects/FastWAM-C2R-eval-final-20261014，第一次直连clone低速超时后原路径确认不存在，第二次直连成功；未绕过同步门禁。原pilot clone未热改。直接读取最终weights payload step=10725。
- 正式结果根 /mnt/workdir/frank/runs/c2r-stereotok-eval-full-20261014。manifest.json固定50任务×clean/randomized×100=10000回合，按长step_limit优先；最终chunk_size从候选10缩小到5（只影响调度，种子/指令/回合数不变），持久模型、准备和rollout流水执行。渲染仍rt/32spp/depth8/oidn；无录视频，不降采样或缩短episode。
- 10:53起提交36个2GPU/48CPU/256Gi/64Gi共享内存的DLC，priority4/ForbiddenQuotaOverSold/360分钟上限，无训练操作；10:55:27一次健康快照24 Running=48GPU、12 Queuing。再提交3个1GPU/24CPU/128Gi分组填零散空位，总上限75GPU/150worker。单卡补位Job dlc4xhukrnm9x6ns、dlc57hfyja7jqvub Running，dlc5hh1cbcjbtr1x Queuing；合计当时50GPU已运行。不能把75提交卡数当75运行卡数。
- 完整39个Job ID/RequestId/资源请求在本机 C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/frank-c2r-eval-20261014/full-jobs.json，同时复制到正式结果根jobs.json。首组dlcbv6akram9k2jy；原36组最后dlckgtpfvzno2oip。全部CreateJob回执均有JobId，未重复提交未知回执。
- 有界实际日志检查：100个prepare claims/100个任务日志已建立，98组已产生有效seed进度，示例open_microwave clean ordinal9/seed100013、click_alarmclock clean ordinal99/seed100115；其余刚启动。errors目录无已发布失败。准备成功不代表全量rollout完成，尚无正式最终成功率。
- 粗估主体2.5–4小时，另5–10分钟验收汇总；基于短/中任务及上游620平均step_limit、实际约50卡起跑、后续排队补位，不承诺固定完工时间。不会沿用旧44卡6h31m当本轮实测时长。
- 自动跟进stereotok已ACTIVE，每10分钟有界检查；用户已授权离开期间继续推进，只在失败、需用户决策或最终完成通知。冻结计划、完整覆盖、去重和错误门禁都通过后再发布正式SR，随后暂停跟进，不恢复旧训练automation。
- 本人单卡准备DSW最后GPU无进程、无其他/mnt/workdir/frank任务，已请求StopInstance；正式DLC独立运行，不依赖DSW或本机聊天保持在线。后续需要检查产物可临时启动该DSW再停止。

复核入口：既有Python环境运行 scripts/c2r/evaluate.py summary --root /mnt/workdir/frank/runs/c2r-stereotok-eval-full-20261014；需等待完整10000结果，summary对seed/instruction/ordinal与plan匹配、重复episode和缺失覆盖均严格校验。每个DLC的workers/<job>/exit.json保存source与各worker退出码；plans保留候选拒绝原因和有效场景RNG，results/logs保留逐回合耗时和成功标记。下一步由跟进任务实时验收并交付逐任务clean/randomized结果。

## 11:12–11:16 有界跟进

- RAM 实时核验仍为 frank。39 个 DLC 一次状态检查：27 Running（24×2 + 3×1 = 51 GPU），12 Queuing（24 GPU）；没有 Failed/Stopped/Succeeded。该状态不代表回合完整性或最终验收。
- 为读取持久结果，按授权启动自己的单卡 DSW `dsw-dqg0h974kfthja8hs9`。实例停在 EnvPreparing，GetInstanceEvents 的 11:12:14/11:12:34 事件明确为可用 reservation 的 CPU/GPU/memory 不足；SSH 映射仍 Allocating，因此未发出 SSH 命令，也未读取本轮 plans/results/errors/worker exit。
- 已 StopInstance，随后 GetInstance 确认 Stopped；没有停止、重启或修改任何评估 DLC，也没有更改训练。
- 有界尝试 `get-pod-logs` 读取首个 DLC `dlcbv6akram9k2jy` 的 20 行日志，仅返回挂载/启动信息。当前 launcher 将逐 worker 日志写到 CPFS，不能用容器启动日志推断回合进展。
- 本轮无代码或评估合同修改。保留所有成功产物及失败证据；下一次跟进继续核对资源是否允许临时读取，待可访问后检查实际回合、错误与退出码。当前无法更新基于实际吞吐的 ETA。
## 11:24–11:30 单卡归一化速度核查

用户要求对比当前全量、ruhu、优化前，排除卡数。统一资源效率定义为有效已完成回合数 / sum(每组GPU数×实际评估占卡小时)，排队不计，已退出分片不继续计；冷启动/等待/故障计入端到端资源效率，并另列局部同条件A/B。未完成任务的任务混合、成功提前退出和硬件差异仍不能自动消除。

- 旧 StereoTok 全量：10000有效回合及覆盖验收见 FastWAM/docs/26-10-09/26-10-09-fastwam-two-tokenizers-training.md。本轮 GetJob 实时核对六组全部 Succeeded；按已记录释放时刻02:38:53–57的中点02:38:55（北京时间10月10日）至各DLC GmtFinishTime计算。五组8卡分别于07:45:16、09:10:03、05:00:03、05:34:40、04:52:42结束，最后4卡05:05:50结束。约162.8811 GPU小时，61.3945回合/GPU小时、58.6372 GPU秒/回合；约4秒释放散差和job退出尾部使此值为近似。排除训练依赖预占等待。44×全墙钟会错误计成约287 GPU小时，不能当实际资源效率。
- ruhu：以Frank通过h100只读 /gpfs/jiuquyun/projects/ruhu/fastwam-c2r-20261011/evaluation/aligned20/results/formal/{demo_clean,demo_randomized}/各task/episode*.json，1468文件全部complete；本轮该目录可读。sacct -j15094,15152 -X -P实际均8GPU，分别23266秒、27587秒，总113.0067 GPU小时，12.9904有效回合/GPU小时、277.1281 GPU秒/回合。含这两个job内的预检、恢复、卡住及未落盘回合开销，不含更早建立共享manifest的GPU预算；不是纯稳态推理效率。旧Stereo/该ruhu作业资源效率约4.73倍，并非本轮优化收益。
- ruhu逐回合壁钟合计318690.9658秒，median193.4232秒；623790 executed_actions、20197次模型推理，推理median .3422926702秒/mean .3438375104秒，总6944.4862秒（约逐回合wall总和2.18%）。不能将wall余量全部归因渲染。合同有every-action get_obs，Wan单目、H100、RoboTwin0aeea2d；当前StereoTok双目L20Y、bf44be5、按replan观察。旧Stereo为qpos/replan24，当前EEF/replan32。跨组并非纯代码优化因果对照。
- 同条件已验证优化：2卡8相同计划回合，1worker/GPU 572.0597s → 2worker/GPU 502.1072s，即25.1722 → 28.6791回合/GPU小时，吞吐提升13.93%，端到端耗时减少12.23%；不是翻倍GPU。指令cache推理 .322300701 → .2915683365s，耗时减9.53%，吞吐增10.54%，四稳态样本且动作bitwise一致。这些均为局部pilot，不能充当当前全量吞吐。
- 当前全量本轮实际完成数仍未知：按授权再次尝试本人DSW，事件11:24:18明确CPU/GPU/memory资源不足、SSH Allocating，未执行远端命令。随后停止，GetInstance确认Stopped。DLC日志API只提供父进程启动信息；不从Running、卡数或之前2.5–4h粗估倒推虚构实际吞吐。下次产物可读时必须按同一快照统计完成回合与每个DLC实际启动/退出时间积分，并尽可能对齐task/phase/动作步数后更新比较。未改正在运行源码、模型、协议或任务。
## 12:07 首次获得当前全量速度与异常

- 不再等待GPU读取实例：以现有DLC请求为模板提交1CPU/4Gi/0GPU、10分钟上限的只读统计作业 dlcgln9qjo42bany，已Succeeded。只用系统python3读取本任务结果目录，无远端文件写入或源码执行；统计输出经GetPodLogs取回。本机请求/回执/输出 readonly-cpu*.json 保存在既有frank-c2r-eval-20261014请求目录。后续可沿用CPU读取机制，避免DSW单卡排队；DSW仍Stopped。
- snapshot time=1791950818.8734612（北京时间12:06:58.873）：97个plan、437个chunk文件、2165个已发布episode。39组状态27Running/12Queuing；用每组实际Pods[0].GmtStartTime至同一快照的GPU时间积分=61.3012564 GPU小时，当前35.3173838回合/GPU小时。排队24GPU未计。约为旧全量61.3945的0.57525倍（当前低42.47%）、ruhu12.9904的2.71873倍。当前长任务优先且含准备/冷启动，未结束，不是完整任务混合的最终速度或纯优化因果倍数。状态原始快照speed-status-1206.json可复算。
- 发现实际rollout异常，不能以Running宣称健康：open_laptop clean offsets5/10/15明确报AttributeError: open_laptop object has no attribute arm_tag。源码check_success读取self.arm_tag；该字段只在play_once设定，当前rollout创建独立env后setup_demo没有调用play_once，准备与rollout进程间未传递该任务状态。需要定向复核官方/ruhu任务初始化合同，保留原计划、指令、成功episode与错误证据，再以新发布源码和独立运行clone修复缺失回合；不得热改d91运行clone或把失败当策略失败。
- GetPodLogs对单条EVAL_SNAPSHOT大JSON在约4KiB截断，故尚未取得完整错误列表/phase counts/exits；后续CPU检查应分行输出简短记录，不再把全部trace堆在一行。437chunk与2165episode存在4个非完整chunk，需逐一核对，不能猜测第四个根因。
## 12:12起资源纠正与慢速调查

- 用户先要求停止75卡、改16卡，已对原39个DLC逐个StopJob并GetJob确认39/39 Stopped。随后用户最新明确为固定资源16卡打底、额外闲时资源补充；额外使用ForceQuotaOverSold，总申请暂沿用原75卡上限。旧39任务不重启。当前自己的DSW单卡仅用于准备/检查，正式固定16卡启动前停止，避免额外固定占卡。
- CPU只读诊断dlc1wbll7tycnj5m成功（1CPU/0GPU/ForceQuotaOverSold），停止后结果488chunks/2420完整回合；四个错误均open_laptop clean offsets0/5/10/15缺失arm_tag。98plans完整，put_bottles_dustbin randomized已准备97/100；put_object_cabinet randomized约36/100，候选拒绝多（last valid seed100725），需保留原门禁，不跳过或缩小100回合。
- 2420回合累积298120.636秒，setup18563.756（6.23%）、observation31900.967（10.70%）、action247654.044（83.07%）；action中包含模型推理16207.037（占总5.44%）。930844动作，median389动作/回合，wall median109.714秒。推理缓存优化作用于很小的占比，不支持全量同比例加速。当前阶段长case优先与旧完整全量任务混合不同，42.5%差不能直接作为同条件回归结论。
- root原因：manifest分进程后没有恢复expert-only任务状态。只读核对ruhu已验证的OpenWAM task_state helper以及本地三个任务play_once/check_success；在evaluate.py rollout setup后按初始物体位姿恢复open_laptop/place_object_scale/put_object_cabinet的arm_tag与cabinet.origin_z，不跑专家动作、不更改种子/指令/成功阈值。
- 为落实用户动态闲时加减卡要求，现有claim改为Linux内核flock租约，逐回合原子落盘，日志/owner按attempt保留；worker退出自动释放锁，存活worker重新扫描未完成工作。只在独立接续root启用，原目录与失败证据不覆盖。已完成case标记减少CPFS扫描；prepare与rollout同worker循环，避免长prepare阻塞全局或被抢占后永久留下mkdir claim。
- 本地6个CPU测试：4通过、2个Linux flock测试明确跳过，远端CPFS仍需执行退出释放测试。已有render benchmark扩展指定task/回合数，旧执行候选eager渲染+每次encode prompt，与当前deferred+缓存在同模型/EEF32/场景计划下对照；两phase反转先后顺序观察冷缓存偏置，保存像素/动作/EEF状态一致性和分阶段时间，不用旧qpos协议替代当前科学合同。尚未完成GPU对照，不提前宣称速度或恢复完成。
## 12:32–12:35 按最新要求启动固定16卡与闲时补位

- 最新明确指令覆盖此前“16卡全闲时”：固定资源16卡打底，剩余闲时补充；随后用户要求立即挂16卡、以7小时为目标。ruhu约14小时只完成1468而非10000回合，不能用8→16卡线性外推7小时全量承诺。保留已验证合同，不把本C2R EEF模型切回旧qpos模型。
- clean发布SHA 9e1079e88875be2264a36e3a4bd47163b2a689d7；独立clone /mnt/workdir/frank/projects/FastWAM-C2R-eval-resume-20261014，原d91 clone未改。实际CPFS上运行6/6 CPU测试通过，包含跨进程独占与SIGKILL后锁释放；没有仅用Windows跳过结果代替Linux验证。
- 新接续root /mnt/workdir/frank/runs/c2r-stereotok-eval-resume-20261014。从原root完整保留2420已发布回合，并从中断chunk的EPISODE日志额外恢复162完整回合，总2582；逐条校验ordinal/seed/instruction与对应plan且无冲突。98个已发布plans和manifest逐字节复制；manifest SHA256 09f4e1f201e0f39c21794054f725574225943555fa42e304e24160cb99f2960f。新root/recovery.json记录来源，旧root所有失败、claims、logs、结果均保留。新summary确认2582/10000、完整SR为空，不冒充完成。
- 同条件诊断4个单卡Job dlcenvqbvuph2c05(click_bell)、dlcexvbpn0jujbv0(open_laptop)、dlcf7ux3fva8jp07(place_object_scale)、dlcfhuih7kl1u90b(put_object_cabinet)。响应用户“现在挂16卡”后提前停止这4个诊断，GetJob全部Stopped，日志保留 /mnt/workdir/frank/runs/c2r-eval-matched-speed-20261014。已获得click_bell clean两回合对照，pixels/actions/EEF states全部相等；排除首次冷启动后的第二回合旧14.2700s、新13.9229s（缩短2.43%），不代表全量优化收益。open_laptop修复后完成首回合且成功，place_object_scale前两回合正常；其余完整A/B和cabinet真实闭环未完成，不能宣称四组通过。DSW独立import诊断SAPIEN2.95s、torch累计6.88s、完整runtime15.61s、policy16.11s；不能凭DLC日志间隔把慢速确定归因CPFS。
- 固定16GPU(ForbiddenQuotaOverSold/priority4)：7个2GPU pipeline共14卡，另2GPU任务dlct3am8bswxyr95(mode prepare、每卡1worker)补put_bottles_dustbin randomized/put_object_cabinet randomized。准备Job完成后，若仍有未完回合，由跟进补一组2GPU固定pipeline，使固定资源维持16而不超过16。pipeline每卡2worker；独立准备进程不加载模型。主7组：dlc1yfnitndhbous、dlc1yzmpl77txcdi、dlc08jnkbba70q5z、dlc08jnkbvrc8xw9、dlcij8y3n2t40e3f、dlcsiubvnbjg1bjf、dlc12ifpn4kpkbtq。
- 额外闲时ForceQuotaOverSold/priority4：29组2GPU+1组1GPU，共59卡申请上限；与固定16合计75。不是额外59固定卡。新pipeline时限1440分钟，使用动态flock租约与逐回合保存，仅计算缺失episode。作业不会因为某分片100回合串行等待而垄断整个task。
- 新38 Job清单/请求在本机既有请求目录 resume-jobs.json、fixed16-resume-*.json、idle-resume-*.json；有界启动快照：固定8作业16卡Running；闲时25作业49卡Running、5作业10卡Queuing。本人准备DSW已GetInstance确认Stopped。全部旧39固定作业仍停止，新诊断4作业已停止；没有额外固定GPU。
- 0GPU健康读取Job dlc15avk7fmd2ic5负责发布新root/jobs.json并读取plans/results/errors/worker退出，输出分行避免API约4KiB截断；不能仅以Running作为回合完成证据。
- 自动跟进stereotok已更新为最新固定16+闲时规则并ACTIVE；检查仅接续root/新清单，禁止恢复旧39作业或训练。定向修复异常、只补缺失，统计新增吞吐时扣除2582恢复回合；最终总成本需另加首轮及诊断实际GPU小时，不把恢复复制当新产出。完整A/B尚未结束，后续不能宣称新配置已全量证实更快。

### 启动并发写入修复
0GPU健康读取发现completed标记写入的临时文件仅按PID命名；不同DLC容器具有相同PID，多个worker完成扫描同时写同一个case标记，tmp被另一容器rename后导致FileNotFoundError。原result/plan仍保留。write_json改用UUID临时名，8路相同PID并发写入barrier回归测试覆盖此根因；不热改9e1079e运行clone，发布新SHA独立clone，仅替换退出/受影响作业。全局唯一临时名也适用于后续每回合保存。

Windows初次并发测试触发目标文件并行replace的WinError5，属于Windows文件句柄语义；测试仅在Windows串行rename，保留8路临时文件写入屏障，Linux仍实际并行rename。未给生产代码加Windows重试旁路，必须以Linux CPFS测试作为部署门禁。

### 12:45 并发修复发布与作业替换
- UUID修复最终SHA 1fcbe1e7a6e758e5234e91cb1ecaf43cede5fe27；CPU DLC dlc8aaosr86dfnoy在独立clean clone /mnt/workdir/frank/projects/FastWAM-C2R-eval-atomic-20261014、实际CPFS测试目录运行7/7通过，包含8线程同时写JSON和SIGKILL释放flock。assets核验dlc11f41zfmbd3dq确认third_party/RoboTwin下两个link与原clone完全同源。
- compact健康读取dlco3njm37qa28ji于12:41:33：98plans、2583回合、无result/prepare error；两个固定Job各2worker因PID临时名碰撞退出。未把其余Job的Running视作完整健康。为防后续case完成时重现该全局缺陷，9e版37个pipeline全部Stop并GetJob确认Stopped，独立2GPU准备dlct3am8bswxyr95保留Running。不热改任何运行clone，所有历史worker/attempt日志和结果保留。
- 发布版重新申请7组固定双卡（14GPU）：dlc1tpyn0r0mcj27、dlc1tzy8ejmk8e37、dlc1u9xtsbvxhmb6、dlc1ujxf63hkmbjl、dlc1utx0jv3wtjv3、dlc1v3wlxnz5832m、dlc1vdw7bf0bnv9i；加准备2GPU共固定16。额外29组双卡+1单卡均ForceQuotaOverSold，59GPU闲时申请。精确registry仍本机resume-jobs.json；旧清单另存resume-jobs-before-atomic.json不得恢复。0GPU dlcee35sbsu9m6kh负责远端registry历史保留后发布和有界首轮健康检查。
- 7小时作为目标，不能承诺：ruhu约14h的1468回合不是当前10000；仅16GPU从2582恢复点7h完成剩余7418，需要66.23回合/GPU小时且不含额外准备尾部。首次快照35.32和旧全量61.39都不能直接当本轮重启稳态速度；额外闲时可加速但随时回收。当前未宣称完成全量，也未宣称代码同条件回归已闭环。

### 跨DLC认领门禁纠正
12:45:59新检查发现相同chunk在不同worker生成相隔约2秒的多个attempt，不能用单容器内flock测试代表跨DLC挂载互斥；已停止1fc版37个pipeline保留独立prepare。此前“内核锁可直接随抢占释放并由其他容器接替”的部署结论撤回。最小修正为CPFS服务端原子mkdir认领，owner绑定DLC_JOB_ID/pid/token；正常完成rename为released目录保留证据，强杀后必须由监控GetJob确认owner作业终态再逐个rename active为retired，不按超时抢活跃任务。动态加卡可立即领未认领片，减卡后的中断片经有界监控回收，不是无监控自动释放。新rollout使用mode rollout，仅等待剩余两plans，避免与保留的旧prepare锁实现混用。正式恢复前新增两个独立0GPU DLC验证真实跨挂载互斥及释放后的接替。

### 12:52 跨挂载实证与固定16恢复
- 两个独立0GPU DLC：holder dlc10b8hl7tplanp、peer dlch60i6jw0351w5均Succeeded。输出明确FLOCK_CROSS_CONTAINER_EXCLUSION False；新mkdir PEER_WHILE_HELD False、PEER_AFTER_RELEASE True、CROSS_CONTAINER_PASS。证实旧flock在实际独立DLC挂载不互斥，不是推测。holder在CPFS执行全部7项测试通过。首次holder请求远端Python解析因嵌套换行SyntaxError失败，未执行任何clone/评估；修复请求并静态compile后定向重试，无数据改动。
- 新发布SHA 8f1dc32374fe5ef0dcf500f5a1405e77eda54e22，clean clone /mnt/workdir/frank/projects/FastWAM-C2R-eval-mkdir-20261014。7固定双卡rollout：dlc1ncascrqko6y3、dlc1nw9z4bbmbrsd、dlc1nw9z4bkd7sf4、dlc1o69ki3h6pa8j、dlc1oq8r9n9my2v0、dlc1oq8r9nhyzx2e、dlc1p08cnfmnqocy。加仍运行的dlct3am8bswxyr95准备2GPU共16固定；59GPU闲时申请不变。
- 新源只以mode rollout启动，沿用既有接续root与成功结果。上轮1fc版37个pipeline均GetJob确认Stopped，registry历史保存在resume-jobs-before-distributed-fix.json，旧registry不可恢复。监控已改原子目录认领与确认终态后逐个rename回收，禁止仅按超时或pid跨容器判断死亡。
- 启动检查0GPU DLC dlc6chqtndqf4umw发布本轮registry并等待90秒读取分片、worker异常、GPU、回合与正式summary完整性；运行状态不代替实际吞吐。12:45:59之前快照2585总回合，不把导入2582或重启前新增3回合计入8f新版本吞吐。
- 首轮门禁实际返回（12:55:03快照）：固定16GPU Running，闲时49GPU Running/10GPU Queuing；126个活动认领对应126个不同chunk、32个rollout Job，未见重复分片认领。已启动worker无Traceback、无result/prepare error，正式summary校验通过但complete=false，总有效结果仍2585、98plans；clean/randomized完整SR均null。新rollout仍在启动/首回合阶段，未形成新增完整回合吞吐，不承诺7小时ETA。部分卡显存由20Gi升至41Gi，不能将单点GPU利用率0判定为故障或证明加载瓶颈。准备持续推进至dustbin randomized ordinal57、cabinet randomized ordinal29；专家拒绝沿原合同记录，不当作已成功回合。跟进需核验新增实际episode和稳定速率，勿重复重启正常启动中的作业。

### 13:04 用户追问ETA与ruhu代码/环境对照
- 通过Frank的h100 SSH只读ruhu指定项目，再核对protocol.json/evaluate_pair.py/run_pair.py与sacct：目标50×2×20=2000，结果仍clean903/randomized565=1468；15094 FAILED 23266s×8，15152 CANCELLED 27587s×8，共113.0067GPU小时、约14.126小时占用墙钟合计。不是8卡14小时完成10000，因此不能与当前16卡全量剩余墙钟直接比较。
- 两组共同EEF20、action horizon/execute32、denoise10、CFG1、shift5、BF16、unseen；评估程序不相同。ruhu RoboTwin0aeea2d、原Wan单目、每动作get_obs、每case启动模型/仿真两runtime RPC；本组bf44be5、StereoTok双目、replan取观察、worker持久模型。ruhu源码ObservationRenderer类存在不等于formal启用：evaluate_pair.py实际每动作observe/get_obs。模拟器/硬件/环境均不同，不能把单位GPU倍率宣称纯代码收益，也没有证据将稳态回归全部归因环境。旧Stereo全量qpos24尤其不是当前EEF32同合同基线。
- 13:02:14（CPU读取dlcdaaeizz10t0e4）仍2585；固定两组遥测已约69Gi、GPU80–95%。13:04:14（dlc1a0yhornwxogq）2614有效回合、98plans、无result/prepare error，8f版新增29；不再说新版本尚无产出。日志首批examples：put_bottles_dustbin clean ordinal45 wall297.17/setup36.34/infer101.89s，blocks_ranking_size clean ordinal73 wall315.02/setup36.24/infer107.23s，stack_bowls_three clean ordinal15 wall290.00/setup35.67/infer111.59s。包含冷启动，不能将首回合平均推理时长当稳态。
- 环境证据：首次Matplotlib字体cache初始化；PyTorch警告/root/.cache/torch/kernels不能创建，内核cache关闭；torch._inductor首次编译warning。证明存在缓存/首次编译开销，但未量化其占全部启动时间，不为此重启正在产出的worker，不改精度。下次必要替换时可定向修复可写cache路径，需验证而非热改运行clone。
- ETA澄清：此前8–13h只是历史两档吞吐的条件外推，不是新版本实测速率。按剩余7386和61.39/35.32回合/GPU小时，16张全用于rollout约7.52–13.07h；当前固定14rollout+2prepare，若准备不释放则约8.59–14.94h。当前49闲时+14固定=63rollout，若持续可用且恢复历史效率约1.91–3.32h；全部59闲时兑现+14固定=73rollout约1.65–2.86h。这些不含未量化准备尾部、抢占与调度退化，稳态ETA需后续同窗口增量。用户侧可暂按当前闲时持续可用时约2–4h规划，明确为低置信条件估计，不能承诺完成时刻。

### 15:58 按新授权先clean后重建randomized
- 用户明确授权引入ruhu修复、排查潜在bug、固定16GPU加闲时先完成clean，再重新跑randomized。已暂停旧自动跟进；旧两个残留闲时Job dlc1pa7y17vcx5zb、dlc1uk08az6gmylk已请求停止，待终态确认后复制clean结果。旧结果和计划均保留。
- 根因归属复核：RoboTwin bf44be51原始源码与FastWAM初始导入2764a1a均有get_cluttered_table默认list原地加bias；不是ruhu新增。ruhu aligned20/scene_defaults.py的两函数实际被sim_runtime.py调用。现将isolate_mutable_defaults/isolate_scene_defaults移入现有evaluate.py（注明来源），在environment创建后统一应用，覆盖准备与rollout；不修改上游共享源码，显式传参保留，默认值已污染则fail closed要求新进程。
- 真实源码AST最小复现：原始xlim[-.59,.59]两次调用后分别[-.29,.89]、[.01,1.19]，即使clean_background_rate提前返回也累加。新增跨task/连续episode隔离与污染检测测试，9项本地CPU测试全部通过。实际clean配置cluttered_table=false，绕过该问题路径；15:51:06逐条校验完成2971clean/2371randomized，无ordinal重复且seed/instruction匹配plans。randomized旧结果仅保留证据，不进入修复版成绩。
- 额外修复故障隔离：ready_cases遇到已失败的条件先记录blocked并让其他条件继续；最后仍非零报告失败，不伪造完整性。已失败chunk不重复把其他worker拉退出，当前worker新异常仍记录后退出，防带损坏CUDA/仿真状态继续。准备失败也仅隔离对应条件。正式summary仍要求所有目标回合完整且无error。
- 针对日志确认的缓存故障，launcher给每worker预建可写kernels/matplotlib目录并设置PYTORCH_KERNEL_CACHE_PATH/MPLCONFIGDIR；不改模型、渲染、数值精度或种子。旧运行clone不热改。
- 分阶段输出：计划新clean root /mnt/workdir/frank/runs/c2r-stereotok-clean-final-20261014，仅含原50clean cases/冻结plans和有效结果，目标5000；randomized新root在clean严格验收后再创建，50randomized cases从seed100000按原专家筛选规则重建100有效回合，旧random计划不复用。最终分别报告两phase，不把旧random混入。

### 16:09 clean 独立恢复与条件 ETA
- 旧两个残留闲时作业均 GetJob 确认 Stopped。发布 f9cb2e89f8d820a40a25239006ad846266f682fa；独立 clean clone /mnt/workdir/frank/projects/FastWAM-C2R-eval-isolated-20261014。bootstrap CPU Job dlcc7z11nwpetepo 在真实 Linux CPFS 运行9项测试全部通过；新 root /mnt/workdir/frank/runs/c2r-stereotok-clean-final-20261014，仅含50个clean条件，复制并校验2975/5000有效回合，剩余2025，完整SR仍为null。原plans/results及recovery来源均保留，旧randomized不混入。
- 已提交8组固定双卡16GPU和29组闲时双卡+1单卡59GPU，全部mode rollout、每卡2worker。精确38Job清单本机既有请求目录 clean-jobs.json，请求 clean-final-*.json，16:09逐个GetJob快照 clean-launch-status.json：固定16卡Running，闲时41卡Running/18卡Queuing。Running只代表调度；首次健康读取0GPU Job dlcftwk5ne165x1o负责发布新root/jobs.json和检查worker日志、错误、认领、实际回合数。
- 用户追问clean分数ETA：2025缺失按历史35.32–61.39回合/GPU小时，固定16卡理想2.06–3.58小时，含启动/长尾暂按2–4小时；当前57卡若持续可用，理想0.58–1.01小时，含启动/长尾暂按1–2小时。均是条件估计，不是修复版稳态实测，不承诺固定完成时刻。clean验收5000后立即报分，再按新授权重建全部randomized计划/回合。
- stereotok跟进已恢复ACTIVE/10分钟，更新到新clean清单、clone、SHA、2975恢复基数和先clean后randomized阶段门禁；禁止恢复旧任务/训练，禁止按超时抢claim。完成后暂停跟进。
- 首次真实健康快照16:10:09（dlcftwk5ne165x1o）：50plans、2975回合、result/prepare errors均0、114活动认领/114worker日志，未见Traceback或exit.json；抽样仍在SAPIEN导入启动阶段。尚无新增回合，不把这次快照解释为稳态吞吐。
