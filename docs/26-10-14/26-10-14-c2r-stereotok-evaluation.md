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