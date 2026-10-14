# C2R StereoTok 全量闭环评估与优化

用户授权优化、必要入口/测试、commit/push、H800 多卡短测并直接启动全量评估；不改训练、不删除历史产物。目标 final step10725，50任务×clean/randomized×100回合，unseen、EEF20/minmax、horizon/replan32、10去噪、shift5、CFG1、模型seed0、专家候选seed100000起、三组双目6/2/2cm。动作合同来自本次C2R训练；不能直接使用旧qpos/replan24入口。

基线：本地frank/c2r-stereotok-2461，origin Frankaaay/FastWAM，9a7c95cf7b7a7ba580ee55c6c90596d470540dbc，fetch成功且HEAD与upstream一致。原有两份26-10-13文档dirty保留，不纳入本次提交。StereoTok DLC dlc1ife1hnwmbq9s与Wan均Succeeded。Stereo completed.json=10725/10725、exit.json=0、final c2r_complete=world8/BS8/GA4、lr1e-6；五个weights文件存在。最终权重全tensor有限及完整state进一步核验中，不能把文件存在当完整载入验收。W&B thob0ldr日志确认step10725与同步完成。

旧H800评估记录：44GPU/88环境，Stereo全量6h31m，Wan5h13m；task-condition100回合粒度仍存在长尾可能，尚未完成阶段计时证明占比。

实现：复用现有Camera双目扩展及render_device定向绑定；policy复用指令embedding，动作队列非空时不请求新观察；新evaluate入口先做官方专家筛选和第二次scene setup，记录有效seed/指令/Python和NumPy随机状态，再动态领取小批回合、持久复用模型。规划仅对原UnStableError跳过种子，意外运行错误失败，不静默缩小覆盖。禁用仅用于视频产物的录像，不改RGB策略观察、仿真步数、成功判定或推理步数。原专家筛选成功判断保留。计划固定Python RNG0以使语言选择可复现（上游未固定Python RNG）；后续Wan必须复用同一plan。

最小CPU验证：独占claim、完整覆盖才聚合、重复episode拒绝，共2测试通过；git diff --check通过。实际GPU吞吐、并发显存、输出数值一致性和闭环smoke尚待验证。DSW dsw-dqg0h974kfthja8hs9已按授权启动，独立8/多卡正式评估未提交。

远端准备：独立clone FastWAM-C2R-eval-20261014，已有Frank assets与task_config仅符号链接引用并精确写入.git/info/exclude；未修改原配置或训练clone。直连fetch首次停滞，核对本任务PID后定向结束，有界重试成功。初次仿真启动发现Device在sapien顶层，已按已有h100_robotwin_gt实现定向修复；未更换依赖。该次尚未领取任务，重试不重复episode。
