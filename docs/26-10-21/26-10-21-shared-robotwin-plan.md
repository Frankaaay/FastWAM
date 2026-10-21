# 三模型共享 RoboTwin 冻结 plan

用户确认以已验收FastWAM的100份冻结plan为基准。保留原文档修改，当前分支已快进336019e；本轮修改未commit/push，未部署。

正式入口统一为 `python scripts/c2r/evaluate.py rollout --root OUTPUT`，汇总使用summary。prepare/pipeline不再可选。固定清单索引是`configs/robotwin_shared_plan.json`，文件哈希锁定在evaluate.py，CPFS目录为`/mnt/workdir/frank/runs/c2r-stereotok-mono-eval-final-20261017/plans`。Clean/Randomized各5000回合，不允许缺失后补采样。

OUTPUT/manifest.json包含：model（必填fastwam/openwam/lingbot）、episodes=100、chunk_size=5、cases（直接从锁定索引选择完整phase）。FastWAM继续提供checkpoint/training_root。OpenWAM policy字段为host/port/stereo；LingBot policy字段为host/port/stereo/video_guidance_scale/action_guidance_scale，实际推理参数仍须核对原模型服务配置。不同模型/checkpoint使用不同OUTPUT目录，不能复用历史结果。

执行环境需要FastWAM src、OpenWAM仓库、XPolicyLab父目录可导入；模型服务分别使用原有环境和启动方式。不要把FastWAM模型推理超参数复制到其他模型。现有H800 launcher仍配置FastWAM解释器，Open/Ling正式请求的客户端依赖与PYTHONPATH需在发布新请求时实际验收，不能仅修改model字段便宣称可运行。

结果绑定plan哈希与完整manifest哈希。缺失/篡改plan、预算变化、模型混用、seed/instruction不一致直接失败。环境Python/NumPy RNG独立于策略；结果保留初始场景与RGB指纹。

本机9项共享测试、既有12项通过；服务器CPU验证真实100份/10000回合hash及Open4项/Ling5项测试。尚未进行真实GPU/SAPIEN三模型联调，不能作为正式上线验收。完整证据与作业ID见 `C:/Project/StereoTok/docs/26-10-21/26-10-21-robotwin-shared-plan.md`。

## 授权发布与单卡验收

2026-10-21 用户明确允许提交/推送本次变更和1张闲时GPU串行验收。具名提交保留其他dirty文件。Fast测试入口tests/check_shared_plan_runtime.py仅用于真实Clean/Randomized初始化和两次原生策略调用，不生成正式分数；固定索引LF换行，LingBot空动作步维度拒绝。实际远端结果另行追加。
