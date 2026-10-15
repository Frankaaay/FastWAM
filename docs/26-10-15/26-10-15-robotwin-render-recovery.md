# RoboTwin H100 渲染中断恢复

- 现场：Job16110在2273 move_stapler_pad超时后退出，单条exit124触发srun kill-on-bad-exit。部分HDF5和7路视频存在，说明已进入回放，不能仅凭没有FRAMES30断言初始化卡死。
- 保留全部已验收和partial。新增120秒重复调用栈及setup/replay阶段日志，复用已有基准入口单独重放2273；不先扩大超时或跳过样本。
- 本地分支frank/fastwam-stereotok-robotwin，9e5f82c，fetch后与origin一致；保留原有untracked。目标脚本/相机自5510750没有其他改动。
- 单卡Job19553复现：180帧、seed27、模型048_stapler/base3。120秒调用栈位于camera.py:322 RenderCameraGroup.take_picture，处于play_once抓取动作的take_dense_action；GPU100%。取消已定位的探针后，用tests入口仅禁用group做同轨迹原串行捕获对照，尚不能断言驱动内部具体缺陷。
- 串行对照19555也出现120秒长等待，调用栈在右目camera.take_picture；因此不能归因为CameraGroup独有问题。增加前30帧开始序号/时间，并把120秒堆栈计时按帧重置，以区分某帧死锁与极慢渲染。保留原捕获质量参数。
- 明确现场：旧7路MP4均可解码且各21帧；带帧边界Job19557在索引21稳定停止，前21帧6–12秒每帧。资产0–6均无非有限顶点/法线/切线或退化三角面。
- 定向原因：2273 pad为Black；实测默认RenderMaterial黑色时specular=metallic=transmission=0。camera.rchit的diffuse/specular/transmission概率总和0仍做除法，未选中BSDF后L及attenuation为0；camera.rgen未终止零能量路径，会继续trace零方向。新增仅在首交点GT保存后终止精确零能量路径的guard，保持32SPP/path8/OIDN、场景材质及动作不变。固定原rgen SHA256 c5d4f1e21e1da1b90bcdd2f2124e87ec72fd5191f4434620abbaaf718f2fe75e；复制到本次个人runtime的独立版本目录，不修改安装包，不热改运行环境。尚待实际回放验证。
- 修复Job19696：2273完整180帧127.25秒通过任务成功、逐帧qpos、标定、JPEG和7路MP4完整解码。rgen修复后SHA256 0d3978c0ff192762201c6db5605580f311e2a8a4add091b94cde5bd2c226def9；前22帧从6–12秒/帧变为约0.45–0.5秒/帧，越过确定性停滞点。此对照支持零能量/零方向射线为触发原因，而非单纯增加超时。
- 修复后前21帧与旧有效MP4不是逐字节相同：7路RGB平均绝对差0.107–0.582/255，最大像素差42–227。终止无贡献路径会改变后续随机数消耗，不能宣称RGB bitwise一致；仍保持32SPP/path8/OIDN和材质。正常样本的全部数值GT/标定/动作与RGB差异继续核对。
- 恢复调度改为单条最多两次独立attempt，失败记录并继续下一条；连续三个不同episode失败才视为系统性故障停止。最终发布先明确检查未验收ID，任一缺失均不得生成ready。新增Linux测试验证隔离/重试/后续领取和缺样本阻止发布，以及系统性故障及时停止。
- 正常样本0000回归：Job19696完整140帧的77个数值HDF5数据集与旧基线逐元素一致，覆盖GT、动作、标定等；7路RGB的MAE为0.147–0.532/255，不宣称逐字节相同。报告位于输出根benchmarks/19696/normal-140-frame-comparison.json。
- 冻结774a985eb6e21405f4655ab280e5fe869f2e0d6c，Job19698先通过全部4项Linux测试，再对2258/2267/2270/2282四个黑垫子样本做单卡四进程回归。恢复Job19700以afterok:19698依赖提交，8卡32进程，复用1184条已验收结果，动态领取剩余1316条。提交前bash -n、sbatch --test-only通过，远端clone clean；未热更新运行clone。
- Job19698全部成功：598帧420.7225秒，1.42136 FPS/GPU；四条均完成完整验收。Job19700于2026-10-15 09:19:53北京时间在xn01-gpu1-0050开始运行。启动健康检查：32日志、8个PCI设备各4进程，32进程均开始出帧，无Python traceback；2273已越过旧索引21卡点至25。GPU瞬时利用率7卡100%、1卡78%，显存约25–26GiB/卡。
- 剩余主体估计从09:20起5–8小时（约14:20–17:20），最终聚合验收另留15–30分钟；此为当前基准和剩余176029帧估算，不是完成承诺。正式作业日志/gpfs/jiuquyun/home/Frank/logs/robotwin-gt-19700.out，单条日志位于输出根logs/*-19700.log，输出根/gpfs/jiuquyun/projects/Frank/robotwin-stereo-gt-20261014。已验收1184条保留；2500条最终ready仍待作业完成。仅做启动健康检查，不持续轮询。
