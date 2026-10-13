# RoboTwin 39 条补齐：只读核验与存储阻塞

> 17:58 更正：下方 17:45 的“存储阻塞”属于仅依据 df/statvfs 的初步判断。后续实际 4 KiB 创建、写入、fsync、关闭后读回全部成功，因此不能把 0 available 直接解释为已证实 ENOSPC。真实剩余容量仍未知，详见末节；不得据旧结论要求用户删除或迁移数据。

## 2026-10-13 17:45（Asia/Shanghai）

- 任务仅补齐 39 条，保留既有 2461 条，不重渲染、不覆盖、不启动训练、不替其他 chat 切换数据。用户授权增加闲时 GPU 加快并行，本任务采用最多 8 张、每容器单 GPU，先验证采集流程再扩展。
- 实际 cwd `C:\Project\StereoTok`。已完整读取交接文件、StereoTok/FastWAM AGENTS.md 与 Frank H800 指南。
- 本地 FastWAM 分支 `frank/fastwam-stereotok-robotwin`，HEAD `f86b842af6ca600fea14a7848d87d5b9064ec529`，origin `https://github.com/Frankaaay/FastWAM.git`。fetch 成功，HEAD 与 origin 差异 0/0；tracked diff/cached diff 均空，保留用户既有 `.agents/`、`AGENTS.md`、`docs/`、`third_party/SAM2Act/` 未跟踪内容，没有 pull、commit、push 或新建分支。
- PAI profile `h800`、region `cn-shenzhen`、workspace `350607`，实核 RAM Frank `205455090823393499`；允许 `quota1bybv9m0ixu` / training / L20Y，research 优先级上限 4。未修改其他 chat 实例或 DLC。
- 新建准备 DSW `dsw-j3tbe3ui1kjny2qq8x`，创建 RequestId `01A134A4-F1B9-5877-A701-1021C08E6530`。资源 1 GPU / 24 CPU / 64 GiB / shared 16 GiB，priority 4，ForceQuotaOverSold。未申请其余 7 GPU。
- 请求及回执目录：`C:\Users\Frank\AppData\Local\AliyunCLI\h800-requests\frank-robotwin-fill39-20261013`。`dsw.json` SHA256 `4a2c8df0bbc3feb5100a9f0502bc983b856bb7279fdf7c34212ec260d2afb224`。沿用团队镜像 `pytorch:2.7.0-gpu-py312-cu128-ubuntu24.04-ngc25.03-deep-ep-4d7302ff-1770029761` 与既有只读数据/CPFS 挂载。
- 实际 SSH 到达 `root@120.77.254.228:1020`；hostname `dsw-387845-5c59f98c4-jkl9s`，GPU `NVIDIA L20Y, 81559 MiB`。`findmnt -T /mnt/workdir` 确认 CPFS。

## 阻塞与实际结果

- `df -h /mnt/workdir` 报告 10T / 100% / available 0；Python `os.statvfs` 再确认 total `10995116277760`、available `0`、free `0` 字节。没有进行试写，没有扫描整个数据盘、删除、迁移或覆盖数据。
- 新 GPU 采集尚未启动，新增成功样本为 0；没有真实补采或 reader 验证证据，无法估算采集/后处理 ETA。
- 已对本实例执行 StopInstance（dry-run 后单次提交），停止 RequestId `01A134A6-8526-5E42-8EF0-83AF4C2EF346`。GetInstance 最终实核 `Stopped`，本任务不再占用 GPU。

## 最终 39 条实读来源

- 权威纳入清单：`/mnt/workdir/frank/datasets/RoboTwin2.0/stereo-clean-fresh-v21-parallel-20261009-99f7b8c/meta/stereo_provenance.json`，SHA256 `20ff630a19fcd4b8719cdc3171cccd74a7891effe22f827c49968f969c210f66`。`source_episode_indices` 实读 2461 个且唯一，取 0..2499 的补集得 39；没有把中途 38-excluded manifest 当最终清单。
- `meta/info.json` 实读 2461 episodes / 540592 frames / 14766 videos / 15 FPS。
- task/episode/seed 映射来自既有 `runs/fastwam-render40-20261009-e2636ae/migration-manifest.json` 全 2500 账本；完整 39 行映射保存在上述本地回执目录的 `read-only-coverage.txt`。
- 39 个 canonical IDs：16, 79, 267, 275, 288, 300, 520, 566, 657, 674, 692, 757, 1028, 1038, 1044, 1270, 1433, 1452, 1453, 1455, 1459, 1474, 1482, 1485, 1492, 1499, 1516, 1524, 1535, 1537, 1545, 1822, 2155, 2238, 2256, 2365, 2386, 2394, 2459。
- 两条非 missing-dense：566=`place_object_scale/16`, seed19；1433=`place_a2b_left/33`, seed40。1433 另从 `worker-33/completed.json` 实读 exit1、verified=false、candidate=false，首根因为 renderer 的 `RuntimeError: Replay failed: place_a2b_left/33, seed=40`，发生在 merge 前成功检查。不推断其失败机制与566相同。
- 其余 37 条有既有不同 Warp/同 seed/连续历史恢复失败证据，本次未重复执行。补采须按任务补足、使用新成功示范，保留 replacement 映射，不能宣称为原始示范恢复。

## 实现门禁与下一步

- 现有 `scripts/render_robotwin_stereo.py` 是严格原轨迹回放/匹配工具，不能用旧状态匹配来验收新采集轨迹。
- `third_party/RoboTwin/script/collect_data.py` 的种子搜索无次数上限，且调用 `remove_data_cache()`，后者在 `_base_task.py` 使用 `shutil.rmtree`。不直接运行此入口，不通过热改活跃环境绕过。
- 下一步须先由用户释放持久存储或指定已授权且有容量的新持久目标。不得批量删除；不使用节点临时盘作为唯一结果存储。
- 必要最小实现方案：复用现有模拟/规划/双目采集方法，独立有界补采入口；每次尝试独立输出并保留缓存，保存六路RGB、EEF/qpos/动作、dense轨迹、场景、标定、帧时序和指令来源。只对新39条做完整验收，以引用已有2461的方式发布共同2500清单及新数据统计。新增入口及正式远端运行所需 commit/push 尚未获授权，须将明确文件范围交用户确认。

## 2026-10-13 存储复核与授权更新

- 用户明确同意本任务专用有界补采/发布脚本、必要测试和 commit/push 到当前 FastWAM 分支。此前代码授权门禁已解除；不扩展至其他 chat 的目录或环境。
- 为核验实际挂载，StartInstance 重启本任务自己的 DSW，RequestId `01A134AE-65F0-57FE-AE40-1A80FCB147AA`，仍为单 GPU。新公网端口 1019；StrictHostKeyChecking=yes，HostKeyAlias 使用此前已信任的 `[120.77.254.228]:1020` 密钥，SSH 成功。Windows ssh-keyscan 因 KEX 不支持未获取到候选密钥，没有绕过主机密钥校验。
- `/mnt/data` 为 `JuiceFS:pfs-wz9ek7d58050mjms` / fuse.juicefs，PAI GetDataset 实时确认 `adb-mount` / JUICEFS / MountAccess=RO；ListDatasets TotalCount=1。显示 1 PiB available 是挂载所报容量，不当作个人写入配额。共享原始 `/mnt/data/data/RoboTwin2.0/dataset` 确实存在且落在此设备上。
- `/mnt/workdir` 与 `/mnt/workspace` 的 mount source 均为同一 `cpfs-02000w4go7xe2f4epd6sv-000001.cn-shenzhen.cpfs.aliyuncs.com:/`，同设备号 114；不是两块独立盘。均报告 total 10 TiB / available 0。
- 精确核验 `official-v21/RoboTwin_lerobot_v21`、`canonical-replay-inputs`、`stereo-clean-fresh-v21-parallel-20261009-99f7b8c` 均在 `/mnt/workdir/frank/datasets/RoboTwin2.0/` 下，真实路径未跳转到其他文件系统，设备号均114。
- 容器 `/tmp` 落在 overlay 根文件系统，可用约179 GiB；不是另一块共享持久盘，不作为补齐结果的唯一存储。
- 最小写入验证：独占创建 `/mnt/workdir/frank/runs/robotwin-fill39-storage-probe-20261013-1755.bin`，写4096字节、flush/fsync、关闭后重新读取4096字节成功；随后只 unlink 此一个明确测试文件。没有 ENOSPC，没有删除其他文件。原“已满不可写”结论撤回；小文件成功不证明有足够容量保存39条及全部缓存。
- 本地证据 `C:\Users\Frank\AppData\Local\AliyunCLI\h800-requests\frank-robotwin-fill39-20261013\storage-audit.txt`。DescribeFileSystems 返回403 Forbbiden.Ram，不能查询底层真实剩余容量/其他NAS实例，未尝试绕过权限。
- 存储建议：当前已确认可写且持久的目标仍为个人 CPFS 路径；新39条拟独立写 `/mnt/workdir/frank/datasets/RoboTwin2.0/stereo-fill39-20261013/`（尚未创建）。旧2461引用复用，不复制整个数据集。未确认其他可写OSS前缀或第二个持久卷，不擅自改 `/mnt/data` 为RW、不上传、不迁移。需区分“当前写入成功”和“真实剩余容量充足”。

## 用户授权继续后的准备

- 用户确认“如果还有空间就继续”。原实例重新启动，实时 64 MiB 随机块写入/fsync/完整SHA256读回通过，完整散列 `017002374f974e6a4dc701ddded429e5e304dc03099d63c181ca971c30b7c8d0`；仅删除本次明确探测文件。可继续小规模真实写入，不能据此声称剩余容量已知。
- 本地 FastWAM fetch成功，当前原分支与origin仍0/0，原未跟踪内容保留。只新增 `scripts/collect_robotwin_missing.py`、`tests/test_collect_robotwin_missing.py` 和本任务文档。
- 入口复用冻结 RoboTwin runtime 的规划、模拟、六路观察、HDF5合并和官方指令生成；规划并采集在同一次模拟，成功后保留dense trajectory、原HDF5、逐帧索引/时间、下一帧qpos动作、初始/最终actor pose及指令来源。每缺口最多12个新seed，每次900秒；每次独立目录且无覆盖/缓存删除，异常基础设施错误停止，不盲换seed。
- 8个固定不相交worker，每容器1GPU；先worker0的canonical16，再放开其余。32SPP/pathdepth8/OIDN、head0.06m/wrists0.02m、240x320、15fps保持既有合同。新39做完整state/EEF/六路图像/全帧标定校验，不重验旧2461全部图像。
- 本地4项unittest、AST、git diff --check通过；尚无真实补采成功证据。用户已授权commit/push，下一步自己的clean远端clone按推送精确SHA执行。

- 补采入口 commit `b30fdf84ac576ac22dfc6ce2cbf6f7ff4824597a` 已推送；自己的clean clone `/mnt/workdir/frank/projects/FastWAM-fill39-20261013`，远端4项测试通过。`plan.json` 已发布到独立数据目录，冻结最终39映射和runtime文件hash。单条真实smoke worker0/canonical16在DSW启动，PID4628，日志 `/mnt/workdir/frank/runs/robotwin-fill39-20261013/smoke-launch.log` 与 `smoke-worker0/0016-00.log`；已进入实际GPU模拟并持续保存帧，验收尚未完成。
- 新增获授权发布脚本 `scripts/publish_robotwin_missing.py`：复用其他chat已经验证的只读2461 EEF20 manifest `/mnt/workdir/frank/runs/c2r-prep-20261013/prepared/manifest.json`（hash `9f277c833fe31d420d8e19eefb387de5865d49023513eb36f82a9ed612eaa8bd`）；仅新增39的完整QA与min/max，旧min/max和新min/max取并集等价全2500统计，rot6d仍固定[-1,1]。原2461只引用路径，不复制/重渲染/修改；按canonical ID构成每任务50条，产出兼容共同C2RDataset的manifest和指令池。未改C2R源码/环境/运行配置，未自动切换训练。发布结果仍须实际Wan/Stereo两reader同左目/action检查。

## 单条验收通过与8GPU并行启动

- canonical16/adjust_bottle，新seed1512，第一次补采exit0，141.832秒，144帧/143窗口；task success、EEF/qpos、六目全帧解码和标定通过。raw SHA256 `cf978900b0de4c55ebfabc7599d250e7370cf36059fb2bb68248234faa3cd398`；dense轨迹SHA256 `eee982600da5aa4a6f48352820a5c865839faf9feec988cd927bc65520cedb83`。左臂5段含position/velocity，右臂未运动列表为空（正常单臂任务）。新数据是替代示范，不是原轨迹恢复。
- 冻结 plan SHA256 `bdff0e19ba52ef7294f665d94d16e7338c0541704d4788645acb9aed19b2af74`；补采producer仍为b30fdf8，不热更新正在运行的clone。发布逻辑另用新clean clone。
- worker0在本任务DSW运行，PID8056，timeout21600秒；其余7个DLC各1GPU/16CPU/64GiB/shared16GiB/6小时/priority4/ForceQuotaOverSold：worker1 `dlc1eggr2jb1cygz`，2 `dlc1f0fxu3zdlykq`，3 `dlc1fafj7vl0n79b`，4 `dlc1fkf4ln5k3xqo`，5 `dlc1fuepzfhnd8wc`，6 `dlc1g4ebd7cnvt0r`，7 `dlc1gedwqzvr9tv6`。全局上限8GPU，未动其他作业。
- 所有请求/回执保存于本地 `frank-robotwin-fill39-20261013`，launcher日志 `/mnt/workdir/frank/runs/robotwin-fill39-20261013/launcher-N.log`，逐条日志/exit记录 `worker-N/`；成功输出 `stereo-fill39-20261013/canonical-NNNN/attempt-NN/accepted.json`，既有16不再采集。
- 发布器追加有界6小时等待，只有39条全部成功才发布2500；任何worker traceback停止发布。独立reader clone固定C2R SHA `633981aaa4d5b9e697774887b7cba4bc3289ea89`，逐新episode首尾及旧首中尾检查Mono/Stereo的左图、动作、proprio、prompt完全相同、右图有效。仅CPU数据reader，不加载模型或训练；文本缓存未生成，ready不能冒充训练启动门禁。
- 单条实测给出主体20–45分钟的粗估（含并行启动和少量规划重试；复杂任务可能延长），发布QA/reader另预估5–10分钟；最终以成功清单计数为准。
