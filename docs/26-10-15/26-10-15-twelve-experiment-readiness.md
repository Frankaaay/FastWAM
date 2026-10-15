# 十二项实验准备

用户要求准备 FastWAM/OpenWAM/LingBot × RoboTwin/LIBERO × Wan/StereoTok 的剩余 smoke 与配置核对，RoboDojo 暂停；不启动正式训练。

本地分支 frank/fastwam-stereotok-robotwin，基线 981f97e82cefc79658d8f823a240ac84b7bdfa6a，与 fetch 后 origin 同步。原有未跟踪文件保留。

本轮具名修改：src/fastwam/models/wan22/stereotok/checkpoints.py、src/fastwam/models/wan22/stereotok_vae.py、src/fastwam/datasets/lerobot/robot_video_dataset.py、tests/test_robotwin_stereotok.py。

- u15853 实际导出只有 format/stage/semantic/geometry/state_dict，无 provenance.generator_updates。以用户选定文件 SHA256 61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae 锁定，不伪造计数、不放宽 strict load。
- codec 保留 RoboTwin 384×320 分区，加入 LIBERO 224×448 左右两个相机的分区；支持已有单相机大小，要求匹配左右目、16 对齐及 1+4k 帧。
- RobotVideoDataset 增加 LIBERO 四相机顺序约束与独立左右横拼，沿用 processor、归一化和原始 action/state 流；必须有 stereo_provenance.json。生产数据转换、正式配置和真实 reader smoke 尚待完成。
- 验证：7 个跨仓库修改 Python 文件 AST 通过；定向执行真实 load_student 函数，对非选定文件在 torch.load 前拒绝通过；git diff --check 通过。Windows 无 torch，未运行完整 torch 单测，未将静态结果算 GPU smoke。

待发布此改动后，服务器从 Git 拉取精确 SHA，再做真实 strict load、数据接入和联合模型 smoke。FastWAM AGENTS.md 明确 Local edits → user pushes → server pulls；既有明确 fork/push 批准只列 OpenWAM/XPolicyLab，FastWAM 本轮发布权限需补齐。没有 scp 未提交源码到服务器绕过该规则。
