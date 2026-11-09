# clean Stage3 u8000 codec 接入

用户授权为 OpenWAM DROID+ABC 双目 midtrain 做代码适配与 smoke。当前分支 frank/fastwam-stereotok-robotwin，基线 a239a95760bef9e4ca3eeafd02b18542e5019707，fetch 后与 origin 同 SHA。已有 fastwam.py、test_robotwin_stereotok.py 及文档修改不属于本次补丁，未覆盖/暂存/提交。

## 权重与最小修改

PPU 权重：/mnt/workdir/frank/checkpoints/stereotok-stage3-clean-u8000-20261109/stereo.pth，4,606,154,269 bytes，SHA256 4c024b1dc9d9280e94a753ab9ccb1de15cc3c24659a1ec5aa86f94a3805f3503。迁移 dlc1yuagp75jp3r7 已输出完整长度及 SHA 校验成功。773 个导出 tensor 此前与指定 ckpt-u8000 的对应 tensor 逐项一致。

checkpoints.py 增加该精确哈希与 stage=3 的绑定准入，保留原 Stage2 哈希；不泛化允许任意权重。student.py 按 e93a3e1e785926a339b788881c9e211b6a441ec8 的 MetricHead 结构实现浅层视差头。加载时先构建 native VAE，再加入 fusion、geometry branch、metric head，最后 strict=True 完整加载。stereotok_vae.py 为新权重记录对应源版本。encode/decode 运算路径未改。

## 验证与状态

使用 C:/Project/StereoTok/.venv/Scripts/python.exe 执行 tests/test_stereotok_stage3_loading.py：通过。测试以小型模型验证 Stage3 全参数 roundtrip、冻结、metric 输出形状、错误 stage 拒绝、缺失 metric tensor 严格失败；不等于 4.6GB 实际 codec strict load 或 PPU 前向。

OpenWAM 侧新增 reader/Alpha codec 替换及部署骨架适配；本地 5 个对应 CPU 合同测试通过。完整 checkpoint strict load、真实数据两路左目/监督逐项对比、PPU 前后向、保存/重载尚未运行。

AGENTS.md 指定 Local edits → user pushes → server pulls，不通过 scp 上传未提交源码。当前修改未 commit/push，须发布精确版本后在个人 clean clone 运行 GPU smoke。未启动正式训练。

用户现已明确授权本次改动commit/push（覆盖此前user-push操作主体限制），以及仅PPU1.5 8卡smoke。正式32卡训练须smoke完成后再次确认。
