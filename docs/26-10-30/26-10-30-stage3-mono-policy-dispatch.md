# Stage3 mono policy codec dispatch repair

The authorized Stage3-u8000 mono-left RoboTwin training dlcx2gg0bfoimvwr failed before any optimizer update: FastWAM._encode_video_latents selected native Wan encoding from requires_stereo=False and accessed StereoTokVAE.scale, which does not exist. The previous CPU codec-load checks missed the policy call site.

Keep the existing frozen codec implementation and training contract. Select the StereoTok codec ABI from its existing binding metadata, independently of whether a right eye is required. Apply the same distinction to single-image encoding and codec input precision. Stereo still rejects missing right input; mono still rejects supplied right input. Native Wan keeps its compiled model.encode/scale path. No new configuration, dependencies or fallback path.

Target branch frank/c2r-stereotok-2461, baseline 950e03b91ed5c8f43324eb7c9d7acb4f19943b06, fetched origin equals HEAD. Existing unrelated historical document edits retained. Regression test in tests/test_c2r_stereotok.py executes the actual video and image encoding methods for mono/stereo/Wan; verifies dispatch, image shape, output dtype, missing/extra right-eye rejection, and native Wan isolation. CPU test requires remote Torch; local default Python lacks Torch. GPU policy training validation remains distinct from the CPU dispatch check.

Requeue uses eight idle GPUs, unchanged original Wan-init recipe, Stage3 checkpoint and W&B run zxqwwhoi (resume=must). Preserve failed output/logs and use a distinct retry output directory. Remote source must be a clean clone of the pushed repair SHA. Receipts and validation logs: C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/repair-20261030.

Validation: 0-GPU DLC dlcxigcujyvuc177 executed the exact modified methods and regression test under Torch; POLICY_DISPATCH_TEST_PASS, 1 test passed. git diff --check passed. No GPU forward/backward evidence yet.
