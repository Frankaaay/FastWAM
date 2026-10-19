# FastWAM LIBERO paired-data evaluation correction

## Scope and source

User authorized one idle GPU, diagnosing the 0/8 smoke, correcting evaluation, and a larger bounded smoke; no full benchmark or training. Local branch `frank/fastwam-stereotok-robotwin`, HEAD `1e248a796150f529e4138ed51d47442f78bb2c0a`; origin Frankaaay/FastWAM, fetch verified. Existing unrelated changes in `docs/26-10-16/26-10-16-libero-paired-initialization.md` were preserved. No commit/push authorized or performed.

DSW `dsw-gwnr2hoew94agtjp7i` was restarted with its verified one-GPU L20Y / priority4 / ForceQuotaOverSold allocation. Work uses Frank's CPFS paths. Training output: `/mnt/workdir/frank/runs/eighteen-preparation-20261014/twelve-ready-u15853/train-fastwam-libero-wan-original20k-20261018`; checkpoint `checkpoints/weights/step_020000.pt`.

## Verified cause

The paired-data exporter (`XPolicyLab/utils/export_libero_stereo_lerobot.py`, published source `912114b...`) preserves simulator actions and OpenCV images, unlike the assumptions in the original evaluator. The source renderer explicitly records pre-action physics state and `macros_image_convention=opencv`.

- Restoring the exact first source demonstration state: raw simulator images are OpenGL-oriented; vertical flip matches both stored camera images exactly (pixel MAE 0). Original 180-degree rotation instead gives MAE 38.855 and 44.849, horizontally mirroring the training observations.
- Exported action and state match HDF5 to float32 tolerance (max errors 2.90e-8 / 1.19e-7). Gripper is native -1=open, +1=close. The original evaluator's inverse 0/1 mapping reverses all 32 audited gripper actions.
- Simulated proprio matches stored training state exactly. Evaluation stats and training stats have identical SHA256. Action normalization roundtrip max error 5.96e-8.
- Checkpoint audit strictly loaded all 1,649 MoT tensors and proprio state with zero missing/unexpected keys; tokenizer binding matches. Live text encoding matches training cached context exactly, including padding behavior.

## Correction

Only `experiments/libero/eval_libero_single.py` changes. Existing `meta/stereo_provenance.json` identifies the paired export; mixed/unknown conventions fail. Paired observations are converted to OpenCV orientation and use the saved processor's deterministic camera transforms/order. Render resolution comes from training metadata (128 here) rather than native evaluator's fixed 256. Paired native gripper actions are not inverted a second time. Original converted data keeps its existing convention. Evaluation checkpoint loading additionally requires strict MoT/proprio loading.

No training data, statistics, weights, success conditions, seeds, action horizon, replanning rate, denoise steps or episode caps were changed.

## Validation and publication boundary

Remote production clone remains clean at the published SHA. Under the repository's local-edit/user-push/server-pull workflow, this smoke loads the exact edited functions into an isolated diagnostic process, not onto disk in the production clone. Patch SHA256 `9ea2bc4588c3df8e20dda1d81d52f44117dca897299960ffc30fd2a301e0ce4d` (LF source) binds the tested candidate. Publication still requires the user's Git workflow.

Independent regression passed: preprocessed training/evaluation pixels bit-identical for both OpenGL and OpenCV simulator conventions; all 32 native gripper actions recovered; action max error 5.96e-8. The old failure is reproduced numerically before the correction.

Expanded smoke uses task IDs 0, 4, 8 in each of Spatial/Object/Goal/Long, two initial states per task (24 total), seed42, denoise10, CFG1, horizon32, replan10, wait30, caps400/700. Original 8 failed cases are included. No full benchmark started. First corrected original case succeeded in 6.66s, compared with failure at the original cap.

Evidence root `/mnt/workdir/frank/runs/fastlib-contract-audit-20261019` (`audit.json`, `orientation.png`); logs `fastlib-contract-audit-20261019-v2.log`, `fastlib-contract-regression-20261019.log`. Smoke root `/mnt/workdir/frank/runs/fastlib-corrected-smoke-20261019`, adjacent `.log`, `episodes.json`, `effective.yaml`, per-task result JSON/videos, final `summary.json`. Local request/diagnostic evidence: `C:/Users/Frank/AppData/Local/AliyunCLI/h800-requests/frank-rolling-two-20261018/fastlib-{contract-audit,contract-regression,corrected-smoke}.py` and corresponding launch files. Final aggregate and resource release are recorded below when verified.

## Final acceptance

24/24 completed successfully (Spatial6/6, Object6/6, Goal6/6, Long6/6); original eight failed cases are now8/8. All twelve native result JSONs were read back and checked for two completed trials each. This is bounded smoke, not a full-benchmark success-rate claim.

Mean episode seconds: Spatial7.509, Object9.222, Goal6.460, Long17.052. Warmed complete-policy-request median0.332629s, mean0.336192s. The correction changes task success, not inference speed. Extrapolating this small sample yields5.589h for2000 episodes, excluding startup/video/task variation; the old17–20h estimate based on failed max-length episodes is obsolete. No full eval started.

`verified-summary.json` and all results/videos persist in the smoke root. Local `fastlib-corrected-results.json` retains audit, episode timings and native results. `git diff --check` passes. Only the evaluator and this new record were changed for this task; pre-existing edits remain. StopInstance requested after completion; final resource status is recorded below. Production Git publication has not occurred.

Final GetInstance confirms dsw-gwnr2hoew94agtjp7i Stopped. The temporary idle GPU is released; CPFS results are retained.

## Authorized publication and stereo evaluation, 2026-10-19

User explicitly authorized commit/push and two one-GPU idle full evaluations after validation. The native evaluator now renders real right views using the training exporter's camera-local +X baselines (agentview 0.06m, wrist 0.02m), restores camera calibration before simulation resumes, composes the same four-camera libero_stereo inputs, and passes input_image_right into the policy. No duplicate-left or synthetic stereo. Wan and StereoTok keep identical evaluation protocol. Added rendering and tensor paths still require remote parity and rollout acceptance before full launches. This record will be updated with measured results.
