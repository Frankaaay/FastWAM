# u8000 mono-left RoboTwin evaluation

User authorized fixed16 GPU evaluation and launch ETA on 2026-10-17. Reuse current frank/c2r-stereotok-mono-u8000 worktree. Git fetch and pull --ff-only passed, starting HEAD9ad69018548728b9856f378514a4508c0cb90833, clean.

Training Job dlc13284y3cnrqx5 succeeded 2026-10-17 02:27:17 Asia/Shanghai; pod exit0, W&B final engine_steps10725, loss0.06802/grad0.12837. Durable acceptance pending before rollout.

Reuse evaluated pipeline from source921bb0c4f598cf858d97b8f6ac64f22e2dc98987: evaluate.py, launch_eval_h800.py, policy prompt cache, associated tests and two RoboTwin rendering files. Preserve native mono codec/model changes. Policy keeps FP32 normalized StereoTok input independently of requires_stereo; mono never reads right-eye images. Scene/render configuration stays identical to previous comparison.

Contract: final step010725, same100 frozen plans copied byte-for-byte from prior Wan eval root, 50tasks x clean/randomized x100, unseen instructions, EEF20, horizon/replan32,10denoise,shift5,CFG1,rt32spp; chunk5,2workers/GPU. Fixed16 only, ForbiddenQuotaOverSold, research quota quotan2i9e2t9tqr,priority4. No extra idle resources. Each2GPU job48CPU/256Gi/64Gi shared. No training restart, old results or claims altered.
