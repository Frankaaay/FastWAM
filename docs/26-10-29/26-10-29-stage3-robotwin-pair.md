# Stage3 RoboTwin mono / stereo pair

2026-10-29 Asia/Shanghai. User authorized checkpoint-parameter loading, necessary validation and commit/push, followed by two 8-GPU idle-resource runs. No native Wan rerun. Baseline remains C2R clean 2500, shared Wan-init, seed 42, global batch 256, 5 epochs / 10725 updates, original LR table and data/augmentation.

Input: `/mnt/workdir/frank/runs/stage3-u8000-wan-eval-20261028/input/stage3-u8000.ckpt`. Read-only DLC `dlc1yime4rubcvi1` verified training format, stereo-v3 architecture, Stage3, generator_updates=8000, 13799251677 bytes. Historical provenance SHA256 is 5388b175591e0045f06b52fb0cfa5f769f1c03c79086c0d71661808e6b8df5e1; deployment must recompute it.

Loader accepts explicit Stage2/Stage3 checkpoints without a hardcoded checksum allowlist. It retains strict RGB/fusion/geometry weight loading; semantic and latent-alignment loss readouts are excluded because inference never invokes them. Reuse the existing FastWAM mono/stereo codec implementation and mode plumbing. Save tokenizer binding for mono as well as stereo. Launcher accepts checkpoint, input mode and independent output directory. OpenWAM uses this same released vendor codec.

Existing unrelated 10/13 document edits are preserved and excluded from this commit. No original runs/checkpoints are overwritten. Local full data test requires unavailable cv2; targeted checkpoint/mode tests execute the actual functions with tiny modules, and AST/CLI/diff checks precede publication. Remote strict load, real input and online W&B checks remain required before formal training; final job IDs and results are recorded in StereoTok's task document.
