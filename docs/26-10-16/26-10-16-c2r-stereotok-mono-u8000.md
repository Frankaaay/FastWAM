# StereoTok u8000 left-only C2R training

2026-10-16 user authorized an isolated worktree/branch from original baseline 9a7c95cf7b7a7ba580ee55c6c90596d470540dbc, necessary mono adaptation, commit/push and fixed 8-GPU H800 training.

Branch: frank/c2r-stereotok-mono-u8000. Worktree: C:/Project/FastWAM-C2R/.worktree/c2r-stereotok-mono-u8000. Existing dirty parent checkout preserved.

Contract: same u8000 frozen student, 2500 episodes, 549034 windows, 5 epochs/10725 updates, BS8/GA4/world8/global256, shared initialization v2, original optimizer/LR/precision/augmentation/text/actions. Save every 2145 updates and retain all five checkpoints. Only three camera left eyes are read; native encode_posterior(left) skips stereo matching/fusion. No duplicated eyes, no latest u15853 changes. Codec binding records input_mode=mono-left. Online W&B required.

Changed existing codec, FastWAM codec routing, dataset stereo flag and unique output name. Added targeted mono routing test; retained existing data parity tests. Local AST and diff checks pass; local Python lacks torch, so no local numerical test claimed. H800 validation and job status pending.

H800 profile h800, region cn-shenzhen, workspace350607, RAM Frank205455090823393499, research quota quotan2i9e2t9tqr. Current role permits priority6; retain original priority4 and ForbiddenQuotaOverSold. Single node8GPU/96CPU/1024Gi/64Gi shared, original image and environment, maximum10080min. No training submitted yet.
