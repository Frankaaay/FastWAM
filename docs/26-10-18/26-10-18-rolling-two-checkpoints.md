# 2026-10-18 Rolling last two checkpoints

User explicitly replaces historical-all/latest-one retention with latest two completed recoverable checkpoints for all downstream runs. Cadence, batch, optimizer and training budget are unchanged. Cleanup happens after completion and removes individual files before empty directories; incomplete saves do not displace completed checkpoints. FastWAM waits for all ranks before the completion marker. OpenWAM uses keep_last_k_ckpts=2 for weights and states. LingBot removes old paired exports/full states together.

Validation: executed actual save/retention code with tiny payloads. Three successful saves retain steps 2 and 3; interrupted third FastWAM save preserves steps 1 and 2. Incomplete Open/Ling state is excluded. These are filesystem/retention tests, not model-training tests. Active jobs must be restarted on the published code; W&B run IDs remain unchanged.
