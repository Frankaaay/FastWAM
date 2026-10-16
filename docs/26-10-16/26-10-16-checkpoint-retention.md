# Checkpoint retention
User approved original weights cadence/history plus latest-only full resume state. RoboTwin save_every is now 2500 optimizer steps, matching the official recipe. New state completes before old resume payload files are individually unlinked; historical weights remain. Batch256, 5 epochs,10725 steps and u15853 are unchanged. Syntax verified; targeted retention check pending.

User clarified original-repo retention: remove latest-only cleanup from FastWAM. Preserve all weight and full-state checkpoints, keeping interval2500 and final saves. No formal training started. Syntax verified; save method restored to original.
