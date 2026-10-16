# Checkpoint retention
User approved original weights cadence/history plus latest-only full resume state. RoboTwin save_every is now 2500 optimizer steps, matching the official recipe. New state completes before old resume payload files are individually unlinked; historical weights remain. Batch256, 5 epochs,10725 steps and u15853 are unchanged. Syntax verified; targeted retention check pending.
