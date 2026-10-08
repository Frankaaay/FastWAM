"""Probe the official task reuse and five-episode render-cache lifecycle.

Experimental branch only: replay open_laptop episodes 0..5 with the original
saved trajectories. Warmups write diagnostics without acceptance markers;
episode5 must pass the unchanged production stereo verifier to be accepted.
This does not reproduce the collector's preceding motion-planning phase.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robotwin-root", "source-root", "output-root", "task-config-path",
                 "canonical-archives-root"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    root = Path(args.robotwin_root).resolve(strict=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "script"))
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    importlib.import_module("test_render").Sapien_TEST()

    task_module = importlib.import_module("envs.open_laptop")
    task = task_module.open_laptop()
    task_module.open_laptop = lambda: task
    original_close = task.close_env

    def collector_close(clear_cache=False):
        original_close(clear_cache=(task.ep_num + 1) % 5 == 0)

    task.close_env = collector_close
    script = Path(__file__).resolve().parents[1] / "scripts/render_robotwin_stereo.py"
    spec = importlib.util.spec_from_file_location("collector_lifecycle_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_verify = module.verify_episode
    records = []

    class WarmupComplete(Exception):
        pass

    def verify(*positional, **options):
        record = dict(episode=task.ep_num, warmup=task.ep_num < 5)
        try:
            result = original_verify(*positional, **options)
            record.update(strict_verified=True, result=result)
        except (ValueError, AssertionError) as error:
            record.update(strict_verified=False, exception=str(error))
            if task.ep_num == 5:
                raise
        finally:
            records.append(record)
            destination = Path(task.save_dir) / "collector-lifecycle-diagnostics.json"
            destination.write_text(json.dumps(dict(
                diagnostic_only=task.ep_num < 5, task_reused=True,
                render_cache_clear_frequency=5, planning_history_reproduced=False,
                records=records), indent=2))
            print(json.dumps(record), flush=True)
        if task.ep_num < 5:
            # Prevent production replay from publishing warmup verified.json.
            raise WarmupComplete()
        return result

    module.verify_episode = verify
    for episode in range(6):
        options = argparse.Namespace(**vars(args), tasks=["open_laptop"],
                                     replan_missing=False, head_baseline_m=0.06,
                                     left_wrist_baseline_m=0.02, right_wrist_baseline_m=0.02,
                                     num_shards=50, shard_index=episode,
                                     pixel_mae_limit=1.0, joint_atol=1e-5)
        try:
            module.replay(options)
        except WarmupComplete:
            pass


if __name__ == "__main__":
    main()
