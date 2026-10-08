"""Experimental probe of the official replacement collection sequence.

Run this on the dedicated experiment branch with the normal replay arguments.
Only adjust_bottle episode16 is supported: the released replacement uses seed
201, after an unsuccessful seed200 attempt with the same planner instance.
Production replay and its strict acceptance criteria remain unchanged.
"""
import importlib
import json
import runpy
import sys
from pathlib import Path


def main():
    arguments = sys.argv[1:]
    root = Path(arguments[arguments.index("--robotwin-root") + 1]).resolve(strict=True)
    sys.path.insert(0, str(root))
    module = importlib.import_module("envs.adjust_bottle")
    unstable_error = importlib.import_module("envs.utils.create_actor").UnStableError
    original = module.adjust_bottle

    class SequenceProbe(original):
        def setup_demo(self, now_ep_num=0, seed=0, **options):
            if options.get("need_plan") and not getattr(self, "sequence_probed", False):
                if now_ep_num != 16 or seed != 201:
                    raise ValueError("This probe only supports official adjust_bottle/16 seed201")
                warmup = dict(options)
                warmup.pop("recorded_actor_models", None)
                self.sequence_probed = True
                print("Probe: original collection attempt seed200 before seed201", flush=True)
                try:
                    super().setup_demo(now_ep_num=now_ep_num, seed=200, **warmup)
                    super().play_once()
                    result = dict(seed=200, plan_success=self.plan_success,
                                  task_success=self.check_success())
                except unstable_error as error:
                    result = dict(seed=200, unstable=str(error))
                finally:
                    self.close_env()
                (Path(options["save_path"]) / "sequence-probe.json").write_text(json.dumps(result))
                print(f"Probe: seed200 result {result}; reuse planner for seed201", flush=True)
            return super().setup_demo(now_ep_num=now_ep_num, seed=seed, **options)

    module.adjust_bottle = SequenceProbe
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/render_robotwin_stereo.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
