"""Experimental probe of the official replacement collection sequence.

Run this on the dedicated experiment branch with the normal replay arguments.
Only adjust_bottle episode16 is supported: the released replacement uses seed
201. By default probe seed200 first; --warmup-start-seed 0 also exercises the
planner state accumulated by earlier collection attempts.
Production replay and its strict acceptance criteria remain unchanged.
"""
import importlib
import json
import runpy
import sys
from pathlib import Path


def main():
    arguments = sys.argv[1:]
    warmup_start = 200
    if "--warmup-start-seed" in arguments:
        position = arguments.index("--warmup-start-seed")
        warmup_start = int(arguments[position + 1])
        del arguments[position:position + 2]
        sys.argv = [sys.argv[0], *arguments]
    if not 0 <= warmup_start <= 200:
        raise ValueError("Warmup seeds must start between 0 and 200")
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
                results = []
                for prior_seed in range(warmup_start, 201):
                    try:
                        super().setup_demo(now_ep_num=now_ep_num, seed=prior_seed, **warmup)
                        super().play_once()
                        result = dict(seed=prior_seed, plan_success=self.plan_success,
                                      task_success=self.check_success())
                    except unstable_error as error:
                        result = dict(seed=prior_seed, unstable=str(error))
                    finally:
                        self.close_env()
                    results.append(result)
                    (Path(options["save_path"]) / "sequence-probe.json").write_text(json.dumps(results))
                    print(f"Probe: {result}; retain planner for next seed", flush=True)
            return super().setup_demo(now_ep_num=now_ep_num, seed=seed, **options)

    module.adjust_bottle = SequenceProbe
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/render_robotwin_stereo.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
