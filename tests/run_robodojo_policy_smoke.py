"""Bound an official RoboDojo evaluation and count completed simulator actions."""
import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--actions", type=int, default=8)
    args, native_args = parser.parse_known_args()
    assert args.actions > 0
    sha = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == "09bd5f5e762b949f5858ece31347ce62057b3a5e", sha
    assert not subprocess.check_output(
        ["git", "-C", str(args.repo), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    entry = args.repo / "src" / "eval_client" / "main.py"
    sys.path[:0] = [str(args.repo), str(args.repo / "XPolicyLab")]
    sys.argv = [str(entry)] + native_args
    namespace = runpy.run_path(str(entry), run_name="robodojo_smoke")
    native_main = namespace["main"]
    original_factory = native_main.__globals__["create_eval_env"]
    completed = 0

    def tracked_factory(*factory_args, **factory_kwargs):
        env = original_factory(*factory_args, **factory_kwargs)
        assert env.num_envs == 1
        original_actions = env.take_action_batch
        original_run = env.run_eval

        def tracked_actions(*action_args, **action_kwargs):
            nonlocal completed
            before = sum(env.take_action_cnt)
            result = original_actions(*action_args, **action_kwargs)
            delta = sum(env.take_action_cnt) - before
            assert delta >= 0
            completed += delta
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            args.evidence.write_text(json.dumps({
                "source_sha": sha,
                "task": env.task_name,
                "policy": env.policy_name,
                "completed_simulator_actions": completed,
                "requested_smoke_actions": args.actions,
                "meaning": "short policy loop, not a benchmark score",
            }, indent=2))
            print("ROBODOJO_COMPLETED_SIMULATOR_ACTIONS", completed, flush=True)
            # IsaacSim can terminate the process inside simulation_app.close(),
            # so emit action evidence before returning to the native shutdown.
            if completed >= args.actions:
                print("ROBODOJO_SHORT_POLICY_LOOP_PASS", completed, flush=True)
            return result

        def short_run():
            env.step_lim = min(env.step_lim, args.actions)
            return original_run()

        env.take_action_batch = tracked_actions
        env.run_eval = short_run
        return env

    native_main.__globals__["create_eval_env"] = tracked_factory
    native_main()
    assert completed > 0, "No real policy action completed"
    print("ROBODOJO_SHORT_POLICY_LOOP_PASS", completed, flush=True)


if __name__ == "__main__":
    main()
