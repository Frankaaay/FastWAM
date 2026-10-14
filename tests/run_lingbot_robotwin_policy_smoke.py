"""Exercise the original LingBot RoboTwin client with bounded real policy actions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


class SmokeComplete(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "robotwin", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--port", type=int, default=29576)
    parser.add_argument("--actions", type=int, default=8)
    args = parser.parse_args()
    assert args.actions > 0
    expected = [(args.repo, "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"),
                (args.robotwin, "2eeec322d95799f537cbfe5f291a8220d965ccb8")]
    for repo, sha in expected:
        assert subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip() == sha
        assert not subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
        ).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    source = args.repo / "evaluation/robotwin/eval_polict_client_openpi.py"
    text = source.read_text()
    placeholder = 'robowin_root = Path("/path/to/your/robowin")'
    assert text.count(placeholder) == 1
    # Bind the author's explicit installation placeholder; keep client logic intact.
    text = text.replace(placeholder, f"robowin_root = Path({str(args.robotwin)!r})")
    native = {"__name__": "lingbot_native_robotwin_smoke", "__file__": str(source)}
    exec(compile(text, str(source), "exec"), native)
    import numpy as np
    state = dict(completed_policy_actions=0, policy_response_received=False)
    environments = []
    factory = native["class_decorator"]
    infer = native["WebsocketClientPolicy"].infer

    def counted_infer(self, observation):
        result = infer(self, observation)
        if "action" in result:
            assert np.isfinite(result["action"]).all()
            state["policy_response_received"] = True
        return result

    def factory_with_evidence(task_name):
        env = factory(task_name)
        environments.append(env)
        take_action = env.take_action

        def counted_action(action, *pos, **kw):
            if state["policy_response_received"]:
                assert np.asarray(action).shape == (16,)
                assert np.isfinite(action).all()
            result = take_action(action, *pos, **kw)
            if state["policy_response_received"]:
                state["completed_policy_actions"] += 1
                evidence = dict(state, action_dim=16, task="adjust_bottle",
                                task_config="demo_clean", source_shas=[s for _, s in expected])
                (args.output / "policy-action-evidence.json").write_text(json.dumps(evidence, indent=2))
                if state["completed_policy_actions"] >= args.actions:
                    raise SmokeComplete
            return result

        env.take_action = counted_action
        return env

    native["WebsocketClientPolicy"].infer = counted_infer
    native["class_decorator"] = factory_with_evidence
    import yaml
    config = yaml.safe_load((args.robotwin / "policy/ACT/deploy_policy.yml").read_text())
    config.update(task_name="adjust_bottle", task_config="demo_clean", policy_name="ACT",
                  train_config_name=0, model_name=0, ckpt_setting=0, seed=0,
                  save_root=str(args.output), port=args.port, test_num=1,
                  video_guidance_scale=5, action_guidance_scale=1)
    try:
        native["Sapien_TEST"]()
        native["main"](config)
    except SmokeComplete:
        print("LINGBOT_NATIVE_ROBOTWIN_POLICY_SMOKE_PASS", state, flush=True)
    finally:
        for env in environments:
            if getattr(env, "eval_video_ffmpeg", None) is not None:
                env._del_eval_video_ffmpeg()
            env.close_env()
    assert state["completed_policy_actions"] >= args.actions, state


if __name__ == "__main__":
    main()
