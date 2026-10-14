"""Execute the native LingBot LIBERO client with a bounded real action count."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


class SmokeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--actions", type=int, default=8)
    args = parser.parse_args()
    assert args.actions > 0
    sys.path.insert(0, str(args.repo))
    spec = importlib.util.spec_from_file_location(
        "lingbot_native_libero_client", args.repo / "evaluation/libero/client.py"
    )
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    original_construct = client.construct_single_env
    original_step = client.env_one_step
    environments = []
    completed = 0

    def tracked_construct(*a, **kw):
        env = original_construct(*a, **kw)
        assert env is not None
        environments.append(env)
        return env

    def tracked_step(env, action):
        nonlocal completed
        import numpy as np
        value = np.asarray(action)
        assert value.size == 7 and np.isfinite(value).all()
        result = original_step(env, action)
        completed += 1
        if completed >= args.actions:
            raise SmokeComplete()
        return result

    client.construct_single_env = tracked_construct
    client.env_one_step = tracked_step
    try:
        client.run("libero_10", args.port, str(args.output), 1, [0, 1])
    except SmokeComplete:
        pass
    finally:
        for env in environments:
            env.close()
    assert completed > 0, "No real policy action completed"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy-smoke-evidence.json").write_text(json.dumps({
        "benchmark": "libero_10", "task_id": 0,
        "completed_policy_actions": completed,
        "requested_smoke_actions": args.actions,
        "meaning": "native policy loop, not a success-rate evaluation",
    }, indent=2))
    print("LINGBOT_NATIVE_LIBERO_POLICY_SMOKE_PASS", completed, flush=True)


if __name__ == "__main__":
    main()
