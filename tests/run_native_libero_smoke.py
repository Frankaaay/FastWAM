"""Run the native evaluator with eight policy actions, never a score evaluation."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "libero"))
from experiments.libero import eval_libero_single as evaluator


def main():
    original_limit = evaluator._get_max_steps
    original_episode = evaluator.run_single_episode
    evidence = []

    def smoke_limit(suite):
        original_limit(suite)  # Retain official suite validation.
        return 8

    def tracked_episode(*args, **kwargs):
        bound = inspect.signature(original_episode).bind(*args, **kwargs)
        cfg = bound.arguments["cfg"]
        assert cfg.data.train._target_.endswith("RobotVideoDataset")
        assert list(cfg.data.train.video_size) == [224, 448]
        assert cfg.data.train.context_len == cfg.model.tokenizer_max_len == 128
        assert cfg.data.train.processor.action_output_dim == 7
        assert cfg.data.train.processor.proprio_output_dim == 8
        env = bound.arguments["env"]
        original_step = env.step
        completed = 0

        def counted_step(action):
            nonlocal completed
            a = np.asarray(action)
            assert a.shape == (7,) and np.isfinite(a).all()
            result = original_step(action)
            completed += 1
            return result

        env.step = counted_step
        try:
            result = original_episode(*args, **kwargs)
        finally:
            env.step = original_step
        policy_steps = completed - int(cfg.EVALUATION.num_steps_wait)
        assert 0 < policy_steps <= 8
        record = dict(suite=str(cfg.EVALUATION.task_suite_name),
                      policy_steps=policy_steps, full_benchmark=False,
                      native_evaluator=True, task_success=bool(result[0]))
        evidence.append(record)
        output = Path(str(cfg.EVALUATION.output_dir))
        output.mkdir(parents=True, exist_ok=True)
        (output / "smoke-action-evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8")
        return result

    evaluator._get_max_steps = smoke_limit
    evaluator.run_single_episode = tracked_episode
    evaluator.eval_single_process()
    assert evidence, "Evaluator returned without executing a policy episode"
    print("NATIVE_LIBERO_ACTION_SMOKE_PASS", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
