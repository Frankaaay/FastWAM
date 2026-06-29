#!/usr/bin/env python
"""Run many LIBERO/LIBERO-plus cases after loading FastWAM once."""

import json
import logging
import sys
import time
from pathlib import Path

import hydra
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    run_single_task,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_multi_process(cfg: DictConfig):
    start = time.time()
    PartialState().config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    task_list_file = cfg.EVALUATION.get("task_list_file")
    if not task_list_file:
        raise ValueError("EVALUATION.task_list_file is required for eval_libero_multi.py")
    cases = json.loads(Path(task_list_file).read_text(encoding="utf-8"))
    gpu_id = int(cfg.gpu_id)
    num_trials = int(cfg.EVALUATION.num_trials)
    logging.info("Worker gpu_id=%s: %d cases from %s (num_trials=%d)", gpu_id, len(cases), task_list_file, num_trials)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(action_horizon_cfg) if action_horizon_cfg is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    output_root = Path(cfg.EVALUATION.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.set_struct(cfg, False)

    suite_cache = {}
    n_done = 0
    n_skip = 0
    n_fail = 0
    for i, case in enumerate(cases):
        suite = case["suite"]
        task_id = int(case["task_id"])
        suite_dir = output_root / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        output_file = suite_dir / f"gpu{gpu_id}_task{task_id}_results.json"

        if output_file.exists():
            n_skip += 1
            continue

        try:
            if suite not in suite_cache:
                suite_cache[suite] = benchmark.get_benchmark_dict()[suite]()
            task_suite = suite_cache[suite]
            task = task_suite.get_task(task_id)

            expected_name = case.get("name")
            actual_name = getattr(task, "name", None)
            if expected_name and actual_name and expected_name != actual_name:
                raise RuntimeError(
                    f"task-name mismatch suite={suite} task_id={task_id}: "
                    f"expected {expected_name!r}, benchmark gave {actual_name!r}"
                )

            initial_states = list(task_suite.get_task_init_states(task_id))
            if not initial_states:
                raise RuntimeError(f"no init states for suite={suite} task_id={task_id}")
            while len(initial_states) < num_trials:
                initial_states.extend(initial_states[: num_trials - len(initial_states)])

            cfg.EVALUATION.task_suite_name = suite
            cfg.EVALUATION.task_id = task_id

            results = {
                "task_suite": suite,
                "task_id": task_id,
                "task_description": None,
                "successes": 0,
                "total_episodes": num_trials,
                "gpu_id": gpu_id,
                "category": case.get("category"),
                "difficulty_level": case.get("difficulty_level"),
                "success_episodes": [],
                "failure_episodes": [],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": 0,
            }
            case_start = time.time()
            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=suite_dir / "videos",
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            results.update(task_results)
            results["category"] = case.get("category")
            results["difficulty_level"] = case.get("difficulty_level")
            results["duration"] = time.time() - case_start
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            n_done += 1
        except Exception as exc:
            n_fail += 1
            logging.exception("case FAILED suite=%s task_id=%s: %s", suite, task_id, exc)
            try:
                (suite_dir / f"gpu{gpu_id}_task{task_id}.FAILED").write_text(str(exc), encoding="utf-8")
            except Exception:
                pass

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (n_done + n_fail) / elapsed if elapsed > 0 else 0.0
            logging.info(
                "gpu%s %d/%d done=%d skip=%d fail=%d %.1fmin %.3f case/s",
                gpu_id,
                i + 1,
                len(cases),
                n_done,
                n_skip,
                n_fail,
                elapsed / 60,
                rate,
            )

    logging.info(
        "gpu%s FINISHED done=%d skip=%d fail=%d total=%d in %.1fmin",
        gpu_id,
        n_done,
        n_skip,
        n_fail,
        len(cases),
        (time.time() - start) / 60,
    )


if __name__ == "__main__":
    eval_multi_process()
