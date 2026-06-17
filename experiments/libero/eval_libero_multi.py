#!/usr/bin/env python
# ----------------------------------------------------------------------------
# eval_libero_multi.py — LIBERO-plus 用的"加载一次模型 → 循环跑一批 case"worker
#
# 为什么需要它:LIBERO-plus 有 10,030 个 case,每个只 rollout 1 次。沿用
# eval_libero_single.py(一进程一 task)会让每个进程都花 ~70s 加载 5B 模型只为
# 跑 1 个 ~40s 的 trial —— 纯加载就要十几小时。这里把模型加载抽出来只做一次,
# 然后顺序跑分配给本 worker 的那一片 case 列表。
#
# 复用 eval_libero_single.py 里的 run_single_task / 模型构建辅助函数,行为与
# 单任务版完全一致,只是外层套了个循环 + 断点续跑 + 每 case 独立 results.json。
#
# 用法(由 scripts/eval_libero_plus.sh 调度,一般不手动跑):
#   CUDA_VISIBLE_DEVICES=$gpu python experiments/libero/eval_libero_multi.py \
#       ckpt=... task=libero_uncond_2cam224_1e-4 model.vae_memory.enabled=true \
#       EVALUATION.num_trials=1 EVALUATION.save_video=false \
#       EVALUATION.dataset_stats_path=... EVALUATION.output_dir=... \
#       gpu_id=$worker_idx EVALUATION.task_list_file=$shard_json
#
# task_list_file:一个 JSON 数组,每项 {"suite","task_id","name","category","difficulty_level"}
#   task_id 是 benchmark 的 0-based 下标(= task_classification.json 里 1-based id - 1)。
#   worker 会断言 benchmark 返回的 task 名字 == "name",以此自检 id 映射是否对。
# ----------------------------------------------------------------------------
import json
import logging
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 复用单任务版的辅助函数与逐 trial 循环(确保行为一致)。导入它也会注册好
# OmegaConf 的 eval/max/split resolver(在 eval_libero_single 模块级执行)。
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    run_single_task,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_multi_process(cfg: DictConfig):
    t0 = time.time()
    PartialState().config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    task_list_file = cfg.EVALUATION.get("task_list_file")
    if not task_list_file:
        raise ValueError("EVALUATION.task_list_file is required for eval_libero_multi.py")
    cases = json.loads(Path(task_list_file).read_text(encoding="utf-8"))
    gpu_id = int(cfg.gpu_id)
    n_trials = int(cfg.EVALUATION.num_trials)
    logging.info("Worker gpu_id=%s: %d cases from %s (num_trials=%d)",
                 gpu_id, len(cases), task_list_file, n_trials)

    # ---- 模型 + processor 只构建一次 ----
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
    action_horizon = (int(action_horizon_cfg) if action_horizon_cfg is not None
                      else int(cfg.data.train.num_frames) - 1)
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    out_root = Path(cfg.EVALUATION.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # cfg 要逐 case 改 task_suite_name/task_id(run_single_task 内部从 cfg 读),开 struct。
    OmegaConf.set_struct(cfg, False)

    suite_cache: dict = {}
    n_done = n_skip = n_fail = 0
    for i, c in enumerate(cases):
        suite = c["suite"]
        tid = int(c["task_id"])
        out_dir = out_root / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"gpu{gpu_id}_task{tid}_results.json"

        if out_file.exists():  # 断点续跑:已完成则跳过
            n_skip += 1
            continue

        try:
            if suite not in suite_cache:
                suite_cache[suite] = benchmark.get_benchmark_dict()[suite]()
            task_suite = suite_cache[suite]
            task = task_suite.get_task(tid)

            # 自检:确认 task_id(0-based) 与 task_classification.json 的 name 对齐,
            # 防止 1-based/0-based 映射错位导致按 factor 聚合时张冠李戴。
            expected = c.get("name")
            got = getattr(task, "name", None)
            if expected and got and expected != got:
                raise RuntimeError(
                    f"task-name mismatch suite={suite} task_id={tid}: "
                    f"expected '{expected}' but benchmark gave '{got}' "
                    f"(id<->index mapping wrong?)")

            initial_states = list(task_suite.get_task_init_states(tid))
            if len(initial_states) == 0:
                raise RuntimeError(f"no init states for suite={suite} task_id={tid}")
            while len(initial_states) < n_trials:
                initial_states.extend(initial_states[: n_trials - len(initial_states)])

            cfg.EVALUATION.task_suite_name = suite
            cfg.EVALUATION.task_id = tid

            video_dir = out_dir / "videos"
            predicted_video_dir = out_dir / "predicted_videos"
            if bool(cfg.EVALUATION.get("save_video", True)):
                video_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "task_suite": suite,
                "task_id": tid,
                "task_description": None,
                "successes": 0,
                "total_episodes": n_trials,
                "gpu_id": gpu_id,
                "category": c.get("category"),
                "difficulty_level": c.get("difficulty_level"),
                "success_episodes": [],
                "failure_episodes": [],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": 0,
            }
            ct0 = time.time()
            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            results.update(task_results)
            results["duration"] = time.time() - ct0
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            n_done += 1
        except Exception as e:  # 单个 case 失败不拖垮整个 worker
            n_fail += 1
            logging.exception("case FAILED suite=%s task_id=%s: %s", suite, tid, e)
            try:
                (out_dir / f"gpu{gpu_id}_task{tid}.FAILED").write_text(str(e), encoding="utf-8")
            except Exception:
                pass

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (n_done + n_fail) / el if el > 0 else 0
            logging.info("gpu%s %d/%d done=%d skip=%d fail=%d %.1fmin %.2f case/s",
                         gpu_id, i + 1, len(cases), n_done, n_skip, n_fail, el / 60, rate)

    logging.info("gpu%s FINISHED done=%d skip=%d fail=%d total=%d in %.1fmin",
                 gpu_id, n_done, n_skip, n_fail, len(cases), (time.time() - t0) / 60)


if __name__ == "__main__":
    eval_multi_process()
