import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.fastwam_online_history import FastWAMOnlineHistoryBuffer  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402

register_default_resolvers()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _require_path(path: str | os.PathLike[str], *, what: str) -> Path:
    resolved = _expand_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{what} not found: {resolved}")
    return resolved


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt: Path) -> Path:
    explicit = cfg.memorybench_eval.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit:
        candidates.append(_expand_path(explicit))
    for parent in ckpt.parents[:6]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Pass "
        "++memorybench_eval.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _configure_eval_dataset(cfg: DictConfig, stats_path: Path) -> Path:
    data_root = _expand_path(
        cfg.memorybench_eval.get(
            "data_root",
            os.environ.get("MEMORYBENCH_DATA_ROOT", "/data/shared/offline/datasets/memorybench"),
        )
    )
    dataset_dir = _expand_path(
        cfg.memorybench_eval.get(
            "dataset_dir",
            data_root / "lerobot" / "memorybench_short_test_v2",
        )
    )
    _require_path(dataset_dir, what="MemoryBench test dataset")

    cfg.data.train.dataset_dirs = [str(dataset_dir)]
    cfg.data.train.is_training_set = False
    cfg.data.train.val_set_proportion = 0.0
    cfg.data.train.pretrained_norm_stats = str(stats_path)

    # Eval needs raw current/history images for VAE encoding inside infer_action.
    cfg.data.train.vae_latent_cache_dir = None
    cfg.data.train.vae_latent_cache_keep_video = True
    return dataset_dir


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return None if value <= 0 else value


def _sample_indices(length: int, *, stride: int, max_samples: int | None) -> list[int]:
    if stride <= 0:
        raise ValueError(f"sample_stride must be positive, got {stride}")
    indices = list(range(0, int(length), int(stride)))
    if max_samples is not None:
        indices = indices[:max_samples]
    if not indices:
        raise ValueError("No eval samples selected.")
    return indices


def _denormalize_action(action: torch.Tensor, dataset) -> torch.Tensor:
    processor = dataset.lerobot_dataset.processor
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("MemoryBench eval currently expects one merged action key.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return normalizer.backward(action.to(dtype=torch.float32)).squeeze(0)


def _init_accumulators(action_dim: int) -> dict[str, Any]:
    return {
        "count": 0,
        "samples": 0,
        "norm_sse": 0.0,
        "norm_sae": 0.0,
        "raw_sse": 0.0,
        "raw_sae": 0.0,
        "per_dim_norm_sse": np.zeros(action_dim, dtype=np.float64),
        "per_dim_norm_sae": np.zeros(action_dim, dtype=np.float64),
        "per_dim_raw_sse": np.zeros(action_dim, dtype=np.float64),
        "per_dim_raw_sae": np.zeros(action_dim, dtype=np.float64),
        "per_dim_count": np.zeros(action_dim, dtype=np.int64),
    }


def _update_metrics(
    acc: dict[str, Any],
    *,
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    valid = valid_mask.bool().unsqueeze(-1).expand_as(target_norm)
    if not bool(valid.any().item()):
        return

    norm_err = (pred_norm - target_norm).to(dtype=torch.float32)
    raw_err = (pred_raw - target_raw).to(dtype=torch.float32)
    norm_err = norm_err[valid]
    raw_err = raw_err[valid]

    acc["count"] += int(norm_err.numel())
    acc["norm_sse"] += float((norm_err * norm_err).sum().item())
    acc["norm_sae"] += float(norm_err.abs().sum().item())
    acc["raw_sse"] += float((raw_err * raw_err).sum().item())
    acc["raw_sae"] += float(raw_err.abs().sum().item())

    valid_np = valid.cpu().numpy()
    norm_np = (pred_norm - target_norm).detach().cpu().numpy()
    raw_np = (pred_raw - target_raw).detach().cpu().numpy()
    for dim in range(target_norm.shape[-1]):
        dim_valid = valid_np[:, dim]
        if not dim_valid.any():
            continue
        norm_dim = norm_np[:, dim][dim_valid]
        raw_dim = raw_np[:, dim][dim_valid]
        acc["per_dim_count"][dim] += int(dim_valid.sum())
        acc["per_dim_norm_sse"][dim] += float(np.square(norm_dim).sum())
        acc["per_dim_norm_sae"][dim] += float(np.abs(norm_dim).sum())
        acc["per_dim_raw_sse"][dim] += float(np.square(raw_dim).sum())
        acc["per_dim_raw_sae"][dim] += float(np.abs(raw_dim).sum())


def _finalize_metrics(acc: dict[str, Any]) -> dict[str, Any]:
    count = max(int(acc["count"]), 1)
    per_dim_count = np.maximum(acc["per_dim_count"], 1)
    return {
        "samples": int(acc["samples"]),
        "action_values": int(acc["count"]),
        "norm_mse": acc["norm_sse"] / count,
        "norm_mae": acc["norm_sae"] / count,
        "raw_mse": acc["raw_sse"] / count,
        "raw_mae": acc["raw_sae"] / count,
        "per_dim_norm_mse": acc["per_dim_norm_sse"] / per_dim_count,
        "per_dim_norm_mae": acc["per_dim_norm_sae"] / per_dim_count,
        "per_dim_raw_mse": acc["per_dim_raw_sse"] / per_dim_count,
        "per_dim_raw_mae": acc["per_dim_raw_sae"] / per_dim_count,
        "per_dim_count": acc["per_dim_count"],
        "gripper_norm_mse": float(acc["per_dim_norm_sse"][-1] / max(acc["per_dim_count"][-1], 1)),
        "gripper_norm_mae": float(acc["per_dim_norm_sae"][-1] / max(acc["per_dim_count"][-1], 1)),
        "gripper_raw_mse": float(acc["per_dim_raw_sse"][-1] / max(acc["per_dim_count"][-1], 1)),
        "gripper_raw_mae": float(acc["per_dim_raw_sae"][-1] / max(acc["per_dim_count"][-1], 1)),
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> dict[str, Any]:
    OmegaConf.set_struct(cfg, False)
    if "memorybench_eval" not in cfg:
        cfg.memorybench_eval = {}

    start = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    ckpt_value = cfg.memorybench_eval.get("ckpt")
    if not ckpt_value:
        raise ValueError("Pass ++memorybench_eval.ckpt=/path/to/step_xxxxxx.pt")
    ckpt = _require_path(ckpt_value, what="checkpoint")
    stats_path = _resolve_dataset_stats_path(cfg, ckpt)
    dataset_dir = _configure_eval_dataset(cfg, stats_path)

    device = str(cfg.memorybench_eval.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(ckpt))
    model = model.to(device).eval()

    dataset = instantiate(cfg.data.train)
    action_dim = int(model.action_expert.action_dim)
    action_horizon_cfg = cfg.memorybench_eval.get("action_horizon")
    sample_stride = int(cfg.memorybench_eval.get("sample_stride", 32))
    max_samples = _as_optional_int(cfg.memorybench_eval.get("max_samples"))
    indices = _sample_indices(len(dataset), stride=sample_stride, max_samples=max_samples)
    num_inference_steps = int(cfg.memorybench_eval.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    text_cfg_scale = float(cfg.memorybench_eval.get("text_cfg_scale", 1.0))
    sigma_shift = cfg.memorybench_eval.get("sigma_shift")
    sigma_shift = None if sigma_shift is None else float(sigma_shift)
    tiled = bool(cfg.memorybench_eval.get("tiled", False))
    rand_device = str(cfg.memorybench_eval.get("rand_device", "cpu"))
    base_seed = None if cfg.get("seed") is None else int(cfg.seed)

    output_dir = _expand_path(
        cfg.memorybench_eval.get(
            "output_dir",
            Path("evaluate_results") / "memorybench_open_loop" / time.strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    use_history = FastWAMOnlineHistoryBuffer.enabled_for_model(model)
    acc = _init_accumulators(action_dim)
    failures: list[dict[str, Any]] = []

    logging.info("MemoryBench open-loop eval")
    logging.info("  ckpt=%s", ckpt)
    logging.info("  dataset=%s", dataset_dir)
    logging.info("  stats=%s", stats_path)
    logging.info("  samples=%d stride=%d use_history=%s", len(indices), sample_stride, use_history)

    pbar = tqdm(indices, desc="MemoryBench eval")
    for sample_order, idx in enumerate(pbar):
        try:
            sample = dataset[idx]
            target_norm = sample["action"].to(dtype=torch.float32)
            action_horizon = int(action_horizon_cfg or target_norm.shape[0])
            action_horizon = min(action_horizon, int(target_norm.shape[0]))

            input_image = sample["video"][:, 0].unsqueeze(0)
            proprio = sample["proprio"][0]
            context = sample["context"]
            context_mask = sample["context_mask"]
            infer_kwargs = {
                "prompt": None,
                "context": context,
                "context_mask": context_mask,
                "input_image": input_image,
                "action_horizon": action_horizon,
                "proprio": proprio,
                "negative_prompt": "",
                "text_cfg_scale": text_cfg_scale,
                "num_inference_steps": num_inference_steps,
                "sigma_shift": sigma_shift,
                "seed": None if base_seed is None else base_seed + int(idx),
                "rand_device": rand_device,
                "tiled": tiled,
            }
            if use_history:
                infer_kwargs.update(
                    {
                        "history_video": sample["history_video"],
                        "history_action": sample["history_action"],
                        "history_video_is_pad": sample["history_video_is_pad"],
                        "history_action_is_pad": sample["history_action_is_pad"],
                    }
                )

            with torch.no_grad():
                pred = model.infer_action(**infer_kwargs)

            pred_norm = pred["action"][:action_horizon].to(dtype=torch.float32)
            target_norm = target_norm[:action_horizon]
            valid_mask = ~sample["action_is_pad"][:action_horizon].bool()
            pred_raw = _denormalize_action(pred_norm, dataset)
            target_raw = _denormalize_action(target_norm, dataset)
            _update_metrics(
                acc,
                pred_norm=pred_norm,
                target_norm=target_norm,
                pred_raw=pred_raw,
                target_raw=target_raw,
                valid_mask=valid_mask,
            )
            acc["samples"] += 1
            if acc["samples"] % 10 == 0:
                cur = _finalize_metrics(acc)
                pbar.set_postfix(norm_mse=f"{cur['norm_mse']:.4g}", raw_mse=f"{cur['raw_mse']:.4g}")
        except Exception as exc:
            failures.append({"idx": int(idx), "error": f"{type(exc).__name__}: {exc}"})
            logging.exception("Failed sample idx=%s", idx)

    metrics = _finalize_metrics(acc)
    results = {
        "mode": "memorybench_open_loop_teacher_forced_history",
        "checkpoint": str(ckpt),
        "dataset_dir": str(dataset_dir),
        "dataset_stats_path": str(stats_path),
        "output_dir": str(output_dir),
        "sample_stride": sample_stride,
        "max_samples": max_samples,
        "selected_samples": len(indices),
        "use_history": use_history,
        "num_inference_steps": num_inference_steps,
        "action_horizon": action_horizon_cfg,
        "device": device,
        "mixed_precision": str(cfg.get("mixed_precision", "bf16")),
        "duration_sec": time.time() - start,
        "metrics": metrics,
        "num_failures": len(failures),
        "failures": failures[:50],
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logging.info("Saved results: %s", results_path)
    logging.info(
        "Done: samples=%d norm_mse=%.6g raw_mse=%.6g failures=%d",
        metrics["samples"],
        metrics["norm_mse"],
        metrics["raw_mse"],
        len(failures),
    )
    return results


if __name__ == "__main__":
    main()
