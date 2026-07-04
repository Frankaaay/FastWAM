#!/usr/bin/env python3
import logging
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from scripts.precompute_vae_latents import (
    _configure_vae_encoder_compile,
    _encode_video_latents,
    _get_cache_cfg,
    _instantiate_train_dataset_without_cache,
    _load_vae,
    _to_bool,
)

register_default_resolvers()
logger = get_logger(__name__)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    ref = reference.detach().float()
    got = candidate.detach().float()
    diff = (ref - got).abs()
    denom = max(float(ref.abs().max().item()), torch.finfo(torch.float32).eps)
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(diff.max().item()),
        "rel": float(diff.max().item()) / denom,
        "mean_abs": float(diff.mean().item()),
        "p999": float(torch.quantile(diff.flatten(), 0.999).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()),
    }


def _print_metric_row(label: str, shape: tuple[int, ...], metrics: dict[str, Any]) -> None:
    print(
        f"{label}\tshape={shape}\texact={metrics['exact']}\t"
        f"max_abs={metrics['max_abs']:.6g}\trel={metrics['rel']:.6g}\t"
        f"mean_abs={metrics['mean_abs']:.6g}\tp999={metrics['p999']:.6g}\t"
        f"cosine={metrics['cosine']:.8f}",
        flush=True,
    )


def _assert_within(label: str, metrics: dict[str, Any], *, abs_tol: float, rel_tol: float) -> None:
    if metrics["max_abs"] > abs_tol or metrics["rel"] > rel_tol:
        raise AssertionError(
            f"{label} exceeded tolerance: max_abs={metrics['max_abs']:.6g} "
            f"(tol={abs_tol}), rel={metrics['rel']:.6g} (tol={rel_tol})"
        )


def _infer_video_frame_counts(cfg: DictConfig) -> tuple[int, int]:
    train_cfg = cfg.data.train
    num_frames = int(train_cfg.get("num_frames", 33))
    ratio = int(train_cfg.get("action_video_freq_ratio", 4))
    history_past_steps = int(train_cfg.get("history_video_past_steps", 16))
    current_index = history_past_steps
    history_frames = len(range(0, current_index + 1, ratio))
    current_frames = len(range(current_index, current_index + num_frames, ratio))
    return current_frames, history_frames


def _random_inputs(cfg: DictConfig, *, batch_size: int, dtype: torch.dtype, device: str) -> list[tuple[str, torch.Tensor]]:
    current_frames, history_frames = _infer_video_frame_counts(cfg)
    height, width = [int(v) for v in cfg.data.train.get("video_size", (384, 320))]
    seed = int(_get_cache_cfg(cfg).get("verify_seed", 0))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = []
    for label, frames in (("current", current_frames), ("history", history_frames)):
        video = torch.randn(
            batch_size,
            3,
            frames,
            height,
            width,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        inputs.append((label, video))
    return inputs


def _dataset_inputs(cfg: DictConfig, *, batch_size: int, dtype: torch.dtype, device: str) -> list[tuple[str, torch.Tensor]]:
    dataset = _instantiate_train_dataset_without_cache(cfg)
    loader = DataLoader(
        Subset(dataset, list(range(batch_size))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(_get_cache_cfg(cfg).get("verify_num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))
    return [
        ("current", batch["video"].to(device=device, dtype=dtype, non_blocking=True)),
        ("history", batch["history_video"].to(device=device, dtype=dtype, non_blocking=True)),
    ]


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    cache_cfg = _get_cache_cfg(cfg)
    work_dir = Path(str(cache_cfg.get("work_dir", "./runs/vae_cache_encode_verify"))).expanduser()
    misc.register_work_dir(work_dir)

    device = str(cache_cfg.get("verify_device", "cuda" if torch.cuda.is_available() else "cpu"))
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    dtype = _mixed_precision_to_model_dtype(mixed_precision)
    batch_size = int(cache_cfg.get("verify_batch_size", 4))
    source = str(cache_cfg.get("verify_source", "random"))
    abs_tol = float(cache_cfg.get("verify_abs_tol", 1e-2))
    rel_tol = float(cache_cfg.get("verify_rel_tol", 1e-3))
    tiled = _to_bool(cache_cfg.get("tiled", False))

    if source == "random":
        inputs = _random_inputs(cfg, batch_size=batch_size, dtype=dtype, device=device)
    elif source == "dataset":
        inputs = _dataset_inputs(cfg, batch_size=batch_size, dtype=dtype, device=device)
    else:
        raise ValueError("vae_latent_cache.verify_source must be 'random' or 'dataset'.")

    vae, _model_id, _vae_path = _load_vae(cfg, device=device, torch_dtype=dtype)
    logger.info(
        "Verifying VAE cache encode equivalence: source=%s batch_size=%d dtype=%s device=%s",
        source,
        batch_size,
        dtype,
        device,
    )

    with torch.inference_mode():
        references: dict[str, torch.Tensor] = {}
        for label, video in inputs:
            reference = _encode_video_latents(
                vae,
                video,
                device=device,
                backend="wrapper",
                tiled=tiled,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            _sync(device)
            references[label] = reference
            batch_native = _encode_video_latents(
                vae,
                video,
                device=device,
                backend="model_batch",
                tiled=tiled,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            _sync(device)
            metrics = _metrics(reference, batch_native)
            _print_metric_row(f"{label}:wrapper_vs_model_batch", tuple(batch_native.shape), metrics)
            _assert_within(f"{label}:wrapper_vs_model_batch", metrics, abs_tol=abs_tol, rel_tol=rel_tol)

        if _to_bool(cache_cfg.get("compile_encoder", False)):
            _configure_vae_encoder_compile(vae, cache_cfg)
            for label, video in inputs:
                compiled_wrapper = _encode_video_latents(
                    vae,
                    video,
                    device=device,
                    backend="wrapper",
                    tiled=tiled,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
                _sync(device)
                metrics = _metrics(references[label], compiled_wrapper)
                _print_metric_row(f"{label}:wrapper_vs_compiled_wrapper", tuple(compiled_wrapper.shape), metrics)
                _assert_within(
                    f"{label}:wrapper_vs_compiled_wrapper",
                    metrics,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )

                compiled = _encode_video_latents(
                    vae,
                    video,
                    device=device,
                    backend="model_batch",
                    tiled=tiled,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
                _sync(device)
                metrics = _metrics(references[label], compiled)
                _print_metric_row(f"{label}:wrapper_vs_compiled_model_batch", tuple(compiled.shape), metrics)
                _assert_within(
                    f"{label}:wrapper_vs_compiled_model_batch",
                    metrics,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )

    print("VAE cache encode equivalence passed.", flush=True)


if __name__ == "__main__":
    main()
