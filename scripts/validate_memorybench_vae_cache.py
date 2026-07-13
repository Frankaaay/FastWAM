#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a MemoryBench VAE latent cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", default="memorybench_short_v4_1e-5")
    parser.add_argument("--minimum-files", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--sample-indices", default=None)
    return parser.parse_args()


def build_dataset(task: str):
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    train_cfg.processor = None
    train_cfg.vae_latent_cache_dir = None
    train_cfg.vae_latent_cache_keep_video = True
    return cfg, instantiate(train_cfg)


def parse_sample_indices(raw: str | None, dataset_size: int) -> list[int]:
    if raw is None:
        return [0, dataset_size // 2, dataset_size - 1]
    indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not indices:
        raise ValueError("--sample-indices must contain at least one index.")
    for index in indices:
        if index < 0 or index >= dataset_size:
            raise ValueError(f"Sample index {index} is outside [0, {dataset_size}).")
    return indices


def validate_payload(path: Path, *, dataset, sample_idx: int, model_id: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "fastwam_robot_video_vae_latents_v1":
        raise ValueError(f"Invalid schema in {path}: {payload.get('schema')}")
    if int(payload.get("sample_idx", -1)) != sample_idx:
        raise ValueError(f"sample_idx mismatch in {path}: {payload.get('sample_idx')} != {sample_idx}")
    if payload.get("fingerprint") != dataset.vae_latent_cache_fingerprint:
        raise ValueError(f"Fingerprint mismatch in {path}: {payload.get('fingerprint')}")
    if payload.get("metadata") != dataset.vae_latent_cache_metadata:
        raise ValueError(f"Metadata mismatch in {path}")
    if payload.get("model_id") != model_id:
        raise ValueError(f"model_id mismatch in {path}: {payload.get('model_id')} != {model_id}")

    summary = {}
    for key in ("input_latents", "history_video_latents"):
        tensor = payload.get(key)
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            raise ValueError(f"{key} must be a 4D tensor in {path}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{key} must be bfloat16 in {path}, got {tensor.dtype}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{key} contains NaN or Inf in {path}")
        summary[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return summary


def main() -> None:
    args = parse_args()
    cfg, dataset = build_dataset(args.task)
    dataset_size = len(dataset)
    cache_dir = args.cache_root.expanduser() / dataset.vae_latent_cache_fingerprint
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache fingerprint directory not found: {cache_dir}")

    cache_files = list(cache_dir.rglob("*.pt"))
    temp_files = list(cache_dir.rglob(".*.tmp.*"))
    if temp_files:
        raise RuntimeError(f"Found {len(temp_files)} incomplete temporary cache files under {cache_dir}")
    if args.minimum_files is not None and len(cache_files) < args.minimum_files:
        raise RuntimeError(f"Expected at least {args.minimum_files} cache files, found {len(cache_files)}")
    if args.require_complete and len(cache_files) != dataset_size:
        raise RuntimeError(f"Expected exactly {dataset_size} cache files, found {len(cache_files)}")

    model_id = str(cfg.data.train.get("vae_latent_cache_model_id"))
    samples = {}
    for sample_idx in parse_sample_indices(args.sample_indices, dataset_size):
        path = dataset.vae_latent_cache_path(sample_idx, args.cache_root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache sample {sample_idx}: {path}")
        samples[str(sample_idx)] = validate_payload(
            path,
            dataset=dataset,
            sample_idx=sample_idx,
            model_id=model_id,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_size": dataset_size,
                "fingerprint": dataset.vae_latent_cache_fingerprint,
                "cache_dir": str(cache_dir),
                "cache_files": len(cache_files),
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
