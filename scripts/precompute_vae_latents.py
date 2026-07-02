import logging
import os
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils import misc

register_default_resolvers()
logger = get_logger(__name__)


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _atomic_torch_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _get_cache_cfg(cfg: DictConfig):
    cache_cfg = cfg.get("vae_latent_cache")
    if cache_cfg is None:
        cache_cfg = {}
    return cache_cfg


def _resolve_cache_output_dir(cfg: DictConfig, cache_cfg) -> Path:
    output_dir = cache_cfg.get("output_dir") if hasattr(cache_cfg, "get") else None
    if output_dir is None and cfg.data is not None and cfg.data.get("train") is not None:
        output_dir = cfg.data.train.get("vae_latent_cache_dir")
    if output_dir is None or not str(output_dir).strip():
        raise ValueError(
            "VAE latent cache output dir is required. Set "
            "`+vae_latent_cache.output_dir=/path/to/cache` or "
            "`+data.train.vae_latent_cache_dir=/path/to/cache`."
        )
    return Path(str(output_dir)).expanduser()


def _instantiate_train_dataset_without_cache(cfg: DictConfig):
    if cfg.data is None or cfg.data.get("train") is None:
        raise ValueError("`cfg.data.train` is required.")
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    train_cfg["vae_latent_cache_dir"] = None
    train_cfg["vae_latent_cache_keep_video"] = True
    return instantiate(train_cfg)


def _load_vae(cfg: DictConfig, *, device: str, torch_dtype: torch.dtype):
    if cfg.model is None:
        raise ValueError("`cfg.model` is required.")
    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))

    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    vae.requires_grad_(False)
    return vae, model_id, str(vae_config.path)


def _filter_indices(dataset, indices: list[int], *, cache_dir: Path, overwrite: bool):
    if overwrite:
        return indices, 0
    remaining = []
    skipped = 0
    for idx in indices:
        if dataset.vae_latent_cache_path(idx, cache_dir).exists():
            skipped += 1
        else:
            remaining.append(idx)
    return remaining, skipped


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    cache_cfg = _get_cache_cfg(cfg)
    cache_dir = _resolve_cache_output_dir(cfg, cache_cfg)
    work_dir = Path(str(cache_cfg.get("work_dir", cache_dir / "_precompute_run"))).expanduser()
    misc.register_work_dir(work_dir)
    if rank == 0:
        OmegaConf.save(config=cfg, f=str(work_dir / "config.yaml"))
    if is_distributed:
        dist.barrier()
    overwrite = _to_bool(cache_cfg.get("overwrite", False))
    batch_size = int(cache_cfg.get("batch_size", cfg.get("batch_size", 1)))
    num_workers = int(cache_cfg.get("num_workers", cfg.get("num_workers", 0)))
    max_samples = cache_cfg.get("max_samples")
    max_samples = None if max_samples in (None, "null") else int(max_samples)
    tiled = _to_bool(cache_cfg.get("tiled", False))
    tile_size = tuple(cache_cfg.get("tile_size", (30, 52)))
    tile_stride = tuple(cache_cfg.get("tile_stride", (15, 26)))

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    torch_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    if rank == 0:
        logger.info(
            "Precomputing VAE latents: cache_dir=%s overwrite=%s batch_size=%d num_workers=%d "
            "max_samples=%s device=%s dtype=%s tiled=%s",
            cache_dir,
            overwrite,
            batch_size,
            num_workers,
            max_samples,
            device,
            torch_dtype,
            tiled,
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not is_distributed:
            logger.info(
                "Multi-GPU available. To shard work, run: "
                "torchrun --standalone --nproc_per_node=%d scripts/precompute_vae_latents.py ...",
                torch.cuda.device_count(),
            )

    dataset = _instantiate_train_dataset_without_cache(cfg)
    all_indices = list(range(len(dataset)))
    if max_samples is not None:
        all_indices = all_indices[:max(max_samples, 0)]
    all_indices, skipped_existing = _filter_indices(
        dataset,
        all_indices,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )
    local_indices = all_indices[rank::world_size]
    if rank == 0:
        logger.info(
            "Dataset size=%d fingerprint=%s to_encode=%d skipped_existing=%d world_size=%d",
            len(dataset),
            dataset.vae_latent_cache_fingerprint,
            len(all_indices),
            skipped_existing,
            world_size,
        )

    vae, model_id, vae_path = _load_vae(cfg, device=device, torch_dtype=torch_dtype)
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    stats = {"new": 0, "overwrite": 0, "skip": skipped_existing if rank == 0 else 0}
    with tqdm(
        total=len(local_indices),
        desc=f"VAE latents rank {rank}/{world_size}",
        unit="sample",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for batch in loader:
                video = batch["video"].to(device=device, dtype=torch_dtype, non_blocking=True)
                history_video = batch["history_video"].to(
                    device=device,
                    dtype=torch_dtype,
                    non_blocking=True,
                )
                input_latents = vae.encode(
                    video,
                    device=device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                history_video_latents = vae.encode(
                    history_video,
                    device=device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )

                sample_indices = batch["sample_idx"].detach().cpu().tolist()
                for i, sample_idx in enumerate(sample_indices):
                    cache_path = dataset.vae_latent_cache_path(sample_idx, cache_dir)
                    if cache_path.exists() and not overwrite:
                        stats["skip"] += 1
                        continue
                    if cache_path.exists():
                        stats["overwrite"] += 1
                    else:
                        stats["new"] += 1
                    payload = dataset.make_vae_latent_cache_payload(
                        sample_idx=int(sample_idx),
                        input_latents=input_latents[i],
                        history_video_latents=history_video_latents[i],
                        model_id=model_id,
                        vae_path=vae_path,
                    )
                    _atomic_torch_save(payload, cache_path)

                pbar.update(len(sample_indices))

    if is_distributed:
        reduce_device = torch.device(device) if str(device).startswith("cuda") else torch.device("cpu")
        stats_tensor = torch.tensor(
            [stats["new"], stats["overwrite"], stats["skip"]],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            stats["new"] = int(stats_tensor[0].item())
            stats["overwrite"] = int(stats_tensor[1].item())
            stats["skip"] = int(stats_tensor[2].item())

    if rank == 0:
        logger.info(
            "Finished VAE latent precompute: cache_dir=%s fingerprint=%s new=%d overwrite=%d skip=%d",
            cache_dir,
            dataset.vae_latent_cache_fingerprint,
            stats["new"],
            stats["overwrite"],
            stats["skip"],
        )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
