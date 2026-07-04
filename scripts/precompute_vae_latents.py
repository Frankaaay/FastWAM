import logging
import inspect
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    train_cfg["vae_latent_cache_precompute_only"] = True
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


def _torch_compile_available() -> bool:
    return hasattr(torch, "compile")


def _to_tuple2(value: Any, *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"`{name}` must contain exactly 2 integers, got {value!r}")
    return int(value[0]), int(value[1])


def _configure_precompute_runtime(cache_cfg) -> None:
    cudnn_benchmark = _to_bool(cache_cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    allow_tf32 = cache_cfg.get("allow_tf32", None)
    if allow_tf32 is not None:
        allow_tf32 = _to_bool(allow_tf32)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32


def _configure_vae_encoder_compile(vae, cache_cfg) -> bool:
    compile_encoder = _to_bool(cache_cfg.get("compile_encoder", False))
    if not compile_encoder:
        return False
    if not _torch_compile_available():
        raise RuntimeError("vae_latent_cache.compile_encoder=true but torch.compile is unavailable.")

    vae_model = getattr(vae, "model", None)
    encoder = getattr(vae_model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Cannot compile VAE encoder: `vae.model.encoder` was not found.")

    mode = str(cache_cfg.get("compile_mode", "default"))
    encoder.forward = torch.compile(
        encoder.forward,
        mode=mode,
        dynamic=False,
        fullgraph=False,
    )
    return True


def _encode_video_latents(
    vae,
    videos: torch.Tensor,
    *,
    device: str,
    backend: str,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> torch.Tensor:
    if backend == "wrapper":
        return vae.encode(
            videos,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    if backend == "model_batch":
        if tiled:
            raise ValueError("vae_latent_cache.encode_backend=model_batch does not support tiled=true.")
        return vae.model.encode(videos.to(device), vae.scale)
    raise ValueError(
        f"Unsupported vae_latent_cache.encode_backend={backend!r}; "
        "expected one of: wrapper, model_batch."
    )


def _sync_if_needed(device: str, enabled: bool) -> None:
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _filter_indices(dataset, indices: list[int], *, cache_dir: Path, overwrite: bool):
    if overwrite:
        return indices, 0
    fingerprint_dir = cache_dir / dataset.vae_latent_cache_fingerprint
    if not fingerprint_dir.exists():
        return indices, 0
    remaining = []
    skipped = 0
    for idx in indices:
        if dataset.vae_latent_cache_path(idx, cache_dir).exists():
            skipped += 1
        else:
            remaining.append(idx)
    return remaining, skipped


def _build_dataloader_kwargs(*, dataset, batch_size: int, num_workers: int, cache_cfg, pin_memory: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        prefetch_factor = cache_cfg.get("prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = _to_bool(cache_cfg.get("persistent_workers", False))
    elif _to_bool(cache_cfg.get("persistent_workers", False)):
        raise ValueError("vae_latent_cache.persistent_workers=true requires num_workers > 0.")

    in_order = cache_cfg.get("in_order", None)
    if in_order is not None:
        if "in_order" not in inspect.signature(DataLoader).parameters:
            raise ValueError("vae_latent_cache.in_order is not supported by this PyTorch DataLoader.")
        kwargs["in_order"] = _to_bool(in_order)
    return kwargs


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _preload_batches_to_ram(loader, *, dtype: torch.dtype, pin_memory: bool, device: str) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    batches = []
    stats = {
        "seconds": 0.0,
        "samples": 0,
        "batches": 0,
        "bytes": 0,
    }
    start = time.perf_counter()
    for batch in loader:
        video = batch["video"].to(dtype=dtype)
        history_video = batch["history_video"].to(dtype=dtype)
        if pin_memory and str(device).startswith("cuda") and torch.cuda.is_available():
            video = video.pin_memory()
            history_video = history_video.pin_memory()
        sample_idx = batch["sample_idx"].detach().cpu()
        batches.append(
            {
                "sample_idx": sample_idx,
                "video": video,
                "history_video": history_video,
            }
        )
        stats["samples"] += int(sample_idx.numel())
        stats["batches"] += 1
        stats["bytes"] += _tensor_bytes(video) + _tensor_bytes(history_video) + _tensor_bytes(sample_idx)
    stats["seconds"] = time.perf_counter() - start
    return batches, stats


def _warmup_vae_encoder(
    vae,
    dataset,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: str,
    backend: str,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    num_batches: int,
    sync_timing: bool,
) -> float:
    if num_batches <= 0:
        return 0.0
    height, width = [int(v) for v in dataset.video_size]
    current_frames = len(dataset.video_sample_indices)
    history_frames = len(dataset.history_video_sample_indices)
    warmup_current = torch.zeros(
        batch_size,
        3,
        current_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    warmup_history = torch.zeros(
        batch_size,
        3,
        history_frames,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    _sync_if_needed(device, sync_timing)
    start = time.perf_counter()
    for _ in range(num_batches):
        _encode_video_latents(
            vae,
            warmup_current,
            device=device,
            backend=backend,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        _encode_video_latents(
            vae,
            warmup_history,
            device=device,
            backend=backend,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    _sync_if_needed(device, True)
    return time.perf_counter() - start


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
    manual_num_shards = cache_cfg.get("num_shards", None)
    manual_shard_index = cache_cfg.get("shard_index", None)
    manual_sharding_enabled = manual_num_shards is not None or manual_shard_index is not None
    if manual_sharding_enabled and is_distributed:
        raise ValueError(
            "vae_latent_cache.num_shards/shard_index are for non-distributed launches. "
            "Do not combine them with torchrun."
        )
    if manual_sharding_enabled:
        if manual_num_shards is None or manual_shard_index is None:
            raise ValueError(
                "Set both vae_latent_cache.num_shards and vae_latent_cache.shard_index."
            )
        shard_world_size = int(manual_num_shards)
        shard_rank = int(manual_shard_index)
        if shard_world_size < 1:
            raise ValueError("vae_latent_cache.num_shards must be >= 1.")
        if shard_rank < 0 or shard_rank >= shard_world_size:
            raise ValueError(
                "vae_latent_cache.shard_index must satisfy 0 <= shard_index < num_shards."
            )
    else:
        shard_world_size = world_size
        shard_rank = rank
    tiled = _to_bool(cache_cfg.get("tiled", False))
    tile_size = _to_tuple2(cache_cfg.get("tile_size", (30, 52)), name="vae_latent_cache.tile_size")
    tile_stride = _to_tuple2(cache_cfg.get("tile_stride", (15, 26)), name="vae_latent_cache.tile_stride")
    encode_backend = str(cache_cfg.get("encode_backend", "wrapper"))
    use_inference_mode = _to_bool(cache_cfg.get("inference_mode", False))
    log_timing = _to_bool(cache_cfg.get("log_timing", False))
    sync_timing = _to_bool(cache_cfg.get("sync_timing", log_timing))
    pin_memory = _to_bool(cache_cfg.get("pin_memory", torch.cuda.is_available()))
    warmup_batches = int(cache_cfg.get("warmup_batches", 0))
    ram_preload = _to_bool(cache_cfg.get("ram_preload", False))
    ram_preload_limit_samples = int(cache_cfg.get("ram_preload_limit_samples", 8192))
    _configure_precompute_runtime(cache_cfg)

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    torch_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    if rank == 0:
        logger.info(
            "Precomputing VAE latents: cache_dir=%s overwrite=%s batch_size=%d num_workers=%d "
            "max_samples=%s device=%s dtype=%s tiled=%s encode_backend=%s inference_mode=%s "
            "pin_memory=%s prefetch_factor=%s persistent_workers=%s cudnn_benchmark=%s "
            "warmup_batches=%d ram_preload=%s ram_preload_limit_samples=%d shard=%d/%d distributed=%s",
            cache_dir,
            overwrite,
            batch_size,
            num_workers,
            max_samples,
            device,
            torch_dtype,
            tiled,
            encode_backend,
            use_inference_mode,
            pin_memory,
            cache_cfg.get("prefetch_factor", None),
            cache_cfg.get("persistent_workers", False),
            torch.backends.cudnn.benchmark,
            warmup_batches,
            ram_preload,
            ram_preload_limit_samples,
            shard_rank,
            shard_world_size,
            is_distributed,
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
    local_indices = all_indices[shard_rank::shard_world_size]
    local_indices, skipped_existing = _filter_indices(
        dataset,
        local_indices,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )
    global_to_encode = len(local_indices)
    global_skipped_existing = skipped_existing
    if is_distributed:
        reduce_device = torch.device(device) if str(device).startswith("cuda") else torch.device("cpu")
        index_stats = torch.tensor(
            [global_to_encode, global_skipped_existing],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(index_stats, op=dist.ReduceOp.SUM)
        global_to_encode = int(index_stats[0].item())
        global_skipped_existing = int(index_stats[1].item())
    if rank == 0:
        logger.info(
            "Dataset size=%d fingerprint=%s to_encode=%d skipped_existing=%d local_to_encode=%d shard=%d/%d",
            len(dataset),
            dataset.vae_latent_cache_fingerprint,
            global_to_encode,
            global_skipped_existing,
            len(local_indices),
            shard_rank,
            shard_world_size,
        )
    if ram_preload and len(local_indices) > ram_preload_limit_samples:
        raise ValueError(
            "vae_latent_cache.ram_preload=true preloads the whole local shard into RAM for benchmarking. "
            f"local_to_encode={len(local_indices)} exceeds ram_preload_limit_samples={ram_preload_limit_samples}. "
            "Use max_samples/num_shards for a bounded chunk or raise the limit intentionally."
        )

    vae, model_id, vae_path = _load_vae(cfg, device=device, torch_dtype=torch_dtype)
    compiled_encoder = _configure_vae_encoder_compile(vae, cache_cfg)
    if rank == 0:
        logger.info(
            "VAE precompute runtime: compile_encoder=%s compile_mode=%s allow_tf32=%s",
            compiled_encoder,
            cache_cfg.get("compile_mode", "default"),
            getattr(torch.backends.cudnn, "allow_tf32", None),
        )

    grad_context = torch.inference_mode if use_inference_mode else torch.no_grad
    with grad_context():
        warmup_seconds = _warmup_vae_encoder(
            vae,
            dataset,
            batch_size=batch_size,
            dtype=torch_dtype,
            device=device,
            backend=encode_backend,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            num_batches=warmup_batches,
            sync_timing=sync_timing,
        )
    if rank == 0 and warmup_batches > 0:
        logger.info("VAE precompute warmup: batches=%d seconds=%.3f", warmup_batches, warmup_seconds)

    loader = DataLoader(
        **_build_dataloader_kwargs(
            dataset=Subset(dataset, local_indices),
            batch_size=batch_size,
            num_workers=num_workers,
            cache_cfg=cache_cfg,
            pin_memory=pin_memory,
        )
    )
    ram_batches = None
    ram_preload_stats = {
        "seconds": 0.0,
        "samples": 0,
        "batches": 0,
        "bytes": 0,
    }
    if ram_preload:
        ram_batches, ram_preload_stats = _preload_batches_to_ram(
            loader,
            dtype=torch_dtype,
            pin_memory=pin_memory,
            device=device,
        )
        if rank == 0:
            gib = ram_preload_stats["bytes"] / (1024 ** 3)
            logger.info(
                "VAE RAM preload: samples=%d batches=%d bytes=%.2fGiB seconds=%.3f samples_per_s=%.3f",
                ram_preload_stats["samples"],
                ram_preload_stats["batches"],
                gib,
                ram_preload_stats["seconds"],
                ram_preload_stats["samples"] / ram_preload_stats["seconds"]
                if ram_preload_stats["seconds"] > 0
                else float("nan"),
            )

    stats = {"new": 0, "overwrite": 0, "skip": skipped_existing}
    timing = {
        "batches": 0,
        "samples": 0,
        "wall_seconds": 0.0,
        "ram_preload_seconds": float(ram_preload_stats["seconds"]),
        "ram_preload_bytes": float(ram_preload_stats["bytes"]),
        "data_wait_seconds": 0.0,
        "h2d_seconds": 0.0,
        "encode_seconds": 0.0,
        "save_seconds": 0.0,
    }
    wall_start = time.perf_counter()
    with tqdm(
        total=len(local_indices),
        desc=f"VAE latents shard {shard_rank}/{shard_world_size}",
        unit="sample",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with grad_context():
            loader_iter = iter(ram_batches if ram_batches is not None else loader)
            while True:
                data_wait_start = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                data_wait_end = time.perf_counter()
                _sync_if_needed(device, sync_timing)
                h2d_start = time.perf_counter()
                video = batch["video"].to(device=device, dtype=torch_dtype, non_blocking=True)
                history_video = batch["history_video"].to(
                    device=device,
                    dtype=torch_dtype,
                    non_blocking=True,
                )
                _sync_if_needed(device, sync_timing)
                h2d_end = time.perf_counter()
                input_latents = _encode_video_latents(
                    vae,
                    video,
                    device=device,
                    backend=encode_backend,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                history_video_latents = _encode_video_latents(
                    vae,
                    history_video,
                    device=device,
                    backend=encode_backend,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                _sync_if_needed(device, sync_timing)
                encode_end = time.perf_counter()

                sample_indices = batch["sample_idx"].detach().cpu().tolist()
                save_start = time.perf_counter()
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
                save_end = time.perf_counter()
                if log_timing:
                    timing["batches"] += 1
                    timing["samples"] += len(sample_indices)
                    timing["data_wait_seconds"] += data_wait_end - data_wait_start
                    timing["h2d_seconds"] += h2d_end - h2d_start
                    timing["encode_seconds"] += encode_end - h2d_end
                    timing["save_seconds"] += save_end - save_start

                pbar.update(len(sample_indices))
    timing["wall_seconds"] = time.perf_counter() - wall_start

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
        if log_timing and timing["samples"] > 0:
            measured = (
                timing["ram_preload_seconds"]
                + timing["data_wait_seconds"]
                + timing["h2d_seconds"]
                + timing["encode_seconds"]
                + timing["save_seconds"]
            )
            active = timing["h2d_seconds"] + timing["encode_seconds"] + timing["save_seconds"]
            streaming_bound = max(timing["ram_preload_seconds"], active)
            logger.info(
                "VAE latent precompute timing: samples=%d batches=%d ram_preload=%.3fs "
                "ram_preload_bytes=%.2fGiB data_wait=%.3fs h2d=%.3fs encode=%.3fs save=%.3fs "
                "active_total=%.3fs measured_total=%.3fs streaming_bound=%.3fs wall=%.3fs "
                "wall_samples_per_s=%.3f active_samples_per_s=%.3f measured_samples_per_s=%.3f "
                "streaming_bound_samples_per_s=%.3f",
                timing["samples"],
                timing["batches"],
                timing["ram_preload_seconds"],
                timing["ram_preload_bytes"] / (1024 ** 3),
                timing["data_wait_seconds"],
                timing["h2d_seconds"],
                timing["encode_seconds"],
                timing["save_seconds"],
                active,
                measured,
                streaming_bound,
                timing["wall_seconds"],
                timing["samples"] / timing["wall_seconds"] if timing["wall_seconds"] > 0 else float("nan"),
                timing["samples"] / active if active > 0 else float("nan"),
                timing["samples"] / measured if measured > 0 else float("nan"),
                timing["samples"] / streaming_bound if streaming_bound > 0 else float("nan"),
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
