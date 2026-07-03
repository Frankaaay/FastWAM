#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_TASK = "fold_clothv4_v4_2epoch"
DEFAULT_OUT = "runs/bench_infer_action"
INFER_STAGE_ORDER = {
    "model/infer/encode_input_image": 0,
    "model/infer/encode_history_video": 1,
    "model/infer/history_video_prefill": 2,
    "model/infer/history_action_prefill": 3,
    "model/infer/current_video_prefill": 4,
    "model/infer/denoise_step": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU FastWAM.infer_action latency benchmark with torch.profiler trace export."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"Hydra task config. Default: {DEFAULT_TASK}")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, repeatable. Example: --override model.mot_torch_compile=true",
    )
    parser.add_argument("--ckpt", default=None, help="Optional FastWAM checkpoint path. Omit to benchmark random weights.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0 or 0. Default: cuda:0")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output directory for chrome trace. Default: {DEFAULT_OUT}")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup infer_action calls before timing.")
    parser.add_argument("--iters", type=int, default=10, help="Timed infer_action calls.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps passed to infer_action.")
    parser.add_argument("--height", type=int, default=240, help="Synthetic input image height; must be multiple of 16.")
    parser.add_argument("--width", type=int, default=320, help="Synthetic input image width; must be multiple of 16.")
    parser.add_argument("--context-len", type=int, default=128, help="Synthetic text context length.")
    parser.add_argument("--history-frames", type=int, default=9, help="Synthetic history video T; must satisfy T %% 4 == 1.")
    parser.add_argument(
        "--history-action-len",
        type=int,
        default=None,
        help="Synthetic history_action length. Default: model.history_action_len when available.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Future action horizon. Default: data.train.num_frames - 1 from the composed config.",
    )
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument("--history", dest="history", action="store_true", help="Benchmark with history conditioning.")
    history_group.add_argument("--no-history", dest="history", action="store_false", help="Benchmark without history conditioning.")
    parser.set_defaults(history=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"--height/--width must be multiples of 16, got {args.height}x{args.width}.")
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive.")
    if args.history_frames <= 0 or args.history_frames % 4 != 1:
        raise ValueError(f"--history-frames must satisfy T % 4 == 1, got {args.history_frames}.")
    if args.history_action_len is not None and args.history_action_len <= 0:
        raise ValueError("--history-action-len must be positive when provided.")
    if args.action_horizon is not None and args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive when provided.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")


def compose_config(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides = [f"task={args.task}"]
    overrides.extend(args.override or [])
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_cuda_device(torch_mod: Any, device_arg: str) -> str:
    device_text = str(device_arg)
    if device_text.isdigit():
        device_text = f"cuda:{device_text}"
    device = torch_mod.device(device_text)
    if device.type != "cuda":
        raise ValueError("This benchmark uses CUDA events and torch.profiler CUDA activity; pass a CUDA device.")
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device_index = 0 if device.index is None else int(device.index)
    if device_index >= torch_mod.cuda.device_count():
        raise ValueError(
            f"Requested cuda:{device_index}, but only {torch_mod.cuda.device_count()} CUDA devices are visible."
        )
    torch_mod.cuda.set_device(device_index)
    return f"cuda:{device_index}"


def infer_action_horizon(cfg: Any, cli_value: int | None) -> int:
    if cli_value is not None:
        return int(cli_value)
    from omegaconf import OmegaConf

    num_frames = OmegaConf.select(cfg, "data.train.num_frames")
    if num_frames is None:
        return 32
    return int(num_frames) - 1


def infer_history_action_len(model: Any, cli_value: int | None) -> int:
    if cli_value is not None:
        return int(cli_value)
    return int(getattr(model, "history_action_len", 20))


def instantiate_model(cfg: Any, device: str):
    import torch
    from hydra.utils import instantiate

    from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.eval()
    # 随机权重路径下 VAE 默认 fp32；统一整模型 dtype，避免 fp32 bias 与 bf16 输入不匹配。
    model.to(dtype=model_dtype)
    torch.set_grad_enabled(False)
    return model, model_dtype, mixed_precision


def load_checkpoint_if_requested(model: Any, ckpt: str | None) -> str:
    if ckpt is None:
        return "random_weights"
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_checkpoint(str(ckpt_path), optimizer=None)
    return str(ckpt_path)


def make_synthetic_inputs(
    torch_mod: Any,
    model: Any,
    args: argparse.Namespace,
    *,
    action_horizon: int,
    history_action_len: int,
) -> dict[str, Any]:
    device = model.device
    dtype = model.torch_dtype
    action_dim = int(model.action_expert.action_dim)
    text_dim = int(getattr(model, "text_dim", 4096))

    input_image = torch_mod.rand((1, 3, args.height, args.width), device=device, dtype=dtype).mul_(2).sub_(1)
    context = torch_mod.randn((1, args.context_len, text_dim), device=device, dtype=dtype)
    context_mask = torch_mod.ones((1, args.context_len), device=device, dtype=torch_mod.bool)

    proprio = None
    if getattr(model, "proprio_dim", None) is not None:
        proprio = torch_mod.randn((1, int(model.proprio_dim)), device=device, dtype=dtype)

    kwargs: dict[str, Any] = {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": int(action_horizon),
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": int(args.num_inference_steps),
        "rand_device": str(device),
    }

    if args.history:
        kwargs["history_video"] = torch_mod.rand(
            (1, 3, args.history_frames, args.height, args.width),
            device=device,
            dtype=dtype,
        ).mul_(2).sub_(1)
        kwargs["history_action"] = torch_mod.randn(
            (1, history_action_len, action_dim),
            device=device,
            dtype=dtype,
        )

    return kwargs


def run_infer(model: Any, kwargs: dict[str, Any]) -> None:
    model.infer_action(**kwargs)


def run_warmup(torch_mod: Any, model: Any, kwargs: dict[str, Any], warmup: int) -> None:
    with torch_mod.inference_mode():
        for _ in range(warmup):
            run_infer(model, kwargs)
    torch_mod.cuda.synchronize()


def time_with_cuda_events(torch_mod: Any, model: Any, kwargs: dict[str, Any], iters: int) -> list[float]:
    times_ms: list[float] = []
    with torch_mod.inference_mode():
        for _ in range(iters):
            start = torch_mod.cuda.Event(enable_timing=True)
            end = torch_mod.cuda.Event(enable_timing=True)
            torch_mod.cuda.synchronize()
            start.record()
            run_infer(model, kwargs)
            end.record()
            torch_mod.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def profiler_time_us(event: Any, cuda_attr: str, device_attr: str) -> float:
    value = getattr(event, cuda_attr, None)
    if value is None:
        value = getattr(event, device_attr, 0.0)
    return float(value or 0.0)


def profiler_rows(prof: Any) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for event in prof.key_averages():
        key = str(event.key)
        if not key.startswith("model/infer/"):
            continue
        cuda_us = profiler_time_us(event, "cuda_time_total", "device_time_total")
        self_cuda_us = profiler_time_us(event, "self_cuda_time_total", "self_device_time_total")
        cpu_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        rows.append(
            {
                "name": key,
                "count": int(getattr(event, "count", 0) or 0),
                "cuda_total_ms": cuda_us / 1000.0,
                "cuda_avg_ms": (cuda_us / max(int(getattr(event, "count", 0) or 0), 1)) / 1000.0,
                "self_cuda_total_ms": self_cuda_us / 1000.0,
                "cpu_total_ms": cpu_us / 1000.0,
            }
        )
    rows.sort(key=lambda row: (INFER_STAGE_ORDER.get(str(row["name"]), 999), str(row["name"])))
    return rows


def run_profiler(torch_mod: Any, model: Any, kwargs: dict[str, Any], out_dir: Path, history_enabled: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    from torch.profiler import ProfilerActivity, profile

    trace_name = f"infer_action_{'history' if history_enabled else 'no_history'}_{datetime.now():%Y%m%d_%H%M%S}.json"
    trace_path = out_dir / trace_name
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:
        with torch_mod.inference_mode():
            run_infer(model, kwargs)
        torch_mod.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    return trace_path, profiler_rows(prof)


def print_results(
    *,
    args: argparse.Namespace,
    device: str,
    mixed_precision: str,
    model_dtype: Any,
    checkpoint_source: str,
    action_dim: int,
    proprio_dim: int | None,
    action_horizon: int,
    history_action_len: int,
    times_ms: list[float],
    trace_path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    mean_ms = statistics.fmean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print("group\tmetric\tvalue\tunit")
    print(f"config\ttask\t{args.task}\t")
    print(f"config\toverrides\t{';'.join(args.override or [])}\t")
    print(f"config\tcheckpoint\t{checkpoint_source}\t")
    print(f"config\tdevice\t{device}\t")
    print(f"config\tmixed_precision\t{mixed_precision}\t")
    print(f"config\tmodel_dtype\t{model_dtype}\t")
    print(f"config\thistory\t{int(bool(args.history))}\tbool")
    print(f"config\timage_h\t{args.height}\tpixels")
    print(f"config\timage_w\t{args.width}\tpixels")
    print(f"config\tcontext_len\t{args.context_len}\ttokens")
    print(f"config\thistory_frames\t{args.history_frames if args.history else 0}\tframes")
    print(f"config\thistory_action_len\t{history_action_len if args.history else 0}\tactions")
    print(f"config\taction_horizon\t{action_horizon}\tactions")
    print(f"config\taction_dim\t{action_dim}\tdims")
    print(f"config\tproprio_dim\t{proprio_dim if proprio_dim is not None else 0}\tdims")
    print(f"config\tnum_inference_steps\t{args.num_inference_steps}\tsteps")
    print(f"latency\twarmup\t{args.warmup}\titers")
    print(f"latency\titers\t{args.iters}\titers")
    print(f"latency\tmean\t{mean_ms:.3f}\tms")
    print(f"latency\tstd\t{std_ms:.3f}\tms")
    print(f"latency\tmin\t{min(times_ms):.3f}\tms")
    print(f"latency\tmax\t{max(times_ms):.3f}\tms")
    print(f"artifact\tchrome_trace\t{trace_path}\tpath")
    print("")
    print("section\tcount\tcuda_total_ms\tcuda_avg_ms\tself_cuda_total_ms\tcpu_total_ms")
    for row in rows:
        print(
            f"{row['name']}\t{row['count']}\t{row['cuda_total_ms']:.3f}\t"
            f"{row['cuda_avg_ms']:.3f}\t{row['self_cuda_total_ms']:.3f}\t{row['cpu_total_ms']:.3f}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch

    cfg = compose_config(args)
    device = resolve_cuda_device(torch, args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model, model_dtype, mixed_precision = instantiate_model(cfg, device)
    checkpoint_source = load_checkpoint_if_requested(model, args.ckpt)

    action_horizon = infer_action_horizon(cfg, args.action_horizon)
    history_action_len = infer_history_action_len(model, args.history_action_len)
    action_dim = int(model.action_expert.action_dim)
    proprio_dim = None if getattr(model, "proprio_dim", None) is None else int(model.proprio_dim)

    kwargs = make_synthetic_inputs(
        torch,
        model,
        args,
        action_horizon=action_horizon,
        history_action_len=history_action_len,
    )

    run_warmup(torch, model, kwargs, args.warmup)
    times_ms = time_with_cuda_events(torch, model, kwargs, args.iters)
    trace_path, rows = run_profiler(torch, model, kwargs, Path(args.out).expanduser(), bool(args.history))

    print_results(
        args=args,
        device=device,
        mixed_precision=mixed_precision,
        model_dtype=model_dtype,
        checkpoint_source=checkpoint_source,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_horizon=action_horizon,
        history_action_len=history_action_len,
        times_ms=times_ms,
        trace_path=trace_path,
        rows=rows,
    )


if __name__ == "__main__":
    main()
