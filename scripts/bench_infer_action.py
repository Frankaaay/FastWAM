#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_TASK = "fold_cloth_orig_2epoch"
DEFAULT_OUT = "runs/bench_infer_action"
INFER_STAGE_ORDER = {
    "model/infer/encode_input_image": 0,
    "model/infer/current_video_prefill": 1,
    "model/infer/denoise_step": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU FastWAM.infer_action latency benchmark.")
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"Hydra task config. Default: {DEFAULT_TASK}")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable Hydra override, e.g. --override model.mot_torch_compile=true",
    )
    parser.add_argument("--ckpt", default=None, help="Optional FastWAM checkpoint. Omit for random weights.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device. Default: cuda:0")
    parser.add_argument("--height", type=int, default=384, help="Synthetic image height. Default: 384.")
    parser.add_argument("--width", type=int, default=320, help="Synthetic image width. Default: 320.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup infer_action calls.")
    parser.add_argument("--iters", type=int, default=10, help="Timed infer_action calls.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output directory. Default: {DEFAULT_OUT}")
    parser.add_argument("--context-len", type=int, default=128, help="Synthetic cached text context length.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Default: data.train.num_frames - 1.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic input and action-noise seed.")
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export one torch.profiler chrome trace after event timing. Default: true.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"--height/--width must be multiples of 16, got {args.height}x{args.width}.")
    if args.context_len <= 0:
        raise ValueError("--context-len must be positive.")
    if args.action_horizon is not None and args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive when provided.")
    if args.num_inference_steps <= 0 or args.iters <= 0:
        raise ValueError("--num-inference-steps and --iters must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")


def compose_config(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", *(args.override or [])])
    OmegaConf.resolve(cfg)
    return cfg


def as_dict(value: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if value is None:
        return {}
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def select_int(cfg: Any, key: str, default: int | None = None) -> int | None:
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key)
    return default if value is None else int(value)


def normalize_device(torch_mod: Any, device_arg: str) -> str:
    text = str(device_arg)
    if text.isdigit():
        text = f"cuda:{text}"
    device = torch_mod.device(text)
    if device.type != "cuda":
        raise ValueError("This benchmark uses CUDA events; pass a CUDA device.")
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    index = 0 if device.index is None else int(device.index)
    if index >= torch_mod.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, only {torch_mod.cuda.device_count()} CUDA devices are visible.")
    torch_mod.cuda.set_device(index)
    return f"cuda:{index}"


def model_dtype_from_cfg(cfg: Any):
    from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    return _mixed_precision_to_model_dtype(mixed_precision), mixed_precision


def instantiate_random_model(cfg: Any, device: str):
    from omegaconf import OmegaConf
    import torch

    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    model_dtype, mixed_precision = model_dtype_from_cfg(cfg)
    model_cfg = cfg.model
    video_cfg = as_dict(model_cfg.video_dit_config)
    action_cfg = as_dict(model_cfg.action_dit_config)
    video_sched = as_dict(model_cfg.get("video_scheduler"))
    action_sched = as_dict(model_cfg.get("action_scheduler"))
    loss_cfg = as_dict(model_cfg.get("loss"))

    video_expert = WanVideoDiT(**video_cfg)
    action_expert = ActionDiT(**action_cfg)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=bool(model_cfg.get("mot_checkpoint_mixed_attn", True)),
        mot_torch_compile=bool(model_cfg.get("mot_torch_compile", False)),
        mot_torch_compile_mode=str(model_cfg.get("mot_torch_compile_mode", "default")),
    )
    proprio_dim = OmegaConf.select(model_cfg, "proprio_dim")
    model = FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=WanVideoVAE38(),
        text_encoder=None,
        tokenizer=None,
        text_dim=int(video_cfg["text_dim"]),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        video_train_shift=float(video_sched.get("train_shift", 5.0)),
        video_infer_shift=float(video_sched.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_sched.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_sched.get("train_shift", 5.0)),
        action_infer_shift=float(action_sched.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_sched.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss_cfg.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss_cfg.get("lambda_action", 1.0)),
        vae_torch_compile=bool(model_cfg.get("vae_torch_compile", False)),
        vae_torch_compile_mode=str(model_cfg.get("vae_torch_compile_mode", "default")),
        vae_encode_functional=bool(model_cfg.get("vae_encode_functional", False)),
        vae_encode_functional_mode=str(model_cfg.get("vae_encode_functional_mode", "reduce-overhead")),
        infer_denoise_cuda_graph=bool(model_cfg.get("infer_denoise_cuda_graph", False)),
    )
    model.eval()
    # 随机构造路径下 VAE 默认 fp32，与真实加载（bf16 权重）不同；统一整模型 dtype，
    # 否则 VAE conv bias(float) 与 bf16 输入不匹配直接报错。
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


def make_inputs(torch_mod: Any, model: Any, args: argparse.Namespace, action_horizon: int) -> dict[str, Any]:
    device = model.device
    dtype = model.torch_dtype
    image = torch_mod.rand((1, 3, args.height, args.width), device=device, dtype=dtype).mul_(2).sub_(1)
    context = torch_mod.randn((1, args.context_len, int(model.text_dim)), device=device, dtype=dtype)
    context_mask = torch_mod.ones((1, args.context_len), device=device, dtype=torch_mod.bool)
    proprio = None
    if getattr(model, "proprio_dim", None) is not None:
        proprio = torch_mod.randn((1, int(model.proprio_dim)), device=device, dtype=dtype)
    return {
        "prompt": None,
        "input_image": image,
        "action_horizon": int(action_horizon),
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "rand_device": str(device),
    }


def time_iters(torch_mod: Any, model: Any, kwargs: dict[str, Any], warmup: int, iters: int) -> list[float]:
    with torch_mod.inference_mode():
        for _ in range(warmup):
            model.infer_action(**kwargs)
    torch_mod.cuda.synchronize()

    times_ms: list[float] = []
    with torch_mod.inference_mode():
        for _ in range(iters):
            start = torch_mod.cuda.Event(enable_timing=True)
            end = torch_mod.cuda.Event(enable_timing=True)
            torch_mod.cuda.synchronize()
            start.record()
            model.infer_action(**kwargs)
            end.record()
            torch_mod.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def profiler_us(event: Any, cuda_attr: str, device_attr: str) -> float:
    value = getattr(event, cuda_attr, None)
    if value is None:
        value = getattr(event, device_attr, 0.0)
    return float(value or 0.0)


def profiler_rows(prof: Any) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for event in prof.key_averages():
        name = str(event.key)
        if not name.startswith("model/infer/"):
            continue
        count = int(getattr(event, "count", 0) or 0)
        cuda_us = profiler_us(event, "cuda_time_total", "device_time_total")
        self_cuda_us = profiler_us(event, "self_cuda_time_total", "self_device_time_total")
        rows.append(
            {
                "name": name,
                "count": count,
                "cuda_total_ms": cuda_us / 1000.0,
                "cuda_avg_ms": (cuda_us / max(count, 1)) / 1000.0,
                "self_cuda_total_ms": self_cuda_us / 1000.0,
                "cpu_total_ms": float(getattr(event, "cpu_time_total", 0.0) or 0.0) / 1000.0,
            }
        )
    return sorted(rows, key=lambda row: (INFER_STAGE_ORDER.get(str(row["name"]), 999), str(row["name"])))


def run_trace(torch_mod: Any, model: Any, kwargs: dict[str, Any], out_dir: Path, stamp: str):
    from torch.profiler import ProfilerActivity, profile

    trace_path = out_dir / f"infer_action_trace_{stamp}.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:
        with torch_mod.inference_mode():
            model.infer_action(**kwargs)
        torch_mod.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    return trace_path, profiler_rows(prof)


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch

    cfg = compose_config(args)
    device = normalize_device(torch, args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    model, model_dtype, mixed_precision = instantiate_random_model(cfg, device)
    checkpoint = load_checkpoint_if_requested(model, args.ckpt)
    action_horizon = int(args.action_horizon or (select_int(cfg, "data.train.num_frames", 33) - 1))
    kwargs = make_inputs(torch, model, args, action_horizon)

    latencies = time_iters(torch, model, kwargs, args.warmup, args.iters)
    mean_ms = statistics.fmean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = None
    stages: list[dict[str, float | int | str]] = []
    if args.trace:
        trace_path, stages = run_trace(torch, model, kwargs, out_dir, stamp)

    result_path = out_dir / f"infer_action_result_{stamp}.json"
    result = {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "per_iter_ms": latencies,
        "config": {
            "task": args.task,
            "overrides": list(args.override or []),
            "checkpoint": checkpoint,
            "device": device,
            "height": int(args.height),
            "width": int(args.width),
            "num_inference_steps": int(args.num_inference_steps),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "context_len": int(args.context_len),
            "action_horizon": action_horizon,
            "action_dim": int(model.action_expert.action_dim),
            "proprio_dim": None if model.proprio_dim is None else int(model.proprio_dim),
            "mixed_precision": mixed_precision,
            "model_dtype": str(model_dtype),
            "seed": int(args.seed),
        },
        "effective_model_flags": {
            "mot_torch_compile": bool(model.mot.mot_torch_compile),
            "mot_torch_compile_enabled": bool(model.mot.mot_torch_compile_enabled),
            "mot_torch_compile_mode": str(model.mot.mot_torch_compile_mode),
            "vae_torch_compile": bool(model.vae_torch_compile),
            "vae_torch_compile_enabled": bool(model.vae_torch_compile_enabled),
            "vae_encode_functional": bool(model.vae_encode_functional),
            "vae_encode_functional_mode": str(model.vae_encode_functional_mode),
            "infer_denoise_cuda_graph": bool(model.infer_denoise_cuda_graph),
        },
        "artifacts": {
            "result_json": str(result_path),
            "chrome_trace": None if trace_path is None else str(trace_path),
        },
        "profiler_stages": stages,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(f"latency: {mean_ms:.3f} +/- {std_ms:.3f} ms")
    print(
        "config: "
        f"steps={args.num_inference_steps} HxW={args.height}x{args.width} "
        f"iters={args.iters} warmup={args.warmup} overrides={';'.join(args.override or [])}"
    )
    print(f"result_json: {result_path}")
    if trace_path is not None:
        print(f"chrome_trace: {trace_path}")
    if stages:
        print("section\tcount\tcuda_total_ms\tcuda_avg_ms\tself_cuda_total_ms\tcpu_total_ms")
        for row in stages:
            print(
                f"{row['name']}\t{row['count']}\t{row['cuda_total_ms']:.3f}\t"
                f"{row['cuda_avg_ms']:.3f}\t{row['self_cuda_total_ms']:.3f}\t{row['cpu_total_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
