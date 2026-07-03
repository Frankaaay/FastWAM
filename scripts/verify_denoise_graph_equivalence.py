#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.bench_infer_action import (  # noqa: E402
    compose_config,
    instantiate_random_model,
    load_checkpoint_if_requested,
    make_inputs,
    normalize_device,
    select_int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify infer_action denoise CUDA graph equivalence.")
    parser.add_argument("--task", default="fold_cloth_orig_2epoch", help="Hydra task config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable Hydra override, e.g. --override model.mot_checkpoint_mixed_attn=false",
    )
    parser.add_argument("--ckpt", default=None, help="Optional FastWAM checkpoint. Omit for random weights.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device. Default: cuda:0")
    parser.add_argument("--height", type=int, default=384, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=320, help="Synthetic image width.")
    parser.add_argument("--context-len", type=int, default=128, help="Synthetic context length.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Default: data.train.num_frames - 1.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic input and denoise seed.")
    parser.add_argument("--abs-tol", type=float, default=0.0, help="Non-zero exit when max_abs exceeds this.")
    parser.add_argument("--allow-fallback", action="store_true", help="Do not fail when graph path falls back.")
    parser.add_argument("--strict", action="store_true", help="Raise AssertionError instead of returning non-zero.")
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
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if args.abs_tol < 0:
        raise ValueError("--abs-tol must be non-negative.")


def state_signature(model: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    return {key: (tuple(value.shape), str(value.dtype)) for key, value in model.state_dict().items()}


def assert_state_compatible(label: str, before: dict[str, tuple[tuple[int, ...], str]], model: Any) -> None:
    after = state_signature(model)
    if after != before:
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
        raise AssertionError(
            f"state_dict signature changed after {label}: "
            f"added={added[:10]} missing={missing[:10]} changed={changed[:10]}"
        )


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float()
    got_f = got.float()
    diff = (ref_f - got_f).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = float(ref_f.abs().max().item())
    rel = max_abs / max(denom, torch.finfo(torch.float32).eps)
    ref_flat = ref_f.flatten()
    got_flat = got_f.flatten()
    if float(ref_flat.norm().item()) == 0.0 or float(got_flat.norm().item()) == 0.0:
        cosine = 1.0 if max_abs == 0.0 else 0.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(ref_flat, got_flat, dim=0).item())
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel": rel, "cosine": cosine}


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| call | graph_status | max_abs | mean_abs | rel | cosine | status |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        metric_row = row["metrics"]
        print(
            f"| {row['call']} | {row['graph_status']} | {metric_row['max_abs']:.6g} | "
            f"{metric_row['mean_abs']:.6g} | {metric_row['rel']:.6g} | "
            f"{metric_row['cosine']:.8f} | {row['status']} |"
        )


@torch.no_grad()
def run_once(torch_mod: Any, model: Any, kwargs: dict[str, Any], enabled: bool) -> tuple[torch.Tensor, str]:
    model.infer_denoise_cuda_graph = bool(enabled)
    model._last_infer_denoise_cuda_graph_status = "pending"
    out = model.infer_action(**kwargs)["action"]
    torch_mod.cuda.synchronize()
    status = str(getattr(model, "_last_infer_denoise_cuda_graph_status", "missing"))
    return out, status


def main() -> int:
    args = parse_args()
    validate_args(args)

    cfg = compose_config(args)
    device = normalize_device(torch, args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    model, _, _ = instantiate_random_model(cfg, device)
    checkpoint = load_checkpoint_if_requested(model, args.ckpt)
    action_horizon = int(args.action_horizon or (select_int(cfg, "data.train.num_frames", 33) - 1))
    kwargs_first = make_inputs(torch, model, args, action_horizon)
    torch.manual_seed(int(args.seed) + 1)
    torch.cuda.manual_seed_all(int(args.seed) + 1)
    kwargs_second = make_inputs(torch, model, args, action_horizon)
    kwargs_second["seed"] = int(args.seed)
    state_before = state_signature(model)

    baseline_first, _baseline_first_status = run_once(torch, model, kwargs_first, enabled=False)
    assert_state_compatible("first eager infer_action", state_before, model)

    baseline_second, _baseline_second_status = run_once(torch, model, kwargs_second, enabled=False)
    assert_state_compatible("second eager infer_action", state_before, model)

    graph_first, graph_first_status = run_once(torch, model, kwargs_first, enabled=True)
    assert_state_compatible("first graph infer_action", state_before, model)

    graph_second, graph_second_status = run_once(torch, model, kwargs_second, enabled=True)
    assert_state_compatible("second graph infer_action", state_before, model)

    first_row = metrics(baseline_first, graph_first)
    second_row = metrics(baseline_second, graph_second)
    expected_statuses = (graph_first_status == "captured" and graph_second_status == "reused")
    graph_ok = expected_statuses or bool(args.allow_fallback)
    first_within_tol = first_row["max_abs"] <= float(args.abs_tol)
    second_within_tol = second_row["max_abs"] <= float(args.abs_tol)
    rows = [
        {
            "call": "first",
            "graph_status": graph_first_status,
            "metrics": first_row,
            "status": "PASS" if graph_ok and first_within_tol else "FAIL",
        },
        {
            "call": "second",
            "graph_status": graph_second_status,
            "metrics": second_row,
            "status": "PASS" if graph_ok and second_within_tol else "FAIL",
        },
    ]
    print_table(rows)
    print(f"checkpoint: {checkpoint}")
    print(f"shape_first: {tuple(baseline_first.shape)}")
    print(f"shape_second: {tuple(baseline_second.shape)}")

    if not graph_ok:
        message = (
            "CUDA graph path did not use persistent replay successfully: "
            f"first={graph_first_status}, second={graph_second_status}, expected first=captured second=reused"
        )
        if args.strict:
            raise AssertionError(message)
        print(message)
        return 2
    if not first_within_tol or not second_within_tol:
        message = (
            f"max_abs first={first_row['max_abs']:.6g} second={second_row['max_abs']:.6g} "
            f"exceeds tolerance {float(args.abs_tol):.6g}"
        )
        if args.strict:
            raise AssertionError(message)
        print(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
