#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.models.wan22.vae_encode_functional import encode_functional
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype={name!r}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metrics(orig: torch.Tensor, got: torch.Tensor) -> dict:
    o = orig.float()
    g = got.float()
    diff = (o - g).abs()
    max_abs = float(diff.max().item())
    denom = float(o.abs().max().item())
    rel = max_abs / max(denom, torch.finfo(torch.float32).eps)
    mean_abs = float(diff.mean().item())
    p999 = float(torch.quantile(diff.flatten().float(), 0.999).item())
    cos = float(
        torch.nn.functional.cosine_similarity(o.flatten(), g.flatten(), dim=0).item()
    )
    return {"max_abs": max_abs, "rel": rel, "mean_abs": mean_abs, "p999": p999, "cos": cos}


def _assert_state_keys(label: str, before: tuple[str, ...], vae: WanVideoVAE38) -> None:
    after = tuple(vae.state_dict().keys())
    if after != before:
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        raise AssertionError(
            f"state_dict keys changed after {label}: added={added[:10]} missing={missing[:10]}"
        )


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify functional VAE encode equivalence.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--abs-tol", type=float, default=1e-2)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="raise on B/C tolerance breach; default reports all levels without failing on compile rounding",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dtype = _dtype_from_name(args.dtype)
    vae = WanVideoVAE38().eval().requires_grad_(False).to(device=device, dtype=dtype)
    state_keys = tuple(vae.state_dict().keys())
    x = torch.randn(
        1,
        3,
        int(args.frames),
        int(args.height),
        int(args.width),
        device=device,
        dtype=dtype,
    )

    orig = vae.model.encode(x, vae.scale)
    _synchronize(device)
    _assert_state_keys("original encode", state_keys, vae)

    rows: list[tuple[str, dict, str]] = []

    func = encode_functional(vae.model, x, vae.scale, compile_mode=None, compiled_cache={})
    _synchronize(device)
    _assert_state_keys("eager functional encode", state_keys, vae)
    m = _metrics(orig, func)
    exact = torch.equal(orig, func)
    rows.append(("A eager functional", m, "PASS" if exact and m["max_abs"] == 0.0 else "FAIL"))
    if not exact or m["max_abs"] != 0.0:
        _print_rows(rows)
        raise AssertionError("A-level eager functional encode is not bitwise equal to original encode")

    breach = []
    for label, mode in (
        ("B compile default", "default"),
        ("C compile reduce-overhead", "reduce-overhead"),
    ):
        try:
            compiled = encode_functional(
                vae.model,
                x,
                vae.scale,
                compile_mode=mode,
                compiled_cache={},
            )
            _synchronize(device)
        except Exception as exc:
            rows.append((label, {"max_abs": float("nan"), "rel": float("nan"), "mean_abs": float("nan"), "p999": float("nan"), "cos": float("nan")}, f"CRASH: {type(exc).__name__}"))
            _print_rows(rows)
            raise
        _assert_state_keys(label, state_keys, vae)
        m = _metrics(orig, compiled)
        passed = m["max_abs"] < float(args.abs_tol) and m["rel"] < float(args.rel_tol)
        rows.append((label, m, "PASS" if passed else "LOOSE"))
        if not passed:
            breach.append(label)

    _print_rows(rows)
    if breach and args.strict:
        raise AssertionError(f"tolerance breach in strict mode: {breach}")
    return 0


def _print_rows(rows: list[tuple[str, dict, str]]) -> None:
    print("| level | max_abs | rel | mean_abs | p99.9 | cosine | status |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for label, m, status in rows:
        print(
            f"| {label} | {m['max_abs']:.6g} | {m['rel']:.6g} | {m['mean_abs']:.6g} "
            f"| {m['p999']:.6g} | {m['cos']:.8f} | {status} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
