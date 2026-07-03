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


def _metrics(orig: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    diff = (orig.float() - got.float()).abs()
    max_abs = float(diff.max().item())
    denom = float(orig.float().abs().max().item())
    rel = max_abs / max(denom, torch.finfo(torch.float32).eps)
    return max_abs, rel


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

    rows: list[tuple[str, float, float, str]] = []

    func = encode_functional(vae.model, x, vae.scale, compile_mode=None, compiled_cache={})
    _synchronize(device)
    _assert_state_keys("eager functional encode", state_keys, vae)
    max_abs, rel = _metrics(orig, func)
    exact = torch.equal(orig, func)
    rows.append(("A eager functional", max_abs, rel, "PASS" if exact and max_abs == 0.0 else "FAIL"))
    if not exact or max_abs != 0.0:
        _print_rows(rows)
        raise AssertionError("A-level eager functional encode is not bitwise equal to original encode")

    for label, mode in (
        ("B compile default", "default"),
        ("C compile reduce-overhead", "reduce-overhead"),
    ):
        compiled = encode_functional(
            vae.model,
            x,
            vae.scale,
            compile_mode=mode,
            compiled_cache={},
        )
        _synchronize(device)
        _assert_state_keys(label, state_keys, vae)
        max_abs, rel = _metrics(orig, compiled)
        passed = max_abs < float(args.abs_tol) and rel < float(args.rel_tol)
        rows.append((label, max_abs, rel, "PASS" if passed else "FAIL"))
        if not passed:
            _print_rows(rows)
            raise AssertionError(
                f"{label} exceeded tolerance: max_abs={max_abs:.6g}, rel={rel:.6g}"
            )

    _print_rows(rows)
    return 0


def _print_rows(rows: list[tuple[str, float, float, str]]) -> None:
    print("| level | max_abs | rel | status |")
    print("| --- | ---: | ---: | --- |")
    for label, max_abs, rel, status in rows:
        print(f"| {label} | {max_abs:.6g} | {rel:.6g} | {status} |")


if __name__ == "__main__":
    raise SystemExit(main())
