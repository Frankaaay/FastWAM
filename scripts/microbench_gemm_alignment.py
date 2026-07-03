#!/usr/bin/env python
"""Probe whether bf16 GEMM kernel choice changes with 16B operand alignment.

Default mode benchmarks torch matmul with logical A[M,K] and B[K,N].
Pass --linear to benchmark F.linear with input A[M,K] and weight B[N,K],
which is closer to the training Linear path.
Pass --fp8 to compare the bf16 baseline with torch._scaled_mm FP8 E4M3.
"""

import argparse
import math

torch = None
F = None


VARIANTS = (
    ("baseline", False, False),
    ("A_misaligned", True, False),
    ("B_misaligned", False, True),
    ("A_B_misaligned", True, True),
)
FP8_VARIANT = "fp8_e4m3"


def _parse_int_list(value):
    try:
        values = [int(part) for part in value.replace(",", " ").split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_shape(value):
    dims = _parse_int_list(value)
    if len(dims) != 3:
        raise argparse.ArgumentTypeError("shape must have exactly 3 integers: M,K,N")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return tuple(dims)


def _build_shapes(args):
    if args.shape:
        return args.shape

    shapes = [(args.m, args.k, n) for n in args.n]
    if not args.no_dgrad:
        shapes.append((args.m, args.dgrad_k, args.dgrad_n))
    return shapes


def _make_bf16_tensor(shape, *, misaligned, device):
    numel = math.prod(shape)
    if misaligned:
        buf = torch.empty(numel + 8, device=device, dtype=torch.bfloat16)
        tensor = buf.narrow(0, 1, numel).view(shape)
        if tensor.data_ptr() % 16 == 0:
            raise RuntimeError("misaligned tensor unexpectedly has 16B-aligned data_ptr")
    else:
        tensor = torch.empty(shape, device=device, dtype=torch.bfloat16)
        if tensor.data_ptr() % 16 != 0:
            raise RuntimeError("baseline tensor unexpectedly has non-16B-aligned data_ptr")
    return tensor


def _make_fp8_tensor(shape, *, column_major, device):
    tensor = torch.empty(shape, device=device, dtype=torch.float8_e4m3fn)
    if not column_major:
        return tensor

    tensor = tensor.t().contiguous().t()
    if tensor.shape != shape or tensor.stride(0) != 1 or tensor.stride(1) != shape[0]:
        raise RuntimeError(
            f"FP8 B tensor is not column-major: shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
        )
    return tensor


def _fp8_skip_reason(m, k, n, device):
    if not hasattr(torch, "_scaled_mm"):
        return "torch._scaled_mm is unavailable"
    if not hasattr(torch, "float8_e4m3fn"):
        return "torch.float8_e4m3fn is unavailable"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (9, 0):
        return f"FP8 _scaled_mm benchmark expects SM90+, got sm_{major}{minor}"
    if k % 16 != 0:
        return f"K must be divisible by 16 for FP8 _scaled_mm, got K={k}"
    if n % 16 != 0:
        return f"N must be divisible by 16 for FP8 _scaled_mm, got N={n}"
    return None


def _cuda_self_time_us(event):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    for attr in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _top_cuda_kernel(run_op, device):
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        with torch.no_grad():
            out = run_op()
        torch.cuda.synchronize(device)
        del out

    fallback = []
    kernels = []
    skip_prefixes = ("aten::", "cuda", "Cuda", "ProfilerStep", "[memory]")
    for event in prof.key_averages():
        time_us = _cuda_self_time_us(event)
        if time_us <= 0:
            continue
        key = event.key
        fallback.append((time_us, key))
        if not key.startswith(skip_prefixes):
            kernels.append((time_us, key))

    ranked = kernels or fallback
    if not ranked:
        return "n/a"
    ranked.sort(reverse=True)
    return ranked[0][1]


def _time_ms(run_op, *, warmup, iters, device):
    with torch.no_grad():
        for _ in range(warmup):
            out = run_op()
            del out
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = run_op()
            del out
        end.record()
        torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(iters, 1)


def _tflops(m, k, n, ms):
    return (2.0 * m * k * n) / (ms / 1000.0) / 1.0e12


def _format_skip_reason(reason):
    text = " ".join(str(reason).split())
    if len(text) > 240:
        text = f"{text[:237]}..."
    return text


def _print_result(shape_label, variant, m, k, n, ms, top_kernel):
    print(f"{shape_label}\t{variant}\t{ms:.4f}\t{_tflops(m, k, n, ms):.3f}\t{top_kernel}", flush=True)


def _print_skip(shape_label, variant, reason):
    print(f"{shape_label}\t{variant}\tskip\tskip\tskip: {_format_skip_reason(reason)}", flush=True)


def _run_bf16_variant(m, k, n, args, device, shape_label, variant, misalign_a, misalign_b):
    a = _make_bf16_tensor((m, k), misaligned=misalign_a, device=device)
    if args.linear:
        b = _make_bf16_tensor((n, k), misaligned=misalign_b, device=device)

        def run_op():
            return F.linear(a, b)

    else:
        b = _make_bf16_tensor((k, n), misaligned=misalign_b, device=device)

        def run_op():
            return a @ b

    ms = _time_ms(run_op, warmup=args.warmup, iters=args.iters, device=device)
    top_kernel = _top_cuda_kernel(run_op, device)
    _print_result(shape_label, variant, m, k, n, ms, top_kernel)
    del a, b
    torch.cuda.empty_cache()


def _run_fp8_variant(m, k, n, args, device, shape_label):
    reason = _fp8_skip_reason(m, k, n, device)
    if reason:
        _print_skip(shape_label, FP8_VARIANT, reason)
        return

    a = b = scale_a = scale_b = None
    try:
        a = _make_fp8_tensor((m, k), column_major=False, device=device)
        b = _make_fp8_tensor((k, n), column_major=True, device=device)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)

        def run_op():
            return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

        ms = _time_ms(run_op, warmup=args.warmup, iters=args.iters, device=device)
        top_kernel = _top_cuda_kernel(run_op, device)
        _print_result(shape_label, FP8_VARIANT, m, k, n, ms, top_kernel)
    except Exception as exc:
        _print_skip(shape_label, FP8_VARIANT, exc)
    finally:
        del a, b, scale_a, scale_b
        torch.cuda.empty_cache()


def _run_one_shape(m, k, n, args, device):
    shape_label = f"M={m},K={k},N={n}"
    if args.fp8:
        _run_bf16_variant(m, k, n, args, device, shape_label, "bf16_baseline", False, False)
        _run_fp8_variant(m, k, n, args, device, shape_label)
        return

    for variant, misalign_a, misalign_b in VARIANTS:
        _run_bf16_variant(m, k, n, args, device, shape_label, variant, misalign_a, misalign_b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=9000, help="Default M for generated shapes.")
    parser.add_argument("--k", type=int, default=3072, help="Default K for generated shapes.")
    parser.add_argument(
        "--n",
        type=_parse_int_list,
        default=[9216, 3072, 14336],
        help="Comma or space separated N values for generated shapes.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Override benchmark shapes as M,K,N. Can be repeated.",
    )
    parser.add_argument("--dgrad-k", type=int, default=14336, help="K for the default dgrad-like shape.")
    parser.add_argument("--dgrad-n", type=int, default=3072, help="N for the default dgrad-like shape.")
    parser.add_argument("--no-dgrad", action="store_true", help="Do not append the default dgrad-like shape.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per variant.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per variant.")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument("--linear", action="store_true", help="Benchmark F.linear instead of torch matmul.")
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="For each shape, output bf16_baseline and fp8_e4m3 rows instead of alignment variants.",
    )
    args = parser.parse_args()

    global torch, F
    import torch as torch_module
    import torch.nn.functional as functional

    torch = torch_module
    F = functional

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.m <= 0 or args.k <= 0 or args.dgrad_k <= 0 or args.dgrad_n <= 0:
        raise ValueError("all generated shape dimensions must be positive")
    if any(n <= 0 for n in args.n):
        raise ValueError("all --n values must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        torch.cuda.set_device(device)

    print("shape\tvariant\tms\tTFLOPS\ttop_kernel", flush=True)
    for m, k, n in _build_shapes(args):
        _run_one_shape(m, k, n, args, device)


if __name__ == "__main__":
    main()
