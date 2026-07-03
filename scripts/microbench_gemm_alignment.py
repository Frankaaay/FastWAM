#!/usr/bin/env python
"""Probe whether bf16 GEMM kernel choice changes with 16B operand alignment.

Default mode benchmarks torch matmul with logical A[M,K] and B[K,N].
Pass --linear to benchmark F.linear with input A[M,K] and weight B[N,K],
which is closer to the training Linear path.
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


def _run_one_shape(m, k, n, args, device):
    shape_label = f"M={m},K={k},N={n}"
    for variant, misalign_a, misalign_b in VARIANTS:
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
        print(f"{shape_label}\t{variant}\t{ms:.4f}\t{_tflops(m, k, n, ms):.3f}\t{top_kernel}", flush=True)
        del a, b
        torch.cuda.empty_cache()


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
