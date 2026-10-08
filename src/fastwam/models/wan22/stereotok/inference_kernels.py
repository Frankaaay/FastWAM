"""Shared inference kernels from peilin/profiling@f125820.

The native-fp16-conv recipe explicitly rounds large convolution operands/output.
Weighted matching and RMS pointwise operations retain the profiled FP32 order.
Imported lazily by the CUDA inference runtime; CPU/training need no Triton.
"""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def install_implicit_spatial_padding(encoder, half_spatial_convs=False):
    """Use cuDNN spatial padding; preserve the original temporal cache/padding."""
    from .vae2_2 import CausalConv3d
    for module in encoder.modules():
        if not isinstance(module, CausalConv3d):
            continue
        if module.padding_mode != "zeros" or module.padding != (0, 0, 0):
            raise ValueError("Implicit spatial padding requires the original causal convolution")
        pw, pr, ph, pb, pt, back = module._padding
        if pw != pr or ph != pb or back != 0:
            raise ValueError("Unsupported asymmetric spatial padding")
        half = half_spatial_convs and module.kernel_size == (3, 3, 3)
        if half:
            if module.weight.requires_grad or module.weight.dtype != torch.float32:
                raise ValueError("Reduced precision candidate requires frozen FP32 masters")
            module.register_buffer("_profiling_half_weight", module.weight.detach().half(), persistent=False)
        def forward(x, cache_x=None, module=module, pw=pw, ph=ph, pt=pt, half=half):
            if cache_x is not None and pt > 0:
                x = torch.cat((cache_x.to(x.device), x), dim=2)
                temporal = pt - cache_x.shape[2]
            else:
                temporal = pt
            if temporal:
                x = F.pad(x, (0, 0, 0, 0, temporal, 0))
            if half:
                # Explicit lossy candidate: half operands/output, FP32 bias and following layers.
                result = F.conv3d(x.half(), module._profiling_half_weight, None, module.stride,
                                  (0, ph, pw), module.dilation, module.groups).float()
                return result if module.bias is None else result + module.bias.view(1, -1, 1, 1, 1)
            return F.conv3d(x, module.weight, module.bias, module.stride,
                            (0, ph, pw), module.dilation, module.groups)
        module.forward = forward


@triton.jit
def _weighted_match(V, P, O, N: tl.constexpr, C: tl.constexpr, T: tl.constexpr,
                    H: tl.constexpr, W: tl.constexpr, D: tl.constexpr,
                    VB: tl.constexpr, VC: tl.constexpr, VT: tl.constexpr,
                    VH: tl.constexpr, VW: tl.constexpr,
                    PB: tl.constexpr, PD: tl.constexpr, PT: tl.constexpr,
                    PH: tl.constexpr, PW: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = index % C
    x = index // C % W
    y = index // (C * W) % H
    t = index // (C * W * H) % T
    b = index // (C * W * H * T)
    value_base = b * VB + c * VC + t * VT + y * VH + x * VW
    weight_base = b * PB + t * PT + y * PH + x * PW
    total = tl.full((BLOCK,), 0, tl.float32)
    # Separate mul/add, original d order, no FMA or reduction reassociation.
    for d in range(D):
        valid = (index < N) & (x >= d)
        value = tl.load(V + value_base - d * VW, valid, 0)
        weight = tl.load(P + weight_base + d * PD, valid, 0)
        product = value * weight
        contribution = tl.where(x >= d, product, 0.)
        total = total + contribution
    tl.store(O + index, total, index < N)


def weighted_match(values, weights, disparities):
    if values.dtype != torch.float32 or weights.dtype != torch.float32:
        raise ValueError("Exact candidate is only implemented for the FP32 profiling contract")
    b, c, t, h, w = values.shape
    result = torch.empty((b, t, h, w, c), device=values.device, dtype=values.dtype)
    n = result.numel()
    _weighted_match[(triton.cdiv(n, 256),)](
        values, weights, result, n, c, t, h, w, disparities,
        *values.stride(), *weights.stride(), 256, enable_fp_fusion=False)
    return result.permute(0, 4, 1, 2, 3)


def install_weighted_match(fusion):
    if fusion.mode != "early":
        raise ValueError("This experiment is restricted to the pinned early checkpoint")
    def forward(left, right, probabilities, feat_cache=None):
        if left.shape != right.shape or probabilities.shape != (left.shape[0], 48, *left.shape[2:]):
            raise ValueError("LAS and Wan quarter-resolution grids must coincide")
        values = fusion.value(right.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        weights = probabilities.to(values.dtype)
        disparities = min(48, left.shape[-1], math.ceil(fusion.max_disparity / 4) + 1)
        matched = weighted_match(values, weights, disparities)
        geometry = fusion.descriptor(weights.permute(0, 2, 3, 4, 1))
        feature = torch.cat((left.permute(0, 2, 3, 4, 1), matched.permute(0, 2, 3, 4, 1), geometry), -1)
        support = probabilities.sum(1, keepdim=True) > 0
        return fusion.aggregate(feature).permute(0, 4, 1, 2, 3) * support
    fusion.forward = forward


@triton.jit
def _norm_pointwise(X, D, G, O, N: tl.constexpr, C: tl.constexpr,
                    T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                    XB: tl.constexpr, XC: tl.constexpr, XT: tl.constexpr,
                    XH: tl.constexpr, XW: tl.constexpr,
                    OB: tl.constexpr, OC: tl.constexpr, OT: tl.constexpr,
                    OH: tl.constexpr, OW: tl.constexpr,
                    SCALE: tl.constexpr, CHANNEL_LAST: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    S: tl.constexpr = T * H * W
    batch = index // (S * C)
    if CHANNEL_LAST:
        channel = index % C
        spatial = index // C % S
    else:
        channel = index // S % C
        spatial = index % S
    time = spatial // (H * W)
    height, width = spatial // W % H, spatial % W
    input_offset = batch * XB + channel * XC + time * XT + height * XH + width * XW
    output_offset = batch * OB + channel * OC + time * OT + height * OH + width * OW
    denominator_index = batch * S + spatial
    x = tl.load(X + input_offset, index < N, 0)
    denominator = tl.load(D + denominator_index, index < N, 1)
    gamma = tl.load(G + channel, channel < C, 1)
    denominator = tl.maximum(denominator, 1.e-12)
    result = tl.div_rn(x, denominator)
    result = result * SCALE
    result = result * gamma
    result = result + 0.
    tl.store(O + output_offset, result, index < N)


def norm_pointwise(x, module, denominator=None):
    """Peilin's FP32 RMS pointwise kernel with the original ATen reduction."""
    if x.dtype != torch.float32 or x.ndim not in (4, 5):
        raise ValueError("Fused RMS norm requires FP32 image/video activations")
    if denominator is None:
        denominator = x.norm(2., dim=1, keepdim=True)
    if not denominator.is_contiguous():
        raise ValueError("Unexpected ATen norm reduction output layout")
    result = torch.empty_like(x)
    n, c = x.numel(), x.shape[1]
    xv = x.unsqueeze(2) if x.ndim == 4 else x
    ov = result.unsqueeze(2) if x.ndim == 4 else result
    channels_last = ov.is_contiguous(memory_format=torch.channels_last_3d)
    _norm_pointwise[(triton.cdiv(n, 256),)](
        x, denominator, module.gamma, result, n, c, *xv.shape[2:],
        *xv.stride(), *ov.stride(), module.scale, channels_last, 256,
        enable_fp_fusion=False)
    return result
