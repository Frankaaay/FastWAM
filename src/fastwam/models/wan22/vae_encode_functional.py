from collections.abc import MutableMapping
from typing import Any

import torch

from .wan_video_vae import VideoVAE38_, VideoVAE_, patchify


_COMPILE_MODES = {"default", "reduce-overhead"}


def _clone_cache_tensors(feat_cache):
    return [item.clone() if torch.is_tensor(item) else item for item in feat_cache]


def _encode_chunk_first(encoder, x, feat_cache):
    local_cache = list(feat_cache)
    feat_idx = [0]
    out, local_cache, _ = encoder(x, feat_cache=local_cache, feat_idx=feat_idx)
    return out, local_cache


def _encode_chunk_next(encoder, x, feat_cache):
    local_cache = list(feat_cache)
    feat_idx = [0]
    out, local_cache, _ = encoder(x, feat_cache=local_cache, feat_idx=feat_idx)
    return out, local_cache


def _compile_chunk_callable(fn, *, mode: str, name: str):
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    try:
        return torch.compile(fn, mode=mode, dynamic=False, fullgraph=False)
    except Exception as exc:
        raise RuntimeError(
            f"failed to compile functional VAE encode {name} chunk with mode={mode!r}: {exc!r}"
        ) from exc


def _get_chunk_callables(vae_model, compile_mode, compiled_cache):
    encoder = getattr(vae_model, "encoder", None)
    if encoder is None:
        raise RuntimeError("functional VAE encode requires `vae_model.encoder`")

    def first_call(x, feat_cache):
        return _encode_chunk_first(encoder, x, feat_cache)

    def next_call(x, feat_cache):
        return _encode_chunk_next(encoder, x, feat_cache)

    if compile_mode is None:
        return first_call, next_call

    if compile_mode not in _COMPILE_MODES:
        raise ValueError(
            f"unsupported functional VAE encode compile_mode={compile_mode!r}; "
            f"expected one of {sorted(_COMPILE_MODES)} or None"
        )

    if compiled_cache is None:
        compiled_cache = {}
    if not isinstance(compiled_cache, MutableMapping):
        raise TypeError(f"`compiled_cache` must be a mutable mapping, got {type(compiled_cache)}")

    cache_key = (id(encoder), compile_mode)
    cached = compiled_cache.get(cache_key)
    if cached is None:
        cached = (
            _compile_chunk_callable(first_call, mode=compile_mode, name="first"),
            _compile_chunk_callable(next_call, mode=compile_mode, name="next"),
        )
        compiled_cache[cache_key] = cached
    return cached


def _prepare_encoder_input(vae_model, x):
    if isinstance(vae_model, VideoVAE38_):
        return patchify(x, patch_size=2)
    if isinstance(vae_model, VideoVAE_):
        return x
    raise TypeError(
        "functional VAE encode supports VideoVAE_ and VideoVAE38_ instances, "
        f"got {type(vae_model).__name__}"
    )


def _apply_scale(vae_model, mu, scale):
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
        return (mu - scale[0].view(1, vae_model.z_dim, 1, 1, 1)) * scale[1].view(
            1, vae_model.z_dim, 1, 1, 1
        )
    scale = scale.to(dtype=mu.dtype, device=mu.device)
    return (mu - scale[0]) * scale[1]


def encode_functional(
    vae_model,
    x: torch.Tensor,
    scale,
    *,
    compile_mode: str | None = None,
    compiled_cache: MutableMapping[Any, Any] | None = None,
) -> torch.Tensor:
    """Functional equivalent of VideoVAE_.encode with explicit cache IO."""

    vae_model.clear_cache()
    x = _prepare_encoder_input(vae_model, x)
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    feat_cache = list(vae_model._enc_feat_map)
    first_call, next_call = _get_chunk_callables(vae_model, compile_mode, compiled_cache)

    out = None
    for i in range(iter_):
        if i == 0:
            out_i, feat_cache = first_call(x[:, :, :1, :, :], feat_cache)
        else:
            feat_cache = _clone_cache_tensors(feat_cache)
            out_i, feat_cache = next_call(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache,
            )
            out = torch.cat([out, out_i], 2)
        feat_cache = _clone_cache_tensors(feat_cache)
        if i == 0:
            out = out_i

    mu, _log_var = vae_model.conv1(out).chunk(2, dim=1)
    mu = _apply_scale(vae_model, mu, scale)

    if isinstance(vae_model, VideoVAE38_):
        vae_model.clear_cache()
    else:
        vae_model._enc_feat_map = feat_cache
        vae_model._enc_conv_idx = [0]
    return mu
