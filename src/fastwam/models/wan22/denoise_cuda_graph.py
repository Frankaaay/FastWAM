from __future__ import annotations

import os
from collections.abc import Callable

import torch

TensorSignature = tuple[tuple[int, ...], str, str]
ConditionSignature = tuple[tuple[str, object], ...]
GraphSignature = tuple[tuple[str, object], ...]
VideoKVCache = list[dict[str, torch.Tensor]]
DenoiseStepFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, VideoKVCache, torch.Tensor, int],
    torch.Tensor,
]


class DenoiseStepGraph:
    """Capture and replay one fixed-shape denoise prediction step."""

    def __init__(self, warmup_iters: int = 5):
        if warmup_iters < 0:
            raise ValueError(f"`warmup_iters` must be non-negative, got {warmup_iters}")
        self.warmup_iters = int(warmup_iters)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool: object | None = None
        self._latents_static: torch.Tensor | None = None
        self._timestep_static: torch.Tensor | None = None
        self._pred_static: torch.Tensor | None = None
        self._context_static: torch.Tensor | None = None
        self._context_mask_static: torch.Tensor | None = None
        self._attention_mask_static: torch.Tensor | None = None
        self._video_kv_cache_static: VideoKVCache | None = None
        self._video_seq_len_static: int | None = None
        self._latents_signature: TensorSignature | None = None
        self._timestep_signature: TensorSignature | None = None
        self._condition_signature: ConditionSignature | None = None
        self._graph_signature: GraphSignature | None = None

    @staticmethod
    def _signature(tensor: torch.Tensor) -> TensorSignature:
        return tuple(tensor.shape), str(tensor.dtype), str(tensor.device)

    @staticmethod
    def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(tensor)}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"`{name}` must be on CUDA for CUDAGraph capture, got {tensor.device}")

    @classmethod
    def _video_kv_cache_signature(
        cls,
        video_kv_cache: VideoKVCache,
    ) -> tuple[tuple[tuple[str, TensorSignature], ...], ...]:
        if not isinstance(video_kv_cache, list):
            raise TypeError(f"`video_kv_cache` must be a list, got {type(video_kv_cache)}")
        layers: list[tuple[tuple[str, TensorSignature], ...]] = []
        for layer_idx, layer_cache in enumerate(video_kv_cache):
            if not isinstance(layer_cache, dict):
                raise TypeError(
                    f"`video_kv_cache[{layer_idx}]` must be a dict, got {type(layer_cache)}"
                )
            entries: list[tuple[str, TensorSignature]] = []
            for key in sorted(layer_cache):
                if not isinstance(key, str):
                    raise TypeError(f"`video_kv_cache[{layer_idx}]` key must be str, got {type(key)}")
                tensor = layer_cache[key]
                cls._require_cuda_tensor(f"video_kv_cache[{layer_idx}][{key!r}]", tensor)
                entries.append((key, cls._signature(tensor)))
            layers.append(tuple(entries))
        return tuple(layers)

    @classmethod
    def make_condition_signature(
        cls,
        *,
        video_kv_cache: VideoKVCache,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> ConditionSignature:
        cls._require_cuda_tensor("context", context)
        cls._require_cuda_tensor("context_mask", context_mask)
        cls._require_cuda_tensor("attention_mask", attention_mask)
        return (
            ("context", cls._signature(context)),
            ("context_mask", cls._signature(context_mask)),
            ("attention_mask", cls._signature(attention_mask)),
            ("video_seq_len", int(video_seq_len)),
            ("video_kv_cache", cls._video_kv_cache_signature(video_kv_cache)),
        )

    @classmethod
    def make_signature(
        cls,
        *,
        latents_sample: torch.Tensor,
        timestep_sample: torch.Tensor,
        video_kv_cache: VideoKVCache,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> GraphSignature:
        cls._require_cuda_tensor("latents_sample", latents_sample)
        cls._require_cuda_tensor("timestep_sample", timestep_sample)
        return (
            ("latents", cls._signature(latents_sample)),
            ("timestep", cls._signature(timestep_sample)),
            *cls.make_condition_signature(
                video_kv_cache=video_kv_cache,
                context=context,
                context_mask=context_mask,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            ),
        )

    @staticmethod
    def _debug_enabled() -> bool:
        value = os.environ.get("FASTWAM_DENOISE_GRAPH_DEBUG", "").strip().lower()
        return value not in {"", "0", "false", "no", "off"}

    @staticmethod
    def _make_graph_pool() -> object | None:
        graph_pool_handle = getattr(torch.cuda, "graph_pool_handle", None)
        if graph_pool_handle is None:
            graphs_mod = getattr(torch.cuda, "graphs", None)
            graph_pool_handle = getattr(graphs_mod, "graph_pool_handle", None)
        if graph_pool_handle is None:
            return None
        return graph_pool_handle()

    @staticmethod
    def _cuda_graph_context(graph: torch.cuda.CUDAGraph, pool: object | None):
        kwargs_candidates: list[dict[str, object]] = []
        if pool is not None:
            kwargs_candidates.append({"pool": pool, "capture_error_mode": "thread_local"})
        kwargs_candidates.append({"capture_error_mode": "thread_local"})
        if pool is not None:
            kwargs_candidates.append({"pool": pool})
        kwargs_candidates.append({})

        last_type_error: TypeError | None = None
        for kwargs in kwargs_candidates:
            try:
                return torch.cuda.graph(graph, **kwargs)
            except TypeError as exc:
                last_type_error = exc
        if last_type_error is not None:
            raise last_type_error
        return torch.cuda.graph(graph)

    @staticmethod
    def _sync_after_capture_failure(device: torch.device) -> Exception | None:
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:  # pragma: no cover - requires a poisoned CUDA context.
            return exc
        return None

    def _format_capture_failure(self, exc: Exception, sync_exc: Exception | None) -> str:
        message = f"CUDA graph capture failed: original={exc!r}"
        if sync_exc is not None:
            message += f"; synchronize_after_failure={sync_exc!r}"
        if self._debug_enabled():
            launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "")
            if launch_blocking != "1":
                message += (
                    "; set CUDA_LAUNCH_BLOCKING=1 before launching the process for a "
                    "per-op CUDA stack"
                )
        return message

    def _check_replay_input(self, name: str, tensor: torch.Tensor, expected) -> None:
        self._require_cuda_tensor(name, tensor)
        got = self._signature(tensor)
        if got != expected:
            raise RuntimeError(f"`{name}` signature changed: got {got}, expected {expected}")

    @staticmethod
    def _clone_static_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor).copy_(tensor)

    @classmethod
    def _clone_static_video_kv_cache(cls, video_kv_cache: VideoKVCache) -> VideoKVCache:
        return [
            {key: cls._clone_static_tensor(layer_cache[key]) for key in sorted(layer_cache)}
            for layer_cache in video_kv_cache
        ]

    def _require_captured_conditions(self) -> tuple[torch.Tensor, torch.Tensor, VideoKVCache, torch.Tensor, int]:
        if (
            self._context_static is None
            or self._context_mask_static is None
            or self._video_kv_cache_static is None
            or self._attention_mask_static is None
            or self._video_seq_len_static is None
            or self._condition_signature is None
        ):
            raise RuntimeError("DenoiseStepGraph condition buffers have not been captured.")
        return (
            self._context_static,
            self._context_mask_static,
            self._video_kv_cache_static,
            self._attention_mask_static,
            self._video_seq_len_static,
        )

    @property
    def signature(self) -> GraphSignature | None:
        return self._graph_signature

    @torch.no_grad()
    def update_conditions(
        self,
        *,
        video_kv_cache: VideoKVCache,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> "DenoiseStepGraph":
        (
            context_static,
            context_mask_static,
            video_kv_cache_static,
            attention_mask_static,
            _video_seq_len_static,
        ) = self._require_captured_conditions()

        got = self.make_condition_signature(
            video_kv_cache=video_kv_cache,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        if got != self._condition_signature:
            raise RuntimeError(
                "denoise CUDA graph condition signature changed: "
                f"got {got}, expected {self._condition_signature}"
            )

        context_static.copy_(context)
        context_mask_static.copy_(context_mask)
        attention_mask_static.copy_(attention_mask)
        for layer_idx, layer_static in enumerate(video_kv_cache_static):
            layer = video_kv_cache[layer_idx]
            for key, tensor_static in layer_static.items():
                tensor_static.copy_(layer[key])
        return self

    @torch.no_grad()
    def capture(
        self,
        step_fn: DenoiseStepFn,
        latents_sample: torch.Tensor,
        timestep_sample: torch.Tensor,
        *,
        video_kv_cache: VideoKVCache,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> "DenoiseStepGraph":
        self._require_cuda_tensor("latents_sample", latents_sample)
        self._require_cuda_tensor("timestep_sample", timestep_sample)
        if latents_sample.device != timestep_sample.device:
            raise RuntimeError(
                "`latents_sample` and `timestep_sample` must be on the same CUDA device, "
                f"got {latents_sample.device} and {timestep_sample.device}"
            )

        device = latents_sample.device
        self._latents_signature = self._signature(latents_sample)
        self._timestep_signature = self._signature(timestep_sample)
        self._condition_signature = self.make_condition_signature(
            video_kv_cache=video_kv_cache,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        self._graph_signature = (
            ("latents", self._latents_signature),
            ("timestep", self._timestep_signature),
            *self._condition_signature,
        )
        self._latents_static = torch.empty_like(latents_sample).copy_(latents_sample)
        self._timestep_static = torch.empty_like(timestep_sample).copy_(timestep_sample)
        self._context_static = self._clone_static_tensor(context)
        self._context_mask_static = self._clone_static_tensor(context_mask)
        self._attention_mask_static = self._clone_static_tensor(attention_mask)
        self._video_kv_cache_static = self._clone_static_video_kv_cache(video_kv_cache)
        self._video_seq_len_static = int(video_seq_len)
        (
            context_static,
            context_mask_static,
            video_kv_cache_static,
            attention_mask_static,
            video_seq_len_static,
        ) = self._require_captured_conditions()

        current_stream = torch.cuda.current_stream(device=device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_iters):
                pred = step_fn(
                    self._latents_static,
                    self._timestep_static,
                    context_static,
                    context_mask_static,
                    video_kv_cache_static,
                    attention_mask_static,
                    video_seq_len_static,
                )
                if self._pred_static is None:
                    self._pred_static = torch.empty_like(pred)
                self._pred_static.copy_(pred)
        current_stream.wait_stream(warmup_stream)

        if self._pred_static is None:
            pred = step_fn(
                self._latents_static,
                self._timestep_static,
                context_static,
                context_mask_static,
                video_kv_cache_static,
                attention_mask_static,
                video_seq_len_static,
            )
            self._pred_static = torch.empty_like(pred)
            self._pred_static.copy_(pred)
        torch.cuda.synchronize(device)

        self._graph_pool = self._make_graph_pool()
        self._graph = torch.cuda.CUDAGraph()
        try:
            with self._cuda_graph_context(self._graph, self._graph_pool):
                pred = step_fn(
                    self._latents_static,
                    self._timestep_static,
                    context_static,
                    context_mask_static,
                    video_kv_cache_static,
                    attention_mask_static,
                    video_seq_len_static,
                )
                self._pred_static.copy_(pred)
        except Exception as exc:
            sync_exc = self._sync_after_capture_failure(device)
            if sync_exc is None and not self._debug_enabled():
                raise
            raise RuntimeError(self._format_capture_failure(exc, sync_exc)) from exc
        return self

    @torch.no_grad()
    def replay(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if (
            self._graph is None
            or self._latents_static is None
            or self._timestep_static is None
            or self._pred_static is None
            or self._latents_signature is None
            or self._timestep_signature is None
        ):
            raise RuntimeError("DenoiseStepGraph has not been captured.")
        self._check_replay_input("latents", latents, self._latents_signature)
        self._check_replay_input("timestep", timestep, self._timestep_signature)

        self._latents_static.copy_(latents)
        self._timestep_static.copy_(timestep)
        self._graph.replay()
        return self._pred_static.clone()
