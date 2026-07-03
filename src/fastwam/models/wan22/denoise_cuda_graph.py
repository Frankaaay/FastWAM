from __future__ import annotations

from collections.abc import Callable

import torch


class DenoiseStepGraph:
    """Capture and replay one fixed-shape denoise prediction step."""

    def __init__(self, warmup_iters: int = 3):
        if warmup_iters < 0:
            raise ValueError(f"`warmup_iters` must be non-negative, got {warmup_iters}")
        self.warmup_iters = int(warmup_iters)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._latents_static: torch.Tensor | None = None
        self._timestep_static: torch.Tensor | None = None
        self._pred_static: torch.Tensor | None = None
        self._latents_signature: tuple[torch.Size, torch.dtype, torch.device] | None = None
        self._timestep_signature: tuple[torch.Size, torch.dtype, torch.device] | None = None

    @staticmethod
    def _signature(tensor: torch.Tensor) -> tuple[torch.Size, torch.dtype, torch.device]:
        return tensor.shape, tensor.dtype, tensor.device

    @staticmethod
    def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(tensor)}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"`{name}` must be on CUDA for CUDAGraph capture, got {tensor.device}")

    def _check_replay_input(self, name: str, tensor: torch.Tensor, expected) -> None:
        self._require_cuda_tensor(name, tensor)
        got = self._signature(tensor)
        if got != expected:
            raise RuntimeError(f"`{name}` signature changed: got {got}, expected {expected}")

    @torch.no_grad()
    def capture(
        self,
        step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        latents_sample: torch.Tensor,
        timestep_sample: torch.Tensor,
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
        self._latents_static = torch.empty_like(latents_sample).copy_(latents_sample)
        self._timestep_static = torch.empty_like(timestep_sample).copy_(timestep_sample)

        current_stream = torch.cuda.current_stream(device=device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_iters):
                pred = step_fn(self._latents_static, self._timestep_static)
                if self._pred_static is None:
                    self._pred_static = torch.empty_like(pred)
                self._pred_static.copy_(pred)
        current_stream.wait_stream(warmup_stream)

        if self._pred_static is None:
            pred = step_fn(self._latents_static, self._timestep_static)
            self._pred_static = torch.empty_like(pred)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            pred = step_fn(self._latents_static, self._timestep_static)
            self._pred_static.copy_(pred)
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
