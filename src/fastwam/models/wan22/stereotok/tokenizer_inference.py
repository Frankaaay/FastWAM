"""Frozen downstream encoding: peilin/profiling@f125820 native-fp16-conv.

The inference view owns its packed convolution weights and nonpersistent FP16
operands. The source model, training forwards and checkpoint keys stay intact.
"""

import copy
import functools
from collections import OrderedDict

import torch

from .vae2_2 import RMS_norm
from .inference_kernels import (install_implicit_spatial_padding,
                               install_weighted_match, norm_pointwise)


class MatcherGraph:
    """Fresh input copies on every replay; bounded full/tail shape cache."""

    def __init__(self, forward):
        self.forward = forward
        self.graphs = OrderedDict()

    @torch.compiler.disable(recursive=False)
    @torch.inference_mode(False)
    @torch.no_grad()
    def __call__(self, left, right):
        signature = (tuple(left.shape), left.stride(), left.dtype, left.device,
                     tuple(right.shape), right.stride(), right.dtype, right.device)
        if signature not in self.graphs:
            if len(self.graphs) == 4:
                self.graphs.popitem(last=False)
            static_left, static_right = left.clone(), right.clone()
            stream = torch.cuda.Stream(device=left.device)
            stream.wait_stream(torch.cuda.current_stream(left.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward(static_left, static_right)
            torch.cuda.current_stream(left.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward(static_left, static_right)
            self.graphs[signature] = (graph, static_left, static_right, output)
        self.graphs.move_to_end(signature)
        graph, static_left, static_right, output = self.graphs[signature]
        static_left.copy_(left)
        static_right.copy_(right)
        graph.replay()
        return output.clone()


class FrozenTokenizerInference:
    @staticmethod
    def signature(model):
        return tuple((id(p), p._version, p.device, p.dtype, p.data_ptr())
                     for block in (model.encoder, model.conv1, model.fusion)
                     for p in block.parameters())

    def __init__(self, source):
        if source.training or any(p.requires_grad for p in source.parameters()):
            raise ValueError("Optimized encoding requires a frozen eval tokenizer")
        blocks = (source.encoder, source.conv1, source.fusion)
        if source.fusion.mode != "early" or any(
                p.dtype != torch.float32 for block in blocks for p in block.parameters()):
            raise ValueError("The accepted inference recipe requires early LAS and FP32 masters")
        self.source_signature = self.signature(source)
        # New Parameter objects share frozen storage until Conv3d packing allocates
        # private storage. Changing their layout/forward never changes the source.
        memo = {id(p): torch.nn.Parameter(p.detach(), requires_grad=False)
                for block in blocks for p in block.parameters()}
        memo.update({id(b): b.detach() for block in blocks for b in block.buffers()})
        self.model = copy.copy(source)
        self.model._modules = dict(source._modules)
        for name in ("encoder", "conv1", "fusion"):
            setattr(self.model, name, copy.deepcopy(getattr(source, name), memo))
        for block in (self.model.encoder, self.model.conv1):
            torch.nn.utils.convert_conv3d_weight_memory_format(block, torch.channels_last_3d)
            install_implicit_spatial_padding(block, half_spatial_convs=True)
        for layer in self.model.encoder.modules():
            if isinstance(layer, RMS_norm):
                if not layer.channel_first or not isinstance(layer.bias, float) or layer.bias != 0.:
                    raise ValueError("Unsupported RMS norm variant")
                layer.forward = functools.partial(norm_pointwise, module=layer)
        fusion = self.model.fusion
        fusion.microbatch = 34
        install_weighted_match(fusion)
        fusion.matcher.forward = MatcherGraph(torch.compile(
            fusion.matcher.forward, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False, "layout_optimization": False,
                     "emulate_precision_casts": True}))
        self.model.encoder.forward = torch.compile(
            self.model.encoder.forward, fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False, "layout_optimization": False,
                     "max_autotune": False})

    @torch.compiler.disable(recursive=False)
    @torch.inference_mode(False)
    @torch.no_grad()
    def encode(self, x, right, content_mask, disparity, match_probabilities, frame_chunk, camera_regions=None):
        # Match the accepted TF32/autotune boundary locally. CUDA Graph capture
        # uses ordinary tensors even when a downstream caller uses inference_mode.
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        benchmark_limit = torch.backends.cudnn.benchmark_limit
        config = torch._dynamo.config
        keys = ("recompile_limit", "accumulated_recompile_limit") if hasattr(
            config, "recompile_limit") else ("cache_size_limit", "accumulated_cache_size_limit")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark_limit = 0
            with config.patch(**dict(zip(keys, (64, 256)))), torch.backends.cudnn.flags(
                    benchmark=True, deterministic=False, allow_tf32=True), \
                    torch.autocast("cuda", enabled=False):
                return self.model._encode_posterior(
                    x.float(), None if right is None else right.float(), content_mask,
                    disparity, match_probabilities, frame_chunk, camera_regions)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.benchmark_limit = benchmark_limit
