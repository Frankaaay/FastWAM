"""Explicit-depth baseline: Wan2.2 encodes the left view and a disparity image as two token streams.

The disparity comes from the same frozen LAS2-H correspondence that StereoTok fuses inside its encoder: the
matching distribution p(d) of the StereoTok matcher is reduced to its expectation, mapped to a fixed log scale,
rendered as a 3-channel image and encoded by the unmodified Wan2.2 VAE. The two latents are joined along width
as in the Wan-LR baseline, so the backbone sees twice the visual tokens and must use the depth stream itself.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .stereotok.checkpoints import load_student

DISPARITY_RANGE = (0.5, 188.0)  # canvas pixels; LAS2-H covers 48 candidates at stride 4


def disparity_image(probabilities):
    """[B,48,T,H/4,W/4] matching distribution -> [B,3,T,H,W] image in [-1,1] (log disparity, fixed scale)."""
    offsets = torch.arange(probabilities.shape[1], device=probabilities.device, dtype=torch.float32) * 4
    support = probabilities.sum(1) > 0
    disparity = (probabilities.float() * offsets.view(1, -1, 1, 1, 1)).sum(1)
    lo, hi = DISPARITY_RANGE
    value = (disparity.clamp(lo, hi).log() - math.log(lo)) / (math.log(hi) - math.log(lo)) * 2 - 1
    value = torch.where(support, value, torch.full_like(value, -1.0))
    b, t, h, w = value.shape
    # Nearest upsampling keeps each value inside its camera panel of the mosaic.
    value = F.interpolate(value.reshape(b * t, 1, h, w), scale_factor=4, mode="nearest").reshape(b, 1, t, 4 * h, 4 * w)
    return value.expand(b, 3, t, 4 * h, 4 * w)


def camera_regions(height, width, device):
    regions = torch.zeros(height // 4, width // 4, dtype=torch.long, device=device)
    if (height, width) == (384, 320):
        regions[64:, :40], regions[64:, 40:] = 1, 2
    elif (height, width) == (224, 448):
        regions[:, 56:] = 1
    return regions


class WanDepthVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True
    width_factor = 2

    def __init__(self, wan_vae, matcher_checkpoint, device):
        super().__init__()
        self.wan = wan_vae.eval().requires_grad_(False)
        path = Path(matcher_checkpoint).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        student = load_student(str(path))
        # Only the frozen correspondence path is kept; no StereoTok encoder or decoder weights are used.
        self.fusion = student.fusion.to(device=device, dtype=torch.float32)
        del student
        self.binding = {"format": "wan2.2-vae-depth-lr", "encode": "per-stream-native-wan",
                        "latent_layout": "cat-width-left-disparity", "matcher_sha256": digest.hexdigest(),
                        "disparity": {"source": "StereoTok LAS2-H matching expectation", "range_px": list(DISPARITY_RANGE),
                                      "mapping": "log, fixed", "upsampling": "nearest"}}
        self.train(False)

    @property
    def model(self):
        return self.wan.model

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def _apply(self, fn, recurse=True):
        # The policy may move its codec to bf16; keep the matcher in FP32 as inside StereoTok.
        result = super()._apply(fn, recurse=recurse)
        self.fusion.float()
        return result

    @torch.no_grad()
    def disparity_video(self, left, right):
        b, _, _, height, width = left.shape
        mask = torch.ones(b, 2, height, width, dtype=torch.bool, device=left.device)
        regions = camera_regions(height, width, left.device)
        images = []
        with torch.autocast(device_type=left.device.type, enabled=False):
            for index in range(0, b, 8):
                probabilities = self.fusion.prepare(left[index:index + 8].float(), right[index:index + 8].float(),
                                                    mask[index:index + 8], self.fusion.max_disparity, camera_regions=regions)
                images.append(disparity_image(probabilities))
        return torch.cat(images)

    @torch.no_grad()
    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, *, right_videos=None):
        if tiled:
            raise ValueError("Wan depth encoding requires tiled=False")
        if right_videos is None:
            raise ValueError("The explicit-depth baseline requires synchronized right eyes")
        dtype = next(self.wan.model.parameters()).dtype
        left = torch.stack(list(videos)).to(device=device)
        right = torch.stack(list(right_videos)).to(device=device)
        if left.shape != right.shape or left.ndim != 5 or left.shape[1] != 3:
            raise ValueError("Expected matching [B,3,T,H,W] stereo inputs")
        if left.shape[-2] % 32 or left.shape[-1] % 32:
            raise ValueError("LAS2 correspondence requires a canvas divisible by 32")
        depth = self.disparity_video(left, right)
        # Same native Wan call and precision as the monocular and Wan-LR branches, once per stream.
        return torch.cat([self.wan.model.encode(left.to(dtype), self.wan.scale),
                          self.wan.model.encode(depth.to(dtype), self.wan.scale)], dim=-1)

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        left, depth = latents.chunk(2, dim=-1)
        return torch.cat([self.wan.decode(left, device, tiled, tile_size, tile_stride),
                          self.wan.decode(depth, device, tiled, tile_size, tile_stride)], dim=-1)
