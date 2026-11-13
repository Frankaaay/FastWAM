"""Naive binocular Wan baseline: the frozen Wan2.2 VAE encodes each eye; latents are joined along width."""
import torch
from torch import nn


class WanBinoVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True
    width_factor = 2

    def __init__(self, wan_vae):
        super().__init__()
        self.wan = wan_vae.eval().requires_grad_(False)
        self.binding = {"format": "wan2.2-vae-bino-lr", "encode": "per-eye-native-wan", "latent_layout": "cat-width-left-right"}

    @property
    def model(self):
        return self.wan.model

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    @torch.no_grad()
    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, *, right_videos=None):
        if tiled:
            raise ValueError("Wan binocular encoding requires tiled=False")
        if right_videos is None:
            raise ValueError("Wan binocular baseline requires synchronized right eyes")
        dtype = next(self.wan.model.parameters()).dtype
        left = torch.stack(list(videos)).to(device=device, dtype=dtype)
        right = torch.stack(list(right_videos)).to(device=device, dtype=dtype)
        if left.shape != right.shape or left.ndim != 5 or left.shape[1] != 3:
            raise ValueError("Expected matching [B,3,T,H,W] stereo inputs")
        # Same native call and precision as the monocular Wan branch, once per eye.
        return torch.cat([self.wan.model.encode(left, self.wan.scale), self.wan.model.encode(right, self.wan.scale)], dim=-1)

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        left, right = latents.chunk(2, dim=-1)
        return torch.cat([self.wan.decode(left, device, tiled, tile_size, tile_stride),
                          self.wan.decode(right, device, tiled, tile_size, tile_stride)], dim=-1)
