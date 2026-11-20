"""GeoSem-style baseline (after StereoVLA): frozen geometric and semantic features fused by a trainable projector.

StereoVLA concatenates a frozen stereo cost-volume feature (FoundationStereo, pooled to the patch grid) with frozen
semantic patch features of the left view (SigLIP + DINOv2) along channels and maps them with an MLP projector trained
with the policy. Adapted to a WAM with the same budget as StereoTok and Wan2.2-VAE:
  * geometry: the matching distribution p(d) of the frozen LAS2-H matcher of the same StereoTok checkpoint (48 disparity
    hypotheses), averaged over each 16x16 latent cell;
  * semantics: frozen DINOv3 ViT-B/16 patch features of the left view (768-d, one per 16x16 cell), per camera panel;
  * appearance: the frozen Wan2.2 latent of the left view.
The three are concatenated per latent cell and an MLP with a zero-initialized output adds the result to the Wan latent
of the observation frame. Token count equals the monocular Wan baseline; future-frame targets stay Wan latents.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .stereotok.checkpoints import load_student
from .wan_depth_vae import camera_regions

DINO_SOURCE_SHA = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
GEO_DIM, SEM_DIM = 48, 768


def panels(height, width):
    """Pixel boxes (top, bottom, left, right) of the camera panels of the FastWAM RoboTwin mosaics."""
    if (height, width) == (384, 320):
        return [(0, 256, 0, 320), (256, 384, 0, 160), (256, 384, 160, 320)]
    if (height, width) == (224, 448):
        return [(0, 224, 0, 224), (0, 224, 224, 448)]
    return [(0, height, 0, width)]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GeoSemProjector(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(48 + GEO_DIM + SEM_DIM)
        self.mlp = nn.Sequential(nn.Linear(48 + GEO_DIM + SEM_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 48)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, latent, features):
        """latent [B,48,T,h,w], features [B,GEO+SEM,T,h,w] -> latent + projector([latent; features])."""
        if latent.shape[0] != features.shape[0] or latent.shape[2:] != features.shape[2:]:
            raise ValueError("GeoSem features must share the latent grid")
        dtype = self.out.weight.dtype
        x = torch.cat([latent, features.to(latent.dtype)], 1).permute(0, 2, 3, 4, 1).to(dtype)
        delta = self.out(self.mlp(self.norm(x))).permute(0, 4, 1, 2, 3)
        return latent + delta.to(latent.dtype)


class WanGeoSemVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True
    stereo_fusion_input = True

    def __init__(self, wan_vae, matcher_checkpoint, dino_repo, dino_checkpoint, device):
        super().__init__()
        self.wan = wan_vae.eval().requires_grad_(False)
        matcher_sha = _sha256(matcher_checkpoint)
        if _sha256(dino_checkpoint) != DINO_SHA256:
            raise ValueError("Unexpected DINOv3 checkpoint")
        import subprocess
        head = subprocess.check_output(["git", "-C", str(dino_repo), "rev-parse", "HEAD"], text=True).strip()
        if head != DINO_SOURCE_SHA:
            raise ValueError("Unexpected DINOv3 source")
        student = load_student(str(Path(matcher_checkpoint).expanduser().resolve(strict=True)))
        self.fusion = student.fusion.to(device=device, dtype=torch.float32)
        del student
        with torch.random.fork_rng(devices=[]):
            dino = torch.hub.load(str(dino_repo), "dinov3_vitb16", source="local", pretrained=False)
        dino.load_state_dict(torch.load(dino_checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.dino = dino.to(device=device, dtype=torch.float32)
        self.binding = {"format": "wan2.2-vae-geosem", "encode": "per-eye-native-wan", "matcher_sha256": matcher_sha,
                        "semantic": {"model": "dinov3_vitb16", "sha256": DINO_SHA256, "source": DINO_SOURCE_SHA},
                        "latent_layout": "left; observation frame fused with [LAS2 p(d) pooled to 16x16 | DINOv3 left patches]"}
        self.train(False)

    @property
    def model(self):
        return self.wan.model

    @property
    def scale(self):
        return self.wan.scale

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.fusion.float()
        self.dino.float()
        return result

    def _stack(self, videos, device, dtype):
        return (videos if isinstance(videos, torch.Tensor) else torch.stack(list(videos))).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, *, right_videos=None):
        if tiled:
            raise ValueError("Wan GeoSem encoding requires tiled=False")
        dtype = next(self.wan.model.parameters()).dtype
        return self.wan.model.encode(self._stack(videos, device, dtype), self.wan.scale)

    @torch.no_grad()
    def features(self, left, right):
        """Observation-frame features [B, GEO+SEM, 1, H/16, W/16] from fp32 images in [-1,1] ([B,3,1,H,W])."""
        b, _, _, height, width = left.shape
        mask = torch.ones(b, 2, height, width, dtype=torch.bool, device=left.device)
        with torch.autocast(device_type=left.device.type, enabled=False):
            probabilities = self.fusion.prepare(left, right, mask, self.fusion.max_disparity,
                                                camera_regions=camera_regions(height, width, left.device))
            geo = F.avg_pool3d(probabilities.float(), (1, 4, 4))                     # [B,48,1,H/16,W/16]
            sem = torch.zeros(b, SEM_DIM, 1, height // 16, width // 16, device=left.device)
            mean = torch.tensor([.485, .456, .406], device=left.device).view(1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225], device=left.device).view(1, 3, 1, 1)
            image = (left[:, :, 0].float().clamp(-1, 1) + 1) / 2
            for top, bottom, x0, x1 in panels(height, width):
                crop = (image[:, :, top:bottom, x0:x1] - mean) / std
                tokens = self.dino.forward_features(crop)["x_norm_patchtokens"].float()   # [B,N,768]
                h, w = (bottom - top) // 16, (x1 - x0) // 16
                sem[:, :, 0, top // 16:bottom // 16, x0 // 16:x1 // 16] = tokens.transpose(1, 2).reshape(b, SEM_DIM, h, w)
        return torch.cat([geo, sem], 1)

    @torch.no_grad()
    def encode_pair(self, videos, device, *, right_videos):
        if right_videos is None:
            raise ValueError("The GeoSem baseline requires synchronized right eyes")
        dtype = next(self.wan.model.parameters()).dtype
        left = self._stack(videos, device, torch.float32)
        right = self._stack(right_videos, device, torch.float32)
        if left.shape != right.shape or left.ndim != 5 or left.shape[1] != 3:
            raise ValueError("Expected matching [B,3,T,H,W] stereo inputs")
        if left.shape[-2] % 32 or left.shape[-1] % 32:
            raise ValueError("LAS2 correspondence requires a canvas divisible by 32")
        latents = self.wan.model.encode(left.to(dtype), self.wan.scale)
        return latents, self.features(left[:, :, :1], right[:, :, :1])

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        return self.wan.decode(latents, device, tiled, tile_size, tile_stride)
