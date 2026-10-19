"""FastWAM codec ABI for a frozen, complete StereoTok student export."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .stereotok.checkpoints import load_student
from .stereotok.vae2_2 import latent_scale


class StereoTokVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True

    def __init__(self, checkpoint, device, input_mode="stereo"):
        super().__init__()
        if input_mode not in ("stereo", "mono-left"):
            raise ValueError(f"Unsupported StereoTok input mode: {input_mode}")
        self.requires_stereo = input_mode == "stereo"
        path = Path(checkpoint).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.binding = {"format": "stereotok-student-v3", "sha256": digest.hexdigest(),
                        "stereotok_source": "4be840f97694e976714d549a4db2a53705384626",
                        "inference_adapter_source": "783b602d128a43814af32285e60a60f270ec2703",
                        "encode": "default-frozen-native-fp16", "input_mode": input_mode}
        self.model = load_student(str(path)).to(device=device, dtype=torch.float32)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def _apply(self, fn, recurse=True):
        # FastWAM moves its codec with the policy; keep tokenizer FP32 masters.
        if hasattr(self, "model"):
            self.model._frozen_inference = None
        def preserve_precision(tensor):
            moved = fn(tensor)
            if tensor.is_floating_point() and moved.dtype != torch.float32:
                return tensor.to(device=moved.device, dtype=torch.float32)
            return moved
        return super()._apply(preserve_precision, recurse=recurse)

    @torch.no_grad()
    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, *, right_videos=None):
        if tiled:
            raise ValueError("StereoTok mosaic encoding requires tiled=False")
        if not self.requires_stereo:
            if right_videos is not None:
                raise ValueError("Mono-left StereoTok must not receive right eyes")
            left = videos if isinstance(videos, torch.Tensor) else torch.stack(list(videos))
            left = left.to(device=device, dtype=torch.float32)
            if left.ndim != 5 or left.shape[1] != 3:
                raise ValueError("Expected [B,3,T,H,W] mono inputs")
            if any(n % 16 for n in left.shape[-2:]) or left.shape[2] < 1 or (left.shape[2] - 1) % 4:
                raise ValueError("StereoTok requires 16-aligned images and 1+4k frames")
            if not torch.isfinite(left).all() or left.abs().max() > 1.001:
                raise ValueError("StereoTok pixels must be finite and normalized to [-1,1]")
            with torch.autocast(device_type=left.device.type, enabled=False):
                mean = torch.cat([self.model.encode_posterior(chunk)[0] for chunk in left.split(8)])
                shift, scale = latent_scale(left.device)
                return (mean.float() - shift.view(1, 48, 1, 1, 1)) * scale.view(1, 48, 1, 1, 1)
        if right_videos is None:
            raise ValueError("StereoTok requires synchronized right eyes")
        left = torch.stack(list(videos)).to(device=device, dtype=torch.float32)
        right = torch.stack(list(right_videos)).to(device=device, dtype=torch.float32)
        if left.shape != right.shape or left.ndim != 5 or left.shape[1] != 3:
            raise ValueError("Expected matching [B,3,T,H,W] stereo inputs")
        height, width = left.shape[-2:]
        if height % 16 or width % 16 or (left.shape[2] - 1) % 4:
            raise ValueError("StereoTok requires 16-aligned images and 1+4k frames")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("Nonfinite stereo pixels")
        if left.abs().max() > 1.001 or right.abs().max() > 1.001:
            raise ValueError("StereoTok pixels must be normalized to [-1,1]")
        mask = torch.ones(left.shape[0], 2, height, width, dtype=torch.bool, device=device)
        regions = torch.zeros(height // 4, width // 4, dtype=torch.long, device=device)
        if (height, width) == (384, 320):
            regions[64:, :40], regions[64:, 40:] = 1, 2
        elif (height, width) == (224, 448):
            regions[:, 56:] = 1
        with torch.autocast(device_type=left.device.type, enabled=False):
            # Bound the inference workspace without changing the policy batch.
            means = []
            for index in range(0, len(left), 8):
                mean, _ = self.model.encode_posterior(left[index:index + 8], right[index:index + 8],
                    mask[index:index + 8], self.model.fusion.max_disparity, camera_regions=regions)
                means.append(mean)
            mean = torch.cat(means)
            shift, scale = latent_scale(left.device)
            return (mean.float() - shift.view(1, 48, 1, 1, 1)) * scale.view(1, 48, 1, 1, 1)

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        if tiled:
            raise ValueError("StereoTok decoding requires tiled=False")
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            rgb, _ = self.model.decode(latents.to(device=device, dtype=torch.float32), latent_scale(device))
        return rgb
