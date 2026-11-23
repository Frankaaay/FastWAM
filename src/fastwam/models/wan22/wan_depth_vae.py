"""Explicit-depth baseline: Wan2.2 encodes the left view and a metric depth image as two token streams.

Depth comes from the complete LAS2-H stereo matcher (official release, iterative refinement, full-resolution output)
run separately on each camera of the stereo mosaic, converted to metric depth with the calibrated f_x * B of that
camera, mapped to a fixed log range (as in SA-WAM), rendered as a 3-channel image and encoded by the unmodified
Wan2.2 VAE. The two latents are joined along width as in the Wan-LR baseline, so the backbone sees twice the visual
tokens and must use the depth stream itself.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

LAS2_H_SHA256 = "758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4"
# RoboTwin cameras: native 320x240, f_x = 358.642 px; head 6 cm, wrist 2 cm (read from the episode HDF5 calibration).
NATIVE_FX = 358.642
CAMERA_BASELINES_M = (0.06, 0.02, 0.02)  # head, left wrist, right wrist
DEPTH_RANGE_M = (0.02, 5.0)
MIN_DISPARITY_PX = 0.5
LAS2_ITERS = 4  # LAS2-H release default, as used for the StereoTok LAS2 teacher
PANEL = (256, 320)  # every camera is matched at native width; the head slot already has it
MOSAIC = (384, 320)


def load_las2(path):
    """Official LAS2-H (upstream 8c97bd4) with the released weights; frozen, eval mode."""
    from .stereotok.las2.liteanystereov2_H import LiteAnyStereoH
    model = LiteAnyStereoH(fnet_pretrained=False, max_disp=192)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if payload and all(k.startswith("module.") for k in payload):
        payload = {k[len("module."):]: v for k, v in payload.items()}
    model.load_state_dict(payload, strict=True)
    return model.eval().requires_grad_(False)


def mosaic_panels(frames):
    """[N,3,384,320] mosaic -> [3N,3,256,320]: head, left wrist, right wrist, each at native camera width."""
    if tuple(frames.shape[-2:]) != MOSAIC:
        raise ValueError(f"Expected the {MOSAIC} RoboTwin stereo mosaic, got {tuple(frames.shape[-2:])}")
    head = frames[..., :256, :]
    up = lambda x: F.interpolate(x, size=PANEL, mode="bilinear", align_corners=False)
    return torch.cat([head, up(frames[..., 256:, :160]), up(frames[..., 256:, 160:])])


def depth_value(disparity, baseline):
    """Disparity at native width (px) -> metric depth, fixed log mapping to [-1, 1]; no match -> far."""
    lo, hi = DEPTH_RANGE_M
    depth = NATIVE_FX * baseline / disparity.clamp_min(MIN_DISPARITY_PX)
    depth = torch.where(disparity >= MIN_DISPARITY_PX, depth, torch.full_like(depth, hi))
    return (depth.clamp(lo, hi).log() - math.log(lo)) / (math.log(hi) - math.log(lo)) * 2 - 1


def depth_mosaic(disparity):
    """[3N,1,256,320] disparity of the three panels -> [N,3,384,320] depth image in the mosaic layout."""
    n = disparity.shape[0] // 3
    head, left, right = (depth_value(d, b) for d, b in zip(disparity.split(n), CAMERA_BASELINES_M))
    down = lambda x: F.interpolate(x, size=(128, 160), mode="area")
    image = torch.cat([head, torch.cat([down(left), down(right)], dim=-1)], dim=-2)
    return image.expand(n, 3, *MOSAIC)


class WanDepthVAE(nn.Module):
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True
    width_factor = 2

    def __init__(self, wan_vae, matcher_checkpoint, device, microbatch=48):
        super().__init__()
        self.wan = wan_vae.eval().requires_grad_(False)
        path = Path(matcher_checkpoint).expanduser().resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != LAS2_H_SHA256:
            raise ValueError("Unexpected LAS2-H checkpoint")
        self.matcher = load_las2(str(path)).to(device=device, dtype=torch.float32)
        self.microbatch = microbatch
        self.binding = {"format": "wan2.2-vae-depthmap-lr", "encode": "per-stream-native-wan",
                        "latent_layout": "cat-width-left-depth", "matcher_sha256": digest.hexdigest(),
                        "depth": {"source": "LAS2-H release, iterative refinement, per camera at native width",
                                  "iters": LAS2_ITERS, "fx": NATIVE_FX, "baselines_m": list(CAMERA_BASELINES_M),
                                  "range_m": list(DEPTH_RANGE_M), "mapping": "log, fixed"}}
        self.train(False)

    @property
    def model(self):
        return self.wan.model

    def train(self, mode=True):
        super().train(False)
        self.requires_grad_(False)
        return self

    def _apply(self, fn, recurse=True):
        # The policy may move its codec to bf16; keep the matcher weights in FP32 (inference runs under fp16 autocast).
        result = super()._apply(fn, recurse=recurse)
        self.matcher.float()
        return result

    @torch.no_grad()
    def disparity(self, left, right):
        """[N,3,256,320] pairs in [-1,1] -> [N,1,256,320] LAS2-H disparity in pixels."""
        outputs = []
        for i in range(0, left.shape[0], self.microbatch):
            l = (left[i:i + self.microbatch].float() + 1) * 127.5
            r = (right[i:i + self.microbatch].float() + 1) * 127.5
            with torch.autocast(device_type=left.device.type, dtype=torch.float16, enabled=left.device.type == "cuda"):
                d = self.matcher(l, r, iters=LAS2_ITERS, test_mode=True)
            outputs.append((d if d.ndim == 4 else d.unsqueeze(1)).float())
        return torch.cat(outputs)

    @torch.no_grad()
    def depth_video(self, left, right):
        b, _, t, h, w = left.shape
        flat = lambda x: x.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        disparity = self.disparity(mosaic_panels(flat(left)), mosaic_panels(flat(right)))
        return depth_mosaic(disparity).reshape(b, t, 3, h, w).permute(0, 2, 1, 3, 4).contiguous()

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
        depth = self.depth_video(left, right)
        # Same native Wan call and precision as the monocular and Wan-LR branches, once per stream.
        return torch.cat([self.wan.model.encode(left.to(dtype), self.wan.scale),
                          self.wan.model.encode(depth.to(dtype), self.wan.scale)], dim=-1)

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        left, depth = latents.chunk(2, dim=-1)
        return torch.cat([self.wan.decode(left, device, tiled, tile_size, tile_stride),
                          self.wan.decode(depth, device, tiled, tile_size, tile_stride)], dim=-1)
