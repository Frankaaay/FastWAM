"""Frozen student inference; parameter names match StereoTok student-v3 exports."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ENCODER_GEOMETRY_RECIPE = dict(version=1, model="LAS2-H", max_disp=192,
    feature="aggregated_cost", candidates=48, descriptor_dim=32, temperature=1.,
    projection="left_matched_geometry_mlp", compression="wan_downsamples_1_2")


class FrozenLAS2Matcher(nn.Module):
    """Official LAS2-H feature/cost trunk; no iterative disparity decoder."""
    def __init__(self, spec, initialize):
        super().__init__()
        from .las2.fnet import FeatureNetFasterNet
        from .las2.aggregation_fasternet import Aggregation
        from .las2.submodule import build_gwc_volume_fast
        if initialize:
            raise ValueError("StereoTok requires a complete student export including LAS weights")
        self.fnet = FeatureNetFasterNet(pretrained=False)
        self.cost_agg = Aggregation(in_channels=48, left_att=True, blocks=[4, 8, 16],
                                    expanse_ratio=4, backbone_channels=self.fnet.feature_channels)
        self.correlation = build_gwc_volume_fast
        self.groups = 8
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self._runtime = None
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, left, right):
        if self._runtime is not None:
            return self._runtime(left, right)
        return self._compute(left, right)

    def _compute(self, left, right):
        # Use the official arithmetic/order, including its [0,255] input domain.
        left = (((left.float() + 1) * 127.5 / 255.0 - self.image_mean) / self.image_std).contiguous()
        right = (((right.float() + 1) * 127.5 / 255.0 - self.image_mean) / self.image_std).contiguous()
        features_left, features_right = self.fnet(left), self.fnet(right)
        volume = self.correlation(features_left[0], features_right[0], 48, self.groups)
        return self.cost_agg(volume.mean(dim=1), features_left)


class LAS2GuidedFusion(nn.Module):
    """LAS supplies correspondence probabilities; Wan supplies right-eye values."""
    def __init__(self, encoder, spec, max_disparity, initialize=True):
        super().__init__()
        if spec["mode"] not in ("early", "late") or not 0 <= max_disparity <= 188:
            raise ValueError("LAS fusion requires early/late and disparity in [0,188]")
        if spec.get("recipe", ENCODER_GEOMETRY_RECIPE) != ENCODER_GEOMETRY_RECIPE:
            raise ValueError("Unsupported encoder geometry recipe")
        self.mode, self.max_disparity = spec["mode"], max_disparity
        self.microbatch = spec["microbatch"]
        self.optimized_match = False
        channels = encoder.dim
        self.matcher = FrozenLAS2Matcher(spec, initialize)
        self.value = nn.Linear(channels, channels, bias=False)
        self.descriptor = nn.Linear(48, 32, bias=False)
        self.aggregate = nn.Sequential(nn.Linear(2 * channels + 32, channels), nn.GELU(),
                                       nn.Linear(channels, channels))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.downsamples = nn.ModuleList(copy.deepcopy(encoder.downsamples[i]) for i in (1, 2)) if self.mode == "late" else nn.ModuleList()

    @torch.no_grad()
    def prepare(self, left, right, content_mask, disparity, camera_regions=None):
        if left.shape != right.shape or left.ndim != 5:
            raise ValueError("LAS fusion requires matching [B,3,T,H,W] eyes")
        b, _, t, h, w = left.shape
        if h % 32 or w % 32:
            raise ValueError("LAS encoder requires a canvas divisible by 32")
        if content_mask.dtype != torch.bool or content_mask.shape not in ((b, 2, h, w), (b, 2, t, h, w)):
            raise ValueError("LAS fusion requires [B,2,H,W] or [B,2,T,H,W] content masks")
        disparity = torch.as_tensor(disparity, device=left.device).expand(b)
        if not torch.isfinite(disparity).all() or (disparity < 0).any() or (disparity > self.max_disparity).any():
            raise ValueError("Disparity exceeds configured LAS support")
        # Flatten independent pairs only; eval-mode LAS never mixes frames/views.
        l = left.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        r = right.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
        outputs = []
        for start in range(0, b * t, self.microbatch):
            with torch.autocast(device_type=left.device.type, dtype=torch.float16, enabled=left.is_cuda):
                outputs.append(self.matcher(l[start:start + self.microbatch], r[start:start + self.microbatch]).float())
        scores = torch.cat(outputs).reshape(b, t, 48, h // 4, w // 4).permute(0, 2, 1, 3, 4)
        return self.probabilities_from_cost(scores, content_mask, disparity, camera_regions)

    @torch.no_grad()
    def probabilities_from_cost(self, scores, content_mask, disparity, camera_regions=None):
        b, _, _, _, width = scores.shape
        disparity = torch.as_tensor(disparity, device=scores.device).expand(b)
        if content_mask.ndim == 4:
            content_mask = content_mask[:, :, None].expand(-1, -1, scores.shape[2], -1, -1)
        mask = F.avg_pool2d(content_mask.flatten(0, 2).unsqueeze(1).float(), 4, 4) == 1
        mask = mask.reshape(b, 2, scores.shape[2], *scores.shape[-2:])
        offsets = torch.arange(48, device=scores.device)
        index = torch.arange(width, device=scores.device)[None, :] - offsets[:, None]
        right_mask = mask[:, 1, :, :, index.clamp_min(0)].permute(0, 3, 1, 2, 4)
        valid = (mask[:, :1] & right_mask & (index >= 0)[None, :, None, None, :]
                 & (offsets[None, :] <= torch.ceil(disparity[:, None] / 4))[:, :, None, None, None])
        if camera_regions is not None:
            if camera_regions.shape != scores.shape[-2:]:
                raise ValueError("Camera regions must match the LAS quarter-resolution grid")
            # Horizontal candidates may not cross the two wrist-camera tiles.
            same_camera = camera_regions[None] == camera_regions[:, index.clamp_min(0)].permute(1, 0, 2)
            valid = valid & same_camera[None, :, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        scores = torch.where(valid.any(1, keepdim=True), scores, torch.zeros_like(scores))
        return scores.softmax(1).masked_fill(~valid, 0)

    def forward(self, left, right, probabilities, feat_cache=None):
        if left.shape != right.shape or probabilities.shape != (left.shape[0], 48, *left.shape[2:]):
            raise ValueError("LAS and Wan quarter-resolution grids must coincide")
        values = self.value(right.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        weights = probabilities.to(values.dtype)
        disparities = min(48, left.shape[-1], math.ceil(self.max_disparity / 4) + 1)
        if self.optimized_match:
            from .inference_kernels import weighted_match
            matched = weighted_match(values, weights, disparities)
        else:
            matched = torch.zeros_like(values)
            for d in range(disparities):
                contribution = values[..., :left.shape[-1] - d] * weights[:, d:d + 1, ..., d:]
                matched = matched + F.pad(contribution, (d, 0))
        geometry = self.descriptor(weights.permute(0, 2, 3, 4, 1))
        feature = torch.cat((left.permute(0, 2, 3, 4, 1), matched.permute(0, 2, 3, 4, 1), geometry), -1)
        support = probabilities.sum(1, keepdim=True) > 0
        delta = self.aggregate(feature).permute(0, 4, 1, 2, 3) * support
        index = [0]
        for layer in self.downsamples:
            delta = layer(delta, feat_cache, index) if feat_cache is not None else layer(delta)
        if self.mode == "late":
            # Biases in copied Wan blocks must not create updates for empty support.
            delta = delta * (F.adaptive_max_pool3d(support.float(), delta.shape[2:]) > 0)
        return delta


class GeometryDecoder(nn.Module):
    """Independent Wan upsampling tail, branching after the shared middle."""
    def __init__(self, decoder):
        super().__init__()
        from .vae2_2 import CausalConv3d

        self.upsamples = copy.deepcopy(decoder.upsamples)
        reference = decoder.head[2].weight
        self.head = nn.Sequential(
            copy.deepcopy(decoder.head[0]), nn.SiLU(),
            CausalConv3d(decoder.dim, 4, 3, padding=1,
                         device=reference.device, dtype=reference.dtype))
        # The RGB projection has different semantics; only this projection is new.
        nn.init.normal_(self.head[2].weight, std=1e-3)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, feature, feat_cache=None, first_chunk=False):
        from .vae2_2 import decode_tail
        return decode_tail(feature, self.upsamples, self.head, feat_cache,
                           first_chunk=first_chunk)


def make_vae():
    from .vae2_2 import WanVAE_
    with torch.device("meta"):
        return WanVAE_(dim=160, dec_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4],
                       num_res_blocks=2, temperal_downsample=[False, True, True])
