"""Policy-level stereo baseline in the style of StereoPolicy: a Stereo Transformer fuses frozen Wan2.2 latents.

Each eye is encoded by the unmodified Wan2.2 VAE. A small trainable module lets every left-view token attend to the
right-view tokens of the same latent row within the same camera panel (rectified cameras: correspondences lie on the
row), and adds the result to the left token through a zero-initialized output projection. The fused tokens replace
only the observation (conditioning) frame; the token count equals the monocular Wan baseline, and the future-frame
targets stay the left-view Wan latents. The module is trained jointly with the policy from action/video losses.
"""
import torch
from torch import nn
import torch.nn.functional as F


def panel_ids(height, width, device):
    """Camera panel of each latent cell for the RoboTwin mosaics used by FastWAM."""
    ids = torch.zeros(height, width, dtype=torch.long, device=device)
    if (height, width) == (24, 20):          # 384x320: head on top, two wrists below
        ids[16:, :10], ids[16:, 10:] = 1, 2
    elif (height, width) == (14, 28):        # 224x448: two panels side by side
        ids[:, 14:] = 1
    return ids


class StereoCrossAttention(nn.Module):
    def __init__(self, channels=48, dim=256, heads=8, layers=2, max_offset=20):
        super().__init__()
        self.heads, self.max_offset = heads, max_offset
        self.left_in = nn.Linear(channels, dim)
        self.right_in = nn.Linear(channels, dim)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                norm_q=nn.LayerNorm(dim), norm_kv=nn.LayerNorm(dim), q=nn.Linear(dim, dim), kv=nn.Linear(dim, 2 * dim),
                o=nn.Linear(dim, dim), norm_mlp=nn.LayerNorm(dim),
                mlp=nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)))))
        # Learned bias per head and signed horizontal offset (right x - left x), as a disparity prior.
        self.offset_bias = nn.Parameter(torch.zeros(heads, 2 * max_offset + 1))
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, left, right):
        """left, right: [B,48,T,H,W] latents -> fused left latents of the same shape."""
        if left.shape != right.shape or left.ndim != 5:
            raise ValueError("Stereo fusion expects matching [B,C,T,H,W] latents")
        b, c, t, h, w = left.shape
        dtype = self.left_in.weight.dtype
        l = left.permute(0, 2, 3, 4, 1).reshape(b * t * h, w, c).to(dtype)
        r = right.permute(0, 2, 3, 4, 1).reshape(b * t * h, w, c).to(dtype)
        x, kv_src = self.left_in(l), self.right_in(r)
        ids = panel_ids(h, w, left.device)
        offset = torch.arange(w, device=left.device)[None, :] - torch.arange(w, device=left.device)[:, None]
        allowed = (ids[:, :, None] == ids[:, None, :])                          # [H,W,W] same panel, same row
        bias = self.offset_bias[:, (offset.clamp(-self.max_offset, self.max_offset) + self.max_offset)]  # [heads,W,W]
        mask = torch.where(allowed[:, None], bias[None].to(dtype), torch.tensor(float('-inf'), dtype=dtype, device=left.device))
        mask = mask.repeat(b * t, 1, 1, 1)                                      # [B*T*H,heads,W,W]
        d = x.shape[-1] // self.heads
        for blk in self.blocks:
            q = blk['q'](blk['norm_q'](x)).view(-1, w, self.heads, d).transpose(1, 2)
            k, v = blk['kv'](blk['norm_kv'](kv_src)).view(-1, w, 2, self.heads, d).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + blk['o'](a.transpose(1, 2).reshape(-1, w, self.heads * d))
            x = x + blk['mlp'](blk['norm_mlp'](x))
        delta = self.out(x).view(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return left + delta.to(left.dtype)


class WanPairVAE(nn.Module):
    """Frozen Wan2.2 VAE that also encodes the right eye of the observation frame for the stereo fusion."""
    upsampling_factor = 16
    temporal_downsample_factor = 4
    requires_stereo = True
    stereo_fusion_input = True

    def __init__(self, wan_vae):
        super().__init__()
        self.wan = wan_vae.eval().requires_grad_(False)
        self.binding = {"format": "wan2.2-vae-stereo-xattn", "encode": "per-eye-native-wan",
                        "latent_layout": "left; right observation frame fused by a policy-side Stereo Transformer"}

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

    def _stack(self, videos, device):
        dtype = next(self.wan.model.parameters()).dtype
        return (videos if isinstance(videos, torch.Tensor) else torch.stack(list(videos))).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, videos, device, tiled=False, tile_size=None, tile_stride=None, *, right_videos=None):
        """Left-view latents only (the targets); the right eye enters through encode_pair."""
        if tiled:
            raise ValueError("Wan stereo encoding requires tiled=False")
        return self.wan.model.encode(self._stack(videos, device), self.wan.scale)

    @torch.no_grad()
    def encode_pair(self, videos, device, *, right_videos):
        if right_videos is None:
            raise ValueError("The stereo cross-attention baseline requires synchronized right eyes")
        left, right = self._stack(videos, device), self._stack(right_videos, device)
        if left.shape != right.shape or left.ndim != 5 or left.shape[1] != 3:
            raise ValueError("Expected matching [B,3,T,H,W] stereo inputs")
        # Causal Wan: the first latent step depends on the first frame only.
        return self.wan.model.encode(left, self.wan.scale), self.wan.model.encode(right[:, :, :1], self.wan.scale)

    @torch.no_grad()
    def decode(self, latents, device, tiled=False, tile_size=None, tile_stride=None):
        return self.wan.decode(latents, device, tiled, tile_size, tile_stride)
