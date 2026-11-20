"""CPU checks for the GeoSem-style baseline (frozen LAS2 p(d) + DINOv3 + Wan latent, trainable projector)."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fastwam.models.wan22 import geosem
from fastwam.models.wan22.geosem import GeoSemProjector, WanGeoSemVAE, panels
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38


class FakeFusion(nn.Module):
    max_disparity = 112

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.regions = None

    def prepare(self, left, right, content_mask, disparity, camera_regions=None):
        self.regions = camera_regions
        b, _, t, h, w = left.shape
        p = torch.zeros(b, 48, t, h // 4, w // 4)
        p[:, 2] = 1
        return p


class FakeStudent:
    def __init__(self):
        self.fusion = FakeFusion()


class FakeDino(nn.Module):
    """Patch token i of a crop equals the crop's mean intensity + i, so panel placement is checkable."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.shapes = []

    def load_state_dict(self, *a, **k):
        return None

    def forward_features(self, x):
        b, _, h, w = x.shape
        self.shapes.append((h, w))
        n = (h // 16) * (w // 16)
        tokens = x.mean((1, 2, 3))[:, None, None] + torch.arange(n, dtype=x.dtype)[None, :, None]
        return {"x_norm_patchtokens": tokens.expand(b, n, 768).clone()}


class GeoSemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.wan = WanVideoVAE38()
        cls.dino = FakeDino()
        with patch.object(geosem, "_sha256", lambda p: geosem.DINO_SHA256), \
             patch.object(geosem, "load_student", lambda p: FakeStudent()), \
             patch("torch.hub.load", lambda *a, **k: cls.dino), \
             patch("torch.load", lambda *a, **k: {}), \
             patch("subprocess.check_output", lambda *a, **k: geosem.DINO_SOURCE_SHA + "\n"):
            cls.codec = WanGeoSemVAE(cls.wan, __file__, ".", __file__, "cpu")

    def test_projector_identity_at_init_and_trainable(self):
        proj = GeoSemProjector()
        latent, feats = torch.randn(2, 48, 1, 24, 20), torch.randn(2, 48 + 768, 1, 24, 20)
        self.assertTrue(torch.equal(proj(latent, feats), latent))
        proj(latent, feats).square().sum().backward()
        self.assertGreater(float(proj.out.weight.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "latent grid"):
            proj(latent, feats[..., :10])

    def test_features_grid_panels_and_geometry(self):
        left = torch.rand(1, 3, 1, 384, 320) * 2 - 1
        left[:, :, :, 256:, 160:] = 1.0                                   # right-wrist panel is white
        feats = self.codec.features(left, left.clone())
        self.assertEqual(tuple(feats.shape), (1, 48 + 768, 1, 24, 20))
        torch.testing.assert_close(feats[:, 2], torch.ones(1, 1, 24, 20))     # pooled p(d) keeps the distribution
        self.assertEqual(float(feats[:, :48].sum(1).min()), 1.0)
        self.assertEqual(self.dino.shapes[-3:], [(256, 320), (128, 160), (128, 160)])
        self.assertEqual(tuple(self.codec.fusion.regions.shape), (96, 80))
        # Right-wrist panel: first patch value = mean of normalized white crop + 0, placed at latent (16, 10).
        sem = feats[:, 48:]
        self.assertAlmostEqual(float(sem[0, 0, 0, 16, 11] - sem[0, 0, 0, 16, 10]), 1.0, places=4)
        self.assertAlmostEqual(float(sem[0, 0, 0, 17, 10] - sem[0, 0, 0, 16, 10]), 10.0, places=4)

    def test_encode_pair_contract(self):
        left = torch.rand(1, 3, 5, 64, 64) * 2 - 1
        z, f = self.codec.encode_pair(left, "cpu", right_videos=left.clone())
        with torch.no_grad():
            native = self.wan.model.encode(left, self.wan.scale)
        self.assertTrue(torch.equal(z, native))
        self.assertEqual(tuple(f.shape), (1, 48 + 768, 1, 4, 4))
        self.assertFalse(any(p.requires_grad for p in self.codec.parameters()))
        with self.assertRaisesRegex(ValueError, "right eyes"):
            self.codec.encode_pair(left, "cpu", right_videos=None)
        self.assertEqual(panels(384, 320)[1], (256, 384, 0, 160))


if __name__ == "__main__":
    unittest.main()
