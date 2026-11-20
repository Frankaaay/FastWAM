"""CPU checks for the explicit-depth baseline codec (left | disparity-image latents joined along width)."""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from fastwam.models.wan22 import wan_depth_vae
from fastwam.models.wan22.wan_depth_vae import WanDepthVAE, disparity_image, DISPARITY_RANGE


class FakeFusion(nn.Module):
    """Stands in for the frozen StereoTok matcher: puts all mass on candidate 3 (12 px) except a left border."""
    max_disparity = 112

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = []

    def prepare(self, left, right, content_mask, disparity, camera_regions=None):
        self.calls.append((left.shape, right.shape, None if camera_regions is None else camera_regions.shape))
        b, _, t, h, w = left.shape
        p = torch.zeros(b, 48, t, h // 4, w // 4)
        p[:, 3, ..., 1:] = 1
        return p


class FakeStudent:
    def __init__(self):
        self.fusion = FakeFusion()


class WanDepthCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.wan = WanVideoVAE38()
        cls.path = __file__
        with patch.object(wan_depth_vae, 'load_student', lambda path: FakeStudent()):
            cls.codec = WanDepthVAE(cls.wan, cls.path, 'cpu')
        cls.left = torch.rand(2, 3, 5, 32, 64) * 2 - 1
        cls.right = torch.rand(2, 3, 5, 32, 64) * 2 - 1

    def test_disparity_image_mapping(self):
        p = torch.zeros(1, 48, 2, 2, 3)
        p[:, 3, ..., 1:] = .5
        p[:, 5, ..., 1:] = .5
        image = disparity_image(p)
        self.assertEqual(tuple(image.shape), (1, 3, 2, 8, 12))
        lo, hi = DISPARITY_RANGE
        expected = (math.log(16.) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 2 - 1
        torch.testing.assert_close(image[0, 0, :, :, 4:], torch.full((2, 8, 8), expected))
        self.assertTrue(torch.equal(image[0, 0, :, :, :4], torch.full((2, 8, 4), -1.)))
        self.assertTrue(torch.equal(image[:, 0], image[:, 2]))

    def test_halves_are_native_wan_of_left_and_disparity(self):
        z = self.codec.encode(self.left, 'cpu', right_videos=self.right)
        self.assertEqual(tuple(z.shape), (2, 48, 2, 2, 8))
        depth = self.codec.disparity_video(self.left, self.right)
        with torch.no_grad():
            left = self.wan.model.encode(self.left, self.wan.scale)
            dz = self.wan.model.encode(depth, self.wan.scale)
        self.assertTrue(torch.equal(z[..., :4], left))
        self.assertTrue(torch.equal(z[..., 4:], dz))

    def test_contract(self):
        self.assertTrue(self.codec.requires_stereo)
        self.assertEqual((self.codec.upsampling_factor, self.codec.width_factor), (16, 2))
        self.assertFalse(any(p.requires_grad for p in self.codec.parameters()))
        self.assertEqual(self.codec.binding['format'], 'wan2.2-vae-depth-lr')
        with self.assertRaisesRegex(ValueError, 'right eyes'):
            self.codec.encode(self.left, 'cpu')
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.codec.encode(self.left, 'cpu', right_videos=self.right[..., :32])
        mosaic = torch.zeros(1, 3, 1, 384, 320)
        regions = wan_depth_vae.camera_regions(384, 320, 'cpu')
        self.assertEqual(int(regions[70, 10]), 1)
        self.assertEqual(int(regions[70, 50]), 2)
        self.assertEqual(int(regions[10, 50]), 0)
        self.assertEqual(tuple(self.codec.disparity_video(mosaic, mosaic).shape), (1, 3, 1, 384, 320))


if __name__ == '__main__':
    unittest.main()
