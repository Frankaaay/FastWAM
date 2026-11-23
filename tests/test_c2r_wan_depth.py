"""CPU checks for the explicit-depth baseline codec (left | metric-depth-image latents joined along width)."""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from fastwam.models.wan22 import wan_depth_vae
from fastwam.models.wan22.wan_depth_vae import (CAMERA_BASELINES_M, DEPTH_RANGE_M, NATIVE_FX, WanDepthVAE,
                                                 depth_mosaic, depth_value, mosaic_panels)


class FakeLAS2(nn.Module):
    """Stands in for LAS2-H: a constant 10 px disparity, except no match in the first column."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, left, right, iters=None, test_mode=False):
        self.calls.append((tuple(left.shape), float(left.min()), float(left.max()), iters, test_mode))
        d = torch.full((left.shape[0], 1, *left.shape[-2:]), 10.)
        d[..., 0] = 0.
        return d


def mapped(depth):
    lo, hi = DEPTH_RANGE_M
    return (math.log(min(max(depth, lo), hi)) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 2 - 1


class WanDepthCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.wan = WanVideoVAE38()
        cls.fake = FakeLAS2()
        with patch.object(wan_depth_vae, 'LAS2_H_SHA256', None), \
             patch.object(wan_depth_vae, 'load_las2', lambda path: cls.fake), \
             patch('hashlib.sha256') as sha:
            sha.return_value.hexdigest.return_value = None
            cls.codec = WanDepthVAE(cls.wan, __file__, 'cpu', microbatch=4)
        cls.left = torch.rand(1, 3, 5, 384, 320) * 2 - 1
        cls.right = torch.rand(1, 3, 5, 384, 320) * 2 - 1

    def test_panels_are_cameras_at_native_width(self):
        x = torch.zeros(2, 3, 384, 320)
        x[..., :256, :] = 1.
        x[..., 256:, :160] = 2.
        x[..., 256:, 160:] = 3.
        p = mosaic_panels(x)
        self.assertEqual(tuple(p.shape), (6, 3, 256, 320))
        for k, v in enumerate((1., 2., 3.)):
            self.assertTrue(torch.equal(p[2 * k:2 * k + 2], torch.full_like(p[:2], v)))
        with self.assertRaisesRegex(ValueError, 'mosaic'):
            mosaic_panels(torch.zeros(1, 3, 256, 256))

    def test_metric_depth_per_camera(self):
        d = torch.full((3, 1, 256, 320), 10.)
        d[..., 0] = 0.
        image = depth_mosaic(d)
        self.assertEqual(tuple(image.shape), (1, 3, 384, 320))
        head, wrist = NATIVE_FX * CAMERA_BASELINES_M[0] / 10, NATIVE_FX * CAMERA_BASELINES_M[1] / 10
        torch.testing.assert_close(image[0, 0, 100, 100], torch.tensor(mapped(head)))
        torch.testing.assert_close(image[0, 0, 300, 50], torch.tensor(mapped(wrist)))
        torch.testing.assert_close(image[0, 0, 300, 250], torch.tensor(mapped(wrist)))
        self.assertEqual(float(image[0, 0, 100, 0]), 1.0)  # no match -> far
        self.assertTrue(torch.equal(image[:, 0], image[:, 2]))
        self.assertEqual(float(depth_value(torch.tensor(1e6), .06)), -1.0)

    def test_halves_are_native_wan_of_left_and_depth(self):
        self.fake.calls.clear()
        z = self.codec.encode(self.left, 'cpu', right_videos=self.right)
        self.assertEqual(tuple(z.shape), (1, 48, 2, 24, 40))
        calls = list(self.fake.calls)
        self.assertEqual(sum(c[0][0] for c in calls), 15)  # 5 frames x 3 cameras
        self.assertTrue(all(c[0][1:] == (3, 256, 320) and c[3] == 4 and c[4] for c in calls))
        self.assertTrue(all(c[1] >= 0 and c[2] <= 255 for c in calls))
        depth = self.codec.depth_video(self.left, self.right)
        with torch.no_grad():
            left = self.wan.model.encode(self.left, self.wan.scale)
            dz = self.wan.model.encode(depth, self.wan.scale)
        self.assertTrue(torch.equal(z[..., :20], left))
        self.assertTrue(torch.equal(z[..., 20:], dz))

    def test_contract(self):
        self.assertTrue(self.codec.requires_stereo)
        self.assertEqual((self.codec.upsampling_factor, self.codec.width_factor), (16, 2))
        self.assertFalse(any(p.requires_grad for p in self.codec.parameters()))
        self.assertEqual(self.codec.binding['format'], 'wan2.2-vae-depthmap-lr')
        with self.assertRaisesRegex(ValueError, 'right eyes'):
            self.codec.encode(self.left, 'cpu')
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.codec.encode(self.left, 'cpu', right_videos=self.right[..., :32])

    def test_rejects_other_checkpoints(self):
        with self.assertRaisesRegex(ValueError, 'LAS2-H checkpoint'):
            WanDepthVAE(self.wan, __file__, 'cpu')


if __name__ == '__main__':
    unittest.main()
