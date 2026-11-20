"""CPU checks for the policy-side stereo cross-attention baseline."""
import unittest
import torch
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from fastwam.models.wan22.stereo_xattn import StereoCrossAttention, WanPairVAE, panel_ids


class StereoXAttnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.wan = WanVideoVAE38()
        cls.codec = WanPairVAE(cls.wan)

    def test_codec_left_targets_and_causal_right_first_frame(self):
        left = torch.rand(1, 3, 5, 32, 48) * 2 - 1
        right = torch.rand(1, 3, 5, 32, 48) * 2 - 1
        zl, zr = self.codec.encode_pair(left, 'cpu', right_videos=right)
        with torch.no_grad():
            full_right = self.wan.model.encode(right, self.wan.scale)
            native_left = self.wan.model.encode(left, self.wan.scale)
        self.assertTrue(torch.equal(zl, native_left))
        self.assertTrue(torch.equal(self.codec.encode(left, 'cpu', right_videos=right), native_left))
        self.assertEqual(tuple(zr.shape), (1, 48, 1, 2, 3))
        torch.testing.assert_close(zr, full_right[:, :, :1], rtol=1e-5, atol=1e-5)
        self.assertFalse(any(p.requires_grad for p in self.codec.parameters()))
        with self.assertRaisesRegex(ValueError, 'right eyes'):
            self.codec.encode_pair(left, 'cpu', right_videos=None)

    def test_fusion_identity_at_init_and_row_panel_locality(self):
        fusion = StereoCrossAttention()
        left, right = torch.randn(2, 48, 1, 24, 20), torch.randn(2, 48, 1, 24, 20)
        self.assertTrue(torch.equal(fusion(left, right), left))
        torch.nn.init.normal_(fusion.out.weight, std=.02)
        base = fusion(left, right)
        changed = right.clone()
        changed[:, :, :, 3, 5] += 10          # head panel, row 3
        delta = (fusion(left, changed) - base).abs().amax(1)[:, 0]
        self.assertGreater(float(delta[:, 3].max()), 0)
        mask = torch.ones_like(delta, dtype=torch.bool); mask[:, 3] = False
        self.assertEqual(float(delta[mask].max()), 0.)
        changed = right.clone()
        changed[:, :, :, 20, 2] += 10         # left-wrist panel
        delta = (fusion(left, changed) - base).abs().amax(1)[:, 0]
        self.assertEqual(float(delta[:, 20, 10:].max()), 0.)
        self.assertGreater(float(delta[:, 20, :10].max()), 0)

    def test_gradients_reach_fusion(self):
        fusion = StereoCrossAttention()
        left, right = torch.randn(1, 48, 1, 24, 20), torch.randn(1, 48, 1, 24, 20)
        fusion(left, right).square().sum().backward()
        self.assertIsNotNone(fusion.out.weight.grad)
        self.assertGreater(float(fusion.out.weight.grad.abs().sum()), 0)
        ids = panel_ids(24, 20, 'cpu')
        self.assertEqual((int(ids[0, 0]), int(ids[20, 0]), int(ids[20, 15])), (0, 1, 2))


if __name__ == '__main__':
    unittest.main()
