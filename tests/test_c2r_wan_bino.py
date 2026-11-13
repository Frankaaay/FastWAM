"""CPU checks for the naive binocular Wan baseline codec (left|right latents joined along width)."""
import unittest
import torch
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from fastwam.models.wan22.wan_bino_vae import WanBinoVAE


class WanBinoCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.wan = WanVideoVAE38()
        cls.bino = WanBinoVAE(cls.wan)
        cls.left = torch.rand(2, 3, 5, 32, 48) * 2 - 1
        cls.right = torch.rand(2, 3, 5, 32, 48) * 2 - 1

    def test_lr_halves_equal_independent_native_encode(self):
        z = self.bino.encode(self.left, 'cpu', right_videos=self.right)
        self.assertEqual(tuple(z.shape), (2, 48, 2, 2, 6))
        with torch.no_grad():
            left = self.wan.model.encode(self.left, self.wan.scale)
            right = self.wan.model.encode(self.right, self.wan.scale)
        self.assertTrue(torch.equal(z[..., :3], left))
        self.assertTrue(torch.equal(z[..., 3:], right))

    def test_ll_halves_identical_and_single_frame(self):
        z = self.bino.encode(self.left, 'cpu', right_videos=self.left)
        self.assertTrue(torch.equal(z[..., :3], z[..., 3:]))
        image = self.left[:1, :, :1]
        self.assertEqual(tuple(self.bino.encode(image, 'cpu', right_videos=image).shape), (1, 48, 1, 2, 6))

    def test_decode_splits_eyes(self):
        z = self.bino.encode(self.left[:1], 'cpu', right_videos=self.right[:1])
        video = self.bino.decode(z, 'cpu')
        self.assertEqual(tuple(video.shape), (1, 3, 5, 32, 96))
        self.assertTrue(torch.equal(video[..., :48], self.wan.decode(z[..., :3], 'cpu')))
        self.assertTrue(torch.equal(video[..., 48:], self.wan.decode(z[..., 3:], 'cpu')))

    def test_contract(self):
        self.assertTrue(self.bino.requires_stereo)
        self.assertEqual(self.bino.model.z_dim, 48)
        self.assertEqual((self.bino.upsampling_factor, self.bino.width_factor), (16, 2))
        self.assertFalse(any(p.requires_grad for p in self.bino.parameters()))
        self.bino.train()
        self.assertFalse(self.bino.training or self.wan.training)
        with self.assertRaisesRegex(ValueError, 'right eyes'):
            self.bino.encode(self.left, 'cpu')
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.bino.encode(self.left, 'cpu', right_videos=self.right[..., :32])


if __name__ == '__main__':
    unittest.main()
