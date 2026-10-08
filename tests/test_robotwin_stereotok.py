"""CPU contracts; no model downloads, checkpoints or simulation assets."""
import ast
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F

from fastwam.datasets.robotwin_stereo import (LEFT_CAMERAS, RIGHT_CAMERAS,
                                            compose_mosaic, observation_mosaics)
from fastwam.models.wan22.stereotok.student import LAS2GuidedFusion
from fastwam.models.wan22.stereotok.checkpoints import load_student
from fastwam.models.wan22.stereotok.vae2_2 import WanVAE_, latent_scale
from fastwam.models.wan22.stereotok_vae import StereoTokVAE

ROOT = Path(__file__).resolve().parents[1]


class TestStereoContracts(unittest.TestCase):
    def test_mosaic_training_rollout_parity(self):
        rng = np.random.default_rng(2)
        raw = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(6)]
        observation = {"observation": {name: {"rgb": image} for name, image in
                                      zip(LEFT_CAMERAS + RIGHT_CAMERAS, raw)}}
        left, right = observation_mosaics(observation)
        resized = [F.resize(torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255,
                            [240, 320], antialias=True) for image in raw]
        torch.testing.assert_close(left, compose_mosaic(torch.stack(resized[:3]))[:, 0][None], rtol=0, atol=0)
        torch.testing.assert_close(right, compose_mosaic(torch.stack(resized[3:]))[:, 0][None], rtol=0, atol=0)
        self.assertEqual(left.shape, (1, 3, 384, 320))
        self.assertFalse(torch.equal(left, right))

    def test_missing_right_eye_fails(self):
        raw = np.zeros((32, 32, 3), np.uint8)
        observation = {"observation": {name: {"rgb": raw} for name in LEFT_CAMERAS}}
        with self.assertRaises(KeyError):
            observation_mosaics(observation)

    def test_correspondence_never_crosses_wrist_tiles(self):
        fusion = LAS2GuidedFusion.__new__(LAS2GuidedFusion)
        nn.Module.__init__(fusion)
        scores = torch.zeros(2, 48, 3, 4, 8)
        mask = torch.ones(2, 2, 3, 16, 32, dtype=torch.bool)
        mask[1, 1, 1] = False
        regions = torch.zeros(4, 8, dtype=torch.long)
        regions[2:, :4], regions[2:, 4:] = 1, 2
        probabilities = fusion.probabilities_from_cost(scores, mask, 112, regions)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertEqual(probabilities[1, :, 1].abs().sum().item(), 0)
        # Right wrist's first column can match itself but cannot look into left wrist.
        self.assertEqual(probabilities[0, 1:, :, 2:, 4].abs().sum().item(), 0)
        torch.testing.assert_close(probabilities[0, :, :, 2:, 4].sum(0), torch.ones(3, 2))
        self.assertGreater(probabilities[0, 1, :, :2, 4].sum().item(), 0)

    def test_codec_shape_normalization_and_frozen_precision(self):
        class Posterior(nn.Module):
            z_dim = 48
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1.000123], dtype=torch.float32))
                self.fusion = types.SimpleNamespace(max_disparity=112)
            def encode_posterior(self, left, right, mask, disparity, camera_regions):
                self.last = (right.clone(), camera_regions.clone())
                shape = (len(left), 48, 1 + (left.shape[2] - 1) // 4, 24, 20)
                return torch.zeros(shape), torch.zeros(shape)
        codec = StereoTokVAE.__new__(StereoTokVAE)
        nn.Module.__init__(codec)
        codec.model = Posterior()
        expected_weight = codec.model.weight.detach().clone()
        codec.train(True).to(dtype=torch.bfloat16)
        self.assertFalse(codec.training)
        self.assertFalse(codec.model.weight.requires_grad)
        self.assertEqual(codec.model.weight.dtype, torch.float32)
        torch.testing.assert_close(codec.model.weight, expected_weight, rtol=0, atol=0)
        left, right = torch.zeros(1, 3, 9, 384, 320), torch.full((1, 3, 9, 384, 320), 0.25)
        encoded = codec.encode(left, "cpu", right_videos=right)
        self.assertEqual(encoded.shape, (1, 48, 3, 24, 20))
        shift, scale = latent_scale("cpu")
        torch.testing.assert_close(encoded[0, :, 0, 0, 0], -shift * scale)
        torch.testing.assert_close(codec.model.last[0], right)
        self.assertEqual(codec.model.last[1][64, 40].item(), 2)
        with self.assertRaises(ValueError):
            codec.encode(left, "cpu")
        with self.assertRaises(ValueError):
            codec.encode(left, "cpu", tiled=True, right_videos=right)

    def test_native_wan_temporal_contract(self):
        torch.manual_seed(1)
        vae = WanVAE_(dim=8, dec_dim=8, z_dim=48, dim_mult=[1, 2, 4, 4],
                      num_res_blocks=1, temperal_downsample=[False, True, True]).eval().requires_grad_(False)
        with torch.no_grad():
            mean, _ = vae.encode_posterior(torch.zeros(1, 3, 9, 32, 32))
        self.assertEqual(mean.shape, (1, 48, 3, 2, 2))

    def test_native_stereo_region_argument_and_temporal_contract(self):
        class Fusion(nn.Module):
            mode = "early"
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.zeros(()))
            def prepare(self, left, right, mask, disparity, regions):
                self.regions = regions
                return torch.zeros(len(left), 48, left.shape[2], left.shape[3] // 4, left.shape[4] // 4)
            def forward(self, left, right, probabilities, cache):
                return torch.zeros_like(left)
        vae = WanVAE_(dim=8, dec_dim=8, z_dim=48, dim_mult=[1, 2, 4, 4],
                      num_res_blocks=1, temperal_downsample=[False, True, True])
        vae.fusion = Fusion()
        vae.eval().requires_grad_(False)
        regions = torch.zeros(8, 8, dtype=torch.long)
        with torch.no_grad():
            mean, _ = vae.encode_posterior(torch.zeros(1, 3, 9, 32, 32), torch.ones(1, 3, 9, 32, 32),
                                           camera_regions=regions)
        self.assertEqual(mean.shape, (1, 48, 3, 2, 2))
        self.assertIs(vae.fusion.regions, regions)

    def test_validation_batch_preserves_right_eye(self):
        tree = ast.parse((ROOT / "src/fastwam/trainer.py").read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_to_batched_eval_sample")
        method.decorator_list = []
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer.py", "exec"), scope)
        sample = dict(video=torch.zeros(3, 9, 32, 32), video_right=torch.ones(3, 9, 32, 32),
                      action=torch.zeros(32, 14), prompt="task")
        result = scope["_to_batched_eval_sample"](sample)
        self.assertEqual(result["action_horizon"], 32)
        torch.testing.assert_close(result["video_right"], sample["video_right"][None])
        sample["video_right"] = torch.ones(3, 1, 32, 32)
        with self.assertRaises(ValueError):
            scope["_to_batched_eval_sample"](sample)

    def test_reject_wrong_student_counter(self):
        with patch("torch.load", return_value={"format": "stereotok-student-v3", "stage": 2,
                                              "provenance": {"generator_updates": 5000}}):
            with self.assertRaisesRegex(ValueError, "u8000"):
                load_student("unused.pt")

    def test_right_camera_translation_preserves_rotation(self):
        # Exercise the actual renderer pose method without importing SAPIEN/Open3D.
        tree = ast.parse((ROOT / "third_party/RoboTwin/envs/camera/camera.py").read_text())
        camera = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Camera")
        method = next(node for node in camera.body if isinstance(node, ast.FunctionDef) and node.name == "_update_stereo_poses")
        scope = {"sapien": types.SimpleNamespace(Pose=lambda matrix: matrix)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "camera.py", "exec"), scope)
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        matrix[:3, 3] = [0.2, 0.3, 1]
        left = types.SimpleNamespace(entity=types.SimpleNamespace(get_pose=lambda:
                                     types.SimpleNamespace(to_transformation_matrix=lambda: matrix)))
        poses = []
        right = types.SimpleNamespace(entity=types.SimpleNamespace(set_pose=poses.append))
        scope["_update_stereo_poses"](types.SimpleNamespace(stereo_pairs=[("head_camera_right", left, right, 0.06)]))
        np.testing.assert_array_equal(poses[0][:3, :3], matrix[:3, :3])
        np.testing.assert_allclose(poses[0][:3, 3] - matrix[:3, 3], [0.06, 0, 0])
        np.testing.assert_array_equal(matrix[:3, 3], [0.2, 0.3, 1])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
