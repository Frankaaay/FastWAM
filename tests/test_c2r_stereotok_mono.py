"""Verify native mono routing without loading deployment weights."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock
import torch
from torch import nn

source = Path(__file__).resolve().parents[1] / "src/fastwam/models/wan22/stereotok_vae.py"
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
namespace = dict(torch=torch, nn=nn, latent_scale=lambda device: (torch.zeros(48), torch.ones(48)))
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)

class MonoContract(unittest.TestCase):
    def test_native_left_only_and_precision(self):
        codec = namespace["StereoTokVAE"].__new__(namespace["StereoTokVAE"])
        nn.Module.__init__(codec)
        codec.model = Mock()
        mean = torch.ones(2, 48, 1, 24, 20)
        codec.model.encode_posterior.return_value = (mean, torch.zeros_like(mean))
        left = torch.zeros(2, 3, 1, 384, 320, dtype=torch.bfloat16)
        out = codec.encode(left, "cpu")
        self.assertFalse(codec.requires_stereo)
        args, kwargs = codec.model.encode_posterior.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(kwargs, {})
        self.assertEqual(args[0].dtype, torch.float32)
        self.assertTrue(torch.equal(args[0], left.float()))
        self.assertTrue(torch.equal(out, mean))
        with self.assertRaises(ValueError):
            codec.encode(left, "cpu", right_videos=left)
        with self.assertRaises(ValueError):
            codec.encode(left + float("nan"), "cpu")

if __name__ == "__main__":
    unittest.main()
