"""CPU regression checks for unified-action supervision; no weights needed."""
import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


# Load the real loss method without importing CUDA/model-download dependencies.
source = Path(__file__).resolve().parents[1] / "src/fastwam/models/wan22/fastwam.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastWAM")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_compute_action_loss_per_sample")
method.decorator_list = []
scope = {"torch": torch, "F": F}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
loss = scope[method.name]


class UnifiedActionMaskTest(unittest.TestCase):
    def test_inactive_dimensions_have_no_loss_or_gradient(self):
        prediction = torch.tensor([[[2., 100.], [4., 100.]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        pad = torch.tensor([[[False, True], [False, True]]])
        value = loss(prediction, target, pad)
        torch.testing.assert_close(value, torch.tensor([10.]))
        value.sum().backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[[2., 0.], [4., 0.]]]))

    def test_time_mask_matches_broadcast_element_mask(self):
        prediction = torch.arange(24.).reshape(2, 3, 4)
        target = torch.zeros_like(prediction)
        pad = torch.tensor([[False, True, False], [True, False, False]])
        torch.testing.assert_close(loss(prediction, target, pad),
                                   loss(prediction, target, pad.unsqueeze(-1).expand_as(prediction)))

    def test_all_padding_returns_zero(self):
        prediction = torch.ones(1, 2, 80, requires_grad=True)
        value = loss(prediction, torch.zeros_like(prediction), torch.ones_like(prediction, dtype=torch.bool))
        torch.testing.assert_close(value, torch.zeros(1))
        value.sum().backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_wrong_width_is_rejected(self):
        with self.assertRaises(ValueError):
            loss(torch.zeros(1, 2, 80), torch.zeros(1, 2, 80), torch.zeros(1, 2, 20, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
