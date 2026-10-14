"""Keep LIBERO's raw instructions out of the RoboDojo prompt template."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "src/fastwam/datasets/openwam_dataset.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name == "unique_prompts")
scope = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)


class CanonicalPromptTest(unittest.TestCase):
    def test_libero_preserves_raw_instruction(self):
        dataset = SimpleNamespace(benchmark="libero", reader=SimpleNamespace(
            _task_idx_to_text={0: "put the cup on the plate", 1: "put the cup on the plate"}))
        self.assertEqual(scope["unique_prompts"](dataset), ["put the cup on the plate"])

    def test_blank_metadata_fails(self):
        dataset = SimpleNamespace(benchmark="libero", reader=SimpleNamespace(
            _task_idx_to_text={0: " "}))
        with self.assertRaises(ValueError):
            scope["unique_prompts"](dataset)


if __name__ == "__main__":
    unittest.main()
