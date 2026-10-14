"""Evaluation coverage, duplicate detection, and fine-grained claim contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[1] / 'scripts/c2r/evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='c2r-eval-test-'))
        (self.root / 'results').mkdir()
        self.manifest = dict(episodes=2, cases=[dict(task='a', phase='clean')])

    def test_claim_has_single_owner(self):
        self.assertTrue(evaluation.claim(self.root, 'a--0'))
        self.assertFalse(evaluation.claim(self.root, 'a--0'))
        self.assertTrue(evaluation.claim(self.root, 'a--1'))

    def test_no_partial_score_and_exact_coverage(self):
        def save(name, ordinal, success):
            evaluation.write_json(self.root / 'results' / name,
                dict(task='a', phase='clean', episodes=[dict(ordinal=ordinal, success=success)]))
        save('0.json', 0, True)
        self.assertIsNone(evaluation.summarize(self.root, self.manifest)['overall']['clean'])
        save('1.json', 1, False)
        result = evaluation.summarize(self.root, self.manifest)
        self.assertTrue(result['complete'])
        self.assertEqual(result['overall']['clean'], .5)
        save('duplicate.json', 1, False)
        with self.assertRaises(ValueError):
            evaluation.summarize(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()
