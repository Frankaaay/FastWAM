import importlib.util
from pathlib import Path
import unittest
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('gt', Path(__file__).resolve().parents[1] / 'scripts/h100_robotwin_gt.py')
gt = importlib.util.module_from_spec(spec); spec.loader.exec_module(gt)


class NativeGT(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Production uses Linux flock')
    def test_isolated_failure_does_not_stop_next_episode_or_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'inputs').mkdir()
            (root / 'inputs/plan.json').write_text(json.dumps({'rows': [
                {'source_episode_index': 11, 'length': 2},
                {'source_episode_index': 12, 'length': 1}]}))
            with patch.object(gt, 'ROOT', root), patch.dict(os.environ, SLURM_JOB_ID='test'), \
                 patch.object(gt.subprocess, 'run', side_effect=[
                     SimpleNamespace(returncode=1), SimpleNamespace(returncode=1),
                     SimpleNamespace(returncode=0)]) as run:
                gt.pool(0, False)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(run.call_args.args[0][-1], '12')
                self.assertEqual(json.loads((root / 'worker-0-test.json').read_text())['failed'], [11])
                with self.assertRaisesRegex(RuntimeError, 'Publication blocked'):
                    gt.finalize()
                self.assertFalse((root / 'ready.json').exists())

    @unittest.skipUnless(sys.platform == 'linux', 'Production uses Linux flock')
    def test_three_consecutive_failures_stop_systemic_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'inputs').mkdir()
            (root / 'inputs/plan.json').write_text(json.dumps({'rows': [
                {'source_episode_index': i, 'length': 1} for i in range(4)]}))
            with patch.object(gt, 'ROOT', root), patch.dict(os.environ, SLURM_JOB_ID='test'), \
                 patch.object(gt.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
                with self.assertRaisesRegex(RuntimeError, 'Three consecutive'):
                    gt.pool(0, False)
                self.assertEqual(run.call_count, 6)

    def test_depth_and_integer_ids(self):
        class Camera:
            def get_picture(self, key):
                if key == 'Position':
                    return np.array([[[0, 0, -2, 1], [np.nan, 0, np.inf, 0]]], np.float32)
                return np.array([[[300, 999, 0, 0], [0, 0, 0, 0]]], np.uint32)
        result = gt.native_gt(Camera().get_picture, .06, 100)
        np.testing.assert_array_equal(result['depth_m'], [[2, 0]])
        np.testing.assert_array_equal(result['valid'], [[1, 0]])
        np.testing.assert_array_equal(result['disparity_px'], [[3, 0]])
        self.assertEqual(result['actor_id'][0, 0], 999)
        self.assertEqual(result['mesh_id'][0, 0], 300)

    def test_occlusion_and_outside_are_distinct(self):
        def eye(depth):
            return dict(depth_m=np.array([depth], np.float32), valid=np.ones((1, 4), np.uint8),
                        disparity_px=np.ones((1, 4), np.float32))
        left, right = eye([2, 2, 2, 2]), eye([1, 2, 2, 2])
        gt.stereo_visibility(left, right)
        np.testing.assert_array_equal(left['outside_right_or_left_fov'], [[1, 0, 0, 0]])
        np.testing.assert_array_equal(left['occluded'], [[0, 1, 0, 0]])


if __name__ == '__main__': unittest.main()
