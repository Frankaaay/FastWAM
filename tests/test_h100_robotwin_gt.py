import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('gt', Path(__file__).resolve().parents[1] / 'scripts/h100_robotwin_gt.py')
gt = importlib.util.module_from_spec(spec); spec.loader.exec_module(gt)


class NativeGT(unittest.TestCase):
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
