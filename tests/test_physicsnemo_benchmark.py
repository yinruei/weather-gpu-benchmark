"""Workload identity and strict numerical checks; optional real FNO smoke test."""
import unittest
import numpy as np
from physicsnemo_benchmark import array_mapping_hash, error_summary, make_model, parse_cases


class PhysicsBenchmarkTest(unittest.TestCase):
    def test_cases_reject_duplicates_and_insufficient_fourier_grid(self):
        self.assertEqual(parse_cases(['4x64'])[0], dict(case='b4_s64', batch=4, side=64))
        for bad in [[], ['0x64'], ['1x8'], ['1x32', '1x32'], ['invalid']]:
            with self.assertRaises(ValueError):
                parse_cases(bad)

    def test_array_hash_covers_names_dtype_shape_and_content(self):
        original = {'x': np.arange(4, dtype=np.float32)}
        self.assertEqual(array_mapping_hash(original), array_mapping_hash(dict(original)))
        for changed in [{'y': original['x']}, {'x': original['x'].astype(np.float64)},
                        {'x': original['x'].reshape(2, 2)}, {'x': original['x']+1}]:
            self.assertNotEqual(array_mapping_hash(original), array_mapping_hash(changed))
        self.assertNotEqual(array_mapping_hash({'x': np.array(1.)}), array_mapping_hash({'x': np.array([1.])}))

    def test_error_check_is_elementwise_and_does_not_hide_nonfinite_output(self):
        reference = np.array([0., 100.], dtype=np.float64)
        self.assertLess(error_summary(np.array([1e-6, 100.001]), reference)['max_tolerance_fraction'], 1)
        for bad in [np.array([.001, 100.]), np.array([np.nan, 100.]), np.array([0.])]:
            with self.assertRaises(ValueError):
                error_summary(bad, reference)

    def test_optional_physicsnemo_fno_forward(self):
        try:
            import torch
            import physicsnemo
        except ImportError:
            self.skipTest('Isolated PhysicsNeMo environment optional')
        torch.manual_seed(1)
        model = make_model()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
