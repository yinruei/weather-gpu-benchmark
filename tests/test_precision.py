"""Independent known answers and failure cases for precision comparisons."""
import unittest
import numpy as np
from precision_benchmark import precision_kernel, reference_metrics

try:
    import torch
except ImportError:
    torch = None


class ReferenceTest(unittest.TestCase):
    def test_known_answer(self):
        values = np.array([1, -2, 3], dtype=np.float32)
        np.testing.assert_allclose(reference_metrics(values, np.zeros(3)),
                                   [2/3, 2, np.sqrt(14/3)], rtol=1e-14)

    def test_invalid_inputs(self):
        for f, o in [(np.zeros(2), np.zeros(1)), (np.array([]), np.array([])),
                     (np.array([np.inf]), np.array([0]))]:
            with self.assertRaises(ValueError):
                reference_metrics(f, o)

    def test_quantization_is_not_recovered_by_wider_arithmetic(self):
        f, o = np.array([280.1], dtype=np.float32), np.array([280.0], dtype=np.float32)
        self.assertGreater(reference_metrics(f, o)[0], .09)
        np.testing.assert_array_equal(reference_metrics(f.astype(np.float16), o.astype(np.float16)), [0, 0, 0])


@unittest.skipIf(torch is None, 'PyTorch not installed')
class TorchPrecisionTest(unittest.TestCase):
    def test_known_answer_all_modes(self):
        for mode in ['fp32', 'fp16_native', 'fp16_fp32_math']:
            dtype = torch.float32 if mode == 'fp32' else torch.float16
            f, o = torch.tensor([1, -2, 3], dtype=dtype), torch.zeros(3, dtype=dtype)
            result = [v.item() for v in precision_kernel(f, o, mode)]
            np.testing.assert_allclose(result, [2/3, 2, np.sqrt(14/3)], rtol=1e-3)

    def test_native_half_square_overflows(self):
        f, o = torch.tensor([3000.], dtype=torch.float16), torch.zeros(1, dtype=torch.float16)
        native = precision_kernel(f, o, 'fp16_native')
        self.assertFalse(torch.isfinite(native[2]).item())
        mixed = [v.item() for v in precision_kernel(f, o, 'fp16_fp32_math')]
        np.testing.assert_allclose(mixed, [3000, 3000, 3000], rtol=1e-6)

    def test_mixed_casts_before_subtraction(self):
        f = torch.tensor([60000.], dtype=torch.float16)
        o = torch.tensor([-60000.], dtype=torch.float16)
        mixed = [v.item() for v in precision_kernel(f, o, 'fp16_fp32_math')]
        np.testing.assert_allclose(mixed, [120000, 120000, 120000], rtol=1e-6)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mixed_matches_quantized_reference(self):
        f = np.array([280.1, 281.7, 279.9], dtype=np.float16)
        o = np.array([280., 281., 280.], dtype=np.float16)
        result = precision_kernel(torch.from_numpy(f).cuda(), torch.from_numpy(o).cuda(), 'fp16_fp32_math')
        np.testing.assert_allclose([v.item() for v in result], reference_metrics(f, o), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
