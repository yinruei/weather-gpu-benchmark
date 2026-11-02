"""Genuine new-input windows, fair updates, graph buffer lifetime and rejection."""
from pathlib import Path
import unittest

import numpy as np

from physicsnemo_benchmark import load_input
from physicsnemo_graph import CASES, METHODS, CudaGraphRunner, load_variant, request_order


DATA = Path(__file__).resolve().parents[1]/'data/ecmwf_batch_case/batch.zarr'


class GraphTest(unittest.TestCase):
    def test_variants_are_disjoint_real_frames_and_preserve_original_A(self):
        for case in CASES:
            a, ia = load_variant(DATA, case, 0)
            b, ib = load_variant(DATA, case, 1)
            expected, old = load_input(DATA, case)
            np.testing.assert_array_equal(a, expected)
            self.assertEqual(ia, old)
            self.assertFalse(np.array_equal(a, b))
            self.assertFalse({tuple(f.values()) for f in ia['frames']} & {tuple(f.values()) for f in ib['frames']})
            self.assertEqual(a.shape, b.shape)
        with self.assertRaises(ValueError):
            load_variant(DATA, CASES[0], 2)

    def test_pairs_use_same_variant_and_preserve_resident_boundary(self):
        jobs = request_order(4, 42)
        self.assertEqual(jobs, request_order(4, 42))
        self.assertNotEqual(jobs, request_order(4, 43))
        for repeat in range(4):
            selected = [j for j in jobs if j['repeat'] == repeat]
            self.assertEqual({(j['case'], j['method']) for j in selected}, {(c['case'], m) for c in CASES for m in METHODS})
            self.assertEqual(len(selected), len(CASES)*len(METHODS))
            for job in selected:
                self.assertEqual(job['variant'], 0 if job['method'].endswith('_resident') else repeat % 2)

    def test_optional_real_cuda_graph_updates_and_strict_input_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CUDA-enabled PyTorch optional')
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for graph capture')
        model = torch.nn.Linear(2, 2, bias=False).cuda().eval()
        with torch.no_grad():
            model.weight.copy_(torch.diag(torch.tensor([2., 3.], device='cuda')))
        a, b = torch.zeros(2, 2, device='cuda'), torch.ones(2, 2)
        runner = CudaGraphRunner(model, a)
        pointers = (runner.input.data_ptr(), runner.output.data_ptr())
        for value in [a, b, a]:
            runner.copy_input(value)
            actual = runner.replay().cpu()
            torch.testing.assert_close(actual, model(value.cuda()).detach().cpu())
            self.assertEqual(pointers, (runner.input.data_ptr(), runner.output.data_ptr()))
        for invalid in [torch.ones(3, 2), torch.ones(2, 2, dtype=torch.float64), torch.ones(2, 2).t()]:
            with self.assertRaises(ValueError):
                runner.copy_input(invalid)
        with self.assertRaises(ValueError):
            CudaGraphRunner(model.train(), a)


if __name__ == '__main__':
    unittest.main()
