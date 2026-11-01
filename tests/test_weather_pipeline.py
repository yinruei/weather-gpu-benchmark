"""Shared-input regional metrics must match the original dense verification."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_weather_batch import fixture
from weather_batch import numpy_grouped, pack_batch
from weather_pipeline import Prepared, Trace, compute, prepare, regions, run_workflow, spatial_subset, torch_shared


class PipelineTest(unittest.TestCase):
    def expected(self, ds, bounds):
        y0, y1, x0, x1 = bounds
        arrays, _ = pack_batch(ds.isel(latitude=slice(y0, y1), longitude=slice(x0, x1)))
        return numpy_grouped(*(a.astype(np.float64) for a in arrays))

    def test_dense_and_shared_match_each_regional_group(self):
        for missing in [False, True]:
            ds = fixture()
            if missing:
                # Missing one member's full-domain point; other members keep it.
                ds.forecast.values[0, 0, 1, 0, 0] = np.nan
            for kind in ['dense', 'shared']:
                prepared = prepare(ds, kind)
                for name, bounds in regions(2, 2):
                    # A one-cell region with no valid data must be rejected.
                    if missing and name == 'northwest':
                        with self.assertRaises(ValueError):
                            self.expected(ds, bounds)
                        continue
                    with self.subTest(missing=missing, kind=kind, region=name):
                        np.testing.assert_allclose(compute(spatial_subset(prepared, bounds)),
                                                   self.expected(ds, bounds), rtol=1e-5, atol=1e-5)

    def test_shared_buffers_do_not_duplicate_reference_or_cache_outputs(self):
        ds = fixture()
        shared, dense = prepare(ds, 'shared'), prepare(ds, 'dense')
        self.assertEqual(shared.reference.shape, ds.reference.shape)
        self.assertEqual(shared.weights.shape, (2,))
        self.assertIsNone(shared.mask)
        self.assertLess(shared.logical_bytes, dense.logical_bytes)
        before = compute(shared).copy()
        # A new calculation uses changed cached input, not a cached metric result.
        shared.forecast += 1.
        after = compute(shared)
        np.testing.assert_allclose(after[:, 0], before[:, 0]+1., rtol=1e-5)
        np.testing.assert_allclose(compute(dense), before, rtol=1e-5, atol=1e-5)

    def test_reopen_once_and_explicit_reuse_execute_same_tasks(self):
        ds = fixture()
        tasks = regions(2, 2)
        expected = np.stack([self.expected(ds, b) for _, b in tasks])
        case = dict(case='fixture', runs=2, leads=2, members=2, side=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'batch.zarr'
            ds.to_zarr(path, zarr_format=2, consolidated=True, write_empty_chunks=True)
            for mode in ['reopen_dense', 'once_dense', 'once_shared', 'reuse_shared']:
                trace = Trace()
                result = run_workflow(path, case, tasks, mode, cache=prepare(ds, 'shared'), trace=trace)
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
                self.assertTrue(all(s >= 0 for s in trace.seconds.values()))
                self.assertEqual('zarr_read_decode' in trace.seconds, mode != 'reuse_shared')
            with self.assertRaises(ValueError):
                run_workflow(path, case, tasks, 'reuse_shared')

    def test_shared_torch_matches_dense_with_member_specific_missing_values(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional')
        ds = fixture()
        ds.forecast.values[0, 1, 1, 0, 0] = np.nan
        ds.reference.values[1, 0, 1, 1] = np.nan
        shared = prepare(ds, 'shared')
        expected = self.expected(ds, (0, 2, 0, 2))
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            buffers = [torch.from_numpy(v).to(device) if v is not None else None
                       for v in [shared.forecast, shared.reference, shared.weights, shared.mask]]
            prepared = Prepared('shared', shared.shape, *buffers)
            np.testing.assert_allclose(torch_shared(prepared).cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
