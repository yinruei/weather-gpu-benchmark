"""Trace accounting and one-return regional output parity."""
import unittest
import numpy as np

from test_weather_batch import fixture
from weather_pipeline import Prepared, prepare, regions, compute, spatial_subset
from weather_dispatch import merged_duration, return_regions, summarize_trace


class DispatchTest(unittest.TestCase):
    def test_trace_busy_time_is_union_not_sum_of_concurrent_events(self):
        self.assertEqual(merged_duration([(0, 4), (2, 6), (8, 12)], 1, 10), 7.)
        def event(category, name, start, duration):
            return dict(ph='X', cat=category, name=name, ts=start, dur=duration)
        document = {'traceEvents': [event('user_annotation', 'call/blocked/cuda_reuse_once/0', 0, 1000),
                                   event('user_annotation', 'dispatch/full', 0, 300),
                                   event('user_annotation', 'return/all', 300, 600),
                                   event('kernel', 'a', 50, 200), event('kernel', 'b', 150, 200),
                                   event('gpu_memcpy', 'Memcpy DtoH', 800, 50)]}
        row = summarize_trace(document)[0]
        self.assertAlmostEqual(row['gpu_kernel_busy_ms'], .3)
        self.assertAlmostEqual(row['gpu_active_union_ms'], .35)
        self.assertEqual(row['dtoh_copies'], 1)
        self.assertAlmostEqual(row['return_range_ms'], .6)

    def test_one_return_preserves_all_region_group_metrics_and_missing_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional')
        ds = fixture()
        shared = prepare(ds, 'shared')
        tasks = regions(2, 2)
        expected = np.stack([compute(spatial_subset(shared, bounds)) for _, bounds in tasks])
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            values = [torch.from_numpy(a).to(device) if a is not None else None for a in
                      [shared.forecast, shared.reference, shared.weights, shared.mask]]
            tensor = Prepared('shared', shared.shape, *values)
            for once in [False, True]:
                np.testing.assert_allclose(return_regions(tensor, tasks, once), expected, rtol=1e-5, atol=1e-5)
        ds.forecast.values[0, 0, 1, 0, 0] = np.nan
        shared = prepare(ds, 'shared')
        values = [torch.from_numpy(a) if a is not None else None for a in
                  [shared.forecast, shared.reference, shared.weights, shared.mask]]
        tensor = Prepared('shared', shared.shape, *values)
        np.testing.assert_allclose(return_regions(tensor, tasks[:1], True)[0], compute(shared), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
