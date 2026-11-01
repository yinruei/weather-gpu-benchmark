"""Schedule coverage and tail summaries for latency experiments."""
import unittest
import numpy as np
from weather_stability import execution_order, summarize


class StabilityTest(unittest.TestCase):
    def test_schedules_preserve_all_methods_and_repetitions(self):
        methods = ['cpu_once', 'cpu_reuse', 'gpu_once', 'gpu_reuse']
        for mode in ['blocked', 'interleaved']:
            rows = execution_order(methods, mode, 12, 8)
            self.assertEqual(len(rows), 48)
            self.assertEqual({(m, r) for m, r, _ in rows}, {(m, r) for m in methods for r in range(12)})
            self.assertEqual(rows, execution_order(methods, mode, 12, 8))
            if mode == 'blocked':
                for start in range(0, 48, 12):
                    self.assertEqual(len({m for m, _, _ in rows[start:start+12]}), 1)
            else:
                for start in range(0, 48, 4):
                    self.assertEqual({m for m, _, _ in rows[start:start+4]}, set(methods))

    def test_tail_summary_exposes_slow_samples_and_rejects_bad_times(self):
        samples = [dict(seconds=1.) for _ in range(90)]+[dict(seconds=10.) for _ in range(10)]
        result = summarize(samples)
        self.assertEqual(result['p50_sec'], 1.)
        self.assertEqual(result['p95_sec'], 10.)
        self.assertEqual(result['p95_over_p50'], 10.)
        for bad in [[], [dict(seconds=0)], [dict(seconds=np.nan)]]:
            with self.assertRaises(ValueError):
                summarize(bad)


if __name__ == '__main__':
    unittest.main()
