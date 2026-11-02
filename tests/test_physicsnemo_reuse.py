"""Reuse must preserve changing selections, source identity and measurement pairing."""
from pathlib import Path
import unittest

import numpy as np
import xarray as xr

from physicsnemo_benchmark import prepare_input, read_from_dataset, read_input
from physicsnemo_reuse import CONDITIONS, WORKFLOWS, job_order


DATA = Path(__file__).resolve().parents[1]/'data/ecmwf_batch_case/batch.zarr'


class ReuseTest(unittest.TestCase):
    def test_reused_handle_reads_distinct_cases_without_stale_input(self):
        with xr.open_zarr(DATA, chunks=None, consolidated=True) as dataset:
            for batch, side in [(16, 64), (1, 32), (64, 64), (16, 64)]:
                case = dict(batch=batch, side=side)
                raw, identity, coords = read_from_dataset(dataset, case)
                expected, expected_identity, expected_coords = read_input(DATA, case)
                np.testing.assert_array_equal(prepare_input(raw), prepare_input(expected))
                self.assertEqual(identity, expected_identity)
                for dim in coords:
                    np.testing.assert_array_equal(coords[dim], expected_coords[dim])
                self.assertEqual(len(identity['frames']), batch)

    def test_pairing_covers_every_case_method_once_per_repeat(self):
        cases = [dict(case='b16_s64'), dict(case='b64_s64')]
        for methods in [WORKFLOWS, CONDITIONS]:
            jobs = job_order(cases, methods, 4, 42)
            self.assertEqual(jobs, job_order(cases, methods, 4, 42))
            self.assertNotEqual(jobs, job_order(cases, methods, 4, 43))
            for repeat in range(4):
                selected = [(name, method) for r, _, name, method in jobs if r == repeat]
                self.assertEqual(len(selected), 2*len(methods))
                self.assertEqual(set(selected), {(c['case'], m) for c in cases for m in methods})


if __name__ == '__main__':
    unittest.main()
