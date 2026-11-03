"""Exact paired frame indices, native coordinates and real local Store get coverage."""
from pathlib import Path
import unittest

import numpy as np
import xarray as xr

from physicsnemo_benchmark import prepare_input
from physicsnemo_graph_pipeline import read_variant
from physicsnemo_reader import frame_indices, probe_reads, read_coverage

DATA = Path(__file__).resolve().parents[1]/'data/ecmwf_batch_case/batch.zarr'


class ExactReaderTest(unittest.TestCase):
    def test_exact_frames_cross_member_lead_and_initialization_boundaries(self):
        with xr.open_zarr(DATA, chunks=None, consolidated=True) as ds:
            for batch, side in [(16, 64), (64, 64), (205, 32), (416, 16)]:
                case = dict(case=f'b{batch}_s{side}', batch=batch, side=side)
                for variant in [0, 1, 0]:
                    expected, identity, coords = read_variant(DATA, case, variant, ds)
                    actual, updated, actual_coords = read_variant(DATA, case, variant, ds, selection='exact')
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(prepare_input(actual), prepare_input(expected))
                    self.assertEqual(updated, identity)
                    for name in coords:
                        np.testing.assert_array_equal(actual_coords[name], coords[name])
                    self.assertEqual(len({tuple(f.values()) for f in updated['frames']}), batch)
        for batch in [16, 64]:
            c = dict(batch=batch, side=64)
            a, identity, _ = read_variant(DATA, c, 1, selection='exact')
            b, expected, _ = read_variant(DATA, c, 1)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(identity, expected)

    def test_exact_selection_reduces_materialization_but_not_fixture_chunks(self):
        for batch, expected_frames in [(16, [16, 32]), (64, [102, 153])]:
            for variant in [0, 1]:
                c = dict(batch=batch, side=64)
                old = read_coverage(DATA, c, variant, 'bounding')
                exact = read_coverage(DATA, c, variant, 'exact')
                self.assertEqual(old['materialized_frames'], expected_frames[variant])
                self.assertEqual(exact['materialized_frames'], batch)
                self.assertEqual(exact['chunk_keys'], old['chunk_keys'])
                self.assertEqual(exact['chunks_touched'], 4)
                self.assertEqual(exact['stored_chunk_bytes'], old['stored_chunk_bytes'])
                self.assertEqual(exact['nominal_chunk_uncompressed_bytes'], 3342336)

    def test_store_probe_observes_full_compressed_chunks_and_coordinate_reads(self):
        for selection in ['bounding', 'exact']:
            probe = probe_reads(DATA, dict(batch=64, side=64), 1, selection)
            self.assertEqual(probe['forecast_gets'], 4)
            self.assertEqual(probe['forecast_returned_bytes'], 833260)
            self.assertTrue(any(r['key'].startswith('latitude/') and r['found'] for r in probe['records']))
            self.assertEqual(probe['identity']['frames'][0]['member'], 13)

    def test_rejects_out_of_bounds_and_wrong_selection(self):
        for batch, variant in [(0, 0), (1.5, 0), (True, 0), (64, 2), (1000, 1)]:
            with self.assertRaises(ValueError):
                frame_indices((4, 8, 51), batch, variant)
        with self.assertRaises(ValueError):
            read_coverage(DATA, dict(batch=16, side=65), 0, 'exact')
        with self.assertRaises(ValueError):
            read_variant(DATA, dict(batch=16, side=64), 0, selection='typo')


if __name__ == '__main__':
    unittest.main()
