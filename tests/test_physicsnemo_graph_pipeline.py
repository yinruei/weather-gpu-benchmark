"""Reader updates and real file contract for selectable pipeline engines."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from physicsnemo_benchmark import prepare_input
from physicsnemo_graph import CASES, load_variant
from physicsnemo_graph_pipeline import ENGINES, READERS, FnoFilePipeline, methods, read_variant, request_order
from physicsnemo_reader import READER_POLICIES
from physicsnemo_pipeline import verify_output


DATA = Path(__file__).resolve().parents[1]/'data/ecmwf_batch_case/batch.zarr'


class GraphPipelineTest(unittest.TestCase):
    def test_both_readers_follow_new_variant_identity_without_stale_frames(self):
        with xr.open_zarr(DATA, chunks=None, consolidated=True) as dataset:
            for case in CASES:
                for variant in [0, 1, 0]:
                    expected, identity = load_variant(DATA, case, variant)
                    for source in [None, dataset]:
                        raw, actual, coordinates = read_variant(DATA, case, variant, source)
                        np.testing.assert_array_equal(prepare_input(raw), expected)
                        self.assertEqual(actual, identity)
                        self.assertEqual(len(coordinates['latitude']), case['side'])
        with self.assertRaises(ValueError):
            read_variant(DATA, CASES[0], 2)

    def test_engine_reader_pairing_and_single_engine_selection(self):
        for engines, readers in [(ENGINES, READERS), (['graph'], ['open_once'])]:
            chosen = methods(engines, readers)
            jobs = request_order(engines, readers, 4, 42)
            self.assertEqual(jobs, request_order(engines, readers, 4, 42))
            for repeat in range(4):
                selected = [j for j in jobs if j['repeat'] == repeat]
                self.assertEqual({(j['case'], j['method']) for j in selected}, {(c['case'], m) for c in CASES for m in chosen})
                self.assertEqual(len(selected), len(CASES)*len(chosen))
                self.assertTrue(all(j['variant'] == repeat % 2 for j in selected))
        for engines, readers in [([], READERS), (['graph', 'graph'], READERS), (['unknown'], READERS), (ENGINES, ['bad'])]:
            with self.assertRaises(ValueError):
                methods(engines, readers)

    def test_optional_cpu_file_engine_roundtrip_and_closed_reader_rejection(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional')
        model = torch.nn.Identity().eval()
        with tempfile.TemporaryDirectory() as temp:
            for reader in READERS:
                with FnoFilePipeline(DATA, CASES[0], model, engine='cpu', reader=reader) as pipeline:
                    for index, variant in enumerate([0, 1, 0]):
                        path = Path(temp)/f'{reader}_{index}.npz'
                        seconds, x, identity, payload = pipeline.execute(variant, path, 'source', 'state')
                        self.assertGreater(seconds, 0)
                        expected, expected_identity = load_variant(DATA, CASES[0], variant)
                        np.testing.assert_array_equal(x, expected)
                        self.assertEqual(identity, expected_identity)
                        verify_output(path, payload, expected.astype(np.float64))
                if READER_POLICIES[reader]['reuse']:
                    with self.assertRaises(ValueError):
                        pipeline.execute(0, Path(temp)/'closed.npz', 'source', 'state')
        with self.assertRaises(ValueError):
            FnoFilePipeline(DATA, CASES[0], model, engine='graph')


if __name__ == '__main__':
    unittest.main()
