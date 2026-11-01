"""Source selection identity, persistence and output contract corruption checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from physicsnemo_benchmark import load_input, prepare_input, read_input
from physicsnemo_pipeline import input_budget, output_payload, verify_output, write_output


DATA = Path(__file__).resolve().parents[1]/'data/ecmwf_batch_case/batch.zarr'


class PhysicsPipelineTest(unittest.TestCase):
    def test_split_read_prepare_preserves_archived_input_and_native_coordinates(self):
        case = dict(case='b64_s64', batch=64, side=64)
        raw, identity, coordinates = read_input(DATA, case)
        expected, old_identity = load_input(DATA, case)
        np.testing.assert_array_equal(prepare_input(raw), expected)
        self.assertEqual(identity, old_identity)
        self.assertEqual(len({tuple(f.values()) for f in identity['frames']}), 64)
        self.assertGreater(coordinates['latitude'][0], coordinates['latitude'][-1])
        self.assertEqual(identity['frames'][51]['lead_hours'], 30)
        with np.load(DATA.parents[2]/'results/rtx3050_physicsnemo_run01/b64_s64.npz') as saved:
            np.testing.assert_array_equal(expected, saved['input_fp32'])

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_input(np.array([np.nan], dtype=np.float32))

    def test_chunk_budget_distinguishes_requested_and_materialized_data(self):
        budget = input_budget(DATA, dict(batch=64, side=64))
        self.assertEqual(budget['materialized_frames'], 102)
        self.assertEqual(budget['packed_input_bytes'], 1048576)
        self.assertEqual(budget['chunks_touched'], 4)
        self.assertEqual(budget['nominal_chunk_uncompressed_bytes'], 3342336)

    def test_file_roundtrip_rejects_coordinate_dtype_and_value_corruption(self):
        case = dict(case='b1_s32', batch=1, side=32)
        raw, identity, coordinates = read_input(DATA, case)
        output = prepare_input(raw)
        payload = output_payload(output, identity, coordinates, case, 'source', 'state')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'output.npz'
            write_output(path, payload)
            verify_output(path, payload, output.astype(np.float64))
            with self.assertRaises(FileExistsError):
                write_output(path, payload)
            changes = [('latitude', coordinates['latitude'][::-1]), ('model_output', output.astype(np.float64)),
                       ('model_output', output+1.), ('member', np.array([42]))]
            for i, (key, value) in enumerate(changes):
                changed = dict(payload)
                changed[key] = value
                bad = Path(temp)/(str(i)+'.npz')
                write_output(bad, changed)
                with self.assertRaises((ValueError, AssertionError)):
                    verify_output(bad, payload, output.astype(np.float64))


if __name__ == '__main__':
    unittest.main()
