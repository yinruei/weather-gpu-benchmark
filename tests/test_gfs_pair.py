"""Offline tests for indexed fields, valid-time pairing and GRIB metadata guards."""
from datetime import datetime, timezone
import unittest
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None
if xr is not None:
    from scripts.prepare_gfs_pair import check_decoded, pair_times, select_message, validate_grib


@unittest.skipIf(xr is None, 'Install requirements-weather.txt')
class GFSPairTest(unittest.TestCase):
    initial = datetime(2024, 5, 6, tzinfo=timezone.utc)

    def test_matching_forecast_and_reference_valid_times(self):
        pairs = pair_times(self.initial, [42, 24])
        self.assertEqual([p[1] for p in pairs], [24, 42])
        self.assertEqual([p[2].isoformat() for p in pairs],
                         ['2024-05-07T00:00:00+00:00', '2024-05-07T18:00:00+00:00'])
        for init, lead, valid in pairs:
            self.assertEqual(valid.hour % 6, 0)
            self.assertEqual((valid-init).total_seconds(), lead*3600)

    def test_invalid_cycles_and_leads_rejected(self):
        for init, leads in [(datetime(2024, 5, 6), [24]),
                            (self.initial.replace(hour=1), [24]),
                            (self.initial, [0]), (self.initial, [25]), (self.initial, [24, 24])]:
            with self.assertRaises(ValueError):
                pair_times(init, leads)

    def test_index_selects_exact_parameter_level_and_step(self):
        index = ('1:0:d=2024050600:TMP:surface:24 hour fcst:\n'
                 '2:100:d=2024050600:TMP:2 m above ground:24 hour fcst:\n'
                 '3:240:d=2024050600:RH:2 m above ground:24 hour fcst:\n')
        start, end, line = select_message(index, self.initial, 24)
        self.assertEqual((start, end), (100, 239))
        self.assertIn(':TMP:2 m above ground:', line)
        with self.assertRaises(ValueError):
            select_message(index, self.initial, 30)

    def test_ambiguous_or_unbounded_message_rejected(self):
        line = '1:0:d=2024050600:TMP:2 m above ground:24 hour fcst:\n'
        for index in [line, line+'2:100:d=2024050600:TMP:2 m above ground:24 hour fcst:\n']:
            with self.assertRaises(ValueError):
                select_message(index, self.initial, 24)

    def test_grib_framing_and_declared_length(self):
        data = b'GRIB'+b'\x00\x00\x00\x02'+(20).to_bytes(8, 'big')+b'7777'
        validate_grib(data)
        for invalid in [b'not GRIB', data[:-1], data[:8]+(21).to_bytes(8, 'big')+data[16:]]:
            with self.assertRaises(ValueError):
                validate_grib(invalid)

    def decoded(self):
        return xr.Dataset({'t2m': (('latitude', 'longitude'), np.ones((2, 2)),
                                  dict(units='K', GRIB_typeOfLevel='heightAboveGround',
                                       GRIB_stepType='instant', GRIB_gridType='regular_ll'))},
                          coords=dict(latitude=[24., 23.75], longitude=[120., 120.25],
                                      time=np.datetime64('2024-05-06'), step=np.timedelta64(24, 'h'),
                                      valid_time=np.datetime64('2024-05-07'), heightAboveGround=2.))

    def test_decoded_metadata_matches_request(self):
        self.assertEqual(check_decoded(self.decoded(), self.initial, 24), np.datetime64('2024-05-07'))
        for data in [self.decoded().assign_coords(valid_time=np.datetime64('2024-05-08')),
                     self.decoded().assign_coords(step=np.timedelta64(30, 'h')),
                     self.decoded().assign_coords(heightAboveGround=10.)]:
            with self.assertRaises(ValueError):
                check_decoded(data, self.initial, 24)

    def test_averaged_or_wrong_level_type_rejected(self):
        for key, value in [('GRIB_stepType', 'avg'), ('GRIB_typeOfLevel', 'isobaricInhPa')]:
            data = self.decoded()
            data.t2m.attrs[key] = value
            with self.assertRaises(ValueError):
                check_decoded(data, self.initial, 24)


if __name__ == '__main__':
    unittest.main()
