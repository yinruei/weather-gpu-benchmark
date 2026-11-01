"""Local Zarr parity, provenance, and exact forecast-axis selection tests."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None
try:
    import zarr
except ImportError:
    zarr = None
if xr is not None:
    from weather_verify import load_pair, numpy_weighted, pack, sha256
    from scripts.prepare_ecmwf_zarr import canonical_forecast, chunk_estimate, save_store, select_forecast


@unittest.skipIf(xr is None, 'Install requirements-weather.txt')
class ForecastSelectionTest(unittest.TestCase):
    def source(self):
        inits = np.array(['2024-05-06', '2024-05-07'], dtype='datetime64[ns]')
        leads = np.array([24, 30], dtype='timedelta64[h]')
        values = np.arange(2*2*2*3*3, dtype=np.float32).reshape(2, 2, 2, 3, 3)
        ds = xr.Dataset({'temperature_2m': (('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude'), values)},
                        coords=dict(init_time=inits, lead_time=leads, ensemble_member=[0, 1],
                                    latitude=[26., 25.75, 25.5], longitude=[116., 116.25, 116.5],
                                    valid_time=(('init_time', 'lead_time'), inits[:, None]+leads[None, :])))
        ds.temperature_2m.attrs.update(units='degree_Celsius', step_type='instant')
        ds.temperature_2m.encoding['chunks'] = (1, 2, 2, 2, 2)
        return ds

    def test_exact_init_lead_member_and_valid_time(self):
        ds = self.source()
        initial = np.datetime64('2024-05-06', 'ns')
        selected = select_forecast(ds, initial, [24], 0, [25.75, 26], [116, 116.25])
        out = canonical_forecast(selected, initial, [24])
        np.testing.assert_array_equal(out.values[0], ds.temperature_2m.values[0, 0, 0, :2, :2])
        np.testing.assert_array_equal(out.time.values, [np.datetime64('2024-05-07')])
        self.assertEqual(int(out.ensemble_member), 0)

    def test_wrong_member_units_or_valid_time_rejected(self):
        initial = np.datetime64('2024-05-06', 'ns')
        with self.assertRaises(ValueError):
            select_forecast(self.source(), initial, [24], 1, [25.5, 26], [116, 116.5])
        wrong_units = self.source()
        wrong_units.temperature_2m.attrs['units'] = 'K'
        with self.assertRaises(ValueError):
            select_forecast(wrong_units, initial, [24], 0, [25.5, 26], [116, 116.5])
        wrong_time = self.source()
        wrong_time.coords['valid_time'] = wrong_time.valid_time+np.timedelta64(1, 'h')
        with self.assertRaises(AssertionError):
            select_forecast(wrong_time, initial, [24], 0, [25.5, 26], [116, 116.5])

    def test_chunk_budget_counts_physical_chunks(self):
        ds = self.source()
        selected = select_forecast(ds, np.datetime64('2024-05-06', 'ns'), [24], 0, [25.75, 26], [116, 116.25])
        estimate = chunk_estimate(ds.temperature_2m, selected)
        self.assertEqual(estimate['spatial_chunks'], 1)
        self.assertEqual(estimate['nominal_uncompressed_source_chunk_bytes'], 64)
        self.assertEqual(estimate['selected_logical_bytes'], 16)

    def test_directory_hash_is_path_independent_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.mkdir(); b.mkdir()
            for name in ['chunk', '.zattrs']: (a/name).write_bytes(name.encode())
            for name in ['.zattrs', 'chunk']: (b/name).write_bytes(name.encode())
            self.assertEqual(sha256(a), sha256(b))
            (b/'chunk').write_bytes(b'changed')
            self.assertNotEqual(sha256(a), sha256(b))


@unittest.skipIf(xr is None or zarr is None, 'Install requirements-zarr.txt')
class ZarrParityTest(unittest.TestCase):
    def test_zero_run_time_and_control_member_chunks_are_explicit(self):
        initial = np.datetime64('2024-05-06', 'ns')
        source = ForecastSelectionTest().source()
        selected = select_forecast(source, initial, [24, 30], 0, [25.75, 26], [116, 116.25])
        field = canonical_forecast(selected, initial, [24, 30])
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)/'control.zarr'
            save_store(store, field)
            self.assertTrue((store/'forecast_reference_time'/'0').is_file())
            self.assertTrue((store/'ensemble_member'/'0').is_file())
            with xr.open_zarr(store, chunks=None) as ds:
                np.testing.assert_array_equal(ds.forecast_reference_time.values, np.full(2, initial))
                self.assertEqual(int(ds.ensemble_member), 0)

    def test_netcdf_zarr_units_missing_values_and_metrics_match(self):
        data = xr.DataArray(np.array([0., 1., 2., np.nan], dtype=np.float32).reshape(1, 2, 2),
                            dims=['time', 'lat', 'lon'],
                            coords=dict(time=np.array(['2024-05-07'], dtype='datetime64[ns]'),
                                        lat=[24., 23.75], lon=[120., 120.25]), attrs=dict(units='degC'))
        reference = data.copy(deep=True)
        reference.values = np.array([273.15, 273.15, np.nan, 273.15], dtype=np.float32).reshape(1, 2, 2)
        reference.attrs['units'] = 'K'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name, field in [('f', data), ('r', reference)]:
                field.to_dataset(name='air').to_netcdf(p/(name+'.nc'), engine='scipy')
                field.to_dataset(name='air').to_zarr(p/(name+'.zarr'), zarr_format=2, consolidated=True)
            a = load_pair(p/'f.nc', p/'r.nc', 'air', 'air', 'coslat')
            b = load_pair(p/'f.zarr', p/'r.zarr', 'air', 'air', 'coslat')
            for va, vb in zip(a, b): np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))
            result_a = numpy_weighted(*pack(a[0].values, a[1].values, a[2], a[3]))
            result_b = numpy_weighted(*pack(b[0].values, b[1].values, b[2], b[3]))
            np.testing.assert_array_equal(result_a, result_b)
            self.assertEqual(int(b[2].sum()), 2)


if __name__ == '__main__':
    unittest.main()
