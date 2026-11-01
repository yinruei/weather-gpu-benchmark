"""Group identity, missing weights and batched CPU/CUDA numerical agreement."""
import unittest
import numpy as np
import xarray as xr

from weather_batch import benchmark_methods, numpy_grouped, pack_batch, torch_grouped, validate_batch
from weather_verify import numpy_weighted, pack, prepare_pair


def fixture():
    inits = np.array(['2024-05-06', '2024-05-07'], dtype='datetime64[ns]')
    leads = np.array([24, 30], dtype='timedelta64[h]')
    valid = inits[:, None]+leads[None, :]
    f = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)+10
    r = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)+8
    ds = xr.Dataset({'forecast': (('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude'), f),
                     'reference': (('init_time', 'lead_time', 'latitude', 'longitude'), r)},
                    coords=dict(init_time=inits, lead_time=leads, ensemble_member=[0, 1],
                                latitude=[24., 23.75], longitude=[120., 120.25],
                                valid_time=(('init_time', 'lead_time'), valid),
                                reference_initialization_time=(('init_time', 'lead_time'), valid),
                                reference_lead_hours=0, height_above_ground=2.))
    for name in ['forecast', 'reference']:
        ds[name].attrs['units'] = 'degree_Celsius'
    ds.latitude.attrs['units'] = 'degrees_north'
    ds.longitude.attrs['units'] = 'degrees_east'
    return ds


class BatchTest(unittest.TestCase):
    def test_each_group_matches_existing_single_forecast_pipeline(self):
        ds = fixture()
        ds.forecast.values[0, 1, 1, 0, 0] = np.nan
        ds.reference.values[1, 0, 1, 1] = np.nan
        arrays, counts = pack_batch(ds)
        actual = numpy_grouped(*(a.astype(np.float64) for a in arrays))
        group = 0
        for i in range(2):
            for lead in range(2):
                for member in range(2):
                    values = []
                    for name in ['forecast', 'reference']:
                        selected = ds[name].isel(init_time=i, lead_time=lead)
                        if name == 'forecast':
                            selected = selected.isel(ensemble_member=member)
                        field = xr.DataArray(selected.values[None, :, :], dims=['time', 'lat', 'lon'],
                                             coords=dict(time=[ds.valid_time.values[i, lead]],
                                                         lat=ds.latitude.values, lon=ds.longitude.values),
                                             attrs=dict(units='degC'))
                        values.append(field)
                    f, r, mask, weights = prepare_pair(*values)
                    packed = pack(f.values, r.values, mask, weights)
                    expected = numpy_weighted(*(a.astype(np.float64) for a in packed))
                    np.testing.assert_allclose(actual[group], expected, atol=1e-12)
                    self.assertEqual(counts[group], mask.sum())
                    group += 1
        self.assertEqual(actual.shape, (8, 3))

    def test_misaligned_reference_or_invalid_units_rejected(self):
        for change in ['time', 'units', 'empty_group', 'member', 'grid']:
            ds = fixture()
            if change == 'time':
                ds.coords['reference_initialization_time'] = ds.valid_time+np.timedelta64(6, 'h')
            elif change == 'units':
                ds.reference.attrs['units'] = 'K'
            elif change == 'empty_group':
                ds.forecast.values[0, 0, 0] = np.nan
            elif change == 'member':
                ds = ds.assign_coords(ensemble_member=[0, 51])
            else:
                ds.latitude.attrs['units'] = 'metres'
            with self.subTest(change=change), self.assertRaises((ValueError, AssertionError)):
                pack_batch(ds)

    def test_methods_are_interleaved_and_every_output_is_validated(self):
        calls = []
        expected = np.array([[1., 2., 3.]])
        methods = {name: (lambda n=name: calls.append(n) or expected.copy()) for name in ['a', 'b', 'c']}
        summaries, rows = benchmark_methods(methods, expected, 6, 1, 42)
        self.assertEqual(len(rows), 18)
        for repeat in range(6):
            self.assertEqual({r['method'] for r in rows if r['repeat'] == repeat}, set(methods))
        self.assertGreater(len({tuple(r['method'] for r in rows if r['repeat'] == i) for i in range(6)}), 1)
        bad = {'bad': lambda: np.array([[1., 2., 9.]])}
        with self.assertRaises(AssertionError):
            benchmark_methods(bad, expected, 2, 1, 42)

    def test_torch_grouped_matches_numpy(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional')
        arrays, _ = pack_batch(fixture())
        expected = numpy_grouped(*(a.astype(np.float64) for a in arrays))
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            result = torch_grouped(*(torch.from_numpy(a).to(device) for a in arrays)).cpu().numpy()
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
