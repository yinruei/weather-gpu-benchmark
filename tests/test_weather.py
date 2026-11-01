"""Weather-coordinate, unit and weighted-metric correctness tests."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None
if xr is not None:
    from weather_verify import load_pair, normalize_temperature, numpy_weighted, pack, prepare_pair


@unittest.skipIf(xr is None, 'Install requirements-weather.txt for NetCDF tests')
class WeatherTest(unittest.TestCase):
    def field(self, values, units='K'):
        return xr.DataArray(np.asarray(values, dtype=np.float32).reshape(1, 2, 2),
                            dims=['time', 'lat', 'lon'],
                            coords=dict(time=np.array(['2013-01-01'], dtype='datetime64[ns]'),
                                        lat=[0., 60.], lon=[230., 240.]), attrs=dict(units=units))

    def test_weighted_known_answer_and_joint_mask(self):
        f = self.field([1, np.nan, 3, 100])
        r = self.field([0, 0, 0, np.nan])
        ff, rr, mask, weights = prepare_pair(f, r)
        self.assertEqual(int(mask.sum()), 2)
        result = numpy_weighted(*pack(ff.values, rr.values, mask, weights))
        np.testing.assert_allclose(result, [5/3, 5/3, np.sqrt(11/3)], rtol=1e-6)

    def test_signed_bias_and_rmse(self):
        f = self.field([-1, -1, 1, 1])
        r = self.field([0, 0, 0, 0])
        ff, rr, mask, weights = prepare_pair(f, r, 'uniform')
        np.testing.assert_allclose(numpy_weighted(*pack(ff.values, rr.values, mask, weights)), [0, 1, 1])

    def test_units_order_and_longitude_convention(self):
        f = self.field([0, 1, 2, 3], 'degC').isel(lat=[1, 0], lon=[1, 0]).transpose('lon', 'time', 'lat')
        r = self.field([273.15, 274.15, 275.15, 276.15]).assign_coords(lon=[-130., -120.])
        ff, rr, mask, weights = prepare_pair(f, r)
        np.testing.assert_allclose(numpy_weighted(*pack(ff.values, rr.values, mask, weights)), [0, 0, 0], atol=1e-6)

    def test_time_or_grid_mismatch_rejected(self):
        r = self.field([0, 0, 0, 0])
        for f in [r.assign_coords(time=np.array(['2013-01-02'], dtype='datetime64[ns]')),
                  r.assign_coords(lat=[0., 50.])]:
            with self.assertRaises(ValueError):
                prepare_pair(f, r)

    def test_unsupported_units_dimensions_and_duplicates_rejected(self):
        base = self.field([0, 0, 0, 0])
        for field in [base.assign_attrs(units='unknown'), base.assign_coords(lon=[0, 360]),
                      base.expand_dims(level=[1000]), base.assign_coords(lat=[0, 0])]:
            with self.assertRaises(ValueError):
                normalize_temperature(field)

    def test_all_missing_or_zero_weight_rejected(self):
        base = self.field([0, 0, 0, 0])
        missing = base.copy(deep=True)
        missing.values[:] = np.nan
        with self.assertRaisesRegex(ValueError, 'No joint valid cells'):
            prepare_pair(missing, base)
        pole = base.isel(lat=[0]).assign_coords(lat=[90.])
        with self.assertRaises(ValueError):
            prepare_pair(pole, pole)

    def test_irregular_grid_rejected_for_cosine_weights(self):
        data = xr.concat([self.field([0, 0, 0, 0]).isel(lat=[0])]*3, dim='lat').assign_coords(lat=[0, 10, 30])
        with self.assertRaises(ValueError):
            prepare_pair(data, data)
        prepare_pair(data, data, 'uniform')

    def test_radian_coordinates_rejected(self):
        data = self.field([0, 0, 0, 0])
        data.lat.attrs['units'] = 'radians'
        with self.assertRaisesRegex(ValueError, 'geographic degrees'):
            normalize_temperature(data)

    def test_netcdf_decode_and_fill_value_roundtrip(self):
        field = self.field([280, np.nan, 282, 283], 'degK')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'air.nc'
            field.to_dataset(name='air').to_netcdf(path, engine='scipy',
                                                   encoding={'air': {'_FillValue': -9999.}})
            f, r, mask, weights = load_pair(path, path, 'air', 'air', 'coslat')
            self.assertEqual(int(mask.sum()), 3)
            np.testing.assert_array_equal(numpy_weighted(*pack(f.values, r.values, mask, weights)), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
