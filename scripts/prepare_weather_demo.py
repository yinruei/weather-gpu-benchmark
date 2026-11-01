"""Prepare a real reanalysis subset plus a clearly manufactured forecast field."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import xarray as xr


SOURCE_URL = 'https://raw.githubusercontent.com/pydata/xarray-data/2baf0c22d9671a2058415f17e71b3fe06239a3dd/air_temperature.nc'
SOURCE_SHA256 = 'c606b89c35970a2983b914b76df4adbb409003ef34aa7cfd7f582e41f307482b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Optional cached copy of the source NetCDF')
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    if args.source:
        payload = args.source.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source checksum mismatch; use the pinned, unmodified tutorial NetCDF')
    with xr.open_dataset(io.BytesIO(payload), engine='scipy') as ds:
        source_attrs = dict(ds.attrs)
        ref = ds.air.sel(time=slice('2013-01-01', '2013-01-01T23:59:59'),
                         lat=slice(50, 30), lon=slice(230, 270)).load().astype(np.float32)
    if ref.shape != (4, 9, 17):
        raise ValueError('Unexpected source subset; expected four times and a 9 x 17 grid')
    ref.attrs = dict(units='K', long_name='Air temperature at sigma level 0.995',
                     experiment_role='real NCEP reanalysis subset, FP32, with explicitly injected demo missing cells')
    lat_phase = np.linspace(0, 2*np.pi, 9)[None, :, None]
    lon_phase = np.linspace(0, 2*np.pi, 17)[None, None, :]
    offset = .5 + .8*np.sin(lon_phase) + .4*np.cos(lat_phase) + .1*np.arange(4)[:, None, None]
    forecast = ref.copy(deep=True)
    forecast.values = (ref.values + offset).astype(np.float32)
    forecast.attrs = dict(units='K', long_name='Manufactured forecast for pipeline validation',
                          experiment_role='manufactured perturbation of reanalysis; not an operational forecast')
    forecast.values[0, 0, 0] = np.nan
    forecast.values[:, 2, 3] = np.nan
    ref.values[1, -1, -1] = np.nan
    ref.values[:, 4, 5] = np.nan
    args.output.mkdir(parents=True)
    for name, field in [('reference.nc', ref), ('manufactured_forecast.nc', forecast)]:
        dataset = field.to_dataset(name='air')
        dataset.attrs = dict(title='One-day North America pipeline validation demo',
                             source=SOURCE_URL,
                             attribution='NCEP reanalysis data provided by NOAA PSL, Boulder, Colorado, USA; distributed here as a subset of the xarray tutorial dataset.',
                             scope='Not ERA5, not 2-m temperature, not Taiwan observations, and not an assessment of forecast skill.')
        dataset.to_netcdf(args.output / name, engine='scipy', format='NETCDF3_64BIT')
    manifest = dict(source_url=SOURCE_URL, source_sha256=hashlib.sha256(payload).hexdigest(),
                    source_documentation='https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html',
                    xarray_example='https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html',
                    source_dataset_attributes=source_attrs,
                    selection=dict(date_utc='2013-01-01', lat_bounds=[30, 50], lon_bounds_0_360=[230, 270],
                                   canonical_lon_bounds=[-130, -90], shape=[4, 9, 17]),
                    variable='air', level='sigma 0.995 (not 2 m)', prepared_dtype='float32',
                    forecast='Manufactured; source temperatures plus 0.5 + 0.8*sin(lon_phase) + 0.4*cos(lat_phase) + 0.1*time_index K; phases span 0..2*pi across the selected grid.',
                    injected_missing_cells=dict(forecast=['[time=0,lat=0,lon=0]', '[:,lat=2,lon=3]'],
                                                reference=['[time=1,lat=-1,lon=-1]', '[:,lat=4,lon=5]']),
                    interpretation='Real reference values with artificial forecast errors and missingness; validates software and timing boundaries, not model skill.',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.nc')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved four valid times, 9 x 17 grid cells per time, to {args.output}')


if __name__ == '__main__':
    main()
