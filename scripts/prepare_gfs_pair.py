"""Fetch real GFS 2-m forecasts and matching f000 analysis/initialization fields.

Only the indexed TMP/2-m GRIB2 message is downloaded with HTTP Range.
No manufactured errors, masking, interpolation or regridding are applied.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr


BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pair_times(initialization, leads):
    if initialization.tzinfo is None:
        raise ValueError('Initialization must be timezone-aware UTC')
    initialization = initialization.astimezone(timezone.utc)
    if initialization.hour % 6 or initialization.minute or initialization.second or initialization.microsecond:
        raise ValueError('GFS initialization must be a 00/06/12/18 UTC cycle')
    if not leads or len(set(leads)) != len(leads) or any(h <= 0 or h % 6 for h in leads):
        raise ValueError('Forecast leads must be unique positive multiples of six hours')
    return [(initialization, h, initialization+timedelta(hours=h)) for h in sorted(leads)]


def select_message(index, initialization, lead):
    entries = [line.split(':') for line in index.splitlines() if line.strip()]
    step = 'anl' if lead == 0 else f'{lead} hour fcst'
    matches = [i for i, parts in enumerate(entries)
               if len(parts) >= 6 and parts[2] == initialization.strftime('d=%Y%m%d%H')
               and parts[3:6] == ['TMP', '2 m above ground', step]]
    if len(matches) != 1:
        raise ValueError(f'Expected one instantaneous TMP/2-m/{step} message, found {len(matches)}')
    i = matches[0]
    start = int(entries[i][1])
    next_offsets = [int(parts[1]) for parts in entries[i+1:] if int(parts[1]) > start]
    if not next_offsets:
        raise ValueError('No bounded next-message offset; refusing an unbounded download')
    end = next_offsets[0]-1
    if start < 0 or end < start or end-start+1 > 16*1024**2:
        raise ValueError('Invalid or unexpectedly large message byte range')
    return start, end, ':'.join(entries[i])


def validate_grib(data):
    if len(data) < 20 or data[:4] != b'GRIB' or data[7] != 2 or data[-4:] != b'7777':
        raise ValueError('Selected response is not one complete GRIB2 message')
    if int.from_bytes(data[8:16], 'big') != len(data):
        raise ValueError('GRIB2 declared length does not match response')


def fetch_field(initialization, lead, cache):
    key = initialization.strftime('%Y%m%d_%H')+f'_f{lead:03d}'
    url = (f'{BASE}/gfs.{initialization:%Y%m%d}/{initialization:%H}/atmos/'
           f'gfs.t{initialization:%H}z.pgrb2.0p25.f{lead:03d}')
    grib, index_path, meta_path = [cache/(key+suffix) for suffix in ['.grib2', '.idx', '.json']]
    if all(p.exists() for p in (grib, index_path, meta_path)):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        payload, index_bytes = grib.read_bytes(), index_path.read_bytes()
        if meta['url'] != url or digest(payload) != meta['grib_sha256'] or digest(index_bytes) != meta['index_sha256']:
            raise ValueError(f'Cache integrity mismatch for {key}')
        start, end, selected = select_message(index_bytes.decode('utf-8'), initialization, lead)
        if meta['byte_range'] != [start, end] or len(payload) != end-start+1:
            raise ValueError('Cached range differs from the checked index')
        validate_grib(payload)
        return grib, meta
    print(f'Downloading {key}: TMP at 2 m', flush=True)
    with urlopen(url+'.idx', timeout=30) as response:
        index_bytes = response.read(512*1024)
    start, end, selected = select_message(index_bytes.decode('utf-8'), initialization, lead)
    request = Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urlopen(request, timeout=60) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or not content_range.startswith(f'bytes {start}-{end}/'):
            raise ValueError('Server did not honor the requested range; refusing a whole-file download')
        payload = response.read(end-start+2)
        etag, modified = response.headers.get('ETag'), response.headers.get('Last-Modified')
    if len(payload) != end-start+1:
        raise ValueError('HTTP range length mismatch')
    validate_grib(payload)
    meta = dict(url=url, index_url=url+'.idx', selected_index_line=selected,
                byte_range=[start, end], downloaded_bytes=len(payload),
                grib_sha256=digest(payload), index_sha256=digest(index_bytes),
                etag=etag, last_modified=modified,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    grib.write_bytes(payload)
    index_path.write_bytes(index_bytes)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return grib, meta


def check_decoded(dataset, initialization, lead):
    expected_init = np.datetime64(initialization.replace(tzinfo=None), 'ns')
    expected_valid = expected_init+np.timedelta64(lead, 'h')
    if np.asarray(dataset.time.values).astype('datetime64[ns]') != expected_init:
        raise ValueError('Decoded GRIB initialization differs from the request')
    if np.asarray(dataset.step.values).astype('timedelta64[ns]') != np.timedelta64(lead, 'h'):
        raise ValueError('Decoded GRIB lead differs from the request')
    if np.asarray(dataset.valid_time.values).astype('datetime64[ns]') != expected_valid:
        raise ValueError('Decoded valid time does not equal initialization plus lead')
    if float(dataset.heightAboveGround.values) != 2.:
        raise ValueError('Decoded level is not 2 m above ground')
    field = dataset['t2m']
    if field.attrs.get('units') != 'K' or field.attrs.get('GRIB_typeOfLevel') != 'heightAboveGround':
        raise ValueError('Unexpected temperature units or vertical coordinate type')
    if field.attrs.get('GRIB_stepType') != 'instant' or field.attrs.get('GRIB_gridType') != 'regular_ll':
        raise ValueError('Require instantaneous temperature on a regular latitude/longitude grid')
    return expected_valid


def decode_field(path, initialization, lead, lat_bounds, lon_bounds):
    with xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}) as dataset:
        valid = check_decoded(dataset, initialization, lead)
        field = dataset.t2m.sel(latitude=slice(lat_bounds[1], lat_bounds[0]),
                               longitude=slice(*lon_bounds)).load()
        if set(field.dims) != {'latitude', 'longitude'} or min(field.shape) == 0:
            raise ValueError('Empty or unexpected grid after geographic selection')
        out = xr.DataArray(field.transpose('latitude', 'longitude').values[None].astype(np.float32),
                           dims=['time', 'lat', 'lon'],
                           coords=dict(time=[valid], lat=field.latitude.values, lon=field.longitude.values,
                                       forecast_reference_time=('time', [np.datetime64(initialization.replace(tzinfo=None), 'ns')]),
                                       lead_hours=('time', [lead]), height_above_ground=2.),
                           name='air', attrs=dict(units='K', standard_name='air_temperature', long_name='2 m air temperature'))
    out.lat.attrs['units'] = 'degrees_north'
    out.lon.attrs['units'] = 'degrees_east'
    out.height_above_ground.attrs.update(units='m', positive='up')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', default='2024-05-06T00:00:00+00:00')
    parser.add_argument('--leads', type=int, nargs='+', default=[24, 30, 36, 42])
    parser.add_argument('--lat', type=float, nargs=2, default=[18., 28.])
    parser.add_argument('--lon', type=float, nargs=2, default=[115., 125.])
    parser.add_argument('--cache', type=Path, default=Path('results/latest/gfs_download'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory; existing data are not overwritten')
    if not (-90 <= args.lat[0] < args.lat[1] <= 90 and 0 <= args.lon[0] < args.lon[1] < 360):
        parser.error('Use ascending lat bounds and non-wrapping 0..360 longitude bounds')
    pairs = pair_times(datetime.fromisoformat(args.init.replace('Z', '+00:00')), args.leads)
    args.cache.mkdir(parents=True, exist_ok=True)
    forecasts, references, sources = [], [], []
    for initial, lead, valid in pairs:
        for role, init, hour, collection in [('forecast', initial, lead, forecasts), ('reference', valid, 0, references)]:
            path, meta = fetch_field(init, hour, args.cache)
            field = decode_field(path, init, hour, args.lat, args.lon)
            collection.append(field)
            sources.append(dict(role=role, initialization_utc=init.isoformat(), lead_hours=hour,
                                valid_time_utc=valid.isoformat(), **meta))
    forecast = xr.concat(forecasts, dim='time', join='exact')
    reference = xr.concat(references, dim='time', join='exact')
    # Indexes must agree before writing; the different initialization/lead coords are intentional.
    xr.align(forecast, reference, join='exact')
    forecast.attrs['experiment_role'] = 'real operational GFS 2-m temperature forecast; original values, no manufactured perturbation'
    reference.attrs['experiment_role'] = 'real GFS f000 analysis/initialization-time 2-m temperature; same model system, not independent station observations'
    args.output.mkdir(parents=True)
    for name, field in [('forecast.nc', forecast), ('reference.nc', reference)]:
        dataset = field.to_dataset(name='air')
        dataset.attrs.update(source='NOAA NCEP GFS via NOAA Open Data on AWS',
                             description='Real forecast/analysis pair. See manifest.json for source messages and valid-time matching.',
                             manufactured_perturbations='none', interpolation='none')
        dataset.to_netcdf(args.output/name, engine='scipy', format='NETCDF3_64BIT')
    manifest = dict(case_kind='gfs_forecast_vs_f000', title='GFS 2-m temperature forecast vs matching GFS f000 analysis',
                    source_registry='https://registry.opendata.aws/noaa-gfs-bdp-pds/',
                    product_documentation='https://www.nco.ncep.noaa.gov/pmb/products/gfs/',
                    forecast_initialization_utc=pairs[0][0].isoformat(), forecast_lead_hours=[p[1] for p in pairs],
                    valid_times_utc=[p[2].isoformat() for p in pairs],
                    variable='air', parameter='TMP at 2 m above ground', units='K', grid_resolution_degrees=.25,
                    lat_bounds=args.lat, lon_bounds=args.lon, shape=list(forecast.shape),
                    reference_kind='GFS f000 analysis/initialization field from the valid-time cycle; not independent station truth',
                    interpretation='One historical forecast cycle/day and a small land-and-ocean region. These are forecast-minus-analysis errors, not an independent station validation or a general skill estimate.',
                    transformations='Spatial subset and FP32 adoption only; no regridding, injected masks or artificial errors.',
                    download_bytes=sum(s['downloaded_bytes'] for s in sources), sources=sources,
                    decoder_versions={k: importlib.metadata.version(k) for k in ['xarray', 'cfgrib', 'eccodes', 'scipy', 'numpy']},
                    files={p.name: digest(p.read_bytes()) for p in args.output.glob('*.nc')})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared genuine GFS pair: {list(forecast.shape)}, {len(sources)} source fields, '
          f'{manifest["download_bytes"]:,} source-message bytes (verified cache reused when available)')


if __name__ == '__main__':
    main()
