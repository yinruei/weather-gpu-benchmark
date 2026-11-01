"""Read a pinned public ECMWF IFS ENS Icechunk/Zarr subset, save local Zarr v2.

Member 0 is the IFS ENS control forecast, not IFS HRES. The matched reference
is the previously acquired NOAA GFS f000 analysis, not station observations.
Remote acquisition requires Python >=3.12 and requirements-icechunk.txt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_verify import prepare_pair, sha256


CATALOG_URL = 'https://stac.dynamical.org/ecmwf-ifs-ens-forecast-15-day-0-25-degree/collection.json'


def select_forecast(dataset, initialization, leads, member, lat_bounds, lon_bounds):
    """Exact selection; retain initialization/lead semantics and validate units."""
    if member != 0:
        raise ValueError('This case is explicitly the ENS control member 0')
    if len(set(leads)) != len(leads) or not leads or any(h <= 0 for h in leads):
        raise ValueError('Positive unique forecast leads required')
    field = dataset['temperature_2m']
    if set(field.dims) != {'init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude'}:
        raise ValueError('Unexpected source dimensions; refuse to infer a run axis')
    if field.attrs.get('units') != 'degree_Celsius' or field.attrs.get('step_type') != 'instant':
        raise ValueError('Unexpected source units or non-instantaneous temperature')
    lead_values = np.array(leads, dtype='timedelta64[h]')
    selected = field.sel(init_time=initialization, lead_time=lead_values, ensemble_member=member,
                         latitude=slice(lat_bounds[1], lat_bounds[0]), longitude=slice(*lon_bounds))
    if min(selected.shape) == 0:
        raise ValueError('Empty geographic subset')
    expected = initialization+lead_values
    if 'valid_time' not in selected.coords:
        raise ValueError('Archive must supply valid-time coordinates')
    np.testing.assert_array_equal(selected.valid_time.values, expected)
    return selected


def chunk_estimate(field, selected):
    chunks = tuple(int(v) for v in field.encoding['chunks'])
    spatial_count = 1
    for dim in ['latitude', 'longitude']:
        indices = field.get_index(dim).get_indexer(selected[dim].values)
        if (indices < 0).any():
            raise ValueError('Selection does not exactly match source grid')
        spatial_count *= len(np.unique(indices//chunks[field.dims.index(dim)]))
    # This source has one init, all leads and all members in each spatial chunk.
    if chunks[field.dims.index('init_time')] != 1 or chunks[field.dims.index('lead_time')] < field.sizes['lead_time'] or chunks[field.dims.index('ensemble_member')] < field.sizes['ensemble_member']:
        raise ValueError('Source chunk layout changed; review the read-budget estimate')
    return dict(chunk_shape=list(chunks), spatial_chunks=int(spatial_count),
                nominal_uncompressed_source_chunk_bytes=int(np.prod(chunks)*field.dtype.itemsize*spatial_count),
                selected_logical_bytes=int(selected.size*selected.dtype.itemsize))


def canonical_forecast(selected, initialization, leads):
    raw = selected.transpose('lead_time', 'latitude', 'longitude').values.astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('Requested historical source subset has missing/nonfinite data')
    result = xr.DataArray(raw, dims=['time', 'lat', 'lon'], name='air',
                          coords=dict(time=initialization+np.array(leads, dtype='timedelta64[h]'),
                                      lat=selected.latitude.values, lon=selected.longitude.values,
                                      forecast_reference_time=('time', np.full(len(leads), initialization)),
                                      lead_hours=('time', leads), ensemble_member=0, height_above_ground=2.),
                          attrs=dict(units='degree_Celsius', standard_name='air_temperature',
                                     long_name='IFS ENS control-member 2 metre temperature',
                                     experiment_role='real ECMWF IFS ENS control member 0 via dynamical.org Zarr; not HRES, no manufactured perturbations'))
    result.lat.attrs['units'] = 'degrees_north'
    result.lon.attrs['units'] = 'degrees_east'
    result.height_above_ground.attrs.update(units='m', positive='up')
    return result


def save_store(path, field):
    dataset = field.to_dataset(name='air').drop_encoding()
    dataset.attrs.update(storage_role='canonical local Zarr v2 subset for offline CPU/CUDA verification',
                         attribution='ECMWF IFS ENS via dynamical.org, and NOAA GFS via NOAA Open Data. See manifest.json for source roles and licenses.')
    # Explicit zero chunks are required for legacy Zarr v2 readers when an
    # integer coordinate has fill_value=None (e.g. run-time offsets/member 0).
    dataset.to_zarr(path, mode='w-', consolidated=True, zarr_format=2, write_empty_chunks=True,
                    encoding={'air': {'chunks': (1, field.sizes['lat'], field.sizes['lon'])}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', default='2024-05-06T00:00:00+00:00')
    parser.add_argument('--leads', type=int, nargs='+', default=[24, 30, 36, 42])
    parser.add_argument('--lat', type=float, nargs=2, default=[18.25, 26.])
    parser.add_argument('--lon', type=float, nargs=2, default=[116., 123.75])
    parser.add_argument('--snapshot-id', help='Pin an immutable Icechunk snapshot; default resolves main once')
    parser.add_argument('--reference', type=Path, default=Path('data/gfs_taiwan_pair/reference.nc'))
    parser.add_argument('--max-source-mib', type=float, default=256.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Remote Icechunk v2 acquisition needs Python >=3.12; evaluation of prepared Zarr does not')
    import icechunk
    if args.output.exists():
        parser.error('Choose a new output directory; no existing stores are overwritten')
    initial = datetime.fromisoformat(args.init.replace('Z', '+00:00'))
    if initial.tzinfo is None:
        parser.error('Initialization must include a UTC offset')
    initial = initial.astimezone(timezone.utc)
    if initial.hour or initial.minute or initial.second or initial.microsecond:
        parser.error('This archive contains 00 UTC IFS ENS initializations only')
    if not (-90 <= args.lat[0] < args.lat[1] <= 90 and -180 <= args.lon[0] < args.lon[1] < 180):
        parser.error('Use ascending, non-wrapping geographic bounds')
    initialization = np.datetime64(initial.replace(tzinfo=None), 'ns')
    start = time.perf_counter()
    with urlopen(CATALOG_URL, timeout=30) as response:
        catalog_bytes = response.read()
    catalog = json.loads(catalog_bytes)
    store_url = catalog['assets']['icechunk-https']['href']
    repo = icechunk.Repository.open(icechunk.http_storage(store_url))
    session = repo.readonly_session(snapshot_id=args.snapshot_id) if args.snapshot_id else repo.readonly_session('main')
    ds = xr.open_zarr(session.store, chunks=None, consolidated=False, decode_timedelta=True)
    selected = select_forecast(ds, initialization, args.leads, 0, args.lat, args.lon)
    estimate = chunk_estimate(ds.temperature_2m, selected)
    if estimate['nominal_uncompressed_source_chunk_bytes'] > args.max_source_mib*2**20:
        raise ValueError('Read exceeds the nominal uncompressed chunk budget; narrow the region or raise --max-source-mib explicitly')
    metadata_sec = time.perf_counter()-start
    print(f'Read-only snapshot {session.snapshot_id}; estimated source chunks: {estimate}', flush=True)
    # Validate valid-time coordinates before dropping other auxiliary coordinates.
    selected = selected.reset_coords(drop=True)
    start = time.perf_counter()
    selected.load()
    read_sec = time.perf_counter()-start
    forecast = canonical_forecast(selected, initialization, args.leads)
    with xr.open_dataset(args.reference) as reference_dataset:
        reference = reference_dataset.air.sel(time=forecast.time.values,
                                               lat=forecast.lat.values, lon=forecast.lon.values).load()
    if reference.attrs.get('units') != 'K' or float(reference.height_above_ground.values) != 2.:
        raise ValueError('Reference must be the documented Kelvin 2-m GFS field')
    np.testing.assert_array_equal(reference.forecast_reference_time.values, reference.time.values)
    np.testing.assert_array_equal(reference.lead_hours.values, np.zeros(len(args.leads)))
    reference.attrs['experiment_role'] = 'real GFS f000 analysis/initialization field copied to Zarr; common reference, not independent station observations'
    prepare_pair(forecast, reference)
    args.output.mkdir(parents=True)
    save_store(args.output/'forecast.zarr', forecast)
    save_store(args.output/'reference.zarr', reference)
    manifest = dict(case_kind='ecmwf_ifs_ens_control_zarr_vs_gfs_f000',
                    title='ECMWF IFS ENS control from Zarr vs common GFS f000 analysis',
                    source_catalog=CATALOG_URL, catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                    source_dataset='ecmwf-ifs-ens-forecast-15-day-0-25-degree', source_asset=store_url,
                    snapshot_id=session.snapshot_id, ensemble_member=0,
                    model_identity='IFS ENS control, not IFS HRES or AIFS',
                    forecast_initialization_utc=initial.isoformat(), forecast_lead_hours=args.leads,
                    valid_times_utc=[str(v) for v in forecast.time.values], variable='air',
                    source_variable='temperature_2m', source_units='degree_Celsius',
                    reference_units='K', grid_resolution_degrees=.25, lat_bounds=args.lat, lon_bounds=args.lon,
                    shape=list(forecast.shape), source_chunk_estimate=estimate,
                    remote_metadata_open_sec=metadata_sec, remote_selected_materialization_sec=read_sec,
                    network_timing_note='Single remote read observation; not a cold-cache throughput estimate. Nominal uncompressed chunk bytes are not measured network bytes.',
                    provider_processing='dynamical.org converts ECMWF Open Data GRIB into Celsius Zarr and rounds floating-point bits for compression; not bitwise native GRIB.',
                    license='ECMWF-derived forecast: CC BY 4.0 and ECMWF Terms of Use; NOAA reference retains its original provenance.',
                    attribution='ECMWF IFS ENS forecast data processed by dynamical.org from ECMWF Open Data.',
                    reference_source_file=args.reference.name, reference_source_sha256=sha256(args.reference),
                    reference_source_manifest='data/gfs_taiwan_pair/manifest.json',
                    interpretation='One historical ENS control forecast against same-valid-time GFS analysis, across land and ocean. GFS analysis is model-dependent and not independent truth; no model ranking or general skill conclusion.',
                    transformations='Exact init/member/lead and spatial selection, no interpolation or artificial errors. Forecast retained in Celsius; reference retained in Kelvin; verification normalizes units.',
                    local_store_format='Zarr v2, consolidated, explicit zero chunks, air chunks=(1,lat,lon)',
                    directory_hash_algorithm='sha256(zarr-directory-v1 NUL + sorted(relative UTF-8 path NUL + binary file SHA-256))',
                    acquisition_versions={k: importlib.metadata.version(k) for k in ['icechunk', 'zarr', 'xarray', 'numpy']},
                    acquisition_python=sys.version.split()[0],
                    files={name: sha256(args.output/name) for name in ['forecast.zarr', 'reference.zarr']})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {list(forecast.shape)} genuine IFS ENS-control values and matched GFS reference as Zarr v2; remote materialization {read_sec:.3f}s')


if __name__ == '__main__':
    main()
