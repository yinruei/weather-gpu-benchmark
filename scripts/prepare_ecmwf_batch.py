"""Acquire a bounded, pinned ECMWF ensemble/GFS analysis Zarr batch (Python >=3.12)."""
import argparse
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
from weather_verify import sha256

EC = 'ecmwf-ifs-ens-forecast-15-day-0-25-degree'
GFS = 'noaa-gfs-forecast'
SNAPSHOTS = {EC: 'G76A041BF9PWGCNRBFVG', GFS: '7FB36MTVTBHMA65ETQB0'}
ASSETS = {
    EC: 'https://dynamical-ecmwf-ifs-ens.s3.us-west-2.amazonaws.com/ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk',
    GFS: 'https://dynamical-noaa-gfs.s3.us-west-2.amazonaws.com/noaa-gfs-forecast/v0.2.7.icechunk',
}


def read_budget(field, selected):
    """Nominal uncompressed chunks touched, not actual HTTP traffic."""
    chunks = field.encoding['chunks']
    count = 1
    for dimension, chunk in zip(field.dims, chunks):
        indices = field.get_index(dimension).get_indexer(np.atleast_1d(selected[dimension].values))
        if (indices < 0).any():
            raise ValueError('Selection is not an exact source-coordinate match')
        count *= len(np.unique(indices // chunk))
    return dict(chunk_shape=list(chunks), chunks=int(count),
                nominal_uncompressed_bytes=int(count*np.prod(chunks)*field.dtype.itemsize),
                selected_logical_bytes=int(selected.size*selected.dtype.itemsize))


def open_source(name, snapshot):
    import icechunk
    start = time.perf_counter()
    catalog_url = f'https://stac.dynamical.org/{name}/collection.json'
    with urlopen(catalog_url, timeout=30) as stream:
        raw = stream.read()
    # Pin the repository location as well as its snapshot: catalog assets may
    # move to a new version that no longer contains the recorded snapshot.
    asset = ASSETS[name]
    repo = icechunk.Repository.open(icechunk.http_storage(asset))
    session = repo.readonly_session(snapshot_id=snapshot)
    dataset = xr.open_zarr(session.store, chunks=None, consolidated=False, decode_timedelta=True)
    field = dataset.temperature_2m
    if field.attrs.get('units') != 'degree_Celsius' or field.attrs.get('step_type') != 'instant':
        raise ValueError(f'Unexpected temperature semantics: {name}')
    provenance = dict(dataset=name, catalog=catalog_url, asset=asset, snapshot_id=session.snapshot_id,
                      catalog_sha256=hashlib.sha256(raw).hexdigest(),
                      metadata_open_seconds=time.perf_counter()-start)
    return dataset, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/ecmwf_batch_case'))
    parser.add_argument('--max-source-mib', type=float, default=1024.)
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Acquisition needs Python >=3.12; offline benchmarking supports the WSL environment')
    if args.output.exists():
        parser.error('Choose a new output path; existing data are preserved')
    ec, ec_source = open_source(EC, SNAPSHOTS[EC])
    gfs, gfs_source = open_source(GFS, SNAPSHOTS[GFS])
    inits = np.arange(np.datetime64('2024-05-06'), np.datetime64('2024-05-10')).astype('datetime64[ns]')
    leads = np.arange(24, 67, 6)
    lead_delta = leads.astype('timedelta64[h]')
    members = np.arange(51)
    valid = inits[:, None]+lead_delta[None, :]
    unique_valid = np.unique(valid)
    region = dict(latitude=slice(34., 18.25), longitude=slice(116., 131.75))
    forecast = ec.temperature_2m.sel(init_time=inits, lead_time=lead_delta,
                                    ensemble_member=members, **region)
    reference = gfs.temperature_2m.sel(init_time=unique_valid, lead_time=np.timedelta64(0, 'h'), **region)
    if forecast.dims != ('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude'):
        raise ValueError('Unexpected forecast axes')
    if reference.dims != ('init_time', 'latitude', 'longitude'):
        raise ValueError('Unexpected reference axes')
    np.testing.assert_array_equal(forecast.valid_time.values, valid)
    np.testing.assert_array_equal(reference.valid_time.values, unique_valid)
    for dim in ['latitude', 'longitude']:
        np.testing.assert_array_equal(forecast[dim].values, reference[dim].values)
        if forecast.sizes[dim] != 64:
            raise ValueError('Expected exact 64 x 64 native grid; no interpolation allowed')
    sources = [ec_source, gfs_source]
    for source, field, selected in [(ec_source, ec.temperature_2m, forecast),
                                     (gfs_source, gfs.temperature_2m, reference)]:
        source['read_budget'] = read_budget(field, selected)
    total = sum(s['read_budget']['nominal_uncompressed_bytes'] for s in sources)
    if not np.isfinite(args.max_source_mib) or args.max_source_mib <= 0 or total > args.max_source_mib*2**20:
        raise ValueError(f'Nominal source read {total/2**20:.1f} MiB exceeds the budget')
    print(f'Bounded read: {total/2**20:.1f} nominal uncompressed MiB, not HTTP bytes', flush=True)
    values = []
    for field, source in [(forecast, ec_source), (reference, gfs_source)]:
        start = time.perf_counter()
        # Drop auxiliaries only after validating run/lead/valid-time identities.
        values.append(field.reset_coords(drop=True).load().values.astype(np.float32))
        source['selected_materialization_seconds'] = time.perf_counter()-start
        print(f"Loaded {source['dataset']} in {source['selected_materialization_seconds']:.3f}s", flush=True)
    f, r = values
    if not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError('Historical source selection contains missing data; inspect before benchmarking')
    reference_indices = np.searchsorted(unique_valid, valid)
    ds = xr.Dataset(
        {'forecast': (('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude'), f),
         'reference': (('init_time', 'lead_time', 'latitude', 'longitude'), r[reference_indices])},
        coords=dict(init_time=inits, lead_time=lead_delta, ensemble_member=members,
                    latitude=forecast.latitude.values, longitude=forecast.longitude.values,
                    valid_time=(('init_time', 'lead_time'), valid),
                    reference_initialization_time=(('init_time', 'lead_time'), valid),
                    reference_lead_hours=0, height_above_ground=2.))
    for name in ['forecast', 'reference']:
        ds[name].attrs.update(units='degree_Celsius', standard_name='air_temperature')
    ds.latitude.attrs['units'] = 'degrees_north'
    ds.longitude.attrs['units'] = 'degrees_east'
    ds.height_above_ground.attrs.update(units='m', positive='up')
    ds.attrs.update(attribution='ECMWF IFS ENS and NOAA GFS processed by dynamical.org; see manifest.json',
                    reference_role='GFS f000 analysis, model-dependent, not independent observations',
                    forecast_role='51 individual IFS ENS members, not HRES or ensemble-mean forecasts')
    args.output.mkdir(parents=True)
    store = args.output/'batch.zarr'
    ds.to_zarr(store, mode='w-', zarr_format=2, consolidated=True, write_empty_chunks=True,
               encoding={'forecast': {'chunks': (1, 4, 51, 32, 32)},
                         'reference': {'chunks': (1, 4, 32, 32)}})
    manifest = dict(case_kind='ecmwf_ens_batch_vs_gfs_analysis', sources=sources,
                    shape=list(f.shape), initializations=[str(v) for v in inits],
                    lead_hours=leads.tolist(), ensemble_members=members.tolist(),
                    valid_times=[[str(v) for v in row] for row in valid],
                    latitude_bounds=[18.25, 34.], longitude_bounds=[116., 131.75],
                    forecast_points=int(f.size), unique_reference_times=len(unique_valid),
                    transformations='Exact source selections; no tiling, artificial perturbations or regridding. Reference indexed by exact valid time and shared across members.',
                    provider_processing='dynamical.org converts native GRIB to Celsius Zarr and rounds floating-point bits; not bitwise native GRIB.',
                    license='CC BY 4.0 for provider archives; ECMWF Terms of Use also apply to ECMWF-derived data.',
                    attribution='ECMWF IFS ENS forecast and NOAA NWS NCEP GFS data processed by dynamical.org from their open-data archives.',
                    interpretation='Separate per-initialization/lead/member spatial metrics; descriptive pooled errors are not ensemble-mean forecast skill. Reference is GFS analysis, not independent truth. Overlapping valid times and related members are not independent samples.',
                    acquisition_python=sys.version.split()[0],
                    acquisition_versions={k: importlib.metadata.version(k) for k in ['icechunk', 'zarr', 'xarray', 'numpy']},
                    files={'batch.zarr': sha256(store)})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {f.size:,} genuine forecast values: {store}', flush=True)


if __name__ == '__main__':
    main()
