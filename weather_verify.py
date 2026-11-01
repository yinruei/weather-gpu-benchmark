"""Verify canonical time/lat/lon temperature NetCDF fields on CPU and CUDA.

Time coordinates must already represent matching UTC valid times. No regridding
or forecast-initialization/lead-time inference is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import xarray as xr


METRICS = ('bias', 'mae', 'rmse')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_temperature(data):
    aliases = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'}
    rename = {k: v for k, v in aliases.items() if k in data.dims}
    if any(v in data.dims for v in rename.values()):
        raise ValueError('Ambiguous dimension aliases')
    data = data.rename(rename)
    if set(data.dims) != {'time', 'lat', 'lon'}:
        raise ValueError('Expected exactly time, lat, lon dimensions; select other levels first')
    data = data.transpose('time', 'lat', 'lon')
    for dim in data.dims:
        if dim not in data.coords or data[dim].dims != (dim,) or data.sizes[dim] == 0:
            raise ValueError(f'{dim} must have a nonempty one-dimensional coordinate')
        if not data.get_index(dim).is_unique:
            raise ValueError(f'Duplicate {dim} coordinates')
    if not np.issubdtype(data.time.dtype, np.datetime64) or np.isnat(data.time.values).any():
        raise ValueError('Time must contain finite datetime64 UTC valid-time labels')
    lat, lon = data.lat.values.astype(float), data.lon.values.astype(float)
    for dim, suffix in [('lat', 'north'), ('lon', 'east')]:
        coord_unit = str(data[dim].attrs.get('units', '')).strip().lower()
        allowed = {'', 'degree', 'degrees', f'degree_{suffix}', f'degrees_{suffix}'}
        allowed |= {f'degree_{suffix[0]}', f'degrees_{suffix[0]}',
                    f'degree{suffix[0]}', f'degrees{suffix[0]}'}
        if coord_unit not in allowed:
            raise ValueError(f'{dim} must use geographic degrees, not {coord_unit!r}')
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError('Latitude/longitude must be finite')
    if (np.abs(lat) > 90).any() or (lon < -180).any() or (lon > 360).any():
        raise ValueError('Invalid latitude/longitude range')
    data = data.assign_coords(lon=(lon + 180) % 360 - 180)
    if not data.get_index('lon').is_unique:
        raise ValueError('Longitude wrap creates duplicate coordinates (e.g. 0 and 360)')
    data = data.sortby(['time', 'lat', 'lon'])
    units = str(data.attrs.get('units', '')).strip().lower()
    kelvin = {'k', 'degk', 'kelvin', 'degree_kelvin', 'degrees_kelvin'}
    celsius = {'c', 'degc', 'celsius', 'degree_celsius', 'degrees_celsius'}
    if units not in kelvin | celsius:
        raise ValueError(f'Unknown temperature units: {units!r}; specify K or degC in the file')
    attrs = dict(data.attrs)
    # Convert in FP64 before adopting the FP32 benchmark representation.
    data = data.astype(np.float64)
    if units in celsius:
        data = data + 273.15
    data = data.astype(np.float32)
    data.attrs = {**attrs, 'units': 'K', 'original_units': units}
    return data


def prepare_pair(forecast, reference, weighting='coslat'):
    f, r = normalize_temperature(forecast), normalize_temperature(reference)
    # No inner join: dropping nonmatching valid times or cells would hide errors.
    f, r = xr.align(f, r, join='exact', copy=False)
    if weighting == 'coslat':
        for dim in ('lat', 'lon'):
            step = np.diff(f[dim].values.astype(float))
            if len(step) and not np.allclose(step, step[0], rtol=1e-6, atol=1e-8):
                raise ValueError('Cosine weights require a regular non-dateline-crossing lat/lon grid')
        lat = f.lat.values.astype(float)
        spatial = np.where(np.abs(lat) >= 90, 0., np.cos(np.deg2rad(lat)))
    elif weighting == 'uniform':
        spatial = np.ones(f.sizes['lat'])
    else:
        raise ValueError('Weighting must be coslat or uniform')
    weights = np.broadcast_to(spatial[None, :, None], f.shape).astype(np.float32)
    mask = np.isfinite(f.values) & np.isfinite(r.values)
    if not mask.any() or weights[mask].sum(dtype=np.float64) <= 0:
        raise ValueError('No joint valid cells with positive weight')
    return f, r, mask, weights


def pack(f, r, mask, weights):
    return tuple(np.ascontiguousarray(v[mask], dtype=np.float32)
                 for v in (np.asarray(f), np.asarray(r), weights))


def numpy_weighted(f, r, weights):
    error = f - r
    denominator = weights.sum()
    return ((error * weights).sum() / denominator,
            (np.abs(error) * weights).sum() / denominator,
            np.sqrt((error * error * weights).sum() / denominator))


def torch_weighted(f, r, weights):
    error = f - r
    denominator = weights.sum()
    return ((error * weights).sum() / denominator,
            (error.abs() * weights).sum() / denominator,
            ((error * error * weights).sum() / denominator).sqrt())


def load_pair(forecast_path, reference_path, variable, reference_variable, weighting):
    fields = []
    for path, name in [(forecast_path, variable), (reference_path, reference_variable)]:
        with xr.open_dataset(path) as dataset:
            fields.append(dataset[name].load())
    return prepare_pair(*fields, weighting=weighting)


def measure(fn, method, repeats, warmup, sync=lambda: None):
    for _ in range(warmup):
        fn()
    raw = []
    for repeat in range(repeats):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        seconds = time.perf_counter() - start
        raw.append(dict(method=method, repeat=repeat, seconds=seconds,
                        **dict(zip(METRICS, map(float, result)))))
    values = [r['seconds'] for r in raw]
    return dict(method=method, median_sec=float(np.median(values)),
                p25_sec=float(np.percentile(values, 25)), p75_sec=float(np.percentile(values, 75))), raw


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--variable', default='air')
    parser.add_argument('--reference-variable')
    parser.add_argument('--weighting', choices=['coslat', 'uniform'], default='coslat')
    parser.add_argument('--backend', choices=['numpy', 'both'], default='numpy')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1:
        parser.error('At least two repeats and one warmup required')
    if args.output.exists():
        parser.error('Choose a new output directory; existing paths are not overwritten')
    args.reference_variable = args.reference_variable or args.variable
    torch = None
    if args.backend == 'both':
        try:
            import torch
        except ImportError:
            parser.error('--backend both requires CUDA-enabled PyTorch')
        if not torch.cuda.is_available():
            parser.error('--backend both requires CUDA; no silent CPU fallback')
        torch.cuda.set_device(0)
        torch.empty(1, device='cuda')
        torch.cuda.synchronize()
    f, r, mask, weights = load_pair(args.forecast, args.reference, args.variable,
                                   args.reference_variable, args.weighting)
    arrays = pack(f.values, r.values, mask, weights)
    truth = numpy_weighted(*(a.astype(np.float64) for a in arrays))
    args.output.mkdir(parents=True)
    env = dict(python=platform.python_version(), numpy=np.__version__, xarray=xr.__version__,
               torch=torch.__version__ if torch else None,
               cuda_runtime=torch.version.cuda if torch else None,
               forecast_sha256=sha256(args.forecast), reference_sha256=sha256(args.reference),
               forecast_file=args.forecast.name, reference_file=args.reference.name,
               forecast_role=f.attrs.get('experiment_role', 'user-supplied field; origin not verified'),
               reference_role=r.attrs.get('experiment_role', 'user-supplied reference; origin not verified'),
               variable=args.variable, reference_variable=args.reference_variable,
               weighting=args.weighting, dtype='float32', metrics_units='K',
               valid_time_policy='time coordinates already represent UTC valid time; exact matching after sorting',
               coordinate_policy='1D regular lat/lon for coslat; longitude wrapped to [-180,180); no regridding',
               missing_policy='joint finite forecast/reference mask; normalize by valid-cell weights',
               repeats=args.repeats, warmup=args.warmup, shape=list(f.shape),
               finite_points=int(mask.sum()), total_points=int(mask.size),
               time_start=str(f.time.values[0]), time_end=str(f.time.values[-1]),
               before=None, after=None,
               timing='resident compute includes scalar return; pipeline includes warm-cache NetCDF read, decode, unit/coordinate checks, mask/weights, packing, allocation/upload (CUDA) and computation; excludes context init, download and output writing',
               timing_order='all repeats of each method in order; not order-balanced',
               numerical_tolerance=dict(rtol=1e-5, atol=1e-5))
    if torch:
        from precision_benchmark import hardware_snapshot
        env['before'] = hardware_snapshot(torch)

    def gpu_result(inputs):
        tensors = [torch.from_numpy(a).to('cuda') for a in inputs]
        return tuple(v.item() for v in torch_weighted(*tensors))

    def pipeline(backend):
        ff, rr, mm, ww = load_pair(args.forecast, args.reference, args.variable,
                                    args.reference_variable, args.weighting)
        aa = pack(ff.values, rr.values, mm, ww)
        return numpy_weighted(*aa) if backend == 'numpy' else gpu_result(aa)

    timings, raw = [], []
    methods = [('numpy_resident', lambda: numpy_weighted(*arrays), lambda: None),
               ('numpy_warm_file_pipeline', lambda: pipeline('numpy'), lambda: None)]
    if torch:
        tensors = [torch.from_numpy(a).to('cuda') for a in arrays]
        torch.cuda.synchronize()
        methods += [('cuda_resident', lambda: tuple(v.item() for v in torch_weighted(*tensors)), torch.cuda.synchronize),
                    ('cuda_warm_file_pipeline', lambda: pipeline('cuda'), torch.cuda.synchronize)]
    for name, fn, sync in methods:
        summary, records = measure(fn, name, args.repeats, args.warmup, sync)
        np.testing.assert_allclose([[r[k] for k in METRICS] for r in records],
                                   np.broadcast_to(truth, (args.repeats, 3)), rtol=1e-5, atol=1e-5)
        timings.append(summary)
        raw.extend(records)
    results = []
    for label, ff, rr, mm, ww in [('all_times', f.values, r.values, mask, weights)] + [
            (str(t), f.values[i], r.values[i], mask[i], weights[i]) for i, t in enumerate(f.time.values)]:
        coverage = dict(scope=label, total_points=int(mm.size), valid_points=int(mm.sum()),
                        valid_fraction=float(mm.mean()), valid_weight_fraction=float(ww[mm].sum(dtype=np.float64)/ww.sum(dtype=np.float64)))
        aa = pack(ff, rr, mm, ww)
        backends = ['reference_fp64', 'numpy_fp32'] + (['cuda_fp32'] if torch else [])
        expected = numpy_weighted(*(a.astype(np.float64) for a in aa)) if len(aa[0]) and aa[2].sum() > 0 else None
        for backend in backends:
            result = None
            if expected is not None:
                result = expected if backend == 'reference_fp64' else numpy_weighted(*aa) if backend == 'numpy_fp32' else gpu_result(aa)
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            results.append({**coverage, 'backend': backend,
                            **dict(zip(METRICS, map(float, result) if result is not None else [None]*3))})
    if torch:
        env['after'] = hardware_snapshot(torch)
    write_csv(args.output / 'metrics.csv', results)
    write_csv(args.output / 'timing_summary.csv', timings)
    write_csv(args.output / 'timing_samples.csv', raw)
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(points=env['total_points'], joint_valid=env['finite_points'],
                          fp64_metrics=dict(zip(METRICS, map(float, truth))), timings=timings), indent=2))
    print(f'PASS: all saved backend results agree with the FP64 reference; outputs in {args.output}')


if __name__ == '__main__':
    main()
