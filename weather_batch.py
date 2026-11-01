"""Measure real Zarr batches with separate init/lead/member spatial metrics."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import xarray as xr

from weather_verify import METRICS, sha256, write_csv


CASES = [
    dict(case='control_small', runs=1, leads=4, members=1, side=32),
    dict(case='control_wide', runs=1, leads=8, members=1, side=64),
    dict(case='control_four_days', runs=4, leads=8, members=1, side=64),
    dict(case='ensemble_small', runs=1, leads=4, members=51, side=32),
    dict(case='ensemble_one_day', runs=1, leads=8, members=51, side=64),
    dict(case='ensemble_two_days', runs=2, leads=8, members=51, side=64),
    dict(case='ensemble_four_days', runs=4, leads=8, members=51, side=64),
]


def validate_batch(ds):
    expected = ('init_time', 'lead_time', 'ensemble_member', 'latitude', 'longitude')
    if ds.forecast.dims != expected or ds.reference.dims != tuple(d for d in expected if d != 'ensemble_member'):
        raise ValueError('Batch must retain explicit initialization, lead, member and spatial axes')
    for dim in expected:
        if ds.sizes[dim] == 0 or not ds.get_index(dim).is_unique:
            raise ValueError('Empty or duplicate batch coordinates')
    if not np.issubdtype(ds.init_time.dtype, np.datetime64) or not np.issubdtype(ds.lead_time.dtype, np.timedelta64):
        raise ValueError('Initialization and lead must be decoded datetime/timedelta coordinates')
    if np.isnat(ds.init_time.values).any() or np.isnat(ds.lead_time.values).any() or (ds.lead_time.values <= np.timedelta64(0, 'h')).any():
        raise ValueError('Expected finite initializations and positive forecast leads')
    valid = ds.init_time.values[:, None]+ds.lead_time.values[None, :]
    np.testing.assert_array_equal(ds.valid_time.values, valid)
    np.testing.assert_array_equal(ds.reference_initialization_time.values, valid)
    if int(ds.reference_lead_hours) != 0 or float(ds.height_above_ground) != 2.:
        raise ValueError('Reference must be same-valid-time f000 analysis at shared 2-m height')
    members = ds.ensemble_member.values
    if not np.issubdtype(members.dtype, np.integer) or (members < 0).any() or (members > 50).any():
        raise ValueError('Expected IFS ENS member identifiers 0..50')
    for name in ['forecast', 'reference']:
        if ds[name].attrs.get('units') != 'degree_Celsius':
            raise ValueError('This archive batch expects explicit Celsius units')
    for dim, limit, unit in [('latitude', 90, 'degrees_north'), ('longitude', 180, 'degrees_east')]:
        coord = ds[dim].values.astype(float)
        delta = np.diff(coord)
        if ds[dim].attrs.get('units') != unit or not np.isfinite(coord).all() or (np.abs(coord) > limit).any():
            raise ValueError('Expected finite geographic degree coordinates')
        if len(delta) and (not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-8) or delta[0] == 0):
            raise ValueError('Cosine weights require a regular geographic grid')


def pack_batch(ds):
    validate_batch(ds)
    # Match weather_verify.py: normalize temperatures in FP64 then adopt FP32.
    f = (ds.forecast.values.astype(np.float64)+273.15).astype(np.float32)
    r = (ds.reference.values.astype(np.float64)+273.15).astype(np.float32)
    r = np.broadcast_to(r[:, :, None, :, :], f.shape)
    mask = np.isfinite(f) & np.isfinite(r)
    lat = ds.latitude.values
    spatial = np.where(np.abs(lat) >= 90, 0., np.cos(np.deg2rad(lat)))
    weights = np.broadcast_to(spatial[None, None, None, :, None], f.shape).astype(np.float32)
    weights = np.where(mask, weights, 0.)
    shape = (-1, ds.sizes['latitude']*ds.sizes['longitude'])
    arrays = tuple(np.ascontiguousarray(a.reshape(shape), dtype=np.float32)
                   for a in (np.where(mask, f, 0.), np.where(mask, r, 0.), weights))
    if (arrays[2].sum(axis=1, dtype=np.float64) <= 0).any():
        raise ValueError('Every forecast group must have positive joint-valid weight')
    return arrays, mask.reshape(shape).sum(axis=1)


def load_batch(path, case):
    with xr.open_zarr(path, chunks=None, consolidated=True) as source:
        for dim, count in [('init_time', case['runs']), ('lead_time', case['leads']),
                            ('ensemble_member', case['members']), ('latitude', case['side']),
                            ('longitude', case['side'])]:
            if source.sizes[dim] < count:
                raise ValueError(f'Insufficient {dim} for {case["case"]}')
        selected = source.isel(init_time=slice(case['runs']), lead_time=slice(case['leads']),
                               ensemble_member=slice(case['members']),
                               latitude=slice(-case['side'], None), longitude=slice(case['side'])).load()
    return selected


def numpy_grouped(f, r, weights):
    error = f-r
    denominator = weights.sum(axis=1)
    return np.stack(((error*weights).sum(axis=1)/denominator,
                     (np.abs(error)*weights).sum(axis=1)/denominator,
                     np.sqrt((error*error*weights).sum(axis=1)/denominator)), axis=1)


def torch_grouped(f, r, weights):
    import torch
    error = f-r
    denominator = weights.sum(dim=1)
    return torch.stack(((error*weights).sum(dim=1)/denominator,
                        (error.abs()*weights).sum(dim=1)/denominator,
                        ((error*error*weights).sum(dim=1)/denominator).sqrt()), dim=1)


def benchmark_methods(methods, expected, repeats, warmup, seed, sync=lambda: None):
    """Shuffle method order within each repetition; validate every returned group."""
    def check(values):
        if np.asarray(values).shape != expected.shape:
            raise ValueError('Missing or merged forecast-group metrics')
        np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-5)
    for method in methods.values():
        for _ in range(warmup):
            check(method())
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        for order, name in enumerate(rng.permutation(list(methods))):
            sync()
            start = time.perf_counter()
            values = methods[name]()
            sync()
            elapsed = time.perf_counter()-start
            check(values)
            rows.append(dict(method=str(name), repeat=repeat, order=order, seconds=elapsed,
                             max_abs_metric_error=float(np.max(np.abs(values-expected)))))
    summaries = []
    for name in methods:
        seconds = [row['seconds'] for row in rows if row['method'] == name]
        summaries.append(dict(method=name, median_sec=float(np.median(seconds)),
                              p25_sec=float(np.percentile(seconds, 25)), p75_sec=float(np.percentile(seconds, 75))))
    return summaries, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['numpy', 'both'], default='both')
    parser.add_argument('--cases', nargs='+', choices=[c['case'] for c in CASES])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--group-loop', action='store_true', help='Also measure per-group CUDA launches on the largest selected case')
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 1 or args.output.exists():
        parser.error('Need >=2 repeats, >=1 warmup, and a new output path')
    torch = None
    if args.backend == 'both':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; no silent CPU fallback')
        torch.cuda.set_device(0)
        torch.empty(1, device='cuda')
        torch.cuda.synchronize()
    manifest = json.loads((args.data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(args.data)
    if digest != manifest['files']['batch.zarr'] or manifest['case_kind'] != 'ecmwf_ens_batch_vs_gfs_analysis':
        raise ValueError('Input data do not match the source manifest')
    chosen = [c for c in CASES if not args.cases or c['case'] in args.cases]
    env = dict(python=platform.python_version(), numpy=np.__version__, xarray=xr.__version__,
               torch=torch.__version__ if torch else None, cuda_runtime=torch.version.cuda if torch else None,
               store_sha256=digest, source_manifest=manifest, cases=chosen,
               repeats=args.repeats, warmup=args.warmup, seed=args.seed,
               tolerance=dict(rtol=1e-5, atol=1e-5),
               order='seeded shuffle of methods within each repetition; cases in listed order',
               metric_semantics='Coslat-weighted spatial Bias/MAE/RMSE separately for every init/lead/member, FP32 Kelvin inputs, FP64 arithmetic reference on identical packed values. Not ensemble-mean forecast metrics.',
               timing='Resident: packed arrays/tensors reused, full group metric return included. Upload: existing packed host arrays to CUDA plus metrics/return. Local pipeline: warm-cache Zarr open/read, checks, unit normalization, mask/weights, packing, allocation/upload as applicable, metrics/return. Remote download, initial CUDA context, validation of returned metrics, input hashing and CSV writes excluded.',
               before=None, after=None)
    if torch:
        from precision_benchmark import hardware_snapshot
        env['before'] = hardware_snapshot(torch)
    args.output.mkdir(parents=True)
    all_summary, all_samples, all_metrics, memory = [], [], [], []
    for case in chosen:
        ds = load_batch(args.data, case)
        arrays, valid_counts = pack_batch(ds)
        truth = numpy_grouped(*(a.astype(np.float64) for a in arrays))
        def pipeline(gpu=False):
            values, _ = pack_batch(load_batch(args.data, case))
            return gpu_upload(values) if gpu else numpy_grouped(*values)
        methods = {'numpy_resident': lambda: numpy_grouped(*arrays),
                   'numpy_local_pipeline': lambda: pipeline()}
        backend_results = {'reference_fp64': truth, 'numpy_fp32': numpy_grouped(*arrays)}
        tensors = None
        if torch:
            tensors = tuple(torch.from_numpy(a).to('cuda') for a in arrays)
            torch.cuda.synchronize()
            def gpu_upload(values):
                inputs = tuple(torch.from_numpy(a).to('cuda') for a in values)
                return torch_grouped(*inputs).cpu().numpy()
            methods.update(cuda_resident=lambda: torch_grouped(*tensors).cpu().numpy(),
                           cuda_upload=lambda: gpu_upload(arrays),
                           cuda_local_pipeline=lambda: pipeline(True))
            if args.group_loop and case == chosen[-1]:
                def group_loop():
                    # Preserve inputs on device; launch/reduce/return one group at a time.
                    return np.concatenate([torch_grouped(*(a[i:i+1] for a in tensors)).cpu().numpy()
                                           for i in range(arrays[0].shape[0])])
                methods['cuda_group_loop'] = group_loop
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            methods['cuda_resident']()
            torch.cuda.synchronize()
            memory.append(dict(case=case['case'], resident_input_bytes=baseline,
                               resident_peak_extra_bytes=torch.cuda.max_memory_allocated()-baseline))
            backend_results['cuda_fp32'] = methods['cuda_resident']()
        summary, samples = benchmark_methods(methods, truth, args.repeats, args.warmup,
                                              args.seed, torch.cuda.synchronize if torch else lambda: None)
        metadata = dict(case=case['case'], groups=arrays[0].shape[0], cells_per_group=arrays[0].shape[1],
                        points=arrays[0].size, valid_points=int(valid_counts.sum()))
        all_summary.extend({**metadata, **row} for row in summary)
        all_samples.extend({**metadata, **row} for row in samples)
        group = 0
        for initial in ds.init_time.values:
            for lead in ds.lead_time.values:
                for member in ds.ensemble_member.values:
                    identity = dict(case=case['case'], initialization=str(initial),
                                    lead_hours=int(lead/np.timedelta64(1, 'h')), valid_time=str(initial+lead),
                                    member=int(member), valid_points=int(valid_counts[group]),
                                    valid_weight_sum=float(arrays[2][group].sum(dtype=np.float64)))
                    for backend, values in backend_results.items():
                        np.testing.assert_allclose(values[group], truth[group], rtol=1e-5, atol=1e-5)
                        all_metrics.append({**identity, 'backend': backend,
                                            **dict(zip(METRICS, map(float, values[group])))})
                    group += 1
        print(json.dumps(dict(**metadata, medians_ms={s['method']: round(s['median_sec']*1000, 4) for s in summary})), flush=True)
        # Do not retain previous case device allocations when measuring the next case.
        methods.clear()
        del tensors, ds, arrays, truth, backend_results
    if torch:
        env['after'] = hardware_snapshot(torch)
    env['memory'] = memory
    write_csv(args.output/'timing_summary.csv', all_summary)
    write_csv(args.output/'timing_samples.csv', all_samples)
    write_csv(args.output/'metrics.csv', all_metrics)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: every measured group agrees with FP64; {len(all_samples)} timing samples', flush=True)


if __name__ == '__main__':
    main()
