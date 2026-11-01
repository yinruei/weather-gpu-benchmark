"""Profile and reuse real Zarr inputs across full-domain/four-quadrant verification."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import xarray as xr

from weather_batch import CASES, numpy_grouped, torch_grouped, validate_batch
from weather_verify import METRICS, sha256, write_csv


def regions(height, width):
    if height < 2 or width < 2:
        raise ValueError('At least two cells along each spatial axis required')
    h, w = height//2, width//2
    # The native archive latitude axis descends; the first row is north.
    return [('full', (0, height, 0, width)), ('northwest', (0, h, 0, w)),
            ('northeast', (0, h, w, width)), ('southwest', (h, height, 0, w)),
            ('southeast', (h, height, w, width))]


class Trace:
    def __init__(self, sync=lambda: None):
        self.sync = sync
        self.seconds = {}

    @contextmanager
    def stage(self, name):
        self.sync()
        start = time.perf_counter()
        yield
        self.sync()
        self.seconds[name] = self.seconds.get(name, 0.)+time.perf_counter()-start


@contextmanager
def stage(trace, name):
    if trace is None:
        yield
    else:
        with trace.stage(name):
            yield


def load_region(path, case, bounds=None, trace=None):
    with stage(trace, 'zarr_read_decode'):
        with xr.open_zarr(path, chunks=None, consolidated=True) as source:
            for dim, count in [('init_time', case['runs']), ('lead_time', case['leads']),
                                ('ensemble_member', case['members']), ('latitude', case['side']),
                                ('longitude', case['side'])]:
                if source.sizes[dim] < count:
                    raise ValueError(f'Insufficient {dim}')
            ds = source.isel(init_time=slice(case['runs']), lead_time=slice(case['leads']),
                             ensemble_member=slice(case['members']), latitude=slice(-case['side'], None),
                             longitude=slice(case['side']))
            if bounds is not None:
                y0, y1, x0, x1 = bounds
                ds = ds.isel(latitude=slice(y0, y1), longitude=slice(x0, x1))
            ds = ds.load()
    return ds


@dataclass
class Prepared:
    kind: str
    shape: tuple
    forecast: object
    reference: object
    weights: object
    mask: object = None

    @property
    def buffers(self):
        return [v for v in [self.forecast, self.reference, self.weights, self.mask] if v is not None]

    @property
    def logical_bytes(self):
        return sum(v.nbytes if isinstance(v, np.ndarray) else v.numel()*v.element_size() for v in self.buffers)


def prepare(ds, kind, trace=None):
    if kind not in ['dense', 'shared']:
        raise ValueError('Unknown representation')
    with stage(trace, 'coordinate_checks'):
        validate_batch(ds)
    with stage(trace, 'unit_normalization'):
        f = (ds.forecast.values.astype(np.float64)+273.15).astype(np.float32)
        r = (ds.reference.values.astype(np.float64)+273.15).astype(np.float32)
    with stage(trace, 'mask_weights'):
        spatial = np.where(np.abs(ds.latitude.values) >= 90, 0.,
                           np.cos(np.deg2rad(ds.latitude.values))).astype(np.float32)
        mask = np.isfinite(f) & np.isfinite(r[:, :, None, :, :])
        if kind == 'dense':
            r = np.broadcast_to(r[:, :, None, :, :], f.shape)
            weights = np.where(mask, np.broadcast_to(spatial[None, None, None, :, None], f.shape), 0.)
        else:
            weights = spatial
            # Share one spatial weight vector and one analysis per init/lead.
            # Retain a member-specific mask only when some input is nonfinite.
            if mask.all():
                mask = None
    with stage(trace, 'pack_arrays'):
        if kind == 'dense':
            f, r = np.where(mask, f, 0.), np.where(mask, r, 0.)
            f, r, weights = [np.ascontiguousarray(a) for a in (f, r, weights)]
            if (weights.sum(axis=(-2, -1), dtype=np.float64) <= 0).any():
                raise ValueError('No positive valid weight in a forecast group')
            mask = None
        else:
            f, r, weights = [np.ascontiguousarray(a) for a in (f, r, weights)]
            valid_weight = weights.sum(dtype=np.float64)*f.shape[-1] if mask is None else (
                mask*weights[None, None, None, :, None]).sum(axis=(-2, -1), dtype=np.float64)
            if np.any(valid_weight <= 0):
                raise ValueError('No positive valid weight in a forecast group')
    return Prepared(kind, f.shape, f, r, weights, mask)


def upload(prepared, trace=None):
    import torch
    with stage(trace, 'upload'):
        values = [torch.from_numpy(v).to('cuda') if v is not None else None for v in
                  [prepared.forecast, prepared.reference, prepared.weights, prepared.mask]]
    return Prepared(prepared.kind, prepared.shape, *values)


def spatial_subset(prepared, bounds, trace=None):
    y0, y1, x0, x1 = bounds
    with stage(trace, 'region_selection'):
        f = prepared.forecast[..., y0:y1, x0:x1]
        r = prepared.reference[..., y0:y1, x0:x1]
        weights = prepared.weights[..., y0:y1, x0:x1] if prepared.kind == 'dense' else prepared.weights[y0:y1]
        mask = None if prepared.mask is None else prepared.mask[..., y0:y1, x0:x1]
    return Prepared(prepared.kind, f.shape, f, r, weights, mask)


def numpy_shared(prepared):
    f, r, w, mask = prepared.forecast, prepared.reference, prepared.weights, prepared.mask
    error = f-r[:, :, None, :, :]
    weights = w[None, None, None, :, None]
    if mask is None:
        denominator = w.sum()*f.shape[-1]
    else:
        error = np.where(mask, error, 0.)
        weights = np.where(mask, weights, 0.)
        denominator = weights.sum(axis=(-2, -1))
    if np.any(denominator <= 0):
        raise ValueError('Region has no positive joint-valid weight')
    return np.stack(((error*weights).sum(axis=(-2, -1))/denominator,
                     (np.abs(error)*weights).sum(axis=(-2, -1))/denominator,
                     np.sqrt((error*error*weights).sum(axis=(-2, -1))/denominator)), axis=-1).reshape(-1, 3)


def torch_shared(prepared):
    import torch
    f, r, w, mask = prepared.forecast, prepared.reference, prepared.weights, prepared.mask
    error = f-r[:, :, None, :, :]
    weights = w[None, None, None, :, None]
    if mask is None:
        denominator = w.sum()*f.shape[-1]
    else:
        error = torch.where(mask, error, 0.)
        weights = torch.where(mask, weights, 0.)
        denominator = weights.sum(dim=(-2, -1))
    # Positive regional weight is validated on the host before any timed GPU work.
    return torch.stack(((error*weights).sum(dim=(-2, -1))/denominator,
                        (error.abs()*weights).sum(dim=(-2, -1))/denominator,
                        ((error*error*weights).sum(dim=(-2, -1))/denominator).sqrt()), dim=-1).reshape(-1, 3)


def compute(prepared, gpu=False, trace=None):
    with stage(trace, 'compute'):
        if prepared.kind == 'dense':
            arrays = [a.reshape(-1, prepared.shape[-2]*prepared.shape[-1]) for a in prepared.buffers]
            result = torch_grouped(*arrays) if gpu else numpy_grouped(*arrays)
        else:
            result = torch_shared(prepared) if gpu else numpy_shared(prepared)
    with stage(trace, 'result_return'):
        return result.cpu().numpy() if gpu else result


def run_workflow(data, case, tasks, mode, gpu=False, cache=None, trace=None):
    results = []
    if mode == 'reopen_dense':
        for _, bounds in tasks:
            ds = load_region(data, case, bounds, trace)
            prepared = prepare(ds, 'dense', trace)
            if gpu:
                prepared = upload(prepared, trace)
            results.append(compute(prepared, gpu, trace))
    else:
        if mode == 'reuse_shared':
            if cache is None:
                raise ValueError('Explicit prepared cache required')
            prepared = cache
        else:
            ds = load_region(data, case, trace=trace)
            prepared = prepare(ds, 'dense' if mode == 'once_dense' else 'shared', trace)
            if gpu:
                prepared = upload(prepared, trace)
        for _, bounds in tasks:
            results.append(compute(spatial_subset(prepared, bounds, trace), gpu, trace))
    with stage(trace, 'result_return'):
        return np.stack(results)


def expected_metrics(data, case, tasks):
    """FP64 reference from the unchanged dense packing path, per region/group."""
    from weather_batch import pack_batch
    expected, coverage = [], []
    for name, bounds in tasks:
        ds = load_region(data, case, bounds)
        arrays, counts = pack_batch(ds)
        expected.append(numpy_grouped(*(a.astype(np.float64) for a in arrays)))
        coverage.append(dict(region=name, bounds=list(bounds), valid_points=counts.tolist(),
                             weight_sum=arrays[2].sum(axis=1, dtype=np.float64).tolist()))
    return np.stack(expected), coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--case', choices=[c['case'] for c in CASES], default='ensemble_four_days')
    parser.add_argument('--backend', choices=['numpy', 'both'], default='both')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--profile-repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2 or args.warmup < 1 or args.profile_repeats < 2:
        parser.error('Need new output path, >=2 repeats/profile repeats and >=1 warmup')
    case = next(c for c in CASES if c['case'] == args.case)
    manifest = json.loads((args.data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(args.data)
    if digest != manifest['files']['batch.zarr']:
        raise ValueError('Source data hash mismatch')
    torch = None
    sync = lambda: None
    if args.backend == 'both':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; no silent CPU fallback')
        torch.cuda.set_device(0)
        torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        sync = torch.cuda.synchronize
    tasks = regions(case['side'], case['side'])
    truth, coverage = expected_metrics(args.data, case, tasks)
    # Both backends receive the same reuse opportunity. Setup is excluded only
    # for explicitly named reuse_shared, and included in every once_* workflow.
    ds = load_region(args.data, case)
    cpu_cache = prepare(ds, 'shared')
    gpu_cache = upload(cpu_cache) if torch else None
    cache_bytes = dict(shared_cpu=cpu_cache.logical_bytes,
                       shared_cuda=gpu_cache.logical_bytes if gpu_cache else None,
                       dense_logical=3*int(np.prod(ds.forecast.shape))*4,
                       all_finite_fast_path=cpu_cache.mask is None)
    env = dict(python=platform.python_version(), numpy=np.__version__, xarray=xr.__version__,
               torch=torch.__version__ if torch else None, cuda_runtime=torch.version.cuda if torch else None,
               case=case, source_sha256=digest, source_manifest=manifest, regions=coverage,
               repeats=args.repeats, warmup=args.warmup, profile_repeats=args.profile_repeats, seed=args.seed,
               cache_bytes=cache_bytes, task_counts=[1, 5], tolerance=dict(rtol=1e-5, atol=1e-5),
               timing='Uninstrumented end-to-end samples are primary. Separate stage profiles synchronize CUDA at boundaries; profiling overhead and synchronization can change timings. All reads are warm local Zarr, not cold storage. CUDA initialization, source hashing, correctness validation, output writing and explicit reuse cache setup excluded. once_* includes read, preparation and upload every invocation.',
               semantics='Same individual-init/lead/member spatial Bias/MAE/RMSE for full domain then four disjoint quadrants. Cache contains inputs, not metric outputs. Each task is recomputed. GFS analysis is model-dependent, not independent observations.',
               before=None, after=None)
    if torch:
        from precision_benchmark import hardware_snapshot
        env['before'] = hardware_snapshot(torch)
    del ds
    args.output.mkdir(parents=True)
    timing, profiles, metrics = [], [], []
    def check(result, expected):
        if result.shape != expected.shape:
            raise ValueError('Missing regional/group metric output')
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for count in env['task_counts']:
        selected, expected = tasks[:count], truth[:count]
        methods = {}
        for backend, gpu in [('cpu', False)]+([('cuda', True)] if torch else []):
            for mode in ['reopen_dense', 'once_dense', 'once_shared', 'reuse_shared']:
                methods[backend+'_'+mode] = (gpu, mode)
        for name, (gpu, mode) in methods.items():
            for _ in range(args.warmup):
                check(run_workflow(args.data, case, selected, mode, gpu, gpu_cache if gpu else cpu_cache), expected)
        rng = np.random.default_rng(args.seed+count)
        for repeat in range(args.repeats):
            for order, name in enumerate(rng.permutation(list(methods))):
                gpu, mode = methods[name]
                sync()
                start = time.perf_counter()
                result = run_workflow(args.data, case, selected, mode, gpu, gpu_cache if gpu else cpu_cache)
                sync()
                elapsed = time.perf_counter()-start
                check(result, expected)
                timing.append(dict(tasks=count, method=str(name), repeat=repeat, order=order,
                                   seconds=elapsed, max_abs_metric_error=float(np.max(np.abs(result-expected)))))
        # Profile separately so extra phase synchronization is never presented
        # as the uninstrumented end-to-end measurement.
        rng = np.random.default_rng(args.seed+100+count)
        for repeat in range(args.profile_repeats):
            for order, name in enumerate(rng.permutation(list(methods))):
                gpu, mode = methods[name]
                trace = Trace(sync if gpu else lambda: None)
                sync()
                start = time.perf_counter()
                result = run_workflow(args.data, case, selected, mode, gpu, gpu_cache if gpu else cpu_cache, trace)
                sync()
                elapsed = time.perf_counter()-start
                check(result, expected)
                residual = elapsed-sum(trace.seconds.values())
                if residual < 0:
                    raise ValueError('Overlapping or invalid stage timers')
                for label, seconds in {**trace.seconds, 'other': residual, 'profile_total': elapsed}.items():
                    profiles.append(dict(tasks=count, method=str(name), repeat=repeat, order=order,
                                         stage=label, seconds=seconds))
        for index, (region, _) in enumerate(selected):
            for group, values in enumerate(expected[index]):
                metrics.append(dict(tasks=count, region=region, group=group,
                                    **dict(zip(METRICS, map(float, values)))))
        summaries = {name: float(np.median([r['seconds'] for r in timing if r['tasks'] == count and r['method'] == name]))*1000 for name in methods}
        print(json.dumps(dict(tasks=count, median_ms=summaries)), flush=True)
    summary = []
    for count in env['task_counts']:
        for name in methods:
            values = [r['seconds'] for r in timing if r['tasks'] == count and r['method'] == name]
            summary.append(dict(tasks=count, method=name, median_sec=float(np.median(values)),
                                p25_sec=float(np.percentile(values, 25)), p75_sec=float(np.percentile(values, 75))))
    if torch:
        env['after'] = hardware_snapshot(torch)
    write_csv(args.output/'timing_samples.csv', timing)
    write_csv(args.output/'timing_summary.csv', summary)
    write_csv(args.output/'stage_samples.csv', profiles)
    write_csv(args.output/'reference_metrics.csv', metrics)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: {len(timing)} end-to-end samples and all stage-profile outputs agree with FP64', flush=True)


if __name__ == '__main__':
    main()
