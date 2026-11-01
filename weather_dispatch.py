"""Diagnose CUDA dispatch/return waits and evaluate one return for five regions."""
import argparse
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import platform
import time

import numpy as np

from weather_batch import CASES
from weather_pipeline import expected_metrics, load_region, prepare, regions, run_workflow, spatial_subset, torch_shared, upload
from weather_stability import execution_order, summarize
from weather_verify import sha256, write_csv


def return_regions(prepared, tasks, return_once, annotate=False):
    """Keep region/group identity; optionally wait/return once after all work."""
    import torch
    def scope(name):
        return torch.profiler.record_function(name) if annotate else nullcontext()
    results = []
    for name, bounds in tasks:
        with scope('dispatch/'+name):
            result = torch_shared(spatial_subset(prepared, bounds))
        if return_once:
            results.append(result)
        else:
            with scope('return/'+name):
                results.append(result.cpu().numpy())
    if return_once:
        with scope('stack_regions'):
            result = torch.stack(results)
        with scope('return/all'):
            return result.cpu().numpy()
    with scope('stack_regions'):
        return np.stack(results)


def traced_workflow(data, case, tasks, cache, first_use, return_once, annotate=False):
    import torch
    def scope(name):
        return torch.profiler.record_function(name) if annotate else nullcontext()
    if first_use:
        with scope('host_prepare'):
            prepared = prepare(load_region(data, case), 'shared')
        with scope('upload_inputs'):
            prepared = upload(prepared)
    else:
        prepared = cache
    return return_regions(prepared, tasks, return_once, annotate)


def merged_duration(intervals, left, right):
    """Union within a call boundary; don't double-count concurrent GPU events."""
    intervals = sorted((max(left, a), min(right, b)) for a, b in intervals if b > left and a < right)
    total, previous_end = 0., left
    for start, end in intervals:
        total += max(0., end-max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def summarize_trace(document):
    """CPU and GPU durations overlap; they are never an additive wall partition."""
    events = [e for e in document['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    roots = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('call/')]
    rows = []
    for root in roots:
        _, schedule, method, repeat = root['name'].split('/')
        left, right = root['ts'], root['ts']+root['dur']
        inside = [e for e in events if e['ts'] >= left-.01 and e['ts']+e['dur'] <= right+.01]
        kernels = [(e['ts'], e['ts']+e['dur']) for e in inside if e.get('cat') == 'kernel']
        copies = [(e['ts'], e['ts']+e['dur']) for e in inside if e.get('cat') == 'gpu_memcpy']
        cpu = [e for e in inside if e.get('cat') == 'user_annotation' and e is not root]
        by_prefix = lambda prefix: sum(e['dur'] for e in cpu if e['name'].startswith(prefix))/1000
        row = dict(schedule=schedule, method=method, repeat=int(repeat), root_wall_ms=root['dur']/1000,
                   host_prepare_range_ms=by_prefix('host_prepare'), upload_range_ms=by_prefix('upload_inputs'),
                   dispatch_range_ms=by_prefix('dispatch/'), return_range_ms=by_prefix('return/'),
                   stack_range_ms=by_prefix('stack_regions'),
                   gpu_kernel_busy_ms=merged_duration(kernels, left, right)/1000,
                   gpu_copy_busy_ms=merged_duration(copies, left, right)/1000,
                   gpu_active_union_ms=merged_duration(kernels+copies, left, right)/1000,
                   kernel_events=len(kernels),
                   dtoh_copies=sum(e.get('cat') == 'gpu_memcpy' and 'DtoH' in e['name'] for e in inside),
                   return_annotations=sum(e['name'].startswith('return/') for e in cpu))
        rows.append(row)
    if not rows:
        raise ValueError('Trace has no annotated workflow calls')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--case', choices=[c['case'] for c in CASES], default='ensemble_four_days')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--trace-repeats', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--schedules', nargs='+', choices=['blocked', 'interleaved'], default=['blocked', 'interleaved'])
    args = parser.parse_args()
    if args.output.exists() or min(args.repeats, args.trace_repeats) < 2 or args.warmup < 1 or len(set(args.schedules)) != len(args.schedules):
        parser.error('Need new output, >=2 repeats/trace repeats, >=1 warmup and unique schedules')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA required; no CPU fallback')
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        parser.error('CPU/CUDA profiler activities required for this diagnostic')
    digest = sha256(args.data)
    manifest = json.loads((args.data.parent/'manifest.json').read_text(encoding='utf-8'))
    if digest != manifest['files']['batch.zarr']:
        raise ValueError('Source input hash mismatch')
    case = next(c for c in CASES if c['case'] == args.case)
    tasks = regions(case['side'], case['side'])
    truth, coverage = expected_metrics(args.data, case, tasks)
    host = prepare(load_region(args.data, case), 'shared')
    torch.cuda.set_device(0)
    cache = upload(host)
    torch.cuda.synchronize()
    methods = ['cpu_once_shared', 'cuda_first_per_region', 'cuda_first_once', 'cuda_reuse_per_region', 'cuda_reuse_once']
    def check(result):
        if result.shape != truth.shape:
            raise ValueError('Missing regional/group outputs')
        np.testing.assert_allclose(result, truth, rtol=1e-5, atol=1e-5)
    def invoke(method, annotate=False):
        if method == 'cpu_once_shared':
            return run_workflow(args.data, case, tasks, 'once_shared', False, host)
        if method.endswith('per_region') and not annotate:
            # Unchanged baseline implementation for primary timings.
            return run_workflow(args.data, case, tasks, 'once_shared' if 'first' in method else 'reuse_shared', True, cache)
        return traced_workflow(args.data, case, tasks, cache, 'first' in method, method.endswith('_once'), annotate)
    from precision_benchmark import hardware_snapshot
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda,
               case=case, source_sha256=digest, source_manifest=manifest, regions=coverage,
               methods=methods, schedules=args.schedules, repeats=args.repeats, trace_repeats=args.trace_repeats,
               warmup=args.warmup, seed=args.seed, before=hardware_snapshot(torch), after=None,
               output_shape=list(truth.shape), output_bytes=int(truth.size*4),
               timing='Primary wall samples run without profiler/record_function ranges; return all five regional/group metrics. First methods include warm local read/preparation/upload, reuse methods exclude input setup equally. Synchronize before/after each call; validate outside timers. CPU first-use method provides realistic interleaved host work.',
               profiler='Separate CPU/CUDA profiling session uses semantic annotations. Timings include profiler overhead and must not replace primary samples. CPU dispatch/return ranges include Python/driver work and waiting. GPU busy intervals are unions of trace kernel/copy events; overlapping CPU/GPU durations are not added.',
               changes='Numerical weighted FP32 kernel unchanged. once variants retain regional results on device, stack them and return once; baseline returns each region. No JAX, torch.compile, driver/power configuration or precision change.',
               tolerance=dict(rtol=1e-5, atol=1e-5))
    args.output.mkdir(parents=True)
    samples = []
    for schedule_index, schedule in enumerate(args.schedules):
        sequence = execution_order(methods, schedule, args.repeats, args.seed+schedule_index)
        if schedule == 'interleaved':
            for method in methods:
                for _ in range(args.warmup):
                    check(invoke(method))
        block = None
        for method, repeat, position in sequence:
            if schedule == 'blocked' and block != position:
                for _ in range(args.warmup):
                    check(invoke(method))
                block = position
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = invoke(method)
            torch.cuda.synchronize()
            seconds = (time.perf_counter_ns()-start)/1e9
            check(result)
            samples.append(dict(schedule=schedule, method=method, repeat=repeat, position=position,
                                seconds=seconds, max_abs_metric_error=float(np.max(np.abs(result-truth)))))
        print(json.dumps(dict(schedule=schedule, medians_ms={m: round(summarize([r for r in samples if r['method'] == m and r['schedule'] == schedule])['p50_sec']*1000, 3) for m in methods})), flush=True)
    # Warm all trace paths before activating the profiler, including stack kernel.
    for method in methods:
        for _ in range(args.warmup):
            check(invoke(method, True))
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as profiler:
        for schedule_index, schedule in enumerate(args.schedules):
            for method, repeat, position in execution_order(methods, schedule, args.trace_repeats, args.seed+100+schedule_index):
                torch.cuda.synchronize()
                with torch.profiler.record_function(f'call/{schedule}/{method}/{repeat}'):
                    result = invoke(method, True)
                    torch.cuda.synchronize()
                check(result)
                profiler.step()
    trace_path = args.output/'trace.json'
    profiler.export_chrome_trace(str(trace_path))
    raw = trace_path.read_bytes()
    document = json.loads(raw)
    trace_rows = summarize_trace(document)
    if not any(r['kernel_events'] > 0 for r in trace_rows if r['method'].startswith('cuda_')):
        raise ValueError('No CUDA kernel events captured; diagnostic cannot infer device activity')
    with gzip.open(args.output/'trace.json.gz', 'wb') as stream:
        stream.write(raw)
    trace_path.unlink()
    env['trace_sha256'] = sha256(args.output/'trace.json.gz')
    env['after'] = hardware_snapshot(torch)
    summaries = [dict(schedule=s, method=m, **summarize([r for r in samples if r['schedule'] == s and r['method'] == m])) for s in args.schedules for m in methods]
    write_csv(args.output/'timing_samples.csv', samples)
    write_csv(args.output/'timing_summary.csv', summaries)
    write_csv(args.output/'trace_calls.csv', trace_rows)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: {len(samples)} primary calls and {len(trace_rows)} profiled calls agree with unchanged FP64 reference', flush=True)


if __name__ == '__main__':
    main()
