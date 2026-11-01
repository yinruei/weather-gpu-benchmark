"""Measure latency distributions under blocked/interleaved regional workflows."""
import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from weather_batch import CASES
from weather_pipeline import expected_metrics, load_region, prepare, regions, run_workflow, upload
from weather_verify import sha256, write_csv

try:
    import resource
except ImportError:
    resource = None


def execution_order(methods, schedule, repeats, seed):
    rng = np.random.default_rng(seed)
    if schedule == 'blocked':
        return [(str(name), repeat, block) for block, name in enumerate(rng.permutation(methods))
                for repeat in range(repeats)]
    if schedule == 'interleaved':
        return [(str(name), repeat, order) for repeat in range(repeats)
                for order, name in enumerate(rng.permutation(methods))]
    raise ValueError('Unknown schedule')


def summarize(rows):
    values = np.array([float(row['seconds']) for row in rows])
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Expected positive finite timing samples')
    result = dict(samples=len(values), mean_sec=float(values.mean()), min_sec=float(values.min()), max_sec=float(values.max()))
    result.update({f'p{q}_sec': float(np.percentile(values, q)) for q in [50, 75, 90, 95, 99]})
    result['p95_over_p50'] = result['p95_sec']/result['p50_sec']
    return result


def process_snapshot():
    values = dict(gc_collections=sum(s['collections'] for s in gc.get_stats()))
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF)
        values.update(user_cpu_sec=r.ru_utime, system_cpu_sec=r.ru_stime,
                      voluntary_switches=r.ru_nvcsw, involuntary_switches=r.ru_nivcsw,
                      minor_faults=r.ru_minflt, major_faults=r.ru_majflt)
    return values


class Telemetry:
    """Optional separate nvidia-smi process; observations do not establish causes."""
    def __init__(self, output, enabled, interval):
        self.enabled, self.interval, self.process = enabled, interval, None
        self.output, self.streams = output, []
        self.info = dict(requested=enabled, interval_ms=interval,
                         timestamp_utc_offset_sec=datetime.now().astimezone().utcoffset().total_seconds(),
                         note='Asynchronous nvidia-smi sampling can perturb measurements; values are coarse device-wide context, not per-kernel counters.')

    def start(self):
        if not self.enabled:
            return
        raw = (self.output/'gpu_telemetry.csv').open('w', encoding='utf-8', newline='')
        errors = (self.output/'gpu_telemetry_stderr.txt').open('w', encoding='utf-8')
        self.streams = [raw, errors]
        fields = 'timestamp,index,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate'
        self.info['fields'] = fields.split(',')
        command = ['nvidia-smi', '--query-gpu='+fields, '--format=csv,noheader,nounits', f'--loop-ms={self.interval}']
        try:
            self.process = subprocess.Popen(command, stdout=raw, stderr=errors)
            self.info['launched'] = True
        except OSError as error:
            self.info.update(launched=False, error=str(error))

    def stop(self):
        if self.process is not None:
            running = self.process.poll() is None
            if running:
                self.process.terminate()
            self.info['exit_code'] = self.process.wait(timeout=5)
            self.info['stopped_by_runner'] = running
        for stream in self.streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--case', choices=[c['case'] for c in CASES], default='ensemble_four_days')
    parser.add_argument('--backend', choices=['numpy', 'both'], default='both')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--schedules', nargs='+', choices=['blocked', 'interleaved'], default=['blocked', 'interleaved'])
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--telemetry', action='store_true')
    parser.add_argument('--telemetry-ms', type=int, default=200)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2 or args.warmup < 1 or len(set(args.schedules)) != len(args.schedules) or args.telemetry_ms < 100:
        parser.error('Need new output, >=2 repeats, >=1 warmup, unique schedules and >=100ms telemetry interval')
    manifest = json.loads((args.data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(args.data)
    if digest != manifest['files']['batch.zarr']:
        raise ValueError('Source data hash mismatch')
    case = next(c for c in CASES if c['case'] == args.case)
    tasks = regions(case['side'], case['side'])
    expected, coverage = expected_metrics(args.data, case, tasks)
    cpu_cache = prepare(load_region(args.data, case), 'shared')
    torch, gpu_cache = None, None
    sync = lambda: None
    if args.backend == 'both':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; no silent CPU fallback')
        torch.cuda.set_device(0)
        gpu_cache = upload(cpu_cache)
        torch.cuda.synchronize()
        sync = torch.cuda.synchronize
    methods = {backend+'_'+mode: (gpu, mode) for backend, gpu in [('cpu', False)]+([('cuda', True)] if torch else [])
               for mode in ['once_shared', 'reuse_shared']}
    env = dict(python=platform.python_version(), numpy=np.__version__,
               torch=torch.__version__ if torch else None, cuda_runtime=torch.version.cuda if torch else None,
               case=case, source_sha256=digest, source_manifest=manifest, region_coverage=coverage,
               method_names=list(methods), schedules=args.schedules, repeats=args.repeats, warmup=args.warmup,
               seed=args.seed, started_utc=datetime.now(timezone.utc).isoformat(),
               timing='Wall latency includes complete five-region metric return. once_shared includes warm local read, validation, normalization, mask/weight preparation and upload when CUDA. reuse_shared excludes identical input-cache setup for both backends. Synchronize before/after each invocation; correctness/resource snapshots and file output outside timed interval.',
               schedule_semantics='Blocked: each method measured consecutively with warmups immediately before its block. Interleaved: warm each method, then shuffle all methods within each repetition. Same-method blocks still include result validation/return between invocations; not a continuously enqueued GPU-kernel throughput test.',
               limitations='Warm local data and initialized CUDA. Case/task shape fixed; both schedules in one process and listed order. Optional telemetry is coarse device-wide sampling and can perturb timing. Resource differences are descriptive process counters, not causal attribution.',
               tolerance=dict(rtol=1e-5, atol=1e-5), before=None, after=None)
    if torch:
        from precision_benchmark import hardware_snapshot
        env['before'] = hardware_snapshot(torch)
    args.output.mkdir(parents=True)
    monitor = Telemetry(args.output, args.telemetry, args.telemetry_ms)
    monitor.start()
    rows = []
    previous_method = 'none'
    def invoke(name):
        gpu, mode = methods[name]
        return run_workflow(args.data, case, tasks, mode, gpu, gpu_cache if gpu else cpu_cache)
    def check(values):
        if values.shape != expected.shape:
            raise ValueError('Missing regional/group outputs')
        np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-5)
    try:
        for schedule_index, schedule in enumerate(args.schedules):
            order = execution_order(list(methods), schedule, args.repeats, args.seed+schedule_index)
            if schedule == 'interleaved':
                for method in methods:
                    for _ in range(args.warmup):
                        check(invoke(method))
                        previous_method = method
            last_block = None
            for name, repeat, position in order:
                if schedule == 'blocked' and last_block != position:
                    for _ in range(args.warmup):
                        check(invoke(name))
                    previous_method, last_block = name, position
                before = process_snapshot()
                sync()
                unix_start_ns = time.time_ns()
                start_ns = time.perf_counter_ns()
                values = invoke(name)
                sync()
                elapsed_ns = time.perf_counter_ns()-start_ns
                after = process_snapshot()
                check(values)
                rows.append(dict(schedule=schedule, method=name, repeat=repeat, position=position,
                                 previous_method=previous_method, unix_start_ns=unix_start_ns,
                                 seconds=elapsed_ns/1e9, max_abs_metric_error=float(np.max(np.abs(values-expected))),
                                 **{k: after[k]-before[k] for k in before}))
                previous_method = name
            stat = {name: {k.replace('_sec', '_ms'): round(v*1000, 3) for k, v in summarize([r for r in rows if r['schedule'] == schedule and r['method'] == name]).items() if k in ['p50_sec', 'p95_sec', 'max_sec']}
                    for name in methods}
            print(json.dumps(dict(schedule=schedule, latency_ms=stat)), flush=True)
    finally:
        monitor.stop()
    env['telemetry'] = monitor.info
    env['ended_utc'] = datetime.now(timezone.utc).isoformat()
    if torch:
        env['after'] = hardware_snapshot(torch)
    summary = [dict(schedule=schedule, method=name, **summarize([r for r in rows if r['schedule'] == schedule and r['method'] == name]))
               for schedule in args.schedules for name in methods]
    write_csv(args.output/'timing_samples.csv', rows)
    write_csv(args.output/'timing_summary.csv', summary)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: all {len(rows)} measured five-region outputs agree with unchanged FP64 reference', flush=True)


if __name__ == '__main__':
    main()
