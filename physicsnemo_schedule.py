"""Paired same-batch versus alternating-batch FNO requests; separate CUDA traces."""
import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from physicsnemo_benchmark import (MODEL_CONFIG, TOLERANCE, array_mapping_hash,
    double_reference, error_summary, load_input, make_model, parse_cases, stats)
from weather_dispatch import merged_duration
from weather_verify import sha256, write_csv


CASES = parse_cases(['16x64', '64x64'])
SCHEDULES = ['grouped', 'alternating']


def execution_plan(rounds, calls_per_case, seed):
    """Counterbalance first shape/schedule; both schedules share first shape per pair."""
    if rounds < 1 or calls_per_case < 2 or seed < 0:
        raise ValueError('Positive rounds, >=2 calls per case, nonnegative seed required')
    rng = np.random.default_rng(seed)
    # All four first-shape/first-schedule combinations balance over every 4 rounds.
    combinations = rng.permutation(np.resize(np.array([0, 3, 1, 2]), rounds))
    first_cases, first_schedules = combinations % 2, combinations // 2
    names = [c['case'] for c in CASES]
    result = []
    for repeat in range(rounds):
        a, b = names[int(first_cases[repeat])], names[1-int(first_cases[repeat])]
        for order in range(2):
            schedule = SCHEDULES[(int(first_schedules[repeat])+order)%2]
            sequence = [a]*calls_per_case+[b]*calls_per_case if schedule == 'grouped' else [a, b]*calls_per_case
            result.append(dict(round=repeat, block_order=order, schedule=schedule, first_case=a, sequence=sequence))
    return result


def sequence_labels(sequence):
    # Common alternating warmup always ends with B64, for both schedules.
    previous = CASES[-1]['case']
    rows = []
    for position, name in enumerate(sequence):
        rows.append(dict(position=position, case=name, previous_case=previous, switched=int(name != previous)))
        previous = name
    return rows


def summarize_trace(document):
    events = [e for e in document['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    roots = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('shape_call/')], key=lambda e: e['ts'])
    rows = []
    for root in roots:
        _, schedule, repeat, position, case = root['name'].split('/')
        left, right = root['ts'], root['ts']+root['dur']
        inside = [e for e in events if e['ts'] >= left-.01 and e['ts']+e['dur'] <= right+.01]
        def intervals(category, predicate=lambda e: True):
            return [(e['ts'], e['ts']+e['dur']) for e in inside if e.get('cat') == category and predicate(e)]
        kernels, copies = intervals('kernel'), intervals('gpu_memcpy')
        annotations = [e for e in inside if e.get('cat') == 'user_annotation']
        dtoh = [e for e in inside if e.get('cat') == 'gpu_memcpy' and 'DtoH' in e['name']]
        htod = [e for e in inside if e.get('cat') == 'gpu_memcpy' and 'HtoD' in e['name']]
        rows.append(dict(schedule=schedule, round=int(repeat), position=int(position), case=case,
            root_wall_ms=root['dur']/1000,
            dispatch_ms=sum(e['dur'] for e in annotations if e['name'] == 'shape_dispatch')/1000,
            return_ms=sum(e['dur'] for e in annotations if e['name'] == 'shape_return')/1000,
            kernel_union_ms=merged_duration(kernels, left, right)/1000,
            copy_union_ms=merged_duration(copies, left, right)/1000,
            active_union_ms=merged_duration(kernels+copies, left, right)/1000,
            runtime_launch_ms=merged_duration(intervals('cuda_runtime', lambda e: 'LaunchKernel' in e['name']), left, right)/1000,
            runtime_wait_ms=merged_duration(intervals('cuda_runtime', lambda e: 'Synchronize' in e['name']), left, right)/1000,
            kernel_events=len(kernels), dtoh_copies=len(dtoh),
            dtoh_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in dtoh),
            htod_copies=len(htod), htod_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in htod)))
    if not rows:
        raise ValueError('Trace has no annotated shape calls')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--calls-per-case', type=int, default=8)
    parser.add_argument('--trace-rounds', type=int, default=3)
    parser.add_argument('--trace-calls-per-case', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.calls_per_case, args.trace_rounds, args.trace_calls_per_case) < 2 or min(args.warmup, args.threads, args.seed) < 1:
        parser.error('New output, >=2 rounds/calls/trace rounds/calls, positive warmup/threads/seed required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        parser.error('CUDA and CPU/CUDA profiler support required; no fallback')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((args.baseline/'environment.json').read_text())
    manifest = json.loads((args.data.parent/'manifest.json').read_text())
    source_hash = sha256(args.data)
    if source_hash != baseline['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or baseline['model_config'] != MODEL_CONFIG:
        raise ValueError('Baseline/source/model mismatch')
    weights_path = args.baseline/'weights.npz'
    if sha256(weights_path) != baseline['weights_sha256']:
        raise ValueError('Weight archive mismatch')
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    if array_mapping_hash(weights) != baseline['model_state_sha256']:
        raise ValueError('Tensor state mismatch')
    model = make_model()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    reference_model = double_reference(model)
    gpu_model = copy.deepcopy(model).cuda()
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    shutil.copyfile(weights_path, args.output/'weights.npz')
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), scope='Fixed untrained FNO shape-schedule performance, no forecast assessment',
        python=platform.python_version(), versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr']},
        cuda_runtime=torch.version.cuda, hardware=hardware_snapshot(torch),
        source_sha256=source_hash, source_manifest=manifest, model_config=MODEL_CONFIG,
        model_state_sha256=baseline['model_state_sha256'], weights_sha256=baseline['weights_sha256'],
        cases=CASES, schedules=SCHEDULES, rounds=args.rounds, calls_per_case=args.calls_per_case,
        trace_rounds=args.trace_rounds, trace_calls_per_case=args.trace_calls_per_case,
        warmup=args.warmup, threads=args.threads, interop_threads=1, seed=args.seed,
        tolerance=TOLERANCE, tf32=False, deterministic_algorithms=True,
        primary='No profiler, annotations or CUDA events. Resident identical inputs/model, sequential forward/full host return, no files/read/pack/upload within timing. Per-call wall time and direct block wall time recorded. Same number of B16/B64 calls, counterbalanced first shape and schedule. Source/reference/cache setup excluded.',
        ordering='Every paired round shares first shape for grouped/alternating schedules. Before each block the identical fixed alternating warmup ends with B64. Output checks/hashes deferred until block ends. All outputs retained on host until checks, bounded to one block plus representatives. Loop overhead is included in block time and recorded as residual relative to per-call sum.',
        profiler='Separate later CPU/CUDA session, same block warmups, shorter trace plan and different seed. shape_dispatch includes Python/ATen/driver submission and waits; shape_return includes preceding-work completion and actual copy. Kernel/copy intervals overlap CPU ranges; union durations are not added to wall time. GPU active union is not occupancy.',
        limits='Batch dimension changes (16 vs 64), spatial grid stays 64x64. Input reuse assumes identical frames/transform. No driver/clock/power/precision/compiler setting changed. Cold startup, I/O throughput, service queueing, prediction quality and scheduling real distinct jobs are not measured. Profiler effects and system-state changes prevent replacing primary timing with trace timing.',
        case_metadata={})
    fixtures, representatives, samples, blocks, trace_checks = {}, {}, [], [], []
    with torch.inference_mode():
        for case in CASES:
            name = case['case']
            x, identity = load_input(args.data, case)
            artifact = args.baseline/(name+'.npz')
            digest = sha256(artifact)
            if digest != baseline['case_metadata'][name]['artifact_sha256']:
                raise ValueError('Baseline case mismatch')
            with np.load(artifact, allow_pickle=False) as saved:
                np.testing.assert_array_equal(x, saved['input_fp32'])
                truth = saved['reference_fp64'].copy()
            fresh = reference_model(torch.from_numpy(x).double()).numpy()
            np.testing.assert_allclose(fresh, truth, rtol=1e-10, atol=1e-11)
            fixtures[name] = dict(input=torch.from_numpy(x).cuda(), truth=truth)
            env['case_metadata'][name] = dict(identity=identity, input_sha256=array_mapping_hash({'input': x}),
                baseline_artifact_sha256=digest, input_bytes=x.nbytes, output_bytes=x.nbytes)
        def invoke(name, annotate=False):
            with torch.profiler.record_function('shape_dispatch') if annotate else nullcontext():
                result = gpu_model(fixtures[name]['input'])
            with torch.profiler.record_function('shape_return') if annotate else nullcontext():
                return result.cpu().numpy()
        def warmup():
            outputs = [(c['case'], invoke(c['case'])) for _ in range(args.warmup) for c in CASES]
            for name, output in outputs:
                error_summary(output, fixtures[name]['truth'])
            torch.cuda.synchronize()
        def checked(row, output):
            name = row['case']
            row.update(error_summary(output, fixtures[name]['truth']), output_sha256=array_mapping_hash({'output': output}))
            representatives[name] = output
            return row
        for block in execution_plan(args.rounds, args.calls_per_case, args.seed):
            warmup()
            pending = []
            block_start = time.perf_counter_ns()
            for label in sequence_labels(block['sequence']):
                start = time.perf_counter_ns()
                output = invoke(label['case'])
                elapsed = (time.perf_counter_ns()-start)/1e9
                pending.append((dict(round=block['round'], block_order=block['block_order'], schedule=block['schedule'], **label, seconds=elapsed), output))
            block_seconds = (time.perf_counter_ns()-block_start)/1e9
            call_seconds = sum(row['seconds'] for row, _ in pending)
            blocks.append(dict(round=block['round'], block_order=block['block_order'], schedule=block['schedule'], first_case=block['first_case'],
                               seconds=block_seconds, sum_calls_sec=call_seconds, residual_sec=block_seconds-call_seconds))
            samples.extend(checked(row, output) for row, output in pending)
        print(f'PASS: {len(samples)} profiler-free shape-schedule outputs checked', flush=True)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            for block in execution_plan(args.trace_rounds, args.trace_calls_per_case, args.seed+1000):
                warmup()
                pending = []
                for label in sequence_labels(block['sequence']):
                    name, position = label['case'], label['position']
                    with torch.profiler.record_function(f'shape_call/{block["schedule"]}/{block["round"]}/{position}/{name}'):
                        output = invoke(name, True)
                    pending.append((dict(round=block['round'], block_order=block['block_order'], schedule=block['schedule'], **label), output))
                trace_checks.extend(checked(row, output) for row, output in pending)
                profiler.step()
        trace_path = args.output/'trace.json'
        profiler.export_chrome_trace(str(trace_path))
        raw = trace_path.read_bytes()
        trace_rows = summarize_trace(json.loads(raw))
        checks = {(r['schedule'], r['round'], r['position'], r['case']): r for r in trace_checks}
        if len(trace_rows) != len(checks):
            raise ValueError('Trace/check call count mismatch')
        for row in trace_rows:
            check = checks[(row['schedule'], row['round'], row['position'], row['case'])]
            row.update({k: v for k, v in check.items() if k not in row})
            if row['kernel_events'] < 1 or row['dtoh_copies'] != 1 or row['dtoh_bytes'] != env['case_metadata'][row['case']]['output_bytes']:
                raise ValueError('Trace missing kernels or exact full-output DtoH copy')
        with gzip.open(args.output/'trace.json.gz', 'wb') as stream:
            stream.write(raw)
        trace_path.unlink()
        env['trace_sha256'] = sha256(args.output/'trace.json.gz')
        for name, output in representatives.items():
            path = args.output/'outputs'/(name+'.npy')
            with path.open('xb') as stream:
                np.save(stream, output, allow_pickle=False)
            env['case_metadata'][name]['output_path'] = path.relative_to(args.output).as_posix()
            env['case_metadata'][name]['output_sha256'] = sha256(path)
        summaries = [dict(schedule=s, case=c['case'], **stats([r for r in samples if r['schedule'] == s and r['case'] == c['case']])) for s in SCHEDULES for c in CASES]
        for filename, records in [('timing_samples.csv', samples), ('timing_summary.csv', summaries), ('block_samples.csv', blocks), ('trace_calls.csv', trace_rows)]:
            write_csv(args.output/filename, records)
        env['hardware_after'] = hardware_snapshot(torch)
        (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
        print(f'PASS: {len(samples)} primary and {len(trace_rows)} profiled calls; full FP64 agreement, real CUDA kernels/copy bytes captured', flush=True)


if __name__ == '__main__':
    main()
