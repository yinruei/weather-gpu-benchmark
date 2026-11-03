"""CPU/eager/graph engines: genuine Zarr input -> FP32 FNO -> full NPZ file."""
import argparse
from contextlib import contextmanager, nullcontext
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
import xarray as xr

from physicsnemo_benchmark import (MODEL_CONFIG, TOLERANCE, array_mapping_hash,
    double_reference, error_summary, make_model, prepare_input, read_from_dataset,
    read_input, stats)
from physicsnemo_graph import CASES, CudaGraphRunner, load_variant
from physicsnemo_pipeline import STAGES, StageClock, output_payload, stage, verify_output, write_output
from weather_dispatch import merged_duration
from weather_verify import sha256, write_csv
from physicsnemo_reader import READER_POLICIES, read_exact, read_gather, read_coverage


ENGINES = ['cpu', 'eager', 'graph']
READERS = list(READER_POLICIES)
DEFAULT_READERS = ['open_once', 'reopen']


def methods(engines, readers):
    if not engines or not readers or len(set(engines)) != len(engines) or len(set(readers)) != len(readers) or any(e not in ENGINES for e in engines) or any(r not in READERS for r in readers):
        raise ValueError('Unique supported engines/readers required')
    return [e+'_'+r for r in readers for e in engines]


def request_order(engines, readers, repeats, seed):
    jobs = [(c['case'], m) for c in CASES for m in methods(engines, readers)]
    rng = np.random.default_rng(seed)
    return [dict(repeat=repeat, order=order, case=jobs[int(i)][0], method=jobs[int(i)][1], variant=repeat % 2)
            for repeat in range(repeats) for order, i in enumerate(rng.permutation(len(jobs)))]


def read_variant(data, case, variant, dataset=None, *, selection='bounding'):
    if selection == 'gather':
        return read_gather(data, case, variant, dataset)
    if selection == 'exact':
        return read_exact(data, case, variant, dataset)
    if selection != 'bounding':
        raise ValueError('Unknown frame selection')
    if variant not in [0, 1]:
        raise ValueError('Two disjoint genuine input variants required')
    count = case['batch']
    extended = dict(case, batch=count*(variant+1))
    raw, identity, coords = read_input(data, extended) if dataset is None else read_from_dataset(dataset, extended)
    start = count*variant
    return raw[start:start+count], dict(identity, frames=identity['frames'][start:start+count]), coords


@contextmanager
def workflow_stage(clock, annotate, name):
    if annotate == 'nvtx':
        import torch
        scope = torch.cuda.nvtx.range('file_'+name)
    elif annotate:
        import torch
        scope = torch.profiler.record_function('file_'+name)
    else:
        scope = nullcontext()
    with stage(clock, name), scope:
        yield


class FnoFilePipeline:
    """Selectable engine/reader; requests always reread/pack input and write a file.

    Models must stay in eval mode; graph/input-buffer lifetimes belong to caller.
    A supplied dataset is shared and is not closed by this worker. Without one,
    open_once owns its handle; use the context manager or close().
    """
    def __init__(self, data, case, model, engine='eager', reader='open_once',
                 runner=None, device_input=None, dataset=None):
        methods([engine], [reader])
        if model.training:
            raise ValueError('Inference eval model required')
        if engine == 'graph' and runner is None:
            raise ValueError('Graph engine requires a captured runner; no silent fallback')
        if engine == 'graph' and runner.model is not model:
            raise ValueError('Graph runner and requested model must be the same object')
        self.data, self.case, self.model = Path(data), dict(case), model
        self.engine, self.reader, self.runner = engine, reader, runner
        self.device_input = runner.input if runner is not None else device_input
        if engine != 'cpu':
            import torch
            shape = (case['batch'], 1, case['side'], case['side'])
            if self.device_input is None or tuple(self.device_input.shape) != shape or self.device_input.dtype != torch.float32 or not self.device_input.is_cuda:
                raise ValueError('Matching CUDA FP32 input buffer required')
        self.policy = READER_POLICIES[reader]
        self.owns_dataset = self.policy['reuse'] and dataset is None
        start = time.perf_counter()
        self.dataset = (xr.open_zarr(data, chunks=None, consolidated=True) if self.owns_dataset else dataset) if self.policy['reuse'] else None
        self.reader_setup_sec = time.perf_counter()-start

    def close(self):
        if self.owns_dataset and self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def execute(self, variant, path, source_hash, state_hash, clock=None, annotate=False):
        import torch
        if self.policy['reuse'] and self.dataset is None:
            raise ValueError('Open-once worker is closed')
        start = time.perf_counter_ns()
        with torch.inference_mode():
            with workflow_stage(clock, annotate, 'read_decode'):
                raw, identity, coordinates = read_variant(self.data, self.case, variant, self.dataset, selection=self.policy['selection'])
            with workflow_stage(clock, annotate, 'prepare'):
                packed = prepare_input(raw)
                tensor = torch.from_numpy(packed)
                payload = output_payload(None, identity, coordinates, self.case, source_hash, state_hash)
            if self.engine != 'cpu':
                with workflow_stage(clock, annotate, 'upload'):
                    if self.runner is not None:
                        self.runner.copy_input(tensor)
                    else:
                        self.device_input.copy_(tensor)
                    tensor = self.device_input
            with workflow_stage(clock, annotate, 'forward'):
                result = self.runner.replay() if self.engine == 'graph' else self.model(tensor)
            with workflow_stage(clock, annotate, 'return'):
                payload['model_output'] = result.cpu().numpy() if self.engine != 'cpu' else result.numpy()
            with workflow_stage(clock, annotate, 'write_close'):
                write_output(path, payload)
        return (time.perf_counter_ns()-start)/1e9, packed, identity, payload


def summarize_trace(document):
    events = [e for e in document['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    roots = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('file_call/')], key=lambda e: e['ts'])
    result = []
    for root in roots:
        _, case, method, repeat, variant = root['name'].split('/')
        left, right = root['ts'], root['ts']+root['dur']
        inside = [e for e in events if e['ts'] >= left-.01 and e['ts']+e['dur'] <= right+.01]
        annotations = [e for e in inside if e.get('cat') == 'user_annotation']
        kernels = [(e['ts'], e['ts']+e['dur']) for e in inside if e.get('cat') == 'kernel']
        copies = [e for e in inside if e.get('cat') == 'gpu_memcpy']
        runtime = [e for e in inside if e.get('cat') == 'cuda_runtime']
        dtoh, htod = [[e for e in copies if d in e['name']] for d in ['DtoH', 'HtoD']]
        row = dict(case=case, method=method, repeat=int(repeat), variant=int(variant), root_wall_ms=root['dur']/1000,
            kernel_events=len(kernels), active_union_ms=merged_duration(kernels+[(e['ts'], e['ts']+e['dur']) for e in copies], left, right)/1000,
            graph_launches=sum('cudaGraphLaunch' in e['name'] for e in runtime),
            dtoh_copies=len(dtoh), dtoh_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in dtoh),
            htod_copies=len(htod), htod_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in htod))
        row.update({s+'_ms': sum(e['dur'] for e in annotations if e['name'] == 'file_'+s)/1000 for s in STAGES})
        result.append(row)
    if not result:
        raise ValueError('No full-file workflow trace calls')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    parser.add_argument('--engines', nargs='+', choices=ENGINES, default=ENGINES)
    parser.add_argument('--readers', nargs='+', choices=READERS, default=DEFAULT_READERS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--profile-repeats', type=int, default=6)
    parser.add_argument('--trace-repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261111)
    args = parser.parse_args()
    chosen = methods(args.engines, args.readers)
    if args.output.exists() or min(args.repeats, args.profile_repeats, args.trace_repeats) < 2 or min(args.warmup, args.threads, args.seed) < 1:
        parser.error('New output, >=2 repeat/profile/trace counts and positive warmup/threads/seed required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    start_setup = time.perf_counter()
    import torch
    has_gpu = any(e != 'cpu' for e in args.engines)
    if has_gpu and (not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()):
        parser.error('Requested engines need CUDA and CPU/CUDA profiler; no fallback')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ref = json.loads((args.reference_run/'environment.json').read_text())
    manifest = json.loads((args.data.parent/'manifest.json').read_text())
    source_hash = sha256(args.data)
    if source_hash != ref['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or ref['model_config'] != MODEL_CONFIG or ref['tolerance'] != TOLERANCE:
        raise ValueError('Reference/source/model protocol mismatch')
    weights_path = args.reference_run/'weights.npz'
    if sha256(weights_path) != ref['weights_sha256']:
        raise ValueError('Reference weights changed')
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    state_hash = array_mapping_hash(weights)
    if state_hash != ref['model_state_sha256']:
        raise ValueError('Tensor state changed')
    cpu_model = make_model()
    cpu_model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    reference_model = double_reference(cpu_model)
    gpu_model = copy.deepcopy(cpu_model).cuda() if has_gpu else None
    sync = torch.cuda.synchronize if has_gpu else lambda: None
    sync()
    model_setup_sec = time.perf_counter()-start_setup
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    shutil.copyfile(weights_path, args.output/'weights.npz')
    start = time.perf_counter()
    shared_dataset = xr.open_zarr(args.data, chunks=None, consolidated=True) if any(READER_POLICIES[r]['reuse'] for r in args.readers) else None
    reader_setup_sec = time.perf_counter()-start
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), scope='Complete local Zarr-to-file CPU/eager/graph inference performance; untrained FNO',
        python=platform.python_version(), versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr']},
        hardware=hardware_snapshot(torch) if has_gpu else dict(cpu_model=platform.processor(), logical_cpus=os.cpu_count()),
        cuda_runtime=torch.version.cuda if has_gpu else None, source_sha256=source_hash, source_manifest=manifest,
        model_config=MODEL_CONFIG, model_state_sha256=state_hash, weights_sha256=ref['weights_sha256'], reference_run=args.reference_run.name,
        cases=CASES, engines=args.engines, readers=args.readers, methods=chosen, stages=STAGES,
        reader_policies={r: READER_POLICIES[r] for r in args.readers},
        read_coverage={c['case']: {selection: [read_coverage(args.data, c, v, selection) for v in [0, 1]]
            for selection in sorted({READER_POLICIES[r]['selection'] for r in args.readers})} for c in CASES},
        repeats=args.repeats, profile_repeats=args.profile_repeats, trace_repeats=args.trace_repeats,
        warmup=args.warmup, threads=args.threads, interop_threads=1, seed=args.seed, tolerance=TOLERANCE,
        tf32=False, deterministic_algorithms=True, model_setup_seconds=model_setup_sec, shared_reader_setup_seconds=reader_setup_sec,
        timing='Direct wall time includes genuine variant read/decode/coordinate validation, FP32 scaling/packing, output metadata, host tensor, upload/copy to fixed GPU input (GPU engines), FNO, full host return, uncompressed NPZ write/close. Models remain resident. open_once/exact_once reuse one shared handle; reopen/exact_reopen open each request. Bounding and exact selection are explicitly recorded in reader_policies. No prepared input/output cache. Primary has no profiler/events/stage synchronization. Readback/hash/FP64 validation/cleanup outside timers.',
        setup='Common imports/source-hash/model loading and shared reader opening are recorded separately. Graph setup includes clone, side-stream warmup, capture/instantiate and first replay; net allocator changes observed after eager warmup. All graphs/source handles coexist even during eager/CPU timings. Setup is excluded from primary; estimated amortization is not measured cold startup.',
        variants='Same two disjoint real windows as pinned reference run: A first B frames, B next B frames. Variant alternates by repeat for all matched engine/reader cases. Each request actually rereads and prepares its requested variant.',
        limits='OS/library caches uncontrolled and warmed by setup/reference/readback. File close is not fsync/durable media persistence. No async overlap, remote retrieval, precision change, training or forecast assessment. Stage profiles and traces are separate later observations and may differ from primary times. GPU allocator probes are not total VRAM or standalone eager memory.', case_metadata={})
    fixtures, workers, samples, profiles, first, memory, trace_checks = {}, {}, [], [], [], [], []
    with torch.inference_mode():
        for case in CASES:
            name = case['case']
            values, identities = zip(*(load_variant(args.data, case, v) for v in [0, 1]))
            reference_path = args.reference_run/ref['case_metadata'][name]['reference_path']
            if sha256(reference_path) != ref['case_metadata'][name]['reference_sha256']:
                raise ValueError('Pinned FP64 archive changed')
            with np.load(reference_path, allow_pickle=False) as saved:
                truths = [saved['reference_'+v].copy() for v in ['A', 'B']]
                for i, label in enumerate(['A', 'B']):
                    np.testing.assert_array_equal(values[i], saved['input_'+label])
                    fresh = reference_model(torch.from_numpy(values[i]).double()).numpy()
                    np.testing.assert_allclose(fresh, truths[i], rtol=1e-10, atol=1e-11)
            graph_setup, graph_memory, runner, device_input = None, None, None, None
            if has_gpu:
                device_input = torch.from_numpy(values[0]).cuda()
                for _ in range(args.warmup):
                    temporary = gpu_model(device_input).cpu().numpy()
                    error_summary(temporary, truths[0])
                del temporary
                sync()
                if 'graph' in args.engines:
                    before_a, before_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                    runner = CudaGraphRunner(gpu_model, device_input, args.warmup)
                    after_a, after_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
                    graph_setup = runner.setup_seconds
                    graph_memory = dict(before_allocated_bytes=before_a, after_allocated_bytes=after_a, extra_allocated_bytes=after_a-before_a,
                        before_reserved_bytes=before_r, after_reserved_bytes=after_r, extra_reserved_bytes=after_r-before_r,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            fixtures[name] = dict(inputs=values, identities=identities, truths=truths)
            env['case_metadata'][name] = dict(identities=list(identities), input_hashes=[array_mapping_hash({'input': x}) for x in values], input_bytes=values[0].nbytes,
                reference_path=reference_path.name, reference_sha256=sha256(reference_path), graph_setup_seconds=graph_setup, graph_setup_memory=graph_memory, outputs={})
            for method in chosen:
                engine, reader = method.split('_', 1)
                workers[(name, method)] = FnoFilePipeline(args.data, case, cpu_model if engine == 'cpu' else gpu_model,
                    engine=engine, reader=reader, runner=runner if engine != 'cpu' else None, device_input=device_input,
                    dataset=shared_dataset if READER_POLICIES[reader]['reuse'] else None)
        def trial(job, retain=False, clock=None, annotate=False):
            name, method, variant = job['case'], job['method'], job['variant']
            path = args.output/'outputs'/f'{name}_{variant}_{method}.npz' if retain else args.output/'trial.npz'
            sync()
            seconds, packed, identity, payload = workers[(name, method)].execute(variant, path, source_hash, state_hash, clock, annotate)
            np.testing.assert_array_equal(packed, fixtures[name]['inputs'][variant])
            if identity != fixtures[name]['identities'][variant]:
                raise ValueError('Workflow input identity changed')
            errors = verify_output(path, payload, fixtures[name]['truths'][variant])
            digest, logical, size = sha256(path), array_mapping_hash(payload), path.stat().st_size
            row = dict(**job, seconds=seconds, file_bytes=size, file_sha256=digest, payload_sha256=logical, **errors)
            if retain:
                env['case_metadata'][name]['outputs'][f'{variant}_{method}'] = dict(path=path.relative_to(args.output).as_posix(), sha256=digest, bytes=size, payload_sha256=logical)
            else:
                path.unlink()
            return row
        for case in CASES:
            for method in chosen:
                first.append(trial(dict(case=case['case'], method=method, variant=0, repeat=0, order=chosen.index(method))))
        for job in request_order(args.engines, args.readers, args.warmup, args.seed-1):
            trial(job)
        for job in request_order(args.engines, args.readers, args.repeats, args.seed):
            samples.append(trial(job, retain=job['repeat'] >= args.repeats-2))
        print(f'PASS: {len(samples)} complete Zarr-to-file primary outputs checked', flush=True)
        for job in request_order(args.engines, args.readers, args.profile_repeats, args.seed+1):
            clock = StageClock(sync if not job['method'].startswith('cpu_') else lambda: None)
            row = trial(job, clock=clock)
            row.update({s+'_sec': value for s, value in clock.seconds.items()})
            row['unattributed_sec'] = row['seconds']-sum(clock.seconds.values())
            profiles.append(row)
        for case in CASES:
            for method in chosen:
                if method.startswith('cpu_'):
                    continue
                job = dict(case=case['case'], method=method, variant=0, repeat=0, order=0)
                sync()
                baseline_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                trial(job)
                peak = torch.cuda.max_memory_allocated()
                memory.append(dict(case=case['case'], method=method, baseline_allocated_bytes=baseline_bytes, peak_allocated_bytes=peak,
                    peak_extra_allocated_bytes=peak-baseline_bytes, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        activities = [torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if has_gpu else [])
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            for job in request_order(args.engines, args.readers, args.trace_repeats, args.seed+1000):
                sync()
                with torch.profiler.record_function(f'file_call/{job["case"]}/{job["method"]}/{job["repeat"]}/{job["variant"]}'):
                    # Validation/cleanup must remain outside the annotated root too.
                    path = args.output/'trace_trial.npz'
                    seconds, packed, identity, payload = workers[(job['case'], job['method'])].execute(job['variant'], path, source_hash, state_hash, annotate=True)
                np.testing.assert_array_equal(packed, fixtures[job['case']]['inputs'][job['variant']])
                if identity != fixtures[job['case']]['identities'][job['variant']]:
                    raise ValueError('Profiled input identity changed')
                errors = verify_output(path, payload, fixtures[job['case']]['truths'][job['variant']])
                trace_checks.append(dict(**job, seconds=seconds, file_bytes=path.stat().st_size, file_sha256=sha256(path), payload_sha256=array_mapping_hash(payload), **errors))
                path.unlink()
                profiler.step()
        path = args.output/'trace.json'
        profiler.export_chrome_trace(str(path))
        raw = path.read_bytes()
        traced = summarize_trace(json.loads(raw))
        index = {(r['case'], r['method'], r['repeat'], r['variant']): r for r in trace_checks}
        if len(traced) != len(index):
            raise ValueError('Trace root count mismatch')
        for row in traced:
            row.update({k: v for k, v in index[(row['case'], row['method'], row['repeat'], row['variant'])].items() if k not in row})
            engine = row['method'].split('_', 1)[0]
            expected_bytes = env['case_metadata'][row['case']]['input_bytes']
            if engine == 'cpu':
                if row['kernel_events'] or row['graph_launches'] or row['dtoh_bytes'] or row['htod_bytes']:
                    raise ValueError('CPU workflow unexpectedly contains CUDA work')
            elif row['kernel_events'] < 1 or row['dtoh_copies'] != 1 or row['dtoh_bytes'] != expected_bytes or row['htod_bytes'] != expected_bytes or (engine == 'graph' and row['graph_launches'] < 1) or (engine == 'eager' and row['graph_launches'] != 0):
                raise ValueError('GPU file workflow missing upload, full output or expected graph launch')
        with gzip.open(args.output/'trace.json.gz', 'wb') as stream:
            stream.write(raw)
        path.unlink()
        env['trace_sha256'] = sha256(args.output/'trace.json.gz')
        summary = [dict(case=c['case'], method=m, **stats([r for r in samples if r['case'] == c['case'] and r['method'] == m])) for c in CASES for m in chosen]
        for filename, records in [('timing_samples.csv', samples), ('timing_summary.csv', summary), ('stage_samples.csv', profiles), ('first_observed.csv', first), ('trace_calls.csv', traced)]:
            write_csv(args.output/filename, records)
        if memory:
            write_csv(args.output/'memory_samples.csv', memory)
        env['hardware_after'] = hardware_snapshot(torch) if has_gpu else None
        (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
        for worker in workers.values():
            worker.close()
        if shared_dataset is not None:
            shared_dataset.close()
        print(f'PASS: {len(samples)} primary files, {len(profiles)} stage profiles, {len(traced)} complete-file traces; genuine A/B inputs and FP64 agreement', flush=True)


if __name__ == '__main__':
    main()
