"""Eager versus CUDA Graph FNO: full output, genuine input updates, setup/memory."""
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
METHODS = ['eager_resident', 'graph_resident', 'eager_device_update',
           'graph_device_update', 'eager_host_update', 'graph_host_update']


def load_variant(store, case, variant):
    if variant not in [0, 1]:
        raise ValueError('Two real, disjoint input variants supported')
    count = case['batch']
    extended = dict(case, batch=count*(variant+1))
    values, identity = load_input(store, extended)
    start = count*variant
    identity = dict(identity, frames=identity['frames'][start:start+count])
    return np.ascontiguousarray(values[start:start+count]), identity


def request_order(repeats, seed):
    jobs = [(c['case'], method) for c in CASES for method in METHODS]
    rng = np.random.default_rng(seed)
    return [dict(repeat=repeat, order=order, case=jobs[int(i)][0], method=jobs[int(i)][1],
                 variant=0 if jobs[int(i)][1].endswith('_resident') else repeat % 2)
            for repeat in range(repeats) for order, i in enumerate(rng.permutation(len(jobs)))]


class CudaGraphRunner:
    """Inference-only fixed-shape runner. Borrowed GPU output is overwritten on replay.

    Keep model/runner alive and weights fixed. Copy new contents into the persistent
    input, never replace its storage. Use on the same current stream as its copies.
    """
    def __init__(self, model, example, warmup=3):
        import torch
        if model.training or not example.is_cuda or example.dtype != torch.float32 or not example.is_contiguous() or warmup < 1:
            raise ValueError('Eval model, contiguous CUDA FP32 input and positive warmup required')
        self.model = model
        self.setup_seconds = {}
        start_all = time.perf_counter()
        with torch.inference_mode():
            start = time.perf_counter()
            self._input = example.detach().clone()
            torch.cuda.synchronize()
            self.setup_seconds['buffer_clone'] = time.perf_counter()-start
            self._stream = torch.cuda.Stream(device=example.device)
            self._stream.wait_stream(torch.cuda.current_stream(example.device))
            start = time.perf_counter()
            with torch.cuda.stream(self._stream):
                for _ in range(warmup):
                    temporary = model(self._input)
            torch.cuda.current_stream(example.device).wait_stream(self._stream)
            torch.cuda.synchronize()
            del temporary
            self.setup_seconds['side_stream_warmup'] = time.perf_counter()-start
            start = time.perf_counter()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._output = model(self._input)
            torch.cuda.synchronize()
            self.setup_seconds['capture_instantiate'] = time.perf_counter()-start
            start = time.perf_counter()
            self._graph.replay()
            torch.cuda.synchronize()
            self.setup_seconds['first_replay'] = time.perf_counter()-start
        self.setup_seconds['total'] = time.perf_counter()-start_all

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    def copy_input(self, value):
        import torch
        if value.shape != self._input.shape or value.dtype != self._input.dtype or not value.is_contiguous():
            raise ValueError('Graph input shape/dtype/contiguous layout must match capture')
        if value.device.type not in ['cpu', 'cuda'] or (value.is_cuda and value.device != self._input.device):
            raise ValueError('Input must be on CPU or the captured CUDA device')
        with torch.inference_mode():
            self._input.copy_(value)

    def replay(self):
        import torch
        with torch.inference_mode():
            self._graph.replay()
        return self._output


def summarize_trace(document):
    events = [e for e in document['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    roots = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('graph_call/')], key=lambda e: e['ts'])
    result = []
    for root in roots:
        _, case, method, repeat, variant = root['name'].split('/')
        left, right = root['ts'], root['ts']+root['dur']
        inside = [e for e in events if e['ts'] >= left-.01 and e['ts']+e['dur'] <= right+.01]
        def intervals(category):
            return [(e['ts'], e['ts']+e['dur']) for e in inside if e.get('cat') == category]
        kernels, copies = intervals('kernel'), intervals('gpu_memcpy')
        runtime = [e for e in inside if e.get('cat') == 'cuda_runtime']
        annotations = [e for e in inside if e.get('cat') == 'user_annotation']
        def copy_events(direction):
            return [e for e in inside if e.get('cat') == 'gpu_memcpy' and direction in e['name']]
        dtoh, htod, dtod = [copy_events(d) for d in ['DtoH', 'HtoD', 'DtoD']]
        result.append(dict(case=case, method=method, repeat=int(repeat), variant=int(variant), root_wall_ms=root['dur']/1000,
            update_ms=sum(e['dur'] for e in annotations if e['name'] == 'graph_input_update')/1000,
            dispatch_ms=sum(e['dur'] for e in annotations if e['name'] == 'graph_dispatch')/1000,
            return_ms=sum(e['dur'] for e in annotations if e['name'] == 'graph_return')/1000,
            kernel_events=len(kernels), kernel_union_ms=merged_duration(kernels, left, right)/1000,
            copy_union_ms=merged_duration(copies, left, right)/1000,
            active_union_ms=merged_duration(kernels+copies, left, right)/1000,
            graph_launches=sum('cudaGraphLaunch' in e['name'] for e in runtime),
            kernel_launches=sum('LaunchKernel' in e['name'] for e in runtime),
            dtoh_copies=len(dtoh), dtoh_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in dtoh),
            htod_copies=len(htod), htod_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in htod),
            dtod_copies=len(dtod), dtod_bytes=sum(int(e.get('args', {}).get('bytes', 0)) for e in dtod)))
    if not result:
        raise ValueError('No annotated graph calls in trace')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--trace-repeats', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    if args.output.exists() or min(args.repeats, args.trace_repeats) < 2 or min(args.warmup, args.threads, args.seed) < 1:
        parser.error('New output, >=2 repetitions/trace repetitions, positive warmup/threads/seed required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        parser.error('CUDA graph and CPU/CUDA profiler support required; no fallback')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    base = json.loads((args.baseline/'environment.json').read_text())
    manifest = json.loads((args.data.parent/'manifest.json').read_text())
    source_hash = sha256(args.data)
    if source_hash != base['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or base['model_config'] != MODEL_CONFIG:
        raise ValueError('Source/model baseline mismatch')
    weight_path = args.baseline/'weights.npz'
    if sha256(weight_path) != base['weights_sha256']:
        raise ValueError('Weight archive mismatch')
    with np.load(weight_path, allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    if array_mapping_hash(weights) != base['model_state_sha256']:
        raise ValueError('Model tensor state mismatch')
    model = make_model()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    reference_model, gpu_model = double_reference(model), copy.deepcopy(model).cuda()
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    shutil.copyfile(weight_path, args.output/'weights.npz')
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), scope='Fixed untrained FNO CUDA Graph inference performance; no forecast assessment',
        python=platform.python_version(), versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr']},
        hardware=hardware_snapshot(torch), cuda_runtime=torch.version.cuda,
        source_sha256=source_hash, source_manifest=manifest, model_config=MODEL_CONFIG,
        model_state_sha256=base['model_state_sha256'], weights_sha256=base['weights_sha256'], cases=CASES, methods=METHODS,
        repeats=args.repeats, warmup=args.warmup, trace_repeats=args.trace_repeats, threads=args.threads, interop_threads=1,
        seed=args.seed, tolerance=TOLERANCE, tf32=False, deterministic_algorithms=True,
        timing='No profiler/events/annotations in primary. All methods use the same persistent CUDA input buffer and full host return. Resident: A reset by a shared untimed D2D precondition for both eager/graph, then forward/return only. Device-update: timed D2D copy + forward/return. Host-update: timed pageable CPU->CUDA copy + forward/return. No Zarr read/pack/write, setup, validation or hashes in timing.',
        variants='A = first B native frames; B = next B disjoint frames in the pinned archive; no synthetic perturbation. Resident uses A, update methods alternate A/B by repeat for matched comparisons. Both cases/all methods shuffled within each repeat.',
        graph='One graph per shape, separate default private pools, fixed weights/input/output addresses. Side-stream eager warmup, capture/instantiate and first replay recorded separately. Input contents updated via copy_, never pointer replacement. Borrowed GPU output overwritten at next replay; host results copied before next call.',
        memory='Setup snapshots bracket graph creation after eager warmup. Net live allocated/reserved increments and setup peaks are PyTorch allocator observations, not total GPU memory. Earlier-shape graphs and all source buffers coexist. Eager runtime probes also run while graph pools remain resident; not standalone eager memory. Driver/cuFFT/graph executable memory may be absent.',
        limitations='Initialized warm process; not cold startup or full application speed. Setup break-even is an estimate from measured setup and request means. Private pools increase persistence. No framework/driver/power/precision/compiler change. Trace is a separate diagnostic run and may perturb timing.',
        case_metadata={})
    fixtures, outputs, samples, update_checks, memories, trace_checks = {}, {}, [], [], [], []
    with torch.inference_mode():
        for case in CASES:
            name = case['case']
            xs, identities = zip(*(load_variant(args.data, case, v) for v in [0, 1]))
            truths = [reference_model(torch.from_numpy(x).double()).numpy().copy() for x in xs]
            if np.array_equal(xs[0], xs[1]) or np.array_equal(truths[0], truths[1]):
                raise ValueError('Distinct genuine input/output variants required')
            baseline_artifact = args.baseline/(name+'.npz')
            if sha256(baseline_artifact) != base['case_metadata'][name]['artifact_sha256']:
                raise ValueError('Baseline case archive mismatch')
            with np.load(baseline_artifact, allow_pickle=False) as saved:
                np.testing.assert_array_equal(xs[0], saved['input_fp32'])
                np.testing.assert_allclose(truths[0], saved['reference_fp64'], rtol=1e-10, atol=1e-11)
            hosts = [torch.from_numpy(x) for x in xs]
            devices = [t.cuda() for t in hosts]
            for device, truth in zip(devices, truths):
                for _ in range(args.warmup):
                    temporary = gpu_model(device).cpu().numpy()
                    error_summary(temporary, truth)
            del temporary
            torch.cuda.synchronize()
            before_a, before_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            try:
                runner = CudaGraphRunner(gpu_model, devices[0], args.warmup)
            except Exception as exc:
                (args.output/'capture_failure.json').write_text(json.dumps(dict(case=name, error=repr(exc), no_fallback=True)), encoding='utf-8')
                raise
            after_a, after_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            setup_memory = dict(before_allocated_bytes=before_a, after_allocated_bytes=after_a, extra_allocated_bytes=after_a-before_a,
                before_reserved_bytes=before_r, after_reserved_bytes=after_r, extra_reserved_bytes=after_r-before_r,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            reference_path = args.output/(name+'_reference.npz')
            np.savez_compressed(reference_path, input_A=xs[0], input_B=xs[1], reference_A=truths[0], reference_B=truths[1])
            env['case_metadata'][name] = dict(identities=list(identities), input_hashes=[array_mapping_hash({'input': x}) for x in xs],
                input_bytes=xs[0].nbytes, reference_path=reference_path.name, reference_sha256=sha256(reference_path),
                baseline_artifact_sha256=base['case_metadata'][name]['artifact_sha256'], setup_seconds=runner.setup_seconds,
                setup_memory=setup_memory, reference_max_variant_difference=float(np.max(np.abs(truths[0]-truths[1]))), outputs={})
            fixtures[name] = dict(hosts=hosts, devices=devices, truths=truths, runner=runner)
            input_pointer, output_pointer = runner.input.data_ptr(), runner.output.data_ptr()
            previous_hash = None
            for step, variant in enumerate([0, 1, 0]):
                runner.copy_input(devices[variant])
                actual = runner.replay().cpu().numpy()
                copied = runner.input.cpu().numpy()
                np.testing.assert_array_equal(copied, xs[variant])
                eager = gpu_model(runner.input).cpu().numpy()
                np.testing.assert_allclose(actual, eager, rtol=1e-6, atol=1e-7)
                digest = array_mapping_hash({'output': actual})
                if previous_hash == digest:
                    raise ValueError('Updated graph output did not change')
                previous_hash = digest
                row = dict(case=name, step=step, variant=variant, input_sha256=array_mapping_hash({'input': copied}), output_sha256=digest,
                    input_pointer_stable=int(runner.input.data_ptr() == input_pointer), output_pointer_stable=int(runner.output.data_ptr() == output_pointer),
                    **error_summary(actual, truths[variant]))
                if row['input_pointer_stable'] != 1 or row['output_pointer_stable'] != 1:
                    raise ValueError('Captured addresses changed')
                update_checks.append(row)
                outputs[(name, variant, 'graph')] = actual
                outputs[(name, variant, 'eager')] = eager
        def invoke(job, annotate=False):
            name, method, variant = job['case'], job['method'], job['variant']
            backend, mode = method.split('_', 1)
            fixture = fixtures[name]
            runner = fixture['runner']
            scope = lambda label: torch.profiler.record_function(label) if annotate else nullcontext()
            if mode != 'resident':
                with scope('graph_input_update'):
                    runner.copy_input(fixture['hosts'][variant] if mode == 'host_update' else fixture['devices'][variant])
            with scope('graph_dispatch'):
                result = runner.replay() if backend == 'graph' else gpu_model(runner.input)
            with scope('graph_return'):
                return result.cpu().numpy()
        def precondition(job):
            if job['method'].endswith('_resident'):
                fixtures[job['case']]['runner'].copy_input(fixtures[job['case']]['devices'][0])
            torch.cuda.synchronize()
        def check(job, output):
            name, variant = job['case'], job['variant']
            backend = job['method'].split('_', 1)[0]
            errors = error_summary(output, fixtures[name]['truths'][variant])
            outputs[(name, variant, backend)] = output
            return dict(**job, output_sha256=array_mapping_hash({'output': output}), **errors)
        for job in request_order(args.warmup, args.seed-1):
            precondition(job)
            check(job, invoke(job))
        for job in request_order(args.repeats, args.seed):
            precondition(job)
            start = time.perf_counter_ns()
            output = invoke(job)
            seconds = (time.perf_counter_ns()-start)/1e9
            samples.append(dict(**check(job, output), seconds=seconds))
        print(f'PASS: {len(samples)} eager/graph primary outputs and genuine A/B updates', flush=True)
        for case in CASES:
            for method in METHODS:
                job = dict(case=case['case'], method=method, variant=0, repeat=0, order=0)
                precondition(job)
                baseline_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                output = invoke(job)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                memories.append(dict(case=case['case'], method=method, baseline_allocated_bytes=baseline_bytes, peak_allocated_bytes=peak,
                                     peak_extra_allocated_bytes=peak-baseline_bytes, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                check(job, output)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            for job in request_order(args.trace_repeats, args.seed+1000):
                precondition(job)
                with torch.profiler.record_function(f'graph_call/{job["case"]}/{job["method"]}/{job["repeat"]}/{job["variant"]}'):
                    output = invoke(job, True)
                trace_checks.append(check(job, output))
                profiler.step()
        path = args.output/'trace.json'
        profiler.export_chrome_trace(str(path))
        raw = path.read_bytes()
        trace_rows = summarize_trace(json.loads(raw))
        trace_index = {(r['case'], r['method'], r['repeat'], r['variant']): r for r in trace_checks}
        if len(trace_rows) != len(trace_checks):
            raise ValueError('Trace call count mismatch')
        for row in trace_rows:
            key = (row['case'], row['method'], row['repeat'], row['variant'])
            row.update({k: v for k, v in trace_index[key].items() if k not in row})
            is_graph = row['method'].startswith('graph_')
            if row['dtoh_copies'] != 1 or row['dtoh_bytes'] != env['case_metadata'][row['case']]['input_bytes'] or row['kernel_events'] < 1 or (is_graph and row['graph_launches'] < 1) or (not is_graph and row['graph_launches'] != 0):
                raise ValueError('Trace lacks full output, kernels or expected graph launch')
        with gzip.open(args.output/'trace.json.gz', 'wb') as stream:
            stream.write(raw)
        path.unlink()
        env['trace_sha256'] = sha256(args.output/'trace.json.gz')
        for (name, variant, backend), output in outputs.items():
            path = args.output/'outputs'/f'{name}_{variant}_{backend}.npy'
            with path.open('xb') as stream:
                np.save(stream, output, allow_pickle=False)
            env['case_metadata'][name]['outputs'][f'{variant}_{backend}'] = dict(path=path.relative_to(args.output).as_posix(), sha256=sha256(path))
        summaries = [dict(case=c['case'], method=m, **stats([r for r in samples if r['case'] == c['case'] and r['method'] == m])) for c in CASES for m in METHODS]
        for filename, records in [('timing_samples.csv', samples), ('timing_summary.csv', summaries), ('update_checks.csv', update_checks), ('memory_samples.csv', memories), ('trace_calls.csv', trace_rows)]:
            write_csv(args.output/filename, records)
        env['hardware_after'] = hardware_snapshot(torch)
        (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
        print(f'PASS: {len(samples)} primary, {len(trace_rows)} profiled calls; fixed-address updates, FP64 outputs and real graph launches verified', flush=True)


if __name__ == '__main__':
    main()
