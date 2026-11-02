"""Paired reuse workflows and GPU cadence diagnostics on identical real inputs."""
import argparse
import copy
from datetime import datetime, timezone
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
    double_reference, error_summary, make_model, parse_cases, prepare_input,
    read_from_dataset, read_input, stats)
from physicsnemo_pipeline import output_payload, verify_output, write_output
from weather_verify import sha256, write_csv


WORKFLOWS = ['cpu_reopen', 'cuda_reopen', 'cpu_open_once', 'cuda_open_once',
             'cpu_host_cache', 'cuda_host_cache', 'cuda_device_cache']
CONDITIONS = ['continuous', 'idle_gap', 'read_gap', 'cpu_gap']


def job_order(cases, methods, repeats, seed):
    jobs = [(c['case'], m) for c in cases for m in methods]
    rng = np.random.default_rng(seed)
    return [(repeat, order, *jobs[int(i)]) for repeat in range(repeats)
            for order, i in enumerate(rng.permutation(len(jobs)))]


def run_workflow(data, fixture, case, model, method, path, source_hash, state_hash):
    import torch
    if method not in WORKFLOWS:
        raise ValueError('Unknown workflow')
    backend, mode = method.split('_', 1)
    start = time.perf_counter_ns()
    if mode in ['reopen', 'open_once']:
        raw, identity, coordinates = (read_input(data, case) if mode == 'reopen'
                                      else read_from_dataset(fixture['dataset'], case))
        packed = prepare_input(raw)
        tensor = torch.from_numpy(packed)
        payload = output_payload(None, identity, coordinates, case, source_hash, state_hash)
    else:
        packed, identity = fixture['x'], fixture['identity']
        tensor = fixture['cpu_input']
        payload = dict(fixture['payload'])
    if backend == 'cuda':
        tensor = fixture['cuda_input'] if mode == 'device_cache' else tensor.to('cuda')
    result = model(tensor)
    host = result.cpu().numpy() if backend == 'cuda' else result.numpy()
    payload['model_output'] = host
    write_output(path, payload)
    seconds = (time.perf_counter_ns()-start)/1e9
    return seconds, packed, identity, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--cases', nargs='+', default=['16x64', '64x64'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--block-repeats', type=int, default=8)
    parser.add_argument('--gap-ms', type=float, default=25.)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    cases = parse_cases(args.cases)
    if args.output.exists() or min(args.repeats, args.blocks, args.block_repeats) < 2 or min(args.warmup, args.threads, args.seed) < 1 or not np.isfinite(args.gap_ms) or args.gap_ms <= 0:
        parser.error('New output path, >=2 repetitions/blocks/inner repeats, positive warmup/threads/gap required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA required; no fallback')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((args.baseline/'environment.json').read_text())
    source_hash = sha256(args.data)
    manifest = json.loads((args.data.parent/'manifest.json').read_text())
    if source_hash != baseline['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or baseline['model_config'] != MODEL_CONFIG or any(c not in baseline['cases'] for c in cases):
        raise ValueError('Baseline/input/config mismatch')
    weights_path = args.baseline/'weights.npz'
    if sha256(weights_path) != baseline['weights_sha256']:
        raise ValueError('Weight file mismatch')
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    state_hash = array_mapping_hash(weights)
    if state_hash != baseline['model_state_sha256']:
        raise ValueError('Model tensor state mismatch')
    cpu_model = make_model()
    cpu_model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    gpu_model = copy.deepcopy(cpu_model).cuda()
    reference_model = double_reference(cpu_model)
    torch.cuda.synchronize()
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    shutil.copyfile(weights_path, args.output/'weights.npz')
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), scope='Fixed untrained FNO performance/numerical consistency; no forecast assessment',
               python=platform.python_version(), versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr']},
               hardware=hardware_snapshot(torch), cuda_runtime=torch.version.cuda,
               source_sha256=source_hash, source_manifest=manifest, model_config=MODEL_CONFIG,
               weights_sha256=baseline['weights_sha256'], model_state_sha256=state_hash,
               cases=cases, workflows=WORKFLOWS, conditions=CONDITIONS, repeats=args.repeats,
               warmup=args.warmup, blocks=args.blocks, block_repeats=args.block_repeats,
               gap_ms=args.gap_ms, threads=args.threads, interop_threads=1, seed=args.seed,
               tolerance=TOLERANCE, tf32=False, deterministic_algorithms=True,
               workflow_boundary='All workflows include FP32 forward, full host output, identical uncompressed NPZ write/close; no fsync. Reopen includes read/pack; open_once reuses only dataset handle while selecting/loading/checking/packing each call; host_cache reuses packed CPU tensor and metadata; device_cache additionally reuses CUDA input. Cache setup is recorded outside request time; only valid for repeated identical input. Models stay resident.',
               ordering='Primary: both cases/all workflows shuffled within every repeat. Cadence: both cases/all conditions shuffled within each block, then consecutive inner calls. Warmup per condition precedes measured inner calls; outputs validated after the whole cadence block, outside GPU timing. No files in cadence experiment.',
               cadence_boundary='Full-output forward wall time includes two CUDA event records, dispatch, forward, event-end record, blocking host copy. Event elapsed time brackets forward on stream and may include dispatch/scheduling gaps; not kernel-active time or occupancy. Precondition times (read/pack+input check, CPU forward, sleep) are recorded separately. Continuous has no intentional precondition but still includes loop/event/host-return overhead.',
               limitations='OS/library cache uncontrolled and warmed by setup/validation; no cold-disk claim. All mode/case caches coexist during measurement. No clock/power/driver setting changed; GPU telemetry is a coarse snapshot, not per-kernel attribution. Case mixing and cadence differ from earlier experiments; use within-session comparisons. Repeated identical weights/inputs only, untrained model.',
               case_metadata={})
    fixtures, workflow_samples, cadence = {}, [], []
    with torch.inference_mode():
        for case in cases:
            name = case['case']
            start = time.perf_counter()
            dataset = xr.open_zarr(args.data, chunks=None, consolidated=True)
            open_sec = time.perf_counter()-start
            start = time.perf_counter()
            raw, identity, coordinates = read_input(args.data, case)
            x = prepare_input(raw)
            cpu_input = torch.from_numpy(x)
            payload = output_payload(None, identity, coordinates, case, source_hash, state_hash)
            host_sec = time.perf_counter()-start
            start = time.perf_counter()
            cuda_input = cpu_input.cuda()
            torch.cuda.synchronize()
            upload_sec = time.perf_counter()-start
            artifact = args.baseline/(name+'.npz')
            artifact_hash = sha256(artifact)
            if artifact_hash != baseline['case_metadata'][name]['artifact_sha256']:
                raise ValueError('Baseline case hash mismatch')
            with np.load(artifact, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['input_fp32'], x)
                truth = saved['reference_fp64'].copy()
            fresh = reference_model(cpu_input.double()).numpy()
            np.testing.assert_allclose(fresh, truth, rtol=1e-10, atol=1e-11)
            fixtures[name] = dict(dataset=dataset, identity=identity, coordinates=coordinates,
                                  x=x, cpu_input=cpu_input, cuda_input=cuda_input,
                                  payload=payload, truth=truth)
            env['case_metadata'][name] = dict(identity=identity, input_sha256=array_mapping_hash({'input': x}),
                baseline_artifact_sha256=artifact_hash, setup_seconds=dict(reopen=0., open_once=open_sec, host_cache=host_sec, device_cache=host_sec+upload_sec),
                cpu_input_bytes=x.nbytes, cuda_input_bytes=x.nbytes, outputs={}, cadence_output={})
        case_lookup = {c['case']: c for c in cases}
        def trial(name, method, repeat, order, retain=False):
            backend = method.split('_', 1)[0]
            path = args.output/'outputs'/(name+'_'+method+'.npz') if retain else args.output/'trial.npz'
            torch.cuda.synchronize()
            seconds, packed, identity, payload = run_workflow(args.data, fixtures[name], case_lookup[name],
                cpu_model if backend == 'cpu' else gpu_model, method, path, source_hash, state_hash)
            np.testing.assert_array_equal(packed, fixtures[name]['x'])
            if identity != fixtures[name]['identity']:
                raise ValueError('Frame identity changed')
            errors = verify_output(path, payload, fixtures[name]['truth'])
            file_hash, logical_hash, size = sha256(path), array_mapping_hash(payload), path.stat().st_size
            row = dict(case=name, method=method, repeat=repeat, order=order, seconds=seconds,
                       file_bytes=size, file_sha256=file_hash, payload_sha256=logical_hash, **errors)
            if retain:
                env['case_metadata'][name]['outputs'][method] = dict(path=path.relative_to(args.output).as_posix(), sha256=file_hash, bytes=size, payload_sha256=logical_hash)
            else:
                path.unlink()
            return row
        for _, _, name, method in job_order(cases, WORKFLOWS, args.warmup, args.seed-1):
            trial(name, method, 0, 0)
        for repeat, order, name, method in job_order(cases, WORKFLOWS, args.repeats, args.seed):
            workflow_samples.append(trial(name, method, repeat, order, retain=repeat == args.repeats-1))
        print('PASS: paired workflow trials completed', flush=True)
        def gpu_call(fixture, condition, events):
            pre_output = None
            start = time.perf_counter_ns()
            if condition == 'idle_gap':
                time.sleep(args.gap_ms/1000)
            elif condition == 'read_gap':
                raw, identity, _ = read_input(args.data, case_lookup[fixture['name']])
                packed = prepare_input(raw)
                np.testing.assert_array_equal(packed, fixture['x'])
                if identity != fixture['identity']:
                    raise ValueError('Precondition frame identity changed')
            elif condition == 'cpu_gap':
                pre_output = cpu_model(fixture['cpu_input']).numpy()
            pre_seconds = 0. if condition == 'continuous' else (time.perf_counter_ns()-start)/1e9
            start = time.perf_counter_ns()
            events[0].record()
            result = gpu_model(fixture['cuda_input'])
            events[1].record()
            host = result.cpu().numpy()
            seconds = (time.perf_counter_ns()-start)/1e9
            event_seconds = events[0].elapsed_time(events[1])/1000
            return dict(seconds=seconds, event_seconds=event_seconds, pre_seconds=pre_seconds), host, pre_output
        for fixture_name, fixture in fixtures.items():
            fixture['name'] = fixture_name
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for block, order, name, condition in job_order(cases, CONDITIONS, args.blocks, args.seed+1):
            fixture = fixtures[name]
            for _ in range(args.warmup):
                _, output, pre_output = gpu_call(fixture, condition, events)
                error_summary(output, fixture['truth'])
                if pre_output is not None:
                    error_summary(pre_output, fixture['truth'])
            pending = []
            for position in range(args.block_repeats):
                timed, output, pre_output = gpu_call(fixture, condition, events)
                pending.append((timed, output, pre_output, position))
            for timed, output, pre_output, position in pending:
                errors = error_summary(output, fixture['truth'])
                if pre_output is not None:
                    error_summary(pre_output, fixture['truth'])
                cadence.append(dict(case=name, condition=condition, block=block, order=order,
                                    position=position, **timed, output_sha256=array_mapping_hash({'output': output}), **errors))
            if block == args.blocks-1:
                # Same deterministic output for every condition, saved once per case.
                path = args.output/'outputs'/(name+'_cadence.npy')
                if not path.exists():
                    with path.open('xb') as stream:
                        np.save(stream, output, allow_pickle=False)
                    env['case_metadata'][name]['cadence_output'] = dict(path=path.relative_to(args.output).as_posix(), sha256=sha256(path))
        summaries = []
        for case in cases:
            name = case['case']
            for method in WORKFLOWS:
                summary = dict(case=name, method=method, **stats([r for r in workflow_samples if r['case'] == name and r['method'] == method]))
                summary['frames_per_second'] = case['batch']/summary['p50_sec']
                summaries.append(summary)
            fixtures[name]['dataset'].close()
        for filename, records in [('workflow_samples.csv', workflow_samples), ('workflow_summary.csv', summaries), ('cadence_samples.csv', cadence)]:
            write_csv(args.output/filename, records)
        env['hardware_after'] = hardware_snapshot(torch)
        (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
        print(f'PASS: {len(workflow_samples)} paired files and {len(cadence)} cadence outputs verified against FP64', flush=True)


if __name__ == '__main__':
    main()
