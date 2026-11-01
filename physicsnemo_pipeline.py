"""Local Zarr -> untrained FNO -> uncompressed NPZ, identical CPU/CUDA contracts."""
import argparse
from contextlib import contextmanager
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

from physicsnemo_benchmark import (MODEL_CONFIG, DEFAULT_CASES, TOLERANCE,
    array_mapping_hash, double_reference, error_summary, load_input, make_model,
    parse_cases, prepare_input, read_input, stats)
from weather_verify import sha256, write_csv


METHODS = ['cpu', 'cuda']
STAGES = ['read_decode', 'prepare', 'upload', 'forward', 'return', 'write_close']
OUTPUT_KEYS = {'model_output', 'initialization', 'lead_hours', 'member', 'valid_time',
               'latitude', 'longitude', 'metadata'}


def output_payload(output, identity, coordinates, case, source_hash, state_hash):
    frames = identity['frames']
    metadata = dict(schema='physicsnemo-pipeline-v1', variable='model_output', units='1',
                    role='Untrained fixed-weight FNO output; no forecast-quality interpretation',
                    dims=['frame', 'channel', 'latitude', 'longitude'], case=case,
                    source_sha256=source_hash, model_state_sha256=state_hash,
                    source_units=identity['source_units'], input_transform=identity['transform'],
                    coordinate_units=dict(latitude='degrees_north', longitude='degrees_east'))
    return dict(model_output=output,
                initialization=np.array([f['initialization'] for f in frames], dtype='datetime64[ns]'),
                lead_hours=np.array([f['lead_hours'] for f in frames], dtype=np.int64),
                member=np.array([f['member'] for f in frames], dtype=np.int64),
                valid_time=np.array([f['valid_time'] for f in frames], dtype='datetime64[ns]'),
                latitude=coordinates['latitude'], longitude=coordinates['longitude'],
                metadata=np.array(json.dumps(metadata, sort_keys=True)))


def write_output(path, payload):
    # Uncompressed ZIP avoids output-value-dependent compression differences.
    # Close is included; durable media flush/fsync is deliberately not requested.
    with path.open('xb') as stream:
        np.savez(stream, **payload)


def verify_output(path, expected_payload, reference):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != OUTPUT_KEYS:
            raise ValueError('Missing or unexpected output arrays')
        for key, expected in expected_payload.items():
            if saved[key].dtype != expected.dtype:
                raise ValueError('Output metadata/array dtype changed')
            np.testing.assert_array_equal(saved[key], expected)
        return error_summary(saved['model_output'], reference)


class StageClock:
    def __init__(self, sync):
        self.sync = sync
        self.seconds = dict.fromkeys(STAGES, 0.)

    @contextmanager
    def stage(self, name):
        start = time.perf_counter_ns()
        yield
        if name in ['upload', 'forward', 'return']:
            self.sync()
        self.seconds[name] = (time.perf_counter_ns()-start)/1e9


@contextmanager
def stage(clock, name):
    if clock is None:
        yield
    else:
        with clock.stage(name):
            yield


def execute(data, case, model, backend, path, source_hash, state_hash, clock=None):
    import torch
    start = time.perf_counter_ns()
    with stage(clock, 'read_decode'):
        raw, identity, coordinates = read_input(data, case)
    with stage(clock, 'prepare'):
        packed = prepare_input(raw)
        tensor = torch.from_numpy(packed)
        # Metadata/coordinate packing is part of the workflow, not validation.
        payload = output_payload(None, identity, coordinates, case, source_hash, state_hash)
    if backend == 'cuda':
        with stage(clock, 'upload'):
            tensor = tensor.to('cuda')
    elif backend != 'cpu':
        raise ValueError('Unknown backend')
    with stage(clock, 'forward'):
        result = model(tensor)
    with stage(clock, 'return'):
        host = result.cpu().numpy() if backend == 'cuda' else result.numpy()
    with stage(clock, 'write_close'):
        payload['model_output'] = host
        write_output(path, payload)
    seconds = (time.perf_counter_ns()-start)/1e9
    return seconds, packed, identity, payload


def input_budget(data, case):
    """Nominal local v2 chunk coverage, not measured physical disk traffic."""
    meta = json.loads((data/'forecast'/'.zarray').read_text())
    ni, nl, nm, height, width = meta['shape']
    batch, side = case['batch'], case['side']
    nr = (batch+nl*nm-1)//(nl*nm)
    leads = min(nl, (batch+nm-1)//nm) if nr == 1 else nl
    counts = [nr, leads, min(nm, batch), side, side]
    starts = [0, 0, 0, height-side, 0]
    touched = int(np.prod([(s+n-1)//c-s//c+1 for s, n, c in zip(starts, counts, meta['chunks'])]))
    return dict(chunk_shape=meta['chunks'], chunks_touched=touched,
                nominal_chunk_uncompressed_bytes=touched*int(np.prod(meta['chunks']))*np.dtype(meta['dtype']).itemsize,
                materialized_frames=int(np.prod(counts[:3])),
                selected_logical_bytes=int(np.prod(counts))*np.dtype(meta['dtype']).itemsize,
                packed_input_bytes=batch*side*side*4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--profile-repeats', type=int, default=8)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--order-seed', type=int, default=20261102)
    args = parser.parse_args()
    cases = parse_cases(args.cases)
    if args.output.exists() or min(args.repeats, args.profile_repeats) < 2 or min(args.warmup, args.threads) < 1:
        parser.error('New output path, >=2 repeat/profile samples and positive warmup/threads required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    setup_start = time.perf_counter()
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled isolated PhysicsNeMo environment required; no CPU fallback')
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
    if source_hash != baseline['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or baseline['model_config'] != MODEL_CONFIG:
        raise ValueError('Baseline architecture or input provenance differs')
    if any(c not in baseline['cases'] for c in cases):
        raise ValueError('Use cases with archived baseline FP64 references')
    weights_path = args.baseline/'weights.npz'
    if sha256(weights_path) != baseline['weights_sha256']:
        raise ValueError('Baseline weight archive hash mismatch')
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    state_hash = array_mapping_hash(weights)
    if state_hash != baseline['model_state_sha256']:
        raise ValueError('Baseline tensor state hash mismatch')
    cpu_model = make_model()
    cpu_model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
    gpu_model = copy.deepcopy(cpu_model).to('cuda')
    reference_model = double_reference(cpu_model)
    torch.cuda.synchronize()
    setup_sec = time.perf_counter()-setup_start
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    shutil.copyfile(weights_path, args.output/'weights.npz')
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), scope='Local pipeline performance and numerical correctness; untrained FNO',
               python=platform.python_version(), versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr', 'scipy']},
               cuda_runtime=torch.version.cuda, hardware=hardware_snapshot(torch),
               source_sha256=source_hash, source_manifest=manifest, model_config=MODEL_CONFIG,
               model_state_sha256=state_hash, weights_sha256=baseline['weights_sha256'], baseline=args.baseline.name,
               cases=cases, methods=METHODS, stages=STAGES, repeats=args.repeats, warmup=args.warmup,
               profile_repeats=args.profile_repeats, threads=args.threads, interop_threads=1, order_seed=args.order_seed,
               tolerance=TOLERANCE, tf32=False, deterministic_algorithms=True, setup_seconds=setup_sec,
               timing='Primary wall time from local Zarr open/read/decode/coordinate validation through scaling/packing, host tensor creation, upload (CUDA), forward, full host return, uncompressed NPZ write and file close. Models/weights remain resident; no input cache is reused. Reference/hash/readback validation and temporary-file cleanup are outside timing. Profiles are separate synchronized executions; no overlap/prefetch.',
               cache='Repeated local reads after setup/reference validation; OS/library caches uncontrolled, no cache clearing. First observations are initialized-process observations, not cold disk/process measurements. Source hash verification and reference setup may already warm caches.',
               write='np.savez, uncompressed ZIP, identical schema and fixed metadata for CPU/CUDA. File creation and close included; no fsync/durable-media guarantee. Every output is read back and checked outside timing. Only last primary output per case/backend retained; other trial outputs removed after validation outside timing.',
               profile='Separate stage-synchronized runs; mean stage times sum with unattributed mean to measured profile mean, never to primary P50.',
               case_metadata={})
    samples, summaries, first_rows, profiles, memory = [], [], [], [], []
    models = dict(cpu=cpu_model, cuda=gpu_model)
    with torch.inference_mode():
        for case in cases:
            name = case['case']
            x, identity = load_input(args.data, case)
            artifact = args.baseline/(name+'.npz')
            artifact_hash = sha256(artifact)
            if artifact_hash != baseline['case_metadata'][name]['artifact_sha256']:
                raise ValueError('Baseline case archive hash mismatch')
            with np.load(artifact, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['input_fp32'], x)
                truth = saved['reference_fp64'].copy()
            fresh = reference_model(torch.from_numpy(x).double()).numpy()
            np.testing.assert_allclose(fresh, truth, rtol=1e-10, atol=1e-11)
            meta = dict(identity=identity, input_sha256=array_mapping_hash({'input': x}),
                        baseline_artifact_sha256=artifact_hash, budget=input_budget(args.data, case), outputs={})
            env['case_metadata'][name] = meta
            def trial(backend, label, repeat, order, retain=False, clock=None):
                path = args.output/'outputs'/(name+'_'+backend+'.npz') if retain else args.output/'trial.npz'
                torch.cuda.synchronize()
                seconds, packed, observed_identity, payload = execute(args.data, case, models[backend], backend, path,
                                                                      source_hash, state_hash, clock)
                np.testing.assert_array_equal(packed, x)
                if observed_identity != identity:
                    raise ValueError('Frame identity changed')
                error = verify_output(path, payload, truth)
                file_hash, file_bytes = sha256(path), path.stat().st_size
                logical_hash = array_mapping_hash(payload)
                row = dict(case=name, method=backend, repeat=repeat, order=order, seconds=seconds,
                           file_bytes=file_bytes, payload_sha256=logical_hash, file_sha256=file_hash, **error)
                if retain:
                    meta['outputs'][backend] = dict(path=path.relative_to(args.output).as_posix(), sha256=file_hash,
                                                    payload_sha256=logical_hash, bytes=file_bytes)
                else:
                    path.unlink()
                return row
            for backend in METHODS:
                first_rows.append(trial(backend, 'first', 0, METHODS.index(backend)))
            for _ in range(args.warmup):
                for backend in METHODS:
                    trial(backend, 'warmup', 0, METHODS.index(backend))
            # One independent full-workflow allocator observation, not timed latency data.
            torch.cuda.synchronize()
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            trial('cuda', 'memory', 0, 0)
            peak = torch.cuda.max_memory_allocated()
            memory.append(dict(case=name, baseline_allocated_bytes=baseline_bytes, peak_allocated_bytes=peak,
                               peak_extra_allocated_bytes=peak-baseline_bytes, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            rng = np.random.default_rng(args.order_seed)
            for repeat in range(args.repeats):
                for order, backend in enumerate(rng.permutation(METHODS)):
                    samples.append(trial(str(backend), 'primary', repeat, order, retain=repeat == args.repeats-1))
            for backend in METHODS:
                summary = dict(case=name, method=backend, **stats([r for r in samples if r['case'] == name and r['method'] == backend]))
                summary['frames_per_second'] = case['batch']/summary['p50_sec']
                summaries.append(summary)
            rng = np.random.default_rng(args.order_seed+1)
            for repeat in range(args.profile_repeats):
                for order, backend in enumerate(rng.permutation(METHODS)):
                    clock = StageClock(torch.cuda.synchronize if backend == 'cuda' else lambda: None)
                    row = trial(str(backend), 'profile', repeat, order, clock=clock)
                    row.update({k+'_sec': v for k, v in clock.seconds.items()})
                    row['unattributed_sec'] = row['seconds']-sum(clock.seconds.values())
                    profiles.append(row)
            print(json.dumps(dict(case=name, p50_ms={r['method']: round(r['p50_sec']*1000, 3) for r in summaries if r['case'] == name})), flush=True)
    env['hardware_after'] = hardware_snapshot(torch)
    for filename, rows in [('timing_samples.csv', samples), ('timing_summary.csv', summaries),
                           ('first_observed.csv', first_rows), ('stage_samples.csv', profiles), ('memory.csv', memory)]:
        write_csv(args.output/filename, rows)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: {len(samples)} primary and {len(profiles)} profiled files read back and checked; local untrained FNO pipeline', flush=True)


if __name__ == '__main__':
    main()
