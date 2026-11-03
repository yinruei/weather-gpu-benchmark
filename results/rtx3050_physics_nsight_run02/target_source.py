"""NVTX-only full-file diagnostic target for Nsight Systems; not primary timing."""
import argparse
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import platform

import numpy as np
import xarray as xr

from physicsnemo_benchmark import array_mapping_hash, make_model
from physicsnemo_graph import CASES, CudaGraphRunner
from physicsnemo_graph_pipeline import FnoFilePipeline, read_variant
from physicsnemo_pipeline import verify_output
from weather_verify import sha256, write_csv

READERS = ['open_once', 'exact_once', 'gather_once']
ENGINES = ['cpu', 'eager', 'graph']
CASE = CASES[-1]


def jobs(repeats, seed):
    choices = [engine+'_'+reader for reader in READERS for engine in ENGINES]
    rng = np.random.default_rng(seed)
    return [dict(case=CASE['case'], method=choices[int(i)], repeat=r, order=o, variant=r % 2)
        for r in range(repeats) for o, i in enumerate(rng.permutation(len(choices)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--capture', action='store_true', help='Use cudaProfilerStart/Stop to bound Nsight capture')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('New output and at least two repetitions required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA required; no fallback')
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ref = json.loads((args.reference_run/'environment.json').read_text())
    source = sha256(args.data)
    if source != ref['source_sha256'] or sha256(args.reference_run/'weights.npz') != ref['weights_sha256']:
        raise ValueError('Source/state provenance mismatch')
    with np.load(args.reference_run/'weights.npz', allow_pickle=False) as saved:
        state = {k: saved[k].copy() for k in saved.files}
    if array_mapping_hash(state) != ref['model_state_sha256']:
        raise ValueError('Tensor state mismatch')
    cpu = make_model()
    cpu.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    gpu = copy.deepcopy(cpu).cuda()
    meta = ref['case_metadata'][CASE['case']]
    reference_path = args.reference_run/meta['reference_path']
    if sha256(reference_path) != meta['reference_sha256']:
        raise ValueError('Pinned reference changed')
    with np.load(reference_path, allow_pickle=False) as saved:
        inputs = [saved['input_'+v].copy() for v in ['A', 'B']]
        truths = [saved['reference_'+v].copy() for v in ['A', 'B']]
    args.output.mkdir(parents=True)
    (args.output/'outputs').mkdir()
    identities, rows, outputs = [], [], {}
    for v in [0, 1]:
        raw, identity, _ = read_variant(args.data, CASE, v)
        from physicsnemo_benchmark import prepare_input
        np.testing.assert_array_equal(prepare_input(raw), inputs[v])
        identities.append(identity)
    with xr.open_zarr(args.data, chunks=None, consolidated=True) as dataset, torch.inference_mode():
        example = torch.from_numpy(inputs[0]).cuda()
        for _ in range(3):
            gpu(example).cpu()
        runner = CudaGraphRunner(gpu, example, 3)
        workers = {e+'_'+r: FnoFilePipeline(args.data, CASE, cpu if e == 'cpu' else gpu,
            engine=e, reader=r, runner=runner if e != 'cpu' else None, dataset=dataset) for r in READERS for e in ENGINES}
        def execute(job, annotated, retain):
            method, variant = job['method'], job['variant']
            path = args.output/'outputs'/f'{variant}_{method}.npz' if retain else args.output/'trial.npz'
            torch.cuda.synchronize()
            scope = torch.cuda.nvtx.range(f'request/{job["case"]}/{method}/{job["repeat"]}/{variant}') if annotated else nullcontext()
            with scope:
                seconds, packed, identity, payload = workers[method].execute(variant, path, source, ref['model_state_sha256'], annotate='nvtx' if annotated else False)
            np.testing.assert_array_equal(packed, inputs[variant])
            if identity != identities[variant]:
                raise ValueError('Input identity changed')
            errors = verify_output(path, payload, truths[variant])
            info = dict(path=path.relative_to(args.output).as_posix(), sha256=sha256(path), bytes=path.stat().st_size,
                payload_sha256=array_mapping_hash(payload))
            if retain:
                outputs[f'{variant}_{method}'] = info
            else:
                path.unlink()
            return dict(**job, seconds=seconds, file_sha256=info['sha256'], file_bytes=info['bytes'], payload_sha256=info['payload_sha256'], **errors)
        for j in jobs(2, args.seed-1):
            execute(j, False, False)
        torch.cuda.synchronize()
        if args.capture:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            for j in jobs(args.repeats, args.seed):
                rows.append(execute(j, True, j['repeat'] >= args.repeats-2))
        finally:
            torch.cuda.synchronize()
            if args.capture:
                torch.cuda.cudart().cudaProfilerStop()
        for worker in workers.values():
            worker.close()
    write_csv(args.output/'calls.csv', rows)
    env = dict(scope='Separate Nsight Systems diagnostic; NVTX annotations, not primary performance measurements',
        python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        source_sha256=source, model_state_sha256=ref['model_state_sha256'], weights_sha256=ref['weights_sha256'],
        reference_run=args.reference_run.name, reference_path=reference_path.name, reference_sha256=sha256(reference_path),
        case=CASE, engines=ENGINES, readers=READERS, repeats=args.repeats, seed=args.seed, threads=6,
        capture=args.capture, tf32=False, deterministic_algorithms=True, identities=identities, outputs=outputs,
        graph_setup_seconds=runner.setup_seconds,
        limits='One B64 graph and shared source handle; resource/shape/cadence state differs from primary benchmark. NVTX ranges cover actual read through NPZ close; validation/hash/cleanup/sync outside request ranges. File readback must agree with pinned CPU FP64. No torch.profiler/CUPTI collector runs inside Nsight. NVTX CPU stages and GPU activity overlap; not occupancy or an additive decomposition of primary P50.')
    (args.output/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    print(f'PASS: {len(rows)} real full-file NVTX requests validated; all requested engines and genuine A/B inputs', flush=True)


if __name__ == '__main__':
    main()
