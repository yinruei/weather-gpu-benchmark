"""Small, untrained PhysicsNeMo FNO forward performance on real Zarr inputs.

This is an architecture/inference workload, not a trained weather forecast.
Use the isolated phase-5 environment; no model checkpoint is downloaded.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from weather_verify import sha256, write_csv


MODEL_CONFIG = dict(in_channels=1, out_channels=1, dimension=2, latent_channels=16,
                    num_fno_layers=2, num_fno_modes=8, padding=0, coord_features=True,
                    decoder_layers=2, decoder_layer_size=16,
                    activation_fn='gelu', decoder_activation_fn='silu')
DEFAULT_CASES = ['1x32', '1x64', '4x64', '16x64', '64x64']
METHODS = ['cpu_forward', 'cuda_forward_return', 'cuda_upload_forward_return', 'cuda_device_only']
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def parse_cases(specifications):
    result = []
    for specification in specifications:
        pieces = specification.split('x')
        if len(pieces) != 2 or not all(p.isdigit() for p in pieces):
            raise ValueError('Cases use batch x square-side, e.g. 4x64')
        batch, side = map(int, pieces)
        if batch < 1 or side < 2*MODEL_CONFIG['num_fno_modes']:
            raise ValueError('Positive batch and a side accommodating Fourier modes required')
        case = dict(case=f'b{batch}_s{side}', batch=batch, side=side)
        if case in result:
            raise ValueError('Duplicate workload case')
        result.append(case)
    if not result:
        raise ValueError('At least one workload required')
    return result


def array_mapping_hash(mapping):
    digest = hashlib.sha256(b'array-mapping-v1\0')
    for name in sorted(mapping):
        original = np.asarray(mapping[name])
        array = np.ascontiguousarray(original)
        if array.dtype.hasobject:
            raise ValueError('Object arrays are not valid numerical inputs')
        digest.update(name.encode()+b'\0'+array.dtype.str.encode()+b'\0')
        digest.update(json.dumps(list(original.shape)).encode()+b'\0'+array.tobytes())
    return digest.hexdigest()


def read_input(store, case):
    """Read the native bounding selection; defer numerical packing to prepare_input."""
    import xarray as xr
    from weather_batch import validate_batch
    batch, side = case['batch'], case['side']
    with xr.open_zarr(store, chunks=None, consolidated=True) as ds:
        validate_batch(ds)
        if side > min(ds.sizes['latitude'], ds.sizes['longitude']):
            raise ValueError('Requested native spatial subset is larger than the source; no tiling')
        ni, nl, nm = [ds.sizes[d] for d in ['init_time', 'lead_time', 'ensemble_member']]
        if batch > ni*nl*nm:
            raise ValueError('Insufficient distinct source frames; no synthetic repetition')
        run_count = (batch+nl*nm-1)//(nl*nm)
        lead_count = min(nl, (batch+nm-1)//nm) if run_count == 1 else nl
        member_count = min(nm, batch)
        selected = ds.forecast.isel(init_time=slice(run_count), lead_time=slice(lead_count),
                                    ensemble_member=slice(member_count), latitude=slice(-side, None),
                                    longitude=slice(side)).load()
        raw = selected.values.reshape(-1, side, side)[:batch, None]
        identities = [dict(initialization=str(initial), lead_hours=int(lead/np.timedelta64(1, 'h')),
                           member=int(member), valid_time=str(initial+lead))
                      for initial in selected.init_time.values for lead in selected.lead_time.values
                      for member in selected.ensemble_member.values][:batch]
        metadata = dict(frames=identities, latitude=[float(selected.latitude.min()), float(selected.latitude.max())],
                        longitude=[float(selected.longitude.min()), float(selected.longitude.max())],
                        source_units='degree_Celsius', transform='(Celsius - 20.0) / 20.0 in FP64, then FP32; fixed benchmark scaling, not trained-model normalization')
        coordinates = {name: selected[name].values.copy() for name in ['latitude', 'longitude']}
    return raw, metadata, coordinates


def prepare_input(raw):
    if not np.isfinite(raw).all():
        raise ValueError('FNO fixture requires finite input; no silent NaN fill')
    return np.ascontiguousarray(((raw.astype(np.float64)-20.)/20.).astype(np.float32))


def load_input(store, case):
    raw, metadata, _ = read_input(store, case)
    return prepare_input(raw), metadata


def error_summary(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Output shape/nonfinite values differ from the numerical reference')
    difference = actual.astype(np.float64)-reference
    scale = TOLERANCE['atol']+TOLERANCE['rtol']*np.abs(reference)
    normalized = float(np.max(np.abs(difference)/scale))
    if normalized > 1:
        raise ValueError(f'Numerical tolerance exceeded: normalized error {normalized}')
    return dict(max_abs_error=float(np.max(np.abs(difference))),
                rmse_error=float(np.sqrt(np.mean(difference*difference))),
                max_tolerance_fraction=normalized)


def make_model(config=MODEL_CONFIG):
    from physicsnemo.models.fno import FNO
    return FNO(**config).eval()


def double_reference(model):
    import torch
    result = copy.deepcopy(model).double()
    # FNO's current spectral weights are real tensors encoding complex values.
    # Reject a future complex64 parameter rather than call it FP64 arithmetic.
    if any(p.dtype not in [torch.float64, torch.complex128] for p in result.parameters()):
        raise ValueError('Reference model parameters are not entirely double precision')
    return result


def stats(rows):
    values = np.array([float(r['seconds']) for r in rows])
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Expected positive finite timing samples')
    return dict(samples=len(values), p50_sec=float(np.percentile(values, 50)),
                p95_sec=float(np.percentile(values, 95)), max_sec=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--order-seed', type=int, default=20261102)
    args = parser.parse_args()
    cases = parse_cases(args.cases)
    if args.output.exists() or args.repeats < 2 or args.warmup < 1 or args.threads < 1:
        parser.error('Need a new output path, >=2 repeats, >=1 warmup and positive CPU threads')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    start = time.perf_counter()
    import torch
    torch_import_sec = time.perf_counter()-start
    if not torch.cuda.is_available():
        parser.error('CUDA-enabled PyTorch required; no silent CPU fallback')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    from physicsnemo.models.fno import FNO
    import_sec = time.perf_counter()-start
    start = time.perf_counter()
    model = make_model()
    build_sec = time.perf_counter()-start
    state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    reference_model = double_reference(model)
    source_sha = sha256(args.data)
    manifest = json.loads((args.data.parent/'manifest.json').read_text(encoding='utf-8'))
    if source_sha != manifest['files']['batch.zarr']:
        raise ValueError('Source data/manifest mismatch')
    start = time.perf_counter()
    torch.cuda.set_device(0)
    torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    context_sec = time.perf_counter()-start
    start = time.perf_counter()
    gpu_model = copy.deepcopy(model).to('cuda')
    torch.cuda.synchronize()
    model_upload_sec = time.perf_counter()-start
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'weights.npz', **state)
    from precision_benchmark import hardware_snapshot
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), stage='5A-forward-and-initial-benchmark',
               scope='Untrained architecture performance and numerical correctness only; no forecast quality assessment',
               python=platform.python_version(), platform=platform.platform(), executable=__import__('sys').executable,
               versions={k: importlib.metadata.version(k) for k in ['torch', 'nvidia-physicsnemo', 'numpy', 'xarray', 'zarr', 'scipy']},
               cuda_runtime=torch.version.cuda, model_class='physicsnemo.models.fno.FNO', model_config=MODEL_CONFIG,
               parameter_count=sum(p.numel() for p in model.parameters()),
               model_state_sha256=array_mapping_hash(state), weights_sha256=sha256(args.output/'weights.npz'),
               model_role='Fixed seeded random initialization; untrained, no pretrained checkpoint or training',
               source_sha256=source_sha, source_manifest=manifest, cases=cases, methods=METHODS,
               repeats=args.repeats, warmup=args.warmup, threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
               seed=args.seed, order_seed=args.order_seed, tolerance=TOLERANCE,
               precision=dict(inputs='float32', model='float32', cpu_reference='FP64 forward with FP32 state/inputs cast to double',
                              tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                              cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
               setup_seconds=dict(torch_import=torch_import_sec, physicsnemo_fno_import=import_sec, cpu_model_build=build_sec,
                                  observed_cuda_context=context_sec, model_upload=model_upload_sec),
               timing='All methods reuse model weights. cpu_forward returns full CPU output; cuda_forward_return reuses GPU input and returns full CPU output; cuda_upload_forward_return uploads existing packed host input each call and returns full CPU output. cuda_device_only leaves output on device and synchronizes completion; its host verification copy is outside timing. Zarr read/transform, model construction/loading, context initialization, FP64 reference, numerical checking and result writing excluded from steady-state timings.',
               ordering='Cases in listed order; methods shuffled within each repetition. Separate first-forward observations precede method warmups; process/model/context already initialized.',
               memory_note='Peak allocated/reserved are PyTorch CUDA allocator observations per isolated invocation after warmup; not total GPU/driver/cuFFT process memory. CUDA input/model reused; no allocator cache clearing between methods.',
               before=hardware_snapshot(torch), after=None, case_metadata={})
    samples, summaries, first, memory = [], [], [], []
    with torch.inference_mode():
        for case in cases:
            start = time.perf_counter()
            x, identity = load_input(args.data, case)
            input_sec = time.perf_counter()-start
            cpu_input = torch.from_numpy(x)
            start = time.perf_counter()
            truth = reference_model(cpu_input.double()).numpy().copy()
            reference_sec = time.perf_counter()-start
            if truth.dtype != np.float64:
                raise ValueError('Reference output is not FP64')
            start = time.perf_counter()
            saved_cpu = model(cpu_input).numpy().copy()
            first.append(dict(case=case['case'], backend='cpu_fp32', seconds=time.perf_counter()-start,
                              **error_summary(saved_cpu, truth)))
            gpu_input = cpu_input.to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            saved_gpu = gpu_model(gpu_input).cpu().numpy().copy()
            torch.cuda.synchronize()
            first.append(dict(case=case['case'], backend='cuda_fp32', seconds=time.perf_counter()-start,
                              **error_summary(saved_gpu, truth)))
            identity.update(input_read_transform_sec=input_sec, reference_forward_sec=reference_sec,
                            input_sha256=array_mapping_hash({'input': x}), input_shape=list(x.shape), output_shape=list(truth.shape))
            env['case_metadata'][case['case']] = identity
            np.savez_compressed(args.output/(case['case']+'.npz'), input_fp32=x,
                                reference_fp64=truth, cpu_fp32=saved_cpu, cuda_fp32=saved_gpu)
            def cpu():
                return model(cpu_input).numpy()
            def resident():
                return gpu_model(gpu_input).cpu().numpy()
            def with_upload():
                return gpu_model(cpu_input.to('cuda')).cpu().numpy()
            def device_only():
                return gpu_model(gpu_input)
            methods = dict(zip(METHODS, [cpu, resident, with_upload, device_only]))
            def check(result):
                output = result.cpu().numpy() if isinstance(result, torch.Tensor) else result
                return error_summary(output, truth)
            for name, fn in methods.items():
                for _ in range(args.warmup):
                    result = fn()
                    check(result)
                    del result
                if name != 'cpu_forward':
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    result = fn()
                    torch.cuda.synchronize()
                    memory.append(dict(case=case['case'], method=name, baseline_allocated_bytes=baseline,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-baseline,
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    check(result)
                    del result
            rng = np.random.default_rng(args.order_seed)
            for repeat in range(args.repeats):
                for order, name in enumerate(rng.permutation(METHODS)):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    result = methods[name]()
                    torch.cuda.synchronize()
                    seconds = (time.perf_counter_ns()-start)/1e9
                    error = check(result)
                    samples.append(dict(case=case['case'], method=str(name), repeat=repeat, order=order, seconds=seconds, **error))
                    del result
            for name in METHODS:
                summary = dict(case=case['case'], method=name, **stats([r for r in samples if r['case'] == case['case'] and r['method'] == name]))
                summary['frames_per_second'] = case['batch']/summary['p50_sec']
                summaries.append(summary)
            env['case_metadata'][case['case']]['artifact_sha256'] = sha256(args.output/(case['case']+'.npz'))
            print(json.dumps(dict(case=case['case'], p50_ms={s['method']: round(s['p50_sec']*1000, 3) for s in summaries if s['case'] == case['case']})), flush=True)
            methods.clear()
            del gpu_input, cpu_input, x, truth, saved_cpu, saved_gpu
    env['after'] = hardware_snapshot(torch)
    write_csv(args.output/'timing_samples.csv', samples)
    write_csv(args.output/'timing_summary.csv', summaries)
    write_csv(args.output/'first_forward.csv', first)
    write_csv(args.output/'memory.csv', memory)
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: {len(samples)} measured outputs agree with CPU FP64 reference; untrained FNO workload', flush=True)


if __name__ == '__main__':
    main()
