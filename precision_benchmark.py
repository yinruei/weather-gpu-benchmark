"""Compare CUDA FP32, native FP16 and FP16 storage with FP32 arithmetic.

Inputs are GPU-resident; every timed call returns three host scalars.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


MODES = ('fp32', 'fp16_native', 'fp16_fp32_math')
METRICS = ('bias', 'mae', 'rmse')


def reference_metrics(forecast, observation):
    """FP64 arithmetic reference for the exact supplied input values."""
    if forecast.shape != observation.shape or forecast.size == 0:
        raise ValueError('Inputs must have identical, nonempty shapes')
    if not (np.isfinite(forecast).all() and np.isfinite(observation).all()):
        raise ValueError('Inputs must be finite')
    error = forecast.astype(np.float64) - observation.astype(np.float64)
    return np.array([error.mean(), np.abs(error).mean(), np.sqrt((error * error).mean())])


def precision_kernel(forecast, observation, mode):
    """Native FP16 keeps intermediates in half; mixed casts BEFORE subtraction."""
    if mode not in MODES:
        raise ValueError(f'Unknown precision mode: {mode}')
    if mode == 'fp16_fp32_math':
        error = forecast.float() - observation.float()
    else:
        error = forecast - observation
    return error.mean(), error.abs().mean(), (error * error).mean().sqrt()


def optional_command(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL,
                                       timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_snapshot(torch):
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu_model = next((s.split(':', 1)[1].strip() for s in cpuinfo.read_text().splitlines()
                          if s.startswith('model name')), cpu_model)
    props = torch.cuda.get_device_properties(0)
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                cpu_model=cpu_model, logical_cpus=os.cpu_count(),
                gpu=props.name, gpu_total_memory_bytes=props.total_memory,
                compute_capability=[props.major, props.minor],
                nvidia_smi=optional_command([
                    'nvidia-smi', '--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw,utilization.gpu',
                    '--format=csv']))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000000, 5000000, 10000000])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are rejected')
    args = parser.parse_args()
    if min(args.sizes) <= 0 or len(set(args.sizes)) != len(args.sizes):
        parser.error('Sizes must be positive and unique')
    if args.repeats < 2 or args.warmup < 1 or args.threads < 1:
        parser.error('At least two repeats, one warmup and one CPU thread are required')
    try:
        import torch
    except ImportError:
        parser.error('Install CUDA-enabled PyTorch in the active environment')
    if not torch.cuda.is_available():
        parser.error('CUDA is required; no CPU fallback is used for this experiment')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory to preserve prior results')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    env = dict(python=platform.python_version(), platform=platform.platform(),
               numpy=np.__version__, torch=torch.__version__, cuda_runtime=torch.version.cuda,
               sizes=args.sizes, modes=list(MODES), repeats=args.repeats, warmup=args.warmup,
               seed=args.seed, torch_threads=args.threads,
               thread_environment={k: os.environ.get(k) for k in
                                   ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
               timing='perf_counter wall time; CUDA synchronized; includes scalar return',
               boundary='resident GPU inputs -> metrics -> three host scalars; excludes upload, host quantization, validation and file IO',
               mixed_boundary='FP16 resident inputs cast to FP32 before subtraction inside every timed call; FP32 intermediates and reductions',
               native_boundary='FP16 tensors and intermediate/output dtypes; internal reduction accumulation is backend-defined',
               reference='NumPy FP64 arithmetic on original FP32 source values; separate FP64 reference on quantized FP16 inputs',
               tolerance=dict(rtol=1e-5, atol=1e-5, meaning='diagnostic agreement only; not an application accuracy requirement'),
               memory='peak torch.cuda.memory_allocated minus pre-input baseline; includes active mode inputs and temporaries; not reserved or total device usage',
               input_distribution='obs normal(280,10); forecast=obs+normal(0.5,2); float32 source arrays',
               mode_order={}, before=hardware_snapshot(torch))
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    rows, samples = [], []
    for n in args.sizes:
        rng = np.random.default_rng(args.seed)
        obs = rng.normal(280, 10, n).astype(np.float32)
        fcst = obs + rng.normal(0.5, 2, n).astype(np.float32)
        source_reference = reference_metrics(fcst, obs)
        half_fcst, half_obs = fcst.astype(np.float16), obs.astype(np.float16)
        half_reference = reference_metrics(half_fcst, half_obs)
        order = np.random.default_rng(args.seed + n).permutation(MODES).tolist()
        env['mode_order'][str(n)] = order
        for mode in order:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            idle_bytes = torch.cuda.memory_allocated()
            fhost, ohost = (fcst, obs) if mode == 'fp32' else (half_fcst, half_obs)
            f = torch.from_numpy(fhost).to('cuda')
            o = torch.from_numpy(ohost).to('cuda')
            torch.cuda.synchronize()
            input_bytes = f.numel() * f.element_size() + o.numel() * o.element_size()

            def execute():
                return tuple(v.item() for v in precision_kernel(f, o, mode))

            for _ in range(args.warmup):
                execute()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times, values = [], []
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = execute()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                times.append(seconds)
                values.append(result)
                record = dict(points=n, mode=mode, repeat=repeat, seconds=seconds,
                              finite=bool(np.isfinite(result).all()))
                record.update({k: float(v) if np.isfinite(v) else None for k, v in zip(METRICS, result)})
                samples.append(record)
            peak_bytes = torch.cuda.max_memory_allocated() - idle_bytes
            values = np.array(values)
            finite = bool(np.isfinite(values).all())
            input_reference = source_reference if mode == 'fp32' else half_reference
            arithmetic_ok = finite and bool(np.allclose(values, input_reference, rtol=1e-5, atol=1e-5))
            source_ok = finite and bool(np.allclose(values, source_reference, rtol=1e-5, atol=1e-5))
            row = dict(points=n, mode=mode, median_sec=float(np.median(times)),
                       p25_sec=float(np.percentile(times, 25)), p75_sec=float(np.percentile(times, 75)),
                       input_mib=input_bytes / 2**20, peak_allocated_mib=peak_bytes / 2**20,
                       idle_allocated_bytes=idle_bytes, finite=finite,
                       agrees_with_source=source_ok, agrees_with_quantized_reference=arithmetic_ok)
            for i, metric in enumerate(METRICS):
                row[metric] = float(np.median(values[:, i])) if np.isfinite(values[:, i]).all() else None
                row[f'{metric}_reference_fp64'] = float(source_reference[i])
                row[f'{metric}_input_reference_fp64'] = float(input_reference[i])
                row[f'{metric}_quantization_delta'] = float(input_reference[i] - source_reference[i])
                row[f'{metric}_max_abs_error'] = float(np.max(np.abs(values[:, i] - source_reference[i]))) if finite else None
                row[f'{metric}_max_abs_arithmetic_error'] = float(np.max(np.abs(values[:, i] - input_reference[i]))) if finite else None
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            del f, o
            torch.cuda.synchronize()
        del fcst, obs, half_fcst, half_obs, fhost, ohost
    env['after'] = hardware_snapshot(torch)
    env['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    write_csv(args.output / 'summary.csv', rows)
    write_csv(args.output / 'samples.csv', samples)
    # Native half may fail its accuracy check; preserve and report that tradeoff.
    if any(not r['agrees_with_quantized_reference'] for r in rows if r['mode'] != 'fp16_native'):
        raise SystemExit('FAIL: FP32 or mixed arithmetic disagrees with its FP64 input reference; results saved for inspection')
    print(f'Saved {len(samples)} timings and per-repeat metric values to {args.output}')


if __name__ == '__main__':
    main()
