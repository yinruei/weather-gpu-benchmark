"""Reproducible FP32 verification benchmark; CUDA is optional."""
import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def numpy_metrics(forecast, observation):
    if forecast.shape != observation.shape or forecast.size == 0:
        raise ValueError('Inputs must have identical, nonempty shapes')
    if not (np.isfinite(forecast).all() and np.isfinite(observation).all()):
        raise ValueError('Inputs must be finite; missing-data policy is not implemented')
    return numpy_kernel(forecast, observation)


def numpy_kernel(forecast, observation):
    error = forecast - observation
    return np.mean(error), np.mean(np.abs(error)), np.sqrt(np.mean(error * error))


def torch_kernel(forecast, observation):
    error = forecast - observation
    return error.mean(), error.abs().mean(), (error * error).mean().sqrt()


def measure(fn, repeats, warmup, sync=lambda: None):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        samples.append(time.perf_counter() - start)
    return result, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000000, 5000000, 10000000, 20000000, 50000000])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU thread count; NumPy thread env vars are recorded')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--numpy-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/latest'))
    args = parser.parse_args()
    if min(args.sizes) <= 0 or args.repeats < 2 or args.warmup < 1 or args.threads < 1:
        parser.error('Positive sizes/threads, at least 2 repeats and 1 warmup required')
    torch = None
    if not args.numpy_only:
        try:
            import torch
        except ImportError:
            parser.error('Install PyTorch or explicitly use --numpy-only')
        torch.set_num_threads(args.threads)
    cuda = torch is not None and torch.cuda.is_available()
    import os
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    processor=platform.processor(), numpy=np.__version__,
                    torch=torch.__version__ if torch else None,
                    cuda_runtime=torch.version.cuda if torch else None,
                    gpu=torch.cuda.get_device_name(0) if cuda else None,
                    torch_threads=args.threads if torch else None,
                    thread_environment={k: os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                    seed=args.seed, sizes=args.sizes, repeats=args.repeats, warmup=args.warmup,
                    dtype='float32', missing_data='reject', aggregation='unweighted global',
                    timing='perf_counter wall time; CUDA synchronized; validation outside timing')
    rows, raw = [], []
    for n in args.sizes:
        rng = np.random.default_rng(args.seed)
        obs = rng.normal(280, 10, n).astype(np.float32)
        fcst = obs + rng.normal(0.5, 2, n).astype(np.float32)
        reference = numpy_metrics(fcst, obs)
        row = dict(points=n, bias=float(reference[0]), mae=float(reference[1]), rmse=float(reference[2]))
        def record(name, fn, sync=lambda: None):
            result, samples = measure(fn, args.repeats, args.warmup, sync)
            row[name + '_median_sec'] = statistics.median(samples)
            row[name + '_p25_sec'] = float(np.percentile(samples, 25))
            row[name + '_p75_sec'] = float(np.percentile(samples, 75))
            raw.extend(dict(points=n, method=name, repeat=i, seconds=s) for i, s in enumerate(samples))
            return result
        record('numpy_cpu', lambda: numpy_kernel(fcst, obs))
        if torch is not None:
            f, o = torch.from_numpy(fcst), torch.from_numpy(obs)
            result = record('torch_cpu', lambda: torch_kernel(f, o))
            np.testing.assert_allclose([v.item() for v in result], reference, rtol=1e-5, atol=1e-5)
            if cuda:
                sync = torch.cuda.synchronize
                # Initialize context before measuring any transfer.
                torch.empty(1, device='cuda'); sync()
                fg, og = f.to('cuda'), o.to('cuda')
                sync()
                torch.cuda.reset_peak_memory_stats()
                result = record('torch_cuda', lambda: torch_kernel(fg, og), sync)
                row['gpu_peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
                np.testing.assert_allclose([v.item() for v in result], reference, rtol=1e-5, atol=1e-5)
                # Reuse allocations: transfer measures two pageable CPU arrays -> device buffers.
                record('h2d', lambda: (fg.copy_(f), og.copy_(o)), sync)
                def end_to_end():
                    fg.copy_(f); og.copy_(o)
                    return tuple(v.item() for v in torch_kernel(fg, og))
                result = record('gpu_e2e', end_to_end, sync)
                np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5)
                row['gpu_compute_speedup'] = row['numpy_cpu_median_sec'] / row['torch_cuda_median_sec']
                row['gpu_e2e_speedup'] = row['numpy_cpu_median_sec'] / row['gpu_e2e_median_sec']
                del fg, og, result
            del f, o
        rows.append(row)
        print(json.dumps(row))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, records in [('summary.csv', rows), ('samples.csv', raw)]:
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (args.output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(records)
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
