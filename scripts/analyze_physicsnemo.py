"""Validate saved FNO artifacts offline; optionally recompute FP64 on CPU."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import MODEL_CONFIG, METHODS, TOLERANCE, array_mapping_hash, double_reference, error_summary, load_input, make_model, stats
from weather_verify import sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data, recompute=False):
    env = json.loads((run/'environment.json').read_text(encoding='utf-8'))
    manifest = json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(data)
    if env['source_sha256'] != digest or manifest != env['source_manifest'] or manifest['files']['batch.zarr'] != digest:
        raise ValueError('Source/provenance mismatch')
    if env['model_config'] != MODEL_CONFIG or env['methods'] != METHODS or env['tolerance'] != TOLERANCE:
        raise ValueError('Unexpected architecture, method or tolerance')
    if env['precision']['tf32_matmul'] or env['precision']['tf32_cudnn']:
        raise ValueError('Expected IEEE FP32 controls for this benchmark')
    weights_path = run/'weights.npz'
    if sha256(weights_path) != env['weights_sha256']:
        raise ValueError('Model state file hash mismatch')
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    if array_mapping_hash(weights) != env['model_state_sha256']:
        raise ValueError('Different model tensor state')
    reference_model = None
    if recompute:
        import torch
        torch.set_num_threads(env['threads'])
        model = make_model(env['model_config'])
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
        reference_model = double_reference(model)
    sample_rows, summary_rows = rows(run/'timing_samples.csv'), rows(run/'timing_summary.csv')
    first_rows, memory_rows = rows(run/'first_forward.csv'), rows(run/'memory.csv')
    summaries = {}
    for row in summary_rows:
        key = (row['case'], row['method'])
        if key in summaries:
            raise ValueError('Duplicate summary')
        summaries[key] = row
    keys = {(c['case'], m) for c in env['cases'] for m in METHODS}
    if set(summaries) != keys:
        raise ValueError('Missing/extra timing group')
    expected_order = []
    output_errors = {}
    for case in env['cases']:
        name = case['case']
        x, identity = load_input(data, case)
        meta = env['case_metadata'][name]
        if any(meta[k] != v for k, v in identity.items()):
            raise ValueError('Source frame identities or normalization changed')
        if meta['input_sha256'] != array_mapping_hash({'input': x}):
            raise ValueError('Input tensor hash mismatch')
        artifact_path = run/(name+'.npz')
        if sha256(artifact_path) != meta['artifact_sha256']:
            raise ValueError('Output artifact hash mismatch')
        with np.load(artifact_path, allow_pickle=False) as archive:
            if set(archive.files) != {'input_fp32', 'reference_fp64', 'cpu_fp32', 'cuda_fp32'}:
                raise ValueError('Missing saved backend/input arrays')
            np.testing.assert_array_equal(archive['input_fp32'], x)
            reference = archive['reference_fp64']
            if reference.dtype != np.float64 or list(reference.shape) != meta['output_shape'] or reference.shape != x.shape:
                raise ValueError('Incorrect FP64 reference dtype or shape')
            output_errors[name] = {backend: error_summary(archive[backend], reference) for backend in ['cpu_fp32', 'cuda_fp32']}
            for backend in ['cpu_fp32', 'cuda_fp32']:
                if archive[backend].dtype != np.float32:
                    raise ValueError('Expected FP32 backend output')
            if recompute:
                import torch
                with torch.inference_mode():
                    expected = reference_model(torch.from_numpy(x).double()).numpy()
                np.testing.assert_allclose(reference, expected, rtol=1e-10, atol=1e-11)
        rng = np.random.default_rng(env['order_seed'])
        expected_order.extend((name, str(method), repeat, order) for repeat in range(env['repeats'])
                              for order, method in enumerate(rng.permutation(METHODS)))
        for method in METHODS:
            selected = [r for r in sample_rows if r['case'] == name and r['method'] == method]
            if sorted(int(r['repeat']) for r in selected) != list(range(env['repeats'])):
                raise ValueError('Missing timing repetitions')
            for field, value in stats(selected).items():
                np.testing.assert_allclose(float(summaries[(name, method)][field]), value, rtol=1e-12)
            np.testing.assert_allclose(float(summaries[(name, method)]['frames_per_second']), case['batch']/float(summaries[(name, method)]['p50_sec']), rtol=1e-12)
        for backend in ['cpu_fp32', 'cuda_fp32']:
            first = [r for r in first_rows if r['case'] == name and r['backend'] == backend]
            if len(first) != 1 or not np.isfinite(float(first[0]['seconds'])) or float(first[0]['seconds']) <= 0:
                raise ValueError('Missing first-forward observation')
            for k, v in output_errors[name][backend].items():
                np.testing.assert_allclose(float(first[0][k]), v, rtol=1e-10, atol=1e-15)
    actual_order = [(r['case'], r['method'], int(r['repeat']), int(r['order'])) for r in sample_rows]
    if actual_order != expected_order:
        raise ValueError('Method execution order differs from seed')
    for row in sample_rows:
        for field in ['max_abs_error', 'rmse_error', 'max_tolerance_fraction']:
            assert np.isfinite(float(row[field])) and float(row[field]) >= 0
        assert float(row['max_tolerance_fraction']) <= 1
    if len(first_rows) != 2*len(env['cases']):
        raise ValueError('Unexpected first-forward rows')
    expected_memory = {(c['case'], m) for c in env['cases'] for m in METHODS if m != 'cpu_forward'}
    if len(memory_rows) != len(expected_memory) or {(r['case'], r['method']) for r in memory_rows} != expected_memory:
        raise ValueError('Missing/duplicate memory observation')
    for row in memory_rows:
        baseline, peak, extra, reserved = [int(row[k]) for k in ['baseline_allocated_bytes', 'peak_allocated_bytes', 'peak_extra_allocated_bytes', 'peak_reserved_bytes']]
        assert 0 <= baseline <= peak <= reserved and peak-baseline == extra
    print(f'PASS {run.name}: {len(sample_rows)} timed outputs, {len(env["cases"])} full backend artifacts, {len(memory_rows)} memory observations'+('; FP64 independently recomputed' if recompute else ''))
    return dict(path=run, env=env, summary=summaries, errors=output_errors, memory=memory_rows, first=first_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recompute-reference', action='store_true', help='Needs the isolated PhysicsNeMo/PyTorch environment; recomputes every saved FP64 output on CPU')
    args = parser.parse_args()
    runs = [validate(path, args.data, args.recompute_reference) for path in args.runs]
    first = runs[0]
    cases = first['env']['cases']
    if any(r['env']['cases'] != cases or r['env']['model_state_sha256'] != first['env']['model_state_sha256'] for r in runs):
        raise ValueError('Sessions must share cases and identical model tensor state')
    def milliseconds(run, case, method, field='p50_sec'):
        return float(run['summary'][(case['case'], method)][field])*1000
    names = [c['case'] for c in cases]
    labels = [f'B{c["batch"]}\n{c["side"]}x{c["side"]}' for c in cases]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for method, label, color, style in [('cpu_forward', 'CPU, full output', '#3366aa', '-'),
                                       ('cuda_forward_return', 'GPU resident input + full return', '#dd7722', '-'),
                                       ('cuda_upload_forward_return', 'GPU upload + full return', '#228855', '-'),
                                       ('cuda_device_only', 'GPU device-only (different boundary)', '#888888', '--')]:
        values = np.array([[milliseconds(r, c, method) for c in cases] for r in runs])
        axes[0, 0].plot(range(len(cases)), np.median(values, axis=0), 'o'+style, color=color, label=label)
        axes[0, 0].fill_between(range(len(cases)), values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
    axes[0, 0].set(yscale='log', ylabel='Median forward latency (ms)', title='FP32 FNO; fixed weights; model/input setup excluded')
    axes[0, 0].legend(fontsize=8)
    for method, label, color in [('cuda_forward_return', 'Resident input + full return', '#dd7722'),
                                  ('cuda_upload_forward_return', 'Upload + full return', '#228855')]:
        values = np.array([[milliseconds(r, c, 'cpu_forward')/milliseconds(r, c, method) for c in cases] for r in runs])
        axes[0, 1].plot(range(len(cases)), np.median(values, axis=0), 'o-', color=color, label=label)
        axes[0, 1].fill_between(range(len(cases)), values.min(axis=0), values.max(axis=0), color=color, alpha=.15)
    axes[0, 1].axhline(1, linestyle='--', color='black', linewidth=1)
    axes[0, 1].set(ylabel='CPU latency / GPU latency', title='Comparable output boundaries; >1 means GPU faster')
    axes[0, 1].legend(fontsize=8)
    for i, (method, label) in enumerate([('cuda_forward_return', 'Resident input'), ('cuda_upload_forward_return', 'Upload each call')]):
        values = [max(int(row['peak_allocated_bytes']) for r in runs for row in r['memory'] if row['case'] == name and row['method'] == method)/2**20 for name in names]
        bars = axes[1, 0].bar(np.arange(len(cases))+(i-.5)*.36, values, width=.34, label=label)
        axes[1, 0].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axes[1, 0].set(ylabel='PyTorch peak allocated (MiB)', title='Warm invocation allocator peak; not total process VRAM')
    axes[1, 0].legend(fontsize=8)
    for backend, label in [('cpu_fp32', 'CPU FP32'), ('cuda_fp32', 'CUDA FP32')]:
        values = [max(r['errors'][name][backend]['max_tolerance_fraction'] for r in runs) for name in names]
        axes[1, 1].plot(range(len(cases)), values, 'o-', label=label)
    axes[1, 1].axhline(1, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set(yscale='log', ylabel='Max |error| / (atol + rtol * |reference|)', title='Numerical tolerance usage: <=1 passes; FP64 reference')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(range(len(cases)), labels)
        ax.set_xlabel('Batch and native input grid size')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Phase 5A: PhysicsNeMo small FNO on RTX 3050\nFixed random weights, untrained architecture; real Zarr input frames; performance and numeric checks only', fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo.png', dpi=160)
    plt.close(fig)
    env = first['env']
    text = ['# Phase 5A: small PhysicsNeMo FNO forward performance', '',
            '**Completed:** isolated Python/CUDA environment, official FNO forward, CPU/RTX 3050 timing/memory and full-output numerical checks. This is a fixed-seed, untrained architecture benchmark, not a pretrained forecast model. No training, model-checkpoint download or forecast-quality assessment is involved. Earth2Studio model pipelines (5C) remain future work.', '',
            f'Environment: Python {env["python"]}, PhysicsNeMo {env["versions"]["nvidia-physicsnemo"]}, PyTorch {env["versions"]["torch"]}, CUDA {env["cuda_runtime"]}; CPU threads {env["threads"]}, interop threads {env["interop_threads"]}. Model has {env["parameter_count"]:,} parameters. [Exact model configuration and recorded controls](../'+first['path'].name+'/environment.json). TF32 is disabled, inference mode/eval are used, and deterministic algorithms are requested.', '',
            '## Comparable steady-state timings', '',
            '| Session | Batch/grid | CPU P50 ms | GPU resident + return P50 ms | CPU/GPU ratio | GPU upload + return P50 ms | GPU resident P95 ms | Resident batch frames/s |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for case in cases:
            cpu, gpu, uploaded = [milliseconds(run, case, m) for m in METHODS[:3]]
            text.append(f'| {run["path"].name} | {case["batch"]} / {case["side"]}x{case["side"]} | {cpu:.4f} | {gpu:.4f} | {cpu/gpu:.2f}x | {uploaded:.4f} | {milliseconds(run,case,"cuda_forward_return","p95_sec"):.4f} | {case["batch"]/(gpu/1000):.1f} |')
    text += ['', 'CPU and the two main GPU paths produce full host outputs. All reuse model weights; GPU resident input is preuploaded. The upload method copies existing packed host input each call. Device-only CUDA is a separately labeled diagnostic with output left on device and completion synchronized; its untimed verification copy is not part of that latency. Batch frames/s is batch size divided by synchronous batch latency, not an asynchronously streamed service-throughput measurement.', '',
             'Zarr reads, input transform, model import/construction/upload, CUDA initialization, FP64 reference work, output validation and NPZ/CSV writing are excluded from steady-state timings. Thus this is not yet the complete read-to-model-to-file application benchmark. Cases run in listed order; methods are shuffled within each repetition with recorded order seeds. Shading is the range of session medians, not a confidence interval. P95 values from these short sessions are sample estimates, not latency guarantees.', '',
             '## Model/input identity and numerical correctness', '',
             'The source is the checked-in pinned ECMWF ensemble Zarr. Batch frames are distinct initialization/lead/member selections in native coordinate order, not repeated/upsampled tiles. Grid 32 uses the southern 32 latitude rows and first 32 longitudes; grid 64 uses the full local native grid. Source Celsius values undergo the fixed benchmark transform `(C - 20)/20` in FP64 then FP32. This is input scaling for an untrained workload, not training-derived weather normalization. Exact frame identities, coordinate bounds, input hashes and source manifest are saved.', '',
             'CPU and GPU use identical serialized FP32 model state. The reference runs the same architecture on CPU after casting those exact weights/inputs to FP64; every parameter and output is checked to be double precision. All timed, warmup, first-forward and memory-probe results are checked elementwise at rtol=1e-4, atol=1e-5. The plotted normalized tolerance fraction must be <=1. These are numerical arithmetic differences, not forecast errors or skill scores.', '',
             'Every case NPZ stores actual packed FP32 input, complete CPU FP64 reference output, CPU FP32 output and CUDA FP32 output. The CPU-only analyzer reloads source frames, checks hashes, compares saved outputs and reconciles quantiles, ordering, first observations and memory arithmetic. In the isolated environment, `--recompute-reference` independently reloads weights and recomputes every FP64 output.', '',
             '## Memory and first-forward boundaries', '',
             'memory.csv records live allocated baseline, peak allocated, extra allocated and peak reserved per warmed GPU invocation. The graph uses the maximum observed allocated peak across sessions for each case/method. Allocator-reserved space may include previous workloads, and driver/cuFFT allocations are not necessarily fully represented; none of these is total process VRAM or a fresh-process per-case peak. Input/weights remain resident, and allocator cache is not cleared between methods.', '',
             'first_forward.csv separately records the first observed FP32 CPU/CUDA forward per shape. Package imports, CPU model construction, observed CUDA setup and model upload are recorded in environment.json. These observations occur in an initialized process, following other setup/reference work; they are not a full cold-process startup measurement. They are kept separate from warmed timing samples.', '',
             '## Reproduce', '', '```bash', 'bash scripts/setup_physicsnemo.sh',
             'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_benchmark.py --output results/my_physicsnemo01',
             'python physicsnemo_benchmark.py --order-seed 20261103 --output results/my_physicsnemo02',
             'python scripts/analyze_physicsnemo.py results/my_physicsnemo01 results/my_physicsnemo02 --recompute-reference --output results/my_physicsnemo_comparison', '```', '',
             'The existing Python 3.10 weather-gpu environment remains a separate baseline. New environment versions and CPU thread controls must be considered before comparing with older metric-kernel experiments, which perform different work. CUDA 12.6 torchvision and the official NVIDIA Warp CUDA12 wheel are pinned to avoid mismatched default binaries; the complete environment lock is checked in. No driver, global Python, JAX or Earth2Studio installation is required by this case.', '',
             '[Official PhysicsNeMo FNO API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/fnos.html), [phase-5 plan](../../docs/phase5_plan_zh.md).', '',
             '![FNO forward timing, batching, allocator peak and numerical agreement](physicsnemo.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Saved validated PhysicsNeMo report and plot: {args.output}')


if __name__ == '__main__':
    main()
