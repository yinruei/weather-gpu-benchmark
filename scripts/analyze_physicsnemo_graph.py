"""Offline CUDA Graph timing/update/memory/trace checks; optional CPU FP64 rebuild."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import MODEL_CONFIG, TOLERANCE, array_mapping_hash, double_reference, error_summary, make_model, stats
from physicsnemo_graph import CASES, METHODS, load_variant, request_order, summarize_trace
from weather_verify import sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data, baseline, recompute=False):
    env = json.loads((run/'environment.json').read_text())
    base = json.loads((baseline/'environment.json').read_text())
    source_hash = sha256(data)
    manifest = json.loads((data.parent/'manifest.json').read_text())
    if source_hash != env['source_sha256'] or source_hash != base['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or env['source_manifest'] != manifest:
        raise ValueError('Source/provenance mismatch')
    if env['cases'] != CASES or env['methods'] != METHODS or env['model_config'] != MODEL_CONFIG or env['tolerance'] != TOLERANCE or env['tf32'] or not env['deterministic_algorithms']:
        raise ValueError('Protocol mismatch')
    if env['weights_sha256'] != base['weights_sha256'] or sha256(run/'weights.npz') != base['weights_sha256'] or env['model_state_sha256'] != base['model_state_sha256']:
        raise ValueError('Weight state mismatch')
    with np.load(run/'weights.npz', allow_pickle=False) as saved:
        weights = {k: saved[k] for k in saved.files}
    if array_mapping_hash(weights) != env['model_state_sha256']:
        raise ValueError('Tensor state mismatch')
    reference_model = None
    if recompute:
        import torch
        torch.set_num_threads(env['threads'])
        model = make_model()
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)
        reference_model = double_reference(model)
    sample, summary, updates, memory, traced = [rows(run/f) for f in ['timing_samples.csv', 'timing_summary.csv', 'update_checks.csv', 'memory_samples.csv', 'trace_calls.csv']]
    def check_order(records, expected):
        if len(records) != len(expected):
            raise ValueError('Missing/extra records')
        for row, wanted in zip(records, expected):
            if any((int(row[k]) if isinstance(v, int) else row[k]) != v for k, v in wanted.items()):
                raise ValueError('Pairing/order/input variant changed')
    check_order(sample, request_order(env['repeats'], env['seed']))
    check_order(traced, request_order(env['trace_repeats'], env['seed']+1000))
    check_order(updates, [dict(case=c['case'], step=i, variant=v) for c in CASES for i, v in enumerate([0, 1, 0])])
    summaries = {(r['case'], r['method']): r for r in summary}
    keys = {(c['case'], m) for c in CASES for m in METHODS}
    if set(summaries) != keys or len(summary) != len(keys):
        raise ValueError('Missing/duplicate timing summaries')
    output_info = {}
    for case in CASES:
        name = case['case']
        meta = env['case_metadata'][name]
        reference_path = name+'_reference.npz'
        if meta['reference_path'] != reference_path or sha256(run/reference_path) != meta['reference_sha256']:
            raise ValueError('Reference file hash/path mismatch')
        baseline_path = baseline/(name+'.npz')
        if meta['baseline_artifact_sha256'] != base['case_metadata'][name]['artifact_sha256'] or sha256(baseline_path) != meta['baseline_artifact_sha256']:
            raise ValueError('Original A reference mismatch')
        with np.load(run/reference_path, allow_pickle=False) as saved:
            if set(saved.files) != {'input_A', 'input_B', 'reference_A', 'reference_B'}:
                raise ValueError('Incomplete reference inputs/outputs')
            references = [saved['reference_'+v].copy() for v in ['A', 'B']]
            for variant, label in enumerate(['A', 'B']):
                x, identity = load_variant(data, case, variant)
                if identity != meta['identities'][variant] or meta['input_hashes'][variant] != array_mapping_hash({'input': x}) or x.nbytes != meta['input_bytes']:
                    raise ValueError('Input frame identity, bytes or transform changed')
                np.testing.assert_array_equal(x, saved['input_'+label])
                if references[variant].dtype != np.float64 or references[variant].shape != x.shape:
                    raise ValueError('FP64 reference dtype/shape mismatch')
                if recompute:
                    import torch
                    with torch.inference_mode():
                        fresh = reference_model(torch.from_numpy(x).double()).numpy()
                    np.testing.assert_allclose(fresh, references[variant], rtol=1e-10, atol=1e-11)
        with np.load(baseline_path, allow_pickle=False) as old:
            np.testing.assert_allclose(references[0], old['reference_fp64'], rtol=1e-10, atol=1e-11)
        if np.array_equal(references[0], references[1]):
            raise ValueError('Input update reference outputs are identical')
        np.testing.assert_allclose(meta['reference_max_variant_difference'], np.max(np.abs(references[0]-references[1])), rtol=1e-12)
        if set(meta['outputs']) != {'0_eager', '0_graph', '1_eager', '1_graph'}:
            raise ValueError('Missing output variants/backends')
        for variant in [0, 1]:
            for backend in ['eager', 'graph']:
                retained = meta['outputs'][f'{variant}_{backend}']
                path = f'outputs/{name}_{variant}_{backend}.npy'
                if retained['path'] != path or sha256(run/path) != retained['sha256']:
                    raise ValueError('Output hash/path mismatch')
                output = np.load(run/path, allow_pickle=False)
                if output.dtype != np.float32:
                    raise ValueError('Expected FP32 model output')
                output_info[(name, variant, backend)] = dict(hash=array_mapping_hash({'output': output}), errors=error_summary(output, references[variant]), output=output)
        for variant in [0, 1]:
            np.testing.assert_allclose(output_info[(name, variant, 'graph')]['output'], output_info[(name, variant, 'eager')]['output'], rtol=1e-6, atol=1e-7)
        if output_info[(name, 0, 'graph')]['hash'] == output_info[(name, 1, 'graph')]['hash']:
            raise ValueError('Updated graph output is stale')
        setup = meta['setup_seconds']
        fields = ['buffer_clone', 'side_stream_warmup', 'capture_instantiate', 'first_replay']
        if set(setup) != set(fields+['total']) or not all(np.isfinite(v) and v > 0 for v in setup.values()) or sum(setup[k] for k in fields) > setup['total']+1e-9:
            raise ValueError('Invalid graph setup timing')
        mem = meta['setup_memory']
        for metric in ['allocated', 'reserved']:
            before, after, extra = [int(mem[f'{k}_{metric}_bytes']) for k in ['before', 'after', 'extra']]
            if min(before, after) < 0 or extra != after-before or int(mem[f'peak_{metric}_bytes']) < max(before, after):
                raise ValueError('Setup memory arithmetic mismatch')
        if mem['peak_reserved_bytes'] < mem['peak_allocated_bytes']:
            raise ValueError('Invalid setup allocator peaks')
    for row in sample+traced+updates:
        backend = row['method'].split('_', 1)[0] if 'method' in row else 'graph'
        info = output_info[(row['case'], int(row['variant']), backend)]
        if row['output_sha256'] != info['hash']:
            raise ValueError('Measured output differs from retained variant/backend')
        for field, value in info['errors'].items():
            np.testing.assert_allclose(float(row[field]), value, rtol=1e-10, atol=1e-15)
        if 'step' in row:
            meta = env['case_metadata'][row['case']]
            if row['input_sha256'] != meta['input_hashes'][int(row['variant'])] or int(row['input_pointer_stable']) != 1 or int(row['output_pointer_stable']) != 1:
                raise ValueError('Graph input update or pointer persistence failed')
    for key in keys:
        selected = [r for r in sample if (r['case'], r['method']) == key]
        for field, value in stats(selected).items():
            np.testing.assert_allclose(float(summaries[key][field]), value, rtol=1e-12)
    if len(memory) != len(keys) or {(r['case'], r['method']) for r in memory} != keys:
        raise ValueError('Missing/duplicate runtime memory probes')
    for row in memory:
        before, peak, extra, reserved = [int(row[k]) for k in ['baseline_allocated_bytes', 'peak_allocated_bytes', 'peak_extra_allocated_bytes', 'peak_reserved_bytes']]
        if not 0 <= before <= peak <= reserved or extra != peak-before:
            raise ValueError('Runtime memory arithmetic mismatch')
    path = run/'trace.json.gz'
    if sha256(path) != env['trace_sha256']:
        raise ValueError('Trace hash mismatch')
    document = json.loads(gzip.decompress(path.read_bytes()))
    recomputed = summarize_trace(document)
    if len(recomputed) != len(traced):
        raise ValueError('Trace root count mismatch')
    for saved, actual in zip(traced, recomputed):
        for field, value in actual.items():
            if isinstance(value, str):
                if saved[field] != value:
                    raise ValueError('Trace annotation mismatch')
            else:
                np.testing.assert_allclose(float(saved[field]), value, rtol=1e-10, atol=1e-9)
        expected_bytes = env['case_metadata'][actual['case']]['input_bytes']
        is_graph = actual['method'].startswith('graph_')
        if actual['dtoh_copies'] != 1 or actual['dtoh_bytes'] != expected_bytes or actual['kernel_events'] < 1 or (is_graph and actual['graph_launches'] < 1) or (not is_graph and actual['graph_launches'] != 0):
            raise ValueError('Missing full output, kernels or graph launches')
        if actual['method'].endswith('_host_update'):
            if actual['htod_bytes'] != expected_bytes or actual['htod_copies'] < 1:
                raise ValueError('Host input not fully uploaded')
        elif actual['htod_bytes'] != 0:
            raise ValueError('Unexpected HtoD copy')
        if actual['method'].endswith('_device_update') and actual['dtod_bytes'] < expected_bytes:
            raise ValueError('Device input update missing')
    print(f'PASS {run.name}: {len(sample)} timings, {len(updates)} A/B/A input checks, {len(memory)} memory probes, {len(traced)} real graph/eager trace calls'+('; FP64 references recomputed' if recompute else ''))
    return dict(path=run, env=env, samples=sample, summary=summaries, memory=memory, trace=traced)


def paired(run, case, mode):
    values = {(r['method'], int(r['repeat'])): float(r['seconds']) for r in run['samples'] if r['case'] == case['case']}
    eager = np.array([values[('eager_'+mode, i)] for i in range(run['env']['repeats'])])
    graph = np.array([values[('graph_'+mode, i)] for i in range(run['env']['repeats'])])
    return eager, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recompute-reference', action='store_true')
    args = parser.parse_args()
    runs = [validate(p, args.data, args.baseline, args.recompute_reference) for p in args.runs]
    modes, labels = ['resident', 'device_update', 'host_update'], ['Resident input A', 'Update from GPU', 'Upload from CPU']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for ax, case in zip(axes[0], CASES):
        for i, backend in enumerate(['eager', 'graph']):
            values = np.array([[float(r['summary'][(case['case'], backend+'_'+mode)]['p50_sec'])*1000 for mode in modes] for r in runs])
            med = np.median(values, axis=0)
            bars = ax.bar(np.arange(3)+(i-.5)*.36, med, width=.34, label=backend.capitalize(),
                          yerr=[med-values.min(axis=0), values.max(axis=0)-med], capsize=4)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.set_xticks(range(3), labels)
        ax.set(ylabel='Primary P50 (ms)', title=f'B{case["batch"]}/64x64: same full host output')
        ax.legend(fontsize=8)
    for i, metric in enumerate(['allocated', 'reserved']):
        values = np.array([[r['env']['case_metadata'][c['case']]['setup_memory']['extra_'+metric+'_bytes']/2**20 for c in CASES] for r in runs])
        med = np.median(values, axis=0)
        bars = axes[1, 0].bar(np.arange(2)+(i-.5)*.36, med, width=.34, label='Net '+metric,
                              yerr=[med-values.min(axis=0), values.max(axis=0)-med], capsize=4)
        axes[1, 0].bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    axes[1, 0].set_xticks(range(2), ['B16 graph', 'B64 graph'])
    axes[1, 0].set(ylabel='PyTorch allocator net change (MiB)', title='Graph setup persistence; not total GPU memory')
    axes[1, 0].legend(fontsize=8)
    for i, case in enumerate(CASES):
        counts = []
        for mode in modes:
            estimates = []
            for run in runs:
                eager, graph = paired(run, case, mode)
                saving = float(np.mean(eager-graph))
                setup = run['env']['case_metadata'][case['case']]['setup_seconds']['total']
                estimates.append(setup/saving if saving > 0 else np.nan)
            counts.append(float(np.mean(estimates)) if np.isfinite(estimates).all() else np.nan)
        bars = axes[1, 1].bar(np.arange(3)+(i-.5)*.36, counts, width=.34, label=f'B{case["batch"]}')
        axes[1, 1].bar_label(bars, labels=[f'{math.ceil(v)}' if np.isfinite(v) else 'No estimate' for v in counts], fontsize=8, padding=3)
    axes[1, 1].set_xticks(range(3), labels)
    axes[1, 1].set(ylabel='Estimated calls to amortize setup', title='Setup / mean saving; not a measured N-call workload')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Phase 5B: fixed-shape CUDA Graph replay versus eager FNO\nGenuine A/B updates; FP32; no I/O/setup/checks in primary time; full output preserved; graph pools coexist', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_graph.png', dpi=160)
    plt.close(fig)
    text = ['# Phase 5B: CUDA Graph replay with genuine input updates', '',
            'The existing untrained FP32 FNO is retained with identical weights and two disjoint genuine ECMWF input windows per B16/B64 shape. No training, forecast assessment, AMP, compiler, JAX, Earth2Studio, driver or power-setting change is involved. One graph per shape is captured; eager remains available.', '',
            '## Primary latency and paired savings', '',
            '| Session | Batch | Boundary | Eager P50 ms | Graph P50 ms | Eager P95 ms | Graph P95 ms | Eager max ms | Graph max ms | Median paired eager/graph | Median paired saving ms |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for case in CASES:
            for mode in modes:
                eager, graph = paired(run, case, mode)
                a, b = [run['summary'][(case['case'], backend+'_'+mode)] for backend in ['eager', 'graph']]
                text.append(f'| {run["path"].name} | {case["batch"]} | {mode} | {float(a["p50_sec"])*1000:.3f} | {float(b["p50_sec"])*1000:.3f} | {float(a["p95_sec"])*1000:.3f} | {float(b["p95_sec"])*1000:.3f} | {float(a["max_sec"])*1000:.3f} | {float(b["max_sec"])*1000:.3f} | {np.median(eager/graph):.3f}x | {np.median(eager-graph)*1000:.3f} |')
    regressions = []
    for run in runs:
        for case in CASES:
            for mode in modes:
                a, b = [float(run['summary'][(case['case'], backend+'_'+mode)]['max_sec'])*1000 for backend in ['eager', 'graph']]
                if b > a:
                    regressions.append(f'{run["path"].name}/B{case["batch"]}/{mode}: {a:.3f} -> {b:.3f} ms')
    if regressions:
        text += ['', 'Observed maximum-latency regressions are retained: '+'; '.join(regressions)+'. Median improvements do not guarantee every invocation or tail statistic improves.']
    text += ['', 'Both cases/all six methods are shuffled per repetition; paired methods use the same A/B variant in that repetition. Resident always uses A and a shared untimed D2D reset/synchronization precondition for both paths, then measures only forward/full return. Device-update includes the same D2D input copy into the static buffer in both paths. Host-update includes the same pageable CPU-to-CUDA input copy in both paths. CPU packing/read and file writes are excluded. Primary timers contain no profiler, annotations or CUDA events; output checks/hashes occur outside timing. Bar heights are medians of session medians and whiskers span session medians, not confidence intervals. All samples/maxima are retained; short-session P95 values and mean-based amortization estimates are sensitive to outliers.', '',
             'A = first B native frames, B = next B frames. No tiling, synthetic perturbation or output caching. A/B/A tests copy actual new input, read back the actual static input, check stable input/output pointers, require changed graph output hashes and compare graph/eager tightly as well as against CPU FP64. Incorrect shape, dtype, noncontiguous layout or wrong CUDA device is rejected. Model weights and runner lifetime must remain fixed; borrowed GPU output storage is overwritten on replay.', '',
             '## Setup, persistence and estimated amortization', '',
             '| Session | Batch | Total setup ms | Capture/instantiate ms | First replay ms | Net allocated MiB | Net reserved MiB | Setup allocated peak MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for case in CASES:
            meta = run['env']['case_metadata'][case['case']]
            setup, memory = meta['setup_seconds'], meta['setup_memory']
            text.append(f'| {run["path"].name} | {case["batch"]} | {setup["total"]*1000:.3f} | {setup["capture_instantiate"]*1000:.3f} | {setup["first_replay"]*1000:.3f} | {memory["extra_allocated_bytes"]/2**20:.3f} | {memory["extra_reserved_bytes"]/2**20:.3f} | {memory["peak_allocated_bytes"]/2**20:.3f} |')
    text += ['', '| Session | Batch | Boundary | Mean saving ms | Estimated calls to recover graph setup |', '|---|---:|---|---:|---:|']
    for run in runs:
        for case in CASES:
            setup = run['env']['case_metadata'][case['case']]['setup_seconds']['total']
            for mode in modes:
                eager, graph = paired(run, case, mode)
                saving = float(np.mean(eager-graph))
                estimate = str(math.ceil(setup/saving)) if saving > 0 else 'No positive mean saving'
                text.append(f'| {run["path"].name} | {case["batch"]} | {mode} | {saving*1000:.3f} | {estimate} |')
    text += ['', 'Setup includes input-buffer clone, side-stream eager warmup, capture/instantiation and the first synchronized replay after an earlier eager warmup. It excludes common model/source/reference setup and validation. Costs are single observations per case/session in an initialized process. Estimated break-even uses setup divided by mean request saving; the plot averages finite per-session estimates then rounds up. This is not a measured multi-request application or cold-start guarantee.', '',
             'Graph inputs/outputs/weights remain alive and each graph owns its default private pool. Allocator snapshots are taken before/after creation; all source buffers and earlier-shape graphs coexist. Net allocated/reserved increases and setup peaks are not total VRAM, graph executable/driver memory or standalone per-case requirements. Runtime memory probes are separate from latency and eager is measured while graph pools remain resident. See memory_samples.csv for baseline/peak/extra/reserved.', '',
             '## Separate CPU/CUDA trace evidence', '',
             '| Batch/boundary/backend | Profiled wall ms | CPU update ms | CPU dispatch ms | CPU return ms | Kernel events | CPU kernel-launch APIs | CPU graph-launch APIs | DtoH bytes | HtoD bytes | DtoD event bytes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        for method in METHODS:
            selected = [row for run in runs for row in run['trace'] if row['case'] == case['case'] and row['method'] == method]
            fields = ['root_wall_ms', 'update_ms', 'dispatch_ms', 'return_ms', 'kernel_events', 'kernel_launches', 'graph_launches', 'dtoh_bytes', 'htod_bytes', 'dtod_bytes']
            text.append(f'| B{case["batch"]}/{method} | '+' | '.join(f'{np.mean([float(r[k]) for r in selected]):.4f}' for k in fields)+' |')
    text += ['', 'Each profiled graph request must contain a real cudaGraphLaunch and kernels; eager has no graph-launch API. Every call returns exactly one full-output DtoH. Host-update must upload all input bytes; device-update includes at least those bytes in DtoD copy events. Resident resets occur outside the root range. Dispatch/return ranges overlap GPU work and cannot be added to kernel/copy intervals to explain wall time.', '',
             'Capture can change internal implementation as well as host launch overhead. In the inspected trace, graph execution contains memcpy128 device kernels where eager reports internal DtoD copy events. Thus fewer DtoD event bytes do not mean device copies vanished, and 102 versus 99 kernel events do not establish more/less mathematical work. Kernel/copy category sums must not be used as a device bandwidth or occupancy claim. The CPU-only analyzer reconstructs all actual trace summaries from the hashed compressed Chrome trace.', '',
             'Trace timing is a separate later experiment with profiler overhead and altered scheduling; primary results determine speed. Neither dispatch-range differences nor graph-launch counts isolate all causes of latency improvement. Full read/prepare/write application gains can be much smaller than these forward/transfer gains.', '',
             '## Reproduce and validate', '', '```bash', 'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_graph.py --output results/my_physics_graph01',
             'python physicsnemo_graph.py --seed 20261110 --output results/my_physics_graph02',
             'python scripts/analyze_physicsnemo_graph.py results/my_physics_graph01 results/my_physics_graph02 --recompute-reference --output results/my_physics_graph_report', '```', '',
             'Use new output paths. Without --recompute-reference, existing CPU data/plot dependencies suffice to validate the saved inputs/references/outputs, state/provenance hashes, A/B/A pointer/update evidence, seeded pairing, quantiles, allocator arithmetic and real trace launches/copy bytes. The optional flag independently rebuilds both CPU FP64 references from the archived exact FP32 state; numerical tolerance remains rtol=1e-4, atol=1e-5.', '',
             '[Official PyTorch CUDA Graph usage and constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).', '',
             '![CUDA Graph latency, setup persistence and estimated amortization](physicsnemo_graph.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated CUDA Graph report:', args.output)


if __name__ == '__main__':
    main()
