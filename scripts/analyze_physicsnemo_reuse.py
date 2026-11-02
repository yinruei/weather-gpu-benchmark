"""Validate paired workflow reuse and cadence evidence using CPU dependencies."""
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
from physicsnemo_benchmark import MODEL_CONFIG, TOLERANCE, array_mapping_hash, error_summary, prepare_input, read_input, stats
from physicsnemo_pipeline import output_payload, verify_output
from physicsnemo_reuse import CONDITIONS, WORKFLOWS, job_order
from weather_verify import sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data, baseline):
    env = json.loads((run/'environment.json').read_text())
    base = json.loads((baseline/'environment.json').read_text())
    source = sha256(data)
    manifest = json.loads((data.parent/'manifest.json').read_text())
    if env['source_sha256'] != source or base['source_sha256'] != source or source != manifest['files']['batch.zarr'] or env['source_manifest'] != manifest:
        raise ValueError('Source/provenance mismatch')
    if env['model_config'] != MODEL_CONFIG or env['tolerance'] != TOLERANCE or env['workflows'] != WORKFLOWS or env['conditions'] != CONDITIONS or env['tf32'] or not env['deterministic_algorithms']:
        raise ValueError('Protocol mismatch')
    if env['weights_sha256'] != base['weights_sha256'] or sha256(run/'weights.npz') != base['weights_sha256'] or env['model_state_sha256'] != base['model_state_sha256']:
        raise ValueError('Weights mismatch')
    with np.load(run/'weights.npz', allow_pickle=False) as saved:
        if array_mapping_hash({k: saved[k] for k in saved.files}) != env['model_state_sha256']:
            raise ValueError('Tensor state mismatch')
    if len({c['case'] for c in env['cases']}) != len(env['cases']) or any(c not in base['cases'] for c in env['cases']):
        raise ValueError('Duplicate/unsupported cases')
    raw, summary, cadence = [rows(run/f) for f in ['workflow_samples.csv', 'workflow_summary.csv', 'cadence_samples.csv']]
    summaries = {(r['case'], r['method']): r for r in summary}
    keys = {(c['case'], m) for c in env['cases'] for m in WORKFLOWS}
    if set(summaries) != keys or len(summary) != len(summaries):
        raise ValueError('Missing/duplicate summaries')
    actual = [(int(r['repeat']), int(r['order']), r['case'], r['method']) for r in raw]
    if actual != job_order(env['cases'], WORKFLOWS, env['repeats'], env['seed']):
        raise ValueError('Workflow pairing/order mismatch')
    actual = [(int(r['block']), int(r['order']), r['case'], r['condition'], int(r['position'])) for r in cadence]
    expected = [(block, order, case, condition, position) for block, order, case, condition in job_order(env['cases'], CONDITIONS, env['blocks'], env['seed']+1) for position in range(env['block_repeats'])]
    if actual != expected:
        raise ValueError('Cadence block pairing/order mismatch')
    for case in env['cases']:
        name = case['case']
        meta = env['case_metadata'][name]
        values, identity, coordinates = read_input(data, case)
        x = prepare_input(values)
        if identity != meta['identity'] or meta['input_sha256'] != array_mapping_hash({'input': x}) or meta['cpu_input_bytes'] != x.nbytes or meta['cuda_input_bytes'] != x.nbytes:
            raise ValueError('Cache input identity/bytes mismatch')
        setup = meta['setup_seconds']
        if set(setup) != {'reopen', 'open_once', 'host_cache', 'device_cache'} or setup['reopen'] != 0 or not all(np.isfinite(v) and v > 0 for k, v in setup.items() if k != 'reopen') or setup['device_cache'] < setup['host_cache']:
            raise ValueError('Invalid setup costs')
        artifact = baseline/(name+'.npz')
        if sha256(artifact) != base['case_metadata'][name]['artifact_sha256'] or meta['baseline_artifact_sha256'] != base['case_metadata'][name]['artifact_sha256']:
            raise ValueError('Reference hash mismatch')
        with np.load(artifact, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['input_fp32'], x)
            reference = saved['reference_fp64'].copy()
        if reference.dtype != np.float64 or reference.shape != x.shape:
            raise ValueError('Reference dtype/shape mismatch')
        file_sizes = set()
        for method in WORKFLOWS:
            retained = meta['outputs'][method]
            expected_path = 'outputs/'+name+'_'+method+'.npz'
            path = run/expected_path
            if retained['path'] != expected_path or sha256(path) != retained['sha256'] or path.stat().st_size != retained['bytes']:
                raise ValueError('Workflow file hash/path/bytes mismatch')
            with np.load(path, allow_pickle=False) as saved:
                output = saved['model_output'].copy()
            if output.dtype != np.float32:
                raise ValueError('Expected FP32 output')
            payload = output_payload(output, identity, coordinates, case, source, env['model_state_sha256'])
            errors = verify_output(path, payload, reference)
            logical_hash = array_mapping_hash(payload)
            if retained['payload_sha256'] != logical_hash:
                raise ValueError('Payload hash mismatch')
            selected = [r for r in raw if r['case'] == name and r['method'] == method]
            for row in selected:
                if int(row['file_bytes']) != retained['bytes'] or row['file_sha256'] != retained['sha256'] or row['payload_sha256'] != logical_hash:
                    raise ValueError('Trial differs from retained output')
                for field, expected in errors.items():
                    np.testing.assert_allclose(float(row[field]), expected, rtol=1e-10, atol=1e-15)
            for field, expected in stats(selected).items():
                np.testing.assert_allclose(float(summaries[(name, method)][field]), expected, rtol=1e-12)
            np.testing.assert_allclose(float(summaries[(name, method)]['frames_per_second']), case['batch']/float(summaries[(name, method)]['p50_sec']), rtol=1e-12)
            file_sizes.add(retained['bytes'])
        if len(file_sizes) != 1:
            raise ValueError('Workflow output sizes differ')
        retained = meta['cadence_output']
        expected_path = 'outputs/'+name+'_cadence.npy'
        if retained['path'] != expected_path or sha256(run/expected_path) != retained['sha256']:
            raise ValueError('Cadence output hash/path mismatch')
        output = np.load(run/expected_path, allow_pickle=False)
        if output.dtype != np.float32:
            raise ValueError('Expected FP32 cadence output')
        errors = error_summary(output, reference)
        digest = array_mapping_hash({'output': output})
        for row in [r for r in cadence if r['case'] == name]:
            values = [float(row[k]) for k in ['seconds', 'event_seconds', 'pre_seconds']]
            if not np.isfinite(values).all() or min(values[:2]) <= 0 or values[2] < 0 or row['output_sha256'] != digest:
                raise ValueError('Invalid cadence timing/output')
            if (row['condition'] == 'continuous' and values[2] != 0) or (row['condition'] != 'continuous' and values[2] <= 0):
                raise ValueError('Unexpected precondition time')
            for field, expected in errors.items():
                np.testing.assert_allclose(float(row[field]), expected, rtol=1e-10, atol=1e-15)
    print(f'PASS {run.name}: {len(raw)} paired workflow files and {len(cadence)} event/wall cadence records reconciled')
    return dict(path=run, env=env, raw=raw, summary=summaries, cadence=cadence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(p, args.data, args.baseline) for p in args.runs]
    cases = runs[0]['env']['cases']
    if any(r['env']['cases'] != cases or r['env']['gap_ms'] != runs[0]['env']['gap_ms'] for r in runs):
        raise ValueError('Cases and idle gap must match')
    gap_ms = runs[0]['env']['gap_ms']
    labels = ['Reopen', 'Open once', 'CPU input cache', 'GPU input cache']
    modes = ['reopen', 'open_once', 'host_cache', 'device_cache']
    colors = ['#4c78a8', '#f58518', '#54a24b', '#b279a2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for ax, case in zip(axes[0], cases[:2]):
        for i, backend in enumerate(['cpu', 'cuda']):
            vals = []
            for mode in modes:
                method = backend+'_'+mode
                vals.append(np.median([float(r['summary'][(case['case'], method)]['p50_sec'])*1000 for r in runs]) if method in WORKFLOWS else np.nan)
            bars = ax.bar(np.arange(4)+(i-.5)*.36, vals, width=.34, label=backend.upper())
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set_xticks(range(4), labels, rotation=12)
        ax.set(ylabel='Request P50 (ms)', title=f'B{case["batch"]}/{case["side"]}x{case["side"]}: full host output + file close')
        ax.legend(fontsize=8)
    for i, case in enumerate(cases):
        wall, event = [], []
        for condition in CONDITIONS:
            block_medians = []
            for run in runs:
                for block in range(run['env']['blocks']):
                    selected = [row for row in run['cadence'] if row['case'] == case['case'] and row['condition'] == condition and int(row['block']) == block]
                    block_medians.append([np.median([float(row[field])*1000 for row in selected]) for field in ['seconds', 'event_seconds']])
            wall.append(np.median([v[0] for v in block_medians]))
            event.append(np.median([v[1] for v in block_medians]))
        axes[1, 0].plot(range(4), wall, 'o-', label=f'B{case["batch"]} host wall + return')
        axes[1, 0].plot(range(4), event, 'o--', label=f'B{case["batch"]} stream event interval')
    axes[1, 0].set_xticks(range(4), ['Continuous', f'{gap_ms:g}ms idle', 'Read/pack first', 'CPU FNO first'], rotation=12)
    axes[1, 0].set(ylabel='Instrumented GPU forward latency (ms)', title='Resident input; precondition time excluded; no files')
    axes[1, 0].legend(fontsize=8)
    largest = cases[-1]
    for mode, label, color in zip(modes, labels, colors):
        values = []
        for n in [1, 4, 16, 64]:
            means = []
            for run in runs:
                selected = [float(r['seconds']) for r in run['raw'] if r['case'] == largest['case'] and r['method'] == 'cuda_'+mode]
                setup = run['env']['case_metadata'][largest['case']]['setup_seconds'][mode]
                means.append((setup/n+np.mean(selected))*1000)
            values.append(np.mean(means))
        axes[1, 1].plot(range(4), values, 'o-', label=label, color=color)
    axes[1, 1].set_xticks(range(4), ['1', '4', '16', '64'])
    axes[1, 1].set(xlabel='Number of uses of the identical input', ylabel='Estimated mean cost / request (ms)', title=f'B{largest["batch"]}: measured setup/N + request mean')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Why GPU speedup shrinks: paired I/O reuse and separate cadence controls\nFixed untrained FNO; same frames/weights/output; cache setup excluded from top panels and included in amortization estimates', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_reuse.png', dpi=160)
    plt.close(fig)
    text = ['# Phase 5B: paired reuse and GPU cadence controls', '',
            'Fixed untrained FNO, identical ECMWF input frames/weights, FP32, full host outputs and identical uncompressed NPZ files. Models remain resident. No forecast assessment, model training, driver/power-setting changes, JAX or Earth2Studio installation.', '',
            '## Paired workflow requests', '',
            '| Session | Batch/grid | Workflow | P50 ms | P95 ms | Setup ms, outside requests |',
            '|---|---|---|---:|---:|---:|']
    for run in runs:
        for case in cases:
            for method in WORKFLOWS:
                mode = method.split('_', 1)[1]
                row = run['summary'][(case['case'], method)]
                setup = run['env']['case_metadata'][case['case']]['setup_seconds'][mode]
                text.append(f'| {run["path"].name} | {case["batch"]}/{case["side"]}x{case["side"]} | {method} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {setup*1000:.3f} |')
    text += ['', '## Within-repetition reuse contrasts', '',
             '| Session | Batch | Backend | Reuse mode | Median paired reopen-minus-mode ms | Median paired reopen/mode |',
             '|---|---:|---|---|---:|---:|']
    for run in runs:
        for case in cases:
            samples = {(r['method'], int(r['repeat'])): float(r['seconds']) for r in run['raw'] if r['case'] == case['case']}
            for method in WORKFLOWS:
                backend, mode = method.split('_', 1)
                if mode == 'reopen':
                    continue
                baseline_values = np.array([samples[(backend+'_reopen', i)] for i in range(run['env']['repeats'])])
                reused_values = np.array([samples[(method, i)] for i in range(run['env']['repeats'])])
                text.append(f'| {run["path"].name} | {case["batch"]} | {backend} | {mode} | {np.median(baseline_values-reused_values)*1000:.3f} | {np.median(baseline_values/reused_values):.2f}x |')
    text += ['', 'Both cases and all workflows are shuffled within each repetition. The paired contrasts match observations by repetition, not by identical clock/power state; they are descriptive, not significance tests. Compare modes/backends within these sessions; earlier experiments used different case/call schedules. Reopen includes every open/read/decode/check/pack/upload/forward/return/write-close. Open-once removes repeated dataset opening but still selects, loads, validates and packs each call; underlying backend/OS caches are uncontrolled. Host-cache removes read/pack and reuses prepared CPU tensor and metadata. Device-cache also removes H2D input copy. All modes keep the same full output and file-close boundary; no fsync.', '',
             'Host/device input caches apply only when the exact same source frames, transform and input are reused; changed data require rebuilding them. They are not read-to-file speedups for arbitrary new inputs. Open handles can serve other valid selections; regression tests cover changing selections. All cases/mode caches coexist here and add memory outside timed requests. Cache setup is observed once per session/case and reported explicitly. Model startup is common and excluded.', '',
             '## Cadence: same GPU-resident input, no writes', '',
             '| Session | Batch | Condition | Precondition P50 ms | Forward+return wall P50 ms | Stream-event P50 ms |',
             '|---|---:|---|---:|---:|---:|']
    for run in runs:
        for case in cases:
            for condition in CONDITIONS:
                selected = [r for r in run['cadence'] if r['case'] == case['case'] and r['condition'] == condition]
                values = [np.median([float(r[k])*1000 for r in selected]) for k in ['pre_seconds', 'seconds', 'event_seconds']]
                text.append(f'| {run["path"].name} | {case["batch"]} | {condition} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    text += ['', f'Cadence runs occur after workflow requests. Case/condition blocks are shuffled; inner calls are consecutive following condition-specific warmups. Continuous has no intentional host work between calls; idle-gap sleeps {gap_ms:g} ms; read-gap opens/reads/packs and checks source input; CPU-gap executes the same CPU FNO. These preconditions are outside the reported GPU forward wall time but included in the precondition column. Their durations and host workloads are not exactly matched, so contrasts do not identify one hardware cause.', '',
             'Wall time includes event records, host dispatch, GPU forward and blocking full host return. CUDA event interval brackets the forward on the stream; it can include host dispatch gaps and GPU scheduling delays and is not the sum of kernel-active times. No utilization, occupancy, Tensor Core use, driver scheduling or GPU clock cause is established by these intervals alone. Telemetry is a coarse before/after snapshot. Output checking is deferred until the end of each measured block to reduce inter-call verification gaps.', '',
             '## Amortization and validation limits', '',
             'Bottom-right plot is an estimate `(one-time cache setup + N * measured request mean)/N`, not a separately timed N-request application. It excludes common model startup, assumes an unchanged input, uses warm repeated-request measurements, and does not establish a cold single-request speedup. Top workflow bars are medians across session medians. Cadence curves take the median of per-block medians; the cadence table reports pooled per-session sample medians. Inner observations in a block are related and are not independent experimental replications. All per-session values and raw pairing/block order are retained above/in CSV.', '',
             'Every workflow invocation writes/readbacks/checks full arrays, dtypes, coordinates, frame identities, payload hashes and elementwise CPU FP64 error outside timing. Each session independently recomputes the archived reference from exact FP32 state/inputs cast to FP64. Cadence outputs and CPU-gap computations are checked against the same reference. Last workflow files plus one actual cadence array per case are retained; all trial file/output hashes are reconciled by the CPU analyzer. Numerical tolerance is rtol=1e-4, atol=1e-5, describing arithmetic consistency only.', '',
             'No OS cache clearing, async prefetch/overlap, compressed output or remote retrieval is measured. Source hash/reference setup and repeated file validation warm caches. Results describe this small architecture/workload/machine, not all GPU models.', '',
             '## Reproduce', '', '```bash', 'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_reuse.py --output results/my_physics_reuse01',
             'python physicsnemo_reuse.py --seed 20261106 --output results/my_physics_reuse02',
             'python scripts/analyze_physicsnemo_reuse.py results/my_physics_reuse01 results/my_physics_reuse02 --output results/my_physics_reuse_report', '```', '',
             'Use new output paths. CPU-only dependencies can regenerate the report and validate archived outputs; running experiments requires the existing isolated CUDA environment.', '',
             '![Paired input reuse and GPU cadence controls](physicsnemo_reuse.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated reuse/cadence report:', args.output)


if __name__ == '__main__':
    main()
