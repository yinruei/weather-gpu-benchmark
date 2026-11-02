"""Recompute paired schedule statistics and real CPU/CUDA trace accounting offline."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import MODEL_CONFIG, TOLERANCE, array_mapping_hash, error_summary, load_input, stats
from physicsnemo_schedule import CASES, SCHEDULES, execution_plan, sequence_labels, summarize_trace
from weather_verify import sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def expected_calls(plan):
    return [dict(round=b['round'], block_order=b['block_order'], schedule=b['schedule'], **label)
            for b in plan for label in sequence_labels(b['sequence'])]


def validate(run, data, baseline):
    env = json.loads((run/'environment.json').read_text())
    base = json.loads((baseline/'environment.json').read_text())
    source_hash = sha256(data)
    manifest = json.loads((data.parent/'manifest.json').read_text())
    if env['source_sha256'] != source_hash or base['source_sha256'] != source_hash or source_hash != manifest['files']['batch.zarr'] or env['source_manifest'] != manifest:
        raise ValueError('Source/provenance mismatch')
    if env['cases'] != CASES or env['schedules'] != SCHEDULES or env['model_config'] != MODEL_CONFIG or env['tolerance'] != TOLERANCE or env['tf32'] or not env['deterministic_algorithms']:
        raise ValueError('Protocol mismatch')
    if env['weights_sha256'] != base['weights_sha256'] or sha256(run/'weights.npz') != base['weights_sha256'] or env['model_state_sha256'] != base['model_state_sha256']:
        raise ValueError('Weight state mismatch')
    with np.load(run/'weights.npz', allow_pickle=False) as saved:
        if array_mapping_hash({k: saved[k] for k in saved.files}) != env['model_state_sha256']:
            raise ValueError('Tensor state mismatch')
    samples, summary, blocks, traced = [rows(run/f) for f in ['timing_samples.csv', 'timing_summary.csv', 'block_samples.csv', 'trace_calls.csv']]
    primary_plan = execution_plan(env['rounds'], env['calls_per_case'], env['seed'])
    trace_plan = execution_plan(env['trace_rounds'], env['trace_calls_per_case'], env['seed']+1000)
    def check_order(records, expected):
        if len(records) != len(expected):
            raise ValueError('Missing/extra call records')
        for row, wanted in zip(records, expected):
            if any((int(row[k]) if isinstance(v, int) else row[k]) != v for k, v in wanted.items()):
                raise ValueError('Schedule/pair/order/shape-transition mismatch')
    check_order(samples, expected_calls(primary_plan))
    check_order(traced, expected_calls(trace_plan))
    check_order(blocks, [{k: v for k, v in b.items() if k != 'sequence'} for b in primary_plan])
    summaries = {(r['schedule'], r['case']): r for r in summary}
    keys = {(s, c['case']) for s in SCHEDULES for c in CASES}
    if set(summaries) != keys or len(summary) != len(keys):
        raise ValueError('Missing/duplicate summaries')
    for case in CASES:
        name = case['case']
        meta = env['case_metadata'][name]
        x, identity = load_input(data, case)
        if meta['identity'] != identity or meta['input_sha256'] != array_mapping_hash({'input': x}) or meta['input_bytes'] != x.nbytes or meta['output_bytes'] != x.nbytes:
            raise ValueError('Input frames, numerical transform or bytes changed')
        artifact = baseline/(name+'.npz')
        if meta['baseline_artifact_sha256'] != base['case_metadata'][name]['artifact_sha256'] or sha256(artifact) != meta['baseline_artifact_sha256']:
            raise ValueError('Reference archive mismatch')
        with np.load(artifact, allow_pickle=False) as saved:
            np.testing.assert_array_equal(x, saved['input_fp32'])
            truth = saved['reference_fp64'].copy()
        path = 'outputs/'+name+'.npy'
        if meta['output_path'] != path or sha256(run/path) != meta['output_sha256']:
            raise ValueError('Saved output hash/path mismatch')
        output = np.load(run/path, allow_pickle=False)
        if output.dtype != np.float32 or output.shape != x.shape or truth.dtype != np.float64:
            raise ValueError('Output/reference dtype/shape mismatch')
        errors = error_summary(output, truth)
        digest = array_mapping_hash({'output': output})
        for row in [r for r in samples+traced if r['case'] == name]:
            if row['output_sha256'] != digest:
                raise ValueError('Trial output differs from retained output')
            for field, value in errors.items():
                np.testing.assert_allclose(float(row[field]), value, rtol=1e-10, atol=1e-15)
    for key in keys:
        selected = [r for r in samples if (r['schedule'], r['case']) == key]
        for field, value in stats(selected).items():
            np.testing.assert_allclose(float(summaries[key][field]), value, rtol=1e-12)
    for block in blocks:
        values = [float(r['seconds']) for r in samples if int(r['round']) == int(block['round']) and r['schedule'] == block['schedule']]
        seconds, total, residual = [float(block[k]) for k in ['seconds', 'sum_calls_sec', 'residual_sec']]
        if not np.isfinite([seconds, total, residual]).all() or seconds <= 0 or residual < 0:
            raise ValueError('Invalid block durations')
        np.testing.assert_allclose(sum(values), total, rtol=1e-12)
        np.testing.assert_allclose(seconds, total+residual, rtol=1e-12)
    trace_path = run/'trace.json.gz'
    if sha256(trace_path) != env['trace_sha256']:
        raise ValueError('Compressed trace hash mismatch')
    recomputed = summarize_trace(json.loads(gzip.decompress(trace_path.read_bytes())))
    if len(recomputed) != len(traced):
        raise ValueError('Trace call count mismatch')
    for row, observed in zip(traced, recomputed):
        for field, expected in observed.items():
            if isinstance(expected, str):
                if row[field] != expected:
                    raise ValueError('Trace annotation differs')
            else:
                np.testing.assert_allclose(float(row[field]), expected, rtol=1e-10, atol=1e-9)
        if observed['kernel_events'] < 1 or observed['dtoh_copies'] != 1 or observed['dtoh_bytes'] != env['case_metadata'][row['case']]['output_bytes']:
            raise ValueError('Missing real kernels or complete output DtoH')
        durations = [float(row[k]) for k in ['root_wall_ms', 'dispatch_ms', 'return_ms', 'kernel_union_ms', 'copy_union_ms', 'active_union_ms', 'runtime_launch_ms', 'runtime_wait_ms']]
        if not np.isfinite(durations).all() or min(durations) < 0 or durations[0] <= 0 or float(row['active_union_ms']) > durations[0]+1e-6:
            raise ValueError('Invalid trace interval durations')
    print(f'PASS {run.name}: {len(samples)} primary calls, {len(blocks)} equal-work blocks, {len(traced)} real CUDA trace calls reconciled')
    return dict(path=run, env=env, samples=samples, summary=summaries, blocks=blocks, trace=traced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(p, args.data, args.baseline) for p in args.runs]
    if any(r['env']['rounds'] != runs[0]['env']['rounds'] or r['env']['calls_per_case'] != runs[0]['env']['calls_per_case'] for r in runs):
        raise ValueError('Primary rounds/call counts must match')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for i, schedule in enumerate(SCHEDULES):
        values = np.array([[float(r['summary'][(schedule, c['case'])]['p50_sec'])*1000 for c in CASES] for r in runs])
        median = np.median(values, axis=0)
        bars = axes[0, 0].bar(np.arange(2)+(i-.5)*.36, median, width=.34, label=schedule.capitalize(),
                              yerr=[median-values.min(axis=0), values.max(axis=0)-median], capsize=4)
        axes[0, 0].bar_label(bars, fmt='%.2f', padding=4, fontsize=8)
    axes[0, 0].set_xticks(range(2), ['B16, 64x64', 'B64, 64x64'])
    axes[0, 0].set(ylabel='Primary per-call P50 (ms)', title='No profiler/events; full host output, no I/O')
    axes[0, 0].legend(fontsize=8)
    for index, run in enumerate(runs):
        values = {(r['schedule'], int(r['round'])): float(r['seconds']) for r in run['blocks']}
        ratios = [values[('alternating', i)]/values[('grouped', i)] for i in range(run['env']['rounds'])]
        axes[0, 1].plot(range(1, len(ratios)+1), ratios, 'o-', alpha=.8, label=f'Run {index+1}')
    axes[0, 1].axhline(1, color='black', linestyle=':', linewidth=1)
    axes[0, 1].set(xlabel='Paired round', ylabel='Alternating block / grouped block wall time', title='Equal counts of both shapes; >1 favors grouped')
    axes[0, 1].legend(fontsize=8)
    selections, labels = [], []
    for case in CASES:
        for schedule in SCHEDULES:
            selections.append([row for r in runs for row in r['trace'] if row['case'] == case['case'] and row['schedule'] == schedule])
            labels.append(f'B{case["batch"]}\n{schedule}')
    bottoms = np.zeros(4)
    for field, label, color in [('dispatch_ms', 'CPU dispatch (inclusive)', '#4c78a8'), ('return_ms', 'CPU return (inclusive)', '#f58518')]:
        values = [np.mean([float(r[field]) for r in selected]) for selected in selections]
        axes[1, 0].bar(range(4), values, bottom=bottoms, label=label, color=color)
        bottoms += values
    roots = np.array([np.mean([float(r['root_wall_ms']) for r in selected]) for selected in selections])
    axes[1, 0].bar(range(4), np.maximum(roots-bottoms, 0), bottom=bottoms, label='Other CPU range', color='#aaaaaa')
    axes[1, 0].plot(range(4), [np.mean([float(r['active_union_ms']) for r in selected]) for selected in selections], 'kD', label='GPU kernel/copy union (overlaps CPU)')
    axes[1, 0].set_xticks(range(4), labels)
    axes[1, 0].set(ylabel='Independent profiler mean (ms)', title='CPU ranges and GPU activity overlap; do not add')
    axes[1, 0].legend(fontsize=7)
    for case, color in zip(CASES, ['#3366aa', '#dd7722']):
        for schedule, marker in [('grouped', 'o'), ('alternating', '^')]:
            selected = [row for r in runs for row in r['trace'] if row['case'] == case['case'] and row['schedule'] == schedule]
            axes[1, 1].scatter([float(r['root_wall_ms']) for r in selected], [float(r['active_union_ms']) for r in selected],
                               label=f'B{case["batch"]} {schedule}', color=color, marker=marker, alpha=.6, s=25)
    maximum = max(float(row['root_wall_ms']) for r in runs for row in r['trace'])
    axes[1, 1].plot([0, maximum], [0, maximum], ':', color='gray', linewidth=1)
    axes[1, 1].set(xlabel='Profiled call wall time (ms)', ylabel='GPU kernel/copy interval union (ms)', title='Separate trace diagnostics; not primary speed')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Phase 5B: isolate B16/B64 scheduling on the same 64x64 FNO inputs\nFixed untrained FP32 model; identical work/full return; top = profiler-free, bottom = separate CPU/CUDA traces', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_schedule.png', dpi=160)
    plt.close(fig)
    text = ['# Phase 5B: grouped versus alternating FNO batch shapes', '',
            'The same fixed untrained FNO, weights and ECMWF input frames are retained. Only request order changes: B16 and B64 both have the same 64x64 spatial grid. GPU inputs/model stay resident; every call recomputes and returns the full output. No reads, packing, input upload, file writes, training or forecast assessment are in these timings.', '',
            '## Profiler-free primary results', '',
            '| Session | Schedule | Case | Samples | P50 ms | P95 ms | Maximum ms |',
            '|---|---|---|---:|---:|---:|---:|']
    for run in runs:
        for schedule in SCHEDULES:
            for case in CASES:
                row = run['summary'][(schedule, case['case'])]
                text.append(f'| {run["path"].name} | {schedule} | B{case["batch"]}/64x64 | {row["samples"]} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    text += ['', '## Paired equal-work block comparison', '',
             '| Session | Grouped block P50 ms | Alternating block P50 ms | Median paired alternating/grouped | Paired rounds where grouped is faster |',
             '|---|---:|---:|---:|---:|']
    paired_medians = []
    for run in runs:
        by_key = {(r['schedule'], int(r['round'])): float(r['seconds']) for r in run['blocks']}
        a = np.array([by_key[('grouped', i)] for i in range(run['env']['rounds'])])
        b = np.array([by_key[('alternating', i)] for i in range(run['env']['rounds'])])
        paired_medians.append(float(np.median(b/a)))
        text.append(f'| {run["path"].name} | {np.median(a)*1000:.3f} | {np.median(b)*1000:.3f} | {np.median(b/a):.3f}x | {int(np.sum(b>a))}/{len(a)} |')
    if min(paired_medians) < 1 < max(paired_medians):
        text += ['', '**Observed result:** session median paired block ratios lie on both sides of 1. These sessions do not show a consistent total-work speedup from grouping the two batch shapes. This does not rule out effects in other workloads or identify the cause of earlier pipeline outliers.']
    text += ['', 'A paired round uses the same first shape and equal numbers of B16/B64 calls in both schedules. All four first-shape/first-schedule combinations balance across each full set of four rounds. Every block starts with the same alternating warmup ending in B64. Block wall time is measured directly; per-call sum plus recorded loop residual reconciles to it. Per-call/inner observations are related, not independent statistical replications. Median ratios and the direction count are descriptive, not confidence/significance tests. All slower calls and maxima remain visible. Top-left bars are medians of session medians; whiskers span session medians, not confidence intervals.', '',
             'Output checks/hashes occur after a whole measured block, so validation does not insert unequal per-call gaps. Primary calls contain no profiler, record_function ranges or CUDA events. The block retains at most one block of host outputs plus representatives, and CPU allocation/dispatch/return behavior remains part of the measured request. Changing order changes those behaviors too; this is not a pure kernel-only experiment.', '',
             '## Real CPU/CUDA trace observations', '',
             '| Case/schedule | Mean profiled wall ms | CPU dispatch ms | CPU return ms | Kernel union ms | Copy union ms | Launch runtime ms | Explicit sync API ms | Kernels/call | DtoH bytes/call | HtoD bytes/call |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        for schedule in SCHEDULES:
            selected = [row for r in runs for row in r['trace'] if row['case'] == case['case'] and row['schedule'] == schedule]
            fields = ['root_wall_ms', 'dispatch_ms', 'return_ms', 'kernel_union_ms', 'copy_union_ms', 'runtime_launch_ms', 'runtime_wait_ms', 'kernel_events', 'dtoh_bytes', 'htod_bytes']
            text.append(f'| B{case["batch"]}/{schedule} | '+' | '.join(f'{np.mean([float(row[k]) for row in selected]):.4f}' for k in fields)+' |')
    text += ['', 'Each trace call must contain real CUDA kernels and exactly one DtoH copy of the full output (262,144 bytes for B16; 1,048,576 for B64). Trace calls are annotated by schedule/round/position/case; the CPU analyzer decompresses the hashed Chrome trace and independently reconstructs every summary, byte count and ordering label.', '',
             'Dispatch/return ranges are inclusive CPU durations and may contain waiting. GPU kernel/copy intervals overlap them; union accounting avoids counting concurrent device activity twice. Return time is not physical copy time alone. CUDA runtime launch and synchronization durations are CPU-side observations, not device execution time. The recorded runtime_wait_ms field counts only API names containing Synchronize; other runtime calls, including copies to pageable host memory, can also wait, so it is not total waiting time. GPU active intervals and wall-minus-active are not occupancy or attribution to one driver/power cause.', '',
             'Traces run in a separate later profiling session with condition warmups, fewer calls and a different plan seed. Instrumentation, CPU allocation and system state can affect trace timing; use primary results for speed. No GPU clock/power/driver, precision, model or compiler setting changed. An observed association with batch-order switching does not prove a specific cuFFT/allocator/driver mechanism. A lack of repeatable improvement does not rule out all other workloads.', '',
             '## Evidence and reproduction', '',
             'Every warmup, primary and profiled output is checked elementwise against the archived CPU FP64 reference at rtol=1e-4, atol=1e-5. Each session independently recomputes that reference from the same FP32 weights/inputs cast to double. Actual retained FP32 outputs, weights, source identities/hashes, raw calls/blocks, trace summaries and gzip Chrome traces are saved. These numerical checks assess arithmetic consistency only.', '',
             'The CPU-only analyzer reloads source frames and retained arrays, checks all output/state/reference hashes, recomputes seeded plans, transitions, quantiles, block sums and trace intervals/copy counts. It validates saved GPU evidence without claiming to rerun GPU work. Chrome trace JSON can be decompressed and opened in a compatible trace viewer.', '',
             '```bash', 'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_schedule.py --output results/my_physics_schedule01',
             'python physicsnemo_schedule.py --seed 20261108 --output results/my_physics_schedule02',
             'python scripts/analyze_physicsnemo_schedule.py results/my_physics_schedule01 results/my_physics_schedule02 --output results/my_physics_schedule_report', '```', '',
             'Use new output paths and the existing isolated PhysicsNeMo/CUDA environment. JAX and Earth2Studio remain deferred. Cached-input numbers describe repeated identical inputs and exclude the earlier read-to-file boundary.', '',
             '![Paired batch schedules and independent real CUDA traces](physicsnemo_schedule.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated shape-schedule report:', args.output)


if __name__ == '__main__':
    main()
