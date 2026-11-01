"""Reconcile scheduling experiments, plot latency tails and coarse GPU context."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_pipeline import expected_metrics, regions
from weather_stability import execution_order, summarize
from weather_verify import sha256, write_csv


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def telemetry(run, env):
    info = env['telemetry']
    if not info.get('requested') or not info.get('launched'):
        return []
    result = []
    tz = timezone(timedelta(seconds=info['timestamp_utc_offset_sec']))
    with (run/'gpu_telemetry.csv').open(encoding='utf-8', newline='') as stream:
        for values in csv.reader(stream):
            if len(values) != 9:
                continue
            values = [v.strip() for v in values]
            try:
                stamp = datetime.strptime(values[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()
                index = int(values[1])
            except ValueError:
                continue
            if index != 0:
                continue
            row = dict(unix_time=stamp, pstate=values[8])
            for name, value in zip(['gpu_util', 'memory_util', 'sm_mhz', 'memory_mhz', 'temperature_c', 'power_w'], values[2:8]):
                try:
                    row[name] = float(value)
                except ValueError:
                    row[name] = None
            result.append(row)
    return result


def validate(run, data):
    env = json.loads((run/'environment.json').read_text(encoding='utf-8'))
    manifest = json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(data)
    if env['source_sha256'] != digest or manifest != env['source_manifest'] or manifest['files']['batch.zarr'] != digest:
        raise ValueError('Source data/manifest mismatch')
    expected, coverage = expected_metrics(data, env['case'], regions(env['case']['side'], env['case']['side']))
    if coverage != env['region_coverage']:
        raise ValueError('Different regional coverage or weights')
    rows = read_rows(run/'timing_samples.csv')
    summary = read_rows(run/'timing_summary.csv')
    methods = [b+'_'+m for b in ['cpu']+(['cuda'] if env['torch'] else []) for m in ['once_shared', 'reuse_shared']]
    if env['method_names'] != methods:
        raise ValueError('Unexpected method set')
    expected_sequence = [(schedule, name, repeat, position) for i, schedule in enumerate(env['schedules'])
                         for name, repeat, position in execution_order(methods, schedule, env['repeats'], env['seed']+i)]
    if [(r['schedule'], r['method'], int(r['repeat']), int(r['position'])) for r in rows] != expected_sequence:
        raise ValueError('Samples do not match the recorded execution schedule')
    stats = {}
    for row in summary:
        key = (row['schedule'], row['method'])
        if key in stats:
            raise ValueError('Duplicate summary')
        stats[key] = row
    keys = {(schedule, method) for schedule in env['schedules'] for method in methods}
    if set(stats) != keys:
        raise ValueError('Missing/extra summary groups')
    for key in keys:
        selected = [r for r in rows if (r['schedule'], r['method']) == key]
        assert sorted(int(r['repeat']) for r in selected) == list(range(env['repeats']))
        actual = summarize(selected)
        for name, value in actual.items():
            np.testing.assert_allclose(float(stats[key][name]), value, rtol=1e-12)
    bound = 1e-5+1e-5*np.max(np.abs(expected))
    previous_start = 0
    previous = None
    for row in rows:
        assert 0 <= float(row['max_abs_metric_error']) <= bound
        assert int(row['unix_start_ns']) > previous_start
        previous_start = int(row['unix_start_ns'])
        for field in ['gc_collections', 'user_cpu_sec', 'system_cpu_sec', 'voluntary_switches', 'involuntary_switches', 'minor_faults', 'major_faults']:
            if field in row:
                assert np.isfinite(float(row[field])) and float(row[field]) >= 0
        if row['schedule'] == 'blocked':
            assert row['previous_method'] == row['method']
        else:
            expected_previous = methods[-1] if previous is None or previous['schedule'] != row['schedule'] else previous['method']
            assert row['previous_method'] == expected_previous
        previous = row
    samples = telemetry(run, env)
    print(f'PASS {run.name}: {len(rows)} checked workflows, {len(stats)} latency groups, {len(samples)} GPU observations')
    return dict(path=run, env=env, rows=rows, stats=stats, telemetry=samples)


def context_summary(runs):
    rows = []
    for run in runs:
        times = np.array([r['unix_time'] for r in run['telemetry']])
        for (schedule, method), stat in run['stats'].items():
            group = [r for r in run['rows'] if r['schedule'] == schedule and r['method'] == method]
            clock, util, states = [], [], set()
            for row in group:
                if not len(times):
                    continue
                midpoint = int(row['unix_start_ns'])/1e9+float(row['seconds'])/2
                index = int(np.argmin(np.abs(times-midpoint)))
                # Coarse context only: avoid matching observations far from a call.
                if abs(times[index]-midpoint) > .6:
                    continue
                item = run['telemetry'][index]
                if item['sm_mhz'] is not None:
                    clock.append(item['sm_mhz'])
                if item['gpu_util'] is not None:
                    util.append(item['gpu_util'])
                states.add(item['pstate'])
            rows.append(dict(session=run['path'].name, schedule=schedule, method=method,
                             matched_clock_samples=len(clock), sm_mhz_min=min(clock) if clock else None,
                             sm_mhz_median=float(np.median(clock)) if clock else None, sm_mhz_max=max(clock) if clock else None,
                             gpu_util_median=float(np.median(util)) if util else None, pstates=';'.join(sorted(states)),
                             gc_collection_invocations=sum(float(r['gc_collections']) > 0 for r in group),
                             involuntary_switches_median=float(np.median([float(r['involuntary_switches']) for r in group]))
                             if all('involuntary_switches' in r for r in group) else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(run, args.data) for run in args.runs]
    first = runs[0]
    if any(r['env']['case'] != first['env']['case'] or r['env']['source_sha256'] != first['env']['source_sha256'] for r in runs):
        raise ValueError('Experiments must use identical inputs/case')
    args.output.mkdir(parents=True, exist_ok=True)
    context = context_summary(runs)
    write_csv(args.output/'context_summary.csv', context)
    if not all(r['env']['torch'] for r in runs):
        print('CPU-only reconciliation complete; GPU plot requires CUDA sessions')
        return
    if any(set(r['env']['schedules']) != {'blocked', 'interleaved'} for r in runs):
        raise ValueError('GPU comparison report requires both blocked and interleaved schedules')
    monitored = [r for r in runs if r['env']['telemetry']['requested']]
    plot_runs = monitored or runs
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    for column, mode in enumerate(['once_shared', 'reuse_shared']):
        for backend, color in [('cpu', '#3366aa'), ('cuda', '#dd7722')]:
            for schedule, style in [('blocked', '-'), ('interleaved', '--')]:
                values = sorted(float(row['seconds'])*1000 for run in plot_runs for row in run['rows'] if row['method'] == backend+'_'+mode and row['schedule'] == schedule)
                if values:
                    axes[0, column].step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, linestyle=style, label=f'{backend.upper()} {schedule}')
        axes[0, column].axhline(.95, color='gray', linewidth=1, linestyle=':')
        axes[0, column].set(xscale='log', xlabel='Five-region wall latency (ms, log scale)', ylabel='Fraction of measured invocations <= latency',
                            title='Includes read/preparation/upload' if column == 0 else 'Prepared input reuse; setup excluded equally')
        axes[0, column].legend(fontsize=8)
    conditions = [('once_shared', 'blocked'), ('once_shared', 'interleaved'), ('reuse_shared', 'blocked'), ('reuse_shared', 'interleaved')]
    width = .75/len(runs)
    for i, run in enumerate(runs):
        med = [float(run['stats'][(s, 'cuda_'+m)]['p50_sec'])*1000 for m, s in conditions]
        high = [float(run['stats'][(s, 'cuda_'+m)]['p95_sec'])*1000 for m, s in conditions]
        bars = axes[1, 0].bar(np.arange(4)+(i-(len(runs)-1)/2)*width, med, width=width*.92,
                              yerr=np.stack([np.zeros(4), np.array(high)-med]), capsize=3,
                              label=run['path'].name.replace('rtx3050_stability_', ''))
        axes[1, 0].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axes[1, 0].set(xticks=np.arange(4), xticklabels=['First-use\nblocked', 'First-use\ninterleaved', 'Reuse\nblocked', 'Reuse\ninterleaved'],
                   ylabel='GPU wall latency (ms)', title='Bars: P50; upper whiskers: P95 (not confidence intervals)')
    axes[1, 0].legend(fontsize=8)
    observed = next((r for r in runs if r['telemetry']), None)
    if observed:
        origin = min(int(row['unix_start_ns']) for row in observed['rows'])/1e9
        records = [r for r in observed['telemetry'] if r['sm_mhz'] is not None]
        axes[1, 1].plot([r['unix_time']-origin for r in records], [r['sm_mhz'] for r in records], color='#446688', alpha=.65, label='Device-wide SM clock')
        axes[1, 1].set(xlabel='Seconds since first measured invocation', ylabel='Coarse SM clock observation (MHz)', title=observed['path'].name+': context, not causal proof')
        secondary = axes[1, 1].twinx()
        for schedule, color in [('blocked', '#228855'), ('interleaved', '#dd7722')]:
            group = [r for r in observed['rows'] if r['method'] == 'cuda_reuse_shared' and r['schedule'] == schedule]
            secondary.scatter([int(r['unix_start_ns'])/1e9-origin for r in group], [float(r['seconds'])*1000 for r in group], s=10, alpha=.55, color=color, label='GPU reuse '+schedule)
        secondary.set(yscale='log', ylabel='GPU reuse wall latency (ms, log scale)')
        handles, labels = axes[1, 1].get_legend_handles_labels()
        other_handles, other_labels = secondary.get_legend_handles_labels()
        axes[1, 1].legend(handles+other_handles, labels+other_labels, fontsize=8)
        if records:
            axes[1, 1].set_ylim(0, max(r['sm_mhz'] for r in records)*1.45)
        latencies = [float(r['seconds'])*1000 for r in observed['rows'] if r['method'] == 'cuda_reuse_shared']
        if latencies:
            secondary.set_ylim(min(latencies)*.8, max(latencies)*3)
    else:
        axes[1, 1].text(.1, .5, 'GPU telemetry unavailable / disabled', transform=axes[1, 1].transAxes)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('RTX 3050 latency stability: same verified five-region work, different execution schedules\nECDFs pool monitored sessions when present; monitoring-off control is shown separately in the P50/P95 panel', fontsize=13)
    fig.savefig(args.output/'stability.png', dpi=160)
    plt.close(fig)
    text = ['# Regional verification latency distributions and scheduling', '',
            'Every invocation computes the same full domain plus four quadrants, retaining individual initialization/lead/member Bias/MAE/RMSE. The source is the checked-in genuine ECMWF ensemble batch matched to GFS f000 analysis. No numerical kernel or precision was changed from the shared-input pipeline.', '',
            'Blocked execution measures one method consecutively, with warmups directly before its block. Interleaved execution shuffles CPU/GPU first-use and reuse methods within every repetition. Consecutive methods still return results and validate them between calls; this is application wall latency, not continuously enqueued device-only throughput. First-use means fresh read/preparation/upload with warm local data and initialized CUDA.', '',
            '## Distribution summaries', '',
            '| Session | Monitor | Schedule | Method | N | P50 ms | P95 ms | P99 ms | Max ms |',
            '|---|---|---|---|---:|---:|---:|---:|---:|']
    for run in runs:
        for schedule in run['env']['schedules']:
            for method in run['env']['method_names']:
                row = run['stats'][(schedule, method)]
                text.append(f'| {run["path"].name} | {"on" if run["env"]["telemetry"]["requested"] else "off"} | {schedule} | {method} | {row["samples"]} | '+' | '.join(f'{float(row[k])*1000:.3f}' for k in ['p50_sec', 'p95_sec', 'p99_sec', 'max_sec'])+' |')
    text += ['', '## Measured schedule differences', '']
    for run in runs:
        for mode in ['once_shared', 'reuse_shared']:
            a = run['stats'][('blocked', 'cuda_'+mode)]
            b = run['stats'][('interleaved', 'cuda_'+mode)]
            text.append(f'- {run["path"].name}, {mode}: blocked P50/P95 {float(a["p50_sec"])*1000:.3f}/{float(a["p95_sec"])*1000:.3f} ms; interleaved {float(b["p50_sec"])*1000:.3f}/{float(b["p95_sec"])*1000:.3f} ms. Interleaved/blocked median ratio {float(b["p50_sec"])/float(a["p50_sec"]):.2f}x.')
    clock_notes = []
    for run in monitored:
        pair = {(r['schedule'], r['method']): r for r in context if r['session'] == run['path'].name}
        blocked_clock = pair[('blocked', 'cuda_reuse_shared')]['sm_mhz_median']
        interleaved_clock = pair[('interleaved', 'cuda_reuse_shared')]['sm_mhz_median']
        if blocked_clock is not None and interleaved_clock is not None:
            clock_notes.append(f'{run["path"].name}: blocked/interleaved matched median SM clock {blocked_clock:g}/{interleaved_clock:g} MHz')
    text += ['', 'Schedule is an observed performance condition, not an identified hardware cause. Device clocks, Windows/WSL scheduling, driver overhead, other device work, allocation and host dispatch are candidate mechanisms; this experiment does not isolate them. CPU and GPU cache policies are identical. A monitoring-off session is an observer-overhead control, not a randomized estimate of monitoring cost: order, seed and external state also differ.', '',
             ('; '.join(clock_notes)+'. ' if clock_notes else '')+'Coarse telemetry can miss short active bursts and sample idle gaps. It does not show the clock of every timed kernel or prove clock behavior caused the latency difference. Kernel/host trace profiling is needed to separate dispatch gaps, device execution and result return.', '',
             '## Reading the figure', '',
             '- Upper left: empirical cumulative distribution of complete local workflows. Upper right: same plot with prepared-input setup excluded equally for CPU/GPU. At a given time on the x-axis, y is the observed fraction completed by that time. Logarithmic x-axes expose the tails; the dotted line marks 95%.',
             '- Lower left: each session is kept separate. Bars show P50, whiskers extend to P95; neither is a confidence interval. The monitoring-off control is included here, not silently pooled into monitored ECDF curves.',
             '- Lower right: one monitored session, with coarse device-wide SM clock samples and wall times of GPU reuse invocations. The two y-axes measure different quantities. Clock sampling cannot resolve individual kernels or attribute activity to this process.', '',
             '## Context and limitations', '',
             '[context_summary.csv](context_summary.csv) matches the nearest GPU observation to each invocation midpoint only if within 0.6 s. Clock observations can be reused for multiple short invocations, so matched counts are not independent telemetry samples. Unsupported counters remain missing. Device utilization reflects a sampling window and all device activity. Process context-switch/page-fault/CPU-time and GC deltas are collected outside timers and are descriptive; snapshots can include a small amount of surrounding instrumentation. None is causal proof.', '',
             'Every warmup and measured returned metric is validated against unchanged dense-path FP64 arithmetic (rtol=atol=1e-5). The analyzer independently checks input hashes, regional coverage, deterministic execution order, previous-method labels, all timing quantiles and recorded maximum numerical error bounds. No outliers are removed. CUDA initialization, source hashing, correctness validation, resource snapshots, CSV writing and explicit reuse input-cache setup are excluded from invocation timers.', '',
             'The number of repetitions and schedule order are recorded per session. P95/P99 are sample estimates, especially uncertain at the far tail; they are not guarantees. All reported measurements use one machine/day/case. GPU monitoring itself may affect timing. These experiments do not rank forecast models or establish general production latency.', '',
             '## Reproduce', '', '```bash',
             'python weather_stability.py --output results/my_stability01 --telemetry',
             'python weather_stability.py --output results/my_stability02 --telemetry --schedules interleaved blocked --seed 20261106',
             'python weather_stability.py --output results/my_stability_control --seed 20261107',
             'python scripts/analyze_stability.py results/my_stability01 results/my_stability02 results/my_stability_control --output results/my_stability_comparison',
             '```', '',
             'Use the existing WSL CUDA environment. `--backend numpy --case control_small --repeats 4 --warmup 1` is the offline CPU smoke case. [Earlier phase-cost report](../rtx3050_pipeline_comparison/analysis.md), [JAX adoption survey](../../docs/jax_survey_zh.md).', '',
             '![Latency distributions, schedule comparison and coarse GPU context](stability.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Saved validated stability report and figure: {args.output}')


if __name__ == '__main__':
    main()
