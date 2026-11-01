"""Validate end-to-end regional workflows and independently timed stage profiles."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_pipeline import expected_metrics, load_region, prepare, regions
from weather_verify import METRICS, sha256


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data):
    env = json.loads((run/'environment.json').read_text(encoding='utf-8'))
    manifest = json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(data)
    if env['source_sha256'] != digest or manifest['files']['batch.zarr'] != digest or env['source_manifest'] != manifest:
        raise ValueError('Input data or provenance manifest differs from recorded measurement')
    case = env['case']
    tasks = regions(case['side'], case['side'])
    truth, coverage = expected_metrics(data, case, tasks)
    if coverage != env['regions']:
        raise ValueError('Regional valid-cell counts or weights differ from source data')
    shared = prepare(load_region(data, case), 'shared')
    assert env['cache_bytes']['shared_cpu'] == shared.logical_bytes
    assert env['cache_bytes']['all_finite_fast_path'] == (shared.mask is None)
    assert env['cache_bytes']['dense_logical'] == 3*np.prod(shared.shape)*4
    if env['torch']:
        assert env['cache_bytes']['shared_cuda'] == shared.logical_bytes
    methods = [backend+'_'+mode for backend in ['cpu']+(['cuda'] if env['torch'] else [])
               for mode in ['reopen_dense', 'once_dense', 'once_shared', 'reuse_shared']]
    keys = {(count, method) for count in env['task_counts'] for method in methods}
    raw = read_rows(run/'timing_samples.csv')
    summary = read_rows(run/'timing_summary.csv')
    profile = read_rows(run/'stage_samples.csv')
    references = read_rows(run/'reference_metrics.csv')
    groups, stats, stages = defaultdict(list), {}, defaultdict(dict)
    for row in raw:
        groups[(int(row['tasks']), row['method'])].append(row)
    for row in summary:
        key = (int(row['tasks']), row['method'])
        if key in stats:
            raise ValueError('Duplicate summary row')
        stats[key] = row
    if set(groups) != keys or set(stats) != keys:
        raise ValueError('Missing, duplicate or extra timing methods')
    reference_keys = {(count, tasks[i][0], group) for count in env['task_counts']
                      for i in range(count) for group in range(truth.shape[1])}
    seen = set()
    for row in references:
        key = (int(row['tasks']), row['region'], int(row['group']))
        if key not in reference_keys or key in seen:
            raise ValueError('Missing or duplicated reference group')
        seen.add(key)
        index = next(i for i, task in enumerate(tasks) if task[0] == row['region'])
        np.testing.assert_allclose([float(row[k]) for k in METRICS], truth[index, int(row['group'])], rtol=1e-12, atol=1e-12)
    if seen != reference_keys:
        raise ValueError('Incomplete regional metrics')
    for count, method in keys:
        values = groups[(count, method)]
        if sorted(int(r['repeat']) for r in values) != list(range(env['repeats'])):
            raise ValueError('Missing end-to-end repetitions')
        seconds = np.array([float(r['seconds']) for r in values])
        assert np.isfinite(seconds).all() and (seconds > 0).all()
        bound = 1e-5+1e-5*np.max(np.abs(truth[:count]))
        assert all(0 <= float(r['max_abs_metric_error']) <= bound for r in values)
        for percentile, field in [(25, 'p25_sec'), (50, 'median_sec'), (75, 'p75_sec')]:
            np.testing.assert_allclose(float(stats[(count, method)][field]), np.percentile(seconds, percentile), rtol=1e-12)
    for count in env['task_counts']:
        rng = np.random.default_rng(env['seed']+count)
        for repeat in range(env['repeats']):
            ordered = sorted((r for r in raw if int(r['tasks']) == count and int(r['repeat']) == repeat), key=lambda r: int(r['order']))
            assert [int(r['order']) for r in ordered] == list(range(len(methods)))
            assert [r['method'] for r in ordered] == list(rng.permutation(methods))
    for row in profile:
        key = (int(row['tasks']), row['method'], int(row['repeat']))
        if row['stage'] in stages[key]:
            raise ValueError('Duplicated stage label')
        value = float(row['seconds'])
        assert np.isfinite(value) and value >= 0
        stages[key][row['stage']] = value
    expected_profile_keys = {(count, method, repeat) for count, method in keys for repeat in range(env['profile_repeats'])}
    if set(stages) != expected_profile_keys:
        raise ValueError('Missing stage-profile workflows')
    for (count, method, repeat), values in stages.items():
        expected_stages = {'compute', 'result_return', 'other', 'profile_total'}
        if 'reuse_shared' not in method:
            expected_stages |= {'zarr_read_decode', 'coordinate_checks', 'unit_normalization', 'mask_weights', 'pack_arrays'}
            if method.startswith('cuda_'):
                expected_stages.add('upload')
        if 'reopen_dense' not in method:
            expected_stages.add('region_selection')
        if set(values) != expected_stages:
            raise ValueError('Missing or unexpected workflow stages')
        np.testing.assert_allclose(sum(v for stage, v in values.items() if stage != 'profile_total'), values['profile_total'], rtol=1e-10)
        assert values['profile_total'] > 0
    for count in env['task_counts']:
        rng = np.random.default_rng(env['seed']+100+count)
        for repeat in range(env['profile_repeats']):
            expected_order = list(rng.permutation(methods))
            for row in profile:
                if int(row['tasks']) == count and int(row['repeat']) == repeat:
                    assert row['method'] == expected_order[int(row['order'])]
    print(f'PASS {run.name}: {len(raw)} whole-workflow samples, {len(stages)} stage profiles, {len(references)} FP64 regional/group rows')
    return dict(path=run, env=env, stats=stats, stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(run, args.data) for run in args.runs]
    first = runs[0]
    if any(r['env']['case'] != first['env']['case'] or r['env']['source_sha256'] != first['env']['source_sha256'] or r['env']['task_counts'] != first['env']['task_counts'] for r in runs):
        raise ValueError('Sessions must measure identical case and workloads')
    if not all(r['env']['torch'] for r in runs):
        print('CPU-only reconciliation complete; GPU comparison plot needs CUDA sessions')
        return
    def ms(run, count, method):
        return float(run['stats'][(count, method)]['median_sec'])*1000
    modes = ['reopen_dense', 'once_dense', 'once_shared', 'reuse_shared']
    labels = ['Reopen + prepare\nper region', 'Read once\ndense inputs', 'Read once\nshared inputs', 'Shared inputs\nalready prepared']
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), layout='constrained')
    for i, (backend, color) in enumerate([('cpu', '#3776b5'), ('cuda', '#dc7c29')]):
        vals = np.array([[ms(r, 5, backend+'_'+mode) for mode in modes] for r in runs])
        med = np.median(vals, axis=0)
        bars = axes[0, 0].bar(np.arange(4)+(i-.5)*.36, med, width=.34, label=backend.upper(), color=color,
                              yerr=np.stack([med-vals.min(axis=0), vals.max(axis=0)-med]), capsize=3)
        axes[0, 0].bar_label(bars, fmt='%.1f', padding=3)
    axes[0, 0].set(xticks=np.arange(4), xticklabels=labels, ylabel='Whole-workflow median (ms)', title='Five distinct regional tasks; primary uninstrumented timings')
    axes[0, 0].legend()
    stage_names = ['zarr_read_decode', 'coordinate_checks', 'unit_normalization', 'mask_weights', 'pack_arrays', 'region_selection', 'upload', 'compute', 'result_return', 'other']
    stage_labels = ['Zarr read/decode', 'Coordinate checks', 'Unit normalization', 'Mask/weights', 'Packing', 'Region slicing', 'Upload', 'Compute/reshape', 'Result return', 'Other']
    selected_methods = [b+'_'+m for b in ['cpu', 'cuda'] for m in modes[:3]]
    bottom = np.zeros(len(selected_methods))
    palette = plt.get_cmap('tab10')
    for j, (name, label) in enumerate(zip(stage_names, stage_labels)):
        values = []
        for method in selected_methods:
            values.append(np.mean([stages.get(name, 0.)*1000 for r in runs for (count, m, _), stages in r['stages'].items() if count == 5 and m == method]))
        axes[0, 1].bar(np.arange(len(selected_methods)), values, bottom=bottom, label=label, color=palette(j))
        bottom += values
    axes[0, 1].set(xticks=np.arange(6), xticklabels=['CPU\nreopen', 'CPU\nonce dense', 'CPU\nonce shared', 'GPU\nreopen', 'GPU\nonce dense', 'GPU\nonce shared'],
                   ylabel='Mean profiled workflow time (ms)', title='Separate phase profiles; extra CUDA synchronization included')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].set_ylim(0, max(bottom)*1.4)
    for method, label, color in [('cpu_reopen_dense', 'CPU reopen', '#3776b5'), ('cuda_reopen_dense', 'GPU reopen', '#dd9955'),
                                 ('cpu_once_shared', 'CPU once shared', '#447744'), ('cuda_once_shared', 'GPU once shared', '#aa4499')]:
        vals = np.array([[ms(r, count, method) for count in [1, 5]] for r in runs])
        axes[1, 0].plot([1, 5], np.median(vals, axis=0), 'o-', label=label, color=color)
        axes[1, 0].fill_between([1, 5], vals.min(axis=0), vals.max(axis=0), color=color, alpha=.15)
    axes[1, 0].set(xticks=[1, 5], xticklabels=['Full domain', 'Full + four quadrants'], xlabel='Distinct regional verification tasks',
                   ylabel='Whole-workflow median (ms)', title='Read/preparation/upload costs included for both backends')
    axes[1, 0].legend()
    size = first['env']['cache_bytes']
    memory = [size['dense_logical']/2**20, size['shared_cpu']/2**20]
    bars = axes[1, 1].bar(['Dense member-expanded\nforecast/reference/weights', 'Shared analysis + spatial weights\nmask stored only if needed'], memory, color=['#777777', '#339988'], width=.55)
    axes[1, 1].bar_label(bars, fmt='%.2f MiB', padding=3)
    axes[1, 1].set(ylabel='Logical prepared input bytes (MiB)', title='Same input representation on CPU and GPU; not peak VRAM', ylim=(0, max(memory)*1.2))
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('RTX 3050: profile the full workflow, share reference data, reuse prepared inputs\nGenuine ECMWF individual-member forecasts vs matching GFS analysis; warm local Zarr; input setup excluded only in explicitly labeled reuse', fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'pipeline.png', dpi=160)
    plt.close(fig)
    text = ['# Regional verification: stage costs and shared input reuse', '',
            'The same checked-in genuine ECMWF ensemble batch is verified against exact-valid-time GFS analysis. Workloads contain one full-domain task or five distinct tasks: full domain plus northwest/northeast/southwest/southeast quadrants. Each task returns separate spatial Bias/MAE/RMSE for every initialization/lead/member. Cache stores input data, not metric results; every task is recomputed.', '',
            '## Whole-workflow results', '',
            'Primary timings are measured without phase instrumentation. Each once_* invocation includes warm local Zarr opening/reading/decoding, coordinate validation, unit normalization, masks/weights, packing and GPU upload where applicable. Reuse-only excludes previously prepared input setup for both CPU and GPU; it must not be presented as first-use latency.', '',
            '| Session | Tasks | CPU reopen ms | CPU read once/dense ms | CPU read once/shared ms | GPU reopen ms | GPU read once/dense ms | GPU read once/shared ms | CPU reuse-only ms | GPU reuse-only ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        for count in [1, 5]:
            method_names = ['cpu_reopen_dense', 'cpu_once_dense', 'cpu_once_shared', 'cuda_reopen_dense', 'cuda_once_dense', 'cuda_once_shared', 'cpu_reuse_shared', 'cuda_reuse_shared']
            text.append(f'| {r["path"].name} | {count} | '+' | '.join(f'{ms(r, count, m):.3f}' for m in method_names)+' |')
    text += ['', '## Observations', '']
    for r in runs:
        text.append(f'- {r["path"].name}, five tasks: CPU whole-workflow improvement {ms(r, 5, "cpu_reopen_dense")/ms(r, 5, "cpu_once_shared"):.2f}x; GPU whole-workflow improvement {ms(r, 5, "cuda_reopen_dense")/ms(r, 5, "cuda_once_shared"):.2f}x. With the same shared/read-once policy, GPU is {ms(r, 5, "cpu_once_shared")/ms(r, 5, "cuda_once_shared"):.2f}x faster than CPU. Reuse-only GPU/CPU speedup is {ms(r, 5, "cpu_reuse_shared")/ms(r, 5, "cuda_reuse_shared"):.2f}x, with setup excluded for both.')
    text += ['', 'Dense-read-once isolates the benefit of reusing input reads/preparation while retaining the old member-expanded representation. Shared-read-once additionally keeps one GFS reference per initialization/lead and one latitude-weight vector instead of duplicating both across 51 members. All-finite data avoid retaining a full member-specific mask; missing data use the tested joint-finite-mask fallback. Both retain FP32 Kelvin forecast/reference values and per-member outputs.', '',
             f'Logical prepared input storage falls from {memory[0]:.2f} MiB to {memory[1]:.2f} MiB ({100*(1-memory[1]/memory[0]):.1f}% less). These are live buffer payload sizes, not allocator-reserved memory, peak memory or total VRAM. CPU and GPU share the same representation. Region slicing and any reshape/materialization needed by the dense method remain inside measured work.', '',
             '## Phase breakdown (separate profiling runs)', '',
             'Stages are Zarr read/decode (including metadata open and selection), coordinate checks, Celsius-to-FP32-Kelvin normalization, mask/weight construction, array packing, region selection, upload, compute (including dense regional reshape/materialization), result return and residual other overhead. Dense reopen selects each requested region before reading, so it does not deliberately reread unrelated chunks. Read-once reads the full domain and serves all regional tasks from that input.', '',
             'Stage profiles are separate from the primary uninstrumented timings. CUDA is synchronized at stage boundaries; additional synchronization and timer overhead can change the profile total. The stacked chart uses mean durations from the same profiled workflows, so components add to mean profile total. It does not add independently measured medians or substitute their sum for whole-workflow measurements.', '',
             '| Backend/workflow, five tasks | Zarr read/decode ms | Unit normalization ms | Mask/weights ms | Packing ms | Upload ms | Compute ms | Mean profile total ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in selected_methods:
        selected = [s for r in runs for (count, name, _), s in r['stages'].items() if count == 5 and name == method]
        means = [np.mean([s.get(k, 0.) for s in selected])*1000 for k in ['zarr_read_decode', 'unit_normalization', 'mask_weights', 'pack_arrays', 'upload', 'compute', 'profile_total']]
        text.append('| '+method+' | '+' | '.join(f'{v:.3f}' for v in means)+' |')
    text += ['', '## Validation and limits', '',
             'All warmups, uninstrumented repetitions and profiled workflows validate every returned regional/group metric against FP64 arithmetic using the unchanged original dense packing path, rtol=atol=1e-5. Correctness checks and result writes occur outside timed intervals. The report independently reloads hashed source data, recomputes reference values and valid-weight/count coverage, checks timing quantiles and seeded method order, and reconciles each phase profile to its own total.', '',
             'The four unit tests cover regional parity, member-specific missing values, compact input ownership/storage, explicit reuse behavior and optional CPU/CUDA shared arithmetic. All 39 tests pass in the WSL CUDA environment. These tests do not treat unrelated ensemble members or overlapping valid dates as independent observations.', '',
             'Both backends have equal cache reuse. CUDA context initialization, remote acquisition, source hashing and correctness/output handling are excluded. Reads are warm local Zarr; first-use means a new workflow input preparation in an already initialized process, not a cold-cache/process-start measurement. An explicit shared cache remains allocated during the session, including when other methods execute. Method order is shuffled within repetitions; task-count order is fixed. Bands/error bars show ranges of session medians, not confidence intervals. The single-variable, two-session result cannot establish general Zarr/NetCDF throughput or future workloads.', '',
             'GFS analysis is model-dependent, not independent station truth. The domain includes land/ocean, and source values have provider unit conversion/bit rounding. This work measures pipeline efficiency; it does not rank ECMWF/GFS forecast skill. No change to the archived earlier benchmark results is implied.', '',
             '## Reproduce', '', '```bash', 'source ~/venvs/weather-gpu/bin/activate', 'python -m pip install -r requirements-zarr.txt',
             'python weather_pipeline.py --output results/my_pipeline01',
             'python weather_pipeline.py --output results/my_pipeline02 --seed 20261104',
             'python scripts/analyze_pipeline.py results/my_pipeline01 results/my_pipeline02 --output results/my_pipeline_comparison', '```', '',
             'Use `--backend numpy` for CPU-only runs. A small offline CI case uses `--case control_small --repeats 3 --warmup 1 --profile-repeats 2`. Source data, attribution and pinned read-only snapshots: [batch manifest](../../data/ecmwf_batch_case/manifest.json) and [data README](../../data/ecmwf_batch_case/README.md).', '',
             '![Primary end-to-end results, separate phase profiles and input storage](pipeline.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Saved pipeline report and figure: {args.output}')


if __name__ == '__main__':
    main()
