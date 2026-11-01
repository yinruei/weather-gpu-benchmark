"""Reconcile batch metrics/timings and compare two measured CPU/CUDA sessions."""
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
from weather_batch import load_batch, numpy_grouped, pack_batch
from weather_verify import METRICS, sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data):
    env = json.loads((run/'environment.json').read_text(encoding='utf-8'))
    manifest = json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    digest = sha256(data)
    if env['store_sha256'] != digest or manifest['files']['batch.zarr'] != digest or env['source_manifest'] != manifest:
        raise ValueError('Batch store or source manifest differs from measured inputs')
    metrics, summary, samples = [rows(run/file) for file in ['metrics.csv', 'timing_summary.csv', 'timing_samples.csv']]
    summaries, sample_groups, metric_groups = {}, defaultdict(list), defaultdict(list)
    for row in summary:
        key = (row['case'], row['method'])
        if key in summaries:
            raise ValueError('Duplicate summary rows')
        summaries[key] = row
    for row in samples:
        sample_groups[(row['case'], row['method'])].append(row)
    for row in metrics:
        metric_groups[row['case']].append(row)
    case_names = {c['case'] for c in env['cases']}
    if set(metric_groups) != case_names or set(sample_groups) != set(summaries):
        raise ValueError('Missing or unexpected cases/methods')
    backends = ['reference_fp64', 'numpy_fp32']+(['cuda_fp32'] if env['torch'] else [])
    reference_by_case = {}
    for case in env['cases']:
        ds = load_batch(data, case)
        arrays, counts = pack_batch(ds)
        truth = numpy_grouped(*(a.astype(np.float64) for a in arrays))
        weight_sum = arrays[2].sum(axis=1, dtype=np.float64)
        identities = [(str(initial), int(lead/np.timedelta64(1, 'h')), str(initial+lead), int(member))
                      for initial in ds.init_time.values for lead in ds.lead_time.values for member in ds.ensemble_member.values]
        expected = {(*identity, backend): i for i, identity in enumerate(identities) for backend in backends}
        actual = metric_groups[case['case']]
        if len(actual) != len(expected):
            raise ValueError('Wrong metric row count')
        seen = set()
        for row in actual:
            key = (row['initialization'], int(row['lead_hours']), row['valid_time'], int(row['member']), row['backend'])
            if key not in expected or key in seen:
                raise ValueError('Duplicated, merged or mislabelled forecast group')
            seen.add(key)
            i = expected[key]
            np.testing.assert_allclose([float(row[k]) for k in METRICS], truth[i], rtol=1e-5, atol=1e-5)
            assert int(row['valid_points']) == counts[i]
            np.testing.assert_allclose(float(row['valid_weight_sum']), weight_sum[i], rtol=1e-12)
        methods = ['numpy_resident', 'numpy_local_pipeline']
        if env['torch']:
            methods += ['cuda_resident', 'cuda_upload', 'cuda_local_pipeline']
        if (case['case'], 'cuda_group_loop') in summaries:
            assert env['torch'] and case == env['cases'][-1]
            methods.append('cuda_group_loop')
        if {method for name, method in summaries if name == case['case']} != set(methods):
            raise ValueError('Unexpected timing method set')
        for method in methods:
            key = (case['case'], method)
            raw, stat = sample_groups[key], summaries[key]
            assert sorted(int(r['repeat']) for r in raw) == list(range(env['repeats']))
            seconds = np.array([float(r['seconds']) for r in raw])
            assert np.isfinite(seconds).all() and (seconds > 0).all()
            assert all(0 <= float(r['max_abs_metric_error']) <= 1e-5+1e-5*np.max(np.abs(truth)) for r in raw)
            for record in [stat]+raw:
                assert int(record['points']) == arrays[0].size
                assert int(record['groups']) == len(identities)
                assert int(record['cells_per_group']) == arrays[0].shape[1]
                assert int(record['valid_points']) == counts.sum()
            for percentile, field in [(25, 'p25_sec'), (50, 'median_sec'), (75, 'p75_sec')]:
                np.testing.assert_allclose(float(stat[field]), np.percentile(seconds, percentile), rtol=1e-12)
        # The recorded within-repeat order must match the seeded randomization.
        rng = np.random.default_rng(env['seed'])
        for repeat in range(env['repeats']):
            ordered = sorted((r for r in samples if r['case'] == case['case'] and int(r['repeat']) == repeat), key=lambda r: int(r['order']))
            assert [int(r['order']) for r in ordered] == list(range(len(methods)))
            assert [r['method'] for r in ordered] == list(rng.permutation(methods))
        reference_by_case[case['case']] = [r for r in actual if r['backend'] == 'reference_fp64']
    if {name for name, _ in summaries} != case_names:
        raise ValueError('Unexpected summary case')
    print(f'PASS {run.name}: {len(metrics)} metric rows and {len(samples)} timing samples reconcile')
    return dict(path=run, env=env, summary=summaries, metrics=reference_by_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(path, args.data) for path in args.runs]
    first = runs[0]
    cases = first['env']['cases']
    if any(r['env']['cases'] != cases or r['env']['store_sha256'] != first['env']['store_sha256'] for r in runs):
        raise ValueError('Sessions must measure identical data and cases')
    if not all(r['env']['torch'] for r in runs):
        print('CPU-only reconciliation complete; GPU comparison report requires CUDA sessions')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    def milliseconds(run, case, method):
        return float(run['summary'][(case['case'], method)]['median_sec'])*1000
    x = np.array([int(first['summary'][(c['case'], 'numpy_resident')]['points']) for c in cases])
    largest = cases[-1]
    last = largest['case']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    colors = dict(numpy_resident='#3366aa', cuda_resident='#dd7722', cuda_upload='#228855')
    for method, label in [('numpy_resident', 'NumPy resident'), ('cuda_resident', 'CUDA resident'), ('cuda_upload', 'CUDA upload + compute')]:
        values = np.array([[milliseconds(r, c, method) for c in cases] for r in runs])
        axes[0, 0].plot(x, np.median(values, axis=0), 'o-', color=colors[method], label=label)
        axes[0, 0].fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[method], alpha=.18)
    axes[0, 0].set(xscale='log', yscale='log', xlabel='Actual forecast values (init x lead x member x cells)', ylabel='Median time (ms)', title='Batch compute scaling; full per-group metric return included')
    axes[0, 0].legend()
    for method, numerator, label, color in [('cuda_resident', 'numpy_resident', 'Resident compute', '#dd7722'),
                                          ('cuda_upload', 'numpy_resident', 'Upload + compute', '#228855'),
                                          ('cuda_local_pipeline', 'numpy_local_pipeline', 'Warm local Zarr pipeline', '#8855aa')]:
        values = np.array([[milliseconds(r, c, numerator)/milliseconds(r, c, method) for c in cases] for r in runs])
        axes[0, 1].plot(x, np.median(values, axis=0), 'o-', color=color, label=label)
        axes[0, 1].fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=.18)
    axes[0, 1].axhline(1, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set(xscale='log', xlabel='Actual forecast values', ylabel='CPU time / CUDA time (>1: CUDA faster)', title='Speedup depends on the measurement boundary')
    axes[0, 1].legend()
    methods = ['numpy_resident', 'cuda_resident', 'cuda_upload', 'numpy_local_pipeline', 'cuda_local_pipeline']
    labels = ['CPU\nresident', 'GPU\nresident', 'GPU\nupload', 'CPU\nlocal pipeline', 'GPU\nlocal pipeline']
    if (last, 'cuda_group_loop') in first['summary']:
        methods.append('cuda_group_loop')
        labels.append('GPU\nper-group loop')
    for i, run in enumerate(runs):
        vals = [milliseconds(run, largest, method) for method in methods]
        bars = axes[1, 0].bar(np.arange(len(methods))+(i-(len(runs)-1)/2)*.35, vals, width=.32, label=run['path'].name.replace('rtx3050_ecmwf_batch_', ''))
        axes[1, 0].bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    axes[1, 0].set(xticks=np.arange(len(methods)), xticklabels=labels, yscale='log', ylabel='Median time (ms)', title=f'{x[-1]:,} values; {int(first["summary"][(last, "numpy_resident")]["groups"]):,} separate forecast groups')
    axes[1, 0].legend()
    by_lead = defaultdict(list)
    for row in first['metrics'][last]:
        by_lead[int(row['lead_hours'])].append(row)
    leads = sorted(by_lead)
    pooled = {}
    for lead in leads:
        group = by_lead[lead]
        w = np.array([float(r['valid_weight_sum']) for r in group])
        pooled[lead] = [np.average([float(r[k]) for r in group], weights=w) if k != 'rmse'
                        else np.sqrt(np.average([float(r[k])**2 for r in group], weights=w)) for k in METRICS]
    for i, name in enumerate(METRICS):
        axes[1, 1].plot(leads, [pooled[lead][i] for lead in leads], 'o-', label=name.upper())
    axes[1, 1].set(xlabel='Forecast lead (hours)', ylabel='Individual-forecast error (K)', title='Individual-member errors vs GFS analysis; not ensemble mean')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('RTX 3050: genuine ECMWF ensemble Zarr batch verification\n4 initializations x 8 leads x 51 members x 64 x 64; shaded range across measured sessions', fontsize=14)
    fig.savefig(args.output/'batch_scaling.png', dpi=160)
    plt.close(fig)
    text = ['# Real ECMWF Zarr batch scaling on RTX 3050', '',
            'Four genuine initializations (2024-05-06..09 00 UTC), leads 24/30/36/42/48/54/60/66 h, members 0..50, native 0.25-degree 64 x 64 grid (18.25..34 N, 116..131.75 E). Total 6,684,672 actual forecast values; no repeated/tiling-generated inputs. GFS f000 reference uses 20 unique matching valid-time cycles. Each init/lead/member is scored separately over space; this is not an ensemble-mean forecast experiment.', '',
            'The data include land and ocean. GFS analysis is model-dependent, not independent observations. Members and overlapping valid times are correlated; the metric counts are not independent sample sizes. Provider unit conversion/bit rounding applies. Different domains, outputs and source processing prevent a direct timing/skill comparison with the earlier single-case report.', '',
            '## Observed medians', '',
            '| Session | Forecast values | Spatial groups | CPU resident ms | GPU resident ms | Resident speedup | GPU upload + compute ms | CPU local pipeline ms | GPU local pipeline ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for case in cases:
            cpu, gpu, upload, cp, gp = [milliseconds(run, case, method) for method in ['numpy_resident', 'cuda_resident', 'cuda_upload', 'numpy_local_pipeline', 'cuda_local_pipeline']]
            row = run['summary'][(case['case'], 'numpy_resident')]
            text.append(f'| {run["path"].name} | {int(row["points"]):,} | {row["groups"]} | {cpu:.4f} | {gpu:.4f} | {cpu/gpu:.2f}x | {upload:.4f} | {cp:.4f} | {gp:.4f} |')
    text += ['', '## What changed GPU efficiency', '',
             '- All forecast groups are represented in one batch tensor. Spatial reductions run across the last axis, retaining initialization/lead/member identity. The full group metric matrix returns to CPU in one transfer.',
             '- Resident compute reuses packed inputs on device. Upload + compute copies existing packed host arrays each invocation; local pipeline additionally opens Zarr, reads/decodes, checks coordinates/units, normalizes Celsius to FP32 Kelvin, masks, builds weights and packs inputs.',
             '- GPU acceleration depends on case shape as well as value count. This seven-case sample is not a universal crossover threshold; cases change dates, member count and spatial size together.',
             '- Resident acceleration is substantial on the larger batches, while the warm local workflow gains are modest. Local pipeline timings exclude remote downloading. No claim of general Zarr versus NetCDF performance follows from these measurements.', '']
    for run in runs:
        cpu, gpu, upload, cp, gp = [milliseconds(run, largest, m) for m in ['numpy_resident', 'cuda_resident', 'cuda_upload', 'numpy_local_pipeline', 'cuda_local_pipeline']]
        line = f'- {run["path"].name}, largest batch: resident {cpu/gpu:.2f}x; upload + compute {cpu/upload:.2f}x; local pipeline {cp/gp:.2f}x.'
        if (last, 'cuda_group_loop') in run['summary']:
            loop = milliseconds(run, largest, 'cuda_group_loop')
            line += f' Per-group CUDA launch/reduce/return loop {loop:.3f} ms versus batched {gpu:.3f} ms ({loop/gpu:.1f}x). Both reuse identical device inputs and return identical per-group metrics; the loop is a dispatch/return overhead diagnostic, not the CPU baseline.'
        text.append(line)
    text += ['', '## Numerical and timing validation', '',
             'Every returned metric of every warmup/measured invocation is checked against FP64 arithmetic on the same FP32 packed temperatures/weights (rtol=atol=1e-5). The report independently reloads the hashed data, recomputes all group metrics and validates group labels, shared-valid weights/counts, summary quantiles, repeat IDs, case sizes and recorded seeded within-repeat method order. Metric checking and result-file writing occur outside timed intervals.', '',
             'Each method uses five warmups and 20 measured repetitions in these sessions. Methods are shuffled within each repetition with recorded seeds; case order remains fixed. CPU/GPU compute includes returning all group metrics. CUDA context initialization, initial packing/upload for resident methods, hashing and remote acquisition are excluded. Warm local reads include OS/library caches; no cold-cache throughput claim. Shading shows the range of session medians, not a confidence interval. Memory entries in environment.json measure live PyTorch allocations and extra resident-method peak, not total process VRAM.', '',
             '## Forecast-minus-analysis errors', '',
             '| Lead h | Bias K | MAE K | RMSE K |', '|---|---:|---:|---:|']
    for lead in leads:
        text.append(f'| {lead} | '+' | '.join(f'{v:.6f}' for v in pooled[lead])+' |')
    text += ['', 'These values pool weighted error moments over individual member forecasts and initializations. RMSE is computed from weighted mean square error, not the average of member RMSE values. No model ranking or general forecast-skill estimate is justified by this short case.', '',
             '## Acquisition and reproduction', '',
             'Source manifests pin immutable read-only Icechunk snapshots. Both temperature archives are distributed in Celsius and processed by dynamical.org: [ECMWF IFS ENS](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) and [NOAA GFS](https://dynamical.org/catalog/noaa-gfs-forecast/). CC BY 4.0 applies; ECMWF Terms of Use also apply to the ECMWF-derived forecast. [Data manifest](../../data/ecmwf_batch_case/manifest.json) records attribution, hashes, exact query and observed remote timings. Nominal uncompressed source-chunk sizes are estimates, not measured HTTP bytes.', '',
             '```bash', 'python -m pip install -r requirements-zarr.txt',
             'python weather_batch.py --output results/my_batch01 --group-loop',
             'python weather_batch.py --output results/my_batch02 --group-loop --seed 20261102',
             'python scripts/analyze_weather_batch.py results/my_batch01 results/my_batch02 --output results/my_batch_comparison', '```', '',
             'The checked-in local Zarr subset can be benchmarked offline on the existing WSL CUDA environment. Remote recreation uses Python >=3.12 and requirements-icechunk.txt: `python scripts/prepare_ecmwf_batch.py --output data/ecmwf_batch_recreated`. Use `--data data/ecmwf_batch_recreated/batch.zarr` to evaluate that store. Metadata/catalog updates may change the manifest even at a fixed data snapshot; the data store directory hash validates the selected bytes.', '',
             '![Real ECMWF batch timing and errors](batch_scaling.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Saved comparison report and plot: {args.output}')


if __name__ == '__main__':
    main()
