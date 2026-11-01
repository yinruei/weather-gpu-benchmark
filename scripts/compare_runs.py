"""Validate two full CPU/CUDA runs and compare their recorded timings.

Usage: python scripts/compare_runs.py results/rtx3050_run01 results/rtx3050_run02
Writes comparison.md and comparison.png into the second run directory.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ['numpy_cpu', 'torch_cpu', 'torch_cuda', 'h2d', 'gpu_e2e']
LABELS = ['NumPy CPU', 'PyTorch CPU', 'CUDA compute', 'Host to device', 'GPU E2E']


def load_run(path):
    env = json.loads((path / 'environment.json').read_text(encoding='utf-8'))
    with (path / 'summary.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with (path / 'samples.csv').open(newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in raw:
        groups[int(row['points']), row['method']].append(
            (int(row['repeat']), float(row['seconds'])))
    expected = {(n, m) for n in env['sizes'] for m in METHODS}
    if set(groups) != expected or [int(r['points']) for r in rows] != env['sizes']:
        raise ValueError(f'{path}: missing, duplicate or extra sizes/methods')
    for row in rows:
        for method in METHODS:
            pairs = groups[int(row['points']), method]
            if sorted(i for i, _ in pairs) != list(range(env['repeats'])):
                raise ValueError(f'{path}: invalid repeat IDs/counts')
            values = np.array([v for _, v in pairs])
            if not (np.isfinite(values).all() and (values > 0).all()):
                raise ValueError(f'{path}: timing samples must be positive and finite')
            for q, key in [(50, 'median'), (25, 'p25'), (75, 'p75')]:
                np.testing.assert_allclose(
                    np.percentile(values, q), float(row[f'{method}_{key}_sec']),
                    rtol=1e-12, atol=0, err_msg=f'{path}: {method} {key}')
        for key, method in [('compute', 'torch_cuda'), ('e2e', 'gpu_e2e')]:
            np.testing.assert_allclose(
                float(row[f'gpu_{key}_speedup']),
                float(row['numpy_cpu_median_sec']) / float(row[f'{method}_median_sec']),
                rtol=1e-12, atol=0)
    return env, rows, len(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('repeat', type=Path)
    args = parser.parse_args()
    first, second = load_run(args.baseline), load_run(args.repeat)
    if first[0] != second[0]:
        raise ValueError('Recorded environment/protocol differs; review before comparison')
    env = first[0]
    lines = [f'# {args.baseline.name} vs {args.repeat.name}', '',
             'Sources: unchanged environment.json, samples.csv and summary.csv in each run directory.', '',
             f'Validation passed: {first[2]} baseline and {second[2]} repeat timing samples. '
             'All timing groups, repeat IDs, positive finite values, medians, quartiles and speedups reconcile. '
             'No samples were removed.', '',
             'Recorded environment and protocol are identical across runs: '
             f"{env['gpu']}, Python {env['python']}, NumPy {env['numpy']}, "
             f"PyTorch {env['torch']}, CUDA runtime {env['cuda_runtime']}, "
             f"{env['torch_threads']} PyTorch CPU thread(s), {env['dtype']}, seed {env['seed']}, "
             f"{env['warmup']} warmups and {env['repeats']} measured repeats per group.", '',
             '## Runtime and within-run variation', '',
             'Change = (repeat median / baseline median - 1). Negative means faster. '
             'IQR/median = (p75 - p25) / median; this measures within-run spread, not a confidence interval.', '',
             '| Points | Method | Baseline ms | Repeat ms | Change | Baseline IQR/median | Repeat IQR/median |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for a, b in zip(first[1], second[1]):
        for method, label in zip(METHODS, LABELS):
            med = [float(r[f'{method}_median_sec']) for r in (a, b)]
            spread = [(float(r[f'{method}_p75_sec']) - float(r[f'{method}_p25_sec'])) / m
                      for r, m in zip((a, b), med)]
            lines.append(f"| {int(a['points']):,} | {label} | {med[0]*1000:.3f} | "
                         f'{med[1]*1000:.3f} | {med[1]/med[0]-1:+.2%} | '
                         f'{spread[0]:.2%} | {spread[1]:.2%} |')
    lines += ['', '## Speedup relative to each run\'s NumPy baseline', '',
              '| Points | Baseline compute | Repeat compute | Baseline E2E | Repeat E2E |',
              '|---:|---:|---:|---:|---:|']
    for a, b in zip(first[1], second[1]):
        values = [float(r[f'gpu_{key}_speedup']) for key in ['compute', 'e2e'] for r in (a, b)]
        lines.append(f"| {int(a['points']):,} | " + ' | '.join(f'{v:.2f}x' for v in values) + ' |')
    same_metrics = all(a[k] == b[k] for a, b in zip(first[1], second[1])
                       for k in ['bias', 'mae', 'rmse'])
    lines += ['', '## Interpretation and limits', '',
              f'- NumPy reference Bias/MAE/RMSE values are identical across runs: {same_metrics}. '
              'Per-backend metric values are not saved, so timing files cannot independently validate CUDA numerical agreement.',
              '- Inspect absolute CPU and GPU times alongside speedup: a slower NumPy baseline raises speedup even when GPU time does not improve.',
              '- Two user-supplied runs provide an initial repeatability check, not enough independent sessions to establish long-term stability or statistical significance.',
              '- Identical recorded metadata does not establish identical system load, clocks or thermals. Exact CPU model, driver, power mode and background load were not recorded.',
              '- H2D and compute are separately sampled medians. Their sum is not a measured decomposition of GPU E2E.', '',
              '## Next experiment', '',
              'Profile a GPU-resident workflow against the current E2E boundary. CUDA compute already uses resident inputs; '
              'a new experiment should explicitly measure a repeated resident-input workflow including the required scalar return, '
              'and account for the initial upload separately. Keep sizes, seed, threads, warmups and repeats fixed. '
              'Record driver, CPU and system-load context before the run.', '',
              'Regenerate: `python scripts/compare_runs.py '
              f'{args.baseline.as_posix()} {args.repeat.as_posix()}`', '',
              '![Runtime medians and interquartile ranges](comparison.png)', '']
    fig, axes = plt.subplots(1, len(METHODS), figsize=(17, 4.3), squeeze=False)
    x = np.arange(len(env['sizes']))
    for ax, method, label in zip(axes[0], METHODS, LABELS):
        for offset, (path, run) in zip([-.16, .16], [(args.baseline, first), (args.repeat, second)]):
            med = np.array([float(r[f'{method}_median_sec'])*1000 for r in run[1]])
            low = np.array([float(r[f'{method}_p25_sec'])*1000 for r in run[1]])
            high = np.array([float(r[f'{method}_p75_sec'])*1000 for r in run[1]])
            ax.bar(x+offset, med, width=.3, yerr=[med-low, high-med], capsize=3, label=path.name)
        ax.set_title(label)
        ax.set_xticks(x, [f'{n/1e6:g}M' for n in env['sizes']])
        ax.set_xlabel('Grid points')
        ax.set_ylabel('Median runtime (ms)')
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('RTX 3050 repeated runs (error bars: p25 to p75; separate y scales)')
    fig.tight_layout()
    fig.savefig(args.repeat / 'comparison.png', dpi=180)
    plt.close(fig)
    (args.repeat / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'PASS: {first[2] + second[2]} samples reconciled; comparison report and chart saved')


if __name__ == '__main__':
    main()
