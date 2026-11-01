"""Validate raw benchmark samples and regenerate a run report and IQR plot."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    env = json.loads((a.run / 'environment.json').read_text())
    with (a.run / 'summary.csv').open() as f:
        summary = list(csv.DictReader(f))
    with (a.run / 'samples.csv').open() as f:
        samples = list(csv.DictReader(f))
    methods = ['numpy_cpu', 'torch_cpu', 'torch_cuda', 'h2d', 'gpu_e2e']
    groups = defaultdict(list)
    for r in samples:
        groups[int(r['points']), r['method']].append((int(r['repeat']), float(r['seconds'])))
    expected = {(n, m) for n in env['sizes'] for m in methods}
    assert set(groups) == expected, 'Missing or extra timing groups'
    assert [int(r['points']) for r in summary] == env['sizes']
    lines = ['# RTX 3050 run01 analysis', '',
             'Source: the three user-uploaded files in this directory. Raw inputs are preserved unchanged.', '',
             f"Environment reported by the run: {env['gpu']}, Python {env['python']}, NumPy {env['numpy']}, PyTorch {env['torch']}, CUDA runtime {env['cuda_runtime']}, WSL2. PyTorch CPU threads: {env['torch_threads']}.", '',
             f"Protocol: FP32 synthetic data, {env['warmup']} warmups and {env['repeats']} measured repetitions per method and size. Timings use synchronized host wall time.", '',
             '| Points | NumPy ms | CUDA compute ms | H2D ms | GPU E2E ms | Compute speedup | E2E speedup |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    variations = ['', '## Timing variation', '', 'IQR/median = (p75 - p25) / median. It describes within-run variation, not a confidence interval.', '',
                  '| Points | Method | IQR / median | Minimum ms | Maximum ms |', '|---:|---|---:|---:|---:|']
    for r in summary:
        n = int(r['points'])
        for m in methods:
            pairs = groups[n, m]
            assert sorted(i for i, _ in pairs) == list(range(env['repeats']))
            v = np.array([v for _, v in pairs])
            assert np.isfinite(v).all() and (v > 0).all()
            for q, k in [(50, 'median'), (25, 'p25'), (75, 'p75')]:
                np.testing.assert_allclose(np.percentile(v, q), float(r[f'{m}_{k}_sec']), rtol=1e-12)
            variations.append(f'| {n:,} | {m} | {(np.percentile(v,75)-np.percentile(v,25))/np.median(v):.2%} | {min(v)*1000:.3f} | {max(v)*1000:.3f} |')
        for key, den in [('compute', 'torch_cuda'), ('e2e', 'gpu_e2e')]:
            np.testing.assert_allclose(float(r[f'gpu_{key}_speedup']), float(r['numpy_cpu_median_sec'])/float(r[f'{den}_median_sec']), rtol=1e-12)
        times = [float(r[f'{m}_median_sec'])*1000 for m in ['numpy_cpu', 'torch_cuda', 'h2d', 'gpu_e2e']]
        lines.append(f"| {n:,} | " + ' | '.join(f'{v:.3f}' for v in times) + f" | {float(r['gpu_compute_speedup']):.2f}x | {float(r['gpu_e2e_speedup']):.2f}x |")
    lines += ['', f'Validation passed: {len(samples)} positive finite timing samples; all group counts, repeat IDs, medians, quartiles and speedups reconcile.', '',
              '## Interpretation', '',
              '- At 1M points GPU E2E is slower than NumPy. GPU E2E is faster at the tested 5M and 10M sizes. No exact crossover threshold was measured.',
              '- At 10M, H2D takes 11.330 ms while GPU compute takes 2.038 ms. Input transfer is the largest measured component. These are separately sampled medians, not an additive decomposition of E2E latency.',
              '- CUDA compute IQR/median is 2.61%–4.14%; PyTorch CPU is 18.42%–36.83%. CPU scheduling, clocks and background load are possible factors, not diagnosed causes.',
              '- The PyTorch CPU median rises from 11.274 ms at 5M to 36.927 ms at 10M. The present data cannot establish the cause. Do not generalize this CPU comparison to all thread settings.',
              '- No timing samples were removed. At 5M CUDA compute has a 2.042 ms maximum versus a 1.074 ms median; the chart shows the middle 50%, not all samples.',
              '- E2E includes two input copies, metric computation and scalar return, with preallocated GPU input buffers. It excludes disk I/O and preprocessing.',
              '- These files do not contain per-backend metric values or correctness logs. The runner checks agreement before saving files; independent numerical correctness cannot be rechecked from timing samples alone.',
              '- This is one user-supplied run, not an independently reproduced GPU experiment. Driver version, exact CPU model, power mode and background load are missing. Software versions are reported as supplied.', '',
              '## Next experiment', '',
              'Repeat the same protocol in a separate output folder to assess run-to-run variation. Then use profiling to test whether keeping model output on GPU avoids the observed transfer cost. Change one condition at a time.', '',
              'Regenerate: `python scripts/analyze_run.py results/rtx3050_run01`', '',
              '![Runtime medians and IQR](benchmark.png)'] + variations
    (a.run / 'analysis.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.array([int(r['points'])/1e6 for r in summary])
    for m, label in zip(methods, ['NumPy CPU', 'PyTorch CPU (1 thread)', 'CUDA compute', 'Host to device', 'GPU E2E']):
        med = np.array([float(r[f'{m}_median_sec'])*1000 for r in summary])
        low = np.array([float(r[f'{m}_p25_sec'])*1000 for r in summary])
        high = np.array([float(r[f'{m}_p75_sec'])*1000 for r in summary])
        axes[0].errorbar(x, med, yerr=[med-low, high-med], marker='o', capsize=4, label=label)
    for m, label in [('compute','Compute'),('e2e','Measured E2E')]:
        axes[1].plot(x, [float(r[f'gpu_{m}_speedup']) for r in summary], 'o-', label=label)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Median runtime (ms, log scale)')
    axes[1].set_ylabel('Ratio of medians vs NumPy'); axes[1].axhline(1, color='gray', linestyle=':')
    for ax in axes:
        ax.set_xlabel('Grid points (millions)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('RTX 3050: 20 repeats, 5 warmups (error bars: p25 to p75)')
    fig.tight_layout(); fig.savefig(a.run/'benchmark.png', dpi=180); plt.close(fig)
    print(f'PASS: {len(samples)} samples reconciled; report and plot regenerated')

if __name__ == '__main__':
    main()
