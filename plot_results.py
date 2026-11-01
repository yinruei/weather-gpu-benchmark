"""Plot historical or current benchmark CSV without pandas."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/benchmark.png'))
    a = p.parse_args()
    with a.csv.open() as f:
        rows = list(csv.DictReader(f))
    legacy = 'numpy_cpu_sec' in rows[0]
    suffix = '_sec' if legacy else '_median_sec'
    x = [float(r['points']) / 1e6 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in [('numpy_cpu', 'NumPy CPU'), ('torch_cpu', 'PyTorch CPU'), ('torch_cuda', 'CUDA compute')]:
        if rows[0].get(key + suffix):
            axes[0].plot(x, [float(r[key + suffix]) * 1000 for r in rows], 'o-', label=label)
    gpu = rows[0].get('torch_cuda' + suffix)
    if gpu:
        cpu = [float(r['numpy_cpu' + suffix]) for r in rows]
        compute = [float(r['torch_cuda' + suffix]) for r in rows]
        e2e = ([float(r['h2d_transfer_sec']) + c for r, c in zip(rows, compute)] if legacy
               else [float(r['gpu_e2e_median_sec']) for r in rows])
        axes[0].plot(x, [t * 1000 for t in e2e], 'o--', label='H2D + compute estimate' if legacy else 'GPU measured E2E')
        axes[1].plot(x, [c / g for c, g in zip(cpu, compute)], 'o-', label='Compute speedup')
        axes[1].plot(x, [c / g for c, g in zip(cpu, e2e)], 'o-', label='Estimated speedup' if legacy else 'Measured E2E speedup')
        axes[1].axhline(1, color='gray', linestyle=':')
        axes[1].legend()
    else:
        axes[1].text(.5, .5, 'CUDA unavailable: no GPU measurements', ha='center', transform=axes[1].transAxes)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Runtime (ms, log scale)')
    axes[1].set_ylabel('Speedup vs NumPy')
    axes[0].legend()
    for ax in axes:
        ax.set_xlabel('Grid points (millions)'); ax.grid(alpha=.2)
    fig.suptitle('Historical RTX 3050 results (unverified timing protocol)' if legacy else 'Weather verification benchmark')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
