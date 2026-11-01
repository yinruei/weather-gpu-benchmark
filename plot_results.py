"""Plot runtime, speedup, memory and transfer costs from a benchmark CSV."""
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
    if not rows:
        p.error('CSV must contain at least one result row')
    legacy = 'numpy_cpu_sec' in rows[0]
    suffix = '_sec' if legacy else '_median_sec'
    x = [float(r['points']) / 1e6 for r in rows]
    fig, grid = plt.subplots(2, 2, figsize=(12, 9))
    runtime, speedup, memory, transfer = grid.flat
    for key, label in [('numpy_cpu', 'NumPy CPU'), ('torch_cpu', 'PyTorch CPU'), ('torch_cuda', 'CUDA compute')]:
        if rows[0].get(key + suffix):
            runtime.plot(x, [float(r[key + suffix]) * 1000 for r in rows], 'o-', label=label)
    gpu = rows[0].get('torch_cuda' + suffix)
    if gpu:
        cpu = [float(r['numpy_cpu' + suffix]) for r in rows]
        compute = [float(r['torch_cuda' + suffix]) for r in rows]
        e2e = ([float(r['h2d_transfer_sec']) + c for r, c in zip(rows, compute)] if legacy
               else [float(r['gpu_e2e_median_sec']) for r in rows])
        runtime.plot(x, [t * 1000 for t in e2e], 'o--', label='H2D + compute estimate' if legacy else 'GPU measured E2E')
        speedup.plot(x, [c / g for c, g in zip(cpu, compute)], 'o-', label='Compute speedup')
        speedup.plot(x, [c / g for c, g in zip(cpu, e2e)], 'o-', label='Estimated speedup' if legacy else 'Measured E2E speedup')
        resident_key = 'gpu_resident_e2e_median_sec'
        if rows[0].get(resident_key):
            resident = [float(r[resident_key]) for r in rows]
            runtime.plot(x, [t * 1000 for t in resident], 'o-', label='Resident + scalar return')
            speedup.plot(x, [c / g for c, g in zip(cpu, resident)], 'o-', label='Resident + scalar return')
        speedup.axhline(1, color='gray', linestyle=':')
        speedup.legend(fontsize=8)
        memory_key = 'gpu_peak_mem_mb' if legacy else 'gpu_peak_allocated_mb'
        if rows[0].get(memory_key):
            memory.plot(x, [float(r[memory_key]) for r in rows], 'o-',
                        label='Legacy reported value' if legacy else 'CUDA compute peak allocated')
            memory.legend(fontsize=8)
        else:
            memory.text(.5, .5, 'No GPU memory measurement', ha='center', transform=memory.transAxes)
        transfer_key = 'h2d_transfer_sec' if legacy else 'h2d_median_sec'
        transfer.plot(x, [float(r[transfer_key])*1000 for r in rows], 'o-', label='Host to device')
        transfer.plot(x, [t*1000 for t in compute], 'o-', label='CUDA compute')
        transfer.legend(fontsize=8)
    else:
        for ax in (speedup, memory, transfer):
            ax.text(.5, .5, 'CUDA unavailable: no GPU measurements', ha='center', transform=ax.transAxes)
    runtime.set_title('Runtime vs size')
    runtime.set_yscale('log'); runtime.set_ylabel('Runtime (ms, log scale)')
    speedup.set_title('GPU speedup vs NumPy'); speedup.set_ylabel('Ratio of runtimes')
    memory.set_title('Reported GPU memory (protocol unknown)' if legacy else 'PyTorch peak allocated during CUDA compute')
    memory.set_ylabel('Legacy reported MB (definition unknown)' if legacy else 'Allocated memory (MiB, not total device usage)')
    memory.set_ylim(bottom=0)
    transfer.set_title('Compute vs transfer (separately measured)')
    transfer.set_ylabel('Runtime (ms)')
    transfer.set_ylim(bottom=0)
    runtime.legend(fontsize=8)
    for ax in grid.flat:
        ax.set_xlabel('Grid points (millions)'); ax.grid(alpha=.2)
    fig.suptitle('Historical RTX 3050 results (unverified timing protocol)' if legacy else 'Weather verification benchmark: medians')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
