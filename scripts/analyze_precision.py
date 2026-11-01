"""Validate recorded precision timings/metrics and generate an accuracy tradeoff report."""
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
from precision_benchmark import METRICS, MODES, reference_metrics


LABELS = dict(fp32='FP32', fp16_native='FP16 native', fp16_fp32_math='FP16 inputs / FP32 math')


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid boolean: {value}')
    return value == 'True'


def validate_run(path):
    env = json.loads((path / 'environment.json').read_text(encoding='utf-8'))
    with (path / 'summary.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with (path / 'samples.csv').open(newline='', encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    expected = {(n, m) for n in env['sizes'] for m in MODES}
    if len(rows) != len(expected) or {(int(r['points']), r['mode']) for r in rows} != expected:
        raise ValueError('Missing or duplicate summary groups')
    groups = defaultdict(list)
    for r in raw:
        groups[int(r['points']), r['mode']].append(r)
    if set(groups) != expected:
        raise ValueError('Missing or extra raw timing groups')
    refs = {}
    for n in env['sizes']:
        rng = np.random.default_rng(env['seed'])
        obs = rng.normal(280, 10, n).astype(np.float32)
        fcst = obs + rng.normal(.5, 2, n).astype(np.float32)
        refs[n] = (reference_metrics(fcst, obs),
                   reference_metrics(fcst.astype(np.float16), obs.astype(np.float16)))
    for row in rows:
        n, mode = int(row['points']), row['mode']
        group = groups[n, mode]
        if sorted(int(r['repeat']) for r in group) != list(range(env['repeats'])):
            raise ValueError('Invalid repeat IDs or count')
        times = np.array([float(r['seconds']) for r in group])
        if not (np.isfinite(times).all() and (times > 0).all()):
            raise ValueError('Timings must be positive and finite')
        for q, key in [(25, 'p25'), (50, 'median'), (75, 'p75')]:
            np.testing.assert_allclose(float(row[f'{key}_sec']), np.percentile(times, q), rtol=1e-12, atol=0)
        values = np.array([[float(r[k]) if r[k] else np.nan for k in METRICS] for r in group])
        for r, value in zip(group, values):
            if boolean(r['finite']) != bool(np.isfinite(value).all()):
                raise ValueError('Raw finite flag disagrees with saved metric values')
        finite = bool(np.isfinite(values).all())
        source, half = refs[n]
        input_ref = source if mode == 'fp32' else half
        if boolean(row['finite']) != finite:
            raise ValueError('Summary finite flag mismatch')
        for flag, ref in [('agrees_with_source', source), ('agrees_with_quantized_reference', input_ref)]:
            ok = finite and bool(np.allclose(values, ref, rtol=env['tolerance']['rtol'],
                                             atol=env['tolerance']['atol']))
            if boolean(row[flag]) != ok:
                raise ValueError(f'{flag} mismatch')
        np.testing.assert_allclose(float(row['input_mib']), n*2*(4 if mode == 'fp32' else 2)/2**20, rtol=1e-12, atol=0)
        peak = float(row['peak_allocated_mib'])
        if not np.isfinite(peak) or peak < float(row['input_mib']):
            raise ValueError('Peak memory is invalid or smaller than active inputs')
        for i, metric in enumerate(METRICS):
            checks = {f'{metric}_reference_fp64': source[i],
                      f'{metric}_input_reference_fp64': input_ref[i],
                      f'{metric}_quantization_delta': input_ref[i]-source[i]}
            if np.isfinite(values[:, i]).all():
                checks[metric] = np.median(values[:, i])
            if finite:
                checks[f'{metric}_max_abs_error'] = np.max(np.abs(values[:, i]-source[i]))
                checks[f'{metric}_max_abs_arithmetic_error'] = np.max(np.abs(values[:, i]-input_ref[i]))
            for key, value in checks.items():
                # Error fields subtract nearly equal FP64 values; tolerate a few
                # cross-platform FP64 ulps without relaxing timing validation.
                np.testing.assert_allclose(float(row[key]), value, rtol=1e-12, atol=1e-14,
                                           err_msg=f'{n} {mode} {key}')
    return env, rows, len(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    env, rows, count = validate_run(args.run)
    by_key = {(int(r['points']), r['mode']): r for r in rows}
    lines = [f'# Precision tradeoff: {args.run.name}', '',
             'Sources: unchanged environment.json, summary.csv and samples.csv in this directory.', '',
             f"Validated {count} timing samples and their saved per-repeat Bias/MAE/RMSE values. "
             'Timing groups, repeat IDs, medians, quartiles, metric summaries, error fields and agreement flags reconcile. '
             'FP64 source and quantized-input references were regenerated from the recorded seed. '
             'Metric-reference reconciliation allows 1e-14 absolute difference for cross-platform FP64 rounding. No samples were removed.', '',
             f"Protocol: GPU-resident synthetic inputs, seed {env['seed']}, {env['warmup']} warmups, "
             f"{env['repeats']} measured repeats per size/mode. Synchronized wall time includes returning three host scalars. "
             'Input generation, input conversion to FP16, upload and validation are outside timing. '
             'The mixed mode casts to FP32 before subtraction inside each timed call.', '',
             f"Hardware: {env['before']['cpu_model']}; {env['before']['gpu']}; "
             f"PyTorch {env['torch']}, CUDA runtime {env['cuda_runtime']}. "
             'Pre/post GPU state snapshots and the seeded per-size mode order are saved in environment.json. '
             'These snapshots do not establish clocks/load during each sample.', '',
             '## Runtime and allocated memory', '',
             '| Points | Mode | Median ms | IQR/median | FP32 / mode runtime | Input MiB | Peak allocated MiB | Finite |',
             '|---:|---|---:|---:|---:|---:|---:|---|']
    for n in env['sizes']:
        base = float(by_key[n, 'fp32']['median_sec'])
        for mode in MODES:
            r = by_key[n, mode]
            med = float(r['median_sec'])
            iqr = (float(r['p75_sec'])-float(r['p25_sec']))/med
            ratio = f'{base/med:.2f}x' if boolean(r['finite']) else 'invalid output'
            lines.append(f"| {n:,} | {LABELS[mode]} | {med*1000:.3f} | {iqr:.2%} | {ratio} | "
                         f"{float(r['input_mib']):.3f} | {float(r['peak_allocated_mib']):.3f} | {r['finite']} |")
    lines += ['', 'Memory is CUDA peak allocated minus the pre-input allocation baseline. It includes active input tensors '
              'and intermediate tensors; only one precision mode is resident at a time. It is not reserved memory or whole-device VRAM. '
              'FP16 input storage is exactly half of FP32; working-memory peaks need not halve.', '',
              '## Metric accuracy', '',
              'Errors below are the maximum absolute error across measured repetitions against FP64 arithmetic on the original '
              'FP32 source inputs. They use synthetic input units. The diagnostic threshold rtol=atol=1e-5 is not a meteorological acceptance criterion.', '',
              '| Points | Mode | Bias abs error | MAE abs error | RMSE abs error | Source agreement | Input-reference agreement |',
              '|---:|---|---:|---:|---:|---|---|']
    for n in env['sizes']:
        for mode in MODES:
            r = by_key[n, mode]
            errors = ' | '.join(f'{float(r[k+"_max_abs_error"]):.6g}' if r[k+'_max_abs_error'] else 'nonfinite' for k in METRICS)
            lines.append(f"| {n:,} | {LABELS[mode]} | {errors} | {r['agrees_with_source']} | {r['agrees_with_quantized_reference']} |")
    largest = max(env['sizes'])
    lines += ['', f'## Error decomposition at {largest:,} points', '',
              'Source reference: FP64 arithmetic on the original FP32 input values. Input reference: FP64 arithmetic on '
              'the values actually stored for a mode. For FP16 modes, the difference between these references isolates input quantization.', '',
              '| Mode | RMSE | Source RMSE | Signed input quantization delta | Signed arithmetic delta |',
              '|---|---:|---:|---:|---:|']
    for mode in MODES:
        r = by_key[largest, mode]
        if boolean(r['finite']):
            lines.append(f"| {LABELS[mode]} | {float(r['rmse']):.9f} | {float(r['rmse_reference_fp64']):.9f} | "
                         f"{float(r['rmse_quantization_delta']):+.6g} | "
                         f"{float(r['rmse'])-float(r['rmse_input_reference_fp64']):+.6g} |")
    lines += ['', 'Signed quantization delta + signed arithmetic delta = signed total metric error. Absolute errors do not '
              'necessarily add: native FP16 arithmetic can partially cancel quantization error. A smaller total error in one synthetic '
              'case does not establish a more reliable algorithm.', '',
              '## Interpretation and limits', '',
              '- Compare speed, allocated memory and all three errors together. A runtime ratio above one is not evidence of acceptable accuracy.',
              '- FP32 arithmetic on FP16 inputs can reduce arithmetic error but cannot recover information lost during input conversion. '
              'Materializing FP32 working tensors also costs time and memory. This eager implementation is not a fused mixed-precision kernel.',
              '- Native FP16 can overflow in intermediate subtraction or squaring even when input values are representable. '
              'The unit tests explicitly exercise this failure. Nonfinite outputs are flagged and receive no valid speedup claim.',
              '- Backend-internal reduction accumulation is not assumed to be FP16; native intermediate and output tensor dtypes are FP16.',
              '- Each size uses a seeded mode order and each mode is timed as one block. Runs with the same seed repeat the same order. '
              'This does not fully control order, cache, clock or background-load effects. Do not infer significance from these short sessions.',
              '- No application error budget or real weather verification has been established. Keep FP32 as the baseline pending real-data validation.', '',
              'Regenerate: `python scripts/analyze_precision.py ' + args.run.as_posix() + '`', '',
              '![Precision runtime, memory and accuracy tradeoff](precision.png)', '']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.array(env['sizes']) / 1e6
    for mode in MODES:
        subset = [by_key[n, mode] for n in env['sizes']]
        med = np.array([float(r['median_sec'])*1000 for r in subset])
        low = np.array([float(r['p25_sec'])*1000 for r in subset])
        high = np.array([float(r['p75_sec'])*1000 for r in subset])
        axes[0, 0].errorbar(x, med, yerr=[med-low, high-med], marker='o', capsize=3, label=LABELS[mode])
        axes[0, 1].plot(x, [float(r['peak_allocated_mib']) for r in subset], 'o-', label=LABELS[mode])
    for j, mode in enumerate(MODES):
        r = by_key[largest, mode]
        offset = (j-1)*.25
        errors = [float(r[k+'_max_abs_error']) if r[k+'_max_abs_error'] else np.nan for k in METRICS]
        axes[1, 0].bar(np.arange(3)+offset, errors, width=.25, label=LABELS[mode])
    positions = np.arange(len(MODES))
    quant, arithmetic, total = [], [], []
    for mode in MODES:
        r = by_key[largest, mode]
        quant.append(float(r['rmse_quantization_delta']))
        arithmetic.append(float(r['rmse'])-float(r['rmse_input_reference_fp64']) if r['rmse'] else np.nan)
        total.append(float(r['rmse'])-float(r['rmse_reference_fp64']) if r['rmse'] else np.nan)
    for offset, v, label in [(-.25, quant, 'Input quantization'), (0, arithmetic, 'Arithmetic'), (.25, total, 'Total')]:
        axes[1, 1].bar(positions+offset, v, width=.25, label=label)
    axes[0, 0].set_title('Resident runtime including scalar return (p25 to p75)')
    axes[0, 0].set_ylabel('Median runtime (ms)')
    axes[0, 1].set_title('Peak allocated memory (active mode only)')
    axes[0, 1].set_ylabel('Allocated memory (MiB)')
    for ax in axes[0]:
        ax.set_xlabel('Grid points (millions)')
        ax.set_ylim(bottom=0)
    axes[1, 0].set_title(f'Max absolute metric error vs source FP64 ({largest/1e6:g}M)')
    axes[1, 0].set_xticks(np.arange(3), ['Bias', 'MAE', 'RMSE'])
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel('Absolute error (synthetic input units, log)')
    axes[1, 1].set_title(f'Signed RMSE error decomposition ({largest/1e6:g}M)')
    axes[1, 1].set_xticks(positions, ['FP32', 'FP16 native', 'FP16 / FP32 math'])
    axes[1, 1].set_ylabel('Signed RMSE error (synthetic input units)')
    axes[1, 1].axhline(0, color='gray', linewidth=.7)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('RTX 3050: FP32 / FP16 speed, memory and accuracy tradeoff')
    fig.tight_layout()
    fig.savefig(args.run / 'precision.png', dpi=180)
    plt.close(fig)
    (args.run / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'PASS: {count} timings and per-repeat metric values validated; report and figure saved')


if __name__ == '__main__':
    main()
