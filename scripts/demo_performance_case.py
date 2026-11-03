"""Validate archived public evidence and rebuild a CPU-only presentation bundle."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = [f'results/rtx3050_physics_gather_pipeline_run0{i}' for i in (1, 2)]
READERS = [f'results/rtx3050_physics_gather_run0{i}' for i in (1, 2)]
CAPTURES = [f'results/rtx3050_physics_nsight_run0{i}' for i in (1, 2, 3)]


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def p50(runs, field, methods):
    result = {method: [] for method in methods}
    for run in runs:
        records = rows(ROOT/run/'timing_summary.csv')
        for method in methods:
            selected = [r for r in records if r['case'] == 'b64_s64' and r[field] == method]
            if len(selected) != 1:
                raise ValueError(f'Expected one validated B64 summary: {run}/{method}')
            result[method].append(float(selected[0]['p50_sec'])*1000)
    return result


def summarize():
    sys.path.insert(0, str(ROOT))
    from nsight_trace import summarize_events

    pairs = []
    for run in PIPELINES:
        records = rows(ROOT/run/'timing_samples.csv')
        matched = []
        for method in ['graph_open_once', 'graph_gather_once']:
            matched.append({(r['repeat'], r['variant']): float(r['seconds']) for r in records
                            if r['case'] == 'b64_s64' and r['method'] == method})
        left, right = matched
        if not left or left.keys() != right.keys():
            raise ValueError('Unmatched graph bounding/gather pairs')
        pairs.append(dict(run=run, median_ratio=statistics.median(left[k]/right[k] for k in left),
                          gather_faster=sum(left[k] > right[k] for k in left), pairs=len(left)))
    captures = []
    for run in CAPTURES:
        calls, complete, _, _ = summarize_events(ROOT/run/'events.sqlite')
        captures.append(dict(run=run, complete_gpu_evidence=complete, requests=len(calls),
                             kernel_events=sum(c['kernel_events'] for c in calls)))
    return dict(mode='CPU validation of archived outputs and traces; no new GPU timings',
                source_runs=dict(pipelines=PIPELINES, readers=READERS, captures=CAPTURES),
                primary_counts=dict(read_only=sum(len(rows(ROOT/r/'timing_samples.csv')) for r in READERS),
                                    complete_file_requests=sum(len(rows(ROOT/r/'timing_samples.csv')) for r in PIPELINES)),
                b64_read_only_p50_ms=p50(READERS, 'reader', ['open_once', 'exact_once', 'gather_once']),
                b64_complete_p50_ms=p50(PIPELINES, 'method', [e+'_'+r for e in ['cpu', 'eager', 'graph']
                                                           for r in ['open_once', 'gather_once']]),
                b64_graph_bounding_over_gather_pairs=pairs, nsight_captures=captures)


def overview(summary):
    text = ['# Validated performance demonstration', '', summary['mode']+'.', '',
            'The fixed FNO is untrained. All times below come from the archived RTX 3050 sessions.', '',
            f'Validated primary observations: {summary["primary_counts"]["read_only"]} read-only; '
            f'{summary["primary_counts"]["complete_file_requests"]} complete-file requests.', '',
            '## B64 reused-handle P50 (two sessions)', '',
            '| Scope | Method | Session 1 ms | Session 2 ms |', '|---|---|---:|---:|']
    for scope, key in [('Read only', 'b64_read_only_p50_ms'), ('Complete request', 'b64_complete_p50_ms')]:
        for method, values in summary[key].items():
            text.append(f'| {scope} | {method} | {values[0]:.3f} | {values[1]:.3f} |')
    text += ['', '## Paired graph comparison', '']
    for pair in summary['b64_graph_bounding_over_gather_pairs']:
        text.append(f'- {Path(pair["run"]).name}: median bounding/gather {pair["median_ratio"]:.3f}x; '
                    f'gather faster in {pair["gather_faster"]}/{pair["pairs"]} matched pairs.')
    text += ['', 'Ratios are descriptive; no consistent general workflow gain is established. Bounding remains the default.', '',
             '## Nsight status', '', '| Capture | Requests | Kernels | Status |', '|---|---:|---:|---|']
    for capture in summary['nsight_captures']:
        status = 'Complete GPU evidence' if capture['complete_gpu_evidence'] else 'PARTIAL: not GPU activity evidence'
        text.append(f'| {Path(capture["run"]).name} | {capture["requests"]} | {capture["kernel_events"]} | {status} |')
    text += ['', 'The return CPU range can wait for preceding GPU work. Diagnostic ranges do not replace primary P50, '
             'measure occupancy, or establish PCIe-only cost.', '',
             '[Full reader report](gather/analysis.md) · [Full Nsight report](nsight/analysis.md) · [Summary JSON](summary.json)', '',
             '![Reader comparison](gather/physicsnemo_gather.png)', '',
             '![Nsight timeline](nsight/physicsnemo_nsight.png)', '']
    return '\n'.join(text)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New directory for regenerated reports and summary')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory; existing files are preserved')
    commands = [
        ['scripts/analyze_physicsnemo_gather.py', *PIPELINES, '--reader-runs', *READERS, '--output', str(output/'gather')],
        ['scripts/analyze_physicsnemo_nsight.py', *CAPTURES, '--output', str(output/'nsight')],
    ]
    output.mkdir(parents=True)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    print('Archived-evidence demonstration: CPU validation, not a new GPU benchmark.', flush=True)
    for index, command in enumerate(commands, 1):
        print(f'[{index}/2] Running {command[0]}', flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment)
        if result.returncode:
            print('Validation failed. No summary was published; retained reports may be incomplete.', file=sys.stderr)
            return result.returncode
    summary = summarize()
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'overview.md').write_text(overview(summary), encoding='utf-8')
    print(f'PASS: both evidence validators completed. Open {output/"overview.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
