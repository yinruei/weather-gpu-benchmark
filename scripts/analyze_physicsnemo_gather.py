"""Compare bounding, vectorized exact and basic-slice/NumPy-gather readers."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_graph import CASES
from physicsnemo_reader import READER_POLICIES
from scripts.analyze_physicsnemo_graph import validate as validate_reference
from scripts.analyze_physicsnemo_graph_pipeline import validate as validate_pipeline
from scripts.analyze_physicsnemo_reader import draw_bars, paired, validate_reads

READERS = ['open_once', 'exact_once', 'gather_once', 'reopen', 'exact_reopen', 'gather_reopen']
GROUPS = [('reuse', ['open_once', 'exact_once', 'gather_once']), ('reopen', ['reopen', 'exact_reopen', 'gather_reopen'])]
SELECTIONS = ['bounding', 'exact', 'gather']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline_runs', nargs='+', type=Path)
    parser.add_argument('--reader-runs', nargs='+', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--recompute-reference', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = validate_reference(args.reference_run, args.data, args.baseline, args.recompute_reference)
    pipes = [validate_pipeline(p, args.data, args.reference_run, reference) for p in args.pipeline_runs]
    reads = [validate_reads(p, args.data, args.reference_run, reference) for p in args.reader_runs]
    if any(set(r['env']['readers']) != set(READERS) or len(r['env']['readers']) != 6 for r in pipes+reads) or any(r['env']['engines'] != ['cpu', 'eager', 'graph'] for r in pipes):
        raise ValueError('All six readers and all three pipeline engines required')
    coverage = reads[0]['env']['coverage']
    if any(r['env']['coverage'] != coverage for r in reads) or any(r['env']['read_coverage'] != coverage for r in pipes):
        raise ValueError('Compared sessions need identical source coverage')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for ax, c in zip(axes[0], CASES):
        values = [[[float(r['summary'][(c['case'], group[i])]['p50_sec'])*1000 for label, group in GROUPS] for r in reads] for i in range(3)]
        draw_bars(ax, ['Reuse handle', 'Reopen'], values, ['Bounding', 'Exact', 'Slice + gather'], 'Read-only P50 ms', f'B{c["batch"]}: independent read/select/decode/check')
    values = [[[float(r['summary'][('b64_s64', engine+'_'+reader)]['p50_sec'])*1000 for engine in ['cpu', 'eager', 'graph']] for r in pipes] for reader in GROUPS[0][1]]
    draw_bars(axes[1, 0], ['CPU', 'EAGER', 'GRAPH'], values, ['Bounding', 'Exact', 'Slice + gather'], 'Complete file-request P50 ms', 'B64 reused handle: actual read through NPZ close')
    labels, source, copied = [], [], []
    for v in [0, 1]:
        for selection in SELECTIONS:
            b = coverage['b64_s64'][selection][v]
            labels.append('AB'[v]+'\n'+selection)
            source.append(b['materialized_raw_bytes']/2**20)
            copied.append((b['returned_backing_bytes']/2**20) if selection == 'gather' else 0)
    axes[1, 1].bar(range(6), source, label='Selected forecast array')
    axes[1, 1].bar(range(6), copied, bottom=source, label='Extra gather output copy')
    axes[1, 1].set_xticks(range(6), labels)
    axes[1, 1].set(ylabel='Logical simultaneous forecast arrays (MiB)', title='B64: slice + gather also allocates a returned copy')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis='y', alpha=.2)
    axes[1, 1].set_axisbelow(True)
    fig.suptitle('Basic-slice reader + NumPy gather versus original bounding and vectorized exact selection\nSame genuine ECMWF A/B frames, chunks, untrained FNO state and full output; session-range whiskers, not confidence intervals', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_gather.png', dpi=160)
    plt.close(fig)
    text = ['# Basic slice plus NumPy gather: bounded reader experiment', '',
        'gather_once/gather_reopen derive the minimal rectangular init/lead/member bounding box of the requested genuine flattened A/B frames. Exactly one basic-slice forecast load is followed by paired NumPy indexing, which copies B frames in canonical order. The xarray/Zarr backend never receives paired vectorized indexers on this path. No rechunking, prepared-input/output cache, interpolation or synthetic frames are used. Original CLI defaults remain bounding.', '',
        '## Forecast array and chunk accounting', '',
        '| Batch | Variant | Selection | Selected frames | Selected array bytes | Returned backing bytes | Source + gather arrays bytes | Store chunks | Store compressed bytes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for c in CASES:
        for v in [0, 1]:
            for s in SELECTIONS:
                b = coverage[c['case']][s][v]
                backing = b.get('returned_backing_bytes', b['materialized_raw_bytes'])
                total = b.get('slice_plus_gather_bytes', b['materialized_raw_bytes'])
                text.append(f'| {c["batch"]} | {"AB"[v]} | {s} | {b["materialized_frames"]} | {b["materialized_raw_bytes"]} | {backing} | {total} | {b["chunks_touched"]} | {b["stored_chunk_bytes"]} |')
    text += ['', 'The gather output is a new B-frame copy, not a view retaining the source box. B64 A/B loads 102/102 frames rather than 102/153, then copies 64; therefore a smaller returned backing does not establish lower peak RAM. The selected box and gathered output coexist during indexing. These logical arrays exclude decoder/indexer temporaries, packing and other process memory. Separate instrumented Store probes verify keys/get counts/compressed bytes and ultimate returned backing; they are not primary timing or physical disk/network traffic. All B16/B64 A/B paths still fetch four full forecast chunks, totaling 833,260 compressed bytes and 3,342,336 nominal uncompressed chunk bytes.', '',
        '## Read-only primary P50/P95/max', '', '| Session | Batch | Reader | P50 ms | P95 ms | Max ms |', '|---|---:|---|---:|---:|---:|']
    for r in reads:
        for c in CASES:
            for reader in READERS:
                row = r['summary'][(c['case'], reader)]
                text.append(f'| {r["path"].name} | {c["batch"]} | {reader} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    text += ['', 'Read-only primary is direct wall time around actual read/select/decode/source validation, including open/close for reopen paths and NumPy gather for gather paths. Packing, hashes, output/reference checks, separate Store probes and report writing are outside. Each repetition shuffles all case/reader jobs and uses the same A/B variant. Setup/probes/validation warm OS/library caches; no cold-disk claim.', '',
        '## Complete-request primary P50/P95/max', '', '| Session | Batch | Method | P50 ms | P95 ms | Max ms |', '|---|---:|---|---:|---:|---:|']
    for r in pipes:
        for c in CASES:
            for method in r['env']['methods']:
                row = r['summary'][(c['case'], method)]
                text.append(f'| {r["path"].name} | {c["batch"]} | {method} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    text += ['', 'Models remain resident. Every complete request actually reads and packs its new A/B input, uploads into the same fixed buffer for GPU engines, runs the fixed untrained FP32 FNO, returns all values, and writes/closes the identical uncompressed NPZ schema/byte count. Setup, FP64 work, file readback/hashes/checks and cleanup are outside primary. No stage synchronization, annotations or profiler are used in primary. File close does not request fsync. Graph resources coexist with eager/CPU. All P95/maxima and regressions remain; current sessions are compared within their protocol, not causally decomposed using older runs.', '',
        '## Paired original/gather reader contrasts', '',
        '| Session | Scope | Batch | Engine | Handle | Median bounding/gather | Median saving ms | Gather-faster pairs |', '|---|---|---:|---|---|---:|---:|---:|']
    for scope, runs in [('read-only', reads), ('complete-file', pipes)]:
        for r in runs:
            for c in CASES:
                for engine in (['reader'] if scope == 'read-only' else ['cpu', 'eager', 'graph']):
                    for label, group in GROUPS:
                        left, right = group[0], group[2]
                        if engine != 'reader':
                            left, right = engine+'_'+left, engine+'_'+right
                        a, b = paired(r, c, left, right, 'reader' if engine == 'reader' else 'method')
                        text.append(f'| {r["path"].name} | {scope} | {c["batch"]} | {engine} | {label} | {np.median(a/b):.3f}x | {np.median(a-b)*1000:.3f} | {np.sum(a>b)}/{len(a)} |')
    text += ['', 'Ratios pair the same repeat/case/lifetime/variant. They are descriptive, not significance tests or guarantees. Both genuine variants are retained, and all input values, identity/time/coordinates and full output contracts agree with the pinned CPU FP64 evidence. Independent synchronized stage profiles and full CPU/CUDA traces remain in each pipeline run, with actual expected graph launches and complete HtoD/DtoH. Those diagnostics are not an additive decomposition of primary P50, and Store gets are not decoder counts or physical storage traffic.', '',
        '## Reproduction and scope', '', '```bash', 'source .venv/physicsnemo/bin/activate',
        'python physicsnemo_graph_pipeline.py --readers open_once exact_once gather_once reopen exact_reopen gather_reopen --seed 20261115 --output results/my_gather_pipeline01',
        'python physicsnemo_graph_pipeline.py --readers open_once exact_once gather_once reopen exact_reopen gather_reopen --seed 20261116 --output results/my_gather_pipeline02',
        'python physicsnemo_reader_benchmark.py --readers open_once exact_once gather_once reopen exact_reopen gather_reopen --seed 20261115 --output results/my_gather01',
        'python physicsnemo_reader_benchmark.py --readers open_once exact_once gather_once reopen exact_reopen gather_reopen --seed 20261116 --output results/my_gather02',
        'python scripts/analyze_physicsnemo_gather.py results/my_gather_pipeline01 results/my_gather_pipeline02 --reader-runs results/my_gather01 results/my_gather02 --recompute-reference --output results/my_gather_report', '```', '',
        'Run benchmarks separately with new output paths. The CPU analyzer reconstructs source selections/coverage and validates actual Store reads, backing bytes, hashes, ordering, quantiles, files, FP64 errors, allocator arithmetic and real traces; --recompute-reference independently rebuilds the common archived FP64 reference with PhysicsNeMo. Existing archived two-/four-reader evidence remains supported. No new forecast-quality study, model training, remote acquisition, driver upgrade or automatic default selection is included.', '',
        '![Gather reader timing and array accounting](physicsnemo_gather.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated gather reader report:', args.output)


if __name__ == '__main__':
    main()
