"""Reconcile exact-frame reader bytes, actual Store gets and complete file requests."""
import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import array_mapping_hash, prepare_input, stats
from physicsnemo_graph import CASES
from physicsnemo_graph_pipeline import READERS, read_variant
from physicsnemo_reader import READER_POLICIES, read_coverage
from physicsnemo_reader_benchmark import reader_order
from scripts.analyze_physicsnemo_graph import validate as validate_reference
from scripts.analyze_physicsnemo_graph_pipeline import rows, validate as validate_pipeline
from weather_verify import sha256

PAIRS = [('open_once', 'exact_once'), ('reopen', 'exact_reopen')]


def validate_reads(run, data, reference_run, reference):
    env = json.loads((run/'environment.json').read_text())
    source = sha256(data)
    ref = reference['env']
    if env['source_sha256'] != source or source != ref['source_sha256'] or env['source_manifest'] != ref['source_manifest'] or env['reference_run'] != reference_run.name:
        raise ValueError('Reader source/reference mismatch')
    if env['cases'] != CASES or env['readers'] != READERS or env['reader_policies'] != READER_POLICIES:
        raise ValueError('Expected all original/exact reader policies')
    records, summaries = rows(run/'timing_samples.csv'), rows(run/'timing_summary.csv')
    order = reader_order(env['readers'], env['repeats'], env['seed'])
    if len(records) != len(order):
        raise ValueError('Reader samples missing/extra')
    for saved, expected in zip(records, order):
        if any((int(saved[k]) if isinstance(v, int) else saved[k]) != v for k, v in expected.items()):
            raise ValueError('Reader variant/order pairing changed')
    expected_keys = {(c['case'], r) for c in CASES for r in env['readers']}
    summary = {(r['case'], r['reader']): r for r in summaries}
    if set(summary) != expected_keys or len(summary) != len(summaries):
        raise ValueError('Reader summary missing/duplicate')
    path = run/'store_probes.json'
    if sha256(path) != env['store_probes_sha256']:
        raise ValueError('Store probe hash mismatch')
    probes = json.loads(path.read_text())
    coverage = {c['case']: {s: [read_coverage(data, c, v, s) for v in [0, 1]] for s in ['bounding', 'exact']} for c in CASES}
    if env['coverage'] != coverage or set(probes) != {c['case'] for c in CASES}:
        raise ValueError('Reader materialization/chunk coverage mismatch')
    for case in CASES:
        name = case['case']
        canonical = ref['case_metadata'][name]
        path = reference_run/canonical['reference_path']
        if sha256(path) != canonical['reference_sha256']:
            raise ValueError('Pinned reference changed')
        if set(probes[name]) != {'bounding', 'exact'} or len(env['fixtures'][name]) != 2:
            raise ValueError('Missing reader probe/fixture variants')
        with np.load(path, allow_pickle=False) as saved:
            for variant, label in enumerate(['A', 'B']):
                fixture = env['fixtures'][name][variant]
                expected_input = saved['input_'+label]
                if fixture['input_hash'] != array_mapping_hash({'input': expected_input}) or fixture['reference_sha256'] != sha256(path):
                    raise ValueError('Reader reference/input hash mismatch')
                for selection in ['bounding', 'exact']:
                    raw, identity, coords = read_variant(data, case, variant, selection=selection)
                    np.testing.assert_array_equal(prepare_input(raw), expected_input)
                    expected_coords = {k: v.tolist() for k, v in coords.items()}
                    if identity != fixture['identity'] or expected_coords != fixture['coordinates']:
                        raise ValueError('Reader genuine frame/coordinate identity changed')
                    probe, budget = probes[name][selection][variant], coverage[name][selection][variant]
                    if probe['identity'] != identity or probe['coordinates'] != expected_coords or probe['raw_bytes'] != budget['requested_raw_bytes'] or probe['backing_array_bytes'] != budget['materialized_raw_bytes']:
                        raise ValueError('Store probe input identity/backing bytes mismatch')
                    forecast = [r for r in probe['records'] if re.fullmatch(r'forecast/\d+(?:\.\d+){4}', r['key'])]
                    if sorted(r['key'] for r in forecast) != budget['chunk_keys'] or probe['forecast_gets'] != len(forecast) or probe['forecast_returned_bytes'] != sum(r['bytes'] for r in forecast) or probe['forecast_returned_bytes'] != budget['stored_chunk_bytes']:
                        raise ValueError('Store get count/keys/compressed bytes mismatch')
                    for record in probe['records']:
                        target = (data/record['key']).resolve()
                        if not target.is_relative_to(data.resolve()) or record['partial'] or record['found'] != target.is_file() or record['bytes'] != (target.stat().st_size if target.is_file() else 0):
                            raise ValueError('Store key/returned byte evidence mismatch')
                for row in [r for r in records if r['case'] == name and int(r['variant']) == variant]:
                    budget = coverage[name][READER_POLICIES[row['reader']]['selection']][variant]
                    if row['input_sha256'] != fixture['input_hash'] or int(row['raw_bytes']) != budget['requested_raw_bytes'] or int(row['backing_array_bytes']) != budget['materialized_raw_bytes']:
                        raise ValueError('Measured reader input/buffer bytes changed')
    for key in expected_keys:
        for field, value in stats([r for r in records if (r['case'], r['reader']) == key]).items():
            np.testing.assert_allclose(float(summary[key][field]), value, rtol=1e-12)
    print(f'PASS {run.name}: {len(records)} read-only samples; exact source frames, backing bytes, actual Store gets/bytes, order and quantiles reconciled')
    return dict(path=run, env=env, samples=records, summary=summary, probes=probes)


def paired(run, case, left, right, field='method', variant=None):
    records = {(r[field], int(r['repeat'])): float(r['seconds']) for r in run['samples'] if r['case'] == case['case'] and (variant is None or int(r['variant']) == variant)}
    repeats = [i for i in range(run['env']['repeats']) if variant is None or i % 2 == variant]
    return [np.array([records[(method, i)] for i in repeats]) for method in [left, right]]


def draw_bars(ax, labels, groups, legends, ylabel, title):
    width = .8/len(groups)
    for i, (values, label) in enumerate(zip(groups, legends)):
        values = np.asarray(values)
        median = np.median(values, axis=0)
        bars = ax.bar(np.arange(len(labels))+(i-(len(groups)-1)/2)*width, median, width*.94, label=label,
            yerr=[median-values.min(axis=0), values.max(axis=0)-median], capsize=3)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)


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
    pipelines = [validate_pipeline(p, args.data, args.reference_run, reference) for p in args.pipeline_runs]
    readers = [validate_reads(p, args.data, args.reference_run, reference) for p in args.reader_runs]
    if any(set(r['env']['readers']) != set(READERS) or len(r['env']['readers']) != len(READERS) or r['env']['engines'] != ['cpu', 'eager', 'graph'] for r in pipelines):
        raise ValueError('Full reader comparison requires all three engines and four readers')
    coverage = readers[0]['env']['coverage']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    draw_bars(axes[0, 0], ['A: frames 0-63', 'B: frames 64-127'],
        [[[coverage['b64_s64'][s][v]['materialized_raw_bytes']/2**20 for v in [0, 1]]] for s in ['bounding', 'exact']],
        ['Bounding', 'Exact'], 'Selected forecast array / backing MiB', 'B64: less materialization; not peak process RAM')
    budget = coverage['b64_s64']
    draw_bars(axes[0, 1], ['A', 'B'], [[[budget[s][v]['stored_chunk_bytes']/1024 for v in [0, 1]]] for s in ['bounding', 'exact']],
        ['Bounding', 'Exact'], 'Compressed forecast bytes returned (KiB)', 'Store probes: same four full chunks per request; not disk I/O')
    labels = [f'B{c["batch"]}\n'+('reuse handle' if a == 'open_once' else 'reopen') for c in CASES for a, b in PAIRS]
    groups = [[[float(r['summary'][(c['case'], pair[i])]['p50_sec'])*1000 for c in CASES for pair in PAIRS] for r in readers] for i in [0, 1]]
    draw_bars(axes[1, 0], labels, groups, ['Bounding', 'Exact'], 'Read-only P50 ms', 'Separate shuffled read/select/decode/check primary timings')
    groups = [[[float(r['summary'][('b64_s64', engine+'_'+reader)]['p50_sec'])*1000 for engine in ['cpu', 'eager', 'graph']] for r in pipelines] for reader in ['open_once', 'exact_once']]
    draw_bars(axes[1, 1], ['CPU', 'EAGER', 'GRAPH'], groups, ['Bounding', 'Exact'], 'Complete request P50 ms', 'B64 reused handle: read -> FNO -> full return -> NPZ close')
    fig.suptitle('Exact native ECMWF frame selection: materialization, Store access and end-to-end evidence\nSame source chunks, genuine A/B inputs, model weights and full output contract; whiskers = session range, not confidence interval', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_reader.png', dpi=160)
    plt.close(fig)
    text = ['# Exact-frame reader and complete-workflow comparison', '',
        'The optional exact_once/exact_reopen readers use shared-dimension xarray vectorized isel for the same native flattened init/lead/member A/B windows. Existing open_once/reopen bounding readers and CLI defaults remain. Models, input scaling/FP32 packing, fixed GPU buffers, output schema, output byte counts, tolerance and graph replay behavior are unchanged within this comparison. No prepared-input/output cache, regridding, rechunking, remote retrieval, training or forecast-quality evaluation is involved.', '',
        '## Selection and actual Store coverage', '',
        '| Batch | Variant | Selection | Materialized frames | Forecast array/backing MiB | Forecast Store gets | Compressed bytes returned | Nominal decoded chunk bytes |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for c in CASES:
        for v in [0, 1]:
            for s in ['bounding', 'exact']:
                b = coverage[c['case']][s][v]
                text.append(f'| {c["batch"]} | {"AB"[v]} | {s} | {b["materialized_frames"]} | {b["materialized_raw_bytes"]/2**20:.3f} | {b["chunks_touched"]} | {b["stored_chunk_bytes"]} | {b["nominal_chunk_uncompressed_bytes"]} |')
    text += ['', 'Separate instrumented LocalStore probes retain all key/byte observations including coordinates and metadata. Forecast chunk gets must match unique expected keys and stored compressed file sizes exactly. Probe returned-view and ultimate backing-array bytes must match selection budgets. These are Store API bytes, not physical disk/network traffic, HTTP requests, decoder invocation counts or peak process RAM. The unchanged fixture chunks are [1,4,51,32,32], so both paths read the same four forecast chunks (833,260 compressed bytes, 3,342,336 nominal uncompressed bytes) for both A/B and B16/B64. Exact selection reduces the forecast result/backing array, not chunk coverage in this fixture.', '',
        '## Independent read-only primary timings', '',
        '| Session | Batch | Reader | P50 ms | P95 ms | Maximum ms |', '|---|---:|---|---:|---:|---:|']
    for run in readers:
        for c in CASES:
            for reader in READERS:
                row = run['summary'][(c['case'], reader)]
                text.append(f'| {run["path"].name} | {c["batch"]} | {reader} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    text += ['', 'Read-only timings include select/read/decode/source validation and, for reopen, dataset open/close. Packing, identity/value/coordinate checks, hashes, probes and report writing are outside primary timing. All cases/readers are shuffled within each repeat and use the same alternating A/B input. No model executes in these measurements. Shared-handle setup/probes/reference checks warm caches; there is no cold-disk claim.', '',
        '### Paired read-only contrasts, split by genuine input variant', '',
        '| Session | Batch | Handle | Variant | Median bounding/exact | Median saving ms | Exact-faster pairs |', '|---|---:|---|---|---:|---:|---:|']
    for run in readers:
        for c in CASES:
            for left, right in PAIRS:
                for v in [0, 1]:
                    old, new = paired(run, c, left, right, 'reader', v)
                    text.append(f'| {run["path"].name} | {c["batch"]} | {left} | {"AB"[v]} | {np.median(old/new):.3f}x | {np.median(old-new)*1000:.3f} | {np.sum(old>new)}/{len(old)} |')
    text += ['', '## Complete Zarr-to-file primary timings', '',
        '| Session | Batch | Engine/reader | P50 ms | P95 ms | Maximum ms |', '|---|---:|---|---:|---:|---:|']
    for run in pipelines:
        for c in CASES:
            for method in run['env']['methods']:
                row = run['summary'][(c['case'], method)]
                text.append(f'| {run["path"].name} | {c["batch"]} | {method} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    text += ['', '### Matched complete-request reader contrasts', '',
        '| Session | Batch | Engine | Handle | Median bounding/exact | Median saving ms | Exact-faster pairs |', '|---|---:|---|---|---:|---:|---:|']
    for run in pipelines:
        for c in CASES:
            for engine in run['env']['engines']:
                for left, right in PAIRS:
                    old, new = paired(run, c, engine+'_'+left, engine+'_'+right)
                    text.append(f'| {run["path"].name} | {c["batch"]} | {engine} | {left} | {np.median(old/new):.3f}x | {np.median(old-new)*1000:.3f} | {np.sum(old>new)}/{len(old)} |')
    text += ['', 'Both methods include actual A/B reads/packing, new-input HtoD for GPU, the same untrained FP32 FNO, full host output and identical uncompressed NPZ creation/close. Models stay resident; eager/graph share fixed GPU input storage and graph resources coexist with other engines. Setup, FP64 reference work, output readback/checks/hashes and cleanup are outside primary timing. Every output file is read back and validated; last A/B outputs for every case/method remain archived. Real full-file traces independently prove the requested engine, full HtoD/DtoH and genuine cudaGraphLaunch only for graph.', '',
        'Independent synchronized stage profiles and allocator observations remain in each pipeline run. They are not an exact decomposition of primary P50 or standalone/total VRAM. File close is not fsync/durable-media persistence. CPU dispatch, GPU execution, OS/library state and profiling can vary; no single driver/power-state cause is isolated. All P95/maxima, regressions and samples remain. Paired ratios/counts describe these sessions, not significance tests or latency guarantees. Do not compare older sessions with different protocols as a causal decomposition.', '',
        '## Validation and reproduction', '',
        'The analyzer validates common pinned GPU evidence, both full-file workflows, source frame identities/coordinates, archived FP64 agreement, equal output byte counts, seeded pairing/quantiles, profile sums, setup/runtime allocator arithmetic and actual trace kernels/launches/copy bytes. It additionally validates reader budgets, observed backing-array sizes, every Store key/byte record against the unchanged source, and read-only input hashes versus pinned native A/B arrays. --recompute-reference independently rebuilds the common FP64 references using the isolated model environment.', '',
        '```bash', 'source .venv/physicsnemo/bin/activate',
        'python physicsnemo_graph_pipeline.py --readers open_once exact_once reopen exact_reopen --seed 20261113 --output results/my_reader_pipeline01',
        'python physicsnemo_graph_pipeline.py --readers open_once exact_once reopen exact_reopen --seed 20261114 --output results/my_reader_pipeline02',
        'python physicsnemo_reader_benchmark.py --seed 20261113 --output results/my_reader01',
        'python physicsnemo_reader_benchmark.py --seed 20261114 --output results/my_reader02',
        'python scripts/analyze_physicsnemo_reader.py results/my_reader_pipeline01 results/my_reader_pipeline02 --reader-runs results/my_reader01 results/my_reader02 --recompute-reference --output results/my_reader_report',
        '# Exact reader, graph alone:',
        'python physicsnemo_graph_pipeline.py --engines graph --readers exact_once --output results/my_exact_graph', '```', '',
        'Run benchmarks separately, use new output paths. Reader-only execution needs the CPU data dependencies; Store probes support Zarr 3 LocalStore and the original environment\'s Zarr 2 DirectoryStore. Recorded formal measurements use the saved isolated-environment versions. Model execution/reference rebuild needs the isolated PhysicsNeMo environment. Without --recompute-reference, CPU dependencies suffice to validate saved GPU evidence and redraw this figure. Existing pipeline defaults retain bounding readers; exact is explicitly selected.', '',
        '![Exact reader coverage, Store bytes, read-only latency and complete workflow](physicsnemo_reader.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated exact-reader report:', args.output)


if __name__ == '__main__':
    main()
