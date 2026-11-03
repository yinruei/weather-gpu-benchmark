"""Validate CPU/eager/graph full-file workflows and independently timed profiles."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import MODEL_CONFIG, TOLERANCE, array_mapping_hash, prepare_input, stats
from physicsnemo_graph import CASES
from physicsnemo_graph_pipeline import methods, read_variant, request_order, summarize_trace
from physicsnemo_pipeline import STAGES, output_payload, verify_output
from scripts.analyze_physicsnemo_graph import validate as validate_reference
from weather_verify import sha256
from physicsnemo_reader import READER_POLICIES, read_coverage


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data, reference_run, reference=None):
    if reference is None:
        reference = validate_reference(reference_run, data, Path('results/rtx3050_physicsnemo_run01'))
    ref = reference['env']
    env = json.loads((run/'environment.json').read_text())
    source = sha256(data)
    manifest = json.loads((data.parent/'manifest.json').read_text())
    if env['source_sha256'] != source or source != ref['source_sha256'] or source != manifest['files']['batch.zarr'] or env['source_manifest'] != manifest:
        raise ValueError('Input/provenance mismatch')
    chosen = methods(env['engines'], env['readers'])
    if 'reader_policies' in env:
        if env['reader_policies'] != {r: READER_POLICIES[r] for r in env['readers']}:
            raise ValueError('Reader policy mismatch')
        expected_coverage = {c['case']: {s: [read_coverage(data, c, v, s) for v in [0, 1]]
            for s in sorted({READER_POLICIES[r]['selection'] for r in env['readers']})} for c in CASES}
        if env.get('read_coverage') != expected_coverage:
            raise ValueError('Reader coverage mismatch')
    elif any(READER_POLICIES[r]['selection'] != 'bounding' for r in env['readers']):
        raise ValueError('Exact reader requires archived selection policies and coverage')
    if env['methods'] != chosen or env['cases'] != CASES or env['stages'] != STAGES or env['model_config'] != MODEL_CONFIG or env['tolerance'] != TOLERANCE or env['tf32'] or not env['deterministic_algorithms']:
        raise ValueError('Protocol mismatch')
    if env['reference_run'] != reference_run.name or env['weights_sha256'] != ref['weights_sha256'] or env['model_state_sha256'] != ref['model_state_sha256'] or sha256(run/'weights.npz') != ref['weights_sha256']:
        raise ValueError('Model/reference mismatch')
    sample, summary, profile, first, traced = [rows(run/f) for f in ['timing_samples.csv', 'timing_summary.csv', 'stage_samples.csv', 'first_observed.csv', 'trace_calls.csv']]
    memory = rows(run/'memory_samples.csv') if (run/'memory_samples.csv').exists() else []
    def check_order(records, expected):
        if len(records) != len(expected):
            raise ValueError('Missing/extra records')
        for actual, wanted in zip(records, expected):
            if any((int(actual[k]) if isinstance(v, int) else actual[k]) != v for k, v in wanted.items()):
                raise ValueError('Pairing/order/input variant changed')
    for records, count, seed in [(sample, env['repeats'], env['seed']), (profile, env['profile_repeats'], env['seed']+1), (traced, env['trace_repeats'], env['seed']+1000)]:
        check_order(records, request_order(env['engines'], env['readers'], count, seed))
    check_order(first, [dict(case=c['case'], method=m, variant=0, repeat=0, order=i) for c in CASES for i, m in enumerate(chosen)])
    keys = {(c['case'], m) for c in CASES for m in chosen}
    summaries = {(r['case'], r['method']): r for r in summary}
    if set(summaries) != keys or len(summary) != len(keys):
        raise ValueError('Missing/duplicate summaries')
    output_info = {}
    for case in CASES:
        name = case['case']
        meta, canonical = env['case_metadata'][name], ref['case_metadata'][name]
        if meta['reference_path'] != canonical['reference_path'] or meta['reference_sha256'] != canonical['reference_sha256'] or sha256(reference_run/meta['reference_path']) != meta['reference_sha256']:
            raise ValueError('Pinned reference archive mismatch')
        if set(meta['outputs']) != {f'{v}_{m}' for v in [0, 1] for m in chosen}:
            raise ValueError('Missing output variants/methods')
        with np.load(reference_run/meta['reference_path'], allow_pickle=False) as saved:
            for variant, label in enumerate(['A', 'B']):
                raw, identity, coordinates = read_variant(data, case, variant)
                x = prepare_input(raw)
                if identity != meta['identities'][variant] or meta['input_hashes'][variant] != array_mapping_hash({'input': x}) or x.nbytes != meta['input_bytes']:
                    raise ValueError('Input frame identity/transform/bytes changed')
                np.testing.assert_array_equal(x, saved['input_'+label])
                truth = saved['reference_'+label]
                file_sizes = set()
                for method in chosen:
                    retained = meta['outputs'][f'{variant}_{method}']
                    path = f'outputs/{name}_{variant}_{method}.npz'
                    if retained['path'] != path or sha256(run/path) != retained['sha256'] or (run/path).stat().st_size != retained['bytes']:
                        raise ValueError('Retained file hash/path/size mismatch')
                    with np.load(run/path, allow_pickle=False) as actual:
                        output = actual['model_output'].copy()
                    if output.dtype != np.float32 or output.shape != x.shape:
                        raise ValueError('Model output dtype/shape mismatch')
                    payload = output_payload(output, identity, coordinates, case, source, env['model_state_sha256'])
                    errors = verify_output(run/path, payload, truth)
                    digest = array_mapping_hash(payload)
                    if retained['payload_sha256'] != digest:
                        raise ValueError('Output array/metadata hash mismatch')
                    output_info[(name, variant, method)] = dict(file=retained, errors=errors, output=output)
                    file_sizes.add(retained['bytes'])
                if len(file_sizes) != 1:
                    raise ValueError('Engines/readers wrote unequal byte counts')
                for reader in env['readers']:
                    if {'eager', 'graph'}.issubset(env['engines']):
                        np.testing.assert_allclose(output_info[(name, variant, 'eager_'+reader)]['output'], output_info[(name, variant, 'graph_'+reader)]['output'], rtol=1e-6, atol=1e-7)
        if 'graph' in env['engines']:
            setup, mem = meta['graph_setup_seconds'], meta['graph_setup_memory']
            parts = ['buffer_clone', 'side_stream_warmup', 'capture_instantiate', 'first_replay']
            if set(setup) != set(parts+['total']) or not all(np.isfinite(v) and v > 0 for v in setup.values()) or sum(setup[k] for k in parts) > setup['total']+1e-9:
                raise ValueError('Graph setup duration mismatch')
            for metric in ['allocated', 'reserved']:
                before, after, extra = [int(mem[f'{k}_{metric}_bytes']) for k in ['before', 'after', 'extra']]
                if min(before, after) < 0 or extra != after-before or mem[f'peak_{metric}_bytes'] < max(before, after):
                    raise ValueError('Graph setup memory arithmetic mismatch')
            if mem['peak_reserved_bytes'] < mem['peak_allocated_bytes']:
                raise ValueError('Invalid setup memory peaks')
        elif meta['graph_setup_seconds'] is not None or meta['graph_setup_memory'] is not None:
            raise ValueError('Unexpected graph setup for non-graph selection')
    for row in sample+profile+first+traced:
        info = output_info[(row['case'], int(row['variant']), row['method'])]
        retained = info['file']
        if not np.isfinite(float(row['seconds'])) or float(row['seconds']) <= 0 or int(row['file_bytes']) != retained['bytes'] or row['file_sha256'] != retained['sha256'] or row['payload_sha256'] != retained['payload_sha256']:
            raise ValueError('Measured file/timing differs from retained output')
        for field, value in info['errors'].items():
            np.testing.assert_allclose(float(row[field]), value, rtol=1e-10, atol=1e-15)
    for key in keys:
        for field, value in stats([r for r in sample if (r['case'], r['method']) == key]).items():
            np.testing.assert_allclose(float(summaries[key][field]), value, rtol=1e-12)
    for row in profile:
        values = [float(row[s+'_sec']) for s in STAGES]+[float(row['unattributed_sec'])]
        if not np.isfinite(values).all() or min(values) < 0 or any(float(row[s+'_sec']) <= 0 for s in STAGES if s != 'upload'):
            raise ValueError('Invalid stage durations')
        if row['method'].startswith('cpu_') and float(row['upload_sec']) != 0:
            raise ValueError('CPU unexpectedly uploads input')
        if not row['method'].startswith('cpu_') and float(row['upload_sec']) <= 0:
            raise ValueError('GPU input update stage missing')
        np.testing.assert_allclose(sum(values), float(row['seconds']), rtol=1e-12, atol=1e-12)
    memory_keys = {k for k in keys if not k[1].startswith('cpu_')}
    if len(memory) != len(memory_keys) or {(r['case'], r['method']) for r in memory} != memory_keys:
        raise ValueError('Missing/extra runtime memory probes')
    for row in memory:
        before, peak, extra, reserved = [int(row[k]) for k in ['baseline_allocated_bytes', 'peak_allocated_bytes', 'peak_extra_allocated_bytes', 'peak_reserved_bytes']]
        if not 0 <= before <= peak <= reserved or extra != peak-before:
            raise ValueError('Runtime allocator arithmetic mismatch')
    path = run/'trace.json.gz'
    if sha256(path) != env['trace_sha256']:
        raise ValueError('Full workflow trace hash mismatch')
    actual_trace = summarize_trace(json.loads(gzip.decompress(path.read_bytes())))
    if len(actual_trace) != len(traced):
        raise ValueError('Trace root count mismatch')
    for saved, actual in zip(traced, actual_trace):
        for field, value in actual.items():
            if isinstance(value, str):
                if saved[field] != value:
                    raise ValueError('Trace annotation mismatch')
            else:
                np.testing.assert_allclose(float(saved[field]), value, rtol=1e-10, atol=1e-9)
        engine = actual['method'].split('_', 1)[0]
        expected = env['case_metadata'][actual['case']]['input_bytes']
        if engine == 'cpu':
            if any(actual[k] for k in ['kernel_events', 'graph_launches', 'dtoh_bytes', 'htod_bytes']):
                raise ValueError('CPU workflow contains CUDA activity')
        elif actual['kernel_events'] < 1 or actual['dtoh_copies'] != 1 or actual['dtoh_bytes'] != expected or actual['htod_bytes'] != expected or (engine == 'graph' and actual['graph_launches'] < 1) or (engine == 'eager' and actual['graph_launches'] != 0):
            raise ValueError('Full workflow lacks input/output copy or actual graph replay')
    print(f'PASS {run.name}: {len(sample)} full-file timings, {len(profile)} stage profiles, {len(traced)} full-file traces; files/provenance/FP64/ordering/quantiles/memory reconciled')
    return dict(path=run, env=env, samples=sample, summary=summaries, profile=profile, first=first, memory=memory, trace=traced)


def paired(run, case, reader):
    records = {(r['method'], int(r['repeat'])): float(r['seconds']) for r in run['samples'] if r['case'] == case['case']}
    return [np.array([records[(engine+'_'+reader, i)] for i in range(run['env']['repeats'])]) for engine in ['eager', 'graph']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--recompute-reference', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reference = validate_reference(args.reference_run, args.data, args.baseline, args.recompute_reference)
    runs = [validate(p, args.data, args.reference_run, reference) for p in args.runs]
    env = runs[0]['env']
    if any(r['env']['methods'] != env['methods'] for r in runs):
        raise ValueError('Compared sessions must share selected engines/readers')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for ax, case in zip(axes[0], CASES):
        width = .8/len(env['engines'])
        for i, engine in enumerate(env['engines']):
            values = np.array([[float(r['summary'][(case['case'], engine+'_'+reader)]['p50_sec'])*1000 for reader in env['readers']] for r in runs])
            median = np.median(values, axis=0)
            bars = ax.bar(np.arange(len(env['readers']))+(i-(len(env['engines'])-1)/2)*width, median, width=width*.94, label=engine.upper(),
                          yerr=[median-values.min(axis=0), values.max(axis=0)-median], capsize=3)
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        ax.set_xticks(range(len(env['readers'])), [r.replace('_', ' ') for r in env['readers']])
        ax.set(ylabel='Complete file-request P50 (ms)', title=f'B{case["batch"]}/64x64; actual alternating A/B inputs')
        ax.legend(fontsize=8)
    comparison = {'eager', 'graph'}.issubset(env['engines'])
    if comparison:
        labels, values = [], []
        for case in CASES:
            for reader in env['readers']:
                labels.append(f'B{case["batch"]}\n{reader.replace("_", " ")}')
                values.append([np.median(paired(r, case, reader)[0]/paired(r, case, reader)[1]) for r in runs])
        med = np.median(values, axis=1)
        axes[1, 0].bar(range(len(labels)), med, color='#dd7722', yerr=[med-np.min(values, axis=1), np.max(values, axis=1)-med], capsize=4)
        axes[1, 0].set_xticks(range(len(labels)), labels)
        axes[1, 0].axhline(1, color='black', linestyle=':', linewidth=1)
        axes[1, 0].set(ylabel='Median paired eager / graph wall time', title='Complete-flow graph benefit; >1 favors graph')
    else:
        axes[1, 0].text(.5, .5, 'Select both eager and graph\nto compare engines', ha='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_axis_off()
    reader = 'open_once' if 'open_once' in env['readers'] else env['readers'][0]
    selections = [[row for run in runs for row in run['profile'] if row['case'] == CASES[-1]['case'] and row['method'] == engine+'_'+reader] for engine in env['engines']]
    bottom = np.zeros(len(selections))
    for name, label, color in zip(STAGES+['unattributed'], ['Read/decode/check', 'Prepare/pack', 'Upload', 'Forward', 'Host return', 'Write/close', 'Other'], ['#4c78a8', '#72b7b2', '#f2cf5b', '#f58518', '#b279a2', '#54a24b', '#aaaaaa']):
        values = [np.mean([float(r[name+'_sec']) for r in selected])*1000 for selected in selections]
        axes[1, 1].bar(range(len(selections)), values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1, 1].set_xticks(range(len(selections)), [e.upper() for e in env['engines']])
    axes[1, 1].set(ylabel='Separate synchronized profile mean (ms)', title=f'B64 {reader}: profiles are not primary decomposition')
    axes[1, 1].legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('CUDA Graph in the complete local Zarr -> FNO -> NPZ workflow\nSame genuine A/B inputs; models resident; no prepared-input cache; primary includes file close, excludes setup/checks', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_graph_pipeline.png', dpi=160)
    plt.close(fig)
    text = ['# Complete Zarr-to-file CPU/eager/CUDA Graph workflow', '',
            'The selectable FnoFilePipeline engine runs the same untrained FP32 FNO, exact archived weights and two disjoint genuine ECMWF windows per shape. Every request rereads and packs its requested A/B input, copies it to the fixed GPU buffer for GPU engines, computes, returns the full output and writes/closes the same uncompressed NPZ schema. No input/output result cache, training or forecast-quality evaluation is involved.', '',
            '## Primary complete-request results', '',
            '| Session | Batch | Method | P50 ms | P95 ms | Maximum ms |', '|---|---:|---|---:|---:|---:|']
    for run in runs:
        for case in CASES:
            for method in env['methods']:
                row = run['summary'][(case['case'], method)]
                text.append(f'| {run["path"].name} | {case["batch"]} | {method} | {float(row["p50_sec"])*1000:.3f} | {float(row["p95_sec"])*1000:.3f} | {float(row["max_sec"])*1000:.3f} |')
    if comparison:
        text += ['', '## Matched eager/graph contrasts and estimated setup amortization', '',
                 '| Session | Batch | Reader | Median paired eager/graph | Median saving ms | Graph-faster pairs | Mean saving ms | Setup ms | Estimated uses to recover setup |',
                 '|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for run in runs:
            for case in CASES:
                setup = run['env']['case_metadata'][case['case']]['graph_setup_seconds']['total']
                for reader in env['readers']:
                    eager, graph = paired(run, case, reader)
                    saving = float(np.mean(eager-graph))
                    estimate = str(math.ceil(setup/saving)) if saving > 0 else 'No positive mean saving'
                    text.append(f'| {run["path"].name} | {case["batch"]} | {reader} | {np.median(eager/graph):.3f}x | {np.median(eager-graph)*1000:.3f} | {int(np.sum(eager>graph))}/{len(eager)} | {saving*1000:.3f} | {setup*1000:.3f} | {estimate} |')
    text += ['', 'Pairing matches repetition/case/reader and the same A/B variant; order of all cases/methods is shuffled. The ratios/counts are descriptive, not significance tests or latency guarantees. All regressions, P95/maxima and samples remain. Top bars show the median of session medians, whiskers their range, not confidence intervals. Graph savings can be diluted or reversed by CPU/I/O/system variation; do not infer whole-workflow speedup from earlier forward-only measurements.', '',
             '## Setup and scope', '',
             'Models stay resident. Open-once reuses one dataset handle shared across all methods/cases, while selecting/loading/checking/packing anew per request. Reopen creates a handle each request. Neither path reuses a prepared tensor as its input. Eager and graph use the same fixed GPU input buffer and both include the host copy inside timing. CPU uses the same archived state/inputs with the recorded thread controls.', '',
             'Common import/source-hash/model setup and shared reader opening are recorded separately. Per-shape graph setup covers buffer clone, side-stream warmup, capture/instantiate and first replay after eager warmup. Its allocator snapshots are net observations with earlier graphs present, not total VRAM, standalone engine memory or graph-executable memory. Eager/CPU timing runs while graph resources coexist. Setup divided by a positive mean end-to-end saving is an estimate sensitive to outliers, not a separately measured N-request application or cold startup.', '',
             'Primary timing goes directly from source read through file close, including metadata/coordinate checks and output metadata construction. Source preparation, FP64 reference work, startup checks, output readback, hashes, cleanup and benchmark CSV/JSON writing are outside timers. File close does not request fsync or guarantee durable media persistence. Source/reference setup and validation may warm OS/library caches; no cold-disk claim or remote-download cost is made. first_observed.csv contains initialized-process observations.', '',
             '## Separate stage profiles and real full-file traces', '',
             '| Session | Batch | Method | Read ms | Prepare ms | Upload ms | Forward ms | Return ms | Write ms | Other ms | Profile mean ms |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for case in CASES:
            for method in env['methods']:
                selected = [r for r in run['profile'] if r['case'] == case['case'] and r['method'] == method]
                values = [np.mean([float(r[s+'_sec']) for r in selected])*1000 for s in STAGES+['unattributed']]
                values.append(np.mean([float(r['seconds']) for r in selected])*1000)
                text.append(f'| {run["path"].name} | {case["batch"]} | {method} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    text += ['', 'Stage profiles are separate later executions with stage GPU synchronization. Their mean segments plus residual sum to their own profile mean, not primary P50; profiles can materially perturb latency. The separate CPU/CUDA trace covers source read, preprocessing, upload, actual engine call, full return and file close, with validation/hash/cleanup outside the root. CPU calls must have no CUDA activity; GPU calls must contain kernels, full input HtoD and full output DtoH, with real cudaGraphLaunch only for graph. Device activity overlaps CPU ranges and is not occupancy or an additive decomposition.', '',
             'Every first/warmup/primary/profile/memory/trace file is read back and checked for exact values/dtypes/coordinates/init/lead/member/valid-time metadata plus elementwise CPU FP64 agreement at rtol=1e-4, atol=1e-5. Both A/B references are independently recomputed in the runner; the optional analyzer flag independently rebuilds the common pinned reference model. Actual last primary files for both variants/all methods are retained; all trial file/payload hashes match the corresponding deterministic retained output. CPU/eager/graph write equal byte counts per case/variant.', '',
             'The CPU analyzer validates the pinned CUDA Graph reference evidence, reconstructs source selections and file contracts, verifies all hashes/errors/pairing/quantiles/stage sums/setup/runtime allocator arithmetic, and reconstructs actual full-file traces from gzip JSON. No silent eager/CPU fallback is allowed for graph. Plain eager and the previous pipeline remain available.', '',
             '## Reproduce', '', '```bash', 'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_graph_pipeline.py --output results/my_graph_pipeline01',
             'python physicsnemo_graph_pipeline.py --seed 20261112 --output results/my_graph_pipeline02',
             'python scripts/analyze_physicsnemo_graph_pipeline.py results/my_graph_pipeline01 results/my_graph_pipeline02 --recompute-reference --output results/my_graph_pipeline_report',
             '# Only graph + open-once reader:',
             'python physicsnemo_graph_pipeline.py --engines graph --readers open_once --output results/my_graph_only',
             '# CPU engine alone:',
             'python physicsnemo_graph_pipeline.py --engines cpu --readers reopen --output results/my_cpu_only', '```', '',
             'Use new output paths. The existing isolated PhysicsNeMo environment is needed to execute models. Without --recompute-reference, existing CPU data/plot dependencies suffice to revalidate saved GPU evidence and regenerate the report. This fixed-case benchmark does not implement arbitrary dynamic-shape serving, global multi-variable forecast models, remote retrieval, compression tuning or async overlap.', '',
             '![Complete-file engine comparison and independent stage profiles](physicsnemo_graph_pipeline.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved validated graph file-pipeline report:', args.output)


if __name__ == '__main__':
    main()
