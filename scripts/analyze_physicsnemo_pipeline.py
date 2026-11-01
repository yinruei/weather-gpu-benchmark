"""Reconcile full-pipeline artifacts, primary timings and independent stage profiles."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physicsnemo_benchmark import MODEL_CONFIG, TOLERANCE, array_mapping_hash, error_summary, prepare_input, read_input, stats
from physicsnemo_pipeline import METHODS, STAGES, input_budget, output_payload, verify_output
from weather_verify import sha256


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(run, data, baseline):
    env = json.loads((run/'environment.json').read_text())
    base = json.loads((baseline/'environment.json').read_text())
    source_hash = sha256(data)
    manifest = json.loads((data.parent/'manifest.json').read_text())
    if source_hash != env['source_sha256'] or source_hash != manifest['files']['batch.zarr'] or env['source_manifest'] != manifest or base['source_sha256'] != source_hash:
        raise ValueError('Source hash/provenance mismatch')
    if env['model_config'] != MODEL_CONFIG or env['tolerance'] != TOLERANCE or env['methods'] != METHODS or env['stages'] != STAGES or env['tf32'] or not env['deterministic_algorithms']:
        raise ValueError('Model, tolerance, method or precision contract changed')
    if env['model_state_sha256'] != base['model_state_sha256'] or env['weights_sha256'] != base['weights_sha256'] or sha256(run/'weights.npz') != base['weights_sha256']:
        raise ValueError('Model state differs from baseline')
    with np.load(run/'weights.npz', allow_pickle=False) as saved:
        if array_mapping_hash({k: saved[k] for k in saved.files}) != env['model_state_sha256']:
            raise ValueError('Model tensor hash mismatch')
    if len({c['case'] for c in env['cases']}) != len(env['cases']) or any(c not in base['cases'] for c in env['cases']):
        raise ValueError('Duplicate or unsupported case')
    sample, summary, first, profile, memory = [rows(run/f) for f in ['timing_samples.csv', 'timing_summary.csv', 'first_observed.csv', 'stage_samples.csv', 'memory.csv']]
    keys = {(c['case'], m) for c in env['cases'] for m in METHODS}
    summaries = {(r['case'], r['method']): r for r in summary}
    if set(summaries) != keys or len(summaries) != len(summary):
        raise ValueError('Missing or duplicate summary')
    errors, payload_hashes, file_hashes, file_sizes = {}, {}, {}, {}
    for case in env['cases']:
        name = case['case']
        meta = env['case_metadata'][name]
        raw, identity, coordinates = read_input(data, case)
        x = prepare_input(raw)
        if meta['identity'] != identity or meta['input_sha256'] != array_mapping_hash({'input': x}) or meta['budget'] != input_budget(data, case):
            raise ValueError('Selected frames, transform, coordinates or read budget changed')
        baseline_path = baseline/(name+'.npz')
        if sha256(baseline_path) != meta['baseline_artifact_sha256'] or meta['baseline_artifact_sha256'] != base['case_metadata'][name]['artifact_sha256']:
            raise ValueError('Archived numerical reference hash mismatch')
        with np.load(baseline_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(x, saved['input_fp32'])
            reference = saved['reference_fp64'].copy()
        if reference.dtype != np.float64 or reference.shape != x.shape:
            raise ValueError('Invalid reference shape/dtype')
        for backend in METHODS:
            key = (name, backend)
            retained = meta['outputs'][backend]
            expected_path = 'outputs/'+name+'_'+backend+'.npz'
            if retained['path'] != expected_path:
                raise ValueError('Unexpected retained output path')
            output_path = run/expected_path
            if sha256(output_path) != retained['sha256'] or output_path.stat().st_size != retained['bytes']:
                raise ValueError('Retained output hash/size mismatch')
            with np.load(output_path, allow_pickle=False) as saved:
                actual = saved['model_output'].copy()
            if actual.dtype != np.float32:
                raise ValueError('Expected FP32 model output')
            payload = output_payload(actual, identity, coordinates, case, source_hash, env['model_state_sha256'])
            errors[key] = verify_output(output_path, payload, reference)
            payload_hashes[key] = array_mapping_hash(payload)
            file_hashes[key], file_sizes[key] = retained['sha256'], retained['bytes']
            if payload_hashes[key] != retained['payload_sha256']:
                raise ValueError('Output tensor/metadata hash mismatch')
        if file_sizes[(name, 'cpu')] != file_sizes[(name, 'cuda')]:
            raise ValueError('CPU/CUDA output byte counts differ')
    def check_record(row):
        key = (row['case'], row['method'])
        if key not in keys or not np.isfinite(float(row['seconds'])) or float(row['seconds']) <= 0:
            raise ValueError('Invalid timing record')
        if row['payload_sha256'] != payload_hashes[key] or row['file_sha256'] != file_hashes[key] or int(row['file_bytes']) != file_sizes[key]:
            raise ValueError('Trial output differs from retained deterministic output')
        for field, expected in errors[key].items():
            np.testing.assert_allclose(float(row[field]), expected, rtol=1e-10, atol=1e-15)
    for row in sample+first+profile:
        check_record(row)
    def check_order(records, repeats, seed):
        expected = []
        for case in env['cases']:
            rng = np.random.default_rng(seed)
            expected.extend((case['case'], str(m), repeat, order) for repeat in range(repeats) for order, m in enumerate(rng.permutation(METHODS)))
        observed = [(r['case'], r['method'], int(r['repeat']), int(r['order'])) for r in records]
        if observed != expected:
            raise ValueError('Missing, extra or out-of-order samples')
    check_order(sample, env['repeats'], env['order_seed'])
    check_order(profile, env['profile_repeats'], env['order_seed']+1)
    if [(r['case'], r['method'], int(r['repeat']), int(r['order'])) for r in first] != [(c['case'], m, 0, i) for c in env['cases'] for i, m in enumerate(METHODS)]:
        raise ValueError('Unexpected first-observation records')
    for key in keys:
        selected = [r for r in sample if (r['case'], r['method']) == key]
        for field, expected in stats(selected).items():
            np.testing.assert_allclose(float(summaries[key][field]), expected, rtol=1e-12)
        batch = next(c['batch'] for c in env['cases'] if c['case'] == key[0])
        np.testing.assert_allclose(float(summaries[key]['frames_per_second']), batch/float(summaries[key]['p50_sec']), rtol=1e-12)
    for row in profile:
        values = np.array([float(row[s+'_sec']) for s in STAGES]+[float(row['unattributed_sec'])])
        if not np.isfinite(values).all() or (values < 0).any() or any(float(row[s+'_sec']) <= 0 for s in STAGES if s != 'upload'):
            raise ValueError('Invalid stage durations')
        if (row['method'] == 'cpu' and float(row['upload_sec']) != 0) or (row['method'] == 'cuda' and float(row['upload_sec']) <= 0):
            raise ValueError('Unexpected upload stage')
        np.testing.assert_allclose(values.sum(), float(row['seconds']), rtol=1e-12, atol=1e-12)
    if len(memory) != len(env['cases']) or {r['case'] for r in memory} != {c['case'] for c in env['cases']}:
        raise ValueError('Missing/extra memory observation')
    for row in memory:
        baseline_bytes, peak, extra, reserved = [int(row[k]) for k in ['baseline_allocated_bytes', 'peak_allocated_bytes', 'peak_extra_allocated_bytes', 'peak_reserved_bytes']]
        if not 0 <= baseline_bytes <= peak <= reserved or extra != peak-baseline_bytes:
            raise ValueError('Invalid memory arithmetic')
    print(f'PASS {run.name}: {len(sample)} full-workflow samples, {len(profile)} profiles, {len(keys)} retained outputs; provenance, FP64 error, persistence, timing and stages reconciled')
    return dict(path=run, env=env, summary=summaries, sample=sample, profile=profile, first=first, memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--baseline', type=Path, default=Path('results/rtx3050_physicsnemo_run01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [validate(p, args.data, args.baseline) for p in args.runs]
    cases = runs[0]['env']['cases']
    if any(r['env']['cases'] != cases or r['env']['model_state_sha256'] != runs[0]['env']['model_state_sha256'] for r in runs):
        raise ValueError('Sessions must share cases and model state')
    baseline_summary = {(r['case'], r['method']): r for r in rows(args.baseline/'timing_summary.csv')}
    def ms(run, case, backend, field='p50_sec'):
        return float(run['summary'][(case['case'], backend)][field])*1000
    labels = [f'B{c["batch"]}\n{c["side"]}x{c["side"]}' for c in cases]
    x = np.arange(len(cases))
    stage_names = STAGES+['unattributed']
    stage_labels = ['Read/decode/check', 'Prepare/pack', 'Upload', 'FNO forward', 'Full host return', 'NPZ write/close', 'Instrumentation/other']
    colors = ['#4c78a8', '#72b7b2', '#f2cf5b', '#f58518', '#b279a2', '#54a24b', '#aaaaaa']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for backend, label, color in [('cpu', 'CPU full local workflow', '#3366aa'), ('cuda', 'CUDA full local workflow', '#dd7722')]:
        values = np.array([[ms(r, c, backend) for c in cases] for r in runs])
        axes[0, 0].plot(x, np.median(values, axis=0), 'o-', label=label, color=color)
        axes[0, 0].fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=.15)
    axes[0, 0].set(ylabel='Primary wall-clock P50 (ms)', title='Local read -> prepare -> model -> file close')
    axes[0, 0].legend(fontsize=8)
    ratios = np.array([[ms(r, c, 'cpu')/ms(r, c, 'cuda') for c in cases] for r in runs])
    axes[0, 1].plot(x, np.median(ratios, axis=0), 'o-', label='Current full workflow', color='#dd7722')
    axes[0, 1].fill_between(x, ratios.min(axis=0), ratios.max(axis=0), alpha=.15, color='#dd7722')
    earlier = [float(baseline_summary[(c['case'], 'cpu_forward')]['p50_sec'])/float(baseline_summary[(c['case'], 'cuda_upload_forward_return')]['p50_sec']) for c in cases]
    axes[0, 1].plot(x, earlier, 'o--', color='#888888', label='Earlier forward/upload/return (context)')
    axes[0, 1].axhline(1, color='black', linestyle=':', linewidth=1)
    axes[0, 1].set(ylabel='CPU P50 / CUDA P50', title='Full-workflow speedup; >1 means CUDA faster')
    axes[0, 1].legend(fontsize=8)
    largest = cases[-1]
    profile_labels, bottoms = [], np.zeros(len(runs)*2)
    selections = []
    for i, run in enumerate(runs):
        for backend in METHODS:
            selections.append([r for r in run['profile'] if r['case'] == largest['case'] and r['method'] == backend])
            profile_labels.append(f'Run {i+1}\n{backend.upper()}')
    for name, label, color in zip(stage_names, stage_labels, colors):
        values = [np.mean([float(r[name+'_sec']) for r in selected])*1000 for selected in selections]
        axes[1, 0].bar(np.arange(len(selections)), values, bottom=bottoms, label=label, color=color)
        bottoms += values
    axes[1, 0].set_xticks(np.arange(len(selections)), profile_labels)
    axes[1, 0].set(ylabel='Separate profile mean (ms)', title=f'B{largest["batch"]}, {largest["side"]}x{largest["side"]}: separate synchronized profiles\nUse primary wall times for speedup')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].bar_label(axes[1, 0].containers[-1], labels=[f'{v:.1f}' for v in bottoms], padding=3, fontsize=8)
    bottoms = np.zeros(len(cases))
    for name, label, color in zip(stage_names, stage_labels, colors):
        values = []
        for case in cases:
            selected = [r for run in runs for r in run['profile'] if r['case'] == case['case'] and r['method'] == 'cuda']
            values.append(100*np.mean([float(r[name+'_sec']) for r in selected])/np.mean([float(r['seconds']) for r in selected]))
        axes[1, 1].bar(x, values, bottom=bottoms, color=color, label=label)
        bottoms += values
    axes[1, 1].set(ylabel='Share of synchronized profile mean (%)', title='CUDA workflow: where the time goes', ylim=(0, 105))
    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax.set_xticks(x, labels)
        ax.set_xlabel('Batch and native grid')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Phase 5B: real local Zarr -> fixed untrained PhysicsNeMo FNO -> uncompressed NPZ\nResident models; repeated reads with uncontrolled OS cache; no fsync; profiles separate from primary timing', fontsize=12)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'physicsnemo_pipeline.png', dpi=160)
    plt.close(fig)
    text = ['# Phase 5B: local Zarr-to-FNO-to-file performance', '',
            'The same fixed, untrained 132,353-parameter FNO and FP32 input transformation are retained from phase 5A. No forecast-quality evaluation, training or Earth2Studio pipeline is involved.', '',
            '## Primary complete-workflow timings', '',
            '| Session | Batch/grid | CPU P50 ms | CUDA P50 ms | CPU/CUDA | CPU P95 ms | CUDA P95 ms | CUDA frames/s |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for c in cases:
            cpu, gpu = ms(run, c, 'cpu'), ms(run, c, 'cuda')
            text.append(f'| {run["path"].name} | {c["batch"]}/{c["side"]}x{c["side"]} | {cpu:.3f} | {gpu:.3f} | {cpu/gpu:.2f}x | {ms(run,c,"cpu","p95_sec"):.3f} | {ms(run,c,"cuda","p95_sec"):.3f} | {c["batch"]/(gpu/1000):.1f} |')
    largest_ratios = [ms(r, largest, 'cpu')/ms(r, largest, 'cuda') for r in runs]
    text += ['', f'For the final listed case B{largest["batch"]}/{largest["side"]}x{largest["side"]}, observed complete-workflow CPU/CUDA ratios range from {min(largest_ratios):.2f}x to {max(largest_ratios):.2f}x. Ratios below 1 elsewhere are retained, including any nonmonotonic batch behavior. These results do not justify assuming every batch benefits from CUDA.', '',
             'Primary wall time is measured directly through file close. It includes Zarr open/read/decode and source coordinate validation, fixed scaling and contiguous FP32 packing, frame/coordinate output metadata construction, host tensor creation, CUDA upload, model forward, full CPU return and uncompressed NPZ creation/write/close. CPU and CUDA models/weights stay resident; each invocation reopens and rereads the source. File readback, reference checks, hashing, cleanup and benchmark CSV/JSON writing are excluded.', '',
             'Primary repetitions are separate from stage-profile repetitions. CUDA completion is guaranteed by the blocking full host copy before writing. Profiles additionally synchronize GPU stages and expose instrumentation cost; their mean segments plus residual sum to the profile mean, not the primary P50. Stage medians are never added to estimate total latency.', '',
             'Profile means can differ materially from primary timings because synchronization boundaries and execution time/system state differ. In particular, GPU forward wall times in this workflow may be longer than the earlier forward-only benchmark. The current samples do not isolate driver scheduling, power state, shape history or other causes. Stage shares describe the synchronized profile experiment, not an exact decomposition of primary latency or proof of a sole bottleneck.', '',
             '## Independent stage profiles', '',
             '| Session | Batch/grid | Backend | Read/decode/check ms | Prepare ms | Upload ms | Forward ms | Return ms | Write/close ms | Other ms | Profile mean ms |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        for c in cases:
            for backend in METHODS:
                selected = [r for r in run['profile'] if r['case'] == c['case'] and r['method'] == backend]
                values = [np.mean([float(r[s+'_sec']) for r in selected])*1000 for s in stage_names]
                total = np.mean([float(r['seconds']) for r in selected])*1000
                text.append(f'| {run["path"].name} | {c["batch"]}/{c["side"]}x{c["side"]} | {backend} | '+' | '.join(f'{v:.3f}' for v in values+[total])+' |')
    text += ['', '## Output/source correctness and interpretation', '',
             'Every first, warmup, primary, profile and memory-probe invocation writes a real NPZ file, reads it back, verifies exact array values/dtypes/coordinates/initialization/lead/member/valid-time metadata and compares the full model output elementwise with the archived CPU FP64 reference (rtol=1e-4, atol=1e-5). Each session independently recomputes that FP64 reference using the archived FP32 weights cast to double. Numerical differences describe arithmetic consistency, not weather error. Output variable is dimensionless model_output, not a temperature forecast.', '',
             'Only the last primary output per case/backend is retained. Deterministic payload/file hashes, output byte counts and numerical checks are recorded for all primary/profile/first observations; discarded trial files are removed after validation outside timing. The CPU-only analyzer checks retained files against original frames, archived weights/reference and every recorded trial hash, then independently reconciles raw quantiles, method ordering and stage sums. CPU/CUDA write identical byte counts with uncompressed np.savez.', '',
             'The source is the pinned local Zarr, not a fresh remote download. Cases follow listed order; primary CPU/CUDA order is seeded and shuffled per repetition, profiles use a separate seed. Reads occur after source hash/reference setup and may use OS/library cache; first_observed.csv is an initialized-process observation, not cold storage/process timing. No cache is dropped. Validation readback and trial deletion also affect cache/system state outside timing. The range shading covers session medians and is not a confidence interval. Short-session P95 values are estimates.', '',
             'Write completion means file close at the application/OS level; fsync and durable media persistence are not requested. Compression, metadata-heavy Zarr output, asynchronous overlap, remote acquisition, model/process startup and training are separate workloads. Setup duration is recorded outside primary timing; GPU allocator memory.csv is a separate warmed full-invocation observation and is not total process VRAM.', '',
             'The earlier forward/upload/return curve is context from an earlier measurement session, not a contemporaneous decomposition of current end-to-end times. Compare current CPU/CUDA within each session for the complete-workflow speedup.', '',
             '## Read coverage', '',
             '| Batch/grid | Packed input bytes | Materialized frames | Selection bytes before truncation | Touched local chunks | Nominal uncompressed chunk bytes |',
             '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        b = runs[0]['env']['case_metadata'][c['case']]['budget']
        text.append(f'| {c["batch"]}/{c["side"]}x{c["side"]} | {b["packed_input_bytes"]} | {b["materialized_frames"]} | {b["selected_logical_bytes"]} | {b["chunks_touched"]} | {b["nominal_chunk_uncompressed_bytes"]} |')
    text += ['', 'Selection matches phase 5A: native rectangular init/lead/member selection, then first B flattened frames. B64 materializes 102 frames before truncation to 64. Chunk estimates use the local v2 chunk layout; they are neither measured physical disk bytes nor remote-archive transfer sizes. This baseline deliberately preserves that selection behavior so later reader changes can be assessed separately.', '',
             '## Reproduce', '', '```bash', 'source .venv/physicsnemo/bin/activate',
             'python physicsnemo_pipeline.py --output results/my_physics_pipeline01',
             'python physicsnemo_pipeline.py --order-seed 20261103 --output results/my_physics_pipeline02',
             'python scripts/analyze_physicsnemo_pipeline.py results/my_physics_pipeline01 results/my_physics_pipeline02 --output results/my_physics_pipeline_comparison', '```', '',
             'Use new output paths. The existing phase-5 isolated environment and exact lock apply; CPU-only dependencies suffice to revalidate saved artifacts and regenerate this report. No JAX installation is required.', '',
             '![Complete local workflow, speedup and separate stage profiles](physicsnemo_pipeline.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Saved validated pipeline report: {args.output}')


if __name__ == '__main__':
    main()
