"""Validate selected real Nsight events, retained outputs and partial-capture status."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsight_trace import summarize_events
from physicsnemo_benchmark import array_mapping_hash, prepare_input
from physicsnemo_graph_pipeline import read_variant
from physicsnemo_nsight import CASE, ENGINES, READERS, jobs
from physicsnemo_pipeline import STAGES, output_payload, verify_output
from scripts.analyze_physicsnemo_graph import validate as validate_reference
from scripts.analyze_physicsnemo_graph_pipeline import rows
from weather_verify import sha256, write_csv


def validate(run, data, ref_run, reference):
    capture = json.loads((run/'capture_environment.json').read_text())
    manifest = json.loads((run/'event_manifest.json').read_text())
    env = json.loads((run/'checks/environment.json').read_text())
    calls = rows(run/'checks/calls.csv')
    canonical = reference['env']
    if capture['target_sha256'] != sha256(run/'target_source.py') or sha256(run/'events.sqlite') != manifest['events_sha256'] or manifest['source_sqlite_sha256'] != capture['sqlite_sha256']:
        raise ValueError('Captured source/event provenance mismatch')
    for name, key in [('capture.log', 'capture_log_sha256'), ('export.log', 'export_log_sha256')]:
        if sha256(run/name) != capture[key]:
            raise ValueError('Collector log hash mismatch')
    if env['source_sha256'] != sha256(data) or env['source_sha256'] != canonical['source_sha256'] or env['model_state_sha256'] != canonical['model_state_sha256'] or env['weights_sha256'] != canonical['weights_sha256']:
        raise ValueError('Nsight source/state differs from canonical benchmark')
    if env['case'] != CASE or env['engines'] != ENGINES or env['readers'] != READERS or env['threads'] != 6 or env['tf32'] or not env['deterministic_algorithms'] or not env['capture']:
        raise ValueError('Nsight protocol mismatch')
    meta = canonical['case_metadata'][CASE['case']]
    if env['reference_path'] != meta['reference_path'] or env['reference_sha256'] != meta['reference_sha256'] or sha256(ref_run/meta['reference_path']) != env['reference_sha256']:
        raise ValueError('Nsight reference mismatch')
    selected, complete, tables, strings = summarize_events(run/'events.sqlite')
    if manifest['counts'] != {t: len(r) for t, r in tables.items()}:
        raise ValueError('Selected-event table counts changed')
    expected = jobs(env['repeats'], env['seed'])
    if len(calls) != len(expected) or len(selected) != len(expected):
        raise ValueError('Missing/extra recorded requests')
    info = {}
    with np.load(ref_run/meta['reference_path'], allow_pickle=False) as archive:
        for variant, label in enumerate(['A', 'B']):
            raw, identity, coords = read_variant(data, CASE, variant)
            np.testing.assert_array_equal(prepare_input(raw), archive['input_'+label])
            if env['identities'][variant] != identity:
                raise ValueError('Nsight source identity changed')
            for reader in READERS:
                for engine in ENGINES:
                    key = f'{variant}_{engine}_{reader}'
                    saved = env['outputs'][key]
                    path = f'outputs/{key}.npz'
                    if saved['path'] != path or sha256(run/'checks'/path) != saved['sha256'] or (run/'checks'/path).stat().st_size != saved['bytes']:
                        raise ValueError('Nsight retained file hash/path/size mismatch')
                    with np.load(run/'checks'/path, allow_pickle=False) as output:
                        values = output['model_output'].copy()
                    if values.dtype != np.float32 or values.shape != archive['input_'+label].shape:
                        raise ValueError('Nsight output shape/dtype mismatch')
                    payload = output_payload(values, identity, coords, CASE, env['source_sha256'], env['model_state_sha256'])
                    error = verify_output(run/'checks'/path, payload, archive['reference_'+label])
                    if array_mapping_hash(payload) != saved['payload_sha256']:
                        raise ValueError('Nsight output payload mismatch')
                    info[key] = saved, error
    if set(env['outputs']) != set(info):
        raise ValueError('Missing/extra retained outputs')
    for call, actual, wanted in zip(calls, selected, expected):
        for field, value in wanted.items():
            if (int(call[field]) if isinstance(value, int) else call[field]) != value or (field != 'order' and actual[field] != value):
                raise ValueError('Nsight shuffled order/input variant mismatch')
        key = f'{wanted["variant"]}_{wanted["method"]}'
        saved, error = info[key]
        if call['file_sha256'] != saved['sha256'] or call['payload_sha256'] != saved['payload_sha256'] or int(call['file_bytes']) != saved['bytes'] or not np.isfinite(float(call['seconds'])) or float(call['seconds']) <= 0:
            raise ValueError('Nsight measured file differs from retained output')
        for field, value in error.items():
            np.testing.assert_allclose(float(call[field]), value, rtol=1e-10, atol=1e-15)
        engine = wanted['method'].split('_', 1)[0]
        if engine == 'cpu':
            if any(actual[k] for k in ['kernel_events', 'graph_launches', 'kernel_launch_apis', 'htod_bytes', 'dtoh_bytes']):
                raise ValueError('CPU root contains CUDA work')
        elif (engine == 'graph' and actual['graph_launches'] != 1) or (engine == 'eager' and (actual['graph_launches'] or actual['kernel_launch_apis'] < 1)):
            raise ValueError('Missing actual expected CUDA dispatch')
        elif complete and (actual['kernel_events'] < 1 or actual['htod_copies'] != 1 or actual['dtoh_copies'] != 1 or actual['htod_bytes'] != 1048576 or actual['dtoh_bytes'] != 1048576 or actual['device_events_outside_root']):
            raise ValueError('Incomplete/misaligned GPU work or full input/output copies')
    if complete and (sum(r['kernel_events'] for r in selected) != len(tables['CUPTI_ACTIVITY_KIND_KERNEL']) or sum(r['htod_copies']+r['dtoh_copies'] for r in selected) > len(tables['CUPTI_ACTIVITY_KIND_MEMCPY'])):
        raise ValueError('Unreconciled/duplicate GPU correlation events')
    print(f'PASS {run.name}: {len(selected)} NVTX full-file calls and outputs; status={"complete GPU trace" if complete else "PARTIAL: no GPU activity evidence"}')
    return dict(path=run, capture=capture, env=env, calls=selected, complete=complete, tables=tables, strings=strings)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    p.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    reference = validate_reference(args.reference_run, args.data, Path('results/rtx3050_physicsnemo_run01'))
    runs = [validate(r, args.data, args.reference_run, reference) for r in args.runs]
    full = [r for r in runs if r['complete']]
    if not full:
        raise ValueError('No complete Nsight GPU capture available; partial roots are not GPU evidence')
    run = full[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    for r in runs:
        write_csv(args.output/(r['path'].name+'_calls.csv'), r['calls'])
    colors = dict(zip(STAGES, ['#4c78a8', '#72b7b2', '#eeca3b', '#f58518', '#b279a2', '#54a24b']))
    methods = ['eager_open_once', 'graph_open_once', 'graph_exact_once', 'graph_gather_once']
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), layout='constrained')
    for ax, method in zip(axes.flat, methods):
        root = next(r for r in run['calls'] if r['method'] == method and r['repeat'] == 0)
        a, b = root['root_start_ns'], root['root_end_ns']
        inside = lambda r: r['start'] >= a and r['end'] <= b
        annotations = [r for r in run['tables']['NVTX_EVENTS'] if inside(r) and r['text'].startswith('file_')]
        runtime = [r for r in run['tables']['CUPTI_ACTIVITY_KIND_RUNTIME'] if inside(r)]
        correlation = {r['correlationId'] for r in runtime}
        for annotation in annotations:
            s = annotation['text'][5:]
            ax.broken_barh([((annotation['start']-a)/1e6, (annotation['end']-annotation['start'])/1e6)], (2.8, .6), facecolors=colors[s])
        launch = [r for r in runtime if 'LaunchKernel' in run['strings'][r['nameId']] or 'cudaGraphLaunch' in run['strings'][r['nameId']]]
        kernels = [r for r in run['tables']['CUPTI_ACTIVITY_KIND_KERNEL'] if r['correlationId'] in correlation]
        copies = [r for r in run['tables']['CUPTI_ACTIVITY_KIND_MEMCPY'] if r['correlationId'] in correlation]
        for records, y, color in [(launch, 1.8, '#222222'), (kernels, .8, '#666666'), (copies, -.2, '#cc3311')]:
            ax.broken_barh([((r['start']-a)/1e6, (r['end']-r['start'])/1e6) for r in records], (y, .6), facecolors=color)
        ax.set_yticks([0, 1, 2, 3], ['GPU copies', 'GPU kernels', 'CPU launch APIs', 'CPU NVTX stages'])
        ax.set(xlabel='Time from request start (ms)', title=method+'; repeat 0, input A', xlim=(0, (b-a)/1e6))
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=colors[s], label=s) for s in STAGES]+[Patch(color='#666666', label='GPU kernels'), Patch(color='#cc3311', label='GPU copies')], loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle('Real Nsight Systems CUDA/NVTX timeline: selected complete file requests\nIndependent diagnostic; WSL CUPTI timestamp conversion; CPU/device intervals overlap; not primary P50 or occupancy', fontsize=12)
    fig.savefig(args.output/'physicsnemo_nsight.png', dpi=160)
    plt.close(fig)
    text = ['# Nsight Systems: real full-file CUDA/NVTX diagnostic', '',
        'The target runs a fixed B64 untrained FNO with the canonical state and genuine A/B frames, shared-handle bounding/exact/gather readers and CPU/eager/graph engines. Every output is written/closed, read back and checked for values, shape/dtype, coordinate/time/member metadata and CPU FP64 agreement. All 18 last A/B files per capture remain. Validation/hashes/cleanup and pre-call synchronization are outside NVTX request roots.', '',
        '## Capture status', '', '| Capture | Nsight version | WSL workaround mode | NVTX calls | GPU kernels | Status |', '|---|---|---|---:|---:|---|']
    for r in runs:
        text.append(f'| {r["path"].name} | {r["capture"]["tool"]["version"]} | {r["capture"].get("config_mode", "none/environment directive")} | {len(r["calls"])} | {sum(c["kernel_events"] for c in r["calls"])} | '+('Complete GPU evidence' if r['complete'] else 'PARTIAL: CUDA API/NVTX only')+' |')
    text += ['', 'The initial 2024.5 capture and 2024.7 environment-directive attempt lack device kernel/copy tables and cannot establish a GPU timeline. The final 2024.7 capture uses CuptiUseRawGpuTimestamps=false in the configuration path returned by nsys -z, restoring the exact original file (or removing the newly created file) after collection. Node-level graph tracing is requested. CUDAProfilerStart/Stop bounds the capture; CPU sampling/context-switch tracing is disabled.', '',
        'NVIDIA documents this WSL timestamp workaround as less precise than the normal conversion. The recovered device activity is suitable for qualitative correlation here; do not treat small CPU/GPU timing offsets as exact scheduling measurements. [NVIDIA WSL troubleshooting](https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536/9), [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/).', '',
        '## Complete-capture diagnostic medians', '',
        '| Method | Calls | NVTX root ms | Read ms | Forward CPU range ms | Return CPU range ms | Kernel union ms | GPU active union ms | Kernel events/call | Kernel-launch APIs/call | Graph launch/call |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for reader in READERS:
        for engine in ENGINES:
            method = engine+'_'+reader
            selected = [c for c in run['calls'] if c['method'] == method]
            fields = ['root_ms', 'read_decode_ms', 'forward_ms', 'return_ms', 'kernel_union_ms', 'gpu_active_union_ms', 'kernel_events', 'kernel_launch_apis', 'graph_launches']
            values = [np.median([r[f] for r in selected]) for f in fields]
            text.append(f'| {method} | {len(selected)} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    text += ['', 'GPU events are matched to NVTX requests via CUDA runtime correlation IDs, not inferred only from nearby timestamps. CPU roots must contain no CUDA work. Each complete GPU root must contain real kernels, one full 1 MiB input HtoD and one full 1 MiB output DtoH. Graph roots require exactly one cudaGraphLaunch, eager roots actual kernel-launch APIs and no graph launch. Kernel counts are reconciled across the entire selected export. Internal device copies are also retained; event-category differences do not prove reduced mathematical work.', '',
        'The CPU forward range measures dispatch plus any blocking host work, while return can wait for preceding GPU work and then transfer. GPU kernels overlap these CPU intervals. A larger return range is not evidence that PCIe transfer alone is slow; launch duration, return duration and GPU activity cannot be added as independent costs. Activity interval union is not GPU occupancy.', '',
        'These calls include NVTX/tool instrumentation, a single B64 graph and a different cadence/resource history from the mixed-shape primary sessions. Their medians do not replace primary P50/P95 or provide an exact primary-stage decomposition. The selected four illustrated requests are examples, with all 36 calls retained in CSV.', '',
        '## Reproduction and event artifacts', '', '```bash', 'source .venv/physicsnemo/bin/activate',
        'python scripts/setup_nsight_cli.py --release 2024.7.1',
        'python scripts/run_physicsnemo_nsight.py --wsl-timestamps --output results/my_nsight',
        'python scripts/analyze_physicsnemo_nsight.py results/my_nsight --output results/my_nsight_report', '```', '',
        'Use new output paths; the installer preserves existing tool directories and validates package SHA256 against NVIDIA repository metadata. Only a local CLI package is extracted; drivers and Python environments are not upgraded. The workaround changes only the specific Nsight config for the collection, then restores it. Other profiling sessions should not run concurrently.', '',
        'Raw capture.nsys-rep and trace.sqlite remain local because Nsight can include process/environment metadata. events.sqlite contains only whitelisted NVTX/CUDA event tables, memcpy enums and referenced API/kernel names; metadata tables and unreferenced strings are omitted. Its manifest links its hash/table counts to the raw-export hash. The archived capture command, tool/package hashes, target source, collector logs and checked outputs are retained. CPU CI independently reconciles the selected events and output contract; it does not rerun Nsight or GPU execution.', '',
        '![Nsight full-file workflow timelines](physicsnemo_nsight.png)', '']
    (args.output/'analysis.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved checked Nsight timeline:', args.output)


if __name__ == '__main__':
    main()
