"""Paired bounding/exact native-frame reader timings and separate Store get probes."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import xarray as xr

from physicsnemo_benchmark import array_mapping_hash, prepare_input, stats
from physicsnemo_graph import CASES
from physicsnemo_graph_pipeline import READERS, read_variant, request_order
from physicsnemo_reader import READER_POLICIES, probe_reads, read_coverage
from weather_verify import sha256, write_csv


def reader_order(readers, repeats, seed):
    return [dict(case=j['case'], reader=j['method'].split('_', 1)[1], repeat=j['repeat'], order=j['order'], variant=j['variant'])
            for j in request_order(['cpu'], readers, repeats, seed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ecmwf_batch_case/batch.zarr'))
    parser.add_argument('--reference-run', type=Path, default=Path('results/rtx3050_physics_graph_run01'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--readers', nargs='+', choices=READERS, default=READERS)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261113)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2 or args.warmup < 1 or args.seed < 1:
        parser.error('New output and positive seed/warmup, >=2 repetitions required')
    reader_order(args.readers, 1, args.seed)  # Reject duplicates before creating output.
    source = sha256(args.data)
    manifest = json.loads((args.data.parent/'manifest.json').read_text())
    ref = json.loads((args.reference_run/'environment.json').read_text())
    if source != manifest['files']['batch.zarr'] or source != ref['source_sha256']:
        raise ValueError('Source/reference provenance mismatch')
    args.output.mkdir(parents=True)
    cases = {c['case']: c for c in CASES}
    fixtures, coverage, probes = {}, {}, {}
    for case in CASES:
        name = case['case']
        canonical = ref['case_metadata'][name]
        path = args.reference_run/canonical['reference_path']
        if sha256(path) != canonical['reference_sha256']:
            raise ValueError('Pinned reference changed')
        with np.load(path, allow_pickle=False) as saved:
            fixtures[name] = []
            for v, label in enumerate(['A', 'B']):
                raw, identity, coords = read_variant(args.data, case, v)
                np.testing.assert_array_equal(prepare_input(raw), saved['input_'+label])
                fixtures[name].append(dict(input_hash=array_mapping_hash({'input': saved['input_'+label]}),
                    identity=identity, coordinates={k: x.tolist() for k, x in coords.items()}, reference_sha256=sha256(path)))
        coverage[name], probes[name] = {}, {}
        for selection in sorted({READER_POLICIES[r]['selection'] for r in args.readers}):
            coverage[name][selection] = [read_coverage(args.data, case, v, selection) for v in [0, 1]]
            probes[name][selection] = [probe_reads(args.data, case, v, selection) for v in [0, 1]]
    setup = time.perf_counter()
    dataset = xr.open_zarr(args.data, chunks=None, consolidated=True) if any(READER_POLICIES[r]['reuse'] for r in args.readers) else None
    reader_setup = time.perf_counter()-setup
    def trial(job):
        case, variant = cases[job['case']], job['variant']
        policy = READER_POLICIES[job['reader']]
        start = time.perf_counter_ns()
        raw, identity, coords = read_variant(args.data, case, variant, dataset if policy['reuse'] else None, selection=policy['selection'])
        seconds = (time.perf_counter_ns()-start)/1e9
        expected = fixtures[job['case']][variant]
        digest = array_mapping_hash({'input': prepare_input(raw)})
        if identity != expected['identity'] or digest != expected['input_hash']:
            raise ValueError('Reader changed genuine source identity/values')
        for k, value in coords.items():
            np.testing.assert_array_equal(value, expected['coordinates'][k])
        backing = raw
        while isinstance(backing.base, np.ndarray):
            backing = backing.base
        return dict(**job, seconds=seconds, input_sha256=digest, raw_bytes=raw.nbytes, backing_array_bytes=backing.nbytes)
    try:
        for job in reader_order(args.readers, args.warmup, args.seed-1):
            trial(job)
        samples = [trial(j) for j in reader_order(args.readers, args.repeats, args.seed)]
    finally:
        if dataset is not None:
            dataset.close()
    summary = [dict(case=c['case'], reader=r, **stats([s for s in samples if s['case'] == c['case'] and s['reader'] == r])) for c in CASES for r in args.readers]
    write_csv(args.output/'timing_samples.csv', samples)
    write_csv(args.output/'timing_summary.csv', summary)
    (args.output/'store_probes.json').write_text(json.dumps(probes, indent=2, allow_nan=False), encoding='utf-8')
    env = dict(captured_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        versions={k: importlib.metadata.version(k) for k in ['numpy', 'xarray', 'zarr']},
        source_sha256=source, source_manifest=manifest, reference_run=args.reference_run.name,
        cases=CASES, readers=args.readers, reader_policies={r: READER_POLICIES[r] for r in args.readers},
        repeats=args.repeats, warmup=args.warmup, seed=args.seed, shared_reader_setup_seconds=reader_setup,
        fixtures=fixtures, coverage=coverage, store_probes_sha256=sha256(args.output/'store_probes.json'),
        timing='Direct wall read/select/decode/coordinate validation only; reopen includes open/close, once readers share a handle. FP32 packing, input/identity/coordinate checks, hashes, probes and CSV/JSON writing are outside primary timing. Genuine A/B per repeat, shuffled case/reader ordering, no prepared-input cache.',
        probe='Separate instrumented fresh-handle LocalStore get observations with all returned key/byte records. Counts forecast compressed bytes returned by Store API, not actual disk/network traffic or decode invocation counts. Not used as primary timings. Logical selected forecast array/backing bytes exclude temporary decoder/indexer arrays and are not peak process RAM.',
        limits='No cold-cache claim; setup/probes/validation warm OS/library state. Source chunks/codec are unchanged. No model execution, remote download, training or forecast-quality evaluation in reader-only timings.')
    (args.output/'environment.json').write_text(json.dumps(env, indent=2, allow_nan=False), encoding='utf-8')
    print(f'PASS: {len(samples)} paired read-only observations; exact identities/values and actual Store chunk gets checked', flush=True)


if __name__ == '__main__':
    main()
