"""Selected Nsight SQLite event export and CPU/GPU correlation accounting."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3

from physicsnemo_pipeline import STAGES
from weather_dispatch import merged_duration
from weather_verify import sha256

TABLES = ['NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL',
          'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET', 'ENUM_CUDA_MEMCPY_OPER']
STRING_FIELDS = {'NVTX_EVENTS': ['textId'], 'CUPTI_ACTIVITY_KIND_RUNTIME': ['nameId'],
                 'CUPTI_ACTIVITY_KIND_KERNEL': ['demangledName', 'shortName', 'mangledName']}


def export_events(run):
    run = Path(run)
    output = run/'events.sqlite'
    if output.exists():
        raise ValueError('New selected-event export required')
    with closing(sqlite3.connect(run/'trace.sqlite')) as source, closing(sqlite3.connect(output)) as target:
        available = {r[0] for r in source.execute("select name from sqlite_master where type='table'")}
        counts, references = {}, set()
        for table in TABLES:
            if table not in available:
                continue
            columns = list(source.execute(f'pragma table_info({table})'))
            names = [c[1] for c in columns]
            declaration = ', '.join('"'+c[1].replace('"', '""')+'" '+(c[2] if c[2] in ['INTEGER', 'REAL', 'TEXT'] else 'TEXT') for c in columns)
            target.execute(f'CREATE TABLE {table} ({declaration})')
            records = list(source.execute(f'SELECT * FROM {table}'))
            if table == 'NVTX_EVENTS':
                text_index = names.index('text')
                records = [r for r in records if r[text_index] and (r[text_index].startswith('request/') or r[text_index] in ['file_'+s for s in STAGES])]
            for key in STRING_FIELDS.get(table, []):
                if key in names:
                    references.update(r[names.index(key)] for r in records if r[names.index(key)] is not None)
            target.executemany(f'INSERT INTO {table} VALUES ('+','.join('?' for _ in names)+')', records)
            counts[table] = len(records)
        target.execute('CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
        strings = [r for r in source.execute('SELECT id,value FROM StringIds') if r[0] in references]
        if any('\x00' in r[1] for r in strings):
            raise ValueError('Unexpected environment-like string in referenced events')
        target.executemany('INSERT INTO StringIds VALUES (?,?)', strings)
        counts['StringIds'] = len(strings)
        target.commit()
    record = dict(source_sqlite_sha256=sha256(run/'trace.sqlite'), events_sha256=sha256(output), counts=counts,
        scope='Only NVTX workflow labels, CUDA runtime/kernel/copy events, memcpy enum and referenced API/kernel strings. Process/environment/system metadata and unreferenced strings omitted. Original raw report/export remains local.')
    (run/'event_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def read_events(path):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        names = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if names-set(TABLES+['StringIds']):
            raise ValueError('Unexpected metadata tables in selected-event artifact')
        tables = {t: [dict(r) for r in db.execute(f'SELECT * FROM {t}')] for t in names}
    strings = {r['id']: r['value'] for r in tables['StringIds']}
    referenced = {r[k] for t, keys in STRING_FIELDS.items() for r in tables.get(t, []) for k in keys if r.get(k) is not None}
    if set(strings) != referenced or len(strings) != len(tables['StringIds']):
        raise ValueError('Missing/extra/duplicate event strings')
    if any('\x00' in s for s in strings.values()):
        raise ValueError('Environment-like string in selected event artifact')
    return tables, strings


def summarize_events(path):
    tables, strings = read_events(path)
    roots = sorted([r for r in tables['NVTX_EVENTS'] if (r.get('text') or '').startswith('request/')], key=lambda r: r['start'])
    if not roots or any(r['end'] is None or r['end'] <= r['start'] for r in roots):
        raise ValueError('Complete NVTX request roots required')
    copies = {r['id']: r['name'] for r in tables.get('ENUM_CUDA_MEMCPY_OPER', [])}
    complete = bool(tables.get('CUPTI_ACTIVITY_KIND_KERNEL')) and bool(tables.get('CUPTI_ACTIVITY_KIND_MEMCPY'))
    result = []
    for root in roots:
        _, case, method, repeat, variant = root['text'].split('/')
        a, b = root['start'], root['end']
        annotations = [r for r in tables['NVTX_EVENTS'] if r['globalTid'] == root['globalTid'] and r['end'] is not None and r['start'] >= a and r['end'] <= b and r['text'].startswith('file_')]
        runtime = [r for r in tables['CUPTI_ACTIVITY_KIND_RUNTIME'] if r['globalTid'] == root['globalTid'] and r['start'] >= a and r['end'] <= b]
        correlation = {r['correlationId'] for r in runtime if r['correlationId'] is not None}
        kernels = [r for r in tables.get('CUPTI_ACTIVITY_KIND_KERNEL', []) if r['correlationId'] in correlation]
        memcpy = [r for r in tables.get('CUPTI_ACTIVITY_KIND_MEMCPY', []) if r['correlationId'] in correlation]
        stage_times = {s+'_ms': sum(r['end']-r['start'] for r in annotations if r['text'] == 'file_'+s)/1e6 for s in STAGES}
        if any(sum(r['text'] == 'file_'+s for r in annotations) != (0 if s == 'upload' and method.startswith('cpu_') else 1) for s in STAGES):
            raise ValueError('Missing/extra stage ranges')
        htod = [r for r in memcpy if copies[r['copyKind']] == 'CUDA_MEMCPY_KIND_HTOD']
        dtoh = [r for r in memcpy if copies[r['copyKind']] == 'CUDA_MEMCPY_KIND_DTOH']
        intervals = [(r['start'], r['end']) for r in kernels+memcpy]
        launches = [r for r in runtime if 'LaunchKernel' in strings[r['nameId']]]
        graph = [r for r in runtime if 'cudaGraphLaunch' in strings[r['nameId']]]
        result.append(dict(case=case, method=method, repeat=int(repeat), variant=int(variant),
            root_start_ns=a, root_end_ns=b, root_ms=(b-a)/1e6,
            kernel_events=len(kernels), graph_launches=len(graph), kernel_launch_apis=len(launches),
            htod_copies=len(htod), htod_bytes=sum(r['bytes'] for r in htod), dtoh_copies=len(dtoh), dtoh_bytes=sum(r['bytes'] for r in dtoh),
            kernel_union_ms=merged_duration([(r['start'], r['end']) for r in kernels], a, b)/1e6,
            gpu_active_union_ms=merged_duration(intervals, a, b)/1e6,
            launch_api_ms=sum(r['end']-r['start'] for r in launches+graph)/1e6,
            device_events_outside_root=sum(r['start'] < a or r['end'] > b for r in kernels+memcpy), **stage_times))
    return result, complete, tables, strings
