"""Capture the checked NVTX target and export a real Nsight Systems SQLite report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsight_trace import export_events


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--installation', type=Path, default=Path('.venv/tools/nsight-systems-2024.7.1/installation.json'))
    parser.add_argument('--wsl-timestamps', action='store_true', help='Temporarily set official Nsight WSL config, restoring exact original bytes afterwards')
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('New output and >=2 repetitions required')
    install = json.loads(args.installation.read_text())
    nsys = install['executable']
    args.output.mkdir(parents=True)
    environment = dict(os.environ)
    config_path, original = None, None
    if args.wsl_timestamps:
        config_path = Path(subprocess.check_output([nsys, '-z'], text=True).strip()).resolve()
        expected_parent = (Path.home()/'.config/NVIDIA Corporation').resolve()
        if config_path.parent != expected_parent or config_path.name != 'nsys-config.ini':
            raise ValueError('Unexpected Nsight config path; preserve it')
        original = config_path.read_bytes() if config_path.exists() else None
        content = original.decode() if original is not None else ''
        content = re.sub(r'(?m)^\s*CuptiUseRawGpuTimestamps\s*=.*$', '', content)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(content.rstrip()+'\nCuptiUseRawGpuTimestamps=false\n')
    target = Path('physicsnemo_nsight.py').resolve()
    shutil.copyfile(target, args.output/'target_source.py')
    command = [nsys, 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
        '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=false',
        '--output='+str((args.output/'capture').resolve()), sys.executable, str(target),
        '--output', str((args.output/'checks').resolve()), '--repeats', str(args.repeats), '--capture']
    commands = [command]
    try:
        result = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        if config_path is not None:
            if original is None:
                config_path.unlink()
            else:
                config_path.write_bytes(original)
    (args.output/'capture.log').write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    report = args.output/'capture.nsys-rep'
    sqlite = args.output/'trace.sqlite'
    export = [nsys, 'export', '--type=sqlite', '--force-overwrite=false', '--output='+str(sqlite.resolve()), str(report.resolve())]
    commands.append(export)
    result = subprocess.run(export, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.output/'export.log').write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    env = dict(tool=install, target_sha256=digest(target), commands=commands,
        wsl_timestamp_workaround=args.wsl_timestamps,
        config_directives='CuptiUseRawGpuTimestamps=false' if args.wsl_timestamps else None,
        config_mode='Temporary official config file; original restored before export' if args.wsl_timestamps else None,
        report_sha256=digest(report), sqlite_sha256=digest(sqlite),
        scope='Real Nsight Systems CUDA/NVTX capture; no CPU sampling/context-switch tracing; independent diagnostic, not primary timing',
        capture_log_sha256=digest(args.output/'capture.log'), export_log_sha256=digest(args.output/'export.log'))
    (args.output/'capture_environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    export_events(args.output)
    with sqlite3.connect(sqlite) as db:
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables or 'CUPTI_ACTIVITY_KIND_MEMCPY' not in tables:
            raise ValueError('Partial Nsight capture: missing actual GPU kernel/copy tables; do not claim complete GPU tracing')
    print('PASS: genuine Nsight Systems report and SQLite export saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
