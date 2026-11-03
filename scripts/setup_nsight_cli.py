"""Extract a hash-verified NVIDIA Nsight Systems CLI into project-local tools."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


BASE = 'https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/'
RELEASES = {'2024.5.1': 'NsightSystems-linux-cli-public-2024.5.1.113-3461954.deb',
            '2024.7.1': 'NsightSystems-linux-cli-public-2024.7.1.84-3512561.deb'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=RELEASES, default='2024.7.1')
    args = parser.parse_args()
    FILENAME = RELEASES[args.release]
    ROOT = Path('.venv/tools/nsight-systems-'+args.release)
    if ROOT.exists():
        raise SystemExit('Tool directory exists; preserve it and use its recorded nsys path')
    metadata = urllib.request.urlopen(BASE+'Packages', timeout=60).read().decode()
    matches = []
    for block in metadata.split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
        if Path(fields.get('Filename', '')).name == FILENAME:
            matches.append(fields)
    if len(matches) != 1 or 'SHA256' not in matches[0]:
        raise ValueError('Unique package and official SHA256 required')
    expected = matches[0]
    downloads = Path('.venv/downloads')
    downloads.mkdir(parents=True, exist_ok=True)
    package = downloads/FILENAME
    if not package.exists():
        temporary = package.with_suffix('.deb.part')
        with urllib.request.urlopen(BASE+FILENAME, timeout=60) as source, temporary.open('xb') as target:
            while chunk := source.read(1024*1024):
                target.write(chunk)
        temporary.rename(package)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    if digest != expected['SHA256'] or package.stat().st_size != int(expected['Size']):
        raise ValueError('Official package hash/size mismatch')
    ROOT.mkdir(parents=True)
    subprocess.run(['dpkg-deb', '-x', str(package), str(ROOT)], check=True)
    executables = [p for p in ROOT.rglob('nsys') if p.is_file() and not p.is_symlink()]
    if len(executables) != 1:
        raise ValueError(f'Expected one nsys executable, found {executables}')
    executable = executables[0].resolve()
    version = subprocess.check_output([str(executable), '--version'], text=True).strip()
    record = dict(url=BASE+FILENAME, metadata_url=BASE+'Packages', sha256=digest,
        bytes=package.stat().st_size, package=expected, executable=str(executable), version=version,
        scope='Project-local package extraction; no apt install, driver change or Python dependency modification')
    (ROOT/'installation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
