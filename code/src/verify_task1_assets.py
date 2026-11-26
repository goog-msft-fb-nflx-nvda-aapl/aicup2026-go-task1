"""Verify each private release asset against release_assets.json."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--manifest', default='release_assets.json')
args = parser.parse_args()
manifest = json.loads((args.directory / args.manifest).read_text())
for item in manifest['assets']:
    path = args.directory / item['name']
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise RuntimeError(f'Artifact checksum mismatch: {path.name}')
    print('Verified', path.name, flush=True)
