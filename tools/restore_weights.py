"""Restore verified public weights into a new inference workspace; never train."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None
    def readable(self):
        return True
    def read(self, n=-1):
        result = bytearray()
        while n < 0 or len(result) < n:
            if self.stream is None:
                try:
                    self.stream = next(self.paths).open('rb')
                except StopIteration:
                    break
            block = self.stream.read(-1 if n < 0 else n - len(result))
            if not block:
                self.stream.close()
                self.stream = None
            else:
                result.extend(block)
        return bytes(result)
    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'config/public_weights.json')
    args = parser.parse_args()
    assets = json.loads(args.manifest.read_text())['assets']
    for asset in assets:
        path = args.directory / asset['name']
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise SystemExit('Missing or different weight download: ' + str(path))
    root = args.root.resolve()
    # Existing data is allowed; existing model outputs are not overwritten.
    for directory in ['results', 'reproduction', 'katago']:
        if (root / directory).exists():
            raise SystemExit('Use a fresh workspace; model directory already exists: ' + str(root / directory))
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    for group, mode in [('parent', 'r|gz'), ('gate', 'r|')]:
        paths = [args.directory / a['name'] for a in sorted(assets, key=lambda x: x['name']) if a.get('archive_group') == group]
        with Parts(paths) as stream, tarfile.open(fileobj=stream, mode=mode) as archive:
            for member in archive:
                target = (root / member.name).resolve()
                if not target.is_relative_to(root / 'results'):
                    raise RuntimeError('Unexpected model archive path: ' + member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or target.suffix not in ['.pt', '.joblib']:
                    raise RuntimeError('Unexpected model archive member: ' + member.name)
                if target.exists():
                    raise RuntimeError('Duplicate model archive member: ' + member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                count += 1
    for asset in assets:
        relative = asset.get('restore_path')
        if asset['name'] == 'KATAGO_NETWORK_LICENSE.txt':
            relative = 'katago/KATAGO_NETWORK_LICENSE.txt'
        if relative:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.directory / asset['name'], target)
            count += 1
    (root / 'public_weights_restored.json').write_text(json.dumps({'manifest': assets, 'restored_model_and_license_files': count}, indent=2) + '\n')
    print('PUBLIC_WEIGHTS_RESTORED', count)


if __name__ == '__main__':
    main()
