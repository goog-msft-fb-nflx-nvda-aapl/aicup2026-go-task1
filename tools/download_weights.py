"""Download and verify the public fitted models, without account credentials."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'config/public_weights.json')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--check-public', action='store_true', help='Check anonymous release availability and GitHub SHA256 metadata only; do not download the full models')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    api = 'https://api.github.com/repos/goog-msft-fb-nflx-nvda-aapl/aicup2026-go-task1/releases/tags/trained-weights-20261127'
    request = urllib.request.Request(api, headers={'User-Agent': 'Go-Task1-anonymous-reproduction'})
    with urllib.request.urlopen(request, timeout=60) as response:
        published = {a['name']: a for a in json.load(response)['assets']}
    errors = []
    for asset in manifest['assets']:
        current = published.get(asset['name'])
        if current is None or current['state'] != 'uploaded':
            errors.append(asset['name'] + ': not published')
        elif current.get('digest') != 'sha256:' + asset['sha256']:
            errors.append(asset['name'] + ': unexpected checkpoint digest')
        elif asset['bytes'] is not None and current['size'] != asset['bytes']:
            errors.append(asset['name'] + ': unexpected size')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('ANONYMOUS_RELEASE_METADATA_VERIFIED', len(manifest['assets']), flush=True)
    if args.check_public:
        return
    if args.directory is None:
        parser.error('--directory is required for downloads')
    args.directory.mkdir(parents=True, exist_ok=True)

    def download(asset):
        target = args.directory / asset['name']
        if target.exists() and digest(target) == asset['sha256']:
            return 'VERIFIED_EXISTING ' + asset['name']
        part = target.with_name(target.name + '.partial')
        offset = part.stat().st_size if part.exists() else 0
        headers = {'User-Agent': 'Go-Task1-anonymous-reproduction'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(manifest['download_base'] + asset['name'], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise RuntimeError('Unexpected range response for ' + asset['name'])
            with part.open('ab' if append else 'wb') as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
        if digest(part) != asset['sha256']:
            raise RuntimeError('SHA256 mismatch: ' + asset['name'])
        part.replace(target)
        return 'DOWNLOADED_AND_HASH_VERIFIED ' + asset['name']

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(download, manifest['assets']):
            print(result, flush=True)
    print('ALL_PUBLIC_WEIGHT_DOWNLOADS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
