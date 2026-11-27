"""Exercise persistent-stage completion and duplicate-worker protection, with no models/data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root = Path(tempfile.mkdtemp(prefix='go_stage_controller_'))
script = Path(__file__).resolve().parents[1] / 'tools/infer.py'
artifact, counter, release = [root / name for name in ['output', 'counter', 'release']]
code = ("from pathlib import Path;import time;"
        "Path(" + repr(str(counter)) + ").open('a').write('executed\\n');"
        "exec(\"while not Path(" + repr(str(release)) + ").exists(): time.sleep(.02)\");"
        "Path(" + repr(str(artifact)) + ").write_text('complete')")
spec = {'name': 'test', 'command': [sys.executable, '-c', code], 'outputs': [str(artifact)],
        'receipt': str(root / 'stage.receipt.json'), 'log': str(root / 'stage.log'),
        'cwd': str(root), 'environment': {}}
path = root / 'stage.spec.json'
path.write_text(json.dumps(spec))
first = subprocess.Popen([sys.executable, str(script), '--stage-worker', str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    deadline = time.monotonic() + 10
    while not Path(spec['receipt']).exists():
        if first.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('First worker failed to start')
        time.sleep(.02)
    second = subprocess.run([sys.executable, str(script), '--stage-worker', str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    assert second.returncode != 0, 'A duplicate worker must not run the command'
    release.touch()
    assert first.wait(timeout=10) == 0
finally:
    release.touch()
    if first.poll() is None:
        first.terminate()
        first.wait(timeout=10)
record = json.loads(Path(spec['receipt']).read_text())
assert counter.read_text().splitlines() == ['executed']
assert record['status'] == 'complete'
assert record['output_hashes'][str(artifact)] == hashlib.sha256(artifact.read_bytes()).hexdigest()

# A command returning zero without its declared artifact must still fail.
bad = dict(spec, name='missing_output', command=[sys.executable, '-c', 'pass'],
           outputs=[str(root / 'missing')], receipt=str(root / 'bad.receipt.json'))
path = root / 'bad.spec.json';path.write_text(json.dumps(bad))
result = subprocess.run([sys.executable, str(script), '--stage-worker', str(path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
assert result.returncode != 0
assert json.loads(Path(bad['receipt']).read_text())['status'] == 'failed'
print('STAGE_CONTROLLER_CHECKS_PASSED: duplicate exclusion, receipts, output hashes, missing-output failure')
