"""Reproduce the verified 0.8630397 candidate with the frozen verifier removed."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--mode', choices=['inference', 'artifacts'], default='artifacts')
args = parser.parse_args()
root = args.root.resolve()
environment = dict(os.environ, GO_ROOT=str(root), PYTHONHASHSEED='0')
subprocess.run([sys.executable, str(root / 'src/codex_build_review_candidate_1118.py'),
                '--root', str(root), '--mode', args.mode], cwd=root, env=environment, check=True)
parent = root / 'sub/go19_cx38.csv'
parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_hash == '0b34e38881dc40ba224d0e188f678d8d2f68b6bf3d76565ed3a083dfbe9ff5ca'
csv.field_size_limit(10**9)
rows = list(csv.reader(parent.open(newline='')))
replacement = {}
for pool, suffix in [('pub2', 'public'), ('priv2', 'private')]:
    scores = np.load(root / 'results' / f'Sfinal_review_candidate_{pool}.npy')
    player_ids = np.load(root / 'results' / f'S_{pool}_pids.npy')
    questions = list(csv.DictReader((root / 'data/tests1v2/tests' /
                                    f'player_identification_test2_{suffix}.csv').open()))
    assert scores.shape == (400, 600) and np.isfinite(scores).all()
    assert len(questions) == 400 and len(set(player_ids)) == 600
    for question, order in zip(questions, np.argsort(-scores, axis=1)[:, :5]):
        question_id = question['question_id']
        replacement[question_id] = [question_id] + [str(player_ids[index]) for index in order]
candidate = [rows[0]] + [replacement.get(row[0], row) for row in rows[1:]]
assert len(candidate) == 1601 and [row[0] for row in candidate] == [row[0] for row in rows]
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in candidate[1:])
assert all(left == right for left, right in zip(candidate[1:], rows[1:])
           if left[0].startswith(('pub_q_', 'priv_q_')))
output = root / 'sub/go19_nover.csv'
with output.open('w', newline='') as handle:
    csv.writer(handle).writerows(candidate)
output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
assert output_hash == 'abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
record = {'mode': args.mode, 'verified': True, 'file': str(output), 'sha256': output_hash,
          'public_score': .8630397, 'parent_sha256': parent_hash,
          'changed_top1': sum(left[1] != right[1] for left, right in zip(candidate[1:], rows[1:])),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root / 'results/production_noverifier_verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
