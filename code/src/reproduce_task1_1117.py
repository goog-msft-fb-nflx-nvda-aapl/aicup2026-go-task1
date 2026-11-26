"""Reproduce the 2026-11-17 Task 1 submission in a user-selected GO_ROOT."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', required=True, type=Path)
parser.add_argument('--config', type=Path)
parser.add_argument('--mode', choices=['artifacts', 'inference'], default='artifacts')
args = parser.parse_args()
root = args.root.resolve()
source = Path(__file__).resolve().parent
config_path = args.config or source.parent / 'reproduction/task1_1117_config.json'
config = json.loads(config_path.read_text())
frozen_verifier = config.get('frozen_verifier_file')
output_name = config.get('submission_filename', 'go_c3334_1117.csv')
for folder in ['src', 'results', 'logs', 'sub']:
    (root / folder).mkdir(parents=True, exist_ok=True)
for path in source.glob('*.py'):
    target = root / 'src' / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
environment = dict(os.environ, GO_ROOT=str(root), PYTHONHASHSEED='0')

def run(script, *arguments, extra_env=None):
    current_env = dict(environment, **(extra_env or {}))
    command = [sys.executable, str(root / 'src' / script), *map(str, arguments)]
    print('Running', ' '.join(command), flush=True)
    subprocess.run(command, cwd=root, env=current_env, check=True)

sample = root / 'data/tests1v2/tests/sample_submission_player_v2.csv'
if not sample.exists():
    raise FileNotFoundError('Place the official Task 1 v2 test package under GO_ROOT/data/tests1v2/tests.')
shutil.copy2(sample, root / 'sub/base.csv')

if args.mode == 'inference':
    if not frozen_verifier and not (root / 'data/train_arr.npz').exists():
        run('prep.py')
    run('id_final6.py', 'id_mlp1000_6_2048_0.2_0.3_0.1_1*.pt', 'x', 'x',
        'sub/base.csv', 'sub/rebuilt_views.csv', 1, 0, .25, 1, .7)
    run('rank_lp_pool.py')
    run('rank_lp_pool_old.py')
    if config.get('new_rank_gate_script'):
        run(config['new_rank_gate_script'])
    for tag in config['production_members']:
        run('codex_idft_new_pools_1118.py' if tag in config.get('new_pool_only_members', []) else 'idft_pools.py', tag)
        if tag in config['original_members']:
            run('idft_qemb.py', tag)
        run('idft_games.py', tag)
    if not frozen_verifier:
        run('holdout_embeds.py', *config['verifier_training_members'])

# Recompute the original-family predictions from the original verQ ensemble.
# This avoids using any previously generated prediction CSV as an input.
original_members = ','.join(config['original_members'])
run('ens_sidft.py', 'reproduceQ1117', *config['original_weighted_members'])
run('mk_gate5.py', 'sub/base.csv', 'sub/original_family_1117.csv',
    2, 3, 1.4, 1., 'reproduceQ1117', 3, 1, .6, .5, .5, original_members, 0,
    extra_env={'SFINAL': 'reproduceQ1117'})
run(config.get('ensemble_script', 'ens_sidft.py'), 'reproduceS1117', *config['weighted_members'])
run(config.get('fusion_script', 'mk_gate5.py'), 'sub/base.csv', 'sub/gate_1117.csv',
    2, 3, 1.4, 1., 'reproduceS1117', 3, 1, .6, .5, .5, original_members, 0,
    extra_env={'SFINAL': 'reproduceS1117'})
verifier_env = {'VERIFIER_EXPORT_DIR': str(root / 'results/verifier_export_1117')}
if frozen_verifier:
    verifier_env['VERIFIER_MODEL_FILE'] = str(root / frozen_verifier)
run('codex_frozen_verifier_1118.py' if frozen_verifier else 'mk_verifier.py',
    'sub/original_family_1117.csv', 'sub/' + output_name,
    'reproduceS1117', ','.join(config['verifier_training_members']),
    ','.join(config['production_members']), 1.,
    extra_env=verifier_env)
output = root / 'sub' / output_name
digest = hashlib.sha256(output.read_bytes()).hexdigest()
with output.open(newline='') as handle:
    rows = list(csv.reader(handle))
assert len(rows) == 1601
assert all(len(row) == 6 and len(set(row[1:])) == 5 for row in rows[1:])
if digest != config['expected_submission_sha256']:
    raise RuntimeError(f'CSV hash mismatch: {digest}; expected {config["expected_submission_sha256"]}.')
report = {'mode': args.mode, 'output': str(output), 'sha256': digest,
          'expected_public_score': config['public_score'], 'rows': len(rows)-1,
          'python': sys.version, 'verified': True}
(root / 'results' / config.get('verification_filename', 'reproduction_1117_verification.json')).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
