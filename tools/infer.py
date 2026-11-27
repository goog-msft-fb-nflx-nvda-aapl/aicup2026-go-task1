"""Complete saved-weight Task 1 inference. No training, uploads or saved prediction inputs.

The default run requires a fresh workspace containing restored model weights,
KataGo source, and the nine official test CSVs. --plan needs none of these.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
POOLS = ['pub', 'priv', 'pub2', 'priv2']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stages(root, config, output):
    def stage(name, script, arguments, outputs):
        return {'name': name, 'command': [sys.executable, str(root / 'src' / script), *map(str, arguments)],
                'outputs': [str(root / p) for p in outputs]}
    result = [stage('base_views', 'id_final6.py',
                    ['id_mlp1000_6_2048_0.2_0.3_0.1_1*.pt', 'x', 'x', 'sub/base.csv',
                     'sub/rebuilt_views.csv', 1, 0, .25, 1, .7],
                    [f'results/S_{p}{suffix}.npy' for p in POOLS for suffix in ['', '_pids']] + ['sub/rebuilt_views.csv']),
              stage('original_rank', 'rank_lp_pool_old.py', [],
                    [f'results/LP{k}_{p}.npy' for p in POOLS[:2] for k in ['c', 'q']]),
              stage('updated_rank', 'codex_rank_gate_1118.py', [],
                    [f'results/LP{k}_codex1118_{p}.npy' for p in POOLS[2:] for k in ['c', 'q']])]
    for tag in config['production_members'] + [config['new_member']]:
        if tag in ['cx38_selected', config['new_member']]:
            script = 'codex_infer_ablation_1118.py'
            arguments = ['--root', root, '--tag', tag]
            pools = POOLS[2:]
        elif tag in config.get('new_pool_only_members', []):
            script = 'codex_idft_new_pools_1118.py'
            arguments = [tag]
            pools = POOLS[2:]
        else:
            script = 'idft_pools.py'
            arguments = [tag]
            pools = POOLS
        outputs = [f'results/Sidft_{tag}_{pool}.npy' for pool in pools]
        if tag in config['original_members']:
            outputs += [f'results/Qemb_{tag}_{pool}.npy' for pool in POOLS[2:]]
        result.append(stage('identity_' + tag, script, arguments, outputs))
    for pool in POOLS[2:]:
        result.append(stage('policy_' + pool, 'codex_policy_test_features_1119.py',
                            ['--root', root, '--pool', pool],
                            [f'reproduction/policy_test1119/{pool}_features.npz',
                             f'reproduction/policy_test1119/{pool}_summary.json']))
    result.append(stage('accepted_fusion', 'inference_final_fusion.py',
                        ['--root', root, '--config', root / 'inference/config/coordinate_fusion.json',
                         '--output', output, '--device', 'cpu'],
                        [str(output), str(output.parent / 'prediction_verification.json')]))
    return result


def model_paths(config):
    parent = json.loads((REPOSITORY / 'config/parent_checkpoint_configs.json').read_text())
    models = [p for p in parent if '/id_mlp' in p]
    models += ['results/rank_svd1000.joblib']
    models += [f'results/lm_lm{i}.pt' for i in range(1, 7)]
    models += [f'results/ft_ft{i}_sord8.pt' for i in range(4, 10)]
    models += [f'results/idft_{tag}.pt' for tag in config['production_members'] + [config['new_member']]]
    models += ['katago/humanv0.ckpt', 'reproduction/policy_metric1119/policy_metric.pt',
               'reproduction/policy_expanded_lda1122/policy_lda.npz']
    return sorted(set(models))


def preflight(root, config, gpu):
    subprocess.run([sys.executable, str(REPOSITORY / 'tools/check_official_data.py'),
                    '--root', str(root), '--inventory', str(REPOSITORY / 'config/official_input_hashes.json'),
                    '--tests-only'], check=True)
    models = model_paths(config)
    missing = [p for p in models if not (root / p).is_file()]
    if missing:
        raise RuntimeError('Restore public weights before inference: ' + ', '.join(missing))
    pins = json.loads((REPOSITORY / 'config/reproduction_contract.json').read_text())
    kgsrc = root / 'katago/kgsrc'
    commit = subprocess.check_output(['git', '-C', str(kgsrc), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != pins['katago_commit']:
        raise RuntimeError('KataGo source commit differs from the report')
    # Ensure runtime source changes are not hidden behind an otherwise matching Git HEAD.
    if subprocess.check_output(['git', '-C', str(kgsrc), 'diff', '--name-only', 'HEAD'], text=True).strip():
        raise RuntimeError('KataGo source has local modifications')
    original = json.loads((REPOSITORY / 'config/public_weights.json').read_text())
    for asset in original['assets']:
        if asset.get('restore_path'):
            path = root / asset['restore_path']
            if sha(path) != asset['sha256']:
                raise RuntimeError('Wrong model version: ' + asset['name'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    tensor = torch.arange(32, device='cuda', dtype=torch.float32)
    if (tensor * tensor).sum().item() != 10416:
        raise RuntimeError('Real GPU tensor smoke test failed')
    files = models + [r['path'] for r in json.loads((REPOSITORY / 'config/official_input_hashes.json').read_text())['files']
                     if r['path'].startswith('data/tests1v2/') and r['path'].endswith('.csv')
                     and 'rank_prediction' not in r['path']]
    return {'input_hashes': {p: sha(root / p) for p in files},
            'source_hashes': {str(p.relative_to(REPOSITORY)): sha(p)
                              for folder in ['code/src', 'config'] for p in sorted((REPOSITORY / folder).glob('*')) if p.is_file()},
            'controller_sha256': sha(Path(__file__).resolve()),
            'katago_commit': commit, 'gpu': str(gpu), 'gpu_name': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'python': sys.version}


def stage_active(receipt):
    # A stage-specific flock cannot be confused with a recycled process ID.
    path = receipt.with_suffix('.worker.lock')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def stage_worker(spec_path):
    spec = json.loads(spec_path.read_text())
    receipt = Path(spec['receipt'])
    worker_lock = receipt.with_suffix('.worker.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running = {'status': 'running', 'pid': os.getpid(), 'started': time.time(), 'stage': spec['name']}
    write_json(receipt, running)
    environment = dict(os.environ, **spec['environment'])
    with Path(spec['log']).open('ab') as log:
        try:
            process = subprocess.run(spec['command'], cwd=spec['cwd'], env=environment, stdout=log, stderr=subprocess.STDOUT)
            running.update(returncode=process.returncode, finished=time.time())
            if process.returncode:
                running['status'] = 'failed'
            else:
                absent = [p for p in spec['outputs'] if not Path(p).is_file()]
                if absent:
                    raise RuntimeError('Stage did not create expected outputs: ' + ', '.join(absent))
                running.update(status='complete', output_hashes={p: sha(Path(p)) for p in spec['outputs']})
        except Exception as error:
            running.update(status='failed', error=str(error), finished=time.time())
            write_json(receipt, running)
            raise
    write_json(receipt, running)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--gpu', default='0', help='One GPU index, passed as CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true', help='Print all inference stages; no data, models, Torch or mutations needed')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse only recorded, hash-verified outputs; monitor confirmed live stages')
    parser.add_argument('--stage-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage_worker:
        stage_worker(args.stage_worker)
        return
    if args.root is None:
        parser.error('--root is required')
    if not args.gpu.isdigit():
        parser.error('--gpu must be one nonnegative GPU index')
    root = args.root.resolve()
    if root == REPOSITORY or root.is_relative_to(REPOSITORY / 'code'):
        parser.error('Use a separate restored inference workspace')
    output = (args.output or root / 'sub/g23_greplace.csv').resolve()
    if not output.is_relative_to(root / 'sub'):
        parser.error('--output must be inside the workspace sub/ directory')
    config = json.loads((REPOSITORY / 'config/coordinate_fusion.json').read_text())
    plan = stages(root, config, output)
    if args.plan:
        print(json.dumps({'inference_only': True, 'training_required': False,
                          'required_models': model_paths(config), 'stages': plan}, indent=2))
        return
    fingerprint = preflight(root, config, args.gpu)
    fingerprint['output'] = str(output)
    fingerprint['stage_plan'] = plan
    if args.preflight_only:
        print('INFERENCE_PREFLIGHT_PASSED', flush=True)
        return
    control = root / 'inference'
    control.mkdir(exist_ok=True)
    lock = (control / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = control / 'run_inputs.json'
    if identity.exists():
        if not args.resume:
            raise RuntimeError('Existing run: inspect its logs and use --resume, or choose a fresh workspace')
        if json.loads(identity.read_text()) != fingerprint:
            raise RuntimeError('Inputs, model files, source, GPU or environment changed since the run started')
    else:
        initialization = control / 'initializing.json'
        initializing = initialization.exists()
        if initializing and (not args.resume or json.loads(initialization.read_text()) != fingerprint):
            raise RuntimeError('Interrupted initialization: use --resume with identical inputs')
        # No inherited scores, features or prediction rows may enter the clean run.
        inherited = list((root / 'results').glob('*.npy')) + list((root / 'sub').glob('*.csv'))
        if (root / 'reproduction/policy_test1119').exists():
            inherited.append(root / 'reproduction/policy_test1119')
        if initializing:
            sample = root / 'data/tests1v2/tests/sample_submission_player_v2.csv'
            inherited = [p for p in inherited if not (p == root / 'sub/base.csv' and sha(p) == sha(sample))]
        if inherited:
            raise RuntimeError('Fresh inference requires an empty generated-output area: ' + str(inherited[:5]))
        write_json(initialization, fingerprint)
        for folder in ['src', 'sub', 'logs', 'inference/config']:
            (root / folder).mkdir(parents=True, exist_ok=True)
        for source in (REPOSITORY / 'code/src').glob('*.py'):
            shutil.copy2(source, root / 'src' / source.name)
        shutil.copy2(REPOSITORY / 'config/coordinate_fusion.json', control / 'config/coordinate_fusion.json')
        shutil.copy2(root / 'data/tests1v2/tests/sample_submission_player_v2.csv', root / 'sub/base.csv')
        write_json(identity, fingerprint)
    if sha(control / 'config/coordinate_fusion.json') != sha(REPOSITORY / 'config/coordinate_fusion.json'):
        raise RuntimeError('Workspace inference configuration changed')
    # Ensure resumed source files are the exact recorded source snapshot.
    for source in (REPOSITORY / 'code/src').glob('*.py'):
        if sha(root / 'src' / source.name) != sha(source):
            raise RuntimeError('Workspace inference source changed: ' + source.name)
    environment = {'GO_ROOT': str(root), 'PYTHONHASHSEED': '0', 'CUDA_VISIBLE_DEVICES': args.gpu,
                   'PYTHONPATH': str(root / 'src') + os.pathsep + str(root / 'katago/kgsrc/python'),
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    for number, spec in enumerate(plan):
        process = None
        receipt = control / (spec['name'] + '.receipt.json')
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['status'] == 'complete':
                if any(not Path(p).is_file() or sha(Path(p)) != h for p, h in previous['output_hashes'].items()):
                    raise RuntimeError('Recorded output changed: ' + spec['name'])
                print('VERIFIED_COMPLETED_STAGE', spec['name'], flush=True)
                continue
            if previous['status'] == 'failed':
                raise RuntimeError('Recorded failed stage; inspect its log before retrying: ' + spec['name'])
            if not stage_active(receipt):
                raise RuntimeError('Interrupted stage has uncertain outputs; inspect before retrying: ' + spec['name'])
            print('MONITORING_LIVE_STAGE', spec['name'], previous['pid'], flush=True)
        else:
            spec.update(receipt=str(receipt), log=str(root / 'logs' / (spec['name'] + '.log')),
                        cwd=str(root), environment=environment)
            spec_path = control / (spec['name'] + '.spec.json')
            write_json(spec_path, spec)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--stage-worker', str(spec_path)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            print('STARTED', number + 1, '/', len(plan), spec['name'], 'worker', process.pid, flush=True)
        while True:
            if receipt.exists():
                status = json.loads(receipt.read_text())
                if status['status'] == 'complete':
                    break
                if status['status'] == 'failed':
                    raise RuntimeError('Inference stage failed; inspect ' + str(root / 'logs' / (spec['name'] + '.log')))
                if not stage_active(receipt):
                    raise RuntimeError('Stage stopped without a final receipt: ' + spec['name'])
            elif process is not None and process.poll() is not None and not stage_active(receipt):
                raise RuntimeError('Stage worker exited before writing its receipt: ' + spec['name'])
            time.sleep(5)
    verification = json.loads((output.parent / 'prediction_verification.json').read_text())
    expected = json.loads((REPOSITORY / 'config/reproduction_contract.json').read_text())['accepted_csv_sha256']
    if verification.get('output') != str(output) or not output.is_file() or sha(output) != expected:
        raise RuntimeError('Output file/path/checksum does not match the accepted prediction')
    if not verification['exact_match']:
        raise RuntimeError('Output was generated but the accepted prediction checksum did not match')
    print('EXACT_ACCEPTED_CSV_REPRODUCED', output, verification['sha256'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('An inference controller or stage is already running; inspect its receipt before resuming.')
    except (RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
