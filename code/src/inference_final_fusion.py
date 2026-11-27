"""Final accepted fusion from scores/features generated in the current inference run.

No prediction CSV, target labels, training arrays or historical score paths are inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = 'eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26'
POOLS = [('pub', 'public', False), ('priv', 'private', False),
         ('pub2', 'public', True), ('priv2', 'private', True)]


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def z(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def softmax(x, temperature):
    x = x / temperature[:, None] if np.ndim(temperature) else x / temperature
    x = x - x.max(1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(1, keepdims=True)


def gaussian(refs, query):
    mu = refs.mean(1)
    variance = refs.var(1)
    variance = .8 * variance.mean(0, keepdims=True) + .2 * variance + 1e-5
    inverse = 1 / variance
    query_mean = query.mean(1)
    distance = ((query_mean * query_mean) @ inverse.T
                - 2 * query_mean @ (mu * inverse).T
                + (mu * mu * inverse).sum(1)[None])
    return -.5 * distance / (1 / query.shape[1] + 1 / refs.shape[1]) - .5 * np.log(variance).sum(1)[None]


def coordinate_fusion(base, member_scores, votes, candidate_lp, query_lp,
                      num_games, query_embeddings=None, updated=False):
    identity = np.mean([z(member_scores[tag]) for tag in votes], axis=0).astype(np.float32)
    score = z(base) + (3 if updated else 1) * z(identity)
    query_temperature = np.ones(len(num_games)) if updated else np.maximum(1., num_games / 5)
    candidate_temperature = 3. if updated else 30.
    gate = np.log(softmax(query_lp, query_temperature)
                  @ softmax(candidate_lp, candidate_temperature).T + 1e-3)
    if updated:
        score = score - .6 * np.sort(score, axis=0)[-3:].mean(0)[None]
        if query_embeddings is None:
            raise ValueError('Updated fusion requires freshly generated query graph embeddings')
        embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        adjacency = embeddings @ embeddings.T
        np.fill_diagonal(adjacency, -1)
        graph = np.zeros_like(adjacency)
        nearest = np.argsort(-adjacency, axis=1)[:, :3]
        for i in range(len(graph)):
            for j in nearest[i]:
                if adjacency[i, j] > .5:
                    graph[i, j] = adjacency[i, j]
                    graph[j, i] = adjacency[i, j]
        graph = graph / (graph.sum(1, keepdims=True) + 1e-9)
        score = score + .5 * (graph @ score)
    score = score + (2 if updated else 3) * z(gate)
    if not updated:
        score = score / 1.4
        score = score - score.max()
        for _ in range(30):
            score = score - np.logaddexp.reduce(score, axis=1, keepdims=True)
            score = score - np.logaddexp.reduce(score, axis=0, keepdims=True)
    if not np.isfinite(score).all():
        raise ValueError('Non-finite coordinate fusion scores')
    return score


def policy_scores(root, data, device='cpu'):
    import torch
    from torch import nn
    from torch.nn import functional as F
    checkpoint = torch.load(root / 'reproduction/policy_metric1119/policy_metric.pt',
                            map_location=device, weights_only=False)
    width = len(checkpoint['mean'])
    model = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, checkpoint['hidden']), nn.GELU(),
                          nn.Dropout(.15), nn.Linear(checkpoint['hidden'], checkpoint['dimension'])).to(device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    features = data['features']
    if features.shape != (7100, width) or not np.isfinite(features).all() or not (data['counts'] > 0).all():
        raise ValueError('Accepted configuration requires valid extracted features for all 7100 games')
    inputs = torch.tensor((features - checkpoint['mean']) / checkpoint['scale'], device=device)
    with torch.inference_mode():
        embedding = np.concatenate([F.normalize(model(inputs[i:i + 1024]), dim=-1).cpu().numpy()
                                    for i in range(0, len(inputs), 1024)])
    valid = np.isfinite(features).all(1) & (data['counts'] > 0)
    # Retain the historical sum/count order rather than replacing it with mean().
    counts = valid[:6000].reshape(600, 10).sum(1)
    prototypes = norm(embedding[:6000].reshape(600, 10, -1).sum(1) / np.maximum(counts[:, None], 1))
    queries = np.zeros((400, checkpoint['dimension']), dtype=np.float32)
    for q in range(400):
        selected = data['query_indices'] == q
        if not selected.any():
            raise ValueError('Missing query games')
        queries[q] = embedding[6000:][selected].mean(0)
    learned = z(norm(queries) @ prototypes.T)
    with np.load(root / 'reproduction/policy_expanded_lda1122/policy_lda.npz') as fitted:
        expanded = norm(((features - fitted['mean']) / fitted['scale']) @ fitted['projection'])
    refs = expanded[:6000].reshape(600, 10, -1)
    likelihood = np.concatenate([gaussian(refs, expanded[6000:][data['query_indices'] == q][None])
                                 for q in range(400)], axis=0)
    return learned, z(likelihood)


def build(root, config, output, device='cpu'):
    csv.field_size_limit(10**9)
    tests = root / 'data/tests1v2/tests'
    sample = list(csv.reader((tests / 'sample_submission_player_v2.csv').open(newline='')))
    predictions = {}
    for pool, suffix, updated in POOLS:
        infix = '2' if updated else ''
        questions = list(csv.DictReader((tests / f'player_identification_test{infix}_{suffix}.csv').open(newline='')))
        candidates = list(csv.DictReader((tests / f'player_identification_candidates{infix}_{suffix}.csv').open(newline='')))
        player_ids = np.array(sorted({r['player_id'] for r in candidates}))
        if not np.array_equal(player_ids, np.load(root / 'results' / f'S_{pool}_pids.npy')):
            raise ValueError('Candidate ordering mismatch: ' + pool)
        votes = (config['weighted_members'] + [config['new_member']] * config['new_member_weight']
                 if updated else config['original_weighted_members'])
        members = {tag: np.load(root / 'results' / f'Sidft_{tag}_{pool}.npy') for tag in set(votes)}
        graph_embeddings = (np.concatenate([np.load(root / 'results' / f'Qemb_{tag}_{pool}.npy')
                                            for tag in config['original_members']], axis=1) if updated else None)
        rank_suffix = f'codex1118_{pool}' if updated else pool
        score = coordinate_fusion(
            np.load(root / 'results' / f'S_{pool}.npy'), members, votes,
            np.load(root / 'results' / f'LPc_{rank_suffix}.npy'),
            np.load(root / 'results' / f'LPq_{rank_suffix}.npy'),
            np.array([int(r['num_games']) for r in questions], dtype=float), graph_embeddings, updated)
        np.save(root / 'results' / f'inference_coordinate_{pool}.npy', score)
        if updated:
            with np.load(root / 'reproduction/policy_test1119' / f'{pool}_features.npz') as features:
                if not np.array_equal(features['player_ids'], player_ids):
                    raise ValueError('Policy candidate alignment mismatch')
                if not np.array_equal(features['question_ids'], [r['question_id'] for r in questions]):
                    raise ValueError('Policy question alignment mismatch')
                learned, likelihood = policy_scores(root, features, device)
            score = z(score) + .25 * learned + .10 * likelihood
        np.save(root / 'results' / f'inference_final_{pool}.npy', score)
        for row, order in zip(questions, np.argsort(-score, axis=1)[:, :5]):
            qid = row['question_id']
            if qid in predictions:
                raise ValueError('Duplicate question ID')
            predictions[qid] = [qid] + [str(player_ids[i]) for i in order]
    if len(sample) != 1601 or len(predictions) != 1600 or set(predictions) != {row[0] for row in sample[1:]}:
        raise ValueError('Official sample and inferred questions do not align')
    rows = [sample[0]] + [predictions[row[0]] for row in sample[1:]]
    if any(len(row) != 6 or len(set(row[1:])) != 5 for row in rows[1:]):
        raise ValueError('Invalid top-five output')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result = {'output': str(output), 'sha256': digest, 'expected_sha256': EXPECTED_SHA256,
              'rows': 1600, 'exact_match': digest == EXPECTED_SHA256,
              'scope': 'Fusion of artifacts generated in this inference workspace; no historical CSV input'}
    (output.parent / 'prediction_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    result = build(args.root.resolve(), json.loads(args.config.read_text()), args.output.resolve(), args.device)
    print(json.dumps(result, indent=2), flush=True)
    if not result['exact_match']:
        raise SystemExit('Prediction generated but does not match the accepted CSV; inspect the saved verification record.')


if __name__ == '__main__':
    main()
