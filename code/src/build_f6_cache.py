"""Build the binary F6 token CSR cache used by svd_only.py."""
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from id2 import mat

root = Path(os.environ['GO_ROOT'])
if os.environ.get('PYTHONHASHSEED') != '0':
    raise RuntimeError('Run with PYTHONHASHSEED=0 to match the recorded token vocabulary.')
archive = np.load(root / 'data/train_arr.npz')
parts = []
for start in range(0, len(archive['M']), 50000):
    stop = min(start + 50000, len(archive['M']))
    parts.append(mat(archive['M'], archive['C'], np.arange(start, stop),
                     archive['col'][start:stop], (1, 1, 1, 1, 1, 1)))
    print('Tokenized games', stop, flush=True)
cache = sp.vstack(parts).tocsr()
np.savez(root / 'results/F6_csr.npz', indptr=cache.indptr.astype(np.int64),
         indices=cache.indices.astype(np.int32))
