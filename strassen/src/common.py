"""Storage and provenance helpers; never used as matrix arithmetic kernels."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import hashlib
import json
from pathlib import Path
import numpy as np


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_pair(root, entry, ca2=True):
    out = []
    for key in ('a', 'b'):
        path = Path(root) / entry[key]
        if sha(path) != entry[key + '_sha256']:
            raise ValueError('input hash mismatch: ' + str(path))
        a = np.load(path, allow_pickle=False)
        n = entry['size']
        if a.shape != (n, n) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('square integer input required')
        if ca2 and (a.min() < 0 or a.max() > 255):
            raise ValueError('CA2 entries must be in [0,255]')
        out.append(a.tolist())
    return out
