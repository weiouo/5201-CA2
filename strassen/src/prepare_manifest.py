"""Index existing shared CA2 matrices without modifying or regenerating them."""
from common import np, sha, write_json
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='strassen/inputs.json')
    args = parser.parse_args()
    root = Path(args.data).resolve()
    output = Path(args.output).resolve()
    entries = []
    for group in range(3):
        for pair, size in enumerate((16, 32, 64, 128, 256, 512, 1024, 2048)):
            entry = dict(group=group, pair=pair, size=size)
            for key in ('a', 'b'):
                path = root / f'G{group}' / f'{key.upper()}_{size}.npy'
                array = np.load(path, allow_pickle=False, mmap_mode='r')
                if (array.shape != (size, size)
                        or not np.issubdtype(array.dtype, np.integer)
                        or array.min() < 0 or array.max() > 255):
                    raise ValueError(f'Invalid CA2 matrix: {path}')
                del array
                entry[key] = Path(os.path.relpath(path, output.parent)).as_posix()
                entry[key + '_sha256'] = sha(path)
            entries.append(entry)
    manifest = dict(entries=entries)
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing manifest differs; use a new --output path.')
        print(f'Unchanged: {output} (24 pairs, 48 files)')
    else:
        write_json(output, manifest)
        print(f'Created: {output} (24 pairs, 48 files)')


if __name__ == '__main__':
    main()
