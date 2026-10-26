"""Run the CA2 Strassen experiment on the shared matrix files."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import ctypes
import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np
from strassen import multiply


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


def load_pair(data, entry):
    arrays = []
    for key in ('a', 'b'):
        path = data / entry[key]
        array = np.load(path, allow_pickle=False)
        n = entry['size']
        if (array.shape != (n, n) or not np.issubdtype(array.dtype, np.integer)
                or array.min() < 0 or array.max() > 255):
            raise ValueError(f'Invalid CA2 matrix: {path}')
        digest = sha(path)
        if key + '_sha256' in entry and entry[key + '_sha256'] != digest:
            raise ValueError(f'Input changed during experiment: {path}')
        entry[key + '_sha256'] = digest
        arrays.append(array)
    return arrays


def digest(c):
    h = hashlib.sha256()
    for row in c:
        # Decimal encoding also supports arbitrary-size integers.
        h.update((','.join(map(str, row)) + '\n').encode('ascii'))
    return h.hexdigest()


def verify(a, b, c, seed, trials=8, samples=12, ca2=False):
    """Freivalds over prime field; false acceptance <= (1/q)^trials.

    Bound assumes a nonzero error modulo q and independent uniform vectors.
    Exact sampled entries complement the modular probabilistic test.
    """
    q = 2147483647
    rng = random.Random(seed)
    n = len(a)
    if len(c) != n or any(len(row) != n for row in c):
        raise AssertionError('incorrect output dimensions')
    if trials < 1 or samples < 1:
        raise ValueError('positive verification counts required')
    if ca2:
        # Both the true and candidate answers lie below q for CA2 sizes;
        # a nonzero error cannot disappear merely by reducing modulo q.
        limit = n * 255 * 255
        if limit >= q:
            raise ValueError('CA2 modular-check bound exceeded')
        if any(not isinstance(x, int) or x < 0 or x > limit for row in c for x in row):
            raise AssertionError('output outside CA2 integer bounds')
    for _ in range(trials):
        r = [rng.randrange(q) for _ in range(n)]
        br = [sum(x * y for x, y in zip(row, r)) % q for row in b]
        abr = [sum(x * y for x, y in zip(row, br)) % q for row in a]
        cr = [sum(x * y for x, y in zip(row, r)) % q for row in c]
        if abr != cr:
            raise AssertionError('Freivalds check failed')
    for _ in range(samples):
        i, j = rng.randrange(n), rng.randrange(n)
        if c[i][j] != sum(a[i][k] * b[k][j] for k in range(n)):
            raise AssertionError('exact entry check failed')
    return digest(c)


def usable_cpus():
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    if os.name == 'nt':
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if kernel.GetProcessAffinityMask(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                        ctypes.byref(process), ctypes.byref(system)):
            return bin(process.value).count('1')
    return os.cpu_count() or 1


def environment():
    info = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                processor=platform.processor(), logical_cpus=os.cpu_count(),
                usable_cpus=usable_cpus(), numpy=np.__version__, start_method=mp.get_context().get_start_method(),
                thread_limits={name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
                source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            info['cpu_name'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'avail', 'page_total',
                                                       'page_avail', 'virtual_total',
                                                       'virtual_avail', 'extended')]
        mem = Memory()
        mem.length = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            info['physical_memory_bytes'] = mem.total
            info['available_memory_bytes'] = mem.avail
    return info


def timed_multiply(a, b, cutoff, processes):
    """Time a fresh Pool lifecycle per product, as in the group scripts."""
    tick = time.perf_counter()
    a, b = a.tolist(), b.tolist()
    if processes:
        with mp.Pool(processes=processes) as pool:
            c = multiply(a, b, cutoff, pool)
    else:
        c = multiply(a, b, cutoff)
    elapsed = time.perf_counter() - tick
    return a, b, c, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-size', type=int, choices=(16,32,64,128,256,512,1024,2048), default=2048)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('use a new output directory')
    entries = [dict(group=g, pair=i, size=n, a=f'G{g}/A_{n}.npy', b=f'G{g}/B_{n}.npy')
               for g in range(3)
               for i, n in enumerate((16,32,64,128,256,512,1024,2048)) if n <= args.max_size]
    manifest = dict(entries=entries)
    # Fixed CA2 settings; one complete run, sequential then all permitted Pool sizes.
    args.groups = [0, 1, 2]
    args.processes = list(range(min(6, usable_cpus()) + 1))
    args.repeats = 1
    args.cutoff = 32
    config = dict(groups=args.groups, processes=args.processes, repeats=1,
                  cutoff=32, max_size=args.max_size, data=str(args.data),
                  timing='Input conversion, multiply and fresh Pool startup/exit included; IO/input checks/output verification excluded')
    write_json(output / 'environment.json', environment())
    write_json(output / 'configuration.json', config)
    baseline = {}
    with open(output / 'measurements.jsonl', 'a', encoding='utf-8', buffering=1) as log:
        for repeat in range(args.repeats):
            for p in args.processes:
                previous_group = None
                cumulative = 0.0
                for entry in entries:
                    a, b = load_pair(args.data, entry)
                    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    write_json(output / 'status.json', dict(state='computing', repeat=repeat,
                               processes=p, group=entry['group'], size=entry['size'], started_utc=started))
                    a, b, c, elapsed = timed_multiply(a, b, args.cutoff, p)
                    if entry['group'] != previous_group:
                        cumulative = 0.0
                        previous_group = entry['group']
                    cumulative += elapsed
                    seed = 50000 + repeat * 10000 + p * 1000 + entry['group'] * 100 + entry['pair']
                    validation_start = time.perf_counter()
                    output_hash = verify(a, b, c, seed, ca2=True)
                    validation_seconds = time.perf_counter() - validation_start
                    key = (entry['group'], entry['pair'])
                    if key in baseline and baseline[key] != output_hash:
                        raise AssertionError('output differs across implementations/repetitions')
                    baseline[key] = output_hash
                    row = dict(repeat=repeat, processes=p, group=entry['group'], pair=entry['pair'],
                               size=entry['size'], seconds=elapsed, output_sha256=output_hash,
                               cumulative_group_seconds=cumulative,
                               validation_seconds=validation_seconds,
                               verified=True, verification_seed=seed, started_utc=started)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    print(f"repeat={repeat} p={p} G{entry['group']} n={entry['size']} {elapsed:.3f}s VERIFIED", flush=True)
                    del a, b, c
    write_json(output / 'input_manifest.json', manifest)
    write_json(output / 'complete.json', dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                            measurements=len(entries) * len(args.processes) * args.repeats))
    write_json(output / 'status.json', dict(state='complete'))


if __name__ == '__main__':
    mp.freeze_support()
    main()
