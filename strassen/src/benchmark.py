"""Reproducible, append-only CA2 experiment driver."""
from common import load_pair, sha, write_json, np
import argparse
import ctypes
import datetime
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
from strassen import multiply
from validation import verify


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
    if processes:
        with mp.Pool(processes=processes) as pool:
            c = multiply(a, b, cutoff, pool)
    else:
        c = multiply(a, b, cutoff)
    return c, time.perf_counter() - tick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON object of benchmark argument defaults')
    parser.add_argument('--manifest', default='strassen/inputs.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cutoff', type=int, default=32)
    parser.add_argument('--processes', type=int, nargs='+', default=list(range(min(6, usable_cpus()) + 1)))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-size', type=int, default=2048)
    parser.add_argument('--groups', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--generic', action='store_true')
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        settings = json.loads(Path(preliminary.config).read_text(encoding='utf-8'))
        permitted = {'manifest', 'cutoff', 'processes', 'repeats', 'max_size', 'groups', 'generic'}
        if set(settings) - permitted:
            parser.error('unsupported configuration fields: ' + str(set(settings) - permitted))
        parser.set_defaults(**settings)
    args = parser.parse_args()
    if args.cutoff < 1 or args.repeats < 1 or any(p < 0 or p > min(6, usable_cpus()) for p in args.processes):
        parser.error('invalid cutoff/repeats/process count')
    if len(set(args.processes)) != len(args.processes) or len(set(args.groups)) != len(args.groups):
        parser.error('processes and groups must not contain duplicates')
    if sorted(args.processes) != args.processes or sorted(args.groups) != args.groups:
        parser.error('use ascending process and group order')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('use a fresh output directory; existing experiments are immutable')
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = sorted([e for e in manifest['entries'] if e['group'] in args.groups and e['size'] <= args.max_size],
                     key=lambda e: (e['group'], e['pair']))
    if not entries:
        parser.error('no matrix pairs selected')
    if not args.generic:
        expected = [(g, i, n) for g in sorted(set(args.groups))
                    for i, n in enumerate((16, 32, 64, 128, 256, 512, 1024, 2048)) if n <= args.max_size]
        if [(e['group'], e['pair'], e['size']) for e in entries] != expected:
            parser.error('CA2 matrix layout mismatch')
    write_json(output / 'environment.json', environment())
    write_json(output / 'configuration.json', dict(**vars(args), manifest_sha256=sha(manifest_path),
               timing='perf_counter; multiply and fresh Pool startup/exit included per pair; IO/conversion/validation excluded',
               parallelism='7 top-level tasks, chunksize=1; workers recurse sequentially',
               validation='8 Freivalds vectors mod 2147483647, 12 exact entries, full SHA256 comparison'))
    write_json(output / 'input_manifest.json', manifest)
    baseline = {}
    with open(output / 'measurements.jsonl', 'a', encoding='utf-8', buffering=1) as log:
        for repeat in range(args.repeats):
            for p in args.processes:
                previous_group = None
                cumulative = 0.0
                for entry in entries:
                    a, b = load_pair(manifest_path.parent, entry, ca2=not args.generic)
                    gc.collect()
                    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    write_json(output / 'status.json', dict(state='computing', repeat=repeat,
                               processes=p, group=entry['group'], size=entry['size'], started_utc=started))
                    c, elapsed = timed_multiply(a, b, args.cutoff, p)
                    if entry['group'] != previous_group:
                        cumulative = 0.0
                        previous_group = entry['group']
                    cumulative += elapsed
                    seed = 50000 + repeat * 10000 + p * 1000 + entry['group'] * 100 + entry['pair']
                    validation_start = time.perf_counter()
                    output_hash = verify(a, b, c, seed, ca2=not args.generic)
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
    write_json(output / 'complete.json', dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                            measurements=len(entries) * len(args.processes) * args.repeats))
    write_json(output / 'status.json', dict(state='complete'))


if __name__ == '__main__':
    mp.freeze_support()
    main()
