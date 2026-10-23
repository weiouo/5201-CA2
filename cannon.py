"""CEG5201 CA2 -- Cannon 4x4 block matrix multiplication.

Core arithmetic is written with explicit Python loops. NumPy is used for
matrix storage, slicing, checking equality, and assembling the output ONLY.
The Pool-based timing can include or exclude worker-pool creation
(default: included, matching the current cw.py). It always includes task
preparation, task serialization/IPC, computing, and result assembly.

Usage, from the 5201-CA2 repository root:
    python cannon.py --mode quick
    python cannon.py --mode verify
    python cannon.py --mode benchmark --max-size 256 --groups 0 --max-procs 4
    python cannon.py --mode benchmark --max-size 2048 --groups 0,1,2 --max-procs 6

Full pure-Python benchmarks may take many hours or longer.
"""

import os

# Apply BEFORE importing NumPy (also inherited by spawned workers on Windows).
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
ALL_SIZES = (16, 32, 64, 128, 256, 512, 1024, 2048)
GRID = 4  # Required by the CA2 assignment (4 x 4 = 16 blocks).


def make_block_tasks(A, B):
    """Create 16 independent output-block tasks in Cannon's cyclic order.

    For output block (i,j), Cannon's initial skew pairs A[i,(i+j)%4]
    with B[(i+j)%4,j]. Each following stage advances this index by 1.
    Each pair is a BLOCK MATRIX MULTIPLICATION, not elementwise product.
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape")
    n = A.shape[0]
    if n % GRID:
        raise ValueError("Matrix size must be divisible by 4")
    b = n // GRID
    a_blocks = [[A[i*b:(i+1)*b, j*b:(j+1)*b]
                 for j in range(GRID)] for i in range(GRID)]
    b_blocks = [[B[i*b:(i+1)*b, j*b:(j+1)*b]
                 for j in range(GRID)] for i in range(GRID)]
    tasks = []
    for i in range(GRID):
        for j in range(GRID):
            indices = [(i + j + stage) % GRID for stage in range(GRID)]
            a_inputs = [a_blocks[i][k] for k in indices]
            b_inputs = [b_blocks[k][j] for k in indices]
            tasks.append((i, j, a_inputs, b_inputs))
    return tasks


def block_worker(task):
    """Calculate ONE C[i,j] output block, accumulating all four stages.

    Inner block multiplications use explicit i-k-j Python loops.
    No np.matmul(), np.dot(), @, vectorized + or * are used.
    """
    block_i, block_j, a_inputs, b_inputs = task
    b = a_inputs[0].shape[0]
    out = [[0] * b for _ in range(b)]

    for stage in range(GRID):
        # This conversion makes scalar arithmetic use Python integers rather
        # than repeated NumPy scalar indexing. It is INSIDE the timed work.
        a = a_inputs[stage].tolist()
        bb = b_inputs[stage].tolist()
        for i in range(b):
            out_row = out[i]
            a_row = a[i]
            for k in range(b):
                x = a_row[k]
                b_row = bb[k]
                for j in range(b):
                    out_row[j] += x * b_row[j]
    return block_i, block_j, out


def assemble_output(n, pieces):
    b = n // GRID
    result = np.empty((n, n), dtype=np.int64)
    for i, j, block in pieces:
        result[i*b:(i+1)*b, j*b:(j+1)*b] = block
    return result


def cannon_sequential(A, B):
    tasks = make_block_tasks(A, B)
    pieces = [block_worker(task) for task in tasks]
    return assemble_output(A.shape[0], pieces)


def cannon_multiprocessing(A, B, pool):
    """Pool has already been created; its startup time is not measured."""
    tasks = make_block_tasks(A, B)
    # A single worker can compute multiple blocks. Pool schedules 16 tasks.
    pieces = pool.map(block_worker, tasks, chunksize=1)
    return assemble_output(A.shape[0], pieces)


def naive_reference(A, B):
    """Independent triple-loop result for correctness tests (small matrices)."""
    n = A.shape[0]
    a, b = A.tolist(), B.tolist()
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            x = a[i][k]
            for j in range(n):
                c[i][j] += x * b[k][j]
    return np.asarray(c, dtype=np.int64)


def verify():
    print("Checking Cannon against independent, explicit-loop multiplication...")
    rng = np.random.default_rng(123)
    for n in (4, 8, 16, 32):
        A = rng.integers(0, 256, (n, n), dtype=np.int64)
        B = rng.integers(0, 256, (n, n), dtype=np.int64)
        expected = naive_reference(A, B)
        seq = cannon_sequential(A, B)
        assert np.array_equal(seq, expected), f"Incorrect serial result at {n}x{n}"
        with Pool(processes=2) as pool:
            mp = cannon_multiprocessing(A, B, pool)
        assert np.array_equal(mp, expected), f"Incorrect multiprocessing result at {n}x{n}"
        print(f"  PASS: {n}x{n}, sequential and 2-process version")


def read_data(group, n):
    folder = ROOT / "data" / f"G{group}"
    A = np.load(folder / f"A_{n}.npy", allow_pickle=False)
    B = np.load(folder / f"B_{n}.npy", allow_pickle=False)
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError(f"Unexpected input shape for G{group}, size {n}")
    return A, B


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved: {path}")


def benchmark(groups, sizes, max_procs, pool_timing="included"):
    """Match teammate CSV schemas; keep raw cumulative times per group."""
    print(f"Timing rule: Pool creation {pool_timing.upper()} for each measured pair;")
    print("             task preparation, computation, IPC, assembly INCLUDED.")
    seq_rows = []
    for group in groups:
        cumulative = 0.0
        for n in sizes:
            A, B = read_data(group, n)
            start = perf_counter()
            _ = cannon_sequential(A, B)
            seconds = perf_counter() - start
            cumulative += seconds
            seq_rows.append((f"G{group}", n, seconds, cumulative))
            print(f"[SEQ] G{group}, n={n:4d}: {seconds:.3f}s, cumulative={cumulative:.3f}s", flush=True)
    write_csv(ROOT / "result" / "cannon_seq.csv",
              ("Group", "Size", "Time", "Cumulative"), seq_rows)

    mp_rows = []
    for p in range(1, max_procs + 1):
        if pool_timing == "included":
            # Match the group member's current cw.py methodology: create
            # and close a fresh Pool inside each pair's timing interval.
            for group in groups:
                cumulative = 0.0
                for n in sizes:
                    A, B = read_data(group, n)
                    start = perf_counter()
                    with Pool(processes=p) as pool:
                        _ = cannon_multiprocessing(A, B, pool)
                    seconds = perf_counter() - start
                    cumulative += seconds
                    mp_rows.append((p, f"G{group}", n, seconds, cumulative))
                    print(f"[MP-{p}] G{group}, n={n:4d}: {seconds:.3f}s, cumulative={cumulative:.3f}s", flush=True)
        else:
            # Alternative methodology: create one Pool per process count,
            # and reuse it across all groups/sizes outside the timed region.
            with Pool(processes=p) as pool:
                for group in groups:
                    cumulative = 0.0
                    for n in sizes:
                        A, B = read_data(group, n)
                        start = perf_counter()
                        _ = cannon_multiprocessing(A, B, pool)
                        seconds = perf_counter() - start
                        cumulative += seconds
                        mp_rows.append((p, f"G{group}", n, seconds, cumulative))
                        print(f"[MP-{p}] G{group}, n={n:4d}: {seconds:.3f}s, cumulative={cumulative:.3f}s", flush=True)
        # Save after each process count: keep intermediate results if stopped.
        write_csv(ROOT / "result" / "cannon_multi.csv",
                  ("Core", "Group", "Size", "Time", "Cumulative"), mp_rows)
    print("Benchmark finished. Compare using seq cumulative / MP cumulative.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cannon matrix multiplication CA2")
    parser.add_argument("--mode", choices=("verify", "quick", "benchmark"),
                        default="quick", help="quick is safe for your first run")
    parser.add_argument("--max-size", type=int, default=2048,
                        help="largest size to benchmark (16,32,...,2048)")
    parser.add_argument("--groups", default="0,1,2", help="comma-separated group IDs")
    parser.add_argument("--max-procs", type=int, default=6,
                        help="maximum Pool processes, capped to 6")
    parser.add_argument("--pool-timing", choices=("included", "excluded"),
                        default="included",
                        help="whether timing includes creating/closing a Pool per pair; default matches cw.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("verify", "quick"):
        verify()
    if args.mode == "quick":
        print("Quick test: only G0; matrix sizes 16, 32, 64; p=1,2.")
        benchmark([0], [16, 32, 64], 2, args.pool_timing)
    elif args.mode == "benchmark":
        groups = [int(x.strip()) for x in args.groups.split(",")]
        if not groups or any(g not in (0, 1, 2) for g in groups):
            raise ValueError("Available groups: 0,1,2")
        sizes = [n for n in ALL_SIZES if n <= args.max_size]
        if not sizes:
            raise ValueError("max-size must be >= 16")
        max_procs = min(args.max_procs, 6, os.cpu_count() or 1)
        if max_procs < 1:
            raise ValueError("max-procs must be >= 1")
        print("WARNING: Pure-Python calculations for large matrices can be VERY slow.")
        benchmark(groups, sizes, max_procs, args.pool_timing)
