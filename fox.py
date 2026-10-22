from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from typing import Sequence

# Set before importing NumPy so each multiprocessing worker stays single-threaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

Matrix = list[list[float]]
Block = tuple[tuple[float, ...], ...]


# ============================================================
# A mutable list representation is used while accumulating a
# block product. Blocks returned from worker processes are made
# immutable tuples so they can be safely serialized by Pool.
# ============================================================

def _zero_block(block_size: int) -> list[list[float]]:
    return [[0.0 for _ in range(block_size)] for _ in range(block_size)]


# ============================================================
# The i-k-j loop order keeps one value from the left block
# active while updating a complete row segment of the result.
# This is the core arithmetic used by both execution modes.
# ============================================================

def _block_product(a: Block, b: Block) -> Block:
    block_size = len(a)
    result = _zero_block(block_size)
    for row in range(block_size):
        for shared in range(block_size):
            value = a[row][shared]
            for column in range(block_size):
                result[row][column] += value * b[shared][column]
    return tuple(tuple(row) for row in result)


# ============================================================
# The result block is updated in-place because each Fox stage
# contributes one partial product to every output block.
# ============================================================

def _add_block(left: list[list[float]], right: Block) -> None:
    for row in range(len(left)):
        for column in range(len(left)):
            left[row][column] += right[row][column]


# ============================================================
# This function is defined at module scope so it can be
# serialized by multiprocessing on both fork and spawn systems.
# ============================================================

def _fox_task(task: tuple[Block, Block]) -> Block:
    return _block_product(task[0], task[1])


# ============================================================
# Internal helper: validate a square matrix
#
# Fox multiplication accepts square matrices of the same size.
# Invalid shapes are reported before any process pool is created.
# ============================================================

def _validate_matrix(matrix: Sequence[Sequence[float]], size: int) -> None:
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("matrix must have exactly size rows and columns")


# ============================================================
# A block size of 64 is used for the benchmark implementation.
# If the matrix dimension is not an exact multiple of that size,
# the final row and column blocks are filled with zeroes.
#
# Return layout:
#   blocks[block_row][block_column][row][column]
# ============================================================

def _prepare_blocks(matrix: Sequence[Sequence[float]], size: int, block_size: int) -> list[list[Block]]:
    padded_size = block_size * ((size + block_size - 1) // block_size)
    grid_size = padded_size // block_size
    blocks: list[list[Block]] = []
    for block_row in range(grid_size):
        row_blocks: list[Block] = []
        for block_column in range(grid_size):
            block = []
            for row in range(block_size):
                source_row = block_row * block_size + row
                block.append(tuple(
                    matrix[source_row][block_column * block_size + column]
                    if source_row < size and block_column * block_size + column < size
                    else 0.0
                    for column in range(block_size)
                ))
            row_blocks.append(tuple(block))
        blocks.append(row_blocks)
    return blocks


# ============================================================
# Only coordinates inside the requested matrix dimension are
# copied. Values belonging to the padded region are discarded.
# ============================================================

def _assemble(blocks: list[list[list[list[float]]]], size: int, block_size: int) -> Matrix:
    result = [[0.0 for _ in range(size)] for _ in range(size)]
    for block_row, row_blocks in enumerate(blocks):
        for block_column, block in enumerate(row_blocks):
            for row in range(block_size):
                for column in range(block_size):
                    global_row = block_row * block_size + row
                    global_column = block_column * block_size + column
                    if global_row < size and global_column < size:
                        result[global_row][global_column] = block[row][column]
    return result


# ============================================================
# For output block C[row][column], stage s selects:
#   A[row][(row + s) mod q]
#   B[(row + s) mod q][column]
#
# Here q is the number of blocks along one matrix dimension.
# ============================================================

def _fox_tasks(a_blocks: list[list[Block]], b_blocks: list[list[Block]], stage: int) -> list[tuple[Block, Block]]:
    grid_size = len(a_blocks)
    return [
        (a_blocks[row][(row + stage) % grid_size],
         b_blocks[(row + stage) % grid_size][column])
        for row in range(grid_size)
        for column in range(grid_size)
    ]


# ============================================================
# The sequential path uses exactly the same block schedule as
# the Pool path, but executes every block product in the parent
# process. This provides a fair correctness and timing baseline.
# ============================================================

def fox_sequential(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    size = len(a)
    _validate_matrix(a, size)
    _validate_matrix(b, size)
    if size == 0:
        return []

    block_size = max(1, min(64, size))
    a_blocks = _prepare_blocks(a, size, block_size)
    b_blocks = _prepare_blocks(b, size, block_size)
    grid_size = len(a_blocks)
    result_blocks = [[_zero_block(block_size) for _ in range(grid_size)]
                     for _ in range(grid_size)]

    for stage in range(grid_size):
        for task_index, task in enumerate(_fox_tasks(a_blocks, b_blocks, stage)):
            row = task_index // grid_size
            column = task_index % grid_size
            _add_block(result_blocks[row][column], _block_product(*task))
    return _assemble(result_blocks, size, block_size)


# ============================================================
# Each Fox stage submits one task for every output block to a
# standard-library multiprocessing.Pool. The parent process
# accumulates the returned partial products before starting the
# next stage, preserving Fox's algorithmic schedule.
#
# Parameters:
#   a, b       - square input matrices
#   processes  - number of Pool worker processes
#
# Post-condition:
#   returns the matrix product A * B
# ============================================================

def fox_parallel(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    processes: int | None = None,
) -> Matrix:
    size = len(a)
    _validate_matrix(a, size)
    _validate_matrix(b, size)
    if size == 0:
        return []

    worker_count = processes or mp.cpu_count()
    if worker_count < 1:
        raise ValueError("processes must be positive")
    block_size = max(1, min(64, size))
    a_blocks = _prepare_blocks(a, size, block_size)
    b_blocks = _prepare_blocks(b, size, block_size)
    grid_size = len(a_blocks)
    result_blocks = [[_zero_block(block_size) for _ in range(grid_size)]
                     for _ in range(grid_size)]

    with mp.Pool(processes=worker_count) as pool:
        for stage in range(grid_size):
            products = pool.map(_fox_task, _fox_tasks(a_blocks, b_blocks, stage))
            product_index = 0
            for row in range(grid_size):
                for column in range(grid_size):
                    _add_block(result_blocks[row][column], products[product_index])
                    product_index += 1
    return _assemble(result_blocks, size, block_size)


# ============================================================
# This independent i-k-j implementation is used only to verify
# the Fox result. It is not used by either benchmarked kernel.
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark Fox matrix multiplication")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--multi",
        action="store_true",
        help="run only the multiprocessing benchmark",
    )
    mode.add_argument(
        "--seq",
        action="store_true",
        help="run only the sequential benchmark",
    )
    args = parser.parse_args()

    sizes = [16, 32, 64, 128, 256, 512]
    core = [1, 2, 3, 4]

    if args.seq or not args.multi:
        with open("result/fox_seq.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Group", "Size", "Time", "Cumulative"])

            for g in range(3):
                total = 0

                for size in sizes:
                    A = np.load(f"data/G{g}/A_{size}.npy")
                    B = np.load(f"data/G{g}/B_{size}.npy")

                    start = time.perf_counter()
                    result = fox_sequential(A, B)
                    elapsed = time.perf_counter() - start

                    total += elapsed

                    writer.writerow([
                        f"G{g}", size, elapsed, total
                    ])

                    file.flush()

    if args.multi or not args.seq:
        with open("result/fox_multi.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Core", "Group", "Size", "Time", "Cumulative"])

            for p in core:
                for g in range(3):
                    total = 0

                    for size in sizes:
                        A = np.load(f"data/G{g}/A_{size}.npy")
                        B = np.load(f"data/G{g}/B_{size}.npy")

                        start = time.perf_counter()
                        result = fox_parallel(A, B, p)
                        elapsed = time.perf_counter() - start

                        total += elapsed

                        writer.writerow([
                            p, f"G{g}", size, elapsed, total
                        ])

                        file.flush()
