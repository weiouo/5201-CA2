import os

# Set before importing NumPy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
from multiprocessing import Pool
import csv
from pathlib import Path


def add(A, B):
    a = A.tolist()
    b = B.tolist()
    c = [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
    return np.asarray(c, dtype=np.int64)


def sub(A, B):
    a = A.tolist()
    b = B.tolist()
    c = [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
    return np.asarray(c, dtype=np.int64)


def mul(A, B):
    n = A.shape[0]
    a = A.tolist()
    b = B.tolist()
    c = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            total = 0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total

    return np.asarray(c, dtype=np.int64)

def calculate_z(name, m1, m2, m3, m4, m5, m6, m7):

    if name == "z11":
        return add(m1, m2)

    elif name == "z12":
        return add(add(add(m1, m3), m5), m6)

    elif name == "z21":
        return sub(add(add(m1, m6), m7), m4)

    elif name == "z22":
        return add(add(add(m1, m5), m6), m7)

def cw_seq(A, B, size):
    if size <= 16 :
        return mul(A,B)
    
    mid = int(size/2)

    a11, a12 = A[:mid, :mid], A[:mid, mid:]
    a21, a22 = A[mid:, :mid], A[mid:, mid:]

    b11, b12 = B[:mid, :mid], B[:mid, mid:]
    b21, b22 = B[mid:, :mid], B[mid:, mid:]

    s1 = add(a21, a22)
    s2 = sub(s1, a11)
    s3 = sub(a11, a21)
    s4 = sub(a12, s2)

    t1 = sub(b12, b11)
    t2 = sub(b22, t1)
    t3 = sub(b22, b12)
    t4 = sub(t2, b21)

    m1 = cw_seq(a11,b11,mid)
    m2 = cw_seq(a12,b21,mid)
    m3 = cw_seq(s4,b22,mid)
    m4 = cw_seq(a22,t4,mid)
    m5 = cw_seq(s1,t1,mid)
    m6 = cw_seq(s2,t2,mid)
    m7 = cw_seq(s3,t3,mid)

    z11 = calculate_z("z11",m1,m2,m3,m4,m5,m6,m7)
    z12 = calculate_z("z12",m1,m2,m3,m4,m5,m6,m7)
    z21 = calculate_z("z21",m1,m2,m3,m4,m5,m6,m7)
    z22 = calculate_z("z22",m1,m2,m3,m4,m5,m6,m7)
    
    C = np.empty((size, size), dtype=np.int64)

    C[:mid, :mid] = z11
    C[:mid, mid:] = z12
    C[mid:, :mid] = z21
    C[mid:, mid:] = z22

    return C

def cw_multi(A, B, size, processes):
    if size <= 16:
        with Pool(processes=processes) as pool:
            result = pool.starmap(mul, [(A, B)])
        return result[0]

    mid = int(size/2)

    a11, a12 = A[:mid, :mid], A[:mid, mid:]
    a21, a22 = A[mid:, :mid], A[mid:, mid:]

    b11, b12 = B[:mid, :mid], B[:mid, mid:]
    b21, b22 = B[mid:, :mid], B[mid:, mid:]

    s1 = add(a21, a22)
    s2 = sub(s1, a11)
    s3 = sub(a11, a21)
    s4 = sub(a12, s2)

    t1 = sub(b12, b11)
    t2 = sub(b22, t1)
    t3 = sub(b22, b12)
    t4 = sub(t2, b21)

    tasks = [
        (a11, b11, mid),
        (a12, b21, mid),
        (s4, b22, mid),
        (a22, t4, mid),
        (s1, t1, mid),
        (s2, t2, mid),
        (s3, t3, mid)
    ]

    with Pool(processes=processes) as pool:
        results = pool.starmap(cw_seq, tasks)

        m1, m2, m3, m4, m5, m6, m7 = results
        

    z11 = calculate_z("z11",m1,m2,m3,m4,m5,m6,m7)
    z12 = calculate_z("z12",m1,m2,m3,m4,m5,m6,m7)
    z21 = calculate_z("z21",m1,m2,m3,m4,m5,m6,m7)
    z22 = calculate_z("z22",m1,m2,m3,m4,m5,m6,m7)

    C = np.empty((size, size), dtype=np.int64)

    C[:mid, :mid] = z11
    C[:mid, mid:] = z12
    C[mid:, :mid] = z21
    C[mid:, mid:] = z22

    return C


if __name__ == "__main__":

    sizes = [16,32,64,128,256,512,1024,2048]
    core = [1,2,3,4,5,6]
    
    with open("result/cw_seq.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Group", "Size", "Time", "Cumulative"])

        for g in range(3):
            total = 0

            for size in sizes:
                A = np.load(f"data/G{g}/A_{size}.npy")
                B = np.load(f"data/G{g}/B_{size}.npy")

                start = time.perf_counter()
                result = cw_seq(A, B, size)
                elapsed = time.perf_counter() - start

                total += elapsed

                writer.writerow([
                    f"G{g}", size, elapsed, total
                ])

                file.flush()
    
    with open("result/cw_multi.csv", "w", newline="") as file:
            writer = csv.writer(file)
    
            writer.writerow(["Core", "Group", "Size", "Time", "Cumulative"])
    
            for p in core:
                for g in range(3):
                    total = 0
    
                    for size in sizes:
                        A = np.load(f"data/G{g}/A_{size}.npy")
                        B = np.load(f"data/G{g}/B_{size}.npy")
    
                        start = time.perf_counter()
                        result = cw_multi(A, B, size, p)
                        elapsed = time.perf_counter() - start
    
                        total += elapsed
    
                        writer.writerow([
                            p, f"G{g}", size, elapsed, total
                        ])

                        file.flush()