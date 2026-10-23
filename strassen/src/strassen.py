"""Pure Python Strassen; no array arithmetic or hidden numerical kernels."""


def classic(a, b):
    """Required i-j-k base case, with Python integer accumulation."""
    n = len(a)
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total
    return c


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def split(a):
    h = len(a) // 2
    return ([r[:h] for r in a[:h]], [r[h:] for r in a[:h]],
            [r[:h] for r in a[h:]], [r[h:] for r in a[h:]])


def operands(a, b):
    """Seven products numbered exactly as CA2 Appendix A."""
    a11, a12, a21, a22 = split(a)
    b11, b12, b21, b22 = split(b)
    return [(add(a11, a21), add(b11, b12)),
            (add(a12, a22), add(b21, b22)),
            (add(a11, a22, -1), add(b11, b22)),
            (a11, add(b12, b22, -1)),
            (add(a21, a22), b11),
            (add(a11, a12), b22),
            (a22, add(b21, b11, -1))]


def combine(m):
    m1, m2, m3, m4, m5, m6, m7 = m
    c11 = add(add(add(m2, m3), m6, -1), m7, -1)
    c12 = add(m4, m6)
    c21 = add(m5, m7)
    c22 = add(add(add(m1, m3, -1), m4, -1), m5, -1)
    return [x + y for x, y in zip(c11, c12)] + [x + y for x, y in zip(c21, c22)]


def recursive(a, b, cutoff):
    if len(a) <= cutoff:
        return classic(a, b)
    return combine([recursive(x, y, cutoff) for x, y in operands(a, b)])


def worker(task):
    return recursive(*task)


def multiply(a, b, cutoff=32, pool=None):
    """Square Python-int matrices; pad arbitrary positive sizes inside timing.

    Input conversion/validation belongs to the loader. Pool workers recurse
    sequentially. Even a base-case MP call is dispatched to a real worker.
    """
    n = len(a)
    if cutoff < 1 or n < 1 or len(b) != n:
        raise ValueError('positive size/cutoff and equal square matrices required')
    if any(len(r) != n for r in a) or any(len(r) != n for r in b):
        raise ValueError('square matrices required')
    size = 1 << (n - 1).bit_length()
    if size != n:
        a = [r + [0] * (size - n) for r in a] + [[0] * size for _ in range(size - n)]
        b = [r + [0] * (size - n) for r in b] + [[0] * size for _ in range(size - n)]
    if pool is None:
        c = recursive(a, b, cutoff)
    elif size <= cutoff:
        c = pool.apply(worker, ((a, b, cutoff),))
    else:
        c = combine(pool.map(worker, [(x, y, cutoff) for x, y in operands(a, b)], chunksize=1))
    return [r[:n] for r in c[:n]] if size != n else c
