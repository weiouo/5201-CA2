"""Independent checks outside benchmark timing."""
import hashlib
import random


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
