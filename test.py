import numpy as np
from ftz import ftz, _ftz_numpy


def test_norm_float():
    number = (np.finfo(np.float32).tiny*2)

    # pick some sizes that are both powers of two and not powers of two
    # offset in both directions
    for size in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 67]:
        for offset in [0, 1]:
            norm = np.array(np.random.randn(size), dtype=np.float32)[offset:]
            result = norm.copy()

            ftz(norm)
            np.testing.assert_array_equal(norm, result)


def test_denorm_float():
    number = (np.finfo(np.float32).tiny / 2)

    # pick some sizes that are both powers of two and not powers of two
    # offset in both directions
    for size in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 67]:
        for offset in [0, 1]:
            denorm = number * np.ones(size, dtype=np.float32)[offset:]
            denorm[0] = 12345.0 # make the first entry not a denorm
            ftz(denorm)

            np.testing.assert_array_equal(denorm[1:], np.zeros_like(denorm)[1:])
            np.testing.assert_array_equal(denorm[0], np.array([12345], dtype=np.float32))


def test_norm_double():
    number = (np.finfo(np.float64).tiny*2)

    # pick some sizes that are both powers of two and not powers of two
    # offset in both directions
    for size in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 67]:
        for offset in [0, 1]:
            norm = np.random.randn(size)[offset:]
            result = norm.copy()

            ftz(norm)
            np.testing.assert_array_equal(norm, result)


def test_denorm_double():
    number = (np.finfo(np.float64).tiny / 2)

    # pick some sizes that are both powers of two and not powers of two
    # offset in both directions
    for size in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 67]:
        for offset in [0, 1]:
            denorm = number * np.ones(size, dtype=np.float32)[offset:]
            denorm[0] = 12345.0 # make the first entry not a denorm

            ftz(denorm)

            np.testing.assert_array_equal(denorm[1:], np.zeros_like(denorm)[1:])
            np.testing.assert_array_equal(denorm[0], np.array([12345], dtype=np.float32))


def test_2d_double():
    a = np.finfo(np.float64).tiny *  np.abs(np.random.randn(20, 20))

    aa = np.copy(a)
    ftz(aa)

    bb = np.copy(a)
    _ftz_numpy(bb)

    np.testing.assert_array_equal(aa, bb)


def test_2d_single():
    a = np.finfo(np.float32).tiny *  np.abs(np.random.randn(20, 20))
    a = np.asarray(a, dtype=np.float32)

    aa = np.copy(a)
    ftz(aa)

    bb = np.copy(a)
    _ftz_numpy(bb)

    np.testing.assert_array_equal(aa, bb)


def bench(prec='single'):
    import time
    r = np.random.rand(10000000)

    if prec == 'single':
        print 'single precision'
        a = np.finfo(np.float32).tiny * r
        a = np.asarray(a, dtype=np.float32)
    elif prec == 'double':
        print 'double precision'
        a = np.finfo(np.float64).tiny = r
    else:
        raise NotImplementedError()

    funcs = [_ftz_numpy, ftz]
    n_trials = 10

    times = np.random.randn(n_trials, len(funcs))

    for i in range(n_trials):
        for j in np.random.permutation(len(funcs)):
            func = funcs[j]
            b = a.copy()

            start = time.time()
            func(b)
            end = time.time()

            times[i, j] = end-start

    print [f.__name__ for f in funcs]
    print np.median(times, axis=0)

if __name__ == '__main__':
    bench('single')
    bench('double')
