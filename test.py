import numpy as np
from ftz import ftz


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
