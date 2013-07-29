import sys
import numpy as np
cimport numpy as np
from libc.stdint cimport intptr_t

cdef extern from "ftz.h":
    int fftz(float* x, size_t length)
    int dftz(double* x, size_t length)

def ftz(np.ndarray data):
    """Flush denormalized numbers to zero in place.

    Denormalized numbers ("denorms") are extremely small numbers (less than
    1.2-38 for single precision or 2.2e-308 in double precision) which are
    handled poorly on most modern archicectures. Arithmetic involving denorms
    can be up to 100x slower than arithmetic on standard floating point
    numbers. Denorms can also give annoying behavior, like overflowing when
    you take their reciprocal.

    Parameters
    ----------
    data : np.ndarray, dtype = {float32 or float64}
        An array of floating point numbers.

    """
    if not (data.flags['C_CONTIGUOUS'] or data.flags['F_CONTIGUOUS']):
        return _ftz_numpy(data)

    if len(data) == 0:
        raise ValueError('data must have length > 0')

    cdef np.ndarray[dtype=np.float32_t] fdata
    cdef np.ndarray[dtype=np.float64_t] ddata

    if data.dtype == np.float32:
        fdata = data.reshape(-1)
        fftz(&fdata[0], len(fdata))

    elif data.dtype == np.float64:
        ddata = data.reshape(-1)
        dftz(&ddata[0], len(ddata))

    else:
        raise TypeError('data must be of type float32 or float64.')


def _ftz_numpy(np.ndarray data):
    """Flush denormalized numbers to zero in place using numpy.

    This code does not require contiguity or alignment, but is not as fast.

    Parameters
    ----------
    data : np.ndarray, dtype = {float32 or float64}
        An array of floating point numbers.
    """
    if data.dtype not in [np.float32, np.float64]:
        raise TypeError('data must be of type float32 or float64.')

    bound = np.finfo(data.dtype).tiny
    mask = np.logical_or(data > bound, data < -bound)
    np.multiply(data, mask, out=data)
