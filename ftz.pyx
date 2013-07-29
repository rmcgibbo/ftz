import numpy as np
cimport numpy as np
from libc.stdint cimport intptr_t

cdef extern from "ftz.h":
    int fftz(float* x, size_t length)
    int dftz(double* x, size_t length)

def ftz(np.ndarray data):
    if not (data.flags['C_CONTIGUOUS'] or data.flags['F_CONTIGUOUS']):
        data = np.copy(data)
    if len(data) == 0:
        raise ValueError('data must have length > 0')

    cdef np.ndarray[dtype=np.float32_t] fdata
    cdef np.ndarray[dtype=np.float64_t] ddata

    if data.dtype == np.float32:
        fdata = data
        print len(fdata)
        fftz(&fdata[0], len(fdata))
        return fdata

    elif data.dtype == np.float64:
        ddata = data
        dftz(&ddata[0], len(ddata))
        return ddata

    else:
        raise TypeError('data must be of type float32 or float64.')
