# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
# Contains functionality for responses

import numpy as np
cimport numpy as np
np.import_array()
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def outer_add(
    np.ndarray[double, ndim=1] in_a,
    unsigned long a_start,
    unsigned long n_a,
    np.ndarray[double, ndim=2] in_b,
    unsigned long b_i,
    np.ndarray[double, ndim=2] out,
    unsigned long out_start,
):
    # get the number of channels in the data
    cdef size_t nspecs = out.shape[0]
    
    j = 0
    for i in range(n_a):
        for s in range(nspecs):
            out[s, out_start + i] += in_a[a_start + i] * in_b[s, b_i]
