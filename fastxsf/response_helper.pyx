# cython: language_level=3,annotate=True,profile=True,fast_fail=True,warning_errors=True
# Contains functionality for responses

import numpy as np
cimport numpy as np
np.import_array()
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_rmf_vectorized_chunked(
    np.ndarray[float, ndim=1] matrix,
    np.ndarray[np.int64_t, ndim=1] indices_channel_lengths,
    np.ndarray[np.int64_t, ndim=1] chunks_size,
    np.ndarray[np.int64_t, ndim=1] chunks_counts_start_idx,
    np.ndarray[np.int64_t, ndim=1] chunks_resp_start_idx,
    np.ndarray[double, ndim=2] specs,
    np.ndarray[double, ndim=2] output,
):
    # get the number of channels in the data
    cdef size_t nspecs = specs.shape[0]
    cdef size_t nchannels = specs.shape[1]
    
    j = 0
    for i in range(nchannels):
        for chunkid in range(indices_channel_lengths[i]):
            chunk_size = chunks_size[j]
            counts_start_idx = chunks_counts_start_idx[j]
            resp_start_idx = chunks_resp_start_idx[j]
            for s in range(nspecs):
                for k in range(chunk_size):
                    output[s, counts_start_idx + k] = matrix[resp_start_idx + k] * specs[s, i]
            j += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_rmf_vectorized(
    np.ndarray[np.float_t, ndim=1] weights,
    np.ndarray[np.int64_t, ndim=1] in_i,
    np.ndarray[np.int64_t, ndim=1] out_i,
    np.ndarray[np.float_t, ndim=2] specs,
    np.ndarray[np.float_t, ndim=2] output,
):
    # get the number of channels in the data
    cdef size_t nmuls = weights.shape[0]
    cdef size_t nspecs = specs.shape[0]
    cdef size_t nchannels = specs.shape[1]
    
    for it in range(nmuls):
        for s in range(nspecs):
            output[s,out_i] = weights[it] * specs[s,in_i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _apply_rmf_vectorized(
    np.ndarray[np.float_t, ndim=2] specs,
    np.ndarray[np.float_t, ndim=2] counts,
    np.ndarray[np.float_t, ndim=2] matrix,
    np.ndarray[object, ndim=1] f_chan,
    np.ndarray[object, ndim=1] n_chan,
    np.ndarray[object, ndim=1] n_grp,
    int offset,
):
    # get the number of channels in the data
    cdef size_t nspecs = specs.shape[0]
    cdef size_t nchannels = specs.shape[1]

    # index for n_chan and f_chan incrementation
    cdef unsigned long k = 0

    # index for the response matrix incrementation
    cdef unsigned long resp_idx = 0

    # loop over all channels
    for i in range(nchannels):
        # this is the current bin in the flux spectrum to
        # be folded
        source_bin_i = specs[:,i]

        # get the current number of groups
        current_num_groups = n_grp[i]
        # loop over the current number of groups
        for j in range(current_num_groups):
            current_num_chans = n_chan[k]
            if current_num_chans == 0:
                k += 1
                resp_idx += current_num_chans
                continue
            else:
                # get the right index for the start of the counts array
                # to put the data into
                counts_idx = f_chan[k] - offset
                # this is the current number of channels to use

                k += 1
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                for l in range(nspecs):
                    counts[l,counts_idx:counts_idx + current_num_chans] += matrix[resp_idx:resp_idx + current_num_chans] * source_bin_i[l]

                # iterate the response index for next round
                resp_idx += current_num_chans

