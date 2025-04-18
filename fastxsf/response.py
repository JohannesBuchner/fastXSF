"""Functionality for linear instrument response."""
# from https://github.com/dhuppenkothen/clarsach/blob/master/clarsach/respond.py
# GPL licenced code from the Clàrsach project

import astropy.io.fits as fits
import numpy as np

__all__ = ["RMF", "ARF"]


class RMF(object):
    """Response matrix file."""

    def __init__(self, filename):
        """
        Initialise.

        Parameters
        ----------
        filename : str
            The file name with the RMF FITS file
        """
        self._load_rmf(filename)

    def _load_rmf(self, filename):
        """
        Load an RMF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the RMF file

        Attributes
        ----------
        n_grp : numpy.ndarray
            the Array with the number of channels in each
            channel set

        f_chan : numpy.ndarray
            The starting channel for each channel group;
            If an element i in n_grp > 1, then the resulting
            row entry in f_chan will be a list of length n_grp[i];
            otherwise it will be a single number

        n_chan : numpy.ndarray
            The number of channels in each channel group. The same
            logic as for f_chan applies

        matrix : numpy.ndarray
            The redistribution matrix as a flattened 1D vector

        energ_lo : numpy.ndarray
            The lower edges of the energy bins

        energ_hi : numpy.ndarray
            The upper edges of the energy bins

        detchans : int
            The number of channels in the detector

        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)

        # get all the extension names
        extnames = np.array([h.name for h in hdulist])

        # figure out the right extension to use
        if "MATRIX" in extnames:
            h = hdulist["MATRIX"]

        elif "SPECRESP MATRIX" in extnames:
            h = hdulist["SPECRESP MATRIX"]
        elif "SPECRESP" in extnames:
            h = hdulist["SPECRESP"]
        else:
            raise AssertionError(f"{extnames} does not contain MATRIX or SPECRESP")

        data = h.data
        hdr = h.header
        hdulist.close()

        # extract + store the attributes described in the docstring
        n_grp = np.array(data.field("N_GRP"))
        f_chan = np.array(data.field('F_CHAN'))
        n_chan = np.array(data.field("N_CHAN"))
        matrix = np.array(data.field("MATRIX"))

        self.energ_lo = np.array(data.field("ENERG_LO"))
        self.energ_hi = np.array(data.field("ENERG_HI"))
        self.energ_unit = data.columns["ENERG_LO"].unit
        self.detchans = hdr["DETCHANS"]
        self.offset = self.__get_tlmin(h)

        # flatten the variable-length arrays
        results = self._flatten_arrays(n_grp, f_chan, n_chan, matrix)
        self.n_grp, self.f_chan, self.n_chan, self.matrix = results
        self.dense_info = None

    def __get_tlmin(self, h):
        """
        Get the tlmin keyword for `F_CHAN`.

        Parameters
        ----------
        h : an astropy.io.fits.hdu.table.BinTableHDU object
            The extension containing the `F_CHAN` column

        Returns
        -------
        tlmin : int
            The tlmin keyword
        """
        # get the header
        hdr = h.header
        # get the keys of all
        keys = np.array(list(hdr.keys()))

        # find the place where the tlmin keyword is defined
        t = np.array(["TLMIN" in k for k in keys])

        # get the index of the TLMIN keyword
        tlmin_idx = np.hstack(np.where(t))[0]

        # get the corresponding value
        tlmin = int(list(hdr.items())[tlmin_idx][1])

        return tlmin

    def _flatten_arrays(self, n_grp, f_chan, n_chan, matrix):

        if not len(n_grp) == len(f_chan) == len(n_chan) == len(matrix):
            raise ValueError("Arrays must be of same length!")

        # find all non-zero groups
        nz_idx = (n_grp > 0)

        # stack all non-zero rows in the matrix
        matrix_flat = np.hstack(matrix[nz_idx], dtype=float)

        # stack all nonzero rows in n_chan and f_chan
        # n_chan_flat = np.hstack(n_chan[nz_idx])
        # f_chan_flat = np.hstack(f_chan[nz_idx])

        # some matrices actually have more elements
        # than groups in `n_grp`, so we'll only pick out
        # those values that have a correspondence in
        # n_grp
        f_chan_new = []
        n_chan_new = []
        for i,t in enumerate(nz_idx):
            if t:
                n = n_grp[i]
                f = f_chan[i]
                nc = n_chan[i]
                if np.size(f) == 1:
                    f_chan_new.append(f.astype(np.int64) - self.offset)
                    n_chan_new.append(nc.astype(np.int64))
                else:
                    f_chan_new.append(f[:n].astype(np.int64) - self.offset)
                    n_chan_new.append(nc[:n].astype(np.int64))

        n_chan_flat = np.hstack(n_chan_new)
        f_chan_flat = np.hstack(f_chan_new)

        return n_grp, f_chan_flat, n_chan_flat, matrix_flat

    def strip(self, channel_mask):
        """
        Strip response matrix of entries outside the channel mask.

        Parameters
        ----------
        channel_mask : array
            Boolean array indicating which detector channel to keep.
        """
        n_grp_new = np.zeros_like(self.n_grp)
        n_chan_new = []
        f_chan_new = []
        matrix_new = []

        in_indices = []
        out_indices = []
        weights = []
        k = 0
        resp_idx = 0
        # loop over all channels
        for i in range(len(self.energ_lo)):
            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups]
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                # outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                mask_valid = channel_mask[counts_idx:counts_idx + current_num_chans]
                if current_num_chans > 0 and mask_valid.any():
                    # add block
                    n_grp_new[i] += 1
                    # length
                    n_chan_new.append(current_num_chans)
                    # location in matrix
                    f_chan_new.append(counts_idx)
                    matrix_new.append(self.matrix[inslice])

                    in_indices.append((i + np.zeros(current_num_chans, dtype=int))[mask_valid])
                    out_indices.append(np.arange(counts_idx, counts_idx + current_num_chans)[mask_valid])
                    weights.append(self.matrix[inslice][mask_valid])
                resp_idx += current_num_chans
            k += current_num_groups

        out_indices = np.hstack(out_indices)
        in_indices = np.hstack(in_indices)
        weights = np.hstack(weights)
        out_index_chunks = []
        for j in np.arange(self.detchans):
            if np.any(out_indices == j):
                out_index_chunks.append((j, in_indices[out_indices == j], weights[out_indices == j]))

        self.n_chan = np.array(n_chan_new)
        self.f_chan = np.array(f_chan_new)
        self.n_grp = n_grp_new
        self.matrix = np.hstack(matrix_new)
        self.dense_info = out_index_chunks

    def apply_rmf(self, spec):
        """
        Fold the spectrum through the redistribution matrix.

        The redistribution matrix is saved as a flattened 1-dimensional
        vector to save space. In reality, for each entry in the flux
        vector, there exists one or more sets of channels that this
        flux is redistributed into. The additional arrays `n_grp`,
        `f_chan` and `n_chan` store this information:
            * `n_group` stores the number of channel groups for each
              energy bin
            * `f_chan` stores the *first channel* that each channel
              for each channel set
            * `n_chan` stores the number of channels in each channel
              set

        As a result, for a given energy bin i, we need to look up the
        number of channel sets in `n_grp` for that energy bin. We
        then need to loop over the number of channel sets. For each
        channel set, we look up the first channel into which flux
        will be distributed as well as the number of channels in the
        group. We then need to also loop over the these channels and
        actually use the corresponding elements in the redistribution
        matrix to redistribute the photon flux into channels.

        All of this is basically a big bookkeeping exercise in making
        sure to get the indices right.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        Returns
        -------
        counts : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel

        """
        if self.dense_info is not None:
            out = np.zeros(self.detchans)
            for i, in_indices_i, weights_i in self.dense_info:
                out[i] = np.dot(spec[in_indices_i], weights_i)
            return out

        # get the number of channels in the data
        nchannels = spec.shape[0]

        # an empty array for the output counts
        counts = np.zeros(nchannels)

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):

            # this is the current bin in the flux spectrum to
            # be folded
            source_bin_i = spec[i]

            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups]
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                counts[outslice] += self.matrix[inslice] * source_bin_i
                # iterate the response index for next round
                resp_idx += current_num_chans
            k += current_num_groups

        return counts[:self.detchans]

    def apply_rmf_vectorized(self, specs):
        """
        Fold the spectrum through the redistribution matrix.

        The redistribution matrix is saved as a flattened 1-dimensional
        vector to save space. In reality, for each entry in the flux
        vector, there exists one or more sets of channels that this
        flux is redistributed into. The additional arrays `n_grp`,
        `f_chan` and `n_chan` store this information:
            * `n_group` stores the number of channel groups for each
              energy bin
            * `f_chan` stores the *first channel* that each channel
              for each channel set
            * `n_chan` stores the number of channels in each channel
              set

        As a result, for a given energy bin i, we need to look up the
        number of channel sets in `n_grp` for that energy bin. We
        then need to loop over the number of channel sets. For each
        channel set, we look up the first channel into which flux
        will be distributed as well as the number of channels in the
        group. We then need to also loop over the these channels and
        actually use the corresponding elements in the redistribution
        matrix to redistribute the photon flux into channels.

        All of this is basically a big bookkeeping exercise in making
        sure to get the indices right.

        Parameters
        ----------
        specs : numpy.ndarray
            The (model) spectra to be folded

        Returns
        -------
        counts : numpy.ndarray
            The (model) spectrum after folding, in counts/s/channel

        """
        # get the number of channels in the data
        nspecs, nchannels = specs.shape

        # an empty array for the output counts
        counts = np.zeros((nspecs, nchannels))

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):

            # this is the current bin in the flux spectrum to
            # be folded
            source_bin_i = specs[:,i]

            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for current_num_chans, counts_idx in zip(
                self.n_chan[k:k + current_num_groups],
                self.f_chan[k:k + current_num_groups]
            ):
                # add the flux to the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels
                to_add = np.outer(source_bin_i, self.matrix[resp_idx:resp_idx + current_num_chans])
                counts[:,counts_idx:counts_idx + current_num_chans] += to_add

                # iterate the response index for next round
                resp_idx += current_num_chans
            k += current_num_groups

        return counts[:,:self.detchans]

    def get_dense_matrix(self):
        """Extract the redistribution matrix as a dense numpy matrix.

        The redistribution matrix is saved as a 1-dimensional
        vector to save space (see apply_rmf for more information).
        This function converts it into a dense array.

        Returns
        -------
        dense_matrix : numpy.ndarray
            The RMF as a dense 2d matrix.
        """
        # get the number of channels in the data
        nchannels = len(self.energ_lo)
        nenergies = self.detchans

        # an empty array for the output counts
        dense_matrix = np.zeros((nchannels, nenergies))

        # index for n_chan and f_chan incrementation
        k = 0

        # index for the response matrix incrementation
        resp_idx = 0

        # loop over all channels
        for i in range(nchannels):
            # get the current number of groups
            current_num_groups = self.n_grp[i]

            # loop over the current number of groups
            for _ in range(current_num_groups):
                current_num_chans = int(self.n_chan[k])
                # get the right index for the start of the counts array
                # to put the data into
                counts_idx = self.f_chan[k]
                # this is the current number of channels to use

                k += 1

                # assign the subarray of the counts array that starts with
                # counts_idx and runs over current_num_chans channels

                outslice = slice(counts_idx, counts_idx + current_num_chans)
                inslice = slice(resp_idx, resp_idx + current_num_chans)
                dense_matrix[i,outslice] = self.matrix[inslice]

                # iterate the response index for next round
                resp_idx += current_num_chans

        return dense_matrix


class ARF(object):
    """Area response file."""

    def __init__(self, filename):
        """Initialise.

        Parameters
        ----------
        filename : str
            The file name with the ARF file
        """
        self._load_arf(filename)
        pass

    def _load_arf(self, filename):
        """Load an ARF from a FITS file.

        Parameters
        ----------
        filename : str
            The file name with the ARF file
        """
        # open the FITS file and extract the MATRIX extension
        # which contains the redistribution matrix and
        # anxillary information
        hdulist = fits.open(filename)
        h = hdulist["SPECRESP"]
        data = h.data
        hdr = h.header
        hdulist.close()

        # extract + store the attributes described in the docstring

        self.e_low = np.array(data.field("ENERG_LO"))
        self.e_high = np.array(data.field("ENERG_HI"))
        self.e_unit = data.columns["ENERG_LO"].unit
        self.specresp = np.array(data.field("SPECRESP"))

        if "EXPOSURE" in list(hdr.keys()):
            self.exposure = hdr["EXPOSURE"]
        else:
            self.exposure = 1.0

        if "FRACEXPO" in data.columns.names:
            self.fracexpo = data["FRACEXPO"]
        else:
            self.fracexpo = 1.0

    def apply_arf(self, spec, exposure=None):
        """
        Fold the spectrum through the ARF.

        The ARF is a single vector encoding the effective area information
        about the detector. A such, applying the ARF is a simple
        multiplication with the input spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            The (model) spectrum to be folded

        exposure : float, default None
            Value for the exposure time. By default, `apply_arf` will use the
            exposure keyword from the ARF file. If this exposure time is not
            correct (for example when simulated spectra use a different exposure
            time and the ARF from a real observation), one can override the
            default exposure by setting the `exposure` keyword to the correct
            value.

        Returns
        -------
        s_arf : numpy.ndarray
            The (model) spectrum after folding, in
            counts/s/channel
        """
        assert spec.shape[0] == self.specresp.shape[0], (
            "Input spectrum and ARF must be of same size.",
            spec.shape, self.specresp.shape)
        e = self.exposure if exposure is None else exposure
        return np.array(spec) * self.specresp * e
