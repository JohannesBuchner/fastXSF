"""Statistical and astrophysical models."""
import itertools

import astropy.io.fits as pyfits
import numpy as np
import tqdm
import xspec_models_cxc as x
from scipy.interpolate import RegularGridInterpolator

from joblib import Memory

mem = Memory('.', verbose=False)


def logPoissonPDF_vectorized(models, counts):
    """Compute poisson probability.

    Parameters
    ----------
    models: array
        expected number of counts. shape is (num_models, len(counts)).
    counts: array
        observed counts (non-negative integer)

    Returns
    -------
    loglikelihood: array
        ln of the Poisson likelihood, neglecting the factorial(counts) factor,
        shape=(num_models,).
    """
    log_models = np.log(np.clip(models, 1e-100, None))
    return np.sum(log_models * counts.reshape((1, -1)), axis=1) - models.sum(axis=1)


def logPoissonPDF(model, counts):
    """Compute poisson probability.

    Parameters
    ----------
    model: array
        expected number of counts
    counts: array
        observed counts (non-negative integer)

    Returns
    -------
    loglikelihood: float
        ln of the Poisson likelihood, neglecting the factorial(counts) factor.
    """
    log_model = np.log(np.clip(model, 1e-100, None))
    return np.sum(log_model * counts) - model.sum()


def xvec(model, energies, pars):
    """Evaluate a model in a vectorized way.

    Parameters
    ----------
    model: object
        xspec model (from fastxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where to evaluate model
    pars: array
        list of parameter vectors

    Returns
    -------
    results: array
        for each parameter vector in pars, evaluates the model
        at the given energies. Has shape (pars.shape[0], energies.shape[0])
    """
    results = np.empty((len(pars), len(energies) - 1))
    for i, pars_i in enumerate(pars):
        results[i, :] = model(energies=energies, pars=pars_i)
    return results


@mem.cache
def check_if_sorted(param_vals, parameter_grid):
    """Check if parameters are stored in a sorted way.

    Parameters
    ----------
    param_vals: array
        list of parameter values stored
    parameter_grid: array
        list of possible values for each parameter

    Returns
    -------
    sorted: bool
        True if param_vals==itertools.product(*parameter_grid)
    """
    for i, params in enumerate(itertools.product(*parameter_grid)):
        if not np.all(param_vals[i] == params):
            return False
    return True


class Table:
    """Additive or multiplicative table model."""

    def __init__(self, filename, method="linear"):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        method: str
            interpolation kind, passed to RegularGridInterpolator
        """
        parameter_grid, data = self._load(filename)
        self.interpolator = RegularGridInterpolator(parameter_grid, data, method=method)

    def _load(self, filename, verbose=True):
        """Load data from file.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        verbose: bool
            whether to print the parameters of a file.

        Returns
        -------
        parameter_grid: list
            list of values for each parameter
        data: array
            array of shape `(len(g) for g in parameter_grid)`
            containing the spectra for each parameter grid point.
        """
        f = pyfits.open(filename)
        assert f[0].header["MODLUNIT"] in ("photons/cm^2/s", "ergs/cm**2/s")
        assert f[0].header["HDUCLASS"] == "OGIP"
        self.parameter_names = f["PARAMETERS"].data["NAME"]
        self.name = f[0].header["MODLNAME"]
        parameter_grid = [
            row["VALUE"][: row["NUMBVALS"]] for row in f["PARAMETERS"].data
        ]
        self.e_model_lo = f["ENERGIES"].data["ENERG_LO"]
        self.e_model_hi = f["ENERGIES"].data["ENERG_HI"]
        self.e_model_mid = (self.e_model_lo + self.e_model_hi) / 2.0
        self.deltae = self.e_model_hi - self.e_model_lo
        specdata = f["SPECTRA"].data
        # nentries = np.product([len(g) for g in parameter_grid])
        print(f'ATABLE "{self.name}"')
        for param_name, param_values in zip(self.parameter_names, parameter_grid):
            print(f"    {param_name}: {param_values.tolist()}")
        is_sorted = check_if_sorted(specdata["PARAMVAL"], parameter_grid)
        shape = tuple([len(g) for g in parameter_grid] + [len(specdata["INTPSPEC"][0])])

        if is_sorted:
            data = specdata["INTPSPEC"].reshape(shape)
        else:
            data = np.nan * np.zeros(
                [len(g) for g in parameter_grid] + [len(specdata["INTPSPEC"][0])]
            )
            for index, params, row in zip(
                tqdm.tqdm(
                    list(itertools.product(*[range(len(g)) for g in parameter_grid]))
                ),
                itertools.product(*parameter_grid),
                sorted(specdata, key=lambda row: tuple(row["PARAMVAL"])),
            ):
                np.testing.assert_allclose(params, row["PARAMVAL"])
                data[index] = row["INTPSPEC"]
        assert np.isfinite(data).all(), data
        return parameter_grid, data

    def __call__(self, energies, pars, vectorized=False):
        """Evaluate spectrum.

        Parameters
        ----------
        energies: array
            energies in keV where spectrum should be computed
        pars: list
            parameter values.
        vectorized: bool
            if true, pars is a list of parameter vectors,
            and the function returns a list of spectra.

        Returns
        -------
        spectrum: array
            photon spectrum, corresponding to the parameter values,
            one entry for each value in energies in phot/cm^2/s.
        """
        if vectorized:
            z = pars[:, -1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            delta_e = e_hi - e_lo
            model_int_spectrum = self.interpolator(pars[:, :-1])
            results = np.empty((len(z), len(e_mid)))
            for i, zi in enumerate(z):
                # this model spectrum contains for each bin [e_lo...e_hi] the integral of energy
                # now we have a new energy, energies
                results[i, :] = (
                    np.interp(
                        # look up in rest-frame, which is at higher energies at higher redshifts
                        x=e_mid * (1 + zi),
                        # in the model spectral grid
                        xp=self.e_model_mid,
                        # use spectral density, which is stretched out if redshifted.
                        fp=model_int_spectrum[i, :] / self.deltae * (1 + zi),
                    ) * delta_e / (1 + zi)
                )
            return results
        else:
            z = pars[-1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            delta_e = e_hi - e_lo
            (model_int_spectrum,) = self.interpolator([pars[:-1]])
            # this model spectrum contains for each bin [e_lo...e_hi] the integral of energy
            # now we have a new energy, energies
            return (
                np.interp(
                    # look up in rest-frame, which is at higher energies at higher redshifts
                    x=e_mid * (1 + z),
                    # in the model spectral grid
                    xp=self.e_model_mid,
                    # use spectral density, which is stretched out if redshifted.
                    fp=model_int_spectrum / self.deltae * (1 + z),
                ) * delta_e / (1 + z)
            )


class FixedTable(Table):
    """Additive or multiplicative table model with fixed energy grid."""

    def __init__(self, filename, energies, redshift=0, method="linear", fix={}):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        energies: array
            energies in keV where spectrum should be computed
        redshift: float
            Redshift
        method: str
            interpolation kind, passed to RegularGridInterpolator
        fix: dict
            dictionary of parameter names and their values to fix
            for faster data loading.
        """
        parameter_grid, data = self._load(filename)
        # interpolate data from original energy grid onto new energy grid
        self.energies = energies
        e_lo = energies[:-1]
        e_hi = energies[1:]
        e_mid = (e_lo + e_hi) / 2.0
        delta_e = e_hi - e_lo
        self.set_shape(list(data.shape), len(e_mid))
        # look up in rest-frame, which is at higher energies at higher redshifts
        self.prepare(
            parameter_grid, data,
            e_mid * (1 + redshift), delta_e / (1 + redshift), self.deltae / (1 + redshift),
            method=method, fix=fix)

    def prepare(self, parameter_grid, data, e_mid_rest, deltae_rest, model_deltae_rest, method, fix={}):
        """Prepare.

        Parameters
        ----------
        parameter_grid: array
            list of possible values for each parameter Returns
        data: array
            data table
        e_mid_rest: array
            Rest frame energy grid
        deltae_rest: array
            Channel grid spacing.
        model_deltae_rest: array
            Energy grid spacing.
        method: str
            interpolation kind, passed to RegularGridInterpolator
        fix: dict
            dictionary of parameter names and their values to fix
            for faster data loading.
        """
        # param_shapes = [len(p) for p in parameter_grid]
        # ndim = len(param_shapes)

        # Flatten the parameter grid into indices
        data_reshaped = data.reshape((-1, data.shape[-1]))
        n_points = data_reshaped.shape[0]
        mask = np.ones(n_points, dtype=bool)

        # Precompute grids
        param_grids = np.meshgrid(*parameter_grid, indexing='ij')
        # same shape as data without last dim
        param_grids_flat = [g.flatten() for g in param_grids]
        # each entry is flattened to match reshaped data
        parameter_names = [str(pname) for pname in self.parameter_names]

        # Now apply fix conditions
        for pname, val in fix.items():
            assert pname in parameter_names, (pname, self.parameter_names)
            param_idx = parameter_names.index(pname)
            mask &= (param_grids_flat[param_idx] == val)
            assert mask.any(), (f'You can only fix parameter {pname} to one of:', parameter_grid[param_idx])

        # Mask valid rows
        valid_data = data_reshaped[mask]
        # Build new parameter grid (only for unfixed parameters)
        newparameter_grid = []
        for p, pname in zip(parameter_grid, self.parameter_names):
            if pname not in fix:
                newparameter_grid.append(p)

        # Interpolate
        newdata = np.zeros((valid_data.shape[0], len(e_mid_rest)))
        for i, row in enumerate(valid_data):
            newdata[i, :] = np.interp(
                x=e_mid_rest,
                xp=self.e_model_mid,
                fp=row / model_deltae_rest,
            ) * deltae_rest

        self.set_shape(tuple([len(g) for g in newparameter_grid] + [len(e_mid_rest)]), len(e_mid_rest))
        self.interpolator = RegularGridInterpolator(
            newparameter_grid, self.prepare_function(newdata),
            method=method)

    def prepare_function(self, y):
        """Modify table rows.

        Parameters
        ----------
        y: array
            table rows

        Returns
        -------
        ynew: array
            Reshaped table.
        """
        return y.reshape(self.newshape)

    def set_shape(self, modelaxes, outlen):
        """Set shape of final table.

        Parameters
        ----------
        modelaxes: tuple
            shape of input table
        outlen: int
            length of last axis
        """
        newshape = list(modelaxes)
        newshape[-1] = outlen
        self.newshape = newshape

    def __call__(self, energies, pars, vectorized=False):
        """Evaluate spectrum.

        Parameters
        ----------
        energies: array
            energies in keV where spectrum should be computed (ignored)
        pars: list
            parameter values.
        vectorized: bool
            if true, pars is a list of parameter vectors,
            and the function returns a list of spectra.

        Returns
        -------
        spectrum: array
            photon spectrum, corresponding to the parameter values,
            one entry for each value in energies in phot/cm^2/s.
        """
        assert np.all(self.energies == energies)
        if vectorized:
            return self.interpolator(pars)
        else:
            return self.interpolator([pars])[0]


class FixedFoldedTable(FixedTable):
    """Additive or multiplicative table model folded through RMF."""

    def prepare_function(self, y):
        """Modify table rows.

        Parameters
        ----------
        y: array
            table rows

        Returns
        -------
        ynew: array
            Reshaped table.
        """
        folded_spectrum = self.RMF.apply_rmf_vectorized(y * self.ARF)
        assert folded_spectrum.shape == (len(y), self.newshape[-1]), (folded_spectrum.shape, y.shape, len(y), self.newshape[-1])
        return folded_spectrum.reshape(self.newshape)

    def set_shape(self, modelaxes, outlen):
        """Set shape of final table.

        Parameters
        ----------
        modelaxes: tuple
            shape of input table
        outlen: int
            ignored.
        """
        newshape = list(modelaxes)
        newshape[-1] = self.RMF.detchans
        self.newshape = newshape

    def __init__(self, filename, energies, RMF, ARF, redshift=0, method="linear", fix={}):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        energies: array
            energies in keV where spectrum should be computed
        redshift: float
            Redshift
        method: str
            interpolation kind, passed to RegularGridInterpolator
        fix: dict
            dictionary of parameter names and their values to fix
            for faster data loading.
        """
        self.ARF = ARF
        self.RMF = RMF
        FixedTable.__init__(self, filename=filename, energies=energies, redshift=redshift, method=method, fix=fix)

    def __call__(self, pars, vectorized=False):
        """Evaluate spectrum.

        Parameters
        ----------
        pars: list
            parameter values.
        vectorized: bool
            if true, pars is a list of parameter vectors,
            and the function returns a list of spectra.

        Returns
        -------
        spectrum: array
            photon spectrum, corresponding to the parameter values,
            one entry for each value in energies in phot/cm^2/s.
        """
        try:
            if vectorized:
                return self.interpolator(pars)
            else:
                return self.interpolator([pars])[0]
        except ValueError as e:
            raise ValueError(f'invalid parameter values passed: {pars}') from e


def prepare_folded_model0d(model, energies, pars, ARF, RMF, nonnegative=True):
    """Prepare a folded spectrum.

    Parameters
    ----------
    model: object
        xspec model (from fastxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where spectrum should be computed (ignored)
    pars: list
        parameter values.
    ARF: array
        vector for multiplication before applying the RMF
    RMF: RMF
        RMF object for folding
    nonnegative: bool
        <MEANING OF nonnegative>

    Returns
    -------
    folded_spectrum: array
        folded spectrum after applying RMF & ARF
    """
    if nonnegative:
        return RMF.apply_rmf(np.clip(model(energies=energies, pars=pars), 0, None) * ARF)
    else:
        return RMF.apply_rmf(model(energies=energies, pars=pars) * ARF)


def prepare_folded_model1d(model, energies, pars, ARF, RMF, nonnegative=True, method='linear'):
    """Prepare a function that returns the folded model.

    Parameters
    ----------
    model: object
        xspec model (from fastxsf.x module, which is xspec_models_cxc)
    energies: array
        energies in keV where spectrum should be computed (ignored)
    pars: list
        parameter values; one of the entries can be an array,
        which will be the interpolation range.
    ARF: array
        vector for multiplication before applying the RMF
    RMF: RMF
        RMF object for folding
    nonnegative: bool
        <MEANING OF nonnegative>
    method: str
        interpolation kind, passed to RegularGridInterpolator

    Returns
    -------
    folded_spectrum: array
        folded spectrum after applying RMF & ARF
    simple_interpolator: func
        function that given the free parameter value returns a folded spectrum.
    """
    mask_fixed = np.array([np.size(p) == 1 for p in pars])
    i_variable = np.where(~(mask_fixed))[0][0]

    data = np.zeros((len(pars[i_variable]), len(ARF)))
    for i, variable in enumerate(pars[i_variable]):
        pars_row = list(pars)
        pars_row[i_variable] = variable
        data[i] = model(energies=energies, pars=pars_row)
    foldeddata = RMF.apply_rmf_vectorized(data * ARF)
    if nonnegative:
        foldeddata = np.clip(foldeddata, 0, None)
    interp = RegularGridInterpolator((pars[i_variable],), foldeddata, method=method)

    def simple_interpolator(par):
        """Interpolator for a single parameter.

        Parameters
        ----------
        par: float
            The value for the one free model parameter

        Returns
        -------
        spectrum: array
            photon spectrum
        """
        try:
            return interp([par])[0]
        except ValueError as e:
            raise ValueError(f'invalid parameter value passed: {par}') from e

    return simple_interpolator
