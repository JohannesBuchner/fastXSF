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

    def __init__(self, filename, energies, redshift=0, method="linear"):
        """Initialise.

        Parameters
        ----------
        filename: str
            filename of a OGIP FITS file.
        method: str
            interpolation kind, passed to RegularGridInterpolator
        """
        parameter_grid, data = self._load(filename)
        # interpolate data from original energy grid onto new energy grid
        self.energies = energies
        e_lo = energies[:-1]
        e_hi = energies[1:]
        e_mid = (e_lo + e_hi) / 2.0
        delta_e = e_hi - e_lo
        newshape = list(data.shape)
        newshape[-1] = len(e_mid)
        newdata = np.zeros((data.size // data.shape[-1], len(e_mid)))
        for i, row in enumerate(data.reshape((-1, data.shape[-1]))):
            newdata[i, :] = np.interp(
                # look up in rest-frame, which is at higher energies at higher redshifts
                x=e_mid * (1 + redshift),
                # in the model spectral grid
                xp=self.e_model_mid,
                # use spectral density, which is stretched out if redshifted.
                fp=row / self.deltae * (1 + redshift),
            ) * delta_e / (1 + redshift)
        self.interpolator = RegularGridInterpolator(
            parameter_grid, newdata.reshape(newshape),
            method=method)

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
