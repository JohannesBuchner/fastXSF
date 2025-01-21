import itertools

import astropy.io.fits as pyfits
import numpy as np
import tqdm
import xspec_models_cxc as x
from scipy.interpolate import RegularGridInterpolator

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
    """<summary sentence of function in imperative>.


    Parameters
    ----------
    model: <TYPE>
        <MEANING OF model>
    energies: <TYPE>
        <MEANING OF energies>
    pars: <TYPE>
        <MEANING OF pars>

    Returns
    -------
    results: <TYPE>
        <MEANING OF results>
    """
    results = np.empty((len(pars), len(energies) - 1))
    for i, pars_i in enumerate(pars):
        results[i, :] = model(energies=energies, pars=pars_i)
    return results


class Table:
    def __init__(self, filename, method="linear"):
        """Create OGIP table.

        Parameters
        ----------
        filename: str
            filename
        method: str
            interpolation kind, passed to RegularGridInterpolator
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
        is_sorted = True
        for i, params in enumerate(itertools.product(*parameter_grid)):
            if not np.all(specdata["PARAMVAL"][i] == params):
                is_sorted = False
                break
        print("sorted:", is_sorted)
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
        self.interpolator = RegularGridInterpolator(parameter_grid, data, method=method)

    def __call__(self, energies, pars, vectorized=False):
        """Evaluate spectrum.

        Parameters
        ----------
        energies: array
            energies in keV where spectrum should be computed
        pars: list
            parameter values.

        Returns
        -------
        spectrum: array
            spectrum, corresponding to the parameter values, one entry for each value in energies.
        vectorized: bool
            <MEANING OF vectorized>
        """
        if vectorized:
            z = pars[:, -1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
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
                        fp=model_int_spectrum[i, :]
                        * self.e_model_mid
                        / self.deltae
                        * (1 + zi),
                    )
                    / 24
                )
            return results
        else:
            z = pars[-1]
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
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
                    fp=model_int_spectrum * self.e_model_mid / self.deltae * (1 + z),
                )
                / 24
            )


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    x.abundance("wilm")
    x.cross_section("vern")
    energies = np.logspace(-0.5, 1, 100)
    e_lo = energies[:-1]
    e_hi = energies[1:]
    e_mid = (e_lo + e_hi) / 2.0
    atable = Table("/home/user/Downloads/specmodels/diskreflect.fits")
    z = 1.0
    Ecut = 400
    PhoIndex = 2.0
    Incl = 87
    plt.plot(e_mid, 2 * atable(energies, [PhoIndex, Ecut, Incl, z]), label="atable")
    plt.plot(
        e_mid,
        x.pexmon(energies=energies, pars=[PhoIndex, Ecut, -1, z, 1, 1, Incl]),
        label="pexmon",
    )
    plt.xlabel("Energy [keV]")
    plt.ylabel("Spectrum [photons/cm$^2$/s]")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("pexmon.pdf")
    plt.close()

    atable = Table("/home/user/Downloads/specmodels/uxclumpy-cutoff.fits")
    NH22 = 0.5
    plt.plot(
        e_mid,
        atable(energies, [NH22, PhoIndex, Ecut, 28.0, 0.1, 45, z]),
        label="atable",
    )
    plt.plot(e_mid, x.powerlaw(energies=energies, pars=[PhoIndex]), label="pl", ls="--")
    plt.plot(
        e_mid,
        x.powerlaw(energies=energies, pars=[PhoIndex])
        * x.TBabs(energies=energies, pars=[NH22]),
        label="pl*tbabs",
    )
    plt.xlabel("Energy [keV]")
    plt.ylabel("Spectrum [photons/cm$^2$/s]")
    plt.ylim(1e-7, 1)
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("abspl.pdf")
