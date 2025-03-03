import numpy as np
from astropy import units as u

def frac_overlap_interval(edges, lo, hi):
    """
    effective_lo = np.maximum(edges[:-1], lo)
    effective_hi = np.minimum(edges[1:], hi)
    assert np.all(effective_hi >= effective_lo), (edges, lo, hi, effective_lo, effective_hi)
    weights = (effective_hi - effective_lo) / (edges[1:] - edges[:-1])
    return weights
    """
    weights = np.zeros(len(edges) - 1)
    for i, (edge_lo, edge_hi) in enumerate(zip(edges[:-1], edges[1:])):
        if edge_lo > lo and edge_hi < hi:
            weight = 1
        elif edge_hi < lo:
            weight = 0
        elif edge_lo > hi:
            weight = 0
        elif hi > edge_hi and lo > edge_lo:
            weight = (edge_hi - lo) / (edge_hi - edge_lo)
        elif lo < edge_lo and hi < edge_hi:
            weight = (hi - edge_lo) / (edge_hi - edge_lo)
        else:
            weight = (min(hi, edge_hi) - max(lo, edge_lo)) / (edge_hi - edge_lo)
        weights[i] = weight
    return weights

def bins_sum(values, edges, lo, hi):
    widths = edges[1:] - edges[:-1]
    fracs = frac_overlap_interval(edges, lo, hi)
    return np.sum(widths * values * fracs)

def bins_integrate1(values, edges, lo, hi):
    mids = (edges[1:] + edges[:-1]) / 2.
    # fracs [0..1]
    fracs = frac_overlap_interval(edges, lo, hi)
    # widths: keV
    # bin local energies: keV
    # values: phot/cm^2/s
    # expected units: keV/cm^2/s --> need to multiply by energies
    return float(np.sum(mids * values * fracs))

def bins_integrate(values, edges, lo, hi):
    return np.sum(values * frac_overlap_interval(edges, lo, hi))

def photon_counts(folded_model_spectrum, chan_energies, energy_lo, energy_hi):
    Nchan = len(chan_energies) - 1
    assert folded_model_spectrum.shape == (Nchan,)
    assert chan_energies.shape == (Nchan + 1,)
    return bins_sum(folded_model_spectrum, chan_energies, energy_lo, energy_hi)
    #assert mask.sum() > 1, ('energy range:', energy_lo, energy_hi, 'not matching available grid of channel energies:', chan_energies)
    #return np.trapz(folded_model_spectrum[mask], chan_energies[mask]) * u.erg/u.cm**2/u.s

def energy_flux(unfolded_model_spectrum, energies, energy_lo, energy_hi):
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape == (Nchan,)
    assert energies.shape == (Nchan + 1,)
    return bins_integrate1(unfolded_model_spectrum, energies, energy_lo, energy_hi) * ((1 * u.keV).to(u.erg))/u.cm**2/u.s

def photon_flux(unfolded_model_spectrum, energies, energy_lo, energy_hi):
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape == (Nchan,)
    assert energies.shape == (Nchan + 1,)
    return bins_integrate(unfolded_model_spectrum, energies, energy_lo, energy_hi) / u.cm**2 / u.s

def luminosity(unfolded_model_spectrum, energies, rest_energy_lo, rest_energy_hi, z, cosmo):
    Nchan = len(energies) - 1
    assert unfolded_model_spectrum.shape == (Nchan,)
    assert energies.shape == (Nchan + 1,)
    rest_energies = energies * (1 + z)
    rest_flux = bins_integrate1(unfolded_model_spectrum, rest_energies, rest_energy_lo, rest_energy_hi) * ((1 * u.keV).to(u.erg))/u.cm**2/u.s / (1 + z)
    #mask = np.logical_and(rest_energies >= rest_energy_lo, rest_energies <= rest_energy_hi)
    #assert mask.sum() > 1, ('energy range:', rest_energy_lo, rest_energy_hi, 'not matching available grid of rest energies:', rest_energies)
    #rest_flux = np.trapz(unfolded_model_spectrum[mask], energies[mask]) * u.erg/u.cm**2/u.s
    DL = cosmo.luminosity_distance(z)
    return (rest_flux * (4 * np.pi * DL**2)).to(u.erg/u.s)
