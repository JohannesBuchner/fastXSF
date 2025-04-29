"""

A new approach to X-ray spectral fitting with Xspec models and
optimized nested sampling.

This script explains how to set up your model.

The idea is that you create a function which computes all model components.

"""
import os
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from optns.sampler import OptNS
from optns.profilelike import GaussianPrior

import fastxsf

# fastxsf.x.chatter(0)
fastxsf.x.abundance('wilm')
fastxsf.x.cross_section('vern')

# let's take a realistic example of a Chandra + NuSTAR FPMA + FPMB spectrum
# with normalisation cross-calibration uncertainty of +-0.2 dex.
# and a soft apec, a pexmon and a UXCLUMPY model, plus a background of course

# we want to make pretty plots of the fit and its components, folded and unfolded,
#  compute 2-10 keV fluxes of the components
#  compute luminosities of the intrinsic power law

filepath = '/mnt/data/daten/PostDoc2/research/agn/eROSITA/xlf/xrayspectra/NuSTARenhance/COSMOS/spectra/102/'

data_sets = {
    'Chandra': fastxsf.load_pha(filepath + 'C.pha', 0.5, 8),
    'NuSTAR-FPMA': fastxsf.load_pha(filepath + 'A.pha', 4, 77),
    'NuSTAR-FPMB': fastxsf.load_pha(filepath + 'B.pha', 4, 77),
}

redshift = data_sets['Chandra']['redshift']

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
galabsos = {
    k: fastxsf.x.TBabs(energies=data['energies'], pars=[data['galnh']])
    for k, data in data_sets.items()
}

# load a Table model
tablepath = os.path.join(os.environ.get('MODELDIR', '.'), 'uxclumpy-cutoff.fits')
import time
#t0 = time.time()
#print("preparing fixed table models...")
#absAGNs = {
#    k: fastxsf.model.FixedTable(tablepath, energies=data['energies'], redshift=redshift)
#    for k, data in data_sets.items()
#}
#print(f'took {time.time() - t0:.3f}s')
t0 = time.time()
print("preparing folded table models...")
absAGN_folded = {
    k: fastxsf.model.FixedFoldedTable(
        tablepath, energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'], redshift=redshift, fix=dict(Ecut=400, Theta_inc=60))
    for k, data in data_sets.items()
}
print(f'took {time.time() - t0:.3f}s')
t0 = time.time()
print("preparing 1d interpolated models...")
scat_folded = {
    k: fastxsf.model.prepare_folded_model1d(fastxsf.x.zpowerlw, pars=[np.arange(1.0, 3.1, 0.01), redshift], energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'])
    for k, data in data_sets.items()
}
apec_folded = {
    k: fastxsf.model.prepare_folded_model1d(fastxsf.x.apec, pars=[10**np.arange(-2, 1.2, 0.01), 1.0, redshift], energies=data['energies'], ARF=data['ARF'] * galabsos[k], RMF=data['RMF_src'])
    for k, data in data_sets.items()
}
print(f'took {time.time() - t0:.3f}s')
print(scat_folded['Chandra'](2.0), scat_folded['Chandra']([2.0]).shape)
assert scat_folded['Chandra'](2.0).shape == data_sets['Chandra']['chan_mask'].shape
print(apec_folded['Chandra'](2.0), apec_folded['Chandra']([2.0]).shape)
assert apec_folded['Chandra'](2.0).shape == data_sets['Chandra']['chan_mask'].shape

# pre-compute the absorption factors -- no need to call this again and again if the parameters do not change!
Incl = 45.0
Ecut = 400

# lets now start using optimized nested sampling.

# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['logNH', 'PhoIndex', 'TORsigma', 'CTKcover', 'kT']

# set up a prior transform
PhoIndex_gauss = scipy.stats.truncnorm(loc=1.95, scale=0.15, a=(1.0 - 1.95) / 0.15, b=(3.0 - 1.95) / 0.15)
def nonlinear_param_transform(cube):
    params = cube.copy()
    params[0] = cube[0] * 6 + 20    # logNH
    params[1] = PhoIndex_gauss.ppf(cube[1])
    params[2] = cube[2] * (80 - 7) + 7
    params[3] = cube[3] * (0.4) + 0
    params[4] = 10**(cube[4] * 2 - 1)  # kT
    return params

#component_names = ['pl', 'scat', 'apec']
linear_param_names = ['Nbkg', 'Npl', 'Nscat', 'Napec']
#for k in data_sets.keys():
#    for name in component_names + ['bkg']:
#        linear_param_names.append(f'norm_{name}_{k}')

bkg_deviations = 0.2
src_deviations = 0.1

Nlinear = len(linear_param_names)
Ndatasets = len(data_sets)
linear_param_prior_Sigma_offset = np.eye(Nlinear * Ndatasets) * 0
linear_param_prior_Sigma = np.eye(Nlinear * Ndatasets) * 0
for j in range(len(data_sets)):
    # for all data-sets, set a parameter prior:
    linear_param_prior_Sigma[j * Nlinear + 3, j * Nlinear + 3] = bkg_deviations**-2
    # across data-sets set a mutual parameter prior for each normalisation
    for k in range(j + 1, len(data_sets)):
        linear_param_prior_Sigma[j * Nlinear + 0, k * Nlinear + 0] = src_deviations**-2
        linear_param_prior_Sigma[j * Nlinear + 1, k * Nlinear + 1] = src_deviations**-2
        linear_param_prior_Sigma[j * Nlinear + 2, k * Nlinear + 2] = src_deviations**-2
    # set a prior, apply it only to the first data-set
    if j == 0:
        # -5 +- 2 for ratio of pl and scat normalisations, only on first data set
        linear_param_prior_Sigma_offset[j * Nlinear + 3, j * Nlinear + 3] = -5
        linear_param_prior_Sigma[j * Nlinear + 3, j * Nlinear + 3] = 2.0**-2

lognorms_prior = GaussianPrior(linear_param_prior_Sigma_offset, linear_param_prior_Sigma)


def linear_param_logprior(params):
    assert np.all(params > 0)
    lognorms = np.log(params.reshape((-1, Nlinear, len(data_sets))))
    Npl = lognorms[:, linear_param_names.index('Npl'), :]
    Nscat = lognorms[:, linear_param_names.index('Nscat'), :]
    #Napec = lognorms[:, component_names.index('apec'), :]
    #Nbkg = lognorms[:, component_names.index('bkg'), :]
    logp = np.where(Nscat > np.log(0.1) + Npl, -np.inf, 0)
    return logp


class LinkedPredictionPacker:
    """Map source and background spectral components to counts,

    Identical components for each dataset.

    pred_counts should look like
    pred_counts should look like
    component1-norm1: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    component2-norm2: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    component3-norm3: [counts_data1, 0, counts_data2, 0, counts_data3, 0]
    background-bkg:   [counts_srcbkg1, counts_bkgbkg1, counts_srcbkg2, counts_bkgbkg2, counts_srcbkg3, counts_bkgbkg3]
    """
    def __init__(self, data_sets, Ncomponents):
        """Initialise."""
        self.data_sets = data_sets
        self.width = 0
        self.Ncomponents = Ncomponents
        self.counts_flat = np.hstack(tuple([
            np.hstack((data['src_region_counts'], data['bkg_region_counts']))
            for k, data in data_sets.items()]))
        self.width, = self.counts_flat.shape

    def pack(self, pred_fluxes):
        # one row for each normalisation,
        pred_counts = np.zeros((self.Ncomponents, self.width))
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            for i, component_spec in enumerate(pred_fluxes[k]):
                pred_counts[i, left:left + Ndata] = component_spec
            # now look at background in the background region
            left += Ndata
            for i, component_spec in enumerate(pred_fluxes[k + '_bkg']):
                # fill in background
                pred_counts[i, left:left + Ndata] = component_spec
            left += Ndata
        return pred_counts

    def unpack(self, pred_counts):
        pred_fluxes = {}
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            pred_fluxes[k] = pred_counts[:, left:left + Ndata]
            # now look at background in the background region
            left += Ndata
            pred_fluxes[k + '_bkg'] = pred_counts[:, left:left + Ndata]
            left += Ndata
        return pred_fluxes

    def prior_predictive_check_plot(self, ax, unit='counts', nsamples=8):
        src_factor = 1
        bkg_factor = 1
        colors = {}
        ylo = np.inf
        yhi = 0
        # now we need to unpack again:
        markers = 'osp><d^v'
        for (k, data), marker in zip(self.data_sets.items(), markers):
            if unit != 'counts':
                src_factor = 1. / data['chan_const_spec_weighting']
                bkg_factor = 1. / (data['chan_const_spec_weighting'] * data["bkg_expoarea"] / data["src_expoarea"])
            x = (data['chan_e_min'] + data['chan_e_max']) / 2.0

            l, = ax.plot(x, data['src_region_counts'] * src_factor, marker=marker, ls=' ', ms=2, label=f'data: {k}')
            colors[k + ' total'] = l.get_color()
            ax.plot(x, data['bkg_region_counts'] * bkg_factor, marker=marker, ls=' ', ms=2, mfc='none', mec=colors[k + ' total'], label=f' bkg: {k}', alpha=0.5)
            ylo = min(ylo, np.min((0.1 + data['src_region_counts']) * src_factor))
            yhi = max(yhi, np.max(1.5 * data['src_region_counts'] * src_factor))

        for i in range(nsamples):
            u = np.random.uniform(size=len(statmodel.nonlinear_param_names))
            nonlinear_params = statmodel.nonlinear_param_transform(u)
            X = statmodel.compute_model_components(nonlinear_params)
            statmodel.statmodel.update_components(X)
            norms = statmodel.statmodel.norms()
            pred_counts = norms @ X.T
            left = 0
            for k, data in self.data_sets.items():
                if unit != 'counts':
                    src_factor = 1. / data['chan_const_spec_weighting']
                    bkg_factor = 1. / (data['chan_const_spec_weighting'] * data["bkg_expoarea"] / data["src_expoarea"])
                x = (data['chan_e_min'] + data['chan_e_max']) / 2.0

                Ndata = data['chan_mask'].sum()
                for j, norm in enumerate(norms):
                    if j == 0:
                        color = colors.get(k + ' total')
                        ls = '--'
                    else:
                        color = colors.get(k + ' ' + linear_param_names[j])
                        if color is None and unit != 'counts':
                            color = colors.get(linear_param_names[j])
                        ls = '-'
                    label = f'{k} {statmodel.linear_param_names[j]}' if i == 0 else None
                    l, = ax.plot(x, norms[j] * X[left:left + Ndata, j] * src_factor, alpha=0.5, lw=1, ls=ls, label=label, color=color)
                    colors[k + ' ' + linear_param_names[j]] = l.get_color()
                    colors[linear_param_names[j]] = l.get_color()

                color = colors.get(k + ' total')
                label = f'{k}: total' if i == 0 else None
                ax.plot(x, pred_counts[left:left + Ndata] * src_factor, alpha=0.5, lw=2, ls='--', color=color, label=label)
                # now look at background in the background region
                left += Ndata
                label = f'{k}: bkg' if i == 0 else None
                ax.plot(x, pred_counts[left:left + Ndata] * bkg_factor, alpha=0.2, lw=2, ls=':', color=color, label=label)
                # skip plotting sub-components for background region
                left += Ndata

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(ylo / 10, yhi)
        ax.set_ylabel('Counts' if unit == 'counts' else 'Counts / cm$^2$ / s / keV')
        ax.set_xlabel('Energy [keV]')


class IndependentPredictionPacker:
    """Map source and background spectral components to counts,

    Independent components for each dataset.

    pred_counts should look like
    component1-norm11: [counts_data1, 0...0]
    component1-norm12: [0, counts_data2...0]
    component1-norm13: [0, ..., counts_data3]
    component2-norm21: [counts_data1, 0...0]
    component2-norm22: [0, counts_data2...0]
    component2-norm23: [0, ..., counts_data3]
    component3-norm31: [counts_data1, 0...0]
    component3-norm32: [0, counts_data2...0]
    component3-norm33: [0, ..., counts_data3]
    background-bkg:   [counts_srcbkg1, counts_bkgbkg1, counts_srcbkg2, counts_bkgbkg2, counts_srcbkg3, counts_bkgbkg3]
    """
    def __init__(self, data_sets, Ncomponents):
        """Initialise."""
        self.data_sets = data_sets
        self.Ndatasets = len(self.data_sets)
        self.Ncomponents = Ncomponents
        self.counts_flat = np.hstack(tuple([
            np.hstack((data['src_region_counts'], data['bkg_region_counts']))
            for k, data in data_sets.items()]))
        self.width, = self.counts_flat.shape

    def pack(self, pred_fluxes):
        # one row for each normalisation,
        pred_counts = np.zeros((self.Ncomponents * self.Ndatasets, self.width))
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            for i, component_spec in enumerate(pred_fluxes[k]):
                row_index = k * self.Ncomponents + i
                pred_counts[row_index, left:left + Ndata] = component_spec
            # now look at background in the background region
            left += Ndata
            for i, component_spec in enumerate(pred_fluxes[k + '_bkg']):
                row_index = k * self.Ncomponents + i
                # fill in background
                pred_counts[row_index, left:left + Ndata] = component_spec
            left += Ndata
        return pred_counts
    def unpack(self, pred_counts):
        pred_fluxes = {}
        # now let's apply the response to each component:
        left = 0
        for k, data in self.data_sets.items():
            Ndata = data['chan_mask'].sum()
            pred_fluxes[k] = pred_counts[:, left:left + Ndata]
            # now look at background in the background region
            left += Ndata
            pred_fluxes[k + '_bkg'] = pred_counts[:, left:left + Ndata]
            left += Ndata
        return pred_fluxes


def compute_model_components(params):
    assert np.isfinite(params).all(), params
    logNH, PhoIndex, TORsigma, CTKcover, kT = params

    pred_counts = {}

    for k, data in data_sets.items():
        # compute model components for each data set:
        #src_spectral_components = [abs_component, scat, apec]
        pred_counts[k] = []
        pred_counts[k].append(data['bkg_model_src_region'] * data['src_expoarea'])
        assert np.all(pred_counts[k][0] >= 0)
        pred_counts[k + '_bkg'] = []
        pred_counts[k + '_bkg'].append(data['bkg_model_bkg_region'] * data['bkg_expoarea'])
        assert np.all(pred_counts[k + '_bkg'][0] >= 0)

        # energies = data['energies']
        # first component: a absorbed power law
        #abs_component = absAGNs[k](energies=energies, pars=[
        #    10**(logNH - 22), PhoIndex, Ecut, TORsigma, CTKcover, Incl])

        # second component, a copy of the unabsorbed power law
        #scat = fastxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, redshift])

        # third component, a apec model
        #apec = np.clip(fastxsf.x.apec(energies=energies, pars=[kT, 1.0, redshift]), 0, None)
        #assert np.all(abs_component >= 0)
        #assert np.all(scat >= 0)
        #assert np.all(apec >= 0), apec[~(apec >= 0)]

        #weighted_src_spectral_components = np.einsum('i,i,ji->ji', data['ARF'], galabsos[k], src_spectral_components)
        #for row in data['RMF_src'].apply_rmf_vectorized(weighted_src_spectral_components):
        #    pred_counts[k].append(row[data['chan_mask']] * data['src_expoarea'])
        #    assert np.all(row >= 0), k
        pred_counts[k].append(absAGN_folded[k](pars=[10**(logNH - 22), PhoIndex, TORsigma, CTKcover])[data['chan_mask']] * data['src_expoarea'])
        pred_counts[k].append(scat_folded[k](PhoIndex)[data['chan_mask']] * data['src_expoarea'])
        pred_counts[k].append(apec_folded[k](kT)[data['chan_mask']] * data['src_expoarea'])
        #for src_spectral_component in weighted_src_spectral_components:
        #    # now let's apply the response to each component:
        #    pred_counts[k].append(data['RMF_src'].apply_rmf(src_spectral_component)[data['chan_mask']] * data['src_expoarea'])
        #    assert np.all(pred_counts[k][-1] >= 0), k
        """
        src_spectral_components = np.array([abs_component, scat, apec])
        assert np.all(abs_component >= 0)
        assert np.all(scat >= 0)
        assert np.all(apec >= 0), apec[~(apec >= 0)]
        pred_counts[k] = np.zeros((4, data['chan_mask'].sum()))
        pred_counts[k][0] = data['bkg_model_src_region'] * data['src_expoarea']
        assert np.all(pred_counts[k][0] >= 0)
        pred_counts[k + '_bkg'] = np.zeros((4, data['chan_mask'].sum()))
        pred_counts[k + '_bkg'][0] = data['bkg_model_bkg_region'] * data['bkg_expoarea']
        assert np.all(pred_counts[k + '_bkg'][0] >= 0)
        #pred_counts[k][1:,:] = data['RMF_src'].apply_rmf_vectorized(
        #    src_spectral_components * (data['ARF'] * galabsos[k]))[:,data['chan_mask']] * data['src_expoarea']
        for j, src_spectral_component in enumerate(src_spectral_components):
            # now let's apply the response to each component:
            pred_counts[k][j] = data['RMF_src'].apply_rmf(
                data['ARF'] * galabsos[k] * src_spectral_component)[data['chan_mask']] * data['src_expoarea']
            assert np.all(pred_counts[k][j] >= 0), (k, j)
        """
    return pred_counts

for k, data in data_sets.items():
    print(k, 'expoarea:', data['src_expoarea'], data['bkg_expoarea'])
    if k.startswith('NuSTAR'):
        data['src_expoarea'] *= 50
        data['bkg_expoarea'] *= 50

X = compute_model_components([24.5, 2.0, 30.0, 0.4, 0.5])
pp = LinkedPredictionPacker(data_sets, 4)
counts_model = pp.pack(X)

print(counts_model.shape, counts_model.sum(axis=0), counts_model.sum(axis=1))

# make it so that spectra have ~10000 counts each
target_counts = np.array([40, 40000, 4, 400])
norms = target_counts / counts_model.sum(axis=1)
norms[0] = 1.0

# let's compute some luminosities
print(f'norms: {norms}')

# simulate spectra and fill in the counts
rng = np.random.RandomState(42)
print('Expected total counts:', norms @ np.sum(counts_model, axis=1))
for k, data in data_sets.items():
    print(f'  Expected counts for {k}: {np.sum(norms @ np.array(X[k]))}')
    counts = rng.poisson(norms @ np.array(X[k]))
    print(f'  Final counts for {k}: {np.sum(counts):d}')
    counts_bkg = rng.poisson(data['bkg_model_bkg_region'] * data['bkg_expoarea'])

    # write result into the data set
    data['src_region_counts'] = counts
    data['bkg_region_counts'] = counts_bkg

pp = LinkedPredictionPacker(data_sets, 4)

def compute_model_components_unnamed(params):
    return pp.pack(compute_model_components(params)).T

# then we need some glue between OptNS and our dictionaries
statmodel = OptNS(
    linear_param_names, nonlinear_param_names, compute_model_components_unnamed,
    nonlinear_param_transform, linear_param_logprior,
    pp.counts_flat, positive=True)

# prior predictive checks:
fig = plt.figure(figsize=(15, 4))
pp.prior_predictive_check_plot(fig.gca())
plt.legend(ncol=4)
plt.savefig('multispecopt-ppc-counts.pdf')
plt.close()

fig = plt.figure(figsize=(15, 4))
pp.prior_predictive_check_plot(fig.gca(), unit='area')
plt.legend(ncol=4)
plt.savefig('multispecopt-ppc.pdf')
plt.close()

print("starting benchmark...")
import time, tqdm
t0 = time.time()
for i in tqdm.trange(1000):
    u = np.random.uniform(size=len(statmodel.nonlinear_param_names))
    nonlinear_params = statmodel.nonlinear_param_transform(u)
    assert np.isfinite(nonlinear_params).all()
    X = statmodel.compute_model_components(nonlinear_params)
    assert np.isfinite(X).all()
    statmodel.statmodel.update_components(X)
    norms = statmodel.statmodel.norms()
    assert np.isfinite(norms).all()
    pred_counts = norms @ X.T
print('Duration:', (time.time() - t0) / 1000)

optsampler = statmodel.ReactiveNestedSampler(
    log_dir='multispecoptjit', resume=True)
# run the UltraNest optimized sampler on the nonlinear parameter space:
optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
optsampler.print_results()
optsampler.plot()
# now for the linear (normalisation) parameters:

# set up a prior log-probability density function for these linear parameters:


"""

# create OptNS object, and give it all of these ingredients,
# as well as our data
# create a UltraNest sampler from this. You can pass additional arguments like here:

# now for postprocessing the results, we want to get the full posterior:
# this samples up to 1000 normalisations for each nonlinear posterior sample:
fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'][:400], 100)
print(f'Obtained {len(fullsamples)} weighted posterior samples')

print('weights:', weights, np.nanmin(weights), np.nanmax(weights), np.mean(weights))
# make a corner plot:
mask = weights > 1e-6 * np.nanmax(weights)
fullsamples_selected = fullsamples[mask,:]
fullsamples_selected[:, :len(linear_param_names)] = np.log10(fullsamples_selected[:, :len(linear_param_names)])

print(f'Obtained {mask.sum()} with not minuscule weight.')
fig = corner.corner(
    fullsamples_selected, weights=weights[mask],
    labels=linear_param_names + nonlinear_param_names,
    show_titles=True, quiet=True,
    plot_datapoints=False, plot_density=False,
    levels=[0.9973, 0.9545, 0.6827, 0.3934], quantiles=[0.15866, 0.5, 0.8413],
    contour_kwargs=dict(linestyles=['-','-.',':','--'], colors=['navy','navy','navy','purple']),
    color='purple'
)
plt.savefig('simpleopt-corner.pdf')
plt.close()

# to obtain equally weighted samples, we resample
# this respects the effective sample size. If you get too few samples here,
# crank up the number just above.
samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
print(f'Obtained {len(samples)} equally weighted posterior samples')


# prior predictive checks:
fig = plt.figure(figsize=(15, 10))
statmodel.posterior_predictive_check_plot(fig.gca(), samples[:100])
plt.legend()
plt.ylim(0.1, counts_flat.max() * 1.1)
plt.yscale('log')
plt.savefig('simpleopt-postpc.pdf')
plt.close()

"""
