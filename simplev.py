import numpy as np
from matplotlib import pyplot as plt
import fastxsf

# fastxsf.x.chatter(0)
fastxsf.x.abundance('wilm')
fastxsf.x.cross_section('vern')

# load the spectrum:
data = fastxsf.load_pha('example/179.pi', 0.5, 8)

# fetch some basic information from our observation
e_lo = data['e_lo']
e_hi = data['e_hi']
e_mid = (data['e_hi'] + data['e_lo']) / 2.
e_width= (data['e_hi'] - data['e_lo'])
energies = np.append(e_lo, e_hi[-1])
RMF_src = data['RMF_src']

bkg_pred_model = data['bkg_model_src_region']
chan_e = (data['chan_e_min'] + data['chan_e_max']) / 2.

absAGN = fastxsf.Table('/home/user/Downloads/specmodels/uxclumpy-cutoff.fits')

galabso = fastxsf.x.TBabs(energies=energies, pars=[data['galnh']])

# define the model:
z = np.array([data['redshift']] * 2)
bkg_norm = np.array([1.0] * 2)
norm = np.array([3e-7] * 2)
scat_norm = norm * 0.08
PhoIndex = np.array([1.9, 2.0])
TORsigma = np.array([28.0] * 2)
CTKcover = np.array([0.1] * 2)
Incl = np.array([45.0] * 2)
NH22 = np.array([1.0] * 2)
Ecut = np.array([400] * 2)

# define a likelihood
def loglikelihood(params, plot=False):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, Incl, Ecut, z, bkg_norm = np.transpose(params)

    abs_component = absAGN(energies=energies, pars=np.transpose([NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z]), vectorized=True)
    scat_norm = norm * rel_scat_norm

    scat_component = fastxsf.xvec(fastxsf.x.zpowerlw, energies=energies, pars=np.transpose([norm, PhoIndex]))
    
    pred_spec = np.einsum('ij,i->ij', abs_component, norm) + np.einsum('ij,i->ij', scat_component, scat_norm)

    pred_counts_src_srcreg = RMF_src.apply_rmf_vectorized(np.einsum('i,i,ji->ji', data['ARF'], galabso, pred_spec))[:,data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = np.einsum('j,i->ij', bkg_pred_model, bkg_norm) * data['src_to_bkg_ratio'] * data['src_expoarea']
    pred_counts_srcreg = pred_counts_src_srcreg + pred_counts_bkg_srcreg
    pred_counts_bkg_bkgreg = np.einsum('j,i->ij', bkg_pred_model, bkg_norm) * data['bkg_expoarea']
    
    if plot:
        plt.figure()
        plt.legend()
        plt.plot(data['chan_e_min'], data['src_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src+bkg')
        plt.plot(data['chan_e_min'], pred_counts_src_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='src')
        plt.plot(data['chan_e_min'], pred_counts_bkg_srcreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('src_region_counts.pdf')
        plt.close()

        plt.figure()
        plt.plot(data['chan_e_min'], data['bkg_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
        plt.plot(data['chan_e_min'], pred_counts_bkg_bkgreg[0] / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
        plt.xlabel('Channel Energy [keV]')
        plt.ylabel('Counts / keV')
        plt.legend()
        plt.savefig('bkg_region_counts.pdf')
        plt.close()

    # compute log Poisson probability
    like_srcreg = fastxsf.logPoissonPDF_vectorized(pred_counts_srcreg, data['src_region_counts'])
    like_bkgreg = fastxsf.logPoissonPDF_vectorized(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    # combined the probabilities. If fitting multiple spectra, you would add them up here as well
    return like_srcreg + like_bkgreg

import scipy.stats
PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)
z_gauss = scipy.stats.norm(data['redshift'], 0.1)

# define a prior transform
def prior_transform(cube):
    params = cube.copy()
    params[:,0] = 10**(cube[:,0] * -10)
    params[:,1] = 10**(cube[:,1] * (2 - -2) + -2)
    params[:,2] = 10**(cube[:,2] * (-1 - -5) + -5)
    params[:,3] = PhoIndex_gauss.ppf(cube[:,3])
    params[:,4] = cube[:,4] * (80 - 7) + 7
    params[:,5] = cube[:,5] * (0.4) + 0
    params[:,6] = cube[:,6] * 90
    params[:,7] = cube[:,7] * (400 - 300) + 300
    params[:,8] = z_gauss.ppf(cube[:,8])
    params[:,9] = 10**(cube[:,9] * (1 - -1) + -1)
    return params

print(loglikelihood(np.transpose([norm, NH22, np.array([0.08]*2), PhoIndex, TORsigma, CTKcover, Incl, Ecut, z, bkg_norm]), plot=True))

param_names = ['norm', 'logNH22', 'scatnorm', 'PhoIndex', 'TORsigma', 'CTKcover', 'Incl', 'Ecut', 'redshift', 'bkg_norm']
from ultranest import ReactiveNestedSampler

sampler = ReactiveNestedSampler(
    param_names, loglikelihood, prior_transform,
    log_dir='simplev', resume=True,
    vectorized=True)
sampler.run(max_num_improvement_loops=0)
sampler.plot()
