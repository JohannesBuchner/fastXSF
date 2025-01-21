import numpy as np
from matplotlib import pyplot as plt
import fastxsf

# fastxsf.x.chatter(0)
fastxsf.x.abundance('wilm')
fastxsf.x.cross_section('vern')

# load the spectrum:
data = fastxsf.load_pha('example/179.pi', 0.5, 8)

e_lo = data['e_lo']
e_hi = data['e_hi']
e_mid = (data['e_hi'] + data['e_lo']) / 2.
e_width= (data['e_hi'] - data['e_lo'])
energies = np.append(e_lo, e_hi[-1])
RMF_src = data['RMF_src']

bkg_pred_model = data['bkg_model_src_region']
#bkg_pred = bkg_pred_model * data['src_to_bkg_ratio']
chan_e = (data['chan_e_min'] + data['chan_e_max']) / 2.

absAGN = fastxsf.Table('/home/user/Downloads/specmodels/uxclumpy-cutoff.fits')

z = data['redshift']
galabso = fastxsf.x.TBabs(energies=energies, pars=[data['galnh']])

# define the model:
bkg_norm = 1.0
norm = 3e-7
scat_norm = norm * 0.08
PhoIndex = 2.0
TORsigma = 28.0
CTKcover = 0.1
Incl = 45.0
NH22 = 1.0
Ecut = 400

def logPoissonPDF(model, counts):
    log_model = np.log(model)
    return np.sum(log_model * counts) - log_model.sum()

# define a likelihood
def loglikelihood(params, plot=False):
    norm, NH22, rel_scat_norm, PhoIndex, TORsigma, CTKcover, Incl, Ecut, bkg_norm = params

    scat_norm = norm * rel_scat_norm
    abs_component = absAGN(energies=energies, pars=[NH22, PhoIndex, Ecut, TORsigma, CTKcover, Incl, z])

    scat_component = fastxsf.x.zpowerlw(energies=energies, pars=[norm, PhoIndex])

    pred_spec = abs_component * norm + scat_component * scat_norm

    pred_counts_src_srcreg = RMF_src.apply_rmf(data['ARF'] * (galabso * pred_spec))[data['chan_mask']] * data['src_expoarea']
    pred_counts_bkg_srcreg = bkg_pred_model * bkg_norm * data['src_to_bkg_ratio'] * data['src_expoarea']
    pred_counts_srcreg = pred_counts_src_srcreg + pred_counts_bkg_srcreg
    pred_counts_bkg_bkgreg = bkg_pred_model * bkg_norm * data['bkg_expoarea']
    
    plt.figure()
    plt.legend()
    plt.plot(data['chan_e_min'], data['src_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
    plt.plot(data['chan_e_min'], pred_counts_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='src+bkg')
    plt.plot(data['chan_e_min'], pred_counts_src_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='src')
    plt.plot(data['chan_e_min'], pred_counts_bkg_srcreg / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
    plt.xlabel('Channel Energy [keV]')
    plt.ylabel('Counts / keV')
    plt.legend()
    plt.savefig('src_region_counts.pdf')
    plt.close()

    plt.figure()
    plt.plot(data['chan_e_min'], data['bkg_region_counts'] / (data['chan_e_max'] - data['chan_e_min']), 'o', label='data', mfc='none')
    plt.plot(data['chan_e_min'], pred_counts_bkg_bkgreg / (data['chan_e_max'] - data['chan_e_min']), label='bkg')
    plt.xlabel('Channel Energy [keV]')
    plt.ylabel('Counts / keV')
    plt.legend()
    plt.savefig('bkg_region_counts.pdf')
    plt.close()

    # compute log Poisson probability
    like_srcreg = logPoissonPDF(pred_counts_srcreg, data['src_region_counts'])
    like_bkgreg = logPoissonPDF(pred_counts_bkg_bkgreg, data['bkg_region_counts'])
    return like_srcreg + like_bkgreg

import scipy.stats
PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)

# define a prior transform
def prior_transform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0] * -10)
    params[1] = 10**(cube[1] * (2 - -2) + -2)
    params[2] = 10**(cube[2] * (-1 - -5) + -5)
    params[3] = PhoIndex_gauss.ppf(cube[3])
    params[4] = cube[4] * (8 - 60) + 8
    params[5] = cube[5] * (0.4) + 0
    params[6] = cube[6] * 90
    params[7] = cube[7] * (400 - 300) + 300
    params[8] = 10**(cube[8] * (1 - -1) + -1)
    return params

print(loglikelihood((norm, NH22, 0.08, PhoIndex, TORsigma, CTKcover, Incl, Ecut, bkg_norm), plot=True))
