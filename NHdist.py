import sys
import numpy as np
import scipy.stats
from scipy.special import logsumexp
from matplotlib import pyplot as plt
import tqdm

filenames = sys.argv[1:]

PhoIndex_gauss = scipy.stats.norm(1.95, 0.15)

PhoIndex_grid = np.arange(1, 3.1, 0.1)
logNH_grid = np.arange(20, 25.1, 0.1)

PhoIndex_logprob = PhoIndex_gauss.logpdf(PhoIndex_grid)

profile_likes = [np.loadtxt(filename) for filename in filenames]

NHmean_grid = np.arange(20, 25, 0.2)
NHstd_grid = np.arange(0.1, 4, 0.1)

# for each NHmean, NHstd
#   for each spectrum
#      for each possible Gamma, NH
#         compute profile likelihood (of background and source)
#         store luminosity
#      compute marginal likelihood integrating over Gamma, NH with distributions
#   compute product of marginal likelihoods
# compute surface of NHmean, NHstds
# no sampling needed!

marglike = np.zeros((len(NHmean_grid), len(NHstd_grid)))
for i, NHmean in enumerate(tqdm.tqdm(NHmean_grid)):
    for j, NHstd in enumerate(NHstd_grid):
        NHlogprob = -0.5 * ((logNH_grid - NHmean) / NHstd)**2
        NHlogprob -= logsumexp(NHlogprob)
        
        # compute marginal likelihood integrating over Gamma, NH with distributions
        for profile_like in profile_likes:
            logprob = profile_like + PhoIndex_logprob.reshape((-1, 1)) + NHlogprob.reshape((1, -11))
            marglike[i,j] += logsumexp(logprob)

extent = [NHstd_grid.min(), NHstd_grid.max(), NHmean_grid.min(), NHmean_grid.max()]        
plt.imshow(-2 * (marglike - marglike.max()), vmin=0, vmax=10,
    extent=extent, origin='lower', cmap='Greys_r', aspect='auto')
plt.ylabel(r'mean($\log N_\mathrm{H}$)')
plt.xlabel(r'std($\log N_\mathrm{H}$)')
plt.colorbar(orientation='horizontal')
plt.contour(-2 * (marglike - marglike.max()), levels=[1, 2, 3],
    extent=extent, origin='lower', colors=['k'] * 3)
plt.savefig(f'distNH_like.pdf')
plt.close()

dchi2 = -2 * (marglike - marglike.max())

for i, NHmean in enumerate(tqdm.tqdm(NHmean_grid)):
    for j, NHstd in enumerate(NHstd_grid):
        if dchi2[i,j] < 3:
            NHprob = np.exp(-0.5 * ((logNH_grid - NHmean) / NHstd)**2)
            NHprob /= NHprob.sum()
            color = 'r' if dchi2[i,j] < 1 else 'orange' if dchi2[i,j] < 2 else 'yellow'
            plt.plot(10**logNH_grid, NHprob, color=color, alpha=0.25)
plt.xscale('log')
plt.xlabel(r'Column density $N_\mathrm{H}$ [#/cm$^2$]')
plt.savefig('distNH_curves.pdf')

