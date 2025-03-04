import numpy as np
import fastxsf
from fastxsf.model import Table
import os
import requests
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose


def download(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        assert response.status_code == 200
        with open(filename, 'wb') as fout:
            fout.write(response.content)

# download file if it does not exist
#download('https://zenodo.org/records/1169181/files/uxclumpy-cutoff.fits?download=1', 'uxclumpy-cutoff.fits')
#download('https://zenodo.org/records/2235505/files/wada-cutoff.fits?download=1', 'wada-cutoff.fits')
#download('https://zenodo.org/records/2235457/files/blob_uniform.fits?download=1', 'blob_uniform.fits')
download('https://zenodo.org/records/2224651/files/wedge.fits?download=1', 'wedge.fits')
download('https://zenodo.org/records/2224472/files/diskreflect.fits?download=1', 'diskreflect.fits')


def test_disk_table():
    energies = np.logspace(-0.5, 2, 400)
    e_lo = energies[:-1]
    e_hi = energies[1:]
    e_mid = (e_lo + e_hi) / 2.0
    deltae = e_hi - e_lo

    fastxsf.x.abundance("angr")
    fastxsf.x.cross_section("vern")
    # compare diskreflect to pexmon
    atable = Table("diskreflect.fits")
    Ecut = 400
    Incl = 70
    PhoIndex = 2.0
    ZHe = 1
    ZFe = 1
    for z in 0.0, 1.0, 2.0, 4.0:
        A = atable(energies, [PhoIndex, Ecut, Incl, z])
        B = fastxsf.x.pexmon(energies=energies, pars=[PhoIndex, Ecut, -1, z, ZHe, ZFe, Incl]) / (1 + z)**2 / 2
        l, = plt.plot(e_mid, A / deltae / (1 + z)**2, label="atable")
        plt.plot(e_mid, B / deltae / (1 + z)**2, label="pexmon", ls=':', color=l.get_color())
        plt.xlabel("Energy [keV]")
        plt.ylabel("Spectrum [photons/cm$^2$/s]")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.savefig("pexmon.pdf")
        #plt.close()
        mask = np.logical_and(energies[:-1] > 8 / (1 + z), energies[:-1] < 80 / (1 + z))
        assert_allclose(A[mask], B[mask], rtol=0.1)

def test_absorber_table():
    fastxsf.x.abundance("angr")
    fastxsf.x.cross_section("bcmc")
    # compare uxclumpy to ztbabs * zpowerlw
    atable = Table("wedge.fits")
    PhoIndex = 1.0
    Incl = 80

    for z in 0, 1, 2:
        plt.figure(figsize=(20, 5))
        print("Redshift:", z)
        for elo, NH22 in (0.2, 0.01), (0.3, 0.1), (0.6, 0.4):
        #for elo, NH22 in (0.3, 0.01),:
            energies = np.geomspace(elo / (1 + z), 10, 400)
            e_lo = energies[:-1]
            e_hi = energies[1:]
            e_mid = (e_lo + e_hi) / 2.0
            deltae = e_hi - e_lo

            A = atable(energies, [NH22, PhoIndex, 45.6, Incl, z])
            B = fastxsf.x.zpowerlw(energies=energies, pars=[PhoIndex, z])
            C = B * fastxsf.x.zphabs(energies=energies, pars=[NH22, z])
            mask = np.logical_and(energies[:-1] > elo / (1 + z), energies[:-1] / (1 + z) < 80)
            #print('A:', np.abs(np.diff(np.log10(C / deltae))))
            #print('B:', np.diff(np.log10(e_mid)))
            #print('C:', np.abs(np.diff(np.log10(C / deltae)) / np.diff(np.log10(e_mid))))
            #for ai in np.where(np.abs(np.diff(np.log10(C / deltae))) > 20)[0]:
            #    mask[np.abs(np.log10(energies[:-1] / energies[ai])) < 0.1] = False
            #for ai in np.where(np.diff(np.log10(A / deltae) / np.log10(deltae)) > 0.03)[0]:
            #    mask[np.abs(np.log10(energies[:-1] / energies[ai])) < 0.1] = False
            #for ai in np.where(np.diff(np.log10(A)) > 0.2)[0]:
            #    mask[ai - 3: ai + 4] = False
            mask[np.abs(energies[:-1] - 6.4 / (1 + z)) < 0.1] = False
            plt.plot(e_mid, A / deltae, label="atable", ls='--', color='k', lw=0.5)
            #plt.plot(e_mid, C / deltae, label="pl*tbabs", ls=':')
            A[~mask] = np.nan
            B[~mask] = np.nan
            C[~mask] = np.nan
            #print(energies[:-1][mask][np.argmax(np.log10(A[mask] / C[mask]))])
            plt.plot(e_mid, A / deltae, label="atable", color='k')
            plt.plot(e_mid, B / deltae, label="pl", ls="--", color='orange', lw=0.5)
            plt.plot(e_mid, C / deltae, label="pl*tbabs", color='r', lw=1)
            plt.xlabel("Energy [keV]")
            plt.ylabel("Spectrum [photons/cm$^2$/s/keV]")
            plt.ylim(0.04, 3)
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.savefig("abspl_z%d.pdf" % z)
            print(energies[np.argmax(np.abs(np.log10(A[mask] / C[mask])))])
            assert_allclose(A[mask], C[mask], rtol=0.2)
        plt.close()
