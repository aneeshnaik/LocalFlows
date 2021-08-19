#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample particles from qDFs.

Created: August 2021
Author: A. P. Naik
"""
import sys
import numpy as np
from os.path import exists
from emcee import EnsembleSampler as Sampler

sys.path.append("../src")
from constants import kpc
from qdf import create_MW_potential, create_qdf_ensemble


def qiso_lndf(theta, qdfs, weights):
    """
    Evaluate log-DF at phase coordinates theta.

    Parameters
    ----------
    theta : numpy array, shape (5)
        Vector containing R, z, vR, vphi, vz. UNITS: m, m, m/s, m/s, m/s.
    lim : float
        Half-width of R, z region, i.e. particles are allowed with
        |R-R_sun| < lim, likewise z.
    qdfs : list
        List of instances of 'quasiisothermaldf' object from galpy.df.
    weights : 1D iterable, length same as qdfs
        Weights of qdfs. Should sum to 1.

    Returns
    -------
    lnf: float
        Natural log of DF evaluated at theta.

    """
    # unpack theta
    R = theta[0]
    z = theta[1]
    vR = theta[2]
    vphi = theta[3]
    vz = theta[4]

    # only allow stars in 1 < R < 16 kpc, |z| < 2.5 kpc
    if (R > 16 * kpc) or (R < 1 * kpc) or (np.abs(z) > 2.5 * kpc):
        return -np.inf

    # galpy units
    u_R = 8 * kpc
    u_v = 220 * 1000

    # calc DF
    f = 0
    for i in range(len(qdfs)):
        df = qdfs[i]
        w = weights[i]
        f += w * df(R / u_R, vR / u_v, vphi / u_v, z / u_R, vz / u_v)[0]
    if f == 0:
        lnf = -np.inf
    else:
        lnf = np.log(f)
    return lnf


def sample(seed, ddtype, savedir):
    """
    Sample 10^6 particles from qDF, adopting given random seed.

    Parameters
    ----------
    seed : int
        Random seed.
    ddtype : int
        0 for no dark disc, 1 for light dark disc, 2 for heavy.
    savedir : str
        Directory into which dataset is saved.

    Returns
    -------
    None.

    """
    # set up MW potential
    mw = create_MW_potential(ddtype)

    # load MAP parameters
    fname = "../data/MAPs.txt"
    data = np.loadtxt(fname, skiprows=1)
    weights = data[:, 2]
    hr = data[:, 3] / 8
    sr = data[:, 4] / 220
    sz = sr / np.sqrt(3)
    hsr = np.ones_like(hr)
    hsz = np.ones_like(hr)

    # set up qDFs
    qdfs = create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot=mw)

    # set up RNG
    rng = np.random.default_rng(seed)

    # dataset hyperparams
    N = 10000

    # set up sampler
    nwalkers, ndim = 40, 5
    n_burnin = 2000
    n_iter = N
    thin = nwalkers
    s = Sampler(nwalkers, ndim, qiso_lndf, args=[qdfs, weights])

    # set up initial walker positions
    R = rng.uniform(low=1 * kpc, high=16 * kpc, size=nwalkers)
    z = rng.uniform(low=-2.5 * kpc, high=2.5 * kpc, size=nwalkers)
    vR = 50000 * rng.normal(size=nwalkers)
    vphi = 200000 + 50000 * rng.normal(size=nwalkers)
    vz = 50000 * rng.normal(size=nwalkers)
    p0 = np.stack((R, z, vR, vphi, vz), axis=-1)

    # burn in
    print("Burning in...")
    s.run_mcmc(p0, n_burnin, progress=True)

    # take final sample
    p0 = s.chain[:, -1, :]
    s.reset()
    print("Taking final sample...")
    s.run_mcmc(p0, n_iter, progress=True, thin=thin)

    # save
    R = s.flatchain[:, 0]
    z = s.flatchain[:, 1]
    vR = s.flatchain[:, 2]
    vphi = s.flatchain[:, 3]
    vz = s.flatchain[:, 4]
    np.savez(savedir + f"{seed}", R=R, z=z, vR=vR, vphi=vphi, vz=vz,
             lnprob=s.lnprobability)
    return


if __name__ == "__main__":

    # parse arguments
    assert len(sys.argv) == 4
    seed = int(sys.argv[1])
    ddtype = int(sys.argv[2])
    savedir = sys.argv[3]
    if savedir[-1] != '/':
        savedir += '/'
    assert ddtype in [0, 1, 2, 3, 4]
    assert exists(savedir)

    # run sampler
    sample(seed, ddtype, savedir)
