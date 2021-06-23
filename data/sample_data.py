#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample particles from qDFs.

Created: June 2021
Author: A. P. Naik
"""
import sys
import numpy as np
from emcee import EnsembleSampler as Sampler

from galpy.potential import PowerSphericalPotentialwCutoff as BulgePhi
from galpy.potential import MiyamotoNagaiPotential as DiscPhi
from galpy.potential import NFWPotential as HaloPhi
from galpy.actionAngle import actionAngleStaeckel
from galpy.df import quasiisothermaldf as qdf

sys.path.append("../src")
from constants import kpc


def qiso_lndf(theta, lim, qdfs, weights):
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

    # only allow heliocentric cube, half-length 100pc
    R0 = 8 * kpc
    z0 = 0.01 * kpc
    if np.abs(R - R0) > lim:
        return -np.inf
    elif np.abs(z - z0) > lim:
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


def sample(ddtype, lim):
    """
    Sample 10^6 particles from qDF, adopting given random seed.

    Parameters
    ----------
    ddtype : int
        0 for no dark disc, 1 for light dark disc, 2 for heavy.
    lim_pc : float
        Half-width of R, z region, i.e. particles are allowed with
        |R-R_sun| < lim, likewise z. UNITS: parsecs.

    Returns
    -------
    None.

    """
    # convert lim to metres
    lim = (lim_pc / 1000) * kpc

    # set up MW potential
    if ddtype == 0:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        halo = HaloPhi(a=16 / 8., normalize=0.35)
        mw = bulge + disc + halo
    elif ddtype == 1:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.02 / 8., normalize=0.03)
        halo = HaloPhi(a=16 / 8., normalize=0.32)
        mw = bulge + disc1 + disc2 + halo
    elif ddtype == 2:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.02 / 8., normalize=0.06)
        halo = HaloPhi(a=16 / 8., normalize=0.29)
        mw = bulge + disc1 + disc2 + halo

    # load MAP parameters
    fname = "../data/MAPs.txt"
    data = np.loadtxt(fname, skiprows=1)

    # set up qDFs
    aA = actionAngleStaeckel(pot=mw, delta=0.45, c=True)
    qdfs = []
    weights = []
    for i in range(6):
        weights.append(data[i, 2])
        hr = data[i, 3] / 8
        sr = data[i, 4] / 220
        sz = sr / np.sqrt(3)
        df = qdf(hr, sr, sz, 1., 1., pot=mw, aA=aA, cutcounter=True)
        qdfs.append(df)

    # set up RNG
    rng = np.random.default_rng(42)

    # dataset hyperparams
    N = 1000000

    # set up sampler
    nwalkers, ndim = 20, 5
    n_burnin = 1000
    n_iter = N
    thin = nwalkers
    s = Sampler(nwalkers, ndim, qiso_lndf, args=[lim, qdfs, weights])

    # set up initial walker positions
    R = 8 * kpc + rng.uniform(low=-lim, high=lim, size=nwalkers)
    z = 0.01 * kpc + rng.uniform(low=-lim, high=lim, size=nwalkers)
    vR = 50000 * rng.normal(size=nwalkers)
    vphi = 220000 + 50000 * rng.normal(size=nwalkers)
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
    np.savez(f"{ddtype}_{lim_pc:.0f}", R=R, z=z, vR=vR, vphi=vphi, vz=vz,
             lnprob=s.lnprobability)
    return


if __name__ == "__main__":

    # parse arguments
    assert len(sys.argv) == 3
    ddtype = int(sys.argv[1])
    lim_pc = float(sys.argv[2])
    assert ddtype in [0, 1, 2]
    assert lim_pc in [100, 400, 1600]

    # run sampler
    sample(ddtype, lim_pc)
