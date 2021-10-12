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

sys.path.append("../src")
from constants import pc, kpc, pi
from qdf import create_MW_potential, create_qdf_ensemble
from utils import sample_magnitudes


def calc_helio_distance(R, phi, z):
    """Convert Galactocentric R, phim z to distance from Sun."""
    # convert to Galacto cartesian
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    # convert to heliocentric cartesian
    xs = x - 8 * kpc
    ys = np.copy(y)
    zs = z - 0.01 * kpc

    # convert to heliocentric sphericals
    d = np.sqrt(xs**2 + ys**2 + zs**2)
    return d


def get_initial_magnitudes(rng, R, phi, z):
    """Sample 1000 * nwalkers abs mags, convert to G, keep only G < 15."""
    # initial sample
    nwalkers = R.size
    MG_tot = sample_magnitudes(1000 * nwalkers, rng).reshape((nwalkers, 1000))
    d = calc_helio_distance(R, phi, z)
    G = MG_tot + 5 * np.log10(d[:, None] / pc) - 5
    assert ((G < 15).sum(axis=1) > 0).all()

    # downsample
    MG = np.zeros(nwalkers)
    for i in range(nwalkers):
        MG[i] = rng.choice(MG_tot[i][(G[i] < 15)])
    return MG


def qiso_lndf(theta, qdfs, weights):
    """
    Evaluate log-DF at phase coordinates theta.

    Parameters
    ----------
    theta : numpy array, shape (5)
        Vector containing R, z, vR, vphi, vz. UNITS: m, m, m/s, m/s, m/s.
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
    phi = theta[1]
    z = theta[2]
    vR = theta[3]
    vphi = theta[4]
    vz = theta[5]
    MG = theta[6]

    # spatial boundaries
    R0 = 8 * kpc
    if np.abs(R - R0) > 1 * kpc:
        return -np.inf
    if np.abs(phi) > pi / 25:
        return -np.inf

    # apparent mag limit
    d = calc_helio_distance(R, phi, z)
    G = MG + 5 * np.log10(d / pc) - 5
    if G > 15:
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
    f *= np.exp(0.55 * MG)

    if f == 0:
        lnf = -np.inf
    else:
        lnf = np.log(f)
    return lnf


def sample(seed):
    """
    Sample 10^6 particles from qDF, adopting given random seed.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    None.

    """
    # set up MW potential
    mw = create_MW_potential()

    # load MAP parameters
    fname = "MAPs.txt"
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
    nwalkers, ndim = 40, 7
    n_burnin = 1000
    n_iter = N
    thin = nwalkers
    s = Sampler(nwalkers, ndim, qiso_lndf, args=[qdfs, weights])

    # set up initial walker positions
    R = rng.uniform(low=7 * kpc, high=9 * kpc, size=nwalkers)
    phi = rng.uniform(low=-pi / 25, high=pi / 25, size=nwalkers)
    sgns = 2 * rng.integers(0, 2, size=nwalkers) - 1
    az = rng.exponential(scale=0.3 * kpc, size=nwalkers)
    z = sgns * az
    vR = 50000 * rng.normal(size=nwalkers)
    vphi = 220000 + 50000 * rng.normal(size=nwalkers)
    vz = 50000 * rng.normal(size=nwalkers)
    MG = get_initial_magnitudes(rng, R, phi, z)
    p0 = np.stack((R, phi, z, vR, vphi, vz, MG), axis=-1)

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
    phi = s.flatchain[:, 1]
    z = s.flatchain[:, 2]
    vR = s.flatchain[:, 3]
    vphi = s.flatchain[:, 4]
    vz = s.flatchain[:, 5]
    MG = s.flatchain[:, 6]
    np.savez(f"maglim/{seed}", R=R, phi=phi, z=z, vR=vR, vphi=vphi, vz=vz,
             MG=MG, lnprob=s.lnprobability)
    return


if __name__ == "__main__":

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # run sampler
    sample(seed)
