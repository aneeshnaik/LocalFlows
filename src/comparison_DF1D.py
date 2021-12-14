#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions used for comparison method: 1D DF-fitting

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
from scipy.integrate import trapezoid
from emcee import EnsembleSampler as Sampler

from constants import pc, G, pi, M_sun, kpc
from utils import load_dset


def calc_potential_DF1D(z, rho1, rho2, rho3, rho4):
    h1 = 40 * pc
    h2 = 100 * pc
    h3 = 300 * pc
    p1 = 4 * pi * G * rho1 * h1**2 * np.log(np.cosh(z / h1))
    p2 = 4 * pi * G * rho2 * h2**2 * np.log(np.cosh(z / h2))
    p3 = 4 * pi * G * rho3 * h3**2 * np.log(np.cosh(z / h3))
    p4 = 2 * pi * G * rho4 * z**2
    p = p1 + p2 + p3 + p4
    return p


def calc_az_DF1D(z, theta):

    rho1 = theta[0]
    rho2 = theta[1]
    rho3 = theta[2]
    rho4 = theta[3]

    h1 = 40 * pc
    h2 = 100 * pc
    h3 = 300 * pc
    a1 = -4 * pi * G * rho1 * h1 * np.tanh(z / h1)
    a2 = -4 * pi * G * rho2 * h2 * np.tanh(z / h2)
    a3 = -4 * pi * G * rho3 * h3 * np.tanh(z / h3)
    a4 = -4 * pi * G * rho4 * z
    a = a1 + a2 + a3 + a4
    return a


def normalise_DF1D(theta, z_cut, N_int=100):

    z_int = np.linspace(-z_cut, z_cut, N_int)

    rho1 = theta[0]
    rho2 = theta[1]
    rho3 = theta[2]
    rho4 = theta[3]
    c2 = theta[4]
    c3 = theta[5]
    sig1 = theta[6]
    sig2 = theta[7]
    sig3 = theta[8]
    c1 = 1 - c2 - c3

    pot_int = calc_potential_DF1D(z_int, rho1, rho2, rho3, rho4)
    t1 = c1 * np.exp(-pot_int / sig1**2)
    t2 = c2 * np.exp(-pot_int / sig2**2)
    t3 = c3 * np.exp(-pot_int / sig3**2)
    integrand = t1 + t2 + t3
    N = trapezoid(integrand, z_int)

    return N


def calc_lnlike_DF1D(theta, z_data, vz_data, z_cut):

    # unpack theta
    rho1 = theta[0]
    rho2 = theta[1]
    rho3 = theta[2]
    rho4 = theta[3]
    c2 = theta[4]
    c3 = theta[5]
    sig1 = theta[6]
    sig2 = theta[7]
    sig3 = theta[8]
    c1 = 1 - c2 - c3

    # bounds
    for rho in [rho1, rho2, rho3, rho4]:
        if rho < 0:
            return -1e+20
        if rho > 0.2 * M_sun / pc**3:
            return -1e+20
    for c in [c1, c2, c3]:
        if c < 0:
            return -1e+20
        if c > 1:
            return -1e+20
    for sig in [sig1, sig2, sig3]:
        if sig < 0:
            return -1e+20
        if sig > 200000:
            return -1e+20

    # potential
    pot = calc_potential_DF1D(z_data, rho1, rho2, rho3, rho4)

    # get DF
    a = vz_data**2 + 2 * pot
    t1 = c1 * np.exp(-a / (2 * sig1**2)) / np.sqrt(2 * pi * sig1**2)
    t2 = c2 * np.exp(-a / (2 * sig2**2)) / np.sqrt(2 * pi * sig2**2)
    t3 = c3 * np.exp(-a / (2 * sig3**2)) / np.sqrt(2 * pi * sig3**2)
    f = t1 + t2 + t3
    N = normalise_DF1D(theta, z_cut=z_cut)
    lnf = np.sum(np.log(f / N))

    return lnf


def get_bestpars_DF1D(dfile):

    # load data
    print("Loading data...")
    z_cut = 2 * kpc
    R_cut = 0.2 * kpc
    data = load_dset(dfile, R_cut=R_cut, z_cut=z_cut)
    z = data[1]
    vz = data[3]

    # set up MCMC sampler
    nwalkers, ndim = 40, 9
    n_burnin = 100
    n_iter = 1000
    thin = 5
    s = Sampler(nwalkers, ndim, calc_lnlike_DF1D, args=[z, vz, z_cut])

    # set up initial walker positions
    rng = np.random.default_rng(42)
    rho1 = rng.uniform(low=0, high=0.2 * M_sun / pc**3, size=nwalkers)
    rho2 = rng.uniform(low=0, high=0.2 * M_sun / pc**3, size=nwalkers)
    rho3 = rng.uniform(low=0, high=0.2 * M_sun / pc**3, size=nwalkers)
    rho4 = rng.uniform(low=0, high=0.2 * M_sun / pc**3, size=nwalkers)
    c2 = rng.uniform(low=0, high=1, size=nwalkers)
    c3 = rng.uniform(low=0, high=1, size=nwalkers)
    sig1 = rng.uniform(low=0, high=150000, size=nwalkers)
    sig2 = rng.uniform(low=0, high=150000, size=nwalkers)
    sig3 = rng.uniform(low=0, high=150000, size=nwalkers)

    # make sure c1 positive for all walkers
    c1 = 1 - c2 - c3
    for i in range(nwalkers):
        if c1[i] < 0:
            c1_neg = True
        else:
            c1_neg = False
        while c1_neg:
            c2[i] = rng.uniform(low=0, high=1)
            c3[i] = rng.uniform(low=0, high=1)
            c1 = 1 - c2 - c3
            if c1[i] > 0:
                c1_neg = False

    # stack walkers
    p0 = np.stack((rho1, rho2, rho3, rho4, c2, c3, sig1, sig2, sig3), axis=-1)

    # burn in
    s.run_mcmc(p0, n_burnin, progress=True)

    # main MCMC run
    p0 = s.chain[:, -1, :]
    s.reset()
    s.run_mcmc(p0, n_iter, progress=True, thin=thin)

    # get max prob parameters
    i = np.unravel_index(s.lnprobability.argmax(), s.lnprobability.shape)
    theta = s.chain[i]

    return theta
