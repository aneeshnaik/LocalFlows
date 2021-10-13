#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions used for comparison method: Jeans analysis

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
from scipy.stats import binned_statistic_2d as bin2d
from emcee import EnsembleSampler as Sampler

from constants import kpc, pi
from utils import concatenate_data


def calc_nu_model(R, z, nu0, hz, hR):
    R0 = 8 * kpc
    nu = nu0 * (1 / np.cosh(z / hz)**2) * np.exp(-(R - R0) / hR)
    return nu


def calc_sig2z_model(R, z, sig2z0, Rsz, alpha):
    R0 = 8 * kpc
    sig2 = sig2z0 * np.exp(-(R - R0) / Rsz) + alpha * (z / kpc)
    return sig2


def calc_sig2R_model(R, z, sig2R0, RsR, beta):
    R0 = 8 * kpc
    sig2 = sig2R0 * np.exp(-(R - R0) / RsR) + beta * (z / kpc)
    return sig2


def calc_lnlike_jeans(
        theta,
        nu_data, nu_err,
        sig2z_data, sig2z_err,
        sig2R_data, sig2R_err,
        R_data, z_data
):

    # unpack theta
    nu0 = 10**theta[0]
    hz = 10**theta[1]
    hR = 10**theta[2]
    sig2z0 = 10**theta[3]
    Rsz = 10**theta[4]
    alpha = theta[5]
    sig2R0 = 10**theta[6]
    RsR = 10**theta[7]
    beta = theta[8]

    # bounds
    if nu0 < 1e-54:
        return -1e+20
    elif nu0 > 1e-52:
        return -1e+20
    if hz < 0.01 * kpc:
        return -1e+20
    elif hz > 10 * kpc:
        return -1e+20
    if hR < 0.1 * kpc:
        return -1e+20
    elif hR > 100 * kpc:
        return -1e+20
    if sig2z0 < 1e+7:
        return -1e+20
    elif sig2z0 > 1e+10:
        return -1e+20
    if Rsz < 0.1 * kpc:
        return -1e+20
    elif Rsz > 100 * kpc:
        return -1e+20
    if alpha < -1e+9:
        return -1e+20
    elif alpha > 1e+9:
        return -1e+20
    if sig2R0 < 1e+7:
        return -1e+20
    elif sig2R0 > 1e+10:
        return -1e+20
    if RsR < 0.1 * kpc:
        return -1e+20
    elif RsR > 100 * kpc:
        return -1e+20
    if beta < 0:
        return -1e+20
    elif beta > 5e+9:
        return -1e+20

    # calculate models
    nu_model = calc_nu_model(R_data, z_data, nu0, hz, hR)
    sig2z_model = calc_sig2z_model(R_data, z_data, sig2z0, Rsz, alpha)
    sig2R_model = calc_sig2R_model(R_data, z_data, sig2R0, RsR, beta)

    # calculate L
    chi2_nu = np.sum((nu_data - nu_model)**2 / nu_err**2)
    chi2_sz = np.sum((sig2z_data - sig2z_model)**2 / sig2z_err**2)
    chi2_sR = np.sum((sig2R_data - sig2R_model)**2 / sig2R_err**2)
    lnL = -0.5 * (chi2_nu + chi2_sz + chi2_sR)
    return lnL


def calc_az_jeans(z, theta):

    hz = 10**theta[1]
    hR = 10**theta[2]
    sig2z0 = 10**theta[3]
    Rsz = 10**theta[4]
    alpha = theta[5]
    sig2R0 = 10**theta[6]
    RsR = 10**theta[7]
    beta = theta[8]

    R0 = 8 * kpc
    R = R0 * np.ones_like(z)
    sig2z = calc_sig2z_model(R, z, sig2z0, Rsz, alpha)
    sig2R = calc_sig2R_model(R, z, sig2R0, RsR, beta)
    sig2Rz = R * z * (sig2R - sig2z) / (R**2 - z**2)

    t1 = -2 * sig2z * np.tanh(z / hz) / hz
    t2 = alpha / kpc
    t3 = sig2Rz / R
    t4 = -sig2Rz / hR
    d = R**2 - z**2
    eRz = np.exp(-(R - R0) / Rsz) / Rsz
    eRR = np.exp(-(R - R0) / RsR) / RsR
    t5 = (R * z / d) * (sig2z0 * eRz - sig2R0 * eRR)
    t6 = - (z / d) * (sig2R - sig2z) * ((R**2 + z**2) / d)

    acc = t1 + t2 + t3 + t4 + t5 + t6
    return acc


def get_bestpars_jeans(datadir):

    # load data
    print("Loading data")
    z_cut = 2.5 * kpc
    R_cut = 0.9 * kpc
    data = concatenate_data(datadir, num_files=2000, R_cut=R_cut, z_cut=z_cut)
    R = data[0]
    z = data[1]
    vR = data[2]
    vz = data[3]

    # flip to north side
    vz[z < 0] *= -1
    z = np.abs(z)

    # construct R, z, bins
    bin_count = 400
    N_zbins = z.size // bin_count
    z_edges = np.array([0])
    z_sorted = np.sort(z)
    for i in range(N_zbins):
        z1 = z_sorted[(i + 1) * bin_count - 1]
        z2 = z_sorted[(i + 1) * bin_count]
        z_edges = np.append(z_edges, 0.5 * (z1 + z2))
    R_edges = np.array([7.1 * kpc, 7.7 * kpc, 8.3 * kpc, 8.9 * kpc])
    z_cen = 0.5 * (z_edges[1:] + z_edges[:-1])
    R_cen = 0.5 * (R_edges[1:] + R_edges[:-1])
    R_grid, z_grid = np.meshgrid(R_cen, z_cen, indexing='ij')

    # density
    bins = [R_edges, z_edges]
    N_grid = bin2d(R, z, np.ones_like(R), bins=bins, statistic='count')[0]
    vol_grid = np.diff(R_edges**2)[:, None] * np.diff(z_edges)[None] * pi / 25
    nu_grid = N_grid / vol_grid
    nu_err_grid = nu_grid / np.sqrt(N_grid)

    # velocity dispersions
    sig2z_grid = bin2d(R, z, vz, bins=bins, statistic='std')[0]**2
    sig2R_grid = bin2d(R, z, vR, bins=bins, statistic='std')[0]**2
    sig2z_err_grid = sig2z_grid / np.sqrt(N_grid)
    sig2R_err_grid = sig2R_grid / np.sqrt(N_grid)

    # flatten into data arrays
    R_data = R_grid.flatten()
    z_data = z_grid.flatten()
    nu_data = nu_grid.flatten()
    nu_err = nu_err_grid.flatten()
    sig2z_data = sig2z_grid.flatten()
    sig2z_err = sig2z_err_grid.flatten()
    sig2R_data = sig2R_grid.flatten()
    sig2R_err = sig2R_err_grid.flatten()

    # set up MCMC
    nwalkers, ndim = 40, 9
    n_burnin = 10000
    n_iter = 2000
    thin = 5
    args = [
        nu_data, nu_err,
        sig2z_data, sig2z_err, sig2R_data,
        sig2R_err, R_data, z_data
    ]
    s = Sampler(nwalkers, ndim, calc_lnlike_jeans, args=args)

    # set up initial walker positions
    rng = np.random.default_rng(42)
    lnu0 = rng.uniform(
        low=-54, high=-52, size=nwalkers
    )
    lhz = rng.uniform(
        low=np.log10(0.01 * kpc), high=np.log10(10 * kpc), size=nwalkers
    )
    lhR = rng.uniform(
        low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers
    )
    lsig2z0 = rng.uniform(
        low=7, high=10, size=nwalkers
    )
    lRsz = rng.uniform(
        low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers
    )
    alpha = rng.uniform(
        low=-1e+9, high=1e+9, size=nwalkers
    )
    lsig2R0 = rng.uniform(
        low=7, high=10, size=nwalkers
    )
    lRsR = rng.uniform(
        low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers
    )
    beta = rng.uniform(
        low=0, high=5e+9, size=nwalkers
    )
    p0 = np.stack(
        (lnu0, lhz, lhR, lsig2z0, lRsz, alpha, lsig2R0, lRsR, beta), axis=-1
    )

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
