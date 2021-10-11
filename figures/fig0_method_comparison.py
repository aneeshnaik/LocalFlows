#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare accelerations derived in our method versus DF fitting and Jeans.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from os.path import exists
from emcee import EnsembleSampler as Sampler
from scipy.integrate import trapezoid
from scipy.stats import binned_statistic_2d as bin2d
from tqdm import trange

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from constants import pc, G, pi, M_sun, kpc, Myr
from utils import concatenate_data
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


def calc_az_widmark(z, theta):

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


def calc_az_salomon(z, theta):

    nu0 = 10**theta[0]
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
    t5 = (R * z / d) * (sig2z0 * np.exp(-(R - R0) / Rsz) / Rsz - sig2R0 * np.exp(-(R - R0) / RsR) / RsR)
    t6 = - (z / d) * (sig2R - sig2z) * ((R**2 + z**2) / d)

    acc = t1 + t2 + t3 + t4 + t5 + t6
    return acc


def potential_widmark(z, rho1, rho2, rho3, rho4):
    h1 = 40 * pc
    h2 = 100 * pc
    h3 = 300 * pc
    p1 = 4 * pi * G * rho1 * h1**2 * np.log(np.cosh(z / h1))
    p2 = 4 * pi * G * rho2 * h2**2 * np.log(np.cosh(z / h2))
    p3 = 4 * pi * G * rho3 * h3**2 * np.log(np.cosh(z / h3))
    p4 = 2 * pi * G * rho4 * z**2
    p = p1 + p2 + p3 + p4
    return p


def normalise_widmark(theta, z_cut, N_int=100):

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

    pot_int = potential_widmark(z_int, rho1, rho2, rho3, rho4)
    t1 = c1 * np.exp(-pot_int / sig1**2)
    t2 = c2 * np.exp(-pot_int / sig2**2)
    t3 = c3 * np.exp(-pot_int / sig3**2)
    integrand = t1 + t2 + t3
    N = trapezoid(integrand, z_int)
    return N


def lnlike_widmark(theta, z_data, vz_data, z_cut):

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
    pot = potential_widmark(z_data, rho1, rho2, rho3, rho4)

    # get DF
    a = vz_data**2 + 2 * pot
    t1 = c1 * np.exp(-a / (2 * sig1**2)) / np.sqrt(2 * pi * sig1**2)
    t2 = c2 * np.exp(-a / (2 * sig2**2)) / np.sqrt(2 * pi * sig2**2)
    t3 = c3 * np.exp(-a / (2 * sig3**2)) / np.sqrt(2 * pi * sig3**2)
    f = t1 + t2 + t3
    N = normalise_widmark(theta, z_cut=z_cut)
    lnf = np.sum(np.log(f / N))

    return lnf


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


def lnlike_salomon(theta, nu_data, nu_err, sig2z_data, sig2z_err, sig2R_data, sig2R_err, R_data, z_data):

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


def get_widmark_bestpars(datadir):

    # load data
    print("Loading data...")
    z_cut = 2 * kpc
    R_cut = 0.2 * kpc
    data = concatenate_data(datadir, num_files=2000, R_cut=R_cut, z_cut=z_cut)
    z = data[1]
    vz = data[3]

    # set up MCMC sampler
    nwalkers, ndim = 40, 9
    n_burnin = 100
    n_iter = 1000
    thin = 5
    s = Sampler(nwalkers, ndim, lnlike_widmark, args=[z, vz, z_cut])

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


def get_salomon_bestpars(datadir):

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
    s = Sampler(nwalkers, ndim, lnlike_salomon, args=args)

    # set up initial walker positions
    rng = np.random.default_rng(42)
    lnu0 = rng.uniform(low=-54, high=-52, size=nwalkers)
    lhz = rng.uniform(low=np.log10(0.01 * kpc), high=np.log10(10 * kpc), size=nwalkers)
    lhR = rng.uniform(low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers)
    lsig2z0 = rng.uniform(low=7, high=10, size=nwalkers)
    lRsz = rng.uniform(low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers)
    alpha = rng.uniform(low=-1e+9, high=1e+9, size=nwalkers)
    lsig2R0 = rng.uniform(low=7, high=10, size=nwalkers)
    lRsR = rng.uniform(low=np.log10(0.1 * kpc), high=np.log10(100 * kpc), size=nwalkers)
    beta = rng.uniform(low=0, high=5e+9, size=nwalkers)
    p0 = np.stack((lnu0, lhz, lhR, lsig2z0, lRsz, alpha, lsig2R0, lRsR, beta), axis=-1)

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


def calc_az_flows(z_arr, flowdir):

    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # set up spatial arrays
    Nx = np.size(z_arr)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # load flows
    flows = load_flow_ensemble(
        '../nflow_models/' + flowdir,
        inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
    )
    df_args = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen,
        'flows': flows
    }

    # get model accels
    a_model = np.zeros_like(z_arr)
    Nv = 1000
    for j in trange(Nx):
        p = sample_velocities(Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]), v_min=10000)
        q = np.tile(pos[j][None], reps=[Nv, 1])
        gxf, gvf = diff_DF(q, p, df_func=calc_DF_model, df_args=df_args)
        a_model[j] = calc_accel_CBE(q, p, gxf, gvf)[2]

    return a_model

datafile = "fig0_data.npz"
if not exists(datafile):

    # set up z array
    Nx = 60    
    z_arr = np.linspace(0, 1.6 * kpc, Nx)

    # get true accels
    mw = create_MW_potential(darkdisc=False, ddtype=None)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    y_true = calc_MW_az(pos, mw) / (pc / Myr**2)

    # model accels
    th_widmark = get_widmark_bestpars("../data/noDD_up_t0/")
    a_widmark = calc_az_widmark(z_arr, th_widmark)
    y1_widmark = a_widmark / (pc / Myr**2)

    th_widmark = get_widmark_bestpars("../data/noDD_p_t5/")
    a_widmark = calc_az_widmark(z_arr, th_widmark)
    y2_widmark = a_widmark / (pc / Myr**2)

    th_salomon = get_salomon_bestpars("../data/noDD_up_t0/")
    a_salomon = calc_az_salomon(z_arr, th_salomon)
    y1_salomon = a_salomon / (pc / Myr**2)

    th_salomon = get_salomon_bestpars("../data/noDD_p_t5/")
    a_salomon = calc_az_salomon(z_arr, th_salomon)
    y2_salomon = a_salomon / (pc / Myr**2)

    a_flows = calc_az_flows(z_arr, "noDD_up_t0")
    y1_flows = a_flows / (pc / Myr**2)

    a_flows = calc_az_flows(z_arr, "noDD_p_t5")
    y2_flows = a_flows / (pc / Myr**2)

    np.savez(datafile, x=z_arr / kpc, y_true=y_true,
             y1_widmark=y1_widmark, y2_widmark=y2_widmark,
             y1_salomon=y1_salomon, y2_salomon=y2_salomon,
             y1_flows=y1_flows, y2_flows=y2_flows)


# load data
data = np.load(datafile)
x = data['x']
y_true = data['y_true']
y1_widmark = data['y1_widmark']
y2_widmark = data['y2_widmark']
y1_salomon = data['y1_salomon']
y2_salomon = data['y2_salomon']
y1_flows = data['y1_flows']
y2_flows = data['y2_flows']


# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(6.9, 3.2), dpi=150)
left = 0.07
right = 0.98
bottom = 0.12
top = 0.92
dX = (right - left) / 2
dY = (top - bottom)
ax1 = fig.add_axes([left, bottom, dX, dY])
ax2 = fig.add_axes([left + dX, bottom, dX, dY])

# colour
c1 = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]
c2 = plt.cm.Spectral(np.linspace(0, 1, 10))[7][None]
c3 = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# plot
ax1.plot(x, -y_true, label="True", c='k', ls='dashed', zorder=10)
ax1.scatter(x, -y1_flows, s=8, c=c1, label="This work")
ax1.scatter(x, -y1_widmark, s=8, c=c2, label="DF-fitting")
ax1.scatter(x, -y1_salomon, s=8, c=c3, label="Jeans analysis")
ax2.plot(x, -y_true, label="True", c='k', ls='dashed', zorder=10)
ax2.scatter(x, -y2_flows, c=c1, s=8)
ax2.scatter(x, -y2_widmark, c=c2, s=8)
ax2.scatter(x, -y2_salomon, c=c3, s=8)

# labels etc
ylim = ax2.get_ylim()
ax1.set_ylim(ylim)
ax1.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax2.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax1.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
ax1.legend(frameon=False)
ax1.tick_params(left=True, right=True, top=True, direction='inout')
ax2.tick_params(left=True, right=True, top=True, direction='inout')
ax2.tick_params(labelleft=False)
ax1.set_title(r"Unperturbed")
ax2.set_title(r"$500\ \mathrm{Myr}$ post-perturbation")

# save
fig.savefig("fig0_method_comparison.pdf")
