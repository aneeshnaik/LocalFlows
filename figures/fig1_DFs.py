#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Comparison of true and modelled local DFs.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from qdf import create_qdf_ensemble, create_MW_potential
from qdf import calc_DF_ensemble as calc_DF_true
from constants import kpc
from scipy.integrate import trapezoid as trapz


def normalise_DF(f, x1, x2):
    """
    Return normalisation of 2D PDF in x1-x2 space, defined by 1D arrays x12.
    """
    N = np.size(x1)
    norm = trapz(np.array([trapz(f[:, i], x1) for i in range(N)]), x2)
    return norm


# set up coordinate arrays
N_px = 128
ones = np.ones((N_px, N_px))
zeros = np.zeros((N_px, N_px))
R0 = 8 * kpc
z0 = 0.
vR0 = 0.
vphi0 = 220000.
vz0 = 0.

Rlim = 1.1 * kpc
zlim = 2.5 * kpc
vlim = 80000
R_arr = np.linspace(R0 - Rlim, R0 + Rlim, N_px)
z_arr = np.linspace(-zlim, zlim, N_px)
vR_arr = np.linspace(vR0 - vlim, vR0 + vlim, N_px)
vphi_arr = np.linspace(vphi0 - vlim, vphi0 + vlim, N_px)
vz_arr = np.linspace(vz0 - vlim, vz0 + vlim, N_px)

dfile = "fig1_data.npz"
if not exists(dfile):

    # load flow ensemble
    flows = load_flow_ensemble(
        flowdir='../flows/fiducial',
        inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64)

    # load qDFs
    fname = "../data/MAPs.txt"
    data = np.loadtxt(fname, skiprows=1)
    weights = data[:, 2]
    hr = data[:, 3] / 8
    sr = data[:, 4] / 220
    sz = sr / np.sqrt(3)
    hsr = np.ones_like(hr)
    hsz = np.ones_like(hr)
    mw = create_MW_potential()
    qdfs = create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot=mw)

    # flow arguments
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # R-z: evaluate DF
    R_grid, z_grid = np.meshgrid(R_arr, z_arr, indexing='ij')
    q = np.stack((R_grid, zeros, z_grid), axis=-1)
    p = np.stack((vR0 * ones, vphi0 * ones, vz0 * ones), axis=-1)
    q = q.reshape((N_px**2, 3))
    p = p.reshape((N_px**2, 3))
    f_model = calc_DF_model(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_model = f_model.reshape((N_px, N_px))
    f_true = calc_DF_true(q, p, qdfs, weights)
    f_true = f_true.reshape((N_px, N_px))
    f_true[np.abs(R_grid - R0) > 1 * kpc] = 0

    # normalise
    norm_true = normalise_DF(f_true, R_arr, z_arr)
    norm_model = normalise_DF(f_model, R_arr, z_arr)
    f_true /= norm_true
    f_model /= norm_model

    # ref value
    f_ref = calc_DF_true(q_cen, p_cen, qdfs, weights) / norm_true
    f1_model = f_model / f_ref
    f1_true = f_true / f_ref

    # calculate residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        res1 = np.divide((f1_model - f1_true), f1_true)

    # vR-vphi: evaluate DF
    vR_grid, vphi_grid = np.meshgrid(vR_arr, vphi_arr, indexing='ij')
    q = np.stack((R0 * ones, zeros, z0 * ones), axis=-1)
    p = np.stack((vR_grid, vphi_grid, vz0 * ones), axis=-1)
    q = q.reshape((N_px**2, 3))
    p = p.reshape((N_px**2, 3))
    f_model = calc_DF_model(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_model = f_model.reshape((N_px, N_px))
    f_true = calc_DF_true(q, p, qdfs, weights)
    f_true = f_true.reshape((N_px, N_px))

    # normalise
    norm_true = normalise_DF(f_true, vR_arr, vphi_arr)
    norm_model = normalise_DF(f_model, vR_arr, vphi_arr)
    f_true /= norm_true
    f_model /= norm_model

    # ref value
    f_ref = calc_DF_true(q_cen, p_cen, qdfs, weights) / norm_true
    f2_model = f_model / f_ref
    f2_true = f_true / f_ref

    # calculate residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        res2 = np.divide((f2_model - f2_true), f2_true)

    # z-vz: evaluate DF
    z_grid, vz_grid = np.meshgrid(z_arr, vz_arr, indexing='ij')
    q = np.stack((R0 * ones, zeros, z_grid), axis=-1)
    p = np.stack((vR0 * ones, vphi0 * ones, vz_grid), axis=-1)
    q = q.reshape((N_px**2, 3))
    p = p.reshape((N_px**2, 3))
    f_model = calc_DF_model(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_model = f_model.reshape((N_px, N_px))
    f_true = calc_DF_true(q, p, qdfs, weights)
    f_true = f_true.reshape((N_px, N_px))

    # normalise
    norm_true = normalise_DF(f_true, z_arr, vz_arr)
    norm_model = normalise_DF(f_model, z_arr, vz_arr)
    f_true /= norm_true
    f_model /= norm_model

    # ref value
    f_ref = calc_DF_true(q_cen, p_cen, qdfs, weights) / norm_true
    f3_model = f_model / f_ref
    f3_true = f_true / f_ref

    # calculate residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        res3 = np.divide((f3_model - f3_true), f3_true)

    np.savez(dfile, f1_true=f1_true, f1_model=f1_model, res1=res1,
             f2_true=f2_true, f2_model=f2_model, res2=res2,
             f3_true=f3_true, f3_model=f3_model, res3=res3)

else:
    data = np.load(dfile)
    f1_true = data['f1_true']
    f1_model = data['f1_model']
    res1 = data['res1']
    f2_true = data['f2_true']
    f2_model = data['f2_model']
    res2 = data['res2']
    f3_true = data['f3_true']
    f3_model = data['f3_model']
    res3 = data['res3']


# set up figure
asp = 6.9 / 8.4
fig = plt.figure(figsize=(6.9, 6.9 / asp), dpi=150)
left = 0.085
right = 0.985
bottom = 0.125
top = 0.97
xgap = 0.0
ygap = 0.05
dX = (right - left - xgap) / 3
dY = asp * dX
CdY = 0.03

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
imargs1 = {
    'origin': 'lower',
    'aspect': 'auto',
    'cmap': 'bone',
    'vmin': 0, 'vmax': 1.05
}
imargs2 = {
    'origin': 'lower',
    'aspect': 'auto',
    'cmap': 'Spectral_r',
    'vmin': -0.09, 'vmax': 0.09
}

# extents
Rmin = (R0 - Rlim) / kpc
Rmax = (R0 + Rlim) / kpc
zmin = -zlim / kpc
zmax = zlim / kpc
vRmin = (vR0 - vlim) / 1000
vRmax = (vR0 + vlim) / 1000
vphimin = (vphi0 - vlim) / 1000
vphimax = (vphi0 + vlim) / 1000
vzmin = (vz0 - vlim) / 1000
vzmax = (vz0 + vlim) / 1000
extent1 = [Rmin, Rmax, zmin, zmax]
extent2 = [vRmin, vRmax, vphimin, vphimax]
extent3 = [zmin, zmax, vzmin, vzmax]

# loop over rows
for i in range(3):

    Y = top - dY - i * (dY + ygap)
    ax1 = fig.add_axes([left, Y, dX, dY])
    ax2 = fig.add_axes([left + dX, Y, dX, dY])
    ax3 = fig.add_axes([left + 2 * dX + xgap, Y, dX, dY])

    f_true = [f1_true, f2_true, f3_true][i]
    f_model = [f1_model, f2_model, f3_model][i]
    res = [res1, res2, res3][i]
    extent = [extent1, extent2, extent3][i]
    xlabel = [
        r'$R\ [\mathrm{kpc}]$',
        r'$v_R\ [\mathrm{km/s}]$',
        r'$z\ [\mathrm{kpc}]$'
    ][i]
    ylabel = [
        r'$z\ [\mathrm{kpc}]$',
        r'$v_\varphi\ [\mathrm{km/s}]$',
        r'$v_z\ [\mathrm{km/s}]$'
    ][i]

    im1 = ax1.imshow(f_true.T, **imargs1, extent=extent)
    im2 = ax2.imshow(f_model.T, **imargs1, extent=extent)
    im3 = ax3.imshow(res.T, **imargs2, extent=extent)

    # ticks, labels etc.
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(direction='inout', right=True, top=True)
    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if i == 0:
        ax1.set_title('Exact')
        ax2.set_title('Model')
        ax3.set_title('Residuals')

# colourbar
CY = top - CdY - 3 * (dY + ygap)
cax1 = fig.add_axes([left, CY, 2 * dX, CdY])
cax2 = fig.add_axes([left + 2 * dX + xgap, CY, dX, CdY])
cbar1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
cbar2 = plt.colorbar(im3, cax=cax2, orientation='horizontal')
cbar1.set_label(r'$f/f_\mathrm{ref}$')
cbar2.set_label(r'$f_\mathrm{model}/f_\mathrm{exact} - 1$')

# save
fig.savefig('fig1_DFs.pdf')
