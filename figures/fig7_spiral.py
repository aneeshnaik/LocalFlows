#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig 7: Phase spiral in z-vz DF slice.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import exists
from matplotlib.colors import SymLogNorm

sys.path.append("../src")
from constants import kpc
from ml import load_flow_ensemble, calc_DF_ensemble

# grid extents
zlim = 2.5 * kpc
vlim = 80000

# check if datafile exists, otherwise create and save
datafile = "fig7_data.npz"
if not exists(datafile):
    
    # set up coordinate arrays
    N_px = 256
    ones = np.ones((N_px, N_px))
    zeros = np.zeros((N_px, N_px))
    R0 = 8 * kpc
    vphi0 = 220000.
    z_arr = np.linspace(-zlim, zlim, N_px)
    vz_arr = np.linspace(-vlim, vlim, N_px)
    
    # load flows
    flows = load_flow_ensemble(
        flowdir='../nflow_models/noDD_p_t2', 
        inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64)    
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # z-vz: evaluate DF
    z_grid, vz_grid = np.meshgrid(z_arr, vz_arr, indexing='ij')
    q = np.stack((R0 * ones, zeros, z_grid), axis=-1)
    p = np.stack((zeros, vphi0 * ones, vz_grid), axis=-1)
    q = q.reshape((N_px**2, 3))
    p = p.reshape((N_px**2, 3))
    f = calc_DF_ensemble(q, p, u_q, u_p, q_cen, p_cen, flows)
    f = f.reshape((N_px, N_px))

    np.savez(datafile, f=f)

# load data
f = np.load(datafile)['f']

# symmetrised DF
f_symm = 0.5 * (f + np.flip(np.flip(f, axis=0), axis=1))
r = f / f_symm - 1


# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
norm = SymLogNorm(linthresh=0.002, linscale=0.002, vmin=-0.1, vmax=0.1, base=10)
imargs = {
        'origin': 'lower',
        'aspect': 'auto',
        'cmap': 'Spectral_r',
        'norm': norm
}

# set up figure
asp = 3.3 / 3.45
fig = plt.figure(figsize=(3.3, 3.3/asp), dpi=150)
bottom = 0.105
left = 0.155
right = 0.94
dX = right - left
dY = asp * dX
cdY = 0.035
ax = fig.add_axes([left, bottom, dX, dY])
cax = fig.add_axes([left, bottom+dY, dX, cdY])

# extents
zmin = -zlim / kpc
zmax = zlim / kpc
vzmin = -vlim / 1000
vzmax = vlim / 1000
extent = [zmin, zmax, vzmin, vzmax]

# plot
im = ax.imshow(r.T, **imargs, extent=extent)

# colorbar
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

# labels etc
ax.set_xlabel(r"$z\ [\mathrm{kpc}]$")
ax.set_ylabel(r"$v_z\ [\mathrm{km/s}]$")
ax.tick_params(left=True, right=True, top=True, direction='inout')
cbar.set_label(r'$f/\bar{f}$ - 1')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_ticks([-0.1, -0.01, 0, 0.01, 0.1])

# save
fig.savefig("fig7_spiral.pdf")