#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from constants import kpc, pc, Myr
from utils import sample_velocities, diff_DF
from qdf import create_qdf_ensemble, create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


# check if datafile exists, otherwise create and save
datafile = "fig2_data.npz"
if not exists(datafile):

    # load flows
    flowdir = "../nflow_models/0_400"
    flows = load_flow_ensemble(
        flowdir=flowdir, inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
    )

    # load qDFs
    fname = "../data/MAPs.txt"
    data = np.loadtxt(fname, skiprows=1)
    weights = data[:, 2]
    hr = data[:, 3] / 8
    sr = data[:, 4] / 220
    sz = sr / np.sqrt(3)
    hsr = np.ones_like(hr)
    hsz = np.ones_like(hr)
    mw = create_MW_potential(ddtype=0)
    qdfs = create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot=mw)

    # flow args
    u_q = (kpc / 5)
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # set up spatial arrays
    Nx = 50
    Nv = 1000
    z0 = 0.01 * kpc
    lim = 0.4 * kpc
    z_arr = np.linspace(z0 - lim, z0 + lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get true accels
    a_true = calc_MW_az(pos, mw)

    # get model accels
    a_model = np.zeros_like(a_true)
    df_args = {
        'u_q': u_q, 'u_p': u_p, 'q_cen': q_cen, 'p_cen': p_cen, 'flows': flows
    }
    for i in tqdm(range(Nx)):
        vel = sample_velocities(
            Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]), v_min=10000
        )
        pos_tiled = np.tile(pos[i][None], reps=[Nv, 1])

        gradxf_model, gradvf_model = diff_DF(
            q=pos_tiled, p=vel, df_func=calc_DF_model, df_args=df_args
        )
        a_model[i] = calc_accel_CBE(
            pos_tiled, vel, gradxf_model, gradvf_model
        )[2]

    # save
    x = z_arr / kpc
    y0 = a_true / (pc / Myr**2)
    y1 = a_model / (pc / Myr**2)
    np.savez(datafile, x=x, y0=y0, y1=y1)

else:
    data = np.load(datafile)
    x = data['x']
    y0 = data['y0']
    y1 = data['y1']


# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(3.3, 3), dpi=150)
left = 0.16
bottom = 0.125
dX = 0.97 - left
dY = 0.97 - bottom
ax = fig.add_axes([left, bottom, dX, dY])

# colour
c = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]

# plot
plt.plot(x, y0, c='k', ls='dashed', zorder=0, label="Exact")
plt.scatter(x, y1, c=c, s=8, zorder=1, label="Reconstruction")

# labels etc
plt.ylim(1.1 * y0.min(), 1.1 * y0.max())
plt.xlabel(r'$z\ [\mathrm{kpc}]$')
plt.ylabel(r'$a\ [\mathrm{kpc/Gyr}^2]$')
ax.tick_params(right=True, top=True, direction='inout')
ax.legend(frameon=False)

# save
fig.savefig('fig2_az.pdf')
