#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoothing length h as function of training dataset size.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import exists
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

sys.path.append("../src")
from constants import pc, kpc, Myr
from ml import load_flow_ensemble, calc_DF_ensemble
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE
from utils import sample_velocities, diff_DF


def calc_dsq(x, y0, y1, sigma):
    dx = np.diff(x)[0]
    n_smooth = sigma / dx
    y_smooth = gaussian_filter1d(y0, n_smooth, mode='nearest')
    inds = np.abs(x) < 0.2
    dsq = np.sum((y_smooth[inds] - y1[inds])**2)
    return dsq


datafile = "fig5alt_data.npz"
if not exists(datafile):

    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # set up spatial arrays
    Nx = 500
    Nv = 1000
    lim = 1.0 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # load MW model
    mw = create_MW_potential(darkdisc=True, ddtype=0)

    # get true accels
    a_true = calc_MW_az(pos, mw)
    y0 = a_true / (pc / Myr**2)

    # loop over Ndata
    y1 = np.zeros((7, Nx))
    for j in range(7):

        # load flows
        N = [100, 500, 900, 1300, 1700, 2100, 2500][j]
        flowdir = f"../nflow_models/DD0L/{N}"
        flows = load_flow_ensemble(
            flowdir=flowdir, inds=np.arange(20),
            n_dim=5, n_layers=8, n_hidden=64
        )

        # get model accels
        a_model = np.zeros_like(a_true)
        df_args = {
            'u_q': u_q, 'u_p': u_p,
            'q_cen': q_cen, 'p_cen': p_cen, 'flows': flows
        }
        for i in tqdm(range(Nx)):
            vel = sample_velocities(
                Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]),
                v_min=10000
            )
            pos_tiled = np.tile(pos[i][None], reps=[Nv, 1])

            gradxf_model, gradvf_model = diff_DF(
                q=pos_tiled, p=vel, df_func=calc_DF_ensemble, df_args=df_args
            )
            a_model[i] = calc_accel_CBE(
                pos_tiled, vel, gradxf_model, gradvf_model
            )[2]
        y1[j] = a_model / (pc / Myr**2)

    np.savez(datafile, x=z_arr / kpc, y0=y0, y1=y1)

# load datafile
data = np.load(datafile)
x = data['x']
y0 = data['y0']
y1 = data['y1']

# loop over Ndata, get smoothing length
h_smooth = np.zeros(7)
for i in range(7):
    h_arr = np.linspace(1, 80, 100) * 1e-3
    dsq = np.array([calc_dsq(x, y0, y1[i], h) for h in h_arr])
    h_smooth[i] = h_arr[dsq.argmin()]*1000

# plot quantities
x = np.array([100, 500, 900, 1300, 1700, 2100, 2500]) // 100
y = h_smooth

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(3.3, 2.7), dpi=150)
bottom = 0.135
top = 0.96
left = 0.135
right = 0.97
dX = right - left
dY = top - bottom
ax = fig.add_axes([left, bottom, dX, dY])

# colours
c = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]

# plot
ax.plot(x, y, c=c, lw=2)

# labels etc
ax.set_xlabel(r"$N_\mathrm{data} / 10^6$")
ax.set_ylabel(r"$h_\mathrm{smooth}\ [\mathrm{pc}]$")
ax.tick_params(left=True, right=True, top=True, direction='inout')
ax.set_ylim(0)
ax.set_xlim(x[0], x[-1])
ax.set_xticks(np.arange(x[0], x[-1] + 2, 2))

# save
fig.savefig('fig5alt_h_Ndata.pdf')
