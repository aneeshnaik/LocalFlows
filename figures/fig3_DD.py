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


datafile = "fig3_data.npz"
if not exists(datafile):

    # flow args
    u_q = (kpc / 5)
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # set up spatial arrays
    Nx = 50
    Nv = 1000
    z0 = 0.01 * kpc
    lim = 0.2 * kpc
    z_arr = np.linspace(z0 - lim, z0 + lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get accels w/o DD
    mw_def = create_MW_potential(ddtype=0)
    a_default = calc_MW_az(pos, mw_def)

    # loop over models
    models = []
    trues = []
    for i in range(3):

        # load flows
        flowdir = ['2_400', '3_400', '4_400'][i]
        flowdir = "../nflow_models/" + flowdir
        flows = load_flow_ensemble(
            flowdir=flowdir, inds=np.arange(20),
            n_dim=5, n_layers=8, n_hidden=64
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
        ddtype = [2, 3, 4][i]
        mw = create_MW_potential(ddtype=ddtype)
        qdfs = create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot=mw)

        # get true accels
        a_true = calc_MW_az(pos, mw)
        trues.append(a_true)

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
                q=pos_tiled, p=vel, df_func=calc_DF_model, df_args=df_args
            )
            a_model[i] = calc_accel_CBE(
                pos_tiled, vel, gradxf_model, gradvf_model
            )[2]
        models.append(a_model)

    np.savez(
        datafile, x=z_arr / kpc, y0=a_default / (pc / Myr**2),
        y0_0=trues[0] / (pc / Myr**2), y1_0=models[0] / (pc / Myr**2),
        y0_1=trues[1] / (pc / Myr**2), y1_1=models[1] / (pc / Myr**2),
        y0_2=trues[2] / (pc / Myr**2), y1_2=models[2] / (pc / Myr**2)
    )


# load datafile
data = np.load(datafile)
x = data['x']
y0 = data['y0']
y0_0 = data['y0_0']
y1_0 = data['y1_0']
y0_1 = data['y0_1']
y1_1 = data['y1_1']
y0_2 = data['y0_2']
y1_2 = data['y1_2']

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(6.9, 3), dpi=150)
left = 0.08
bottom = 0.12
top = 0.93
dX = (0.98 - left) / 3
dY = top - bottom
ax0 = fig.add_axes([left, bottom, dX, dY])
ax1 = fig.add_axes([left + dX, bottom, dX, dY])
ax2 = fig.add_axes([left + 2 * dX, bottom, dX, dY])

# colour
c = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]

# plot
for i in range(3):
    ax = [ax0, ax1, ax2][i]
    y_true = [y0_0, y0_1, y0_2][i]
    y_model = [y1_0, y1_1, y1_2][i]
    ax.plot(x, y_true, c='k', ls='dashed', zorder=0, label="Exact")
    ax.plot(x, y0, c='k', ls='dotted', zorder=0, label="No DD")
    ax.scatter(x, y_model, c=c, s=8, zorder=1, label="Reconstruction")

# ticks, labels limits etc
titles = [
    r'$h_\mathrm{DD}=10\,\mathrm{pc}$',
    r'$25\,\mathrm{pc}$',
    r'$50\,\mathrm{pc}$'
]
for i in range(3):
    ax = [ax0, ax1, ax2][i]
    ax.tick_params(right=True, top=True, direction='inout')
    ax.set_ylim(-1.275, 1.275)
    ax.set_xlim(-0.23, 0.23)
    ax.set_title(titles[i])
ax0.set_ylabel(r'$a\ [\mathrm{kpc/Gyr}^2]$')
ax1.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax0.legend(frameon=False)

# save
fig.savefig('fig3_DD.pdf')
