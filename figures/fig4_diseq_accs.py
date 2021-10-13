#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6: Measured acceleration in perturbed datasets.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from constants import kpc, pc, Myr
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


# check if datafile exists, otherwise create and save
datafile = "fig4_data.npz"
if not exists(datafile):

    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # load MW model
    mw = create_MW_potential()

    # set up spatial arrays
    Nx = 60
    lim = 1.75 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get true accels
    y_true = calc_MW_az(pos, mw) / (pc / Myr**2)

    # loop over models
    flowdirs = ['perturbed_t0', 'perturbed_t2', 'perturbed_t5']
    y_model = np.zeros((3, Nx))
    for i in range(3):

        # load flows
        flows = load_flow_ensemble(
            '../flows/' + flowdirs[i],
            inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
        )
        df_args = {
            'u_q': u_q, 'u_p': u_p,
            'q_cen': q_cen, 'p_cen': p_cen,
            'flows': flows
        }

        # get model accels
        Nv = 1000
        v_args = {'Nv': Nv, 'v_max': 50000, 'v_min': 10000, 'v_mean': p_cen}
        for j in trange(Nx):
            p = sample_velocities(**v_args)
            q = np.tile(pos[j][None], reps=[Nv, 1])
            gxf, gvf = diff_DF(q, p, df_func=calc_DF_model, df_args=df_args)
            y_model[i, j] = calc_accel_CBE(q, p, gxf, gvf)[2] / (pc / Myr**2)

    # save
    x = z_arr / kpc
    np.savez(datafile, x=x, y_true=y_true, y_model=y_model)


# load data
data = np.load(datafile)
x = data['x']
y_true = data['y_true']
y_model = data['y_model']

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(6.9, 3.5), dpi=150)
left = 0.075
right = 0.99
top = 0.935
bottom = 0.115
rfrac = 1 / 4
dX = (right - left) / 3
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
ax0 = fig.add_axes([left, bottom + rdY, dX, dY])
ax1 = fig.add_axes([left + dX, bottom + rdY, dX, dY])
ax2 = fig.add_axes([left + 2 * dX, bottom + rdY, dX, dY])
ax0r = fig.add_axes([left, bottom, dX, rdY])
ax1r = fig.add_axes([left + dX, bottom, dX, rdY])
ax2r = fig.add_axes([left + 2 * dX, bottom, dX, rdY])

# colour
c_p = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# main plots
ax0.plot(x, y_true, c='k', ls='dashed', zorder=0)
ax1.plot(x, y_true, c='k', ls='dashed', zorder=0)
ax2.plot(x, y_true, c='k', ls='dashed', zorder=0, label="Exact")
ax0.scatter(x, y_model[0], c=c_p, s=8, zorder=1)
ax1.scatter(x, y_model[1], c=c_p, s=8, zorder=1)
ax2.scatter(x, y_model[2], c=c_p, s=8, zorder=1, label="Model")

# residuals
r = y_model / y_true - 1
ax0r.plot(x, r[0], lw=2, c=c_p)
ax1r.plot(x, r[1], lw=2, c=c_p)
ax2r.plot(x, r[2], lw=2, c=c_p)
ax0r.plot(x, np.zeros_like(x), lw=0.5, ls='dashed', c='k')
ax1r.plot(x, np.zeros_like(x), lw=0.5, ls='dashed', c='k')
ax2r.plot(x, np.zeros_like(x), lw=0.5, ls='dashed', c='k')

# labels etc
ax1r.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax0.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
ax2.legend(frameon=False)
for ax in [ax0r, ax1r, ax2r]:
    ax.set_ylim(-0.2, 0.2)
for ax in [ax0, ax1, ax2]:
    ax.set_ylim(-2.6, 2.6)
    ax.tick_params(labelbottom=False)
for ax in fig.axes:
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(left=True, right=True, top=True, direction='inout')
    ax.set_xticks(np.linspace(-1.5, 1.5, 7))
for ax in [ax0r, ax1r, ax2r]:
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
for ax in [ax1, ax1r, ax2, ax2r]:
    ax.tick_params(labelleft=False)
ax0r.set_ylabel("Model/Exact - 1")
ax0.set_title(r"Initial")
ax1.set_title(r"$200\ \mathrm{Myr}$")
ax2.set_title(r"$500\ \mathrm{Myr}$")

# save
fig.savefig('fig4_diseq_accs.pdf')
