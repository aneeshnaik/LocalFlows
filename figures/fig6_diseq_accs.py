#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5: Dependence of eta (central density accuracy) against # data points.

Created: September 2021
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
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


# check if datafile exists, otherwise create and save
datafile = "fig6_data.npz"
if not exists(datafile):

    # hyperparams
    Nv = 1000
    Nx = 60
    
    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # load MW model
    mw = create_MW_potential(darkdisc=False, ddtype=None)

    # set up spatial arrays
    lim = 1.1 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    
    # get true accels
    y_true = calc_MW_az(pos, mw) / (pc / Myr**2)
    
    # loop over models
    flowdirs = [
        'noDD_initial_unperturbed',
        'noDD_initial_perturbed',
        'noDD_final_unperturbed',
        'noDD_final_perturbed',
    ]
    y_model = np.zeros((4, Nx))
    for i in range(4):
    
        # load flows
        flows = load_flow_ensemble(
            '../nflow_models/' + flowdirs[i], 
            inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
        )
        df_args = {
            'u_q': u_q, 'u_p': u_p,
            'q_cen': q_cen, 'p_cen': p_cen,
            'flows': flows
        }

        # get model accels
        for j in tqdm(range(Nx)):
            p = sample_velocities(Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]), v_min=10000)
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
left = 0.08
right = 0.975
top = 0.935
bottom = 0.115
rfrac = 1/4
dX = (right - left) / 2
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
ax0 = fig.add_axes([left, bottom + rdY, dX, dY])
ax1 = fig.add_axes([left + dX, bottom + rdY, dX, dY])
ax0r = fig.add_axes([left, bottom, dX, rdY])
ax1r = fig.add_axes([left + dX, bottom, dX, rdY])

# colour
c_up = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]
c_p = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# main plots
ax0.plot(x, y_true, c='k', ls='dashed', zorder=0)
ax1.plot(x, y_true, c='k', ls='dashed', zorder=0, label="Exact")
ax0.scatter(x, y_model[0], c=c_up, s=8, zorder=1)
ax0.scatter(x, y_model[1], c=c_p, s=8, zorder=1)
ax1.scatter(x, y_model[2], c=c_up, s=8, zorder=1, label="Unperturbed")
ax1.scatter(x, y_model[3], c=c_p, s=8, zorder=1, label="Perturbed")

# residuals
r = y_model / y_true - 1
ax0r.plot(x, r[0], lw=2, c=c_up)
ax0r.plot(x, r[1], lw=2, c=c_p)
ax1r.plot(x, r[2], lw=2, c=c_up)
ax1r.plot(x, r[3], lw=2, c=c_p)
ax0r.plot(x, np.zeros_like(x), lw=0.5, ls='dashed', c='k')
ax1r.plot(x, np.zeros_like(x), lw=0.5, ls='dashed', c='k')

# labels etc
ax0r.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax1r.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax0.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
ax1.legend(frameon=False)
for ax in [ax0r, ax1r]:
    ax.set_ylim(-0.2, 0.2)
for ax in [ax0, ax1]:
    ax.set_ylim(-2.4, 2.4)
for ax in [ax0, ax0r, ax1, ax1r]:
    ax.set_xticks(np.linspace(-1, 1, 9))
for ax in [ax0r, ax1r]:
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
for ax in [ax0, ax0r, ax1, ax1r]:
    ax.set_xlim(x[0], x[-1])
ax0.tick_params(labelbottom=False)
ax1.tick_params(labelbottom=False)
for ax in [ax0, ax1, ax0r, ax1r]:
    ax.tick_params(left=True, right=True, top=True, direction='inout')
for ax in [ax1, ax1r]:
    ax.tick_params(labelleft=False)
ax0r.set_ylabel("Model/Exact - 1")
ax0.set_title(r"$t = 0$")
ax1.set_title(r"$t = 500\ \mathrm{Myr}$")

# save
fig.savefig('fig6_diseq_accs.pdf')
