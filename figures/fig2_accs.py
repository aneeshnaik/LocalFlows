#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Local accelerations.

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
from qdf import create_MW_potential, calc_MW_az, calc_MW_aR
from cbe import calc_accel_CBE


# check if datafile exists, otherwise create and save
datafile = "fig2_data.npz"
if not exists(datafile):
    
    # load flows
    flows = load_flow_ensemble(
        '../nflow_models/noDD_up_t0', 
        inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
    )
    
    # load MW model
    mw = create_MW_potential(darkdisc=False, ddtype=None)
    
    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])
    df_args = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen,
        'flows': flows
    }
    
    # hyperparams
    Nv = 1000
    Nx = 60
    
    # Z ACCELS
    # set up spatial arrays
    lim = 1.75 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    
    # get true accels
    az_true = calc_MW_az(pos, mw)
    
    # get model accels
    az_model = np.zeros_like(az_true)
    for i in tqdm(range(Nx)):
        vel = sample_velocities(Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]), v_min=10000)
        pos_tiled = np.tile(pos[i][None], reps=[Nv, 1])
        gradxf_model, gradvf_model = diff_DF(q=pos_tiled, p=vel, df_func=calc_DF_model, df_args=df_args)
        az_model[i] = calc_accel_CBE(pos_tiled, vel, gradxf_model, gradvf_model)[2]
    x0 = z_arr / kpc
    y0_true = az_true / (pc / Myr**2)
    y0_model = az_model / (pc / Myr**2)
    
    
    # R ACCELS
    # set up spatial arrays
    R_arr = np.linspace(7, 9, Nx) * kpc
    z_arr = np.zeros_like(R_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    
    # get true accels
    aR_true = calc_MW_aR(pos, mw)
    
    # get model accels
    aR_model = np.zeros_like(aR_true)
    for i in tqdm(range(Nx)):
        vel = sample_velocities(Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]), v_min=10000)
        pos_tiled = np.tile(pos[i][None], reps=[Nv, 1])
        gradxf_model, gradvf_model = diff_DF(q=pos_tiled, p=vel, df_func=calc_DF_model, df_args=df_args)
        aR_model[i] = calc_accel_CBE(pos_tiled, vel, gradxf_model, gradvf_model)[0]
    x1 = R_arr / kpc
    y1_true = aR_true / (pc / Myr**2)
    y1_model = aR_model / (pc / Myr**2)
    
    # save
    np.savez(
        datafile,
        x0=x0, y0_true=y0_true, y0_model=y0_model,
        x1=x1, y1_true=y1_true, y1_model=y1_model
    )

else:
    data = np.load(datafile)
    x0 = data['x0']
    y0_true = data['y0_true']
    y0_model = data['y0_model']
    x1 = data['x1']
    y1_true = data['y1_true']
    y1_model = data['y1_model']


# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(6.9, 3.5), dpi=150)
left = 0.08
right = 0.915
top = 0.935
bottom = 0.125
gap = 0.02
rfrac = 1/4
dX = (right - left - gap) / 2
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
ax0 = fig.add_axes([left, bottom + rdY, dX, dY])
ax1 = fig.add_axes([left + dX + gap, bottom + rdY, dX, dY])
ax0r = fig.add_axes([left, bottom, dX, rdY])
ax1r = fig.add_axes([left + dX + gap, bottom, dX, rdY])

# colour
c = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]

# main plots
ax0.plot(x0, y0_true, c='k', ls='dashed', zorder=0)
ax0.scatter(x0, y0_model, c=c, s=8, zorder=1)
ax1.plot(x1, y1_true, c='k', ls='dashed', zorder=0, label="Exact")
ax1.scatter(x1, y1_model, c=c, s=8, zorder=1, label="Model")

# residuals
r0 = y0_model / y0_true - 1
r1 = y1_model / y1_true - 1
ax0r.plot(x0, r0, lw=2, c=c)
ax1r.plot(x1, r1, lw=2, c=c)
ax0r.plot(x0, np.zeros_like(x0), lw=0.5, ls='dashed', c='k')
ax1r.plot(x1, np.zeros_like(x1), lw=0.5, ls='dashed', c='k')

# labels etc
ax0r.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax0.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
ax1r.set_xlabel(r'$R\ [\mathrm{kpc}]$')
ax1.set_ylabel(r'$a_R\ \left[\mathrm{pc/Myr}^2\right]$')
ax1.legend(frameon=False)
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1r.yaxis.tick_right()
ax1r.yaxis.set_label_position('right')
ax1.set_ylim(-7.9, 0)
ax0r.set_ylim(-0.2, 0.2)
ax1r.set_ylim(-0.2, 0.2)
for ax in [ax0, ax0r]:
    ax.set_xticks(np.linspace(-1.5, 1.5, 7))
for ax in [ax1, ax1r]:
    ax.set_xticks(np.linspace(7, 9, 9))
for ax in [ax0r, ax1r]:
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
for ax in [ax0, ax0r]:
    ax.set_xlim(x0[0], x0[-1])
for ax in [ax1, ax1r]:
    ax.set_xlim(x1[0], x1[-1])
ax0.tick_params(labelbottom=False)
ax1.tick_params(labelbottom=False)
for ax in [ax0, ax1, ax0r, ax1r]:
    ax.tick_params(left=True, right=True, top=True, direction='inout')
ax0r.set_ylabel("Model/Exact - 1")
ax0.set_title('Vertical Accelerations')
ax1.set_title('Radial Accelerations')

# save
fig.savefig('fig2_accs.pdf')

