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

from tqdm import trange

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from constants import pc, kpc, Myr
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE
from comparison_DF1D import get_bestpars_DF1D, calc_az_DF1D
from comparison_jeans import get_bestpars_jeans, calc_az_jeans


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
    v_args = {'Nv': Nv, 'v_max': 50000, 'v_min': 10000, 'v_mean': p_cen}
    for j in trange(Nx):
        p = sample_velocities(**v_args)
        q = np.tile(pos[j][None], reps=[Nv, 1])
        gxf, gvf = diff_DF(q, p, df_func=calc_DF_model, df_args=df_args)
        a_model[j] = calc_accel_CBE(q, p, gxf, gvf)[2]

    return a_model


datafile = "fig6_data.npz"
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
    th_widmark = get_bestpars_DF1D("../data/fiducial/")
    a_widmark = calc_az_DF1D(z_arr, th_widmark)
    y1_widmark = a_widmark / (pc / Myr**2)

    th_widmark = get_bestpars_DF1D("../data/perturbed_t5/")
    a_widmark = calc_az_DF1D(z_arr, th_widmark)
    y2_widmark = a_widmark / (pc / Myr**2)

    th_salomon = get_bestpars_jeans("../data/fiducial/")
    a_salomon = calc_az_jeans(z_arr, th_salomon)
    y1_salomon = a_salomon / (pc / Myr**2)

    th_salomon = get_bestpars_jeans("../data/perturbed_t5/")
    a_salomon = calc_az_jeans(z_arr, th_salomon)
    y2_salomon = a_salomon / (pc / Myr**2)

    a_flows = calc_az_flows(z_arr, "fiducial")
    y1_flows = a_flows / (pc / Myr**2)

    a_flows = calc_az_flows(z_arr, "perturbed_t5")
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

# residuals
with np.errstate(divide='ignore', invalid='ignore'):
    r1_widmark = y1_widmark / y_true - 1
    r2_widmark = y2_widmark / y_true - 1
    r1_salomon = y1_salomon / y_true - 1
    r2_salomon = y2_salomon / y_true - 1
    r1_flows = y1_flows / y_true - 1
    r2_flows = y2_flows / y_true - 1

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(6.9, 3.2), dpi=150)
left = 0.075
right = 0.98
bottom = 0.12
top = 0.92
rfrac = 1 / 4
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
dX = (right - left) / 2
ax1 = fig.add_axes([left, bottom + rdY, dX, dY])
ax2 = fig.add_axes([left + dX, bottom + rdY, dX, dY])
ax1r = fig.add_axes([left, bottom, dX, rdY])
ax2r = fig.add_axes([left + dX, bottom, dX, rdY])


# colour
c1 = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]
c2 = plt.cm.Spectral(np.linspace(0, 1, 10))[7][None]
c3 = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# plot
ax1.plot(x, -y_true, label="True", c='k', ls='dashed')
ax1.plot(x, -y1_flows, lw=2, c=c1, label="This work")
ax1.plot(x, -y1_widmark, lw=2, c=c2, label="DF-fitting")
ax1.plot(x, -y1_salomon, lw=2, c=c3, label="Jeans analysis")
ax2.plot(x, -y_true, label="True", c='k', ls='dashed')
ax2.plot(x, -y2_flows, c=c1, lw=2)
ax2.plot(x, -y2_widmark, c=c2, lw=2)
ax2.plot(x, -y2_salomon, c=c3, lw=2)
ax1r.plot(x, r1_flows, c=c1)
ax1r.plot(x, r1_widmark, c=c2)
ax1r.plot(x, r1_salomon, c=c3)
ax2r.plot(x, r2_flows, c=c1)
ax2r.plot(x, r2_widmark, c=c2)
ax2r.plot(x, r2_salomon, c=c3)
ax1r.plot([-0.1, 1.7], [0, 0], lw=0.5, ls='dashed', c='k')
ax2r.plot([-0.1, 1.7], [0, 0], lw=0.5, ls='dashed', c='k')

# labels etc
ylim = ax2.get_ylim()
ax1.set_ylim(ylim)
for ax in [ax1r, ax2r]:
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax1.set_ylabel(r'$|a_z|\ \left[\mathrm{pc/Myr}^2\right]$')
ax1r.set_ylabel("Recon./Exact - 1")
ax1.legend(frameon=False)
for ax in fig.axes:
    ax.tick_params(left=True, right=True, top=True, direction='inout')
    ax.set_xlim(-0.07, 1.67)
ax2.tick_params(labelleft=False)
ax2r.tick_params(labelleft=False)
ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax1.set_title(r"Unperturbed")
ax2.set_title(r"$500\ \mathrm{Myr}$ post-perturbation")

# save
fig.savefig("fig6_method_comparison.pdf")
