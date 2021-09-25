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
from constants import kpc, pc, Myr
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


def f(x, A, B):
    """Straight line func for fitting."""
    return A*x + B


datafile = "fig5_data.npz"
if not exists(datafile):

    # load MW potential
    ddtype = 0
    mw = create_MW_potential(darkdisc=True, ddtype=ddtype)

    # set up spatial arrays
    lim = 0.005 * kpc
    Nx = 10
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get true accels
    a_true = calc_MW_az(pos, mw)

    # loop over Ndata
    N_arr = np.array([100, 500, 900, 1300, 1700, 2100, 2500])
    y = np.zeros(7)
    for i in tqdm(range(7)):
        
        # load flows
        flows = load_flow_ensemble(
            f'../nflow_models/DD0L/{N_arr[i]}',
            inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
        )
        
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

        # get model accels
        a_model = np.zeros_like(a_true)
        Nv = 750
        for k in range(Nx):
            p = sample_velocities(
                Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]),
                v_min=10000
            )
            q = np.tile(pos[k][None], reps=[Nv, 1])
            gxf, gvf = diff_DF(q, p, df_func=calc_DF_model, df_args=df_args)
            a_model[k] = calc_accel_CBE(q, p, gxf, gvf)[2]        

        # gradients
        x = z_arr / kpc
        y0 = a_true / (pc/Myr**2)
        y1 = a_model / (pc/Myr**2)
        r0 = np.polyfit(x, y0, 1)[0]
        r1 = np.polyfit(x, y1, 1)[0]
        y[i] = r1 / r0

    np.savez(datafile, x=N_arr // 100, y=y)


# load data
data = np.load(datafile)
x = data['x']
y = data['y']

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

# plots
ax.plot(x, y, c=c, lw=2)
ax.plot(x, np.ones_like(x), c='k', ls='dotted')

# labels etc
ax.set_xlabel(r"$N_\mathrm{data} / 10^6$")
ax.set_ylabel(r"$\eta \equiv a_\mathrm{model}'(z=0)/ a_\mathrm{true}'(z=0)$")
ax.tick_params(left=True, right=True, top=True, direction='inout')
ax.set_ylim(0, 1.05)
ax.set_xlim(x[0], x[-1])
ax.set_xticks(np.arange(x[0], x[-1]+2, 2))

# save
fig.savefig('fig5_eta_Ndata.pdf')