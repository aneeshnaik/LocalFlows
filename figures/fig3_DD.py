#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: Vertical accelerations under a dark disc model.

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
from constants import kpc, pc, Myr, pi, G, M_sun
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE

def acc2sig(x):
    a_ms2 = x * (pc / Myr**2)
    sig_kgm3 = a_ms2 / (2 * pi * G)
    sig_Msunpc2 = sig_kgm3 / (M_sun / pc**2)
    return  sig_Msunpc2

def sig2acc(x):
    sig_kgm3 = x * (M_sun / pc**2)
    a_ms2 = sig_kgm3 * (2 * pi * G)
    a_pcMyr2 = a_ms2 / (pc / Myr**2)
    return a_pcMyr2


datafile = "fig3_data.npz"
if not exists(datafile):

    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])

    # set up spatial arrays
    Nx = 50
    Nv = 1000
    lim = 0.2 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get accels w/o DD
    mw_def = create_MW_potential(darkdisc=False)
    a_default = calc_MW_az(pos, mw_def)

    # load flows
    flowdir = "../nflow_models/DD0"
    flows = load_flow_ensemble(
        flowdir=flowdir, inds=np.arange(20),
        n_dim=5, n_layers=8, n_hidden=64
    )

    # load MW model
    mw = create_MW_potential(darkdisc=True, ddtype=0)

    # get true accels
    a_true = calc_MW_az(pos, mw)

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

    np.savez(
        datafile,
        x=z_arr / kpc,
        y0=a_default / (pc / Myr**2),
        y1=a_true / (pc / Myr**2),
        y2=a_model / (pc / Myr**2)
    )


# load datafile
data = np.load(datafile)
x = data['x']
y0 = data['y0']
y1 = data['y1']
y2 = data['y2']

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(3.3, 3), dpi=150)
bottom = 0.13
top = 0.97
left = 0.17
right = 0.84
dX = right - left
dY = top - bottom
ax = fig.add_axes([left, bottom, dX, dY])

# colour
c = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]

# plots
ax.plot(x, y0, c='k', ls='dotted', zorder=0, label="Exact, no DD")
ax.plot(x, y1, c='k', ls='dashed', zorder=0, label=r"Exact, DD")
ax.scatter(x, y2, c=c, s=8, zorder=1, label="Model")

# secondary axis
sax = ax.secondary_yaxis('right', functions=(acc2sig, sig2acc))

# labels etc
ax.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
sax.set_ylabel(r'$\mathrm{sgn}(z)\Sigma\ \left[\mathrm{M_\odot/pc^2}\right]$')
ax.tick_params(left=True, top=True, direction='inout')
sax.tick_params(right=True, direction='inout')
ax.legend(frameon=False)

# save
fig.savefig('fig3_DD.pdf')