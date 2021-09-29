#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measured accelerations in a mag-limited sample, pre- and post- correction.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble
from constants import kpc, pc, Myr, pi
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


def calc_S(q):

    R_arr = q[:, 0]
    z_arr = q[:, 2]
    N_pts = R_arr.shape[0]

    # set up array of phi
    N_phi = 250
    phi_min = -pi / 25
    phi_max = pi / 25
    phi_arr = torch.linspace(phi_min, phi_max, N_phi)
    dphi = np.diff(phi_arr)[0]
    Dphi = phi_max - phi_min

    # stack into arrays
    phi_tiled = phi_arr[None].tile([N_pts, 1])
    R_tiled = R_arr[:, None].tile([1, N_phi])
    z_tiled = z_arr[:, None].tile([1, N_phi])

    # distance at each position
    x_GC = R_tiled * torch.cos(phi_tiled)
    y_GC = R_tiled * torch.sin(phi_tiled)
    z_GC = z_tiled
    x_HC = x_GC - 8 * kpc
    y_HC = y_GC
    z_HC = z_GC - 0.01 * kpc
    d = torch.sqrt(x_HC**2 + y_HC**2 + z_HC**2)
    d_pc = d / pc

    # limiting absolute mag
    x_lim = 25 - 5 * torch.log10(d_pc)

    # cap this to 12
    x_lim[x_lim > 12] = 12

    # observed fraction
    alpha = 0.55
    x1 = -5
    x2 = 12
    eax1 = np.exp(alpha * x1)
    eax2 = np.exp(alpha * x2)
    p_obs = (torch.exp(alpha * x_lim) - eax1) / (eax2 - eax1)

    # integrate to S
    S = (torch.sum(p_obs, dim=-1) * dphi) / Dphi
    return S


def calc_DF_selcorr(q, p, u_q, u_p, q_cen, p_cen, flows):
    S = calc_S(q)
    f_obs = calc_DF_ensemble(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_true = f_obs / S
    return f_true


datafile = "figB2_data.npz"
if not exists(datafile):
    
    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])
    
    # set up spatial arrays
    Nx = 60
    Nv = 1000
    lim = 1.75 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    
    # get true accels
    mw = create_MW_potential(darkdisc=False)
    a_true = calc_MW_az(pos, mw)
    
    # load flows
    flowdir = "../nflow_models/test_maglim/mag/"
    f1 = load_flow_ensemble(
        flowdir=flowdir, inds=np.arange(20),
        n_dim=5, n_layers=8, n_hidden=64
    )
    # load flows
    flowdir = "../nflow_models/test_maglim/vol/"
    f2 = load_flow_ensemble(
        flowdir=flowdir, inds=np.arange(20),
        n_dim=5, n_layers=8, n_hidden=64
    )
    
    # get model accels
    a_v = np.zeros_like(a_true)
    a_m = np.zeros_like(a_true)
    a_mcorr = np.zeros_like(a_true)
    args1 = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen, 'flows': f1
    }
    args2 = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen, 'flows': f2
    }
    for i in tqdm(range(Nx)):
        p = sample_velocities(
            Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]),
            v_min=10000
        )
        q = np.tile(pos[i][None], reps=[Nv, 1])
    
        gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_ensemble, df_args=args1)
        a_m[i] = calc_accel_CBE(q, p, gxf, gvf)[2]
        
        gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_selcorr, df_args=args1)
        a_mcorr[i] = calc_accel_CBE(q, p, gxf, gvf)[2]
        
        gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_ensemble, df_args=args2)
        a_v[i] = calc_accel_CBE(q, p, gxf, gvf)[2]

    np.savez(
        datafile,
        x=z_arr / kpc,
        y0=a_true / (pc / Myr**2),
        y1=a_v / (pc / Myr**2),
        y2=a_m / (pc / Myr**2),
        y3=a_mcorr / (pc / Myr**2)
    )


# load datafile
data = np.load(datafile)
x = data['x']
y0 = data['y0']
y1 = data['y1']
y2 = data['y2']
y3 = data['y3']
r1 = y1 / y0 - 1
r2 = y2 / y0 - 1
r3 = y3 / y0 - 1

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
fig = plt.figure(figsize=(3.3, 3.5), dpi=150)
bottom = 0.11
top = 0.985
left = 0.155
right = 0.97
rfrac = 0.2
dX = right - left
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
ax = fig.add_axes([left, bottom + rdY, dX, dY])
axr = fig.add_axes([left, bottom, dX, rdY])

# colour
c1 = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]
c2 = plt.cm.Spectral(np.linspace(0, 1, 10))[7][None]
c3 = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# plots
ax.plot(x, y0, c='k', ls='dashed', zorder=0, label=r"Exact")
ax.scatter(x, y1, c=c1, s=8, zorder=1, label="Volume limited")
ax.scatter(x, y2, c=c2, s=8, zorder=1, label="Mag. limited, uncorrected")
ax.scatter(x, y3, c=c3, s=8, zorder=1, label="Mag. limited, corrected")
axr.plot(x, r1, lw=2, c=c1)
axr.plot(x, r2, lw=2, c=c2)
axr.plot(x, r3, lw=2, c=c3)

# labels etc
axr.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
axr.set_ylabel("Model/Exact - 1")
ax.legend(frameon=False, loc='lower left')
axr.set_ylim(-0.25, 0.5)
ax.set_ylim(-3.5, 2.8)
axr.set_yticks([-0.2, 0, 0.2, 0.4])
for ax in [ax, axr]:
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(np.linspace(-1.5, 1.5, 7))
    ax.tick_params(right=True, top=True, direction='inout')

# save
fig.savefig("figB2_maglim_accs.pdf")
