#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-z projection of Gaia-like selection function.

Created: September 2021
Author: A. P. Naik
"""
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../src")
from constants import kpc, pc, pi


def calc_S(q):

    R_arr = q[:, 0]
    z_arr = q[:, 2]
    N_pts = R_arr.shape[0]

    # set up array of phi
    N_phi = 100
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

# get selection function
N_px = 128
R_arr = np.linspace(7 * kpc, 9 * kpc, N_px)
z_arr = np.linspace(0, 2 * kpc, N_px)
R_grid, z_grid = np.meshgrid(R_arr, z_arr, indexing='ij')
R_grid = R_grid.reshape(N_px**2)
z_grid = z_grid.reshape(N_px**2)
q = np.stack((R_grid, np.zeros_like(R_grid), z_grid), axis=-1)
q = torch.tensor(q, requires_grad=True)
S = calc_S(q).detach().numpy().reshape((N_px, N_px))

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
c = plt.cm.Spectral(np.linspace(0, 1, 40)[:12])

# set up figure
asp = 3.3 / 3.15
fig = plt.figure(figsize=(3.3, 3.3 / asp), dpi=150)
bottom = 0.12
left = 0.15
right = 0.95
dX = right - left
dY = asp * dX
ax = fig.add_axes([left, bottom, dX, dY])

# plot
X = R_grid.reshape((N_px, N_px))/kpc
Y = z_grid.reshape((N_px, N_px))/kpc
levels = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
cs = ax.contour(X, Y, S, levels=levels, colors=c)

# labels etc
l_locs = [(7.975, 1.35), (7.975, 1.2), (7.975, 0.71),
          (7.975, 0.54), (7.975, 0.38), (7.975, 0.25)]
ax.clabel(cs, levels=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
          fmt='%1.1f', manual=l_locs)
ax.tick_params(direction='inout', right=True, top=True)
ax.set_xlabel(r'$R\ [\mathrm{kpc}]$')
ax.set_ylabel(r'$z\ [\mathrm{kpc}]$')

# save
fig.savefig("figA1_sel_fun.pdf")
