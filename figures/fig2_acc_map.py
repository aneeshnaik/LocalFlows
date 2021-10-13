#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: 2D maps of a_R and a_z

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import exists
from tqdm import trange

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble as calc_DF_model
from qdf import create_MW_potential, calc_MW_az, calc_MW_aR
from constants import pc, kpc, Myr
from utils import sample_velocities, diff_DF
from cbe import calc_accel_CBE

datafile = "fig2_data.npz"
if not exists(datafile):

    # load flows
    flows = load_flow_ensemble(
        '../flows/fiducial',
        inds=np.arange(20), n_dim=5, n_layers=8, n_hidden=64
    )

    # load MW model
    mw = create_MW_potential()

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

    # create spatial arrays
    Nx = 128
    R_arr = np.linspace(7 * kpc, 9 * kpc, Nx)
    z_arr = np.linspace(-2 * kpc, 2 * kpc, Nx)
    R_grid, z_grid = np.meshgrid(R_arr, z_arr, indexing='ij')
    phi_grid = np.zeros_like(R_grid)
    pos = np.stack((R_grid, phi_grid, z_grid), axis=-1).reshape((Nx**2, 3))

    # get true accels
    az_true = calc_MW_az(pos, mw)
    aR_true = calc_MW_aR(pos, mw)

    # get model accels
    az_model = np.zeros_like(az_true)
    aR_model = np.zeros_like(aR_true)
    Nv = 1000
    vav = np.array([0, 220000, 0])
    for i in trange(Nx**2):
        p = sample_velocities(Nv=Nv, v_max=50000, v_mean=vav, v_min=10000)
        q = np.tile(pos[i][None], reps=[Nv, 1])
        gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_model, df_args=df_args)
        acc = calc_accel_CBE(q, p, gxf, gvf)
        aR_model[i] = acc[0]
        az_model[i] = acc[2]

    # reshape and rescale
    az_true = az_true.reshape((Nx, Nx)) / (pc / Myr**2)
    aR_true = aR_true.reshape((Nx, Nx)) / (pc / Myr**2)
    az_model = az_model.reshape((Nx, Nx)) / (pc / Myr**2)
    aR_model = aR_model.reshape((Nx, Nx)) / (pc / Myr**2)

    np.savez(datafile, az_true=az_true, aR_true=aR_true,
             az_model=az_model, aR_model=aR_model)


# load plot data
data = np.load(datafile)
az_true = data['az_true']
aR_true = data['aR_true']
az_model = data['az_model']
aR_model = data['aR_model']

# residuals
with np.errstate(divide='ignore', invalid='ignore'):
    resz = np.abs(az_model) / np.abs(az_true) - 1
    resR = np.abs(aR_model) / np.abs(aR_true) - 1

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8

# set up figure
asp = 6.9 / 8.4
fig = plt.figure(figsize=(6.9, 6.9 / asp), dpi=150)
left = 0.145
right = 0.855
bottom = 0.05
top = 0.9
dX = (right - left) / 3
dY = asp * dX * 2
cdY = 0.025
gap = 0.07
ax0 = fig.add_axes([left, bottom + dY + cdY + gap, dX, dY])
ax1 = fig.add_axes([left + dX, bottom + dY + cdY + gap, dX, dY])
ax2 = fig.add_axes([left + 2 * dX, bottom + dY + cdY + gap, dX, dY])
ax3 = fig.add_axes([left, bottom + cdY, dX, dY])
ax4 = fig.add_axes([left + dX, bottom + cdY, dX, dY])
ax5 = fig.add_axes([left + 2 * dX, bottom + cdY, dX, dY])

# image settings
extent = [7, 9, -2, 2]
azmin = np.abs(az_true).min()
azmax = np.abs(az_true).max()
aRmin = np.abs(aR_true).min()
aRmax = np.abs(aR_true).max()
im_args = {'origin': 'lower', 'aspect': 'auto', 'extent': extent}
az_args = {**im_args, 'cmap': 'bone', 'vmin': azmin, 'vmax': azmax}
aR_args = {**im_args, 'cmap': 'bone', 'vmin': aRmin, 'vmax': aRmax}
res_args = {**im_args, 'cmap': 'Spectral_r', 'vmin': -0.12, 'vmax': 0.12}

# plot
im0 = ax0.imshow(np.abs(az_model).T, **az_args)
im1 = ax1.imshow(np.abs(az_true).T, **az_args)
im2 = ax2.imshow(resz.T, **res_args)
im3 = ax3.imshow(np.abs(aR_model).T, **aR_args)
im4 = ax4.imshow(np.abs(aR_true).T, **aR_args)
im5 = ax5.imshow(resR.T, **res_args)

# colourbars
cY_z = bottom + 2 * dY + cdY + gap
cY_R = bottom
cax_z = fig.add_axes([left, cY_z, 2 * dX, cdY])
cax_rz = fig.add_axes([left + 2 * dX, cY_z, dX, cdY])
cax_R = fig.add_axes([left, cY_R, 2 * dX, cdY])
cax_rR = fig.add_axes([left + 2 * dX, cY_R, dX, cdY])
cbar_z = plt.colorbar(im0, cax=cax_z, orientation='horizontal')
cbar_rz = plt.colorbar(im2, cax=cax_rz, orientation='horizontal')
cbar_R = plt.colorbar(im3, cax=cax_R, orientation='horizontal')
cbar_rR = plt.colorbar(im5, cax=cax_rR, orientation='horizontal')

# ticks, labels etc
cbar_z.ax.xaxis.set_ticks_position('top')
cbar_rz.ax.xaxis.set_ticks_position('top')
cbar_z.ax.xaxis.set_label_position('top')
cbar_rz.ax.xaxis.set_label_position('top')
for ax in [ax3, ax4, ax5]:
    ax.tick_params(labelbottom=False, labeltop=True)
for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.tick_params(direction='inout', right=True, top=True)
    ticks = ax.xaxis.get_major_ticks()
    ticks[0].label1.set_visible(False)
    ticks[-1].label1.set_visible(False)
    ticks[0].label2.set_visible(False)
    ticks[-1].label2.set_visible(False)
for ax in [ax1, ax2, ax4, ax5]:
    ax.tick_params(labelleft=False)
ax1.set_xlabel(r"$R\ [\mathrm{kpc}]$")
ax0.set_ylabel(r"$z\ [\mathrm{kpc}]$")
ax3.set_ylabel(r"$z\ [\mathrm{kpc}]$")
cbar_z.set_label(r'$|a_z|\ \left[\mathrm{pc/Myr}^2\right]$')
cbar_R.set_label(r'$|a_R|\ \left[\mathrm{pc/Myr}^2\right]$')
cbar_rz.set_label("Model/Exact - 1")
cbar_rR.set_label("Model/Exact - 1")
bbox = dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5)
ax0.text(0.1, 0.925, "Model", ha='left', va='center',
         transform=ax0.transAxes, bbox=bbox, zorder=100, fontsize='large')
ax1.text(0.1, 0.925, "Exact", ha='left', va='center',
         transform=ax1.transAxes, bbox=bbox, zorder=100, fontsize='large')
ax2.text(0.1, 0.925, "Residuals", ha='left', va='center',
         transform=ax2.transAxes, bbox=bbox, zorder=100, fontsize='large')
largs = {"ha": "left", "va": "center", "fontsize": "large"}
ax2.text(1.075, 0.5, "Vertical", transform=ax2.transAxes, **largs)
ax5.text(1.075, 0.5, "Radial", transform=ax5.transAxes, **largs)

# save
fig.savefig("fig2_acc_map.pdf")
