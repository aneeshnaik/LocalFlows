#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: a_R and a_z from mock dataset with proper motion errors.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from os.path import exists

import torch
from torch import nn

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.distributions.normal import ConditionalDiagonalNormal as CDN
from nflows.transforms import CompositeTransform
from nflows.transforms import ReversePermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform as MAAT

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_single
from constants import kpc, pc, Myr
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az, calc_MW_aR
from cbe import calc_accel_CBE


def setup_nu_flow(n_layers, n_hidden):

    # base distribution
    base_dist = StandardNormal(shape=[2])

    # loop over layers in flow
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MAAT(features=2, hidden_features=n_hidden))

    # assemble flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)

    return flow


def setup_pvx_flow(n_layers, n_hidden):

    # base distribution
    base_dist = CDN(shape=[3], context_encoder=nn.Linear(2, 6))

    # loop over layers
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=3))
        transforms.append(MAAT(
            features=3, hidden_features=n_hidden, context_features=2)
        )

    # assemble flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def load_nu_flow(state_dict, n_layers, n_hidden):
    flow = setup_nu_flow(n_layers=n_layers, n_hidden=n_hidden)
    flow.load_state_dict(torch.load(state_dict))
    flow.eval()
    return flow


def load_pvx_flow(state_dict, n_layers, n_hidden):
    flow = setup_pvx_flow(n_layers=n_layers, n_hidden=n_hidden)
    flow.load_state_dict(torch.load(state_dict))
    flow.eval()
    return flow


def load_nu_flow_ensemble(flowdir, inds, n_layers, n_hidden):

    # check if dir ends in '/', otherwise append
    if flowdir[-1] != '/':
        flowdir += '/'

    # loop over inds and load flows
    flows = []
    for i in inds:
        fname = flowdir + f"{i}_best.pth"
        flows.append(load_nu_flow(fname, n_layers, n_hidden))
    return flows


def load_pvx_flow_ensemble(flowdir, inds, n_layers, n_hidden):

    # check if dir ends in '/', otherwise append
    if flowdir[-1] != '/':
        flowdir += '/'

    # loop over inds and load flows
    flows = []
    for i in inds:
        fname = flowdir + f"{i}_best.pth"
        flows.append(load_pvx_flow(fname, n_layers, n_hidden))
    return flows


def calc_nu_single(q, u_q, q_cen, flow):

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)

    # rescale units
    q = (q - torch.tensor(q_cen)) / u_q

    # eval f from flow
    f = flow.log_prob(q[:, [0, 2]].float()).exp()

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_pvx_single(q, p, u_q, u_p, q_cen, p_cen, flow):

    # check shapes match
    assert q.shape == p.shape

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
        p = p[None]
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)
        p = torch.tensor(p)

    # rescale units
    q = (q - torch.tensor(q_cen)) / u_q
    p = (p - torch.tensor(p_cen)) / u_p

    # eval f from flow
    f = flow.log_prob(inputs=p.float(), context=q[:, [0, 2]].float()).exp()

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_DF_split(q, p, u_q, u_p, q_cen, p_cen, nu_flows, pvx_flows):
    # loop over density flows
    N = len(nu_flows)
    for i in range(N):
        if i == 0:
            nu = calc_nu_single(q, u_q, q_cen, nu_flows[i]) / N
        else:
            nu = nu + calc_nu_single(q, u_q, q_cen, nu_flows[i]) / N

    # loop over pvx flows
    N = len(pvx_flows)
    for i in range(N):
        if i == 0:
            pvx = calc_pvx_single(q, p, u_q, u_p, q_cen, p_cen, pvx_flows[i]) / N
        else:
            pvx = pvx + calc_pvx_single(q, p, u_q, u_p, q_cen, p_cen, pvx_flows[i]) / N

    # DF
    f = nu * pvx
    return f


# check if datafile exists, otherwise create and save
datafile = "fig3_data.npz"
if not exists(datafile):

    # load flows
    Nflows = 100
    flows = load_flow_ensemble(
        '../flows/test_errors',
        inds=np.arange(Nflows), n_dim=5, n_layers=8, n_hidden=64
    )
    nu_flows = load_nu_flow_ensemble(
        '../flows/test_RV/nu',
        inds=np.arange(20), n_layers=8, n_hidden=64
    )
    pvx_flows = load_pvx_flow_ensemble(
        '../flows/test_RV/pvx',
        inds=np.arange(20), n_layers=8, n_hidden=64
    )

    # load MW model
    mw = create_MW_potential()

    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])
    RV_args = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen,
        'nu_flows': nu_flows, 'pvx_flows': pvx_flows
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
    az_errs = np.zeros((Nx, Nflows))
    az_RV = np.zeros_like(az_true)
    for i in trange(Nx):
        p = sample_velocities(Nv=Nv, v_max=50000, v_mean=p_cen, v_min=10000)
        q = np.tile(pos[i][None], reps=[Nv, 1])
        for j in range(Nflows):
            args = {
                'u_q': u_q, 'u_p': u_p,
                'q_cen': q_cen, 'p_cen': p_cen,
                'flow': flows[j]
            }
            gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_single, df_args=args)
            az_errs[i, j] = calc_accel_CBE(q, p, gxf, gvf)[2]
        gxf, gvf = diff_DF(q, p, df_func=calc_DF_split, df_args=RV_args)
        az_RV[i] = calc_accel_CBE(q, p, gxf, gvf)[2]

    x0 = z_arr / kpc
    y0_true = az_true / (pc / Myr**2)
    y0_errs = az_errs / (pc / Myr**2)
    y0_RV = az_RV / (pc / Myr**2)

    # R ACCELS
    # set up spatial arrays
    R_arr = np.linspace(7, 9, Nx) * kpc
    z_arr = np.zeros_like(R_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)

    # get true accels
    aR_true = calc_MW_aR(pos, mw)

    # get model accels
    aR_errs = np.zeros((Nx, Nflows))
    aR_RV = np.zeros_like(aR_true)
    for i in trange(Nx):
        p = sample_velocities(Nv=Nv, v_max=50000, v_mean=p_cen, v_min=10000)
        q = np.tile(pos[i][None], reps=[Nv, 1])
        for j in range(Nflows):
            args = {
                'u_q': u_q, 'u_p': u_p,
                'q_cen': q_cen, 'p_cen': p_cen,
                'flow': flows[j]
            }
            gxf, gvf = diff_DF(q=q, p=p, df_func=calc_DF_single, df_args=args)
            aR_errs[i, j] = calc_accel_CBE(q, p, gxf, gvf)[0]
        gxf, gvf = diff_DF(q, p, df_func=calc_DF_split, df_args=RV_args)
        aR_RV[i] = calc_accel_CBE(q, p, gxf, gvf)[0]

    x1 = R_arr / kpc
    y1_true = aR_true / (pc / Myr**2)
    y1_errs = aR_errs / (pc / Myr**2)
    y1_RV = aR_RV / (pc / Myr**2)

    # save
    np.savez(
        datafile,
        x0=x0, y0_true=y0_true, y0_errs=y0_errs, y0_RV=y0_RV,
        x1=x1, y1_true=y1_true, y1_errs=y1_errs, y1_RV=y1_RV
    )


# load data
data = np.load(datafile)
x0 = data['x0']
y0_true = data['y0_true']
y0_errs = data['y0_errs']
y0_RV = data['y0_RV']
x1 = data['x1']
y1_true = data['y1_true']
y1_errs = data['y1_errs']
y1_RV = data['y1_RV']

# medians and errors
y0_median = np.median(y0_errs, axis=-1)
y0_q16 = np.percentile(y0_errs, 16, axis=-1)
y0_q84 = np.percentile(y0_errs, 84, axis=-1)
y0_err = np.stack((y0_median - y0_q16, y0_q84 - y0_median))
y1_median = np.median(y1_errs, axis=-1)
y1_q16 = np.percentile(y1_errs, 16, axis=-1)
y1_q84 = np.percentile(y1_errs, 84, axis=-1)
y1_err = np.stack((y1_median - y1_q16, y1_q84 - y1_median))

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
rfrac = 1 / 4
dX = (right - left - gap) / 2
dY = (top - bottom) * (1 - rfrac)
rdY = (top - bottom) * rfrac
ax0 = fig.add_axes([left, bottom + rdY, dX, dY])
ax1 = fig.add_axes([left + dX + gap, bottom + rdY, dX, dY])
ax0r = fig.add_axes([left, bottom, dX, rdY])
ax1r = fig.add_axes([left + dX + gap, bottom, dX, rdY])

# colour
c1 = plt.cm.Spectral(np.linspace(0, 1, 10))[2][None]
c2 = plt.cm.Spectral(np.linspace(0, 1, 10))[8][None]

# main plots
ax0.plot(x0, y0_true, c='k', ls='dashed', zorder=0, lw=1)
ax0.errorbar(x0, y0_median, y0_err, c=c1, fmt='.', ms=4)
ax0.scatter(x0, y0_RV, c=c2, s=8, zorder=1)
ax1.plot(x1, y1_true, c='k', ls='dashed', zorder=0, label="Exact")
ax1.errorbar(x1, y1_median, y1_err, c=c1, fmt='.', ms=4, label=r"\textit{Gaia}-like uncertainties")
ax1.scatter(x1, y1_RV, c=c2, s=8, zorder=1, label="Missing RVs")

# residuals
r0_err = y0_median / y0_true - 1
r1_err = y1_median / y1_true - 1
r0_RV = y0_RV / y0_true - 1
r1_RV = y1_RV / y1_true - 1
ax0r.plot(x0, r0_err, lw=2, c=c1)
ax1r.plot(x1, r1_err, lw=2, c=c1)
ax0r.plot(x0, r0_RV, lw=2, c=c2)
ax1r.plot(x1, r1_RV, lw=2, c=c2)
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
ax0r.set_ylabel("Recon./Exact - 1")
ax0.set_title('Vertical Accelerations')
ax1.set_title('Radial Accelerations')

# save
fig.savefig("fig3_validation.pdf")
