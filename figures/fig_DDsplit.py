#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble
from constants import kpc, pc, Myr, pi, G, M_sun
from utils import sample_velocities, diff_DF
from qdf import create_MW_potential, calc_MW_az
from cbe import calc_accel_CBE


def calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, q_flow, p_flow):
    """
    Evaluate model DF at given phase positions from single flow.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    u_q : float
        Rescaling units for positions (i.e. rescaling used to train flow).
    u_p : float
        Rescaling units for velocities (i.e. rescaling used to train flow).
    q_cen : np.array or torch.tensor, shape (3)
        Position centre. UNITS: m.
    p_cen : np.array or torch.tensor, shape (3)
        Velocity centre. UNITS: m/s.
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows, generated
        from e.g. load_flow() or setup_MAF().

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.

    """
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
    
    # just keep R, z
    q = q[:, [0, 2]]

    # turn to floats
    q = q.float()
    p = p.float()

    # eval f from flow
    f = q_flow.log_prob(q).exp() * p_flow.log_prob(p).exp()

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_DF_ensemble(q, p, u_q, u_p, q_cen, p_cen, q_flows, p_flows):
    """
    Evaluate model DF at given phase positions from ensemble of flows.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    u_q : float
        Rescaling units for positions (i.e. rescaling used to train flow).
    u_p : float
        Rescaling units for velocities (i.e. rescaling used to train flow).
    q_cen : np.array or torch.tensor, shape (3)
        Position centre. UNITS: m.
    p_cen : np.array or torch.tensor, shape (3)
        Velocity centre. UNITS: m/s.
    flows : list of nflows.flows.Flow objects
        List of normalising flows, each is an instance of Flow object from
        nflows.flows, generated from e.g. load_flow() or setup_MAF().

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.

    """
    assert len(p_flows) == len(q_flows)

    # loop over flows
    N = len(q_flows)
    for i in range(N):
        q_flow = q_flows[i]
        p_flow = p_flows[i]
        if i == 0:
            f = calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, q_flow, p_flow) / N
        else:
            f = f + calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, q_flow, p_flow) / N
    return f


datafile = "fig_ddsplit_data.npz"
if not exists(datafile):
    
    # flow args
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])
    
    # set up spatial arrays
    Nx = 10
    Nv = 1000
    lim = 0.005 * kpc
    z_arr = np.linspace(-lim, lim, Nx)
    R_arr = 8 * kpc * np.ones_like(z_arr)
    phi_arr = np.zeros_like(z_arr)
    pos = np.stack((R_arr, phi_arr, z_arr), axis=-1)
    
    # load flows
    flowdir = "../nflow_models/DD0split_pos"
    q_flows = load_flow_ensemble(
    flowdir=flowdir, inds=np.arange(15),
    n_dim=2, n_layers=8, n_hidden=64
    )
    flowdir = "../nflow_models/DD0split_vel"
    p_flows = load_flow_ensemble(
    flowdir=flowdir, inds=np.arange(15),
    n_dim=3, n_layers=8, n_hidden=64
    )
    
    
    # load MW model
    mw = create_MW_potential(darkdisc=True, ddtype=0)
    
    # get true accels
    a_true = calc_MW_az(pos, mw)
    
    # get model accels
    a_model = np.zeros_like(a_true)
    df_args = {
        'u_q': u_q, 'u_p': u_p,
        'q_cen': q_cen, 'p_cen': p_cen, 'q_flows': q_flows, 'p_flows': p_flows,
    }
    
    for i in tqdm(range(Nx)):
        vel = sample_velocities(
            Nv=Nv, v_max=50000, v_mean=np.array([0, 220000, 0]),
            v_min=10000
        )
        pos_tiled = np.tile(pos[i][None], reps=[Nv, 1])
    
        gradxf_model, gradvf_model = diff_DF(
            q=pos_tiled, p=vel, df_func=calc_DF_ensemble, df_args=df_args
        )
        a_model[i] = calc_accel_CBE(
            pos_tiled, vel, gradxf_model, gradvf_model
        )[2]
    
    
    np.savez(
        datafile,
        x=z_arr / kpc,
        y0=a_true / (pc / Myr**2),
        y1=a_model / (pc / Myr**2)
    )


# load datafile
data = np.load(datafile)
x = data['x']
y0 = data['y0']
y1 = data['y1']

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
ax.plot(x, y0, c='k', ls='dashed', zorder=0, label=r"Exact, DD")
ax.scatter(x, y1, c=c, s=8, zorder=1, label="Model")

# secondary axis
#sax = ax.secondary_yaxis('right', functions=(acc2sig, sig2acc))

# labels etc
ax.set_xlabel(r'$z\ [\mathrm{kpc}]$')
ax.set_ylabel(r'$a_z\ \left[\mathrm{pc/Myr}^2\right]$')
#sax.set_ylabel(r'$\mathrm{sgn}(z)\Sigma\ \left[\mathrm{M_\odot/pc^2}\right]$')
ax.tick_params(left=True, top=True, direction='inout')
#sax.tick_params(right=True, direction='inout')
ax.legend(frameon=False)