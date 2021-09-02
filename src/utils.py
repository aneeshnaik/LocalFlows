#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import torch
from scipy.integrate import trapezoid

from constants import pi, kpc


def norm_pdf(f, x):
    """Normalise a PDF over array x."""
    norm = trapezoid(f, x)
    f /= norm
    return f


def sample_velocities(Nv, v_max, v_mean, v_min):
    """Uniformly sample Nv velocities in ball of radius v_max centred on v_mean."""
    # magnitude
    v_mag = v_min + (v_max - v_min) * np.random.rand(Nv)

    # orientation
    phi = 2 * pi * np.random.rand(Nv)
    theta = np.arccos(2 * np.random.rand(Nv) - 1)

    # convert to Cartesian
    vx = v_mag * np.sin(theta) * np.cos(phi)
    vy = v_mag * np.sin(theta) * np.sin(phi)
    vz = v_mag * np.cos(theta)

    # stack
    vel = v_mean + np.stack((vx, vy, vz), axis=-1)
    return vel


def diff_DF(q, p, df_func, df_args):
    """
    Calculate spatial and velocity gradients of DF.

    Parameters
    ----------
    q : np.array, shape (N, 3) or (3)
        Positions at which to evaluate DF gradients. Either an array shaped
        (N, 3) for N different phase points, or shape (3) for single phase
        point. UNITS: metres.
    p : np.array, shape (N, 3) or (3)
        Velocities at which to evaluate DF gradients. UNITS: m/s.
    df_func : function
        Function that evaluates DF for given q and p as described above, e.g.
        either calc_DF function in hernquist.py or either calc_DF function in
        ml.py.
    df_args : dict
        Additional arguments for df_func, e.g. M and a for the hernquist.py
        functions.

    Returns
    -------
    gradxf : np.array, shape (N, 3) or (3)
        Spatial gradient of DF. UNITS: [DF units] / metres.
    gradvf : np.array, shape (N, 3) or (3)
        Velocity gradient of DF. UNITS: [DF units] / (m/s).

    """
    # check if 1D
    oneD = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
        p = p[None]

    # convert to torch tensors
    q = torch.tensor(q, requires_grad=True)
    p = torch.tensor(p, requires_grad=True)

    # evaluate DF
    f = df_func(q, p, **df_args)

    # calculate f gradients; dfdq and dfdp both have shape (Nv, 3)
    grad = torch.autograd.grad
    dfdq = grad(f, q, torch.ones_like(f), create_graph=True)[0]
    dfdp = grad(f, p, torch.ones_like(f), create_graph=True)[0]
    if oneD:
        dfdq = dfdq[0]
        dfdp = dfdp[0]

    gradxf = dfdq.detach().numpy()
    gradvf = dfdp.detach().numpy()
    return gradxf, gradvf


def logit(x):
    """Logit function."""
    return np.log(x / (1 - x))


def get_rescaled_tensor(datadir, num_files, u_pos, u_vel, cen, R_cut=None):
    """
    Load data, rescale, recentre, shuffle, and make into torch tensor.

    Parameters
    ----------
    dfile : str
        Path to .npz file containing dataset, e.g. files in /data directory.
    u_pos : float
        Rescaling for R, z. UNITS: m.
    u_vel : float
        Velocity rescaling. UNITS: m/s.
    cen : np.array, shape (5)
        Phase point about which to re-centre data, (R, z, vR, vphi, vz). UNITS:
        (m, m, m/s, m/s, m/s).

    Returns
    -------
    data_tensor : torch.Tensor, shape (N, 5)
        Torch tensor containing dataset ready to train flow. Shape is (N, 5),
        where N is number of data points.

    """
    # load data
    print("Loading data...", flush=True)
    R, z, vR, vz, vphi = concatenate_data(datadir, num_files, R_cut=R_cut, R_cen=cen[0])

    # shift and rescale positions
    R = (R - cen[0]) / u_pos
    z = (z - cen[1]) / u_pos
    vR = (vR - cen[2]) / u_vel
    vphi = (vphi - cen[3]) / u_vel
    vz = (vz - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R, z, vR, vphi, vz), axis=-1)
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))

    return data_tensor


def concatenate_data(datadir, num_files, R_cut=None, R_cen=8 * kpc):

    # check if dir ends in '/', otherwise append
    if datadir[-1] != '/':
        datadir += '/'

    # loop over files
    R = np.array([])
    z = np.array([])
    vR = np.array([])
    vz = np.array([])
    vphi = np.array([])
    for k in range(num_files):

        # load file
        d = np.load(datadir + f"{k}.npz")

        # keep only stars within R_cut if specified
        if R_cut is not None:
            assert type(R_cut) in [float, np.float32, np.float64]
            assert type(R_cen) in [float, np.float32, np.float64]
            inds = np.abs(d['R'] - R_cen) < R_cut
        else:
            inds = np.ones(d['R'].shape, dtype=bool)

        # append data
        R = np.append(R, d['R'][inds])
        z = np.append(z, d['z'][inds])
        vR = np.append(vR, d['vR'][inds])
        vz = np.append(vz, d['vz'][inds])
        vphi = np.append(vphi, d['vphi'][inds])

    return R, z, vR, vz, vphi
