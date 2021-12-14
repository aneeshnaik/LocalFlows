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
from tqdm import tqdm

from constants import pi, kpc


def vcart_to_vsph(vx, vy, vz, x, y, z):
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(R**2 + z**2)

    vr = x/r * vx + y/r * vy + z/r * vz
    vth = - (r / R) * (vz - (z / r) * vr)
    vphi = (1 / R) * (x * vy - y * vx)
    return vr, vth, vphi


def vsph_to_vcart(vr, vth, vphi, r, theta, phi):
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    vx = vr * cp * st + vth * cp * ct - vphi * sp
    vy = vr * sp * st + vth * sp * ct + vphi * cp
    vz = vr * ct - vth * st
    return vx, vy, vz


def norm_pdf(f, x):
    """Normalise a PDF over array x."""
    norm = trapezoid(f, x)
    f /= norm
    return f


def sample_magnitudes(N, rng):

    # params
    alpha = 0.55
    x1 = -5
    x2 = 12

    # normalisation
    eax1 = np.exp(alpha * x1)
    eax2 = np.exp(alpha * x2)
    A = alpha / (eax2 - eax1)

    # sample
    U = rng.uniform(size=N)
    M = np.log(alpha * U / A + eax1) / alpha
    return M


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


def get_rescaled_tensor(datadir, num_files, u_pos, u_vel, cen, R_cut=None, z_cut=None):
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
    R, z, vR, vz, vphi = concatenate_data(datadir, num_files, R_cut=R_cut, z_cut=z_cut)

    # shift and rescale positions
    R = (R - cen[0]) / u_pos
    z = (z - cen[1]) / u_pos
    vR = (vR - cen[2]) / u_vel
    vphi = (vphi - cen[3]) / u_vel
    vz = (vz - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R, z, vR, vphi, vz), axis=-1)
    rng = np.random.default_rng(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))

    return data_tensor


def load_dset(dfile, R_cut=None, z_cut=None):
    """
    Read saved .npz dataset, keeping only stars within specified cuts.

    Parameters
    ----------
    dfile : str
        .npz file containing stellar data.
    R_cut : float, optional
        Keep only stars within `R_cut` of 8 kpc. UNITS: metres. The default is
        None.
    z_cut : float, optional
        Keep only stars within `z_cut` of the midplane. The default is None.

    Returns
    -------
    R : 1D numpy array
        Galactocentric cylindrical radius. UNITS: metres.
    z : 1D numpy array
        Height above Galactic midplane. UNITS: metres.
    vR : 1D numpy array
        Galactocentric cylindrical radial velocity. UNITS: metres/s.
    vz : 1D numpy array
        Galactocentric vertical velocity. UNITS: metres/s.
    vphi : 1D numpy array
        Galactocentric azimuthal velocity. UNITS: metres/s.

    """

    # load data
    d = np.load(dfile)
    R = d['R']
    z = d['z']
    vR = d['vR']
    vz = d['vz']
    vphi = d['vphi']

    # apply spatial cut if requested
    if R_cut is not None:

        # check R_cut and z_cut makes sense
        if z_cut is None:
            raise ValueError("Need non-null z_cut if R_cut non-null.")
        floats = [float, np.float32, np.float64]
        if type(R_cut) not in floats:
            raise TypeError("Wrong type for R_cut")
        if type(z_cut) not in floats:
            raise TypeError("Wrong type for z_cut")

        # keep only data within cuts
        inds = (np.abs(R - 8 * kpc) < R_cut) & (np.abs(z) < z_cut)
        R = R[inds]
        z = z[inds]
        vR = vR[inds]
        vz = vz[inds]
        vphi = vphi[inds]

    return R, z, vR, vz, vphi
