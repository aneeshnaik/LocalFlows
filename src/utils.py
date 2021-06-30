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


def norm_pdf(f, x):
    """Normalise a PDF over array x."""
    norm = trapezoid(f, x)
    f /= norm
    return f


def logit(x):
    """Logit function."""
    return np.log(x / (1 - x))


def get_rescaled_tensor(dfile, u_pos, u_vel, cen):
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
    data = np.load(dfile)

    # shift and rescale positions
    R = (data['R'] - cen[0]) / u_pos
    z = (data['z'] - cen[1]) / u_pos
    vR = (data['vR'] - cen[2]) / u_vel
    vphi = (data['vphi'] - cen[3]) / u_vel
    vz = (data['vz'] - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R, z, vR, vphi, vz), axis=-1)
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))

    return data_tensor


def get_rescaled_tensor_trans(dfile, u_pos, u_vel, cen):
    """
    Load data, rescale, recentre, transform position coordinates, shuffle,
    and make into torch tensor.

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
    print("Loading data...")
    data = np.load(dfile)

    # shift and rescale positions
    R = (data['R'] - cen[0]) / u_pos + 0.5
    z = (data['z'] - cen[1]) / u_pos + 0.5
    vR = (data['vR'] - cen[2]) / u_vel
    vphi = (data['vphi'] - cen[3]) / u_vel
    vz = (data['vz'] - cen[4]) / u_vel

    # transform poositions
    qR = logit(R)
    qz = logit(z)

    # get rid of data points with very high transformed radii
    mask = (np.abs(qR) < 8) & (np.abs(qz) < 8)
    print(f"Keeping {mask.sum()} data points...")
    qR = qR[mask]
    qz = qz[mask]
    vR = vR[mask]
    vphi = vphi[mask]
    vz = vz[mask]

    # stack and shuffle data
    data = np.stack((qR, qz, vR, vphi, vz), axis=-1)
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))

    return data_tensor
