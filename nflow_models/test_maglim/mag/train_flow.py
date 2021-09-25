#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct mag-limited sample then train flows on it.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import torch

sys.path.append("../../../src")
from utils import concatenate_data
from constants import kpc, pi
from ml import train_flow


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


def get_maglim_sample():
    
    # set up RNG
    rng = np.random.default_rng(42)
    
    # load initial dataset
    dfile = "../../../data/test_maglim"
    data = concatenate_data(
        dfile, num_files=400, R_cut=1*kpc, z_cut=2.5*kpc, verbose=True
    )
    R = data[0]
    z = data[1]
    N = R.size
    
    # assign absolute magnitudes
    M = sample_magnitudes(N, rng)
    
    # helioc. distances: sample phi, -> cartesians,
    phi = rng.uniform(low=-pi/10, high=pi/10, size=N)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    xs = x - 8 * kpc
    ys = np.copy(y)
    zs = z - 0.01 * kpc
    d = np.sqrt(xs**2 + ys**2 + zs**2)
    
    # convert M to apparent mag
    d_pc = 1000 * d / kpc
    m = M + 5*np.log10(d_pc) - 5
    
    # keep only data with m < 20
    R = data[0][m < 20]
    z = data[1][m < 20]
    vR = data[2][m < 20]
    vz = data[3][m < 20]
    vphi = data[4][m < 20]
    
    # downsample again to 10^6
    N = R.size
    inds = rng.choice(np.arange(N), size=1000000, replace=False)
    R = R[inds]
    z = z[inds]
    vR = vR[inds]
    vz = vz[inds]
    vphi = vphi[inds]
    
    # shift and rescale positions
    u_pos = kpc
    u_vel = 100000
    cen = np.array([8 * kpc, 0.01 * kpc, 0, 220000, 0])
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


if __name__ == '__main__':

    # load data
    data = get_maglim_sample()

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
