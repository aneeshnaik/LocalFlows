#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct vol-limited sample then train flows on it.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import torch

sys.path.append("../../../src")
from utils import concatenate_data
from constants import kpc
from ml import train_flow


def get_vollim_sample():
    
    # set up RNG
    rng = np.random.default_rng(42)
    
    # load initial dataset
    dfile = "../../../data/test_maglim"
    data = concatenate_data(
        dfile, num_files=400, R_cut=1*kpc, z_cut=2.5*kpc, verbose=True
    )
    R = data[0]
    z = data[1]
    vR = data[2]
    vz = data[3]
    vphi = data[4]

    # downsample to 10^6
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
    data = get_vollim_sample()

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
