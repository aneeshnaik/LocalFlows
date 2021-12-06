#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train normalising flow.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import torch

sys.path.append("../../src")
from constants import kpc
from ml import train_flow


if __name__ == '__main__':

    # load data
    data = np.load("../../data/test_RV/data.npz")
    R = data['R']
    z = data['z']
    vR = data['vR']
    vz = data['vz']
    vphi = data['vphi']

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
    rng = np.random.default_rng(42)
    rng.shuffle(data)

    # make torch tensor
    data = torch.from_numpy(data.astype(np.float32))

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
