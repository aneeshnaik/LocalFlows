#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train normalising flow.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import sys

sys.path.append("../../src")
from constants import kpc
from ml import train_flow
from utils import get_rescaled_tensor


if __name__ == '__main__':

    # load data
    u_pos = (kpc / 5)
    u_vel = 100000
    cen = np.array([8 * kpc, 0.01 * kpc, 0, 220000, 0])
    dfile = "../../data/datasets/3_400/compiled.npz"
    data = get_rescaled_tensor(dfile, u_pos, u_vel, cen)

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
