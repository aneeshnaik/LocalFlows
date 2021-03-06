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
    data = get_rescaled_tensor(
        dfile="../../data/fiducial/dset.npz",
        u_pos=kpc, u_vel=100000,
        cen=np.array([8 * kpc, 0, 0, 220000, 0]),
        R_cut=1 * kpc, z_cut=2.5 * kpc
    )
    print(f"Found {data.shape[0]} stars", flush=True)

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
