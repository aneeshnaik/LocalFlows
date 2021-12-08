#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make two datasets for RV test: full 2D dataset and half-size 5D dataset.

Created: December 2021
Author: A. P. Naik
"""
import numpy as np
import sys

sys.path.append("../src")
from utils import concatenate_data
from constants import kpc

if __name__ == "__main__":

    # load fiducial dataset
    R, z, vR, vz, vphi = concatenate_data("fiducial", num_files=2000,
                                          R_cut=1 * kpc, z_cut=2.5 * kpc,
                                          verbose=True)

    # save full dataset
    np.savez("test_RV/full_dset", R=R, z=z)

    # discard random half of dataset
    N_tot = R.size
    N_RV = N_tot // 2
    inds = np.random.choice(np.arange(N_tot), size=N_RV, replace=False)
    R = R[inds]
    z = z[inds]
    vR = vR[inds]
    vz = vz[inds]
    vphi = vphi[inds]

    # save remaining half dataset
    np.savez("test_RV/RV_dset", R=R, z=z, vR=vR, vphi=vphi, vz=vz)
