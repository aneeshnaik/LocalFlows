#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'Kick' stars from fiducial dataset, then save kicked dataset.

Individual data files are loaded one by one from "fiducial" directory. 10% are
randomly subsampled, and 'kicked' by 20 km/s in +z direction. Individual files
are then saved in "perturbed_t0" directory.

Created: August 2021
Author: A. P. Naik
"""
import numpy as np
from os.path import exists
from tqdm import trange


if __name__ == "__main__":

    # check subdir exists
    if not exists("perturbed_t0"):
        raise FileNotFoundError(
            "Expected to find subdir 'perturbed_t2', try `mkdir perturbed_t0`"
        )

    for idx in trange(2000):

        # load dataset; convert data to galpy units; stack
        data = np.load(f"fiducial/{idx}.npz")
        R = data['R']
        z = data['z']
        vR = data['vR']
        vz = data['vz']
        vphi = data['vphi']

        # perturbation: 10% of stars kicked by 20km/s in +z direction
        np.random.seed(42)
        N = vz.shape[0]
        N_kick = N // 10
        kick_inds = np.random.choice(np.arange(N), size=N_kick, replace=False)
        kick = 20000
        vz[kick_inds] += kick

        # save
        np.savez(
            f"perturbed_t0/{idx}.npz",
            R=R, z=z, vR=vR, vphi=vphi, vz=vz,
        )
