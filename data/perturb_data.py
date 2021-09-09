#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'Kick' stars, then save kicked dataset.

Created: August 2021
Author: A. P. Naik
"""
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    for idx in tqdm(range(2000)):

        # load dataset; convert data to galpy units; stack
        data = np.load(f"noDD_initial_unperturbed/{idx}.npz")
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
            f"noDD_initial_perturbed/{idx}.npz",
            R=R, z=z, vR=vR, vphi=vphi, vz=vz,
        )
