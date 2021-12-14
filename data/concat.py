#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In each data subdirectory, concatenate individual sample files into 1 file.

Created: December 2021
Author: A. P. Naik
"""
import numpy as np
from tqdm import trange


if __name__ == '__main__':

    # loop over subdirs
    subdirs = ['fiducial', 'perturbed_t0', 'perturbed_t2', 'perturbed_t5']
    for subdir in subdirs:

        # print message
        print("Concetanating " + subdir)

        # loop over files
        R = np.array([])
        z = np.array([])
        vR = np.array([])
        vz = np.array([])
        vphi = np.array([])
        for k in trange(2000):

            # load file
            d = np.load(subdir + f"/{k}.npz")

            # append data
            R = np.append(R, d['R'])
            z = np.append(z, d['z'])
            vR = np.append(vR, d['vR'])
            vz = np.append(vz, d['vz'])
            vphi = np.append(vphi, d['vphi'])

        # save
        np.savez(subdir + '/dset', R=R, z=z, vR=vR, vz=vz, vphi=vphi)
