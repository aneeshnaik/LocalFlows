#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each dataset, compile all sample files into single file.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
from tqdm import tqdm

# loop over datasets
dir = 'datasets/unrestricted'

# loop over files
R = np.array([])
z = np.array([])
vR = np.array([])
vz = np.array([])
vphi = np.array([])
for k in tqdm(range(2000)):

    # load file
    d = np.load(dir + f"/{k}.npz")

    # append data
    R = np.append(R, d['R'])
    z = np.append(z, d['z'])
    vR = np.append(vR, d['vR'])
    vz = np.append(vz, d['vz'])
    vphi = np.append(vphi, d['vphi'])

# save compiled dataset
np.savez(dir + '/compiled', R=R, z=z, vR=vR, vz=vz, vphi=vphi)
