#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create several realisations of error distribution.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys

sys.path.append("../src")
from utils import concatenate_data
from constants import kpc

# set up RNG
rng = np.random.default_rng(42)

# load original (unshifted) data
R, z, vR, vz, vphi = concatenate_data(
    "../data/noDD_up_t0/",
    num_files=2000,
    R_cut=2 * kpc,
    z_cut=2.5 * kpc,
    verbose=True
)


