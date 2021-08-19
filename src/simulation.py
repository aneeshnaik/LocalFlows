#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np

from time import perf_counter as time
from galpy.orbit import Orbit
from galpy.util.bovy_conversion import time_in_Gyr

from qdf import create_MW_potential
from constants import kpc

if __name__ == "__main__":

    # load dataset; convert data to galpy units; stack
    data = np.load("../data/datasets/0_400/compiled.npz")
    u_x = 8 * kpc
    u_v = 220000
    R = data['R'] / u_x
    z = data['z'] / u_x
    vR = data['vR'] / u_v
    vz = data['vz'] / u_v
    vphi = data['vphi'] / u_v
    eta = np.stack((R, vR, vphi, z, vz), axis=-1)

    # set up MW potential
    mw = create_MW_potential(0)

    # initialise orbit
    o = Orbit(eta[:1000])

    # orbit times: 500 Myr
    ts = np.linspace(0, 0.5 / time_in_Gyr(220., 8.), 10000)

    # integrate orbit
    t0 = time()
    o.integrate(ts, mw, method='leapfrog_c')
    t1 = time()
    print(f"Took {t1-t0:.1f} seconds")
