#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'Kick' stars, then evolve by 500 Myr in MW potential.

Created: August 2021
Author: A. P. Naik
"""
import numpy as np
import sys

from galpy.orbit import Orbit
from galpy.util.bovy_conversion import time_in_Gyr

sys.path.append("../src")
from qdf import create_MW_potential
from constants import kpc


if __name__ == "__main__":

    # parse arguments
    idx = 0
    #assert len(sys.argv) == 2
    #idx = int(sys.argv[1])

    # load dataset; convert data to galpy units; stack
    data = np.load(f"full_MW_initial/{idx}.npz")
    u_x = 8 * kpc
    u_v = 220000
    R = data['R'] / u_x
    z = data['z'] / u_x
    vR = data['vR'] / u_v
    vz = data['vz'] / u_v
    vphi = data['vphi'] / u_v
    eta = np.stack((R, vR, vphi, z, vz), axis=-1)

    # perturbation: 10% of stars kicked by 20km/s in +z direction
    np.random.seed(42)
    N = eta.shape[0]
    N_kick = N // 10
    kick_inds = np.random.choice(np.arange(N), size=N_kick, replace=False)
    kick = 20000 / u_v
    eta[kick_inds, -1] += kick

    # set up MW potential
    mw = create_MW_potential(0)

    # initialise orbit
    o = Orbit(eta)

    # orbit times: 500 Myr
    ts = np.linspace(0, 0.5 / time_in_Gyr(220., 8.), 2)

    # integrate orbit
    o.integrate(ts, mw, method='leapfrog_c')

    # save
    R = o.getOrbit()[:, -1, 0] * u_x
    vR = o.getOrbit()[:, -1, 1] * u_v
    vphi = o.getOrbit()[:, -1, 2] * u_v
    z = o.getOrbit()[:, -1, 3] * u_x
    vz = o.getOrbit()[:, -1, 4] * u_v
    np.savez(
        f"full_MW_final_perturbed/{idx}.npz",
        R=R, z=z, vR=vR, vphi=vphi, vz=vz,
    )

