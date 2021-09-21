#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolve stars by 500 Myr in MW potential.

Created: August 2021
Author: A. P. Naik
"""
import numpy as np
import sys

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

sys.path.append("../src")
from qdf import create_MW_potential
from constants import kpc


if __name__ == "__main__":

    # parse arguments
    assert len(sys.argv) == 2
    idx = int(sys.argv[1])

    # load dataset; convert data to galpy units; stack
    data = np.load(f"noDD_up_t0/{idx}.npz")
    u_x = 8 * kpc
    u_v = 220000
    R = data['R'] / u_x
    z = data['z'] / u_x
    vR = data['vR'] / u_v
    vz = data['vz'] / u_v
    vphi = data['vphi'] / u_v
    eta = np.stack((R, vR, vphi, z, vz), axis=-1)

    # set up MW potential
    mw = create_MW_potential()

    # initialise orbit
    o = Orbit(eta)

    # orbit times: 500 Myr
    t_array = np.array([0, 0.1, 0.2, 0.5]) / time_in_Gyr(220., 8.)

    # integrate orbit
    o.integrate(t_array, mw, method='leapfrog_c')

    # save
    dirnames = ['t1', 't2', 't5']
    for i in range(3):

        R = o.getOrbit()[:, i + 1, 0] * u_x
        vR = o.getOrbit()[:, i + 1, 1] * u_v
        vphi = o.getOrbit()[:, i + 1, 2] * u_v
        z = o.getOrbit()[:, i + 1, 3] * u_x
        vz = o.getOrbit()[:, i + 1, 4] * u_v
        np.savez(
            f"noDD_up_{dirnames[i]}/{idx}.npz",
            R=R, z=z, vR=vR, vphi=vphi, vz=vz,
        )
