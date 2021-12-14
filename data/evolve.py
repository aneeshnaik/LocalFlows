#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolve stars in perturbed dataset by 500 Myr in MW potential.

Script takes integer argument from 0-1999, corresponding to file in
"perturbed_t0" directory. Stars in that data file are then evolved in MW
potential. Snapshots are saved at 200 and 500 Myr in "perturbed_t2" and
"perturbed_t5" directory respectively.

Created: August 2021
Author: A. P. Naik
"""
import numpy as np
import sys
from os.path import exists

from galpy.orbit import Orbit
from galpy.util.conversion import time_in_Gyr

sys.path.append("../src")
from qdf import create_MW_potential
from constants import kpc


if __name__ == "__main__":

    # check subdir exists
    if not exists("perturbed_t2"):
        raise FileNotFoundError(
            "Expected to find subdir 'perturbed_t2', try `mkdir perturbed_t2`"
        )
    if not exists("perturbed_t5"):
        raise FileNotFoundError(
            "Expected to find subdir 'perturbed_t5', try `mkdir perturbed_t5`"
        )

    # parse arguments
    assert len(sys.argv) == 2
    idx = int(sys.argv[1])

    # load dataset; convert data to galpy units; stack
    data = np.load(f"perturbed_t0/{idx}.npz")
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
    t_array = np.array([0, 0.2, 0.5]) / time_in_Gyr(220., 8.)

    # integrate orbit
    o.integrate(t_array, mw, method='leapfrog_c')

    # save
    dirnames = ['t2', 't5']
    for i in range(2):

        R = o.getOrbit()[:, i + 1, 0] * u_x
        vR = o.getOrbit()[:, i + 1, 1] * u_v
        vphi = o.getOrbit()[:, i + 1, 2] * u_v
        z = o.getOrbit()[:, i + 1, 3] * u_x
        vz = o.getOrbit()[:, i + 1, 4] * u_v
        np.savez(
            f"perturbed_{dirnames[i]}/{idx}.npz",
            R=R, z=z, vR=vR, vphi=vphi, vz=vz,
        )
