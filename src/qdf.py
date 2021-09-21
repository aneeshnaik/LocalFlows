#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to the quasi-isothermal DF (qDF).

Created: June 2021
Author: A. P. Naik
"""
import numpy as np

from galpy.potential import PowerSphericalPotentialwCutoff as Bulge
from galpy.potential import MiyamotoNagaiPotential as MNDisc
from galpy.potential import MN3ExponentialDiskPotential as MN3Disc
from galpy.potential import NFWPotential as Halo
from galpy.potential import evaluatezforces, evaluateRforces
from galpy.util.conversion import force_in_pcMyr2
from galpy.actionAngle import actionAngleStaeckel
from galpy.df import quasiisothermaldf as qdf

from constants import kpc, pc, Myr


def create_MW_potential(darkdisc=False, ddtype=None):
    """Create galpy MW potential with desired dark disc type."""
    if darkdisc:

        # check ddtype allowed
        if ddtype not in np.arange(20):
            assert False, "Not supported yet!"

        # load param file
        inds, heights, norms = np.genfromtxt(
            "../data/DD_params.txt", dtype=None, skip_header=1, unpack=True,
            usecols=[0, 1, 2]
        )
        h_DD = heights[inds == ddtype][0] / 1000
        norm_DD = norms[inds == ddtype][0]
        norm_halo = 0.35 - norm_DD

        # put MW together
        bulge = Bulge(alpha=1.8, rc=1.9 / 8, normalize=0.05)
        disc1 = MNDisc(a=3. / 8., b=0.28 / 8, normalize=0.6)
        disc2 = MN3Disc(hr=3. / 8., hz=h_DD / 8, normalize=norm_DD)
        halo = Halo(a=16 / 8., normalize=norm_halo)
        mw = bulge + disc1 + disc2 + halo

    else:

        # put MW together
        bulge = Bulge(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc = MNDisc(a=3. / 8., b=0.28 / 8., normalize=0.6)
        halo = Halo(a=16 / 8., normalize=0.35)
        mw = bulge + disc + halo

    return mw


def calc_MW_az(pos, pot):
    """
    Evaluate true z-acceleration under MW model.

    pos: (N, 3) or (3) np array, units: metres.
    pot: galpy potential
    """
    # unpack
    if pos.ndim == 2:
        R = pos[:, 0]
        z = pos[:, 2]
    else:
        R = pos[0]
        z = pos[2]

    # evaluate acc
    u = 8 * kpc
    a = evaluatezforces(pot, R / u, z / u)
    a *= force_in_pcMyr2(220., 8.) * (pc / Myr**2)
    return a


def calc_MW_aR(pos, pot):
    """
    Evaluate true R-acceleration under MW model.

    pos: (N, 3) or (3) np array, units: metres.
    pot: galpy potential
    """
    # unpack
    if pos.ndim == 2:
        R = pos[:, 0]
        z = pos[:, 2]
    else:
        R = pos[0]
        z = pos[2]

    # evaluate acc
    u = 8 * kpc
    a = evaluateRforces(pot, R / u, z / u)
    a *= force_in_pcMyr2(220., 8.) * (pc / Myr**2)
    return a


def create_qdf(hr, sr, sz, hsr, hsz, pot):
    aA = actionAngleStaeckel(pot=pot, delta=0.45, c=True)
    df = qdf(hr, sr, sz, hsr, hsz, pot=pot, aA=aA, cutcounter=True)
    return df


def create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot):
    N = hr.size
    qdfs = []
    for i in range(N):
        qdfs.append(create_qdf(hr[i], sr[i], sz[i], hsr[i], hsz[i], pot))
    return qdfs


def calc_DF_single(q, p, qdf):
    """
    Evaluate single qDF.

    pos: (N, 3) or (3) np array. vel: ditto.
    Units: metres and m/s respectively.
    """
    # check shapes match
    assert q.shape == p.shape

    # galpy units
    ux = 8 * kpc
    uv = 220000

    # unpack and evaluate DF
    if q.ndim == 2:
        N = q.shape[0]
        R = q[:, 0]
        z = q[:, 2]
        vR = p[:, 0]
        vphi = p[:, 1]
        vz = p[:, 2]
        f = np.array([qdf(R[i] / ux, vR[i] / uv, vphi[i] / uv, z[i] / ux, vz[i] / uv)[0] for i in range(N)])
    else:
        R = q[0]
        z = q[2]
        vR = p[0]
        vphi = p[1]
        vz = p[2]
        f = qdf(R / ux, vR / uv, vphi / uv, z / ux, vz / uv)[0]
    return f


def calc_DF_ensemble(q, p, qdfs, weights):
    """
    Evaluate ensemble of qDFs.

    pos: (N, 3) or (3) np array. vel: ditto.
    Units: metres and m/s respectively.
    """
    # check shapes match
    assert q.shape == p.shape

    # loop over qdfs
    f = 0
    for i in range(len(qdfs)):
        f += weights[i] * calc_DF_single(q, p, qdfs[i])
    return f
