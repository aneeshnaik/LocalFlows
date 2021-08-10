#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to the quasi-isothermal DF (qDF).

Created: June 2021
Author: A. P. Naik
"""
import numpy as np

from galpy.potential import PowerSphericalPotentialwCutoff as BulgePhi
from galpy.potential import MiyamotoNagaiPotential as DiscPhi
from galpy.potential import NFWPotential as HaloPhi
from galpy.potential import evaluatezforces
from galpy.util.bovy_conversion import force_in_pcMyr2
from galpy.actionAngle import actionAngleStaeckel
from galpy.df import quasiisothermaldf as qdf

from constants import kpc, pc, Myr


def create_MW_potential(ddtype):
    """Create galpy MW potential with desired dark disc type."""
    if ddtype == 0:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        halo = HaloPhi(a=16 / 8., normalize=0.35)
        mw = bulge + disc + halo
    elif ddtype == 1:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.02 / 8., normalize=0.03)
        halo = HaloPhi(a=16 / 8., normalize=0.32)
        mw = bulge + disc1 + disc2 + halo
    elif ddtype == 2:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.02 / 8., normalize=0.06)
        halo = HaloPhi(a=16 / 8., normalize=0.29)
        mw = bulge + disc1 + disc2 + halo
    elif ddtype == 3:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.05 / 8., normalize=0.06)
        halo = HaloPhi(a=16 / 8., normalize=0.29)
        mw = bulge + disc1 + disc2 + halo
    elif ddtype == 4:
        bulge = BulgePhi(alpha=1.8, rc=1.9 / 8., normalize=0.05)
        disc1 = DiscPhi(a=3. / 8., b=0.28 / 8., normalize=0.6)
        disc2 = DiscPhi(a=3. / 8., b=0.1 / 8., normalize=0.06)
        halo = HaloPhi(a=16 / 8., normalize=0.29)
        mw = bulge + disc1 + disc2 + halo
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
