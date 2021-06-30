#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to the quasi-isothermal DF (qDF).

Created: June 2021
Author: A. P. Naik
"""
from galpy.potential import PowerSphericalPotentialwCutoff as BulgePhi
from galpy.potential import MiyamotoNagaiPotential as DiscPhi
from galpy.potential import NFWPotential as HaloPhi
from galpy.actionAngle import actionAngleStaeckel
from galpy.df import quasiisothermaldf as qdf


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
    return mw


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
