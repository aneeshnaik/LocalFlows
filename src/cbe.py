#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just one function: calc_accel_CBE, calculates accelerations from DF gradients.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np


def calc_accel_CBE(pos, vel, gradxf, gradvf):
    r"""
    Convert DF gradients to acceleration via CBE inversion.

    This function implements Eq. 26 of An et al., (2021) in cylindrical coords,
    calculating gravitational accelerations at a single point in configuration
    space via the gradients of the DF at a number of points in velocity space.

    Note: the scaling of the DF does not matter when evaluating its
    derivatives: the scaling of :math:`\nabla\Phi` looks something like:

    .. math:: [\nabla\Phi] = \frac{[v][\nabla_x F][\nabla_v F]}{[\nabla_v F]^2}

    Thus, the scaling of F cancels.

    Parameters
    ----------
    pos : np.array, shape (N, 3)
        Single position (R, phi, z) at which acceleration is to be calculated,
        tiled into shape (N, 3). UNITS: m/s.
    vel : np.array, shape (N, 3)
        Velocity points to be summed over, at which DF gradients are evaluated.
        This is (vR, vphi, vz). UNITS: m/s.
    gradxf : np.array, shape (N, 3)
        Spatial gradient of DF, evaluated at N points in phase space specified
        by vel above. UNITS: [f units] / metres.
    gradvf : TYPE
        Velocity gradient of DF, evaluated at N points in phase space specified
        by vel above. UNITS: [f units] / (m/s).

    Returns
    -------
    acc : np.array, shape (3)
        Gravitational acceleration. UNITS: m/s^2.
    """
    S = np.sum(vel * gradxf, axis=-1)
    r = pos[:, 0]
    t1 = (vel[:, 1] * vel[:, 1] / r) * gradvf[:, 0]
    t2 = (vel[:, 0] * vel[:, 1] / r) * gradvf[:, 1]
    S += t1 - t2
    R = np.sum(S[:, None] * gradvf, axis=0)
    A = gradvf[:, None] * gradvf[..., None]
    A = np.sum(A, axis=0)
    Ainv = np.linalg.inv(A)
    gphi = np.matmul(Ainv, R)
    return -gphi
