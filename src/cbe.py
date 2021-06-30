#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np


def calc_accel_CBE(pos, vel, gradxf, gradvf):
    """Convert DF gradients to accleration via Wyn's method. vel, gradxf and gradvf all np arrays shape (N, 3)."""
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