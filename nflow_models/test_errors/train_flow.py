#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct mag-limited sample then train flows on it.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import torch

sys.path.append("../../../src")
from utils import concatenate_data
from constants import pc, kpc, pi
from ml import train_flow


def sample_magnitudes(N, rng):

    # params
    alpha = 0.55
    x1 = -5
    x2 = 12

    # normalisation
    eax1 = np.exp(alpha * x1)
    eax2 = np.exp(alpha * x2)
    A = alpha / (eax2 - eax1)

    # sample
    U = rng.uniform(size=N)
    M = np.log(alpha * U / A + eax1) / alpha
    return M


def vcart_to_vsph(vx, vy, vz, x, y, z):
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(R**2 + z**2)

    vr = x/r * vx + y/r * vy + z/r * vz
    vth = - (r / R) * (vz - (z / r) * vr)
    vphi = (1 / R) * (x * vy - y * vx)
    return vr, vth, vphi


def vsph_to_vcart(vr, vth, vphi, r, theta, phi):
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    vx = vr * cp * st + vth * cp * ct - vphi * sp
    vy = vr * sp * st + vth * sp * ct + vphi * cp
    vz = vr * ct - vth * st
    return vx, vy, vz


def get_shifted_sample(seed):

    # set up RNG for mag/phi assignment
    rng = np.random.default_rng(42)

    # load original (unshifted) data
    R_old, z_old, vR_old, vz_old, vphi_old = concatenate_data(
        "../data/errors",
        num_files=800,
        R_cut=2 * kpc,
        z_cut=2.5 * kpc,
        verbose=True
    )

    # assign absolute magnitudes
    N = R_old.size
    M = sample_magnitudes(N, rng)

    # sample phi, convert to galactocentric cartesian
    phi_old = rng.uniform(low=-pi/25, high=pi/25, size=N)
    x = R_old * np.cos(phi_old)
    y = R_old * np.sin(phi_old)
    z = np.copy(z_old)
    vx = vR_old * np.cos(phi_old) - vphi_old * np.sin(phi_old)
    vy = vR_old * np.sin(phi_old) + vphi_old * np.cos(phi_old)
    vz = np.copy(vz_old)

    # convert to heliocentric cartesian
    xs = x - 8 * kpc
    ys = np.copy(y)
    zs = z - 0.01 * kpc
    vxs = vx + 10000
    vys = vy - 11000
    vzs = vz - 7000

    # convert to heliocentric sphericals
    d = np.sqrt(xs**2 + ys**2 + zs**2)
    phi = np.arctan2(ys, xs)
    theta = np.arccos(zs / d)
    vlos, vth, vphi = vcart_to_vsph(vxs, vys, vzs, xs, ys, zs)

    # get parallaxes and apparent mags
    d_pc = d / pc
    par = 1000 / d_pc
    m = M + 5*np.log10(d_pc) - 5

    # assign errors
    xp = [15, 20]
    fp = [np.log10(0.02), np.log10(0.5)]
    sig_par = 10**np.interp(m, xp=xp, fp=fp)
    sig_vlos = 2000 * np.ones_like(vlos)

    # new RNG for error assignment
    rng = np.random.default_rng(seed)

    # shift values
    vlos_new = rng.normal(vlos, scale=sig_vlos)
    par_new = rng.normal(par, scale=sig_par)
    d_pc_new = 1000 / par_new
    d_new = d_pc_new * pc

    # back to heliocentric cartesians
    x_new = d_new * np.cos(phi) * np.sin(theta)
    y_new = d_new * np.sin(phi) * np.sin(theta)
    z_new = d_new * np.cos(theta)
    vx_new, vy_new, vz_new = vsph_to_vcart(vlos_new, vth, vphi, d_new, theta, phi)

    # to galactocentric cartesians
    x_new = x_new + 8 * kpc
    y_new = np.copy(y_new)
    z_new = z_new + 0.01 * kpc
    vx_new = vx_new - 10000
    vy_new = vy_new + 11000
    vz_new = vz_new + 7000

    # to galactocentric cylindricals
    R_new = np.sqrt(x_new**2 + y_new**2)
    phi_new = np.arctan2(y_new, x_new)
    vR_new = vx_new * np.cos(phi_new) + vy_new * np.sin(phi_new)
    vphi_new = -vx_new * np.sin(phi_new) + vy_new * np.cos(phi_new)

    # keep only m < 20, +ve distance, within R and phi bounds
    mask = ((m < 20) & (d_new > 0) & (R_new > 7 * kpc) & (R_new < 9 * kpc) & (phi_new > -pi/25) & (phi_new < pi/25))
    R_new = R_new[mask]
    z_new = z_new[mask]
    vR_new = vR_new[mask]
    vz_new = vz_new[mask]
    vphi_new = vphi_new[mask]

    # downsample again to 10^6
    N_mask = mask.sum()
    inds = rng.choice(np.arange(N_mask), size=1000000, replace=False)
    R_new = R_new[inds]
    z_new = z_new[inds]
    vR_new = vR_new[inds]
    vz_new = vz_new[inds]
    vphi_new = vphi_new[inds]

    # shift and rescale positions
    u_pos = kpc
    u_vel = 100000
    cen = np.array([8 * kpc, 0.01 * kpc, 0, 220000, 0])
    R_new = (R_new - cen[0]) / u_pos
    z_new = (z_new - cen[1]) / u_pos
    vR_new = (vR_new - cen[2]) / u_vel
    vphi_new = (vphi_new - cen[3]) / u_vel
    vz_new = (vz_new - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R_new, z_new, vR_new, vphi_new, vz_new), axis=-1)
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))
    return data_tensor


if __name__ == '__main__':

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # load data
    data = get_shifted_sample(seed)

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
