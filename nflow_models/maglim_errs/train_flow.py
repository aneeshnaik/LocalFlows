#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train normalising flow.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import torch
from tqdm import trange

sys.path.append("../../src")
from constants import pc, kpc, pi, year
from ml import train_flow


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


def load_original_data():

    datadir = "../../data/maglim/"
    num_files = 100

    # loop over files
    R = np.array([])
    phi = np.array([])
    z = np.array([])
    vR = np.array([])
    vz = np.array([])
    vphi = np.array([])
    MG = np.array([])
    iterator = trange(num_files)
    for k in iterator:

        # load file
        d = np.load(datadir + f"{k}.npz")

        # append data
        R = np.append(R, d['R'])
        phi = np.append(phi, d['phi'])
        z = np.append(z, d['z'])
        vR = np.append(vR, d['vR'])
        vz = np.append(vz, d['vz'])
        vphi = np.append(vphi, d['vphi'])
        MG = np.append(MG, d['MG'])

    return R, phi, z, vR, vphi, vz, MG


def get_shifted_sample(seed):

    # set up RNG for mag/phi assignment
    rng = np.random.default_rng(seed)

    # original dataset
    R, phi, z, vR, vphi, vz, MG = load_original_data()

    # convert to galactocentric cartesian
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    vx = vR * np.cos(phi) - vphi * np.sin(phi)
    vy = vR * np.sin(phi) + vphi * np.cos(phi)

    # convert to heliocentric cartesian
    xs = x - 8 * kpc
    ys = np.copy(y)
    zs = z - 0.01 * kpc
    vxs = vx + 10000
    vys = vy - 11000
    vzs = vz - 7000

    # convert to heliocentric sphericals
    ds = np.sqrt(xs**2 + ys**2 + zs**2)
    phis = np.arctan2(ys, xs)
    thetas = np.arccos(zs / ds)
    vlos, vth, vphi = vcart_to_vsph(vxs, vys, vzs, xs, ys, zs)

    # get apparent mags
    G = MG + 5 * np.log10(ds / pc) - 5

    # PMs
    pmth = vth / ds
    pmphi = vphi / ds

    # convert to mas/yr
    pmth_masyr = pmth * (648000000 * year) / pi
    pmphi_masyr = pmphi * (648000000 * year) / pi

    # generate errors, shift, convert back to vels
    sig_PM = 1 / ((21.5 - G) * (1 + 0.1 * (21.5 - G)**2))
    sig_PM = np.maximum(sig_PM, 0.015 * np.ones_like(sig_PM))
    pmth_masyr_new = rng.normal(loc=pmth_masyr, scale=sig_PM)
    pmphi_masyr_new = rng.normal(loc=pmphi_masyr, scale=sig_PM)
    pmth_new = pmth_masyr_new * pi / (648000000 * year)
    pmphi_new = pmphi_masyr_new * pi / (648000000 * year)
    vth_new = ds * pmth_new
    vphi_new = ds * pmphi_new

    # back to heliocentric cartesians
    vxs_new, vys_new, vzs_new = vsph_to_vcart(vlos, vth_new, vphi_new, ds, thetas, phis)

    # to galactocentric cartesians
    vx_new = vxs_new - 10000
    vy_new = vys_new + 11000
    vz_new = vzs_new + 7000

    # cylindricals
    vR_new = vx_new * np.cos(phi) + vy_new * np.sin(phi)
    vphi_new = -vx_new * np.sin(phi) + vy_new * np.cos(phi)

    # shift and rescale positions
    u_pos = kpc
    u_vel = 100000
    cen = np.array([8 * kpc, 0.01 * kpc, 0, 220000, 0])
    R_new = (R - cen[0]) / u_pos
    z_new = (z - cen[1]) / u_pos
    vR_new = (vR_new - cen[2]) / u_vel
    vphi_new = (vphi_new - cen[3]) / u_vel
    vz_new = (vz_new - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R_new, z_new, vR_new, vphi_new, vz_new), axis=-1)
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
