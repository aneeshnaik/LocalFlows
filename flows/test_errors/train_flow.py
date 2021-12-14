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

sys.path.append("../../src")
from constants import kpc, pi, year
from ml import train_flow
from utils import load_dset, vsph_to_vcart, vcart_to_vsph


def get_shifted_sample(seed):

    # set up RNG for mag/phi assignment
    rng = np.random.default_rng(seed)

    # load data
    print("Loading data...", flush=True)
    R, z, vR, vz, vphi = load_dset("../../data/fiducial/dset.npz")

    # randomly assign phi
    print("Shifting data...", flush=True)
    phi = rng.uniform(low=-pi / 25, high=pi / 25, size=R.size)

    # randomly assign phi
    phi = rng.uniform(low=-pi / 25, high=pi / 25, size=R.size)

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
    vlos, vths, vphis = vcart_to_vsph(vxs, vys, vzs, xs, ys, zs)

    # PMs in mas/yr
    pmth = (vths / ds) * (648000000 * year) / pi
    pmphi = (vphis / ds) * (648000000 * year) / pi

    # parallax in mas
    par = kpc / ds

    # generate errors and shift data
    sig_pm = 0.025
    sig_par = 0.025
    sig_vlos = 1000
    pmth_new = rng.normal(loc=pmth, scale=sig_pm)
    pmphi_new = rng.normal(loc=pmphi, scale=sig_pm)
    par_new = rng.normal(loc=par, scale=sig_par)
    vlos_new = rng.normal(loc=vlos, scale=sig_vlos)

    # convert pms/pars back to vels/dist
    ds_new = kpc / par_new
    vths_new = ds_new * pmth_new * pi / (648000000 * year)
    vphis_new = ds_new * pmphi_new * pi / (648000000 * year)

    # back to heliocentric cartesians
    xs_new = ds_new * np.sin(thetas) * np.cos(phis)
    ys_new = ds_new * np.sin(thetas) * np.sin(phis)
    zs_new = ds_new * np.cos(thetas)
    vxs_new, vys_new, vzs_new = vsph_to_vcart(
        vlos_new, vths_new, vphis_new, ds_new, thetas, phis
    )

    # to galactocentric cartesians
    x_new = xs_new + 8 * kpc
    y_new = np.copy(ys_new)
    z_new = zs_new + 0.01 * kpc
    vx_new = vxs_new - 10000
    vy_new = vys_new + 11000
    vz_new = vzs_new + 7000

    # cylindricals
    R_new = np.sqrt(x_new**2 + y_new**2)
    phi_new = np.arctan2(y_new, x_new)
    vR_new = vx_new * np.cos(phi_new) + vy_new * np.sin(phi_new)
    vphi_new = -vx_new * np.sin(phi_new) + vy * np.cos(phi_new)

    # keep only those within region of interest
    m = (np.abs(R_new - 8 * kpc) < 1 * kpc) & (np.abs(z_new) < 2.5 * kpc)
    print(f"{m.sum()} stars in region of interest", flush=True)
    R_new = R_new[m]
    z_new = z_new[m]
    vR_new = vR_new[m]
    vphi_new = vphi_new[m]
    vz_new = vz_new[m]

    # shift and rescale positions
    u_pos = kpc
    u_vel = 100000
    cen = np.array([8 * kpc, 0, 0, 220000, 0])
    R_new = (R_new - cen[0]) / u_pos
    z_new = (z_new - cen[1]) / u_pos
    vR_new = (vR_new - cen[2]) / u_vel
    vphi_new = (vphi_new - cen[3]) / u_vel
    vz_new = (vz_new - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R_new, z_new, vR_new, vphi_new, vz_new), axis=-1)
    rng = np.random.default_rng(42)
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
