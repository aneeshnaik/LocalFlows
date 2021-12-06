#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guess missing line-of-sight velocities with neural network.

Created: December 2021
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR

sys.path.append("../src")
from utils import concatenate_data, vcart_to_vsph, vsph_to_vcart
from constants import kpc, pi

rng = np.random.default_rng(42)


def load_data():
    """Load fiducial dataset."""

    # load data file
    datadir = "../data/fiducial/"
    num_files = 2000
    R, z, vR, vz, vphi = concatenate_data(
        datadir, num_files, R_cut=1 * kpc, z_cut=2.5 * kpc, verbose=True
    )

    # randomly assign phi
    phi = rng.uniform(low=-pi / 25, high=pi / 25, size=R.size)

    return R, phi, z, vR, vphi, vz


def get_helio_data(R, phi, z, vR, vphi, vz):
    """Load fiducial dataset, convert to heliocentric spherical coords."""

    # convert to galacto. cartesian
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    vx = vR * np.cos(phi) - vphi * np.sin(phi)
    vy = vR * np.sin(phi) + vphi * np.cos(phi)

    # convert to helio. cartesian
    xs = x - 8 * kpc
    ys = np.copy(y)
    zs = z - 0.01 * kpc
    vxs = vx + 10000
    vys = vy - 11000
    vzs = vz - 7000

    # convert to helio. spherical
    ds = np.sqrt(xs**2 + ys**2 + zs**2)
    phis = np.arctan2(ys, xs)
    thetas = np.arccos(zs / ds)
    vlos, vthetas, vphis = vcart_to_vsph(vxs, vys, vzs, xs, ys, zs)

    return ds, thetas, phis, vlos, vthetas, vphis


def make_datasets(R, phi, z, vR, vphi, vz, u_d, u_ang, u_v):
    """Load data, rescale, split, and return as torch Dataset instances."""

    # convert data to heliocentric sphericals
    d, theta, phi, vlos, vth, vphi = get_helio_data(R, phi, z, vR, vphi, vz)

    # rescale
    d = d / u_d
    theta = theta / u_ang
    phi = phi / u_ang
    vlos = vlos / u_v
    vth = vth / u_v
    vphi = vphi / u_v

    # stack
    X = np.stack((d, theta, phi, vth, vphi), axis=-1)
    y = vlos[:, None]

    # discard missing RVs
    N_tot = d.size
    N_RVs = int(d.size / 2)
    RV_mask = np.zeros(N_tot, dtype=bool)
    RV_inds = rng.choice(np.arange(N_tot), size=N_RVs, replace=False)
    RV_mask[RV_inds] = True
    X_RV = X[RV_mask]
    y_RV = y[RV_mask]
    X_noRV = X[~RV_mask]
    y_noRV = y[~RV_mask]

    # train/test split
    N_train = int(N_RVs * 0.8)
    train_mask = np.zeros(N_RVs, dtype=bool)
    train_inds = rng.choice(np.arange(N_RVs), size=N_train, replace=False)
    train_mask[train_inds] = True
    X_train = X_RV[train_mask]
    y_train = y_RV[train_mask]
    X_test = X_RV[~train_mask]
    y_test = y_RV[~train_mask]

    # set up torch datasets
    data_noRV = RVDataset(X_noRV, y_noRV)
    data_train = RVDataset(X_train, y_train)
    data_test = RVDataset(X_test, y_test)

    return data_noRV, data_train, data_test


def train_nn(data_train, data_test):

    # set up neural network
    model = nn.Sequential(
        nn.Linear(5, 128),
        nn.BatchNorm1d(128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.Sigmoid(),
        nn.Linear(32, 1),
    )

    # various objects for training: data loader, loss, optimiser
    loader = DataLoader(data_train, batch_size=5000, shuffle=True)
    lossfn = nn.MSELoss()
    optm = Adam(model.parameters(), lr=0.01)
    sched = ReduceLR(optm, factor=0.5, patience=5,
                     min_lr=2e-6, threshold=1e-6, cooldown=5)

    # evaluate test loss before training
    model.eval()
    test_loss = lossfn(model(data_test.X), data_test.y)
    print(f'Pre-training test loss : {test_loss}', flush=True)

    # training loop
    N_epochs = 100
    for epoch in range(N_epochs):
        epoch_loss = 0
        for batch in loader:
            X, y = batch[0], batch[1]
            loss = train_epoch(model, X, y, optm, lossfn)
            epoch_loss += loss / len(loader)

        sched.step(epoch_loss)
        lr = optm.param_groups[0]['lr']
        print(f'Epoch {epoch+1} training loss : {epoch_loss}, lr: {lr:.3e}')

        model.eval()
        test_loss = lossfn(model(data_test.X), data_test.y)
        print(f'Epoch {epoch+1} test loss : {test_loss}')

    return model


def train_epoch(model, X, y, optimizer, criterion):
    model.train()
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss


class RVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        return

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def recombine_data(y_noRV, data_noRV, data_train, data_test, u_d, u_ang, u_v):

    # combine all data
    y = torch.vstack((data_train.y, data_test.y, y_noRV)).detach().numpy().astype(np.float64)
    X = torch.vstack((data_train.X, data_test.X, data_noRV.X)).detach().numpy().astype(np.float64)

    # unroll
    ds, thetas, phis, vths, vphis = X.T
    vlos = y[:, 0]

    # rescale
    ds = ds * u_d
    thetas = thetas * u_ang
    phis = phis * u_ang
    vths = vths * u_v
    vphis = vphis * u_v
    vlos = vlos * u_v

    # back to heliocentric cartesians
    xs = ds * np.sin(thetas) * np.cos(phis)
    ys = ds * np.sin(thetas) * np.sin(phis)
    zs = ds * np.cos(thetas)
    vxs, vys, vzs = vsph_to_vcart(vlos, vths, vphis, ds, thetas, phis)

    # to galactocentric cartesians
    x = xs + 8 * kpc
    y = np.copy(ys)
    z = zs + 0.01 * kpc
    vx = vxs - 10000
    vy = vys + 11000
    vz = vzs + 7000

    # cylindricals
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    vR = vx * np.cos(phi) + vy * np.sin(phi)
    vphi = -vx * np.sin(phi) + vy * np.cos(phi)

    return R, phi, z, vR, vphi, vz


if __name__ == '__main__':

    # load fiducial dataset
    R, phi, z, vR, vphi, vz = load_data()

    # data rescaling units
    u_d = 3 * kpc
    u_ang = pi
    u_v = 400000

    # split data into no RV, training and testing torch Datasets
    data_noRV, data_train, data_test = make_datasets(
        R, phi, z, vR, vphi, vz,
        u_d, u_ang, u_v
    )

    # train NN on data
    model = train_nn(data_train, data_test)

    # guess missing RVs
    y_noRV = model(data_noRV.X)

    # recombine all data with missing RVs, return as galactocentric cyls.
    R, phi, z, vR, vphi, vz = recombine_data(
        y_noRV,
        data_noRV, data_train, data_test,
        u_d, u_ang, u_v
    )

    # save
    np.savez("test_RV/data", R=R, z=z, vR=vR, vphi=vphi, vz=vz)
