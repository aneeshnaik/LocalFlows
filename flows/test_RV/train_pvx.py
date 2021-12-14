#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train normalising flow.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import copy
from time import perf_counter as time

import torch
from torch import nn
from torch.utils.data import DataLoader as DL, TensorDataset as TDS
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR

from nflows.flows import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal as CDN
from nflows.transforms import CompositeTransform
from nflows.transforms import ReversePermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform as MAAT

sys.path.append("../../src")
from constants import kpc


def setup_flow(n_layers, n_hidden):

    # base distribution
    base_dist = CDN(shape=[3], context_encoder=nn.Linear(2, 6))

    # loop over layers
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=3))
        transforms.append(MAAT(
            features=3, hidden_features=n_hidden, context_features=2)
        )

    # assemble flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def train_epoch(flow, data, batch_size, optimiser, scheduler):
    # set to training mode
    flow.train()

    # make data loader for training data
    loader = DL(TDS(data), batch_size=batch_size, shuffle=True)

    # loop over batches in data
    losses = np.array([])
    for batch_idx, batch in enumerate(loader):
        batch = batch[0]
        optimiser.zero_grad()
        loss = -flow.log_prob(inputs=batch[:, 2:], context=batch[:, :2]).mean()
        loss.backward()
        optimiser.step()
        losses = np.append(losses, loss.item())

    # compute total loss at end of epoch
    loss = calc_total_loss(flow, data)

    # step the lr scheduler
    scheduler.step(loss)

    return loss


def calc_total_loss(flow, data):
    flow.eval()
    with torch.no_grad():
        loss = -flow.log_prob(inputs=data[:, 2:], context=data[:, :2]).mean().item()
    return loss


def train_flow(data, seed, n_layers=8, n_hidden=64,
               lr=1e-3, lrgamma=0.5, lrpatience=5,
               lrmin=2e-6, lrthres=1e-6, lrcooldown=10,
               weight_decay=0, batch_size=10000, num_epochs=500):

    # set RNG seed
    torch.manual_seed(seed)

    # set up MAF
    flow = setup_flow(n_layers=n_layers, n_hidden=n_hidden)

    # set up optimiser
    optimiser = Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLR(optimiser, factor=lrgamma, patience=lrpatience,
                         min_lr=lrmin, threshold=lrthres, cooldown=lrcooldown)

    # compute loss pre-training
    losses = np.zeros(num_epochs + 1)
    loss = calc_total_loss(flow, data)
    losses[0] = loss
    print(f"Pre-training total loss={loss:.6e}", flush=True)

    # train; loop over epochs
    best_epoch = -1
    best_loss = np.inf
    best_model = copy.deepcopy(flow)
    for epoch in range(num_epochs):

        # start stopclock
        t0 = time()

        # print start-of-epoch message
        lr = optimiser.param_groups[0]['lr']
        print(f'\nStarting epoch {epoch}; lr={lr:.4e}', flush=True)

        # train 1 epoch
        loss = train_epoch(flow, data, batch_size, optimiser, scheduler)
        losses[epoch + 1] = loss

        # if best so far, save model
        if loss < best_loss:
            best_epoch = epoch
            prevbest_loss = best_loss
            best_loss = loss
            best_model = copy.deepcopy(flow)

        # construct end-of-epoch message
        str = f"Finished epoch {epoch}; total loss={loss:.6e}"
        if epoch != 0:
            if best_epoch == epoch:
                f = (best_loss - prevbest_loss) / np.abs(prevbest_loss)
                str += f"\nBest epoch so far. Cf prev. best: dL/L={f:.2e}"
            else:
                f = (loss - best_loss) / np.abs(best_loss)
                str += f"\nBest epoch was {best_epoch}. Cf best: dL/L={f:.2e}"
        t1 = time()
        t = t1 - t0
        str += f"\nNum bad epochs: {scheduler.num_bad_epochs}"
        str += f"\nTime taken for epoch: {int(t)} seconds"
        print(str, flush=True)

        # stop loop if we've reached min learning rate
        if lr <= lrmin:
            break

    # save best model and loss history
    torch.save(best_model.state_dict(), f'pvx/{seed}_best.pth')
    np.save(f'pvx/{seed}_losses', losses)
    return


if __name__ == '__main__':

    # load data
    data = np.load("../../data/test_RV/RV_dset.npz")
    R = data['R']
    z = data['z']
    vR = data['vR']
    vz = data['vz']
    vphi = data['vphi']

    # shift and rescale positions
    u_pos = kpc
    u_vel = 100000
    cen = np.array([8 * kpc, 0, 0, 220000, 0])
    R = (R - cen[0]) / u_pos
    z = (z - cen[1]) / u_pos
    vR = (vR - cen[2]) / u_vel
    vphi = (vphi - cen[3]) / u_vel
    vz = (vz - cen[4]) / u_vel

    # stack and shuffle data
    data = np.stack((R, z, vR, vphi, vz), axis=-1)
    rng = np.random.default_rng(42)
    rng.shuffle(data)

    # make torch tensor
    data = torch.from_numpy(data.astype(np.float32))

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
