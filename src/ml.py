#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to training and analysing the normalising flows.

Created: June 2021
Author: A. P. Naik
"""
import numpy as np
import copy
from time import perf_counter as time

import torch
from torch.utils.data import DataLoader as DL, TensorDataset as TDS
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms import ReversePermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform as MAAT


def setup_MAF(n_dim, n_layers, n_hidden):
    """
    Set up masked autoregressive flow.

    Parameters
    ----------
    n_dim : int
        Number of flow input dimensions, i.e. dimensionality of dataset.
    n_layers : int
        Number of transformations along the flow.
    n_hidden : int
        Number of hidden layers in each transformation.

    Returns
    -------
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows.

    """
    # base distribution
    base_dist = StandardNormal(shape=[n_dim])

    # loop over layers in flow
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=n_dim))
        transforms.append(MAAT(features=n_dim, hidden_features=n_hidden))

    # assemble flow
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def load_flow(state_dict, n_dim, n_layers, n_hidden):
    """
    Load flow from state dict.

    Parameters
    ----------
    state_dict : str
        Path to pytorch state dict (e.g. .pth) file.
    n_dim : int
        Number of flow input dimensions, i.e. dimensionality of dataset.
    n_layers : int
        Number of transformations along the flow.
    n_hidden : int
        Number of hidden layers in each transformation.

    Returns
    -------
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows.

    """
    flow = setup_MAF(n_dim=n_dim, n_layers=n_layers, n_hidden=n_hidden)
    flow.load_state_dict(torch.load(state_dict))
    flow.eval()
    return flow


def load_flow_ensemble(flowdir, inds, n_dim, n_layers, n_hidden):
    """
    Load ensemble of flows from array of state dicts.

    Individual flows are expected to be saved as state dicts with filenames
    'n_best.pth', where x is some integer.

    Parameters
    ----------
    flowdir : str
        Directory in which state dicts are saved
    inds : array-like
        Indices of flows, i.e. [n1, n2, n3, ...], where flows are saved as
        'n1_best.pth' etc.
    n_dim : int
        Number of flow input dimensions, i.e. dimensionality of dataset.
    n_layers : int
        Number of transformations along the flow.
    n_hidden : int
        Number of hidden layers in each transformation.

    Returns
    -------
    flows : list
        List containing normalising flows, i.e. instances of Flow object from
        nflows.flows.
    """
    # check if dir ends in '/', otherwise append
    if flowdir[-1] != '/':
        flowdir += '/'

    # loop over inds and load flows
    flows = []
    for i in inds:
        fname = flowdir + f"{i}_best.pth"
        flows.append(load_flow(fname, n_dim, n_layers, n_hidden))
    return flows


def calc_total_loss(flow, data):
    """
    Compute total loss of model on data.

    Parameters
    ----------
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows, generated
        from e.g. load_flow() or setup_MAF().
    data : torch.Tensor, shape (N, n_dim)
        Torch tensor containing data on which to evaluate flow loss. Tensor
        should have shape (N, n_dim), where N is number of data points and
        n_dim is number of input dims of flow.

    Returns
    -------
    loss : float
        Model loss evaluated on given dataset.

    """
    flow.eval()
    with torch.no_grad():
        loss = -flow.log_prob(inputs=data, context=None).mean().item()
    return loss


def train_epoch(flow, data, batch_size, optimiser, scheduler, total_losses):
    """
    Train flow model on data for one epoch.

    Parameters
    ----------
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows, generated
        from e.g. load_flow() or setup_MAF().
    data : torch.Tensor, shape (N, n_dim)
        Torch tensor containing data on which to evaluate flow loss. Tensor
        should have shape (N, n_dim), where N is number of data points and
        n_dim is number of input dims of flow.
    batch_size : int
        Number of data points in batch.
    optimiser : torch.optim Optimizer
        A pytorch Optimizer instance, e.g. Adam
    scheduler : torch..optim.lr_scheduler Scheduler
        A scheduler from torch.optim.lr_scheduler, e.g. ReduceLROnPlateau.

    Returns
    -------
    loss : float
        Model loss evaluated on given dataset.

    """
    # set to training mode
    flow.train()

    # make data loader for training data
    loader = DL(TDS(data), batch_size=batch_size, shuffle=True)

    # loop over batches in data
    losses = np.array([])
    for batch_idx, batch in enumerate(loader):
        batch = batch[0]
        optimiser.zero_grad()
        loss = -flow.log_prob(inputs=batch, context=None).mean()
        loss.backward()
        optimiser.step()
        losses = np.append(losses, loss.item())

    # compute total loss at end of epoch
    if total_losses:
        loss = calc_total_loss(flow, data)
    else:
        loss = np.mean(losses)

    # step the lr scheduler
    scheduler.step(loss)

    return loss


def calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, flow):
    """
    Evaluate model DF at given phase positions from single flow.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    u_q : float
        Rescaling units for positions (i.e. rescaling used to train flow).
    u_p : float
        Rescaling units for velocities (i.e. rescaling used to train flow).
    q_cen : np.array or torch.tensor, shape (3)
        Position centre. UNITS: m.
    p_cen : np.array or torch.tensor, shape (3)
        Velocity centre. UNITS: m/s.
    flow : nflows.flows.Flow
        Normalising flow, instance of Flow object from nflows.flows, generated
        from e.g. load_flow() or setup_MAF().

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.

    """
    # check shapes match
    assert q.shape == p.shape

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
        p = p[None]
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)
        p = torch.tensor(p)

    # rescale units
    q = (q - torch.tensor(q_cen)) / u_q
    p = (p - torch.tensor(p_cen)) / u_p

    # concat
    eta = torch.cat((q[:, [0, 2]], p), dim=-1).float()

    # eval f from flow
    f = flow.log_prob(eta).exp()

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_DF_ensemble(q, p, u_q, u_p, q_cen, p_cen, flows):
    """
    Evaluate model DF at given phase positions from ensemble of flows.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    u_q : float
        Rescaling units for positions (i.e. rescaling used to train flow).
    u_p : float
        Rescaling units for velocities (i.e. rescaling used to train flow).
    q_cen : np.array or torch.tensor, shape (3)
        Position centre. UNITS: m.
    p_cen : np.array or torch.tensor, shape (3)
        Velocity centre. UNITS: m/s.
    flow : list of nflows.flows.Flow objects
        List of normalising flows, each is an instance of Flow object from
        nflows.flows, generated from e.g. load_flow() or setup_MAF().

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.

    """
    # loop over flows
    N = len(flows)
    for i in range(N):
        if i == 0:
            f = calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, flows[i]) / N
        else:
            f = f + calc_DF_single(q, p, u_q, u_p, q_cen, p_cen, flows[i]) / N
    return f


def train_flow(data, seed, n_dim=5, n_layers=8, n_hidden=64,
               lr=1e-3, lrgamma=0.5, lrpatience=5,
               lrmin=2e-6, lrthres=1e-6, lrcooldown=10,
               weight_decay=0, batch_size=10000, num_epochs=500,
               save_intermediate=False, save_interval=10, cut_early=True,
               total_losses=True):
    """
    Train normalising flow on data.

    Parameters
    ----------
    data : torch.Tensor, shape (N, n_dim)
        Torch tensor containing data on which to evaluate flow loss. Tensor
        should have shape (N, n_dim), where N is number of data points and
        n_dim is number of input dims of flow, matching 'n_dim' argument.
    seed : int
        Random seed to initialise flow.
    n_dim : int
        Number of flow input dimensions, i.e. dimensionality of dataset. The
        default is 5.
    n_layers : int
        Number of transformations along the flow. The default is 8.
    n_hidden : int
        Number of hidden layers in each transformation. The default is 64.
    lr : float, optional
        Initial learning rate. The default is 1e-3.
    lrgamma : float, optional
        Factor by which learning rate is decreased on plateau. The default is
        0.5.
    lrpatience : float, optional
        Number of training epochs to wait on plateau before reducing lr. The
        default is 5.
    lrmin : float, optional
        Minimum learning rate beyond which to decrease no further.
        Additionally, if 'cut_early' is True then the training is truncated
        when the learning rate hits lrmin. The default is 2e-6.
    lrthres : float, optional
        Threshold for measuring the new optimum, to only focus on significant
        changes. The default is 1e-6.
    lrcooldown : int, optional
        Number of training epochs to wait before resuming normal operation
        (i.e. resuming counting bad epochs) after reducing the learning rate.
        The default is 10.
    weight_decay : float, optional
        Weight decay for L2 regularisation. The default is 0.
    batch_size : int, optional
        Batch size for gradient descent. The default is 10000.
    num_epochs : int, optional
        Number of training epochs to loop over. If 'cut_early' is True then
        the training can stop before num_epochs if the learning rate hits
        lrmin. The default is 500.
    save_intermediate : bool, optional
        Whether to save snapshots of flow before the training has finished. The
        frequency of saves if set by 'save_interval' below. The default is
        False.
    save_interval : int, optional
        If 'save_intermediate' is True, the number of epochs between each save
        file. The default is 10.
    cut_early : bool, optional
        If True, training is truncated early (i.e. before num_epochs) if
        learning rate reaches lrmin. The default is True.
    total_losses : bool, optional
        If True, total (i.e. full dataset) losses are calculated at end of each
        epoch. Otherwise, losses are averaged across epoch. The default is True.

    Returns
    -------
    None.

    """
    # set RNG seed
    torch.manual_seed(seed)

    # set up MAF
    flow = setup_MAF(n_dim=n_dim, n_layers=n_layers, n_hidden=n_hidden)

    # set up optimiser
    optimiser = Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLR(optimiser, factor=lrgamma, patience=lrpatience,
                         min_lr=lrmin, threshold=lrthres, cooldown=lrcooldown)

    # compute loss pre-training
    losses = np.zeros(num_epochs + 1)
    if total_losses:
        loss = calc_total_loss(flow, data)
    else:
        flow.eval()
        loader = DL(TDS(data), batch_size=batch_size, shuffle=True)

        # loop over batches in data
        loss_arr = np.array([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = batch[0]
                loss = -flow.log_prob(inputs=batch, context=None).mean().item()
                loss_arr = np.append(loss_arr, loss)
        loss = np.mean(loss_arr)
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
        loss = train_epoch(flow, data, batch_size, optimiser, scheduler, total_losses)
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

        # save model if epoch = interval
        if save_intermediate and ((epoch + 1) % save_interval == 0):
            torch.save(flow.state_dict(), f'{seed}_{epoch+1}.pth')

        # stop loop if we've reached min learning rate
        if cut_early and lr <= lrmin:
            break

    # save best model and loss history
    torch.save(best_model.state_dict(), f'{seed}_best.pth')
    np.save(f'{seed}_losses', losses)
    return
