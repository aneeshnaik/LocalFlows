import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from os.path import exists

sys.path.append("../src")
from ml import load_flow_ensemble, calc_DF_ensemble
from qdf import create_qdf_ensemble, create_MW_potential
from qdf import calc_DF_ensemble as calc_DF_true
from constants import kpc, pi,pc
from scipy.integrate import trapezoid as trapz


def calc_S(q):

    R_arr = q[:, 0]
    z_arr = q[:, 2]
    N_pts = R_arr.shape[0]

    # set up array of phi
    N_phi = 1000
    phi_min = -pi / 25
    phi_max = pi / 25
    phi_arr = torch.linspace(phi_min, phi_max, N_phi)
    dphi = np.diff(phi_arr)[0]
    Dphi = phi_max - phi_min

    # stack into arrays
    phi_tiled = phi_arr[None].tile([N_pts, 1])
    R_tiled = R_arr[:, None].tile([1, N_phi])
    z_tiled = z_arr[:, None].tile([1, N_phi])

    # distance at each position
    x_GC = R_tiled * torch.cos(phi_tiled)
    y_GC = R_tiled * torch.sin(phi_tiled)
    z_GC = z_tiled
    x_HC = x_GC - 8 * kpc
    y_HC = y_GC
    z_HC = z_GC - 0.01 * kpc
    d = torch.sqrt(x_HC**2 + y_HC**2 + z_HC**2)
    d_pc = d / pc

    # limiting absolute mag
    x_lim = 20 - 5 * torch.log10(d_pc)

    # cap this to 12
    x_lim[x_lim > 12] = 12

    # observed fraction
    alpha = 0.55
    x1 = -5
    x2 = 12
    eax1 = np.exp(alpha * x1)
    eax2 = np.exp(alpha * x2)
    p_obs = (torch.exp(alpha * x_lim) - eax1) / (eax2 - eax1)

    # integrate to S
    S = (torch.sum(p_obs, dim=-1) * dphi) / Dphi
    return S


def calc_DF_model(q, p, u_q, u_p, q_cen, p_cen, flows):
    S = calc_S(torch.tensor(q))
    f_obs = calc_DF_ensemble(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_true = f_obs / S
    return f_true


def normalise_DF(f, x1, x2):
    """
    Return normalisation of 2D PDF in x1-x2 space, defined by 1D arrays x12.
    """
    N = np.size(x1)
    norm = trapz(np.array([trapz(f[:, i], x1) for i in range(N)]), x2)
    return norm


# set up coordinate arrays
N_px = 128
ones = np.ones((N_px, N_px))
zeros = np.zeros((N_px, N_px))
R0 = 8 * kpc
z0 = 0.
vR0 = 0.
vphi0 = 220000.
vz0 = 0.

Rlim = 0.4 * kpc
zlim = 0.4 * kpc
vlim = 80000
R_arr = np.linspace(R0 - Rlim, R0 + Rlim, N_px)
z_arr = np.linspace(-zlim, zlim, N_px)
vR_arr = np.linspace(vR0 - vlim, vR0 + vlim, N_px)
vphi_arr = np.linspace(vphi0 - vlim, vphi0 + vlim, N_px)
vz_arr = np.linspace(vz0 - vlim, vz0 + vlim, N_px)

datafile = "DF_alt_data.npz"
if not exists(datafile):


    
    
    # load flow ensemble
    flows = load_flow_ensemble(
        flowdir='../nflow_models/maglim/', 
        inds=np.arange(13), n_dim=5, n_layers=8, n_hidden=64)
    
    # load qDFs
    fname = "../data/MAPs.txt"
    data = np.loadtxt(fname, skiprows=1)
    weights = data[:, 2]
    hr = data[:, 3] / 8
    sr = data[:, 4] / 220
    sz = sr / np.sqrt(3)
    hsr = np.ones_like(hr)
    hsz = np.ones_like(hr)
    mw = create_MW_potential(ddtype=0) 
    qdfs = create_qdf_ensemble(hr, sr, sz, hsr, hsz, pot=mw)
    
    # flow arguments
    u_q = kpc
    u_p = 100000
    q_cen = np.array([8 * kpc, 0, 0.01 * kpc])
    p_cen = np.array([0, 220000, 0])
    
    # R-z: evaluate DF
    R_grid, z_grid = np.meshgrid(R_arr, z_arr, indexing='ij')
    q = np.stack((R_grid, zeros, z_grid), axis=-1)
    p = np.stack((vR0 * ones, vphi0 * ones, vz0 * ones), axis=-1)
    q = q.reshape((N_px**2, 3))
    p = p.reshape((N_px**2, 3))
    f_model = calc_DF_model(q, p, u_q, u_p, q_cen, p_cen, flows)
    f_model = f_model.reshape((N_px, N_px))
    f_true = calc_DF_true(q, p, qdfs, weights)
    f_true = f_true.reshape((N_px, N_px))
    f_true[np.abs(R_grid - R0) > 1 * kpc] = 0
    
    # normalise
    norm_true = normalise_DF(f_true, R_arr, z_arr)
    norm_model = normalise_DF(f_model, R_arr, z_arr)
    f_true /= norm_true
    f_model /= norm_model
    
    
    # ref value
    f_ref = calc_DF_true(q_cen, p_cen, qdfs, weights) / norm_true
    f1_model = f_model / f_ref
    f1_true = f_true / f_ref
    
    # calculate residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        res1 = np.divide((f1_model - f1_true), f1_true)
    
    np.savez(datafile, f_model=f1_model, f_true=f1_true, res=res1)

data = np.load(datafile)
f_model = data['f_model']
f_true = data['f_true']
res = data['res']


Rmin = (R0 - Rlim) / kpc
Rmax = (R0 + Rlim) / kpc
zmin = -zlim / kpc
zmax = zlim / kpc
extent = [Rmin, Rmax, zmin, zmax]

plt.imshow(res.T, origin='lower', cmap='Spectral', vmin=-0.1, vmax=0.1, extent=extent)
