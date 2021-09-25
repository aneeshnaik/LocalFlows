#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample absolute magnitudes from exponential distribution.

Created: September 2021
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt


# params
alpha = 0.55
x1 = -5
x2 = 12

# normalisation
eax1 = np.exp(alpha * x1)
eax2 = np.exp(alpha * x2)
A = alpha / (eax2 - eax1)

# set up RNG
rng = np.random.default_rng(42)

# uniform sample
U = rng.uniform(size=1000000)

# transform to magnitudes
M = np.log(alpha * U / A + eax1) / alpha

N_bins = 200
x_edges = np.linspace(x1, x2, N_bins + 1)
x_cens = 0.5 * (x_edges[1:] + x_edges[:-1])
f_cens = A * np.exp(alpha * x_cens)

plt.hist(M, bins=x_edges, density=True)
plt.plot(x_cens, f_cens)


plt.yscale('log')