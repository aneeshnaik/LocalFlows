# LocalFlows


## Summary

This code was used to generate the results in the article Naik, An, Burrage, and Evans (2021). An earlier paper in the series ([An et al., 2021](https://arxiv.org/abs/2106.05981)) described a new method for calculating gravitational accelerations from a known stellar distribution function. Naik et al. (2021) then applied the method to mock data representing stars in the solar neighbourhood.

The method is in two stages:
1. We use normalising flows to 'learn' the distribution function of a 6D mock dataset. Here, our code is built around the implementation of masked autoregressive flows in the package `nflows`.
2. We then convert these learned DFs to accelerations using our new technique.

Please see our two papers for more details about the technique.

## Citation

Our code is freely available for use under the MIT License. For details, see LICENSE.

If using our code, please cite our two papers, [An et al. (2021)](https://arxiv.org/abs/2106.05981) and Naik et al. (2021). 

Additionally, if using the normalising flow part of our code to learn DFs, please also consider citing the package `nflows` ([link](https://github.com/bayesiains/nflows)) around which that part of our code is built, as well as the article by [Green and Ting (2020)](https://arxiv.org/abs/2011.04673) in which the idea was first proposed.

## Structure

This code is structured as follows:
- `/src` contains all of the 'source code', including code underlying the normalising flows, the stellar distribution functions used to generate the mock data, the conversion to accelerations, and various other utility functions etc.
- `/data` contains the mock datasets, as well as the scripts used to generate them.
- `/flows` contains the trained normalising flow models, as well as the scripts used to generate them.
- `/figures` contains the plotting scripts used for the paper.

The various subsections below describe these components in further detail.

### `/data`

This directory contains the mock datasets and the scripts used to generate the mock datasets.

- `sample_data_fiducial.py`: To be run with an integer argument `i`; samples 10000 from the DF described in our paper using random seed `i`. The positions and velocities of these stars are then stored as a numpy zipped file in `/data/fiducial/i.npz`. For our analysis, we ran this 2000 times, with seeds 0-1999 inclusive, thus generating 20 million stars between 1 and 16 kpc.
- `perturb_data.py`: Also to be run with integer argument `i`; takes stars in `/data/fiducial/{i}.npz` and 'perturbs' them (see Sec 4.2 of our paper), then saves the perturbed stars in `/data/perturbed_t0/{i}.npz`. Again, we ran this 2000 times, `i` between 0-1999.
- `evolve_data.py`: Also to be run with integer argument `i`; takes perturbed stars in `/data/perturbed_t0/{i}.npz`, evolves their orbits and stores snapshots at 200 Myr and 500 Myr in `/data/perturbed_t2/{i}.npz` and `/data/perturbed_t5/{i}.npz` respectively. Again, we ran this 2000 times, `i` between 0-1999.
- `split_RV_datasets.py`: For the missing radial velocity test in Figure 3 of the paper. Given the ~1 million stars in the fiducial dataset between 7 and 9 kpc, store their 2D position data in `/data/test_RV/full_dset.npz`, then randomly choose half of them, keep 5D position+velocity data in `/data/test_RV/RV_dset.npz`.


### `/figures`

This directory contains the four figures from our paper (in .pdf format), along with the scripts used to generate them. The general file pattern is:
- `fign_x.py`
- `fign_x.pdf`
- `fign_data.npz`

Here, `x` is some brief descriptor describing the figure. The python script generates both the `.npz` file containing the figure data and the figure itself. On subsequent runs, the script will find the `.npz` file it has previously created and generate the figure directly from the saved figure data, saving the trouble of generating the data anew.

These scripts give different example use cases for how to read and analyse the normalising flow models and how to convert their learned DFs into accelerations.
