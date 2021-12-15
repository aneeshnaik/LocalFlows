# LocalFlows


## Summary

This code was used to generate the results in the article [Naik, An, Burrage, and Evans (2021)](https://arxiv.org/abs/2112.07657): Charting galactic accelerations II: how to 'learn' accelerations in the solar neighbourhood. An earlier paper in the series ([An et al., 2021](https://arxiv.org/abs/2106.05981)) described a new method for calculating gravitational accelerations from a known stellar distribution function. Naik et al. (2021) then applied the method to mock data representing stars in the solar neighbourhood.

The method is in two stages:
1. We use normalising flows to 'learn' the distribution function of a 6D mock dataset. Here, our code is built around the implementation of masked autoregressive flows in the package `nflows`.
2. We then convert these learned DFs to accelerations using our new technique.

Please see our two papers for more details about the technique.

As described below, the mock datasets used to generate the results in the paper are not stored in this repository, but [here](https://doi.org/10.5281/zenodo.5781350). Alternatively, the scripts in the `/data` folder can be used to generate the mock datasets anew.

## Citation

Our code is freely available for use under the MIT License. For details, see LICENSE.

If using our code, please cite our two papers, [An et al. (2021)](https://arxiv.org/abs/2106.05981) and [Naik et al. (2021)](https://arxiv.org/abs/2112.07657). 

Additionally, if using the normalising flow part of our code to learn DFs, please also consider citing the package `nflows` ([link](https://github.com/bayesiains/nflows)) around which that part of our code is built, as well as the article by [Green and Ting (2020)](https://arxiv.org/abs/2011.04673) in which the idea was first proposed.


## Structure

This code is structured as follows:
- `/src` contains all of the 'source code', including code underlying the normalising flows, the stellar distribution functions used to generate the mock data, the conversion to accelerations, and various other utility functions etc.
- `/data` contains the mock datasets, as well as the scripts used to generate them.
- `/flows` contains the trained normalising flow models, as well as the scripts used to generate them.
- `/figures` contains the plotting scripts used for the paper.

The various subsections below describe these components in further detail.

### `/src`

This directory contains all of the 'source code' underlying the project. There are 7 scripts within this section:

- `constants.py`: definitions of various useful physical constants and unit conversions
- `qdf.py`: various functions relating to the quasi-isothermal DF (qDF) used to sample the mock data
- `utils.py`: various utility functions
- `cbe.py`: function to calculate acceleration from known DF (cf. eq. 4 in our paper)
- `ml.py`: various functions relating to setting up, training, and evaluating the normalizing flows
- `comparison_jeans`: implementation of the Jeans analysis procedure used in our methods comparison section (Sec. 5, see also Appendix A for technicals)
- `comparison_DF1D`: implementation of the 1D DF-fitting procedure used in our methods comparison section (Sec. 5, see also Appendix B for technicals)


### `/data`

This directory contains the mock datasets and the scripts used to generate the mock datasets.

- `sample_fiducial.py`: To be run with an integer argument `i`; samples 10000 from the DF described in our paper using random seed `i`. The positions and velocities of these stars are then stored as a numpy zipped file in `/data/fiducial/i.npz`. For our analysis, we ran this 2000 times, with seeds 0-1999 inclusive, thus generating 20 million stars between 1 and 16 kpc.
- `perturb.py`: Also to be run with integer argument `i`, and only after `sample_fiducial.py`; takes stars in `/data/fiducial/{i}.npz` and 'perturbs' them (see Sec 4.2 of our paper), then saves the perturbed stars in `/data/perturbed_t0/{i}.npz`. Again, we ran this 2000 times, `i` between 0-1999.
- `evolve.py`: Also to be run with integer argument `i`, and only after `perturb.py`; takes perturbed stars in `/data/perturbed_t0/{i}.npz`, evolves their orbits and stores snapshots at 200 Myr and 500 Myr in `/data/perturbed_t2/{i}.npz` and `/data/perturbed_t5/{i}.npz` respectively. Again, we ran this 2000 times, `i` between 0-1999.
- `concat.py`: Only to be run after `evolve.py`: In each of the subdirectories `fiducial`, `perturbed_t0`, `perturbed_t2`, concatenate the 2000 individual sample files into single `dset.npz` files.
- `split_RV.py`: For the missing radial velocity test in Figure 3 of the paper. Given the ~1 million stars in the fiducial dataset between 7 and 9 kpc, store their 2D position data in `/data/test_RV/full_dset.npz`, then randomly choose half of them, keep 5D position+velocity data in `/data/test_RV/RV_dset.npz`.

Before running these scripts, one has to create the following empty subdirectories within `data`:
- `fiducial`
- `perturbed_t0`
- `perturbed_t2`
- `perturbed_t5`
- `test_RV`

After running the scripts in sequence, these subdirectories are all filled with datasets in the form of `.npz` files. These filled subdirectories have been left out of this git repository to avoid bloating the repo. As an alternative to generating the data from scratch, you can download the pre-made datasets directly from [Zenodo](https://doi.org/10.5281/zenodo.5781350), where the 5 directories above are stored as `.zip` files. Simply unzip them and place the resulting directories within `/data`.


### `/flows`

This directory contains the trained normalising flow models, as well as the scripts used to generate them. The four subdirectories `fiducial`, `perturbed_t0`, `perturbed_t2`, `perturbed_t5` follow the same basic pattern: the script `train_flow.py` within the subdirectory loads the corresponding mock data from `/data`, then trains a normalising flow using various functions from `src/ml.py`. `train_flow.py` takes an integer argument `i`, which is used as a random seed for initialising the parameters of the flow. At the end of the run, the flow is saved as `{i}_best.pth` (the losses are also saved as `{i}_losses.npy`). In each case, we trained an ensemble of 20 flows.

Things are a bit more complicated for `test_errors` and `test_RV`. For `test_errors`, the data are shifted by random amounts before training the flows. This is all performed within `test_errors/train_flow.py`. Meanwhile, in `test_RV`, two ensembles of flows are trained, one for the 2D dataset in `/data/test_RV`, and one for the 5D dataset there. The runscripts for these are respectively `test_RV/test_nu.py` and `test_RV/test_pvx.py`.


### `/figures`

This directory contains the four figures from our paper (in .pdf format), along with the scripts used to generate them. The general file pattern is:
- `fign_x.py`
- `fign_x.pdf`
- `fign_data.npz`

Here, `x` is some brief descriptor describing the figure. The python script generates both the `.npz` file containing the figure data and the figure itself. On subsequent runs, the script will find the `.npz` file it has previously created and generate the figure directly from the saved figure data, saving the trouble of generating the data anew.

These scripts give different example use cases for how to read and analyse the normalising flow models and how to convert their learned DFs into accelerations.


## Prerequisites

This section lists the various dependencies of our code. The version number in parenthesis is not (necessarily) a requirement, but simply the version of a given library we used at the time of publication. Earlier/later versions of a library could work equally well.

- `numpy` (1.20.3)
- `pytorch` (1.9.0)
- `galpy` (1.7.0)
- `tqdm` (4.62.1)
- `emcee` (3.1.1)
- `matplotlib` (3.4.2)
- `nflows` (0.14)
- `scipy` (1.7.1)
