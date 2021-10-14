# LocalFlows


## Summary

This code was used to generate the results in the article Naik, An, Burrage, and Evans (2021). An earlier paper in the series (An et al, 2021) described a new method for calculating gravitational accelerations from a known stellar distribution function. Naik et al. (2021) then applied the method to mock data representing stars in the solar neighbourhood.

The method is in two stages:
1. We use normalising flows to 'learn' the distribution function of a 6D mock dataset. Here, our code is built around the implementation of masked autoregressive flows in the package `nflows`.
2. We then convert these learned DFs to accelerations using our new technique.

Please see our paper and (An et al, 2021) for more details about the technique.

## Citation

Our code is freely available for use under the MIT License. For details, see LICENSE.

If using our code, please cite our two papers, An et al. (2021) and Naik et al. (2021). 

Additionally, if using the normalising flow part of our code to learn DFs, please also consider citing the package `nflows` ([link](https://github.com/bayesiains/nflows)) around which that part of our code is built, as well as the article by Green & Ting (2020) ([link](https://arxiv.org/abs/2011.04673)) in which the idea was first proposed.

## Structure

This code is structured as follows:
- `/data` contains the mock datasets, as well as the scripts used to generate them.
- `/figures` contains the plotting scripts used for the paper.
- `/flows` contains the trained normalising flow models, as well as the scripts used to generate them.
- `/src` contains all of the 'source code', including code underlying the normalising flows, the stellar distribution functions used to generate the mock data, the conversion to accelerations, and various other utility functions etc. 
