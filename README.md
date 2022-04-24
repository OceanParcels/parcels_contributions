# Parcels Contributions Repository
Repository where users can contribute Kernels and code snippets

## What can you do in this repo.?
* Browse frequent questions, if you have a specific question, to see if an answer exists: https://github.com/OceanParcels/parcels_contributions/wiki/Frequently-Asked-Questions-(FAQ)
* Browse available code snippets from experienced Parcels users (and group members)
* Provide a code snipped (via PR) - either if you have a generally-applicable short code people frequently reuse, or you you want to provide a snippet to a still-open question
* Provide a question (if not already existent) via the Parcels main issue tracker: https://github.com/OceanParcels/parcels/issues

## How to add a kernel or code snippet?
Please do so via a pull request.

## Dependencies

### General

* Numerical libraries: numpy, scipy 
* Libraries for accessing NetCDF and HDF5 data: netcdf4, xarray, h5py 
* Libraries for plotting: matplotlib, cartopy

### Kernels and Examples

Those scripts represent either novel application kernels or more extensive example scripts for Parcels simulations, and thus require [parcels](https://github.com/OceanParcels/parcels) as a package.

* Libraries for simulations: parcels (on conda-forge)


### VTK transform script

For running the 'transform_to_vtk.py' scripts in the _Snippets_ folder, you need to get the VTK library (also available via conda):

* visualisation library: vtk (on anaconda and conda-forge)
