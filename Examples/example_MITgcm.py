import math
from datetime import timedelta as delta
from os import path
from glob import glob

import numpy as np
import dask

from parcels import AdvectionRK4
from parcels import Field
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import ScipyParticle
from parcels import Variable

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def fieldset_from_MITgcm(chunk_mode):
    data_path = path.join("/data/oceanparcels/input_data", "MITgcm4km/")
    data_file = path.join(data_path, "RGEMS3_2008_Surf.nc")
    mesh_mask = path.join(data_path, "RGEMS3_Surf_grid.nc")

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'data': data_file},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'data': data_file}}
    variables = {'U': 'UVEL',
                 'V': 'VVEL'}
    dimensions = {'U': {'lon': 'XG', 'lat': 'YG', 'time': 'time'},
                  'V': {'lon': 'XG', 'lat': 'YG', 'time': 'time'}}
    chs = False
    if chunk_mode == 'auto':
        chs = 'auto'
    elif chunk_mode == 'specific':
        chs = {'U': {'time': 1, 'XC': 80, 'XG': 80, 'YC': 60, 'YG': 60},
               'V': {'time': 1, 'XC': 80, 'XG': 80, 'YC': 60, 'YG': 60}}

    fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions, field_chunksize=chs, time_periodic=delta(days=366), tracer_interp_method='cgrid_velocity')
    return fieldset


def fieldset_from_MITgcm_plus_WaveWatch3(chunk_mode):
    data_path_mitgcm = path.join("/data/oceanparcels/input_data", "MITgcm4km/")
    data_file_mitgcm = path.join(data_path_mitgcm, "RGEMS3_2008_Surf.nc")
    mesh_mask_mitgcm = path.join(data_path_mitgcm, "RGEMS3_Surf_grid.nc")

    filenames_mitgcm = {'U': {'lon': mesh_mask_mitgcm, 'lat': mesh_mask_mitgcm, 'data': data_file_mitgcm},
                 'V': {'lon': mesh_mask_mitgcm, 'lat': mesh_mask_mitgcm, 'data': data_file_mitgcm}}
    variables_mitgcm = {'U': 'UVEL',
                 'V': 'VVEL'}
    dimensions_mitgcm = {'U': {'lon': 'XG', 'lat': 'YG', 'time': 'time'},
                  'V': {'lon': 'XG', 'lat': 'YG', 'time': 'time'}}
    chs_mitgcm = False
    if chunk_mode == 'auto':
        chs_mitgcm = 'auto'
    elif chunk_mode == 'specific':
        chs_mitgcm = {'U': {'time': 1, 'XC': 80, 'XG': 80, 'YC': 60, 'YG': 60},
                      'V': {'time': 1, 'XC': 80, 'XG': 80, 'YC': 60, 'YG': 60}}

    fieldset_mitgcm = FieldSet.from_c_grid_dataset(filenames_mitgcm, variables_mitgcm, dimensions_mitgcm, field_chunksize=chs_mitgcm,
                                                   time_periodic=delta(days=366), tracer_interp_method='cgrid_velocity')
    data_path_stokes = path.join("/data/oceanparcels/input_data", "WaveWatch3data", "CFSR")
    data_files_stokes = sorted(glob(data_path_stokes + "/WW3-GLOB-30M_2008*_uss.nc"))
    variables_stokes = {'U': 'uuss',
                        'V': 'vuss'}
    dimensions_stokes = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
                         'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}}
    chs_stokes = False
    if chunk_mode == 'auto':
        chs_stokes = 'auto'
    elif chunk_mode == 'specific':
        chs_stokes = {'time': 1, 'latitude': 32, 'longitude': 16}
    # fieldset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
    fieldset_stokes = FieldSet.from_netcdf(data_files_stokes, variables_stokes, dimensions_stokes, field_chunksize=chs_stokes, time_periodic=delta(days=366))
    fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    fieldset = FieldSet(U=fieldset_mitgcm.U + fieldset_stokes.U, V=fieldset_mitgcm.V + fieldset_stokes.V)
    return fieldset


def compute_MITgcm_particle_advection(field_set, mode, lonp, latp):

    def periodicBC(particle, fieldSet, time):
        dlon = -89.0+91.8
        dlat = 0.7+1.4
        if particle.lon > -89.0:
            particle.lon -= dlon
        if particle.lon < -91.8:
            particle.lon += dlon
        if particle.lat > 0.7:
            particle.lat -= dlat
        if particle.lat < -1.4:
            particle.lat += dlat

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("MITgcm_particles_chunk", pset, outputdt=delta(days=1))
    kernels = pset.Kernel(periodicBC) + pset.Kernel(AdvectionRK4)
    pset.execute(kernels, runtime=delta(days=30), dt=delta(hours=12), output_file=pfile)
    return pset


def test_MITgcm(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '2MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_MITgcm(chunk_mode)
    npart = 20
    lonp = -89.0 * np.ones(npart)
    latp = [i for i in -0.5+(-1.0+np.random.rand(npart)*2.0*1.0)]
    compute_MITgcm_particle_advection(field_set, mode, lonp, latp)
    # MITgcm sample file dimensions: y=600, x=840
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert ( len(field_set.U.grid.load_chunk) == (int(math.ceil(600.0/60.0)) * int(math.ceil(840.0/80.0))) )
    return True


def test_MITgcm_WaveWatchStokes(mode, chunk_mode):
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '2MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    field_set = fieldset_from_MITgcm_plus_WaveWatch3(chunk_mode)
    npart = 20
    lonp = -89.0 * np.ones(npart)
    latp = [i for i in -0.5+(-1.0+np.random.rand(npart)*2.0*1.0)]
    compute_MITgcm_particle_advection(field_set, mode, lonp, latp)
    # MITgcm sample file dimensions: y=600, x=840
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        assert (len(field_set.U.grid.load_chunk) == (1 * int(math.ceil(600.0/60.0)) * int(math.ceil(840.0/80.0))))
    return True


if __name__ == "__main__":
    assert (test_MITgcm('jit', False))
    assert (test_MITgcm('jit', 'auto'))
    assert (test_MITgcm('jit', 'specific'))
    assert (test_MITgcm_WaveWatchStokes('jit', False))
    assert (test_MITgcm_WaveWatchStokes('jit', 'auto'))
    assert (test_MITgcm_WaveWatchStokes('jit', 'specific'))
