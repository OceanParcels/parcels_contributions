import math
from datetime import timedelta as delta
from os import path
from glob import glob

import numpy as np
from parcels import rng as random
from parcels import ErrorCode
import dask

from parcels import AdvectionRK4_3D
from parcels import Field
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleFile
from parcels import ParticleSet
from parcels import ScipyParticle
from parcels import Variable

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def fieldset_from_OFAM(chunk_mode):
    # data_path = path.join("/data/oceanparcels/input_data", "OFAM3_2017", "3D")
    data_path = path.join("/data", "OFAM3_2017_daily", "3D")
    u_files = [path.join(data_path,dr) for dr in ['ocean_u_1993_04.nc','ocean_u_1993_05.nc','ocean_u_1993_06.nc']]  # Lat/lon subset of 0.1 deg grid with daily output.
    v_files = [path.join(data_path,dr) for dr in ['ocean_v_1993_04.nc','ocean_v_1993_05.nc','ocean_v_1993_06.nc']]
    w_files = [path.join(data_path,dr) for dr in ['ocean_w_1993_04.nc','ocean_w_1993_05.nc','ocean_w_1993_06.nc']]

    filenames = {'U': {'lon': u_files[0], 'lat': u_files[0], 'depth': u_files[0], 'data': u_files},
                 'V': {'lon': u_files[0], 'lat': u_files[0], 'depth': u_files[0], 'data': v_files},
                 'W': {'lon': u_files[0], 'lat': u_files[0], 'depth': u_files[0], 'data': w_files}}
    variables = {'U': 'u',
                 'V': 'v',
                 'W': 'w'}
    # dimensions = {'time': 'Time', 'depth': 'sw_ocean', 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
    dimensions = {'time': 'Time', 'depth': 'st_ocean', 'lat': 'yu_ocean', 'lon': 'xu_ocean'}
    chs = False
    netdcf_dim_name_map = None
    if chunk_mode == 'auto':
        chs = 'auto'
        netdcf_dim_name_map = {'lon': ['xu_ocean', 'xt_ocean'],
                               'lat': ['yu_ocean', 'yt_ocean'],
                               'depth': ['st_ocean', 'sw_ocean'],   # , 'st_edges_ocean'
                               'time': ['Time']}
    elif chunk_mode == 'specific':  # , 'st_edges_ocean': 8 # , 'st_ocean': 8
        chs = {'U': {'Time': 1, 'xu_ocean': 128, 'yu_ocean': 96, 'st_ocean': 8, 'xt_ocean': 128, 'yt_ocean': 96, 'sw_ocean': 8},
               'V': {'Time': 1, 'xu_ocean': 128, 'yu_ocean': 96, 'st_ocean': 8, 'xt_ocean': 128, 'yt_ocean': 96, 'sw_ocean': 8},
               'W': {'Time': 1, 'xu_ocean': 128, 'yu_ocean': 96, 'st_ocean': 8, 'xt_ocean': 128, 'yt_ocean': 96, 'sw_ocean': 8}}
        netdcf_dim_name_map = {'lon': ['xu_ocean', 'xt_ocean'],
                               'lat': ['yu_ocean', 'yt_ocean'],
                               'depth': ['st_ocean', 'sw_ocean'],
                               'time': ['Time']}

    fieldset = FieldSet.from_b_grid_dataset(filenames, variables, dimensions, field_chunksize=chs, time_periodic=delta(days=366), mesh='spherical', name_maps = netdcf_dim_name_map)
    return fieldset


def compute_OFAM_particle_advection(field_set, mode, lonp, latp):

    def DeleteParticle(particle, fieldset, time):
        particle.delete()

    def RenewParticle(particle, fieldset, time):
        # particle.lon = random.random() * 360.0
        particle.lon = -20.0 - 5.0 + random.random() * 2.0 * 5.0
        particle.lat = -75.0 + random.random() * 2.0 * 75.0
        particle.state = ErrorCode.Evaluate

    def periodicBC(particle, fieldSet, time):
        dlon = 360.0
        dlat = 75.0+75.0
        if particle.lon > 360.0:
            particle.lon -= dlon
        if particle.lon < 0.0:
            particle.lon += dlon
        if particle.lat > 75.0:
            particle.lat -= dlat
        if particle.lat < -75.0:
            particle.lat += dlat

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lonp, lat=latp)
    pfile = ParticleFile("OFAM_particles_chunk", pset, outputdt=delta(days=7))
    kernels = pset.Kernel(periodicBC) + pset.Kernel(AdvectionRK4_3D)
    pset.execute(kernels, runtime=delta(days=60), dt=delta(hours=12), output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    return pset


def test_OFAM(mode, chunk_mode):
    # here, coordinates for lon are 0 < lon < 360
    if chunk_mode == 'auto':
        dask.config.set({'array.chunk-size': '4MiB'})
    else:
        dask.config.set({'array.chunk-size': '128MiB'})
    random.seed(0)
    field_set = fieldset_from_OFAM(chunk_mode)
    npart = 4096
    lonp = [i for i in -20.0 - 5.0 + np.random.rand(npart) * 2.0 * 5.0] # -20.0 * np.ones(npart)
    latp = [i for i in -75.0 + np.random.rand(npart) * 2.0 * 75.0]
    compute_OFAM_particle_advection(field_set, mode, lonp, latp)
    # MITgcm sample file dimensions: y=1500, x=3600, w=51
    assert (len(field_set.U.grid.load_chunk) == len(field_set.V.grid.load_chunk))
    assert (len(field_set.U.grid.load_chunk) == len(field_set.W.grid.load_chunk))
    if chunk_mode is False:
        assert (len(field_set.U.grid.load_chunk) == 1)
    elif chunk_mode == 'auto':
        assert (len(field_set.U.grid.load_chunk) != 1)
    elif chunk_mode == 'specific':
        numblocks = [i for i in field_set.U.grid.chunk_info[1:4]]
        dblocks = 0
        for bsize in field_set.U.grid.chunk_info[4:4+numblocks[0]]:
            dblocks += bsize
        vblocks = 0
        for bsize in field_set.U.grid.chunk_info[4+numblocks[0]:4+numblocks[0]+numblocks[1]]:
            vblocks += bsize
        ublocks = 0
        for bsize in field_set.U.grid.chunk_info[4+numblocks[0]+numblocks[1]:4+numblocks[0]+numblocks[1]+numblocks[2]]:
            ublocks += bsize
        matching_numblocks = (ublocks==3600 and vblocks==1500 and dblocks==51)
        matching_fields = (field_set.U.grid.chunk_info == field_set.V.grid.chunk_info == field_set.W.grid.chunk_info)
        matching_uniformblocks = (len(field_set.U.grid.load_chunk) == (int(math.ceil(51.0/8.0)) * int(math.ceil(1500.0/96.0)) * int(math.ceil(3600.0/128.0))))
        assert ( matching_uniformblocks or (matching_fields and matching_numblocks) )
    return True

if __name__ == "__main__":
    # assert (test_OFAM('jit', False)) - won't work cause too big to load in 1 go in memory
    assert (test_OFAM('jit', 'auto'))
    assert (test_OFAM('jit', 'specific'))

