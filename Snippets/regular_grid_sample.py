"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import FieldSet, ScipyParticle, JITParticle, Variable, RectilinearZGrid, StateCode, OperationCode, ErrorCode
from parcels.particleset import ParticleSet
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
import datetime
import numpy as np
import xarray as xr
import os
# from scipy.interpolate import interpn
from glob import glob
import h5py

a = 3.6 * 1e2  # by definition: arcdeg
b = 1.8 * 1e2  # by definiton: arcdeg
DBG_MSG = False


def convert_timearray(t_array, dt_minutes, ns_per_sec, debug=False, array_name="time array"):
    """
    Helper function for time-conversion from the calendar format
    :param t_array: 2D array of time values in either calendar- or float-time format; dim-0 = object entities, dim-1 = time steps (or 1D with just timesteps)
    :param dt_minutes: expected delta_t als float-value (in minutes)
    :param ns_per_sec: conversion value of number of nanoseconds within 1 second
    :param debug: parameter telling to print debug messages or not
    :param array_name: name of the array (for debug prints)
    :return: converted t_array
    """
    ta = t_array
    while len(ta.shape) > 1:
        ta = ta[0]
    if isinstance(ta[0], datetime.datetime) or isinstance(ta[0], datetime.timedelta) or isinstance(ta[0], np.timedelta64) or isinstance(ta[0], np.datetime64) or np.float64(ta[1]-ta[0]) > (dt_minutes+dt_minutes/2.0):
        if debug:
            print("{}.dtype before conversion: {}".format(array_name, t_array.dtype))
        t_array = (t_array / ns_per_sec).astype(np.float64)
        ta = (ta / ns_per_sec).astype(np.float64)
        if debug:
            print("{0}.range and {0}.dtype after conversion: ({1}, {2}) \t {3}".format(array_name, ta.min(), ta.max(), ta.dtype))
    else:
        if debug:
            print("{0}.range and {0}.dtype: ({1}, {2}) \t {3} \t(no conversion applied)".format(array_name, ta.min(), ta.max(), ta.dtype))
        pass
    return t_array


def create_CMEMS_fieldset(datahead, periodic_wrap=False):
    ddir = os.path.join(datahead, "CMEMS", "GLOBAL_REANALYSIS_PHY_001_030")
    print(ddir)
    files = sorted(glob(os.path.join(ddir, "mercatorglorys12v1_gl12_mean_201607*.nc")))
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    global ttotal
    ttotal = 31  # days
    chs = 'auto'
    fieldset = None
    if periodic_wrap:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, time_periodic=delta(days=366))
    else:
        fieldset = FieldSet.from_netcdf(files, variables, dimensions, chunksize=chs, allow_time_extrapolation=True)
    global tsteps
    tsteps = len(fieldset.U.grid.time_full)
    global tstepsize
    tstepsize = int(math.floor(ttotal/tsteps))
    return fieldset


class SampleParticle(JITParticle):
    valid = Variable('valid', dtype=np.int32, initial=-1, to_write=True)
    sample_u = Variable('sample_u', initial=0., dtype=np.float32, to_write=True)
    sample_v = Variable('sample_v', initial=0., dtype=np.float32, to_write=True)


def sample_uv(particle, fieldset, time):
    particle.sample_u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.sample_v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    if particle.valid < 0:
        if (math.isnan(particle.time) == True):
            particle.valid = 0
        else:
            particle.valid = 1


def DeleteParticle(particle, fieldset, time):
    if particle.valid < 0:
        particle.valid = 0
    particle.delete()


if __name__=='__main__':
    parser = ArgumentParser(description="Resample curvilinear CMEMS grid on regular-spaced (A)-grid")
    parser.add_argument("-F", "--field_dir", dest="field_dir", type=str, default="/data/CMEMS", required=True, help="directory path of a CMEMS (input) field files")
    parser.add_argument("-O", "--out_dir", dest="out_dir", type=str, default="None", help="output directory path the interpolated (output) field files (HDF5)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="5", help="number of cells per arc-degree or metre (default: 5)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=720, help="computational delta_t time stepping in minutes (default: 720min = 12h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1440, help="repeating release rate of added particles in minutes (default: 1440min = 24h)")
    args = parser.parse_args()

    gres = int(float(eval(args.gres)))
    datahead =args.field_dir
    outdir = args.out_dir
    outdir = eval(outdir) if outdir == "None" else outdir
    if outdir is None:
        outdir = datahead
    if outdir is None:
        outdir = datahead
    out_fname = "CMEMS"
    dt_minutes = args.dt
    outdt_minutes = args.outdt
    time_in_days = args.time_in_days
    delete_func = DeleteParticle

    fT_start = np.datetime64('2016-01-01 00:00:00')
    # fT_end = np.datetime64('2016-12-31 11:59:59')
    fT_end = fT_start + np.timedelta64(delta(days=time_in_days))
    timerange = (fT_end - fT_start).astype('timedelta64')
    sec_per_day = 86400.0
    # ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    us_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    global_fT = np.arange(0, timerange, np.timedelta64(delta(minutes=outdt_minutes)))
    fT = (global_fT/us_per_sec).astype(np.float64)
    time_in_min = np.nanmin(fT, axis=0)
    time_in_max = np.nanmax(fT, axis=0)

    step = 1.0/gres
    xsteps = int(np.floor(a * gres))
    ysteps = int(np.floor(b * gres))

    xval = np.arange(start=-a*0.5, stop=a*0.5, step=step, dtype=np.float32)
    yval = np.arange(start=-b*0.5, stop=b*0.5, step=step, dtype=np.float32)
    centers_x = xval + step/2.0
    centers_y = yval + step/2.0
    us = np.zeros((centers_y.shape[0], centers_x.shape[0]))
    vs = np.zeros((centers_y.shape[0], centers_x.shape[0]))

    grid_file = h5py.File(os.path.join(outdir, "grid.h5"), "w")
    grid_lon_ds = grid_file.create_dataset("longitude", data=centers_x, compression="gzip", compression_opts=4)
    grid_lon_ds.attrs['unit'] = "arc degree"
    grid_lon_ds.attrs['name'] = 'longitude'
    grid_lon_ds.attrs['min'] = centers_x.min()
    grid_lon_ds.attrs['max'] = centers_x.max()
    grid_lat_ds = grid_file.create_dataset("latitude", data=centers_y, compression="gzip", compression_opts=4)
    grid_lat_ds.attrs['unit'] = "arc degree"
    grid_lat_ds.attrs['name'] = 'latitude'
    grid_lat_ds.attrs['min'] = centers_y.min()
    grid_lat_ds.attrs['max'] = centers_y.max()
    grid_time_ds = grid_file.create_dataset("times", data=fT, compression="gzip", compression_opts=4)
    grid_time_ds.attrs['unit'] = "seconds"
    grid_time_ds.attrs['name'] = 'time'
    grid_time_ds.attrs['min'] = np.nanmin(fT)
    grid_time_ds.attrs['max'] = np.nanmax(fT)
    grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file = h5py.File(os.path.join(outdir, "hydrodynamic_U.h5"), "w")
    us_file_ds = us_file.create_dataset("uo", shape=(1, us.shape[0], us.shape[1]), dtype=us.dtype, maxshape=(fT.shape[0], us.shape[0], us.shape[1]), compression="gzip", compression_opts=4)
    us_file_ds.attrs['unit'] = "m/s"
    us_file_ds.attrs['name'] = 'meridional_velocity'

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file = h5py.File(os.path.join(outdir, "hydrodynamic_V.h5"), "w")
    vs_file_ds = vs_file.create_dataset("vo", shape=(1, vs.shape[0], vs.shape[1]), dtype=vs.dtype, maxshape=(fT.shape[0], vs.shape[0], vs.shape[1]), compression="gzip", compression_opts=4)
    vs_file_ds.attrs['unit'] = "m/s"
    vs_file_ds.attrs['name'] = 'zonal_velocity'

    print("Sampling UV on CMEMS grid ...")
    sample_time = 0
    fieldset = create_CMEMS_fieldset(datahead=datahead, periodic_wrap=True)
    p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
    sample_pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=np.array(p_center_x).flatten(), lat=np.array(p_center_y).flatten(), time=sample_time)
    sample_kernel = sample_pset.Kernel(sample_uv)
    sample_outname = out_fname + "_sampleuv"
    sample_output_file = sample_pset.ParticleFile(name=os.path.join(outdir, sample_outname+".nc"), outputdt=delta(minutes=outdt_minutes))
    sample_pset.execute(sample_kernel, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=sample_output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func})
    sample_output_file.close()
    del sample_output_file
    del sample_pset
    del sample_kernel
    print("UV on CMEMS grid sampled.")

    print("Load sampled data ...")
    sample_xarray = xr.open_dataset(os.path.join(outdir, sample_outname + ".nc"))
    N_s = sample_xarray['lon'].shape[0]
    tN_s = sample_xarray['lon'].shape[1]
    if DBG_MSG:
        print("N: {}, t_N: {}".format(N_s, tN_s))
    valid_array = np.maximum(np.max(np.array(sample_xarray['valid'][:, 0:2]), axis=1), 0).astype(np.bool)
    if DBG_MSG:
        print("Valid array: any true ? {}; all true ? {}".format(valid_array.any(), valid_array.all()))
    ctime_array_s = sample_xarray['time'].data
    time_in_min_s = np.nanmin(ctime_array_s, axis=0)
    time_in_max_s = np.nanmax(ctime_array_s, axis=0)
    assert ctime_array_s.shape[1] == time_in_min.shape[0]
    mask_array_s = valid_array
    for ti in range(ctime_array_s.shape[1]):
        replace_indices = np.isnan(ctime_array_s[:, ti])
        ctime_array_s[replace_indices, ti] = time_in_max_s[ti]  # in this application, it should always work cause there's no delauyed release
    if DBG_MSG:
        print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(ctime_array_s.shape, type(ctime_array_s[0 ,0]), np.min(ctime_array_s[0]), np.max(ctime_array_s[0])))
    timebase_s = time_in_max_s[0]
    dtime_array_s = ctime_array_s - timebase_s
    if DBG_MSG:
        print("time info from file after baselining: shape = {} type = {} range = {}".format( dtime_array_s.shape, type(dtime_array_s[0 ,0]), (np.min(dtime_array_s), np.max(dtime_array_s)) ))

    psX = sample_xarray['lon']
    psY = sample_xarray['lat']
    psZ = None
    if 'depth' in sample_xarray.keys():
        psZ = sample_xarray['depth']
    elif 'z' in sample_xarray.keys():
        psZ = sample_xarray['z']
    psT = dtime_array_s
    global_psT = time_in_max_s -time_in_max_s[0]
    psT = convert_timearray(psT, outdt_minutes*60, us_per_sec, debug=DBG_MSG, array_name="psT")
    global_psT = convert_timearray(global_psT, outdt_minutes*60, us_per_sec, debug=DBG_MSG, array_name="global_psT")
    psU = sample_xarray['sample_u']
    psV = sample_xarray['sample_v']
    print("Sampled data loaded.")

    print("Interpolating UV on a regular-square grid ...")
    total_items = psT.shape[1]
    for ti in range(psT.shape[1]):
        us_local = np.expand_dims(psU[:, ti], axis=1)
        us_local[~mask_array_s, :] = 0
        vs_local = np.expand_dims(psV[:, ti], axis=1)
        vs_local[~mask_array_s, :] = 0
        if ti == 0 and DBG_MSG:
            print("us.shape {}; us_local.shape: {}; psU.shape: {}; p_center_y.shape: {}".format(us.shape, us_local.shape, psU.shape, p_center_y.shape))

        us[:, :] = np.reshape(us_local, p_center_y.shape)
        vs[:, :] = np.reshape(vs_local, p_center_y.shape)

        us_minmax = [min(us_minmax[0], us.min()), max(us_minmax[1], us.max())]
        us_statistics[0] += us.mean()
        us_statistics[1] += us.std()
        us_file_ds.resize((ti+1), axis=0)
        us_file_ds[ti, :, :] = us
        vs_minmax = [min(vs_minmax[0], vs.min()), max(vs_minmax[1], vs.max())]
        vs_statistics[0] += vs.mean()
        vs_statistics[1] += vs.std()
        vs_file_ds.resize((ti+1), axis=0)
        vs_file_ds[ti, :, :] = vs

        current_item = ti
        workdone = current_item / total_items
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
    print("\nFinished UV-interpolation.")

    us_file_ds.attrs['min'] = us_minmax[0]
    us_file_ds.attrs['max'] = us_minmax[1]
    us_file_ds.attrs['mean'] = us_statistics[0] / float(fT.shape[0])
    us_file_ds.attrs['std'] = us_statistics[1] / float(fT.shape[0])
    us_file.close()
    vs_file_ds.attrs['min'] = vs_minmax[0]
    vs_file_ds.attrs['max'] = vs_minmax[1]
    vs_file_ds.attrs['mean'] = vs_statistics[0] / float(fT.shape[0])
    vs_file_ds.attrs['std'] = vs_statistics[1] / float(fT.shape[0])
    vs_file.close()

    del centers_x
    del centers_y
    del xval
    del yval
    del global_fT