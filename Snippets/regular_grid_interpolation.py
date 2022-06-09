"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""
from scipy.interpolate import interpn
from datetime import timedelta as delta
from argparse import ArgumentParser
import datetime
import numpy as np
import h5py
import math
import sys
import os

# ===================================================================================================== #
# This snippet interpolates a regular-gridded UV field on a (differently-shaped) grid.                  #
# Input:                                                                                                #
#   - U and V: a 3D-array with dimensionality/coordinates [time_index][lat_index][lon_index]            #
#   - fT: 1D array of float times; dimensionality [time_index]                                          #
#   - flats: 1D array of float latitudes; dimensionality [lat_index]                                    #
#   - flons: 1D array of float longitutes; dimensionality [lon_index]                                   #
#   - centers_x: x-direction cell centers of the target grid (or field); dimensionality [new_lon_index] #
#   - centers_y: y-direction cell centers of the target grid (or field); dimensionality [new_lat_index] #
#   - us and vs: output 3D array with dimensionality [time_index][new_lat_index][new_lon_index]         #
# ===================================================================================================== #

a = 3.6 * 1e2  # by definition: arcdeg
b = 1.8 * 1e2  # by definiton: arcdeg
tsteps = 122 # in steps
tstepsize = 6.0 # unitary
tscale = 12.0*60.0*60.0 # in seconds
gyre_rotation_speed = 366.0*24.0*60.0*60.0  # assume 1 rotation every 52 weeks
# ==== INFO FROM NEMO-MEDUSA: realistic values are 0-2.5 [m/s] ==== #
# scalefac = (40.0 / (1000.0/ (60.0 * 60.0)))  # 40 km/h
scalefactor = ((4.0*1000) / (60.0*60.0))  # 4 km/h
vertical_scale = (800.0 / (24*60.0*60.0))  # 800 m/d
# ==== ONLY APPLY BELOW SCALING IF MESH IS FLAT AND (a, b) are below 100,000 [m] ==== #
v_scale_small = 1./1000.0 # this is to adapt, cause 1 U = 1 m/s = 1 spatial unit / time unit; spatial scale; domain = 1920 m x 960 m -> scale needs to be adapted to to interpret speed on a 1920 km x 960 km grid


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


def doublegyre_from_numpy(xdim=960, ydim=480, steady=False, mesh='flat'):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
    A = 0.3
    epsilon = 0.25
    omega = 2 * np.pi

    scalefac = scalefactor
    if 'flat' in mesh and np.maximum(a, b) > 370.0 and np.maximum(a, b) < 100000:
        scalefac *= v_scale_small

    lon = np.linspace(-a*0.5, a*0.5, xdim, dtype=np.float32)
    # lonrange = lon.max()-lon.min()
    sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, ydim, dtype=np.float32)
    # latrange = lat.max() - lat.min()
    sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = (tsteps * tstepsize) * tscale
    times = np.linspace(0., totime, tsteps, dtype=np.float64)
    sys.stdout.write("time field: {}\n".format(times.size))
    dx, dy = lon[2]-lon[1], lat[2]-lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    freqs = np.ones(times.size, dtype=np.float32)
    if not steady:
        for ti in range(times.shape[0]):
            time_f = times[ti] / gyre_rotation_speed
            freqs[ti] *= omega * time_f
    else:
        freqs = (freqs * 0.5) * omega

    for ti in range(times.shape[0]):
        freq = freqs[ti]
        # print(freq)
        for i in range(lon.shape[0]):
            for j in range(lat.shape[0]):
                x1 = ((lon[i]*2.0 + a) / a)  # - dx / 2
                x2 = ((lat[j]*2.0 + b) / (2.0*b))  # - dy / 2
                f_xt = (epsilon * np.sin(freq) * x1**2.0) + (1.0 - (2.0 * epsilon * np.sin(freq))) * x1
                U[ti, j, i] = -np.pi * A * np.sin(np.pi * f_xt) * np.cos(np.pi * x2)
                V[ti, j, i] = np.pi * A * np.cos(np.pi * f_xt) * np.sin(np.pi * x2) * (2 * epsilon * np.sin(freq) * x1 + 1 - 2 * epsilon * np.sin(freq))
    U *= scalefac
    V *= scalefac
    return lon, lat, times, U, V


if __name__=='__main__':
    parser = ArgumentParser(description="Resample curvilinear CMEMS grid on regular-spaced (A)-grid")
    parser.add_argument("-O", "--out_dir", dest="out_dir", type=str, default="None", required=True, help="output directory path the interpolated (output) field files (HDF5)")
    parser.add_argument("-gres", "--grid_resolution", dest="gres", type=str, default="5", help="number of cells per arc-degree or metre (default: 5)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-dt", "--deltatime", dest="dt", type=int, default=720, help="computational delta_t time stepping in minutes (default: 720min = 12h)")
    parser.add_argument("-ot", "--outputtime", dest="outdt", type=int, default=1440, help="repeating release rate of added particles in minutes (default: 1440min = 24h)")
    parser.add_argument("-fsx", "--field_sx", dest="field_sx", type=int, default="480", help="number of original field cells in x-direction")
    parser.add_argument("-fsy", "--field_sy", dest="field_sy", type=int, default="240", help="number of original field cells in y-direction")
    args = parser.parse_args()

    field_sx = args.field_sx
    field_sy = args.field_sy
    gres = float(eval(args.gres))
    outdir = args.out_dir
    out_fname = "CMEMS"
    dt_minutes = args.dt
    outdt_minutes = args.outdt
    time_in_days = args.time_in_days

    flons, flats, ftimes, U, V = doublegyre_from_numpy(xdim=field_sx, ydim=field_sy, mesh='spherical')
    fT_start = np.datetime64('2016-01-01 00:00:00')
    fT_end = fT_start + np.timedelta64(delta(days=time_in_days))
    timerange = (fT_end - fT_start).astype('timedelta64')
    sec_per_day = 86400.0
    # ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    us_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    global_fT = np.arange(0, timerange, np.timedelta64(delta(minutes=outdt_minutes)))
    fT = (global_fT/us_per_sec).astype(np.float64)
    # time_in_min = np.nanmin(fT, axis=0)
    # time_in_max = np.nanmax(fT, axis=0)

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
    grid_time_ds.attrs['min'] = np.min(fT)
    grid_time_ds.attrs['max'] = np.max(fT)
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

    print("Interpolating UV on a regular-square grid ...")
    total_items = fT.shape[0]
    for ti in range(fT.shape[0]):
        uv_ti = ti % U.shape[0]
        mgrid = (flats, flons)
        p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
        gcenters = (p_center_y.flatten(), p_center_x.flatten())
        us_local = interpn(mgrid, U[uv_ti], gcenters, method='linear', fill_value=.0)
        vs_local = interpn(mgrid, V[uv_ti], gcenters, method='linear', fill_value=.0)
        us[:, :] = np.reshape(us_local, p_center_y.shape)
        vs[:, :] = np.reshape(vs_local, p_center_y.shape)

        us_minmax = [min(us_minmax[0], us.min(initial=0.0)), max(us_minmax[1], us.max(initial=0.0))]
        us_statistics[0] += us.mean()
        us_statistics[1] += us.std()
        us_file_ds.resize((ti+1), axis=0)
        us_file_ds[ti, :, :] = us
        vs_minmax = [min(vs_minmax[0], vs.min(initial=0.0)), max(vs_minmax[1], vs.max(initial=0.0))]
        vs_statistics[0] += vs.mean()
        vs_statistics[1] += vs.std()
        vs_file_ds.resize((ti+1), axis=0)
        vs_file_ds[ti, :, :] = vs

        del us_local
        del vs_local
        del p_center_y
        del p_center_x

        current_item = ti+1
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

    del gcenters
    del centers_x
    del centers_y
    del xval
    del yval
    del fT
