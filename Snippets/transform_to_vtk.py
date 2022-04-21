import os
import gc
import sys
import h5py
import math
from scipy.io import netcdf
import xarray as xr
import numpy as np
import fnmatch
from scipy.interpolate import interpn
from datetime import timedelta
import datetime
from argparse import ArgumentParser
from glob import glob
import fnmatch
import vtk
from vtkmodules.util import numpy_support as np_vtk
from vtkmodules.vtkCommonCore import (
    vtkDoubleArray,
    vtkFloatArray,
    vtkMath,
    vtkPoints
)
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkThreshold, vtkThresholdPoints, vtkWindowedSincPolyDataFilter
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkPolyLine, vtkCellArray, vtkImageData
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter, vtkXMLPolyDataWriter, vtkXMLImageDataWriter
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
import warnings

DBG_MSG = False

# Helper function for time-conversion from the calendar format
def convert_timearray(t_array, dt_minutes, ns_per_sec, debug=False, array_name="time array"):
    """

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

def convert_timevalue(in_val, t0, ns_per_sec, debug=False):
    """
    :param in_val: input value
    :param t0: reference time value, in format of 'datetime.datetime' or 'np.datetime'
    :param ns_per_sec: float64 value of nanoseconds per second
    :param debug: debug-switch to print debug information
    """
    if debug:
        print("input value: {}".format(in_val))
    tval = in_val
    if isinstance(tval, datetime.datetime) or isinstance(tval, np.datetime64):
        tval = tval - t0
        if debug:
            print("converted timestep to time difference: {}".format(tval))
    if isinstance(tval, datetime.timedelta) or isinstance(tval, np.timedelta64):
        tval = np.array([tval / ns_per_sec], dtype=np.float64)[0]
        if debug:
            print("converted timedelta-value to float value: {}".format(tval))
    return tval

# -------------------------------------------------------------------------------------------------------------------- #
def get_data_of_ndarray_nc(data_array):
    """
    :param data_array: input field
    :return: tuple of data_array np.nanmin, np.nanmax, data0, data_dx
    """
    if data_array is None or data_array.shape[0] == 0:
        return None, None, None, None
    darray = data_array.data
    dmin = np.nanmin(data_array)
    dmax = np.nanmax(data_array)
    d0 = None
    data_dx = None
    if len(data_array.shape) == 1:
        d0 = darray[0]
        if data_array.shape[0] > 1:
            data_dx = darray[1] - darray[0]
    del darray
    return dmin, dmax, d0, data_dx

def get_data_of_ndarray_h5(data_array):
    """
    :param data_array: input field
    :return: tuple of data_array np.nanmin, np.nanmax, data0, data_dx
    """
    if data_array is None or data_array.shape[0] == 0:
        return None, None, None, None
    darray = data_array[()]
    dmin = np.nanmin(data_array)
    dmax = np.nanmax(data_array)
    d0 = None
    data_dx = None
    if len(data_array.shape) == 1:
        d0 = darray[0]
        if data_array.shape[0] > 1:
            data_dx = darray[1] - darray[0]
    del darray
    return dmin, dmax, d0, data_dx

# -------------------------------------------------------------------------------------------------------------------- #

def time_index_value(tx, _ft, periodic, _ft_dt=None):  # , _ft_min=None, _ft_max=None
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = tx / f_dt
    ti = int(math.floor(f_interp))
    if periodic:
        ti = ti % ft.shape[0]
    else:
        ti = max(0, min(ft.shape[0]-1, ti))
    return ti


def time_partion_value(tx, _ft, periodic, _ft_dt=None):  # , _ft_min=None, _ft_max=None
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = abs(tx / f_dt)
    if periodic:
        # print("f_interp = math.fmod({}, {})".format(f_interp, float(ft.shape[0])))
        f_interp = math.fmod(f_interp, float(ft.shape[0]))
    else:
        # print("f_interp = max({}, min({}, {}))".format(ft[0], ft[-1], f_interp))
        f_interp = max(0.0, min(float(ft.shape[0]-1), f_interp))
    f_t = f_interp - math.floor(f_interp)
    return f_t


def lat_index_value(lat, _fl):
    # expect fl to be forward-linear
    fl = _fl
    if isinstance(_fl, xr.DataArray):
        fl = fl.data
    f_dL = fl[1] - fl[0]
    f_interp = lat / f_dL
    lati = int(math.floor(f_interp))
    return lati


def lat_partion_value(lat, _fl):
    # expect ft to be forward-linear
    fl = _fl
    if isinstance(_fl, xr.DataArray):
        fl = fl.data
    f_dL = fl[1] - fl[0]
    f_interp = lat / f_dL
    lat_t = f_interp - math.floor(f_interp)
    return lat_t


def depth_index_value(dx, _fd):
    # expect ft to be forward-linear
    fd = _fd
    if isinstance(_fd, xr.DataArray):
        fd = fd.data
    f_dD = fd[1] - fd[0]
    f_interp = dx / f_dD
    di = int(math.floor(f_interp))
    return di


def depth_partion_value(dx, _fd):
    # expect ft to be forward-linear
    fd = _fd
    if isinstance(_fd, xr.DataArray):
        fd = fd.data
    f_dD = fd[1] - fd[0]
    f_interp = dx / f_dD
    f_d = f_interp - math.floor(f_interp)
    return f_d


if __name__ =='__main__':
    parser = ArgumentParser(description="Transforming UV[W] flow fields and particle data to VTK")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str, default="/var/scratch/experiments/NNvsGeostatistics/data", help="head directory containing all input data and also are the store target for output files", required=True)
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, default="None", help="head output directory")
    parser.add_argument("-U", "--Upattern", dest="Upattern", type=str, default='*U.nc', help="pattern of U-file(s)")
    parser.add_argument("--uvar", dest="uvar", type=str, default='vozocrtx', help="variable name of U")
    parser.add_argument("-V", "--Vpattern", dest="Vpattern", type=str, default='*V.nc', help="pattern of V-file(s)")
    parser.add_argument("--vvar", dest="vvar", type=str, default='vomecrty', help="variable name of V")
    parser.add_argument("-W", "--Wpattern", dest="Wpattern", type=str, default='*W.nc', help="pattern of W-file(s)")
    parser.add_argument("--wvar", dest="wvar", type=str, default='W', help="variable name of W")
    parser.add_argument("--xvar", dest="xvar", type=str, default="None", help="variable name of x")
    parser.add_argument("--xuvar", dest="xuvar", type=str, default="None", help="variable name of x in field 'U', if differing between fields.")
    parser.add_argument("--xvvar", dest="xvvar", type=str, default="None", help="variable name of x in field 'V', if differing between fields.")
    parser.add_argument("--xwvar", dest="xwvar", type=str, default="None", help="variable name of x in field 'W', if differing between fields.")
    parser.add_argument("--yvar", dest="yvar", type=str, default="None", help="variable name of y")
    parser.add_argument("--yuvar", dest="yuvar", type=str, default="None", help="variable name of y in field 'U', if differing between fields.")
    parser.add_argument("--yvvar", dest="yvvar", type=str, default="None", help="variable name of y in field 'V', if differing between fields.")
    parser.add_argument("--ywvar", dest="ywvar", type=str, default="None", help="variable name of y in field 'W', if differing between fields.")
    parser.add_argument("--zvar", dest="zvar", type=str, default="None", help="variable name of z")
    parser.add_argument("--zuvar", dest="zuvar", type=str, default="None", help="variable name of z in field 'U', if differing between fields.")
    parser.add_argument("--zvvar", dest="zvvar", type=str, default="None", help="variable name of z in field 'V', if differing between fields.")
    parser.add_argument("--zwvar", dest="zwvar", type=str, default="None", help="variable name of z in field 'W', if differing between fields.")
    parser.add_argument("--tvar", dest="tvar", type=str, default="None", help="variable name of t")
    parser.add_argument("--tuvar", dest="tuvar", type=str, default="None", help="variable name of t in field 'U', if differing between fields.")
    parser.add_argument("--tvvar", dest="tvvar", type=str, default="None", help="variable name of t in field 'V', if differing between fields.")
    parser.add_argument("--twvar", dest="twvar", type=str, default="None", help="variable name of t in field 'W', if differing between fields.")
    parser.add_argument("-F", "--format", dest="format", choices=['nc', 'h5'], default='nc', help="type of field files to evaluate, NetCDF (nc) or HDF5 (h5). Default: nc")
    parser.add_argument("-p", "--pfile", dest="pfile", type=str, default="*.nc", help="particle file of particles to-be-plotted", required=True)
    parser.add_argument("--plain_field_dump", dest="plain_field_dump", action="store_true", default=False, help="dumps the plain NetCDF field files to VTK StructuresGrids")
    parser.add_argument("--plain_particle_dump", dest="plain_particle_dump", action="store_true", default=False, help="dumps the plain NetCDF particle files to VTK PolyData")
    parser.add_argument("--interpolate", dest="interpolate", action="store_true", default=False, help="Time interpolation")
    parser.add_argument("--interpolate_particles_only", dest="interpolate_particles_only", action="store_true", default=False, help="Time-interpolate only the particles")
    parser.add_argument("--interpolate_field_only", dest="interpolate_field_only", action="store_true", default=False, help="Time-interpolate only the fields")
    parser.add_argument("--fixZ", dest="fixZ", action="store_true", default=False, help="transform z-Axis to display height, e.g. depth is negative")
    parser.add_argument("--bathymetry", dest="bathymetry", action="store_true", default=False, help="Obtains Bathymetry mesh by iso-countring the 0-velocity isosurface.")
    parser.add_argument("--store_as_sgrid", dest="store_sgrid", action="store_true", default=False, help="Do not interpolate in z-axis; store results as StructuredGrid")
    parser.add_argument("-LOm", "--lonmin", dest="lonmin", type=float, default=None, help="min. longitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LOM", "--lonmax", dest="lonmax", type=float, default=None, help="max. longitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LAm", "--latmin", dest="latmin", type=float, default=None, help="min. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LAM", "--latmax", dest="latmax", type=float, default=None, help="max. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-DM", "--depthmax", dest="depthmax", type=float, default=None, help="max. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-TIm", "--timin", dest="timin", type=int, default=None, help="min. time index to plot - only effective when interpolating")
    parser.add_argument("-TIM", "--timax", dest="timax", type=int, default=None, help="max. time index to plot - only effective when interpolating")
    parser.add_argument("-dt", "--interpolate_dt", dest="interpolate_dt", type=float, default=-1.0, help="Target interpolation dt (different from article-dt or field-dt) in day(s). Only positive values valid. If value is negative (default), the interpolation-dt is taken from the dt of the particle(s).")

    # parser.add_argument("-T", "--time", dest="time", type=str, default="-1.0", help="timestamp (in seconds) at which to be plotted")
    # parser.add_argument("-L", "--lat", dest="lat", type=float, default=0.0, help="latitude (in arcdegrees or metres) at which to be plotted")
    # parser.add_argument("-D", "--depth", dest="depth", type=float, default=1.0, help="depth (in metres) at which to be plotted")
    args = parser.parse_args()

    filedir = args.filedir
    outdir = args.outdir
    outdir = eval(outdir) if outdir == "None" else outdir
    if outdir is None:
        outdir = filedir
    if outdir is None:
        outdir = filedir
    pfile_name = args.pfile
    # timestamp = float(eval(args.time))
    # latitude = args.lat
    # depth_level = args.depth
    dump_field_plain = args.plain_field_dump
    dump_particle_plain = args.plain_particle_dump
    particles_only = args.interpolate_particles_only
    fields_only = args.interpolate_field_only
    interpolate = args.interpolate
    store_sgrid = args.store_sgrid
    periodicFlag = True
    hasW = False
    is3D = True
    Pn = 128
    idt = args.interpolate_dt * 86400.0

    fileformat = args.format
    Upattern = args.Upattern
    if '.' in Upattern:
        p_index = str.rfind(Upattern, '.')
        Upattern = Upattern[0:p_index]
    Vpattern = args.Vpattern
    if '.' in Upattern:
        p_index = str.rfind(Vpattern, '.')
        Vpattern = Vpattern[0:p_index]
    Wpattern = args.Wpattern
    if '.' in Wpattern:
        p_index = str.rfind(Wpattern, '.')
        Wpattern = Wpattern[0:p_index]

    if interpolate and "h5" in fileformat:
        warnings.warn("HDF5 data are expected to already be regularly-gridded, so they will not be interpolated")
        if not dump_field_plain and not dump_particle_plain:
            warnings.warn("As HDF5 files are not interpolated, we need to dump the plain particle- and field data.")
            dump_field_plain = True
            dumo_particle_plain = True

    xuvar = None
    xvvar = None
    xwvar = None
    yuvar = None
    yvvar = None
    ywvar = None
    zuvar = None
    zvvar = None
    zwvar = None
    tuvar = None
    tvvar = None
    twvar = None

    xvar = args.xvar
    xvar = eval(xvar) if xvar=="None" else xvar
    if xvar is None:
        xuvar = args.xuvar
        xuvar = eval(xuvar) if xuvar == "None" else xuvar
        xvvar = args.xvvar
        xvvar = eval(xvvar) if xvvar == "None" else xvvar
        xwvar = args.xwvar
        xwvar = eval(xwvar) if xwvar == "None" else xwvar
    else:
        xuvar = xvar
        xvvar = xvar
        xwvar = xvar
    yvar = args.yvar
    yvar = eval(yvar) if yvar == "None" else yvar
    if yvar is None:
        yuvar = args.yuvar
        yuvar = eval(yuvar) if yuvar == "None" else yuvar
        yvvar = args.yvvar
        yvvar = eval(yvvar) if yvvar == "None" else yvvar
        ywvar = args.ywvar
        ywvar = eval(ywvar) if ywvar == "None" else ywvar
    else:
        yuvar = yvar
        yvvar = yvar
        ywvar = yvar
    zvar = args.zvar
    zvar = eval(zvar) if zvar == "None" else zvar
    if zvar is None:
        zuvar = args.zuvar
        zuvar = eval(zuvar) if zuvar == "None" else zuvar
        zvvar = args.zvvar
        zvvar = eval(zvvar) if zvvar == "None" else zvvar
        zwvar = args.zwvar
        zwvar = eval(zwvar) if zwvar == "None" else zwvar
    else:
        zuvar = zvar
        zvvar = zvar
        zwvar = zvar
    if (zuvar is None) or (zuvar is None):
        is3D = False
    tvar = args.tvar
    tvar = eval(tvar) if tvar == "None" else tvar
    if tvar is None:
        tuvar = args.tuvar
        tuvar = eval(tuvar) if tuvar == "None" else tuvar
        tvvar = args.tvvar
        tvvar = eval(tvvar) if tvvar == "None" else tvvar
        twvar = args.twvar
        twvar = eval(twvar) if twvar == "None" else twvar
    else:
        tuvar = tvar
        tvvar = tvar
        twvar = tvar
    uvar = args.uvar
    uvar = eval(uvar) if uvar == "None" else uvar
    assert uvar is not None
    vvar = args.vvar
    vvar = eval(vvar) if vvar == "None" else vvar
    assert vvar is not None
    wvar = args.wvar
    wvar = eval(wvar) if wvar == "None" else wvar
    if wvar is None:
        hasW = False
    else:
        hasW = True
    particle_fpath = os.path.join(filedir, pfile_name) if pfile_name[0] != '/' else pfile_name
    # ==== temporal- and space resampling will be necessary ==== #
    multifile = False
    time_adaptive = False
    grid_adaptive = False
    plain_write = True
    # ==== spatial conversion   ==== #
    equatorial_a_radius = 63781370.0  # in [m]
    polar_b_radius = 63567523.0  # [m]
    # ==== time conversion data ==== #
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    if DBG_MSG:
        print("ns_per_sec = {}".format((ns_per_sec/np.timedelta64(1, 'ns')).astype(np.float64)))
    sec_per_day = 86400.0

    # fU_nc = None
    # fV_nc = None
    # fW_nc = None
    fX_nc = None
    fY_nc = None
    fZ_nc = None
    fX_nc_shape, fX_nc_len, fX_nc_min, fX_nc_max = None, None, None, None
    fY_nc_shape, fY_nc_len, fY_nc_min, fY_nc_max = None, None, None, None
    fZ_nc_shape, fZ_nc_len, fZ_nc_min, fZ_nc_max = None, None, None, None
    fT_nc = None
    speed_nc = None
    fU_ext_nc = None
    fV_ext_nc = None
    fW_ext_nc = None
    f_velmag_ext_nc = None
    extents_nc = None
    uvel_fpath_nc = sorted(glob(os.path.join(filedir, Upattern + ".nc")))
    vvel_fpath_nc = sorted(glob(os.path.join(filedir, Vpattern + ".nc")))
    wvel_fpath_nc = None
    if hasW:
        wvel_fpath_nc = sorted(glob(os.path.join(filedir, Wpattern + ".nc")))
    if "nc" in fileformat:
        if hasW:
            assert len(wvel_fpath_nc) == len(uvel_fpath_nc)
        if len(uvel_fpath_nc) > 1 and len(vvel_fpath_nc) > 1:
            multifile = True
        if len(uvel_fpath_nc) > 0 and os.path.exists(uvel_fpath_nc[0]):
            f_u = xr.open_dataset(uvel_fpath_nc[0], decode_cf=True, engine='netcdf4')
            fT_nc = f_u.variables[tuvar].data
            fX_nc = f_u.variables[xuvar]
            fX_nc_min, fX_nc_max, fX_nc_0, fX_nc_dx = get_data_of_ndarray_nc(fX_nc)
            fX_nc = fX_nc.data
            fX_nc_shape, fX_nc_len = fX_nc.shape, fX_nc.shape[0]
            fY_nc = f_u.variables[yuvar]
            fY_nc_min, fY_nc_max, fY_nc_0, fY_nc_dy = get_data_of_ndarray_nc(fY_nc)
            fY_nc = fY_nc.data
            fY_nc_shape, fY_nc_len = fY_nc.shape, fY_nc.shape[0]
            if is3D:
                fZ_nc = f_u.variables[zuvar] if zuvar in f_u.variables.keys() else None
                fZ_nc_min, fZ_nc_max, fZ_nc_0, fZ_nc_dz = get_data_of_ndarray_nc(fZ_nc)
                fZ_nc = fZ_nc.data
                fZ_nc_shape, fZ_nc_len = fZ_nc.shape, fZ_nc.shape[0]
            if fZ_nc is None:
                is3D = False
            extents_nc = (fX_nc.min(), fX_nc.max(), fY_nc.min(), fY_nc.max(), fZ_nc.min(), fZ_nc.max()) if is3D else (fX_nc.min(), fX_nc.max(), fY_nc.min(), fY_nc.max())
            f_u.close()
            del f_u
            if not multifile:
                uvel_fpath_nc = uvel_fpath_nc[0]
        if len(vvel_fpath_nc) > 0 and os.path.exists(vvel_fpath_nc[0]):
            if not multifile:
                vvel_fpath_nc = vvel_fpath_nc[0]
        if hasW and  len(wvel_fpath_nc) > 0 and os.path.exists(wvel_fpath_nc[0]):
            if not multifile:
                wvel_fpath_nc = wvel_fpath_nc[0]

        # print("fX_nc: {}".format(fX_nc))
        # print("fY_nc: {}".format(fY_nc))
        # print("fZ_nc: {}".format(fZ_nc))
        # print("fT_nc: {}".format(fT_nc))
        print("extends XYZ (NetCDF): {}".format(extents_nc))

    # fU_h5 = None
    # fV_h5 = None
    # fW_h5 = None
    fX_h5 = None
    fY_h5 = None
    fZ_h5 = None
    fX_h5_shape, fX_h5_len, fX_h5_min, fX_h5_max = None, None, None, None
    fY_h5_shape, fY_h5_len, fY_h5_min, fY_h5_max = None, None, None, None
    fZ_h5_shape, fZ_h5_len, fZ_h5_min, fZ_h5_max = None, None, None, None
    fT_h5 = None
    speed_h5 = None
    fU_ext_h5 = None
    fV_ext_h5 = None
    fW_ext_h5 = None
    f_velmag_ext_h5 = None
    extents_h5 = None
    uvel_fpath_h5 = glob(os.path.join(filedir, Upattern + ".h5"))
    vvel_fpath_h5 = glob(os.path.join(filedir, Vpattern + ".h5"))
    wvel_fpath_h5 = None
    if hasW:
        wvel_fpath_h5 = glob(os.path.join(filedir, Wpattern + ".h5"))
    grid_fpath_h5 = os.path.join(filedir, 'grid.h5')
    if "h5" in fileformat:
        if hasW:
            assert len(wvel_fpath_h5) == len(uvel_fpath_h5)
        if len(uvel_fpath_h5) > 1 and len(vvel_fpath_h5) > 1:
            multifile |= True
        if len(uvel_fpath_h5) > 0 and os.path.exists(uvel_fpath_h5[0]):
            if not multifile:
                uvel_fpath_h5 = uvel_fpath_h5[0]
        if len(vvel_fpath_h5) > 0 and os.path.exists(vvel_fpath_h5[0]):
            if not multifile:
                vvel_fpath_h5 = vvel_fpath_h5[0]
        if hasW and len(wvel_fpath_h5) > 0 and os.path.exists(wvel_fpath_h5[0]):
            if not multifile:
                wvel_fpath_h5 = wvel_fpath_h5[0]
        if os.path.exists(grid_fpath_h5):
            fZ_h5 = None
            f_grid = h5py.File(grid_fpath_h5, "r")
            fX_h5 = f_grid['longitude']
            fX_h5_min, fX_h5_max, fX_h5_0, fX_h5_dx = get_data_of_ndarray_h5(fX_h5)
            fX_h5 = fX_h5[()]
            fX_h5_shape, fX_h5_len = fX_h5.shape, fX_h5.shape[0]
            fY_h5 = f_grid['latitude']
            fY_h5_min, fY_h5_max, fY_h5_0, fY_h5_dy = get_data_of_ndarray_h5(fY_h5)
            fY_h5 = fY_h5[()]
            fY_h5_shape, fY_h5_len = fY_h5.shape, fY_h5.shape[0]
            if is3D:
                fZ_h5 = f_grid['depths'] if 'depths' in f_grid else None
                fZ_h5_min, fZ_h5_max, fZ_h5_0, fZ_h5_dz = get_data_of_ndarray_h5(fZ_h5)
                fZ_h5 = fZ_h5[()]
                fZ_h5_shape, fZ_h5_len = fZ_h5.shape, fZ_h5.shape[0]
            if fZ_h5 is None:
                is3D = False
            fT_h5 = f_grid['times'][()]
            extents_h5 = (fX_h5.min(), fX_h5.max(), fY_h5.min(), fY_h5.max(), fZ_h5.min(), fZ_h5.max()) if is3D else (fX_h5.min(), fX_h5.max(), fY_h5.min(), fY_h5.max())
            f_grid.close()
            del f_grid

        # print("fX_h5: {}".format(fX_h5))
        # print("fY_h5: {}".format(fY_h5))
        # print("fZ_h5: {}".format(fZ_h5))
        # print("fT_h5: {}".format(fT_h5))
        print("extends XYZ (NetCDF): {}".format(extents_h5))

    print("multifile: {}".format(multifile))
    timebase = None
    dtime_array = None
    fT = None
    fT_dt = 0
    t0 = None
    fT_fpath_mapping = []  # stores tuples with (<index of file in all files>, <filepath U>, <local index of ti in fT(f_u)>)
    if "nc" in fileformat:
        fXb_ft_nc = []
        fYb_ft_nc = []
        fZb_ft_nc = []
        fT_ft_nc = []
        fU_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fV_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fW_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        if multifile:
            print("U-files: {}".format(uvel_fpath_nc))
            i = 0
            for fpath in uvel_fpath_nc:
                f_u = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                xnc = f_u.variables[xuvar]
                xnc_min, xnc_max, xnc_0, xnc_dx = get_data_of_ndarray_nc(xnc)
                # print("xnc_min = {}, xnc_max = {}, xnc_0 = {}, xnc_dx = {}".format(xnc_min, xnc_max, xnc_0, xnc_dx))
                ync = f_u.variables[yuvar]
                ync_min, ync_max, ync_0, ync_dy = get_data_of_ndarray_nc(ync)
                znc = None
                znc_min, znc_max, znc_0, znc_dz = None, None, None, None
                if is3D:
                    znc = f_u.variables[zuvar] if zuvar in f_u.variables.keys() else None
                    znc_min, znc_max, znc_0, znc_dz = get_data_of_ndarray_nc(znc)
                tnc = f_u.variables[tuvar].data
                if len(fXb_ft_nc) > 0 and fXb_ft_nc[0][0] == xnc_min and fXb_ft_nc[0][1] == xnc_max:
                    grid_adaptive &= False
                fXb_ft_nc.append((xnc_min, xnc_max))
                if len(fYb_ft_nc) > 0 and fYb_ft_nc[0][0] == ync_min and fYb_ft_nc[0][1] == ync_max:
                    grid_adaptive &= False
                fYb_ft_nc.append((ync_min, ync_max))
                if is3D and (znc is not None):
                    if len(fZb_ft_nc) > 0 and fZb_ft_nc[0][0] == znc_min and fZb_ft_nc[0][1] == znc_max:
                        grid_adaptive &= False
                    fZb_ft_nc.append((znc_min, znc_max))
                if t0 is None:
                    t0 = tnc[0] if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else tnc
                if np.isclose(fT_dt, 0.0):
                    if len(tnc) > 1:
                        ft_0 = convert_timevalue(tnc[0], t0, ns_per_sec, debug=(i == 0))
                        ft_1 = convert_timevalue(tnc[1], t0, ns_per_sec, debug=(i == 0))
                        fT_dt = ft_1 - ft_0
                    elif len(fT_ft_nc) > 0:
                        ft_0 = (convert_timevalue(tnc[0], t0, ns_per_sec, debug=(i == 0)) if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else convert_timevalue(tnc, t0, ns_per_sec, debug=(i == 0)))
                        fT_dt = ft_0 - fT_ft_nc[0]

                if isinstance(tnc, list):
                    tnc = np.ndarray(tnc)
                if isinstance(tnc, np.ndarray):
                    for ti in range(tnc.shape[0]):
                        fT_ft_nc.append(convert_timevalue(tnc[ti], t0, ns_per_sec, debug=(i == 0)))
                        fT_fpath_mapping.append((i, fpath, ti))
                else:
                    fT_ft_nc.append(convert_timevalue(tnc, t0, ns_per_sec, debug=(i == 0)))
                    fT_fpath_mapping.append((i, fpath, 0))

                xi_same = True
                for xG in fXb_ft_nc:
                    xi_same &= ((xnc_min, xnc_max) != xG)
                yi_same = True
                for yG in fYb_ft_nc:
                    yi_same &= ((ync_min, ync_max) != yG)
                zi_same = True
                if is3D and znc is not None and len(fZb_ft_nc) > 0:
                    for zG in fZb_ft_nc:
                        zi_same &= ((znc_min, znc_max) != zG)
                if xi_same and yi_same and zi_same:
                    grid_adaptive &= False
                else:
                    grid_adaptive |= True

                fU_nc = f_u.variables[uvar].data
                if i == 0:
                    print("fU - shape: {}".format(fU_nc.shape))
                max_u_value = np.maximum(np.maximum(np.abs(np.nanmin(fU_nc)), np.abs(np.nanmax(fU_nc))), np.maximum(np.abs(fU_ext_nc[0]), np.abs(fU_ext_nc[1])))
                fU_ext_nc = (-max_u_value, +max_u_value)
                f_u.close()
                del xnc
                del ync
                del znc
                del fU_nc
                del f_u
                i += 1
            print("fU - ext.: {}".format(fU_ext_nc))
            print("V-files: {}".format(vvel_fpath_nc))
            i = 0
            for fpath in vvel_fpath_nc:
                f_v = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                fV_nc = f_v.variables[vvar].data
                if i == 0:
                    print("fV - shape: {}".format(fV_nc.shape))
                max_v_value = np.maximum(np.maximum(np.abs(np.nanmin(fV_nc)), np.abs(np.nanmax(fV_nc))), np.maximum(np.abs(fV_ext_nc[0]), np.abs(fV_ext_nc[1])))
                fV_ext_nc = (-max_v_value, +max_v_value)
                f_v.close()
                del fV_nc
                del f_v
                i += 1
            print("fV - ext.: {}".format(fV_ext_nc))
            if hasW:
                print("W-files: {}".format(wvel_fpath_nc))
                i = 0
                for fpath in wvel_fpath_nc:
                    f_w = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                    fW_nc = f_w.variables[wvar].data
                    if i == 0:
                        print("fW - shape: {}".format(fW_nc.shape))
                    max_w_value = np.maximum(np.maximum(np.abs(np.nanmin(fW_nc)), np.abs(np.nanmax(fW_nc))), np.maximum(np.abs(fW_ext_nc[0]), np.abs(fW_ext_nc[1])))
                    fW_ext_nc = (-max_w_value, +max_w_value)
                    f_w.close()
                    del fW_nc
                    del f_w
                    i += 1
                print("fW - ext.: {}".format(fW_ext_nc))
            print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_nc), fT_dt, fT_ft_nc))
            # ==== check for dt consistency ==== #
            if len(fT_ft_nc) > 1:
                for ti in range(1, len(fT_ft_nc)):
                    delta_dt = (fT_ft_nc[ti] - fT_ft_nc[ti-1])
                    time_adaptive |= not np.isclose(delta_dt, fT_dt)
            else:
                time_adaptive = False
            # if grid_adaptive:
            #     fX = None
            #     fY = None
            #     fZ = None
            fT = np.array(fT_ft_nc)
        else:
            print("U-file: {}".format(uvel_fpath_nc))
            # ======== u-velocity ======== #
            f_u = xr.open_dataset(uvel_fpath_nc, decode_cf=True, engine='netcdf4')
            xnc = f_u.variables[xuvar]
            xnc_min, xnc_max, xnc_0, xnc_dx = get_data_of_ndarray_nc(xnc)
            fXb_ft_nc.append((xnc_min, xnc_max))
            ync = f_u.variables[yuvar]
            ync_min, ync_max, ync_0, ync_dy = get_data_of_ndarray_nc(ync)
            fYb_ft_nc.append((ync_min, ync_max))
            znc = None
            znc_min, znc_max, znc_0, znc_dz = None, None, None, None
            if is3D:
                znc = f_u.variables[zuvar].data if zuvar in f_u.variables.keys() else None
                znc_min, znc_max, znc_0, znc_dz = get_data_of_ndarray_nc(znc)
                fZb_ft_nc.append((znc_min, znc_max))
            tnc = f_u.variables[tuvar].data
            grid_adaptive = False

            if t0 is None:
                t0 = tnc[0] if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else tnc
            if np.isclose(fT_dt, 0.0):
                if len(tnc) > 1:
                    ft_0 = convert_timevalue(tnc[0], t0, ns_per_sec, debug=True)
                    ft_1 = convert_timevalue(tnc[1], t0, ns_per_sec, debug=True)
                    fT_dt = ft_1 - ft_0
                elif len(fT_ft_nc) > 0:
                    ft_0 = (convert_timevalue(tnc[0], t0, ns_per_sec, debug=True) if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else convert_timevalue(tnc, t0, ns_per_sec, debug=True))
                    fT_dt = ft_0 - fT_ft_nc[0]

            if isinstance(tnc, np.ndarray):
                # fT_ft_nc = convert_timearray(tnc, fT_dt*60, ns_per_sec, True).tolist()
                fT_ft_nc = tnc.tolist()
            else:
                fT_ft_nc.append(tnc)
            for ti in range(len(fT_ft_nc)):
                fT_fpath_mapping.append((None, uvel_fpath_nc, ti))
            fT = np.array(fT_ft_nc)

            fU_nc = f_u.variables[uvar].data
            max_u_value = np.maximum(np.abs(fU_nc.min()), np.abs(fU_nc.max()))
            fU_ext_nc = (-max_u_value, +max_u_value)
            f_u.close()
            del xnc
            del ync
            del znc
            del fU_nc
            del f_u
            print("fU - ext.: {}".format(fU_ext_nc))
            print("V-file: {}".format(vvel_fpath_nc))
            # ======== v-velocity ======== #
            f_v = xr.open_dataset(vvel_fpath_nc, decode_cf=True, engine='netcdf4')
            fV_nc = f_v.variables[vvar].data
            max_v_value = np.maximum(np.abs(fV_nc.min()), np.abs(fV_nc.max()))
            fV_ext_nc = (-max_v_value, +max_v_value)
            f_v.close()
            del fV_nc
            del f_v
            print("fV - ext.: {}".format(fV_ext_nc))
            if hasW:
                print("W-file: {}".format(wvel_fpath_nc))
                # ======== w-velocity ======== #
                f_w = xr.open_dataset(wvel_fpath_nc, decode_cf=True, engine='netcdf4')
                fW_nc = f_w.variables[wvar].data
                max_w_value = np.maximum(np.abs(fW_nc.min()), np.abs(fW_nc.max()))
                fW_ext_nc = (-max_w_value, +max_w_value)
                f_w.close()
                del fW_nc
                del f_w
                print("fW - ext.: {}".format(fW_ext_nc))

        # ======== Time post-processing ======== #
        time_in_min = np.nanmin(fT, axis=0)
        time_in_max = np.nanmax(fT, axis=0)
        if DBG_MSG:
            print("Times:\n\tmin = {}\n\tmax = {}".format(time_in_min, time_in_max))
        # assert fT.shape[1] == time_in_min.shape[0]
        timebase = time_in_max[0] if (isinstance(time_in_max, list) or isinstance(time_in_max, np.ndarray)) else time_in_max
        dtime_array = fT - timebase
        fT = convert_timearray(fT, fT_dt, ns_per_sec, debug=DBG_MSG, array_name="fT")

        del fXb_ft_nc
        del fYb_ft_nc
        del fZb_ft_nc
        del fT_ft_nc
    else:
        fXb_ft_h5 = []
        fYb_ft_h5 = []
        fZb_ft_h5 = []
        fT_ft_h5 = []
        fU_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fV_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fW_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        assert (isinstance(fT_h5, list) or isinstance(fT_h5, np.ndarray))
        # ==== ==== consistent global grid file ==== ==== #
        if os.path.exists(grid_fpath_h5):
            grid_adaptive = False
            fXb_ft_h5.append((fX_h5_min, fX_h5_max))
            fYb_ft_h5.append((fY_h5_min, fY_h5_max))
            if is3D:
                fZb_ft_h5.append((fZ_h5_min, fZ_h5_max))

            if t0 is None:
                t0 = fT_h5[0]
            if np.isclose(fT_dt, 0.0):
                if len(fT_h5) > 1 or fT_h5.shape[0] > 1:
                    ft_0 = convert_timevalue(fT_h5[0], t0, ns_per_sec, debug=True)
                    ft_1 = convert_timevalue(fT_h5[1], t0, ns_per_sec, debug=True)
                    fT_dt = ft_1 - ft_0
                elif len(fT_ft_h5) > 0:
                    ft_0 = convert_timevalue(fT_h5[0], t0, ns_per_sec, debug=True)
                    fT_dt = ft_0 - fT_ft_h5[0]

            if isinstance(fT_h5, np.ndarray):
                # fT_ft_nc = convert_timearray(tnc, fT_dt*60, ns_per_sec, True).tolist()
                fT_ft_h5 = fT_h5.tolist()
            else:
                fT_ft_h5.append(fT_h5)
            for ti in range(len(fT_ft_h5)):
                fT_fpath_mapping.append((None, uvel_fpath_nc, ti))

            xi_same = True
            for xG in fXb_ft_h5:
                xi_same &= ((fX_h5_min, fX_h5_max) != xG)
            yi_same = True
            for yG in fYb_ft_h5:
                yi_same &= ((fY_h5_min, fY_h5_max) != yG)
            zi_same = True
            if is3D and fZ_h5 is not None and len(fZb_ft_h5) > 0:
                for zG in fZb_ft_h5:
                    zi_same &= ((fZ_h5_min, fZ_h5_max) != zG)
            if xi_same and yi_same and zi_same:
                grid_adaptive &= False
            else:
                grid_adaptive |= True

            fT = np.array(fT_ft_h5)
        # ==== ==== consistent global grid file ==== ==== #
        if multifile:
            print("U-files: {}".format(uvel_fpath_h5))
            i = 0
            for fpath in uvel_fpath_h5:
                f_u = h5py.File(fpath, "r")
                fU_h5 = f_u[uvar][()]
                max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
                fU_ext_h5 = (-max_u_value, +max_u_value)
                f_u.close()
                del fU_h5
                del f_u
                i += 1
            print("fU - ext.: {}".format(fU_ext_h5))
            print("V-files: {}".format(vvel_fpath_h5))
            i = 0
            for fpath in vvel_fpath_h5:
                f_v = h5py.File(fpath, "r")
                fV_h5 = f_v[vvar][()]
                max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
                fV_ext_h5 = (-max_v_value, +max_v_value)
                f_v.close()
                del fV_h5
                del f_v
                i += 1
            print("fV - ext.: {}".format(fV_ext_h5))
            if hasW:
                print("W-files: {}".format(wvel_fpath_h5))
                i = 0
                for fpath in wvel_fpath_h5:
                    f_w = h5py.File(fpath, "r")
                    fW_h5 = f_w[wvar][()]
                    max_w_value = np.maximum(np.abs(fW_h5.min()), np.abs(fW_h5.max()))
                    fW_ext_h5 = (-max_w_value, +max_w_value)
                    f_w.close()
                    del fW_h5
                    del f_w
                    i += 1
                print("fW - ext.: {}".format(fW_ext_h5))
            print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_h5), fT_dt, fT_ft_h5))
            # ==== check for dt consistency ==== #
            if len(fT_ft_h5) > 1:
                for ti in range(1, len(fT_ft_h5)):
                    delta_dt = (fT_ft_h5[ti] - fT_ft_h5[ti-1])
                    time_adaptive |= not np.isclose(delta_dt, fT_dt)
            else:
                time_adaptive = False
            fT = np.array(fT_ft_h5)
        else:
            print("U-file: {}".format(uvel_fpath_h5))
            f_u = h5py.File(uvel_fpath_h5, "r")
            fU_h5 = f_u[uvar][()]
            max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
            fU_ext_h5 = (-max_u_value, +max_u_value)
            f_u.close()
            del fU_h5
            del f_u
            print("fU - ext.: {}".format(fU_ext_h5))
            print("V-files: {}".format(vvel_fpath_h5))
            f_v = h5py.File(vvel_fpath_h5, "r")
            fV_h5 = f_v[vvar][()]
            max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
            fV_ext_h5 = (-max_v_value, +max_v_value)
            f_v.close()
            del fV_h5
            del f_v
            print("fV - ext.: {}".format(fV_ext_h5))
            if hasW:
                print("W-files: {}".format(wvel_fpath_h5))
                f_w = h5py.File(wvel_fpath_h5, "r")
                fW_h5 = f_w[wvar][()]
                max_w_value = np.maximum(np.abs(fW_h5.min()), np.abs(fW_h5.max()))
                fW_ext_h5 = (-max_w_value, +max_w_value)
                f_w.close()
                del fW_h5
                del f_w
                print("fW - ext.: {}".format(fW_ext_h5))
            print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_h5), fT_dt, fT_ft_h5))

        # ======== Time post-processing ======== #
        time_in_min = np.nanmin(fT, axis=0)
        time_in_max = np.nanmax(fT, axis=0)
        if DBG_MSG:
            print("Times:\n\tmin = {}\n\tmax = {}".format(time_in_min, time_in_max))
        # assert fT.shape[1] == time_in_min.shape[0]
        timebase = time_in_max[0] if (isinstance(time_in_max, list) or isinstance(time_in_max, np.ndarray)) else time_in_max
        dtime_array = fT - timebase
        fT = convert_timearray(fT, fT_dt, ns_per_sec, debug=DBG_MSG, array_name="fT")

        del fXb_ft_h5
        del fYb_ft_h5
        del fZb_ft_h5
        del fT_ft_h5

    fX = None
    fY = None
    fZ = None
    fX_shape, fX_len, fX_min, fX_max = None, None, None, None
    fY_shape, fY_len, fY_min, fY_max = None, None, None, None
    fZ_shape, fZ_len, fZ_min, fZ_max = None, None, None, None
    speed = None
    fU_ext = None
    fV_ext = None
    fW_ext = None
    extents = None
    if "nc" in fileformat:
        fX = fX_nc
        fY = fY_nc
        fZ = fZ_nc
        fX_shape, fX_len, fX_min, fX_max = fX_nc_shape, fX_nc_len, fX_nc_min, fX_nc_max
        fY_shape, fY_len, fY_min, fY_max = fY_nc_shape, fY_nc_len, fY_nc_min, fY_nc_max
        fZ_shape, fZ_len, fZ_min, fZ_max = fZ_nc_shape, fZ_nc_len, fZ_nc_min, fZ_nc_max
        fU_ext, fV_ext, fW_ext = fU_ext_nc, fV_ext_nc, fW_ext_nc
        extents = extents_nc
    elif "h5" in fileformat:
        fX = fX_h5
        fY = fY_h5
        fZ = fZ_h5
        fX_shape, fX_len, fX_min, fX_max = fX_h5_shape, fX_h5_len, fX_h5_min, fX_h5_max
        fY_shape, fY_len, fY_min, fY_max = fY_h5_shape, fY_h5_len, fY_h5_min, fY_h5_max
        fZ_shape, fZ_len, fZ_min, fZ_max = fZ_h5_shape, fZ_h5_len, fZ_h5_min, fZ_h5_max
        fU_ext, fV_ext, fW_ext = fU_ext_h5, fV_ext_h5, fW_ext_h5
        extents = extents_h5
    else:
        exit()
    print("grid adaptive: {}".format(grid_adaptive))
    print("time adaptive: {}".format(time_adaptive))
    print("fX - shape: {}; |fX|: {}".format(fX_shape, fX_len))
    print("fY - shape: {}; |fY|: {}".format(fY_shape, fY_len))
    print("fZ - shape: {}; |fZ|: {}".format(fZ_shape, fZ_len))
    print("fT - shape: {}; |fT|: {}".format(fT.shape, len(fT)))
    fX_ext = (fX_min, fX_max)
    fY_ext = (fY_min, fY_max)
    fZ_ext = (fZ_min, fZ_max)
    fT_ext = (fT.min(), fT.max())
    print("fX ext. (in) - {}".format(fX_ext))
    print("fY ext. (in) - {}".format(fY_ext))
    print("fZ ext. (in) - {}".format(fZ_ext))
    print("fT ext. (in) - {}".format(fT_ext))
    print("fT: {}".format(fT))
    sX = fX_ext[1] - fX_ext[0]
    sY = fY_ext[1] - fY_ext[0]
    sZ = fZ_ext[1] - fZ_ext[0]
    sT = fT_ext[1] - fT_ext[0]
    resample_x = 0
    resample_y = 0
    if (fX_ext[0] >= -180.1) and (fX_ext[1] <= 180.1):
        fX = (fX / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
        fX_ext = (fX.min(), fX.max())
        sX = fX_ext[1] - fX_ext[0]
        resample_x = 1
    elif (fX_ext[0] >= 0.0) and (fX_ext[1] <= 360.1):
        # fX = (fX / 360.0) * (2.0 * np.pi * equatorial_a_radius)
        fX = ((fX-180.0) / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
        fX_ext = (fX.min(), fX.max())
        sX = fX_ext[1] - fX_ext[0]
        resample_x = 2
    if (fY_ext[0] >= -90.1) and (fY_ext[1] <= 90.1):
        fY = (fY / 90.0) * ((np.pi * polar_b_radius) / 2.0)
        fY_ext = (fY.min(), fY.max())
        sY = fY_ext[1] - fY_ext[0]
        resample_y = 1
    elif (fY_ext[0] >= 0.0) and (fY_ext[1] <= 180.1):
        # fY = (fY / 180.0) * (np.pi * polar_b_radius)
        fY = ((fY-90.0) / 90.0) * ((np.pi * polar_b_radius) / 2.0)
        fY_ext = (fY.min(), fY.max())
        sY = fY_ext[1] - fY_ext[0]
        resample_y = 2
    fX_dx_1 = fX[1:]
    fX_dx_0 = fX[0:-1]
    fX_dx = fX_dx_1-fX_dx_0
    fX_dx_min = np.nanmin(fX_dx)
    fX_dx_max = np.nanmax(fX_dx)
    del fX_dx_0
    del fX_dx_1
    del fX_dx
    fY_dy_1 = fY[1:]
    fY_dy_0 = fY[0:-1]
    fY_dy = fY_dy_1-fY_dy_0
    fY_dy_min = np.nanmin(fY_dy)
    fY_dy_max = np.nanmax(fY_dy)
    del fY_dy_0
    del fY_dy_1
    del fY_dy
    fZ_dz_1 = fZ[1:]
    fZ_dz_0 = fZ[0:-1]
    fZ_dz = fZ_dz_1-fZ_dz_0
    fZ_dz_min = np.nanmin(fZ_dz)
    fZ_dz_max = np.nanmax(fZ_dz)
    del fZ_dz_0
    del fZ_dz_1
    del fZ_dz
    lateral_gres = min(fX_dx_min, fY_dy_max)
    vertical_gres = fZ_dz_min

    del fX
    del fY
    del fZ
    print("sX - {}".format(sX))
    print("sY - {}".format(sY))
    print("sZ - {}".format(sZ))
    print("sT - {}".format(sT))
    print("fX ext. (out) - {}".format(fX_ext))
    print("fY ext. (out) - {}".format(fY_ext))
    print("fZ ext. (out) - {}".format(fZ_ext))
    print("fT ext. (out) - {}".format(fT_ext))
    print("fX_dx - min: {}; max {}".format(fX_dx_min, fX_dx_max))
    print("fY_dy - min: {}; max {}".format(fY_dy_min, fY_dy_max))
    print("fZ_dz - min: {}; max {}".format(fZ_dz_min, fZ_dz_max))
    print("lateral gres: {}; vertical gres: {}".format(lateral_gres, vertical_gres))
    dt = fT[1] - fT[0]
    print("dT: {}; fT_dt: {}".format(dt, fT_dt))
    gc.collect()

    # dump_field_plain = args.plain_field_dump
    # dump_particle_plain = args.plain_particle_dump

    # ==== write original data ==== #
    if dump_field_plain:
        if "nc" in fileformat:
            if multifile:
                total_items = 3 * 2 * fT.shape[0] * fX_shape[0] * fY_shape[0] * fZ_shape[0]
                current_item = 0
                workdone = 0
                i = 0
                for fpath in uvel_fpath_nc:
                    f_u = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                    fX_ft_nc = f_u.variables[xuvar].data
                    fY_ft_nc = f_u.variables[yuvar].data
                    fZ_ft_nc = np.array([0.0], dtype=np.float32)
                    if is3D:
                        fZ_ft_nc = f_u.variables[zuvar].data if zuvar in f_u.variables.keys() else fZ_ft_nc
                    fT_ft_nc = f_u.variables[tuvar].data
                    fU_ft_nc = f_u.variables[uvar].data
                    ndims = 2
                    ndims = ndims+1 if is3D else ndims
                    # dims = (fU_ft_nc.shape[1], fU_ft_nc.shape[2], fU_ft_nc.shape[3]) if is3D else (fU_ft_nc.shape[1], fU_ft_nc.shape[2])
                    dims = (fU_ft_nc.shape[3], fU_ft_nc.shape[2], fU_ft_nc.shape[1]) if is3D else (fU_ft_nc.shape[2], fU_ft_nc.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                    pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                    psize = 1
                    for sdim in pdims:
                        psize *= sdim
                    print("U - dsize: {}; psize: {}".format(dsize, psize))

                    for ti in range(fT_ft_nc.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        udata_vtk = vtkDoubleArray()
                        udata_vtk.SetNumberOfComponents(1)
                        udata_vtk.SetNumberOfTuples(dsize)
                        points_vtk = vtkPoints()
                        # points_vtk.Allocate(int(psize))

                        offset = 0
                        pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        zDim = 1 if not is3D else fZ_ft_nc.shape[0]
                        for di in range(zDim):
                            # kOffset = di * pdims[1] * pdims[0]
                            depth = 0.0  if not is3D else fZ_ft_nc[di]
                            for lati in range(fY_ft_nc.shape[0]):
                                # jOffset = lati * pdims[0]
                                lat = fY_ft_nc[lati]
                                for loni in range(fX_ft_nc.shape[0]):
                                    # offset = loni + jOffset + kOffset
                                    lon = fX_ft_nc[loni]
                                    pt[0] = lon
                                    pt[1] = lat
                                    pt[2] = depth
                                    points_vtk.InsertNextPoint(pt)
                                    # points_vtk.SetPoint(offset, pt)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fU_ft_nc = np.nan_to_num(fU_ft_nc, nan=0.0)
                        zDim = 1 if not is3D else fU_ft_nc.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fU_ft_nc.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fU_ft_nc.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fU_ft_nc[ti, zi, yi, xi] if is3D else fU_ft_nc[ti, yi, xi]
                                    # udata_vtk.InsertNextValue(val)
                                    udata_vtk.SetValue(offset, val)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(udata_vtk)
                        sgrid.GetPointData().SetScalars(udata_vtk)

                        sgrid_file = os.path.join(outdir, "U_original_%d_%d.vts" % (i, ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()

                        del sgrid
                        del udata_vtk
                        del points_vtk
                        gc.collect()

                    print("\nGenerated NetCDF U data.")
                    del fX_ft_nc
                    del fY_ft_nc
                    del fZ_ft_nc
                    del fT_ft_nc
                    del fU_ft_nc

                    f_u.close()
                    del f_u
                    i += 1
                i = 0
                for fpath in vvel_fpath_nc:
                    f_v = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                    fX_ft_nc = f_v.variables[xvvar].data
                    fY_ft_nc = f_v.variables[yvvar].data
                    fZ_ft_nc = np.array([0.0], dtype=np.float32)
                    if is3D:
                        fZ_ft_nc = f_v.variables[zvvar].data if zvvar in f_v.variables.keys() else fZ_ft_nc
                    fT_ft_nc = f_v.variables[tvvar].data
                    fV_ft_nc = f_v.variables[vvar].data
                    ndims = 2
                    ndims = ndims+1 if is3D else ndims
                    # dims = (fV_ft_nc.shape[1], fV_ft_nc.shape[2], fV_ft_nc.shape[3]) if is3D else (fV_ft_nc.shape[1], fV_ft_nc.shape[2])
                    dims = (fV_ft_nc.shape[3], fV_ft_nc.shape[2], fV_ft_nc.shape[1]) if is3D else (fV_ft_nc.shape[2], fV_ft_nc.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                    pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                    psize = 1
                    for sdim in pdims:
                        psize *= sdim
                    print("V - dsize: {}; psize: {}".format(dsize, psize))

                    for ti in range(fT_ft_nc.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        vdata_vtk = vtkDoubleArray()
                        vdata_vtk.SetNumberOfComponents(1)
                        vdata_vtk.SetNumberOfTuples(dsize)
                        points_vtk = vtkPoints()
                        # points_vtk.Allocate(int(psize))

                        offset = 0
                        pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        zDim = 1 if not is3D else fZ_ft_nc.shape[0]
                        for di in range(zDim):
                            # kOffset = di * pdims[1] * pdims[0]
                            depth = 0.0  if not is3D else fZ_ft_nc[di]
                            for lati in range(fY_ft_nc.shape[0]):
                                # jOffset = lati * pdims[0]
                                lat = fY_ft_nc[lati]
                                for loni in range(fX_ft_nc.shape[0]):
                                    # offset = loni + jOffset + kOffset
                                    lon = fX_ft_nc[loni]
                                    pt[0] = lon
                                    pt[1] = lat
                                    pt[2] = depth
                                    points_vtk.InsertNextPoint(pt)
                                    # points_vtk.SetPoint(offset, pt)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fV_ft_nc = np.nan_to_num(fV_ft_nc, nan=0.0)
                        zDim = 1 if not is3D else fV_ft_nc.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fV_ft_nc.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fV_ft_nc.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fV_ft_nc[ti, zi, yi, xi] if is3D else fV_ft_nc[ti, yi, xi]
                                    # vdata_vtk.InsertNextValue(val)
                                    vdata_vtk.SetValue(offset, val)
                                    current_item += 1

                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(vdata_vtk)
                        sgrid.GetPointData().SetScalars(vdata_vtk)

                        sgrid_file = os.path.join(outdir, "V_original_%d_%d.vts" % (i, ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()

                        del sgrid
                        del vdata_vtk
                        del points_vtk
                        gc.collect()

                    print("\nGenerated NetCDF V data.")
                    del fX_ft_nc
                    del fY_ft_nc
                    del fZ_ft_nc
                    del fT_ft_nc
                    del fV_ft_nc

                    f_v.close()
                    del f_v
                    i += 1
                i = 0
                if hasW:
                    for fpath in wvel_fpath_nc:
                        f_w = xr.open_dataset(fpath, decode_cf=True, engine='netcdf4')
                        fX_ft_nc = f_w.variables[xwvar].data
                        fY_ft_nc = f_w.variables[ywvar].data
                        fZ_ft_nc = np.array([0.0], dtype=np.float32)
                        if is3D:
                            fZ_ft_nc = f_w.variables[zwvar].data if zwvar in f_w.variables.keys() else fZ_ft_nc
                        fT_ft_nc = f_w.variables[twvar].data
                        fW_ft_nc = f_w.variables[wvar].data
                        ndims = 2
                        ndims = ndims+1 if is3D else ndims
                        # dims = (fW_ft_nc.shape[1], fW_ft_nc.shape[2], fW_ft_nc.shape[3]) if is3D else (fW_ft_nc.shape[1], fW_ft_nc.shape[2])
                        dims = (fW_ft_nc.shape[3], fW_ft_nc.shape[2], fW_ft_nc.shape[1]) if is3D else (fW_ft_nc.shape[2], fW_ft_nc.shape[1])
                        dsize = 1
                        for sdim in dims:
                            dsize *= sdim

                        # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                        pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                        psize = 1
                        for sdim in pdims:
                            psize *= sdim
                        print("W - dsize: {}; psize: {}".format(dsize, psize))

                        for ti in range(fT_ft_nc.shape[0]):
                            sgrid = vtkStructuredGrid()
                            sgrid.SetDimensions(dims)
                            wdata_vtk = vtkDoubleArray()
                            wdata_vtk.SetNumberOfComponents(1)
                            wdata_vtk.SetNumberOfTuples(dsize)
                            points_vtk = vtkPoints()
                            # points_vtk.Allocate(int(psize))

                            offset = 0
                            pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            zDim = 1 if not is3D else fZ_ft_nc.shape[0]
                            for di in range(zDim):
                                # kOffset = di * pdims[1] * pdims[0]
                                depth = 0.0  if not is3D else fZ_ft_nc[di]
                                for lati in range(fY_ft_nc.shape[0]):
                                    # jOffset = lati * pdims[0]
                                    lat = fY_ft_nc[lati]
                                    for loni in range(fX_ft_nc.shape[0]):
                                        # offset = loni + jOffset + kOffset
                                        lon = fX_ft_nc[loni]
                                        pt[0] = lon
                                        pt[1] = lat
                                        pt[2] = depth
                                        points_vtk.InsertNextPoint(pt)
                                        # points_vtk.SetPoint(offset, pt)
                                        current_item += 1
                                workdone = current_item / total_items
                                print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                            offset = 0
                            with np.errstate(invalid='ignore'):
                                fW_ft_nc = np.nan_to_num(fW_ft_nc, nan=0.0)
                            zDim = 1 if not is3D else fW_ft_nc.shape[1]
                            for zi in range(zDim):
                                kOffset = zi * dims[1] * dims[0]
                                for yi in range(fW_ft_nc.shape[2]):
                                    jOffset = yi * dims[0]
                                    for xi in range(fW_ft_nc.shape[3]):
                                        offset = xi + jOffset + kOffset
                                        val = fW_ft_nc[ti, zi, yi, xi] if is3D else fW_ft_nc[ti, yi, xi]
                                        # wdata_vtk.InsertNextValue(val)
                                        wdata_vtk.SetValue(offset, val)
                                        current_item += 1
                                workdone = current_item / total_items
                                print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                            sgrid.SetPoints(points_vtk)
                            # sgrid.GetCellData().SetScalars(wdata_vtk)
                            sgrid.GetPointData().SetScalars(wdata_vtk)

                            sgrid_file = os.path.join(outdir, "W_original_%d_%d.vts" % (i, ti))
                            writer = vtkXMLStructuredGridWriter()
                            writer.SetFileName(sgrid_file)
                            writer.SetInputData(sgrid)
                            writer.Write()

                            del sgrid
                            del wdata_vtk
                            del points_vtk
                            gc.collect()

                        print("\nGenerated NetCDF W data.")
                        del fX_ft_nc
                        del fY_ft_nc
                        del fZ_ft_nc
                        del fT_ft_nc
                        del fW_ft_nc

                        f_w.close()
                        del f_w
                        i += 1
            else:
                total_items = 3 * 2 * fT.shape[0] * fX_shape[0] * fY_shape[0] * fZ_shape[0]
                current_item = 0
                workdone = 0
                # ======== u-velocity ======== #
                f_u = xr.open_dataset(uvel_fpath_nc, decode_cf=True, engine='netcdf4')
                fX_ft_nc = f_u.variables[xuvar].data
                fY_ft_nc = f_u.variables[yuvar].data
                fZ_ft_nc = np.array([0.0], dtype=np.float32)
                if is3D:
                    fZ_ft_nc = f_u.variables[zuvar].data if zuvar in f_u.variables.keys() else fZ_ft_nc
                fT_ft_nc = f_u.variables[tuvar].data
                fU_ft_nc = f_u.variables[uvar].data
                ndims = 2
                ndims = ndims + 1 if is3D else ndims
                # dims = (fU_ft_nc.shape[1], fU_ft_nc.shape[2], fU_ft_nc.shape[3]) if is3D else (fU_ft_nc.shape[1], fU_ft_nc.shape[2])
                dims = (fU_ft_nc.shape[3], fU_ft_nc.shape[2], fU_ft_nc.shape[1]) if is3D else (fU_ft_nc.shape[2], fU_ft_nc.shape[1])
                dsize = 1
                for sdim in dims:
                    dsize *= sdim

                # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                psize = 1
                for sdim in pdims:
                    psize *= sdim

                for ti in range(fT_ft_nc.shape[0]):
                    sgrid = vtkStructuredGrid()
                    sgrid.SetDimensions(dims)
                    udata_vtk = vtkDoubleArray()
                    udata_vtk.SetNumberOfComponents(1)
                    udata_vtk.SetNumberOfTuples(dsize)
                    udata_vtk.SetName("U")
                    points_vtk = vtkPoints()
                    # points_vtk.Allocate(int(psize))

                    offset = 0
                    pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    zDim = 1 if not is3D else fZ_ft_nc.shape[0]
                    for di in range(zDim):
                        # kOffset = di * pdims[1] * pdims[0]
                        depth = fZ_ft_nc[di]
                        for lati in range(fY_ft_nc.shape[0]):
                            # jOffset = lati * pdims[0]
                            lat = fY_ft_nc[lati]
                            for loni in range(fX_ft_nc.shape[0]):
                                # offset = loni + jOffset + kOffset
                                lon = fX_ft_nc[loni]
                                pt[0] = lon
                                pt[1] = lat
                                pt[2] = depth
                                points_vtk.InsertNextPoint(pt)
                                # points_vtk.SetPoint(offset, pt)
                                current_item += 1
                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    offset = 0
                    with np.errstate(invalid='ignore'):
                        fU_ft_nc = np.nan_to_num(fU_ft_nc, nan=0.0)
                    zDim = 1 if not is3D else fU_ft_nc.shape[1]
                    for zi in range(zDim):
                        kOffset = zi * dims[1] * dims[0]
                        for yi in range(fU_ft_nc.shape[2]):
                            jOffset = yi * dims[0]
                            for xi in range(fU_ft_nc.shape[3]):
                                offset = xi + jOffset + kOffset
                                val = fU_ft_nc[ti, zi, yi, xi] if is3D else fU_ft_nc[ti, yi, xi]
                                # udata_vtk.InsertNextValue(val)
                                udata_vtk.SetValue(offset, val)
                                current_item += 1

                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    sgrid.SetPoints(points_vtk)
                    # sgrid.GetCellData().SetScalars(udata_vtk)
                    sgrid.GetPointData().SetScalars(udata_vtk)
                    sgrid.GetPointData().SetActiveScalars("U")

                    sgrid_file = os.path.join(outdir, "U_original_%d.vts" % (ti))
                    writer = vtkXMLStructuredGridWriter()
                    writer.SetFileName(sgrid_file)
                    writer.SetInputData(sgrid)
                    writer.Write()

                print("\nGenerated NetCDF U data.")
                del fX_ft_nc
                del fY_ft_nc
                del fZ_ft_nc
                del fT_ft_nc
                del fU_ft_nc

                f_u.close()
                del f_u
                # ======== v-velocity ======== #
                f_v = xr.open_dataset(vvel_fpath_nc, decode_cf=True, engine='netcdf4')
                fX_ft_nc = f_v.variables[xvvar].data
                fY_ft_nc = f_v.variables[yvvar].data
                fZ_ft_nc = np.array([0.0], dtype=np.float32)
                if is3D:
                    fZ_ft_nc = f_v.variables[zvvar].data if zvvar in f_v.variables.keys() else fZ_ft_nc

                fT_ft_nc = f_v.variables[tvvar].data
                fV_ft_nc = f_v.variables[vvar].data
                ndims = 2
                ndims = ndims + 1 if is3D else ndims
                # dims = (fV_ft_nc.shape[1], fV_ft_nc.shape[2], fV_ft_nc.shape[3]) if is3D else (fV_ft_nc.shape[1], fV_ft_nc.shape[2])
                dims = (fV_ft_nc.shape[3], fV_ft_nc.shape[2], fV_ft_nc.shape[1]) if is3D else (fV_ft_nc.shape[2], fV_ft_nc.shape[1])
                dsize = 1
                for sdim in dims:
                    dsize *= sdim

                # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                psize = 1
                for sdim in pdims:
                    psize *= sdim

                for ti in range(fT_ft_nc.shape[0]):
                    sgrid = vtkStructuredGrid()
                    sgrid.SetDimensions(dims)
                    vdata_vtk = vtkDoubleArray()
                    vdata_vtk.SetNumberOfComponents(1)
                    vdata_vtk.SetNumberOfTuples(dsize)
                    vdata_vtk.SetName("V")
                    points_vtk = vtkPoints()
                    # points_vtk.Allocate(int(psize))

                    offset = 0
                    pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    zDim = 1 if not is3D else fZ_ft_nc.shape[0]
                    for di in range(zDim):
                        # kOffset = di * pdims[1] * pdims[0]
                        depth = fZ_ft_nc[di]
                        for lati in range(fY_ft_nc.shape[0]):
                            # jOffset = lati * pdims[0]
                            lat = fY_ft_nc[lati]
                            for loni in range(fX_ft_nc.shape[0]):
                                # offset = loni + jOffset + kOffset
                                lon = fX_ft_nc[loni]
                                pt[0] = lon
                                pt[1] = lat
                                pt[2] = depth
                                points_vtk.InsertNextPoint(pt)
                                # points_vtk.SetPoint(offset, pt)
                                current_item += 1
                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    offset = 0
                    with np.errstate(invalid='ignore'):
                        fV_ft_nc = np.nan_to_num(fV_ft_nc, nan=0.0)
                    zDim = 1 if not is3D else fV_ft_nc.shape[1]
                    for zi in range(zDim):
                        kOffset = zi * dims[1] * dims[0]
                        for yi in range(fV_ft_nc.shape[2]):
                            jOffset = yi * dims[0]
                            for xi in range(fV_ft_nc.shape[3]):
                                offset = xi + jOffset + kOffset
                                val = fV_ft_nc[ti, zi, yi, xi] if is3D else fV_ft_nc[ti, yi, xi]
                                # vdata_vtk.InsertNextValue(val)
                                vdata_vtk.SetValue(offset, val)
                                current_item += 1
                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    sgrid.SetPoints(points_vtk)
                    # sgrid.GetCellData().SetScalars(vdata_vtk)
                    sgrid.GetPointData().SetScalars(vdata_vtk)
                    sgrid.GetPointData().SetActiveScalars("V")

                    sgrid_file = os.path.join(outdir, "V_original_%d.vts" % (ti))
                    writer = vtkXMLStructuredGridWriter()
                    writer.SetFileName(sgrid_file)
                    writer.SetInputData(sgrid)
                    writer.Write()

                print("\nGenerated NetCDF V data.")
                del fX_ft_nc
                del fY_ft_nc
                del fZ_ft_nc
                del fT_ft_nc
                del fV_ft_nc

                f_v.close()
                del f_v
                # ======== w-velocity ======== #
                if hasW:
                    f_w = xr.open_dataset(wvel_fpath_nc, decode_cf=True, engine='netcdf4')
                    fX_ft_nc = f_w.variables[xwvar].data
                    fY_ft_nc = f_w.variables[ywvar].data
                    fZ_ft_nc = np.array([0.0], dtype=np.float32)
                    if is3D:
                        fZ_ft_nc = f_w.variables[zwvar].data if zwvar in f_w.variables.keys() else fZ_ft_nc
                    fT_ft_nc = f_w.variables[twvar].data
                    fW_ft_nc = f_w.variables[wvar].data
                    ndims = 2
                    ndims = ndims + 1 if is3D else ndims
                    # dims = (fW_ft_nc.shape[1], fW_ft_nc.shape[2], fW_ft_nc.shape[3]) if is3D else (fW_ft_nc.shape[1], fW_ft_nc.shape[2])
                    dims = (fW_ft_nc.shape[3], fW_ft_nc.shape[2], fW_ft_nc.shape[1]) if is3D else (fW_ft_nc.shape[2], fW_ft_nc.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    # pdims = (fZ_ft_nc.shape[0], fY_ft_nc.shape[0], fX_ft_nc.shape[0])
                    pdims = (fX_ft_nc.shape[0], fY_ft_nc.shape[0], fZ_ft_nc.shape[0]) if is3D else (fX_ft_nc.shape[0], fY_ft_nc.shape[0])
                    psize = 1
                    for sdim in pdims:
                        psize *= sdim

                    for ti in range(fT_ft_nc.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        wdata_vtk = vtkDoubleArray()
                        wdata_vtk.SetNumberOfComponents(1)
                        wdata_vtk.SetNumberOfTuples(dsize)
                        wdata_vtk.SetName("W")
                        points_vtk = vtkPoints()

                        offset = 0
                        pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        for di in range(fZ_ft_nc.shape[0]):
                            # kOffset = di * pdims[1] * pdims[0]
                            depth = fZ_ft_nc[di]
                            for lati in range(fY_ft_nc.shape[0]):
                                # jOffset = lati * pdims[0]
                                lat = fY_ft_nc[lati]
                                for loni in range(fX_ft_nc.shape[0]):
                                    # offset = loni + jOffset + kOffset
                                    lon = fX_ft_nc[loni]
                                    pt[0] = lon
                                    pt[1] = lat
                                    pt[2] = depth
                                    points_vtk.InsertNextPoint(pt)
                                    # points_vtk.SetPoint(offset, pt)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fW_ft_nc = np.nan_to_num(fW_ft_nc, nan=0.0)
                        zDim = 1 if not is3D else fW_ft_nc.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fW_ft_nc.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fW_ft_nc.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fW_ft_nc[ti, zi, yi, xi] if is3D else fW_ft_nc[ti, yi, xi]
                                    # wdata_vtk.InsertNextValue(val)
                                    wdata_vtk.SetValue(offset, val)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(wdata_vtk)
                        sgrid.GetPointData().SetScalars(wdata_vtk)
                        sgrid.GetPointData().SetActiveScalars("W")

                        sgrid_file = os.path.join(outdir, "W_original_%d.vts" % (ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()

                    print("\nGenerated NetCDF W data.")
                    del fX_ft_nc
                    del fY_ft_nc
                    del fZ_ft_nc
                    del fT_ft_nc
                    del fW_ft_nc

                    f_w.close()
                    del f_w
        elif "h5" in fileformat:
            if multifile:
                total_items = 3 * 2 * fT.shape[0] * fX_shape[0] * fY_shape[0] * fZ_shape[0]
                current_item = 0
                workdone = 0
                ndims = 2
                ndims = ndims + 1 if is3D else ndims
                f_grid = h5py.File(grid_fpath_h5, "r")
                fX_ft_h5 = f_grid['longitude'][()]
                fY_ft_h5 = f_grid['latitude'][()]
                fZ_ft_h5 = np.array([0.0], dtype=np.float32)
                if is3D:
                    fZ_ft_h5 = f_grid['depths'][()] if 'depths' in f_grid else fZ_ft_h5
                fT_ft_h5 = f_grid['times'][()]

                pdims = (fX_ft_h5.shape[0], fY_ft_h5.shape[0], fZ_ft_h5.shape[0]) if is3D else (fX_ft_h5.shape[0], fY_ft_h5.shape[0])
                psize = 1
                for sdim in pdims:
                    psize *= sdim

                points_vtk = vtkPoints()
                offset = 0
                pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                zDim = 1 if not is3D else fZ_ft_h5.shape[0]
                for di in range(zDim):
                    # kOffset = di * pdims[1] * pdims[0]
                    depth = fZ_ft_h5[di]
                    for lati in range(fY_ft_h5.shape[0]):
                        # jOffset = lati * pdims[0]
                        lat = fY_ft_h5[lati]
                        for loni in range(fX_ft_h5.shape[0]):
                            # offset = loni + jOffset + kOffset
                            lon = fX_ft_h5[loni]
                            pt[0] = lon
                            pt[1] = lat
                            pt[2] = depth
                            points_vtk.InsertNextPoint(pt)
                            # points_vtk.SetPoint(offset, pt)
                            current_item += 1
                    workdone = current_item / total_items
                    print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)

                f_grid.close()
                del f_grid
                # ======== u-velocity ======== #
                i = 0
                for fpath in uvel_fpath_h5:
                    f_u = h5py.File(fpath, "r")
                    fU_ft_h5 = f_u[uvar][()]

                    dims = (fU_ft_h5.shape[3], fU_ft_h5.shape[2], fU_ft_h5.shape[1]) if is3D else (fU_ft_h5.shape[2], fU_ft_h5.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    for ti in range(fT_ft_h5.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        udata_vtk = vtkDoubleArray()
                        udata_vtk.SetNumberOfComponents(1)
                        udata_vtk.SetNumberOfTuples(dsize)
                        udata_vtk.SetName("U")
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fU_ft_h5 = np.nan_to_num(fU_ft_h5, nan=0.0)
                        zDim = 1 if not is3D else fU_ft_h5.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fU_ft_h5.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fU_ft_h5.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fU_ft_h5[ti, zi, yi, xi] if is3D else fU_ft_h5[ti, yi, xi]
                                    # udata_vtk.InsertNextValue(val)
                                    udata_vtk.SetValue(offset, val)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress u-data: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(udata_vtk)
                        sgrid.GetPointData().SetScalars(udata_vtk)
                        sgrid.GetPointData().SetActiveScalars("U")

                        sgrid_file = os.path.join(outdir, "U_original_%d_%d.vts" % (i, ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()
                    print("\nGenerated NetCDF U data.")
                    f_u.close()
                    del fU_ft_h5
                    del f_u
                    i += 1
                # ======== v-velocity ======== #
                i = 0
                for fpath in vvel_fpath_h5:
                    f_v = h5py.File(fpath, "r")
                    fV_ft_h5 = f_v[uvar][()]

                    dims = (fV_ft_h5.shape[3], fV_ft_h5.shape[2], fV_ft_h5.shape[1]) if is3D else (fV_ft_h5.shape[2], fV_ft_h5.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    for ti in range(fT_ft_h5.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        vdata_vtk = vtkDoubleArray()
                        vdata_vtk.SetNumberOfComponents(1)
                        vdata_vtk.SetNumberOfTuples(dsize)
                        vdata_vtk.SetName("V")
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fV_ft_h5 = np.nan_to_num(fV_ft_h5, nan=0.0)
                        zDim = 1 if not is3D else fV_ft_h5.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fV_ft_h5.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fV_ft_h5.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fV_ft_h5[ti, zi, yi, xi] if is3D else fV_ft_h5[ti, yi, xi]
                                    # vdata_vtk.InsertNextValue(val)
                                    vdata_vtk.SetValue(offset, val)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress v-data: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(vdata_vtk)
                        sgrid.GetPointData().SetScalars(vdata_vtk)
                        sgrid.GetPointData().SetActiveScalars("V")

                        sgrid_file = os.path.join(outdir, "V_original_%d_%d.vts" % (i, ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()
                    print("\nGenerated NetCDF V data.")
                    f_v.close()
                    del fV_ft_h5
                    del f_v
                    i += 1
                i = 0
                # ======== w-velocity ======== #
                if hasW:
                    for fpath in wvel_fpath_h5:
                        f_w = h5py.File(fpath, "r")
                        fW_ft_h5 = f_w[uvar][()]

                        dims = (fW_ft_h5.shape[3], fW_ft_h5.shape[2], fW_ft_h5.shape[1]) if is3D else (
                        fW_ft_h5.shape[2], fW_ft_h5.shape[1])
                        dsize = 1
                        for sdim in dims:
                            dsize *= sdim

                        for ti in range(fT_ft_h5.shape[0]):
                            sgrid = vtkStructuredGrid()
                            sgrid.SetDimensions(dims)
                            wdata_vtk = vtkDoubleArray()
                            wdata_vtk.SetNumberOfComponents(1)
                            wdata_vtk.SetNumberOfTuples(dsize)
                            wdata_vtk.SetName("W")
                            offset = 0
                            with np.errstate(invalid='ignore'):
                                fW_ft_h5 = np.nan_to_num(fW_ft_h5, nan=0.0)
                            zDim = 1 if not is3D else fW_ft_h5.shape[1]
                            for zi in range(zDim):
                                kOffset = zi * dims[1] * dims[0]
                                for yi in range(fW_ft_h5.shape[2]):
                                    jOffset = yi * dims[0]
                                    for xi in range(fW_ft_h5.shape[3]):
                                        offset = xi + jOffset + kOffset
                                        val = fW_ft_h5[ti, zi, yi, xi] if is3D else fW_ft_h5[ti, yi, xi]
                                        # wdata_vtk.InsertNextValue(val)
                                        wdata_vtk.SetValue(offset, val)
                                        current_item += 1
                                workdone = current_item / total_items
                                print("\rProgress w-data: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset),
                                      end="", flush=True)
                            sgrid.SetPoints(points_vtk)
                            # sgrid.GetCellData().SetScalars(wdata_vtk)
                            sgrid.GetPointData().SetScalars(wdata_vtk)
                            sgrid.GetPointData().SetActiveScalars("W")

                            sgrid_file = os.path.join(outdir, "W_original_%d_%d.vts" % (i, ti))
                            writer = vtkXMLStructuredGridWriter()
                            writer.SetFileName(sgrid_file)
                            writer.SetInputData(sgrid)
                            writer.Write()
                        print("\nGenerated NetCDF W data.")
                        f_w.close()
                        del fW_ft_h5
                        del f_w
                        i += 1

                del fX_ft_h5
                del fY_ft_h5
                del fZ_ft_h5
                del fT_ft_h5
            else:
                total_items = 3 * 2 * fT.shape[0] * fX_shape[0] * fY_shape[0] * fZ_shape[0]
                current_item = 0
                workdone = 0
                ndims = 2
                ndims = ndims + 1 if is3D else ndims
                f_grid = h5py.File(grid_fpath_h5, "r")
                fX_ft_h5 = f_grid['longitude'][()]
                fY_ft_h5 = f_grid['latitude'][()]
                fZ_ft_h5 = np.array([0.0], dtype=np.float32)
                if is3D:
                    fZ_ft_h5 = f_grid['depths'][()] if 'depths' in f_grid else fZ_ft_h5
                fT_ft_h5 = f_grid['times'][()]

                pdims = (fX_ft_h5.shape[0], fY_ft_h5.shape[0], fZ_ft_h5.shape[0]) if is3D else (fX_ft_h5.shape[0], fY_ft_h5.shape[0])
                psize = 1
                for sdim in pdims:
                    psize *= sdim

                points_vtk = vtkPoints()
                offset = 0
                pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                zDim = 1 if not is3D else fZ_ft_h5.shape[0]
                for di in range(zDim):
                    # kOffset = di * pdims[1] * pdims[0]
                    depth = fZ_ft_h5[di]
                    for lati in range(fY_ft_h5.shape[0]):
                        # jOffset = lati * pdims[0]
                        lat = fY_ft_h5[lati]
                        for loni in range(fX_ft_h5.shape[0]):
                            # offset = loni + jOffset + kOffset
                            lon = fX_ft_h5[loni]
                            pt[0] = lon
                            pt[1] = lat
                            pt[2] = depth
                            points_vtk.InsertNextPoint(pt)
                            # points_vtk.SetPoint(offset, pt)
                            current_item += 1
                    workdone = current_item / total_items
                    print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)

                f_grid.close()
                del f_grid
                # ======== u-velocity ======== #
                f_u = h5py.File(uvel_fpath_h5, "r")
                fU_ft_h5 = f_u[uvar][()]

                dims = (fU_ft_h5.shape[3], fU_ft_h5.shape[2], fU_ft_h5.shape[1]) if is3D else (fU_ft_h5.shape[2], fU_ft_h5.shape[1])
                dsize = 1
                for sdim in dims:
                    dsize *= sdim

                for ti in range(fT_ft_h5.shape[0]):
                    sgrid = vtkStructuredGrid()
                    sgrid.SetDimensions(dims)
                    udata_vtk = vtkDoubleArray()
                    udata_vtk.SetNumberOfComponents(1)
                    udata_vtk.SetNumberOfTuples(dsize)
                    udata_vtk.SetName("U")
                    offset = 0
                    with np.errstate(invalid='ignore'):
                        fU_ft_h5 = np.nan_to_num(fU_ft_h5, nan=0.0)
                    zDim = 1 if not is3D else fU_ft_h5.shape[1]
                    for zi in range(zDim):
                        kOffset = zi * dims[1] * dims[0]
                        for yi in range(fU_ft_h5.shape[2]):
                            jOffset = yi * dims[0]
                            for xi in range(fU_ft_h5.shape[3]):
                                offset = xi + jOffset + kOffset
                                val = fU_ft_h5[ti, zi, yi, xi] if is3D else fU_ft_h5[ti, yi, xi]
                                # udata_vtk.InsertNextValue(val)
                                udata_vtk.SetValue(offset, val)
                                current_item += 1
                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    sgrid.SetPoints(points_vtk)
                    # sgrid.GetCellData().SetScalars(udata_vtk)
                    sgrid.GetPointData().SetScalars(udata_vtk)
                    sgrid.GetPointData().SetActiveScalars("U")

                    sgrid_file = os.path.join(outdir, "U_original_%d.vts" % (ti))
                    writer = vtkXMLStructuredGridWriter()
                    writer.SetFileName(sgrid_file)
                    writer.SetInputData(sgrid)
                    writer.Write()
                print("\nGenerated NetCDF U data.")
                f_u.close()
                del fU_ft_h5
                del f_u
                # ======== v-velocity ======== #
                f_v = h5py.File(vvel_fpath_h5, "r")
                fV_ft_h5 = f_v[uvar][()]

                dims = (fV_ft_h5.shape[3], fV_ft_h5.shape[2], fV_ft_h5.shape[1]) if is3D else (fV_ft_h5.shape[2], fV_ft_h5.shape[1])
                dsize = 1
                for sdim in dims:
                    dsize *= sdim

                for ti in range(fT_ft_h5.shape[0]):
                    sgrid = vtkStructuredGrid()
                    sgrid.SetDimensions(dims)
                    vdata_vtk = vtkDoubleArray()
                    vdata_vtk.SetNumberOfComponents(1)
                    vdata_vtk.SetNumberOfTuples(dsize)
                    vdata_vtk.SetName("V")
                    offset = 0
                    with np.errstate(invalid='ignore'):
                        fV_ft_h5 = np.nan_to_num(fV_ft_h5, nan=0.0)
                    zDim = 1 if not is3D else fV_ft_h5.shape[1]
                    for zi in range(zDim):
                        kOffset = zi * dims[1] * dims[0]
                        for yi in range(fV_ft_h5.shape[2]):
                            jOffset = yi * dims[0]
                            for xi in range(fV_ft_h5.shape[3]):
                                offset = xi + jOffset + kOffset
                                val = fV_ft_h5[ti, zi, yi, xi] if is3D else fV_ft_h5[ti, yi, xi]
                                # vdata_vtk.InsertNextValue(val)
                                vdata_vtk.SetValue(offset, val)
                                current_item += 1
                        workdone = current_item / total_items
                        print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                    sgrid.SetPoints(points_vtk)
                    # sgrid.GetCellData().SetScalars(vdata_vtk)
                    sgrid.GetPointData().SetScalars(vdata_vtk)
                    sgrid.GetPointData().SetActiveScalars("V")

                    sgrid_file = os.path.join(outdir, "V_original_%d.vts" % (ti))
                    writer = vtkXMLStructuredGridWriter()
                    writer.SetFileName(sgrid_file)
                    writer.SetInputData(sgrid)
                    writer.Write()
                print("\nGenerated NetCDF V data.")
                f_v.close()
                del fV_ft_h5
                del f_v
                # ======== w-velocity ======== #
                if hasW:
                    f_w = h5py.File(wvel_fpath_h5, "r")
                    fW_ft_h5 = f_w[uvar][()]

                    dims = (fW_ft_h5.shape[3], fW_ft_h5.shape[2], fW_ft_h5.shape[1]) if is3D else (fW_ft_h5.shape[2], fW_ft_h5.shape[1])
                    dsize = 1
                    for sdim in dims:
                        dsize *= sdim

                    for ti in range(fT_ft_h5.shape[0]):
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(dims)
                        wdata_vtk = vtkDoubleArray()
                        wdata_vtk.SetNumberOfComponents(1)
                        wdata_vtk.SetNumberOfTuples(dsize)
                        wdata_vtk.SetName("W")
                        offset = 0
                        with np.errstate(invalid='ignore'):
                            fW_ft_h5 = np.nan_to_num(fW_ft_h5, nan=0.0)
                        zDim = 1 if not is3D else fW_ft_h5.shape[1]
                        for zi in range(zDim):
                            kOffset = zi * dims[1] * dims[0]
                            for yi in range(fW_ft_h5.shape[2]):
                                jOffset = yi * dims[0]
                                for xi in range(fW_ft_h5.shape[3]):
                                    offset = xi + jOffset + kOffset
                                    val = fW_ft_h5[ti, zi, yi, xi] if is3D else fW_ft_h5[ti, yi, xi]
                                    # wdata_vtk.InsertNextValue(val)
                                    wdata_vtk.SetValue(offset, val)
                                    current_item += 1
                            workdone = current_item / total_items
                            print("\rProgress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(workdone * 50), workdone * 100, offset), end="", flush=True)
                        sgrid.SetPoints(points_vtk)
                        # sgrid.GetCellData().SetScalars(wdata_vtk)
                        sgrid.GetPointData().SetScalars(wdata_vtk)
                        sgrid.GetPointData().SetActiveScalars("W")

                        sgrid_file = os.path.join(outdir, "W_original_%d.vts" % (ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()
                    print("\nGenerated NetCDF W data.")
                    f_w.close()
                    del fW_ft_h5
                    del f_w

                del fX_ft_h5
                del fY_ft_h5
                del fZ_ft_h5
                del fT_ft_h5


    indices = np.random.randint(0, 100, Pn, dtype=int)
    indices_set = False
    reserved_attribs = ['lon', 'lat', 'depth', 'depthu', 'depthv', 'depthw', 'x', 'y', 'z', 'trajectory', 'time']
    # ==== write particle data ==== #
    if dump_particle_plain:
        if "nc" in fileformat:
            pt_file = xr.open_dataset(particle_fpath, decode_cf=True, engine='netcdf4')
            pX = pt_file['lon']  # .data
            pY = pt_file['lat']  # .data
            pZ = None
            if 'depth' in pt_file.keys():
                pZ = pt_file['depth']  # .data
            elif 'z' in pt_file.keys():
                pZ = pt_file['z']  # .data
            N = int(pt_file['lon'].shape[0])
            tN = int(pt_file['lon'].shape[1])
            indices = np.random.randint(0, N - 1, Pn, dtype=int)
            indices_set = True
            if DBG_MSG:
                print("N: {}, t_N: {}".format(N, tN))
            valid_array = None
            if 'valid' in pt_file.keys():
                valid_array = np.maximum(np.max(np.array(pt_file['valid'][:, 0:2]), axis=1), 0).astype(np.bool_)
                if DBG_MSG:
                    print("Valid array: any true ? {}; all true ? {}".format(np.any(valid_array), np.all(valid_array)))
            else:
                pX = pX.data
                nonstationary = np.ones(N, dtype=np.bool_)
                for p_ti in range(1, tN):
                    px_f1_i = np.nonzero(np.isnan(pX[:, p_ti]))[0]
                    px_f0_i = np.nonzero(np.isnan(pX[:, p_ti-1]))[0]
                    pX_i = np.intersect1d(px_f0_i, px_f1_i)
                    px_f0 = pX[pX_i, p_ti-1]
                    if DBG_MSG:
                        print("px_f0 - shape: {}".format(px_f0.shape))
                    px_f1 = pX[pX_i, p_ti]
                    if DBG_MSG:
                        print("px_f1 - shape: {}".format(px_f1.shape))
                    if pX_i.shape[0] > 0:
                        nonstationary[pX_i] = np.logical_and(nonstationary[pX_i], ~np.isclose(px_f0, px_f1).squeeze())
                valid_array = nonstationary
                if DBG_MSG:
                    print("Valid array: any true ? {}; all true ? {}".format(np.any(valid_array), np.all(valid_array)))
                del pX
                pX = pt_file['lon']
            pT_init = pt_file['time'].data
            pT_min = np.nanmin(pT_init, axis=0)
            pT_max = np.nanmax(pT_init, axis=0)
            if DBG_MSG:
                print("Times:\n\tmin = {}\n\tmax = {}".format(pT_min, pT_max))
            assert pT_init.shape[1] == pT_min.shape[0]
            mask_array = valid_array
            for ti in range(pT_init.shape[1]):
                replace_indices = np.isnan(pT_init[:, ti])
                pT_init[replace_indices, ti] = pT_max[ti]  # this ONLY works if there is no delayed start
            if DBG_MSG:
                print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(pT_init.shape, type(pT_init[0, 0]), np.min(pT_init[0]), np.max(pT_init[0])))
            pT_0 = pT_max[0]
            pT = pT_init - pT_0
            if DBG_MSG:
                print("time info from file after baselining: \n\tshape = {}".format(pT.shape))
                print("\ttype = {}".format(type(pT[0, 0])))
                print("\trange = {}".format( (np.nanmin(pT), np.nanmax(pT)) ))
            pAttr = []
            for attribute in pt_file.keys():
                if attribute not in reserved_attribs:
                    pAttr.append(attribute)
            # pAttr = list(pt_file.keys())

            # px_t_0 = np.array(pX[:, ti0])
            # px_t_0 = np.take_along_axis(px_t_0, indices, axis=0)
            # py_t_0 = np.array(pY[:, ti0])
            # py_t_0 = np.take_along_axis(py_t_0, indices, axis=0)

            for p_ti in range(0, tN):
                pdata_vtk = vtkPolyData()
                points_vtk = vtkPoints()
                # points_vtk.Allocate(N)
                pX_ft = np.array(pX[:, p_ti])
                pY_ft = np.array(pY[:, p_ti])
                pZ_ft = np.array(pZ[:, p_ti])
                pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                # traji = 0
                for pi in range(N):
                    pt[0] = pX_ft[pi]
                    pt[1] = pY_ft[pi]
                    pt[2] = pZ_ft[pi]
                    # points_vtk.SetPoint(pi, pt)
                    points_vtk.InsertNextPoint(pt)
                    # if pi in indices:
                    #     traj_points_vtk.SetPoint(traji, pt)
                    #     # traj_points_vtk.InsertNextPoint(pt)
                    #     traji += 1
                pdata_vtk.SetPoints(points_vtk)

                for attribute in pAttr:
                    attr_array_vtk = vtkDoubleArray()
                    attr_array_vtk.SetNumberOfComponents(1)
                    attr_array_vtk.SetNumberOfTuples(N)
                    attr_array_vtk.SetName(attribute)
                    pattr_array = np.array(pt_file[attribute][:, p_ti])
                    for pi in range(N):
                        attr_array_vtk.SetValue(pi, pattr_array[pi])
                        # if pi in indices:
                        #     # traj_attr_vtk[attribute].SetValue(pi, pattr_array[pi])
                        #     traj_attr_vtk[attribute].InsertNextValue(pattr_array[pi])
                    pdata_vtk.GetPointData().AddArray(attr_array_vtk)
                    del pattr_array

                pdata_file = os.path.join(outdir, "Particles_original_%d.vtp" % (p_ti))
                pdata_writer = vtkXMLPolyDataWriter()
                # pdata_writer.SetFileTypeToBinary()
                pdata_writer.SetFileName(pdata_file)
                pdata_writer.SetInputData(pdata_vtk)
                pdata_writer.Write()


            trajectory_vtk = vtkPolyData()
            traj_points_vtk = vtkPoints()
            # traj_points_vtk.Allocate(Pn * tN)
            traj_attr_vtk = {}
            for attribute in pAttr:
                traj_attr_vtk[attribute] = vtkDoubleArray()
                traj_attr_vtk[attribute].SetNumberOfComponents(1)
                # traj_attr_vtk[attribute].SetNumberOfTuples(N)
                traj_attr_vtk[attribute].SetName(attribute)

            traj_cells_vtk = vtkCellArray()
            traj_polyline_vtk = []
            for traji in range(Pn):
                pline_vtk = vtkPolyLine()
                pline_vtk.GetPointIds().SetNumberOfIds(tN)
                traj_polyline_vtk.append(pline_vtk)

            for index in range(indices.shape[0]):
                pi = indices[index]
                pX_ft = np.array(pX[pi, :])
                pY_ft = np.array(pY[pi, :])
                pZ_ft = np.array(pZ[pi, :])
                pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                for p_ti in range(0, tN):
                    pt[0] = pX_ft[p_ti]
                    pt[1] = pY_ft[p_ti]
                    pt[2] = pZ_ft[p_ti]
                    pt_id = (index * tN) + p_ti
                    # traj_points_vtk.SetPoint(pt_id, pt)
                    traj_points_vtk.InsertNextPoint(pt)
                    traj_polyline_vtk[index].GetPointIds().SetId(p_ti, pt_id)
                for attribute in pAttr:
                    pattr_array = np.array(pt_file[attribute][pi, :])
                    for p_ti in range(0, tN):
                        traj_attr_vtk[attribute].InsertNextValue(pattr_array[p_ti])

            trajectory_vtk.SetPoints(traj_points_vtk)
            for pline in traj_polyline_vtk:
                traj_cells_vtk.InsertNextCell(pline)
            for attribute in pAttr:
                trajectory_vtk.GetPointData().AddArray(traj_attr_vtk[attribute])
            trajectory_vtk.SetLines(traj_cells_vtk)

            traj_file = os.path.join(outdir, "trajectories.vtp")
            traj_writer = vtkXMLPolyDataWriter()
            # traj_writer.SetFileTypeToBinary()
            traj_writer.SetFileName(traj_file)
            traj_writer.SetInputData(trajectory_vtk)
            traj_writer.Write()

            pt_file.close()
        elif "h5" in fileformat:
            # TODO ==== implement particles from HDF5 ==== #
            pass

    clamp = False
    if args.lonmin is not None and args.lonmax is not None:
        clamp = True
        lonmin = fX_ext[0] if args.lonmin is None else args.lonmin
        lonmax = fX_ext[1] if args.lonmax is None else args.lonmax
        if resample_x == 1 or resample_x == 2:
            fX_ext = ((lonmin / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0),
                      (lonmax / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0))
        else:
            fX_ext = (lonmin, lonmax)
        sX = fX_ext[1] - fX_ext[0]
    if args.latmin is not None and args.latmax is not None:
        clamp = True
        latmin = fY_ext[0] if args.latmin is None else args.latmin
        latmax = fY_ext[1] if args.latmax is None else args.latmax
        if resample_y == 1 or resample_y == 2:
            fY_ext = ((latmin / 90.0) * ((np.pi * polar_b_radius) / 2.0),
                      (latmax / 90.0) * ((np.pi * polar_b_radius) / 2.0))
        else:
            fY_ext = (latmin, latmax)
        sY = fY_ext[1] - fY_ext[0]
    if args.depthmax is not None and is3D:
        clamp = True
        fZ_ext = (fZ_ext[0], args.depthmax)
        sZ = fZ_ext[1] - fZ_ext[0]
    if args.fixZ and is3D:
        fZ_ext = (fZ_ext[1] * -1.0, fZ_ext[0] * -1.0)
        sZ = fZ_ext[1] - fZ_ext[0]
    print("clamp: {}".format(clamp))
    if clamp:
        print("sX (clamp) - {}".format(sX))
        print("sY (clamp) - {}".format(sY))
        print("sZ (clamp) - {}".format(sZ))
        print("fX ext. (clamp) - {}".format(fX_ext))
        print("fY ext. (clamp) - {}".format(fY_ext))
        print("fZ ext. (clamp) - {}".format(fZ_ext))
        print("fT ext. (clamp) - {}".format(fT_ext))
    if interpolate:
        xsteps = int(math.floor(sX / lateral_gres))
        ysteps = int(math.floor(sY / lateral_gres))
        zsteps = 1
        if is3D:
            zsteps = int(math.floor(sZ / vertical_gres))
        uvel_fpath = None
        vvel_fpath = None
        wvel_fpath = None
        if "nc" in fileformat:
            uvel_fpath = uvel_fpath_nc
            vvel_fpath = vvel_fpath_nc
            wvel_fpath = wvel_fpath_nc
            # ======== LOAD PARTICLE INFORMATION ======== #
            print("Load particle information grid")
            pt_file = xr.open_dataset(particle_fpath, decode_cf=True, engine='netcdf4')
            pX = pt_file['lon']
            if resample_x == 1 or resample_x == 2:
                pX = (pX / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
            pY = pt_file['lat']
            if resample_y == 1 or resample_y == 2:
                pY = (pY / 90.0) * ((np.pi * polar_b_radius) / 2.0)
            pZ = None
            if 'depth' in pt_file.keys() and is3D:
                pZ = pt_file['depth']
            elif 'z' in pt_file.keys() and is3D:
                pZ = pt_file['z']
            if (pZ is not None and is3D) and args.fixZ:
                pZ = pZ * -1.0
            N = int(pt_file['lon'].shape[0])
            tN = int(pt_file['lon'].shape[1])
            indices = np.random.randint(0, N - 1, Pn, dtype=int)
            indices_set = True
            if DBG_MSG:
                print("N: {}, t_N: {}".format(N, tN))
            valid_array = None
            if 'valid' in pt_file.keys():
                valid_array = np.maximum(np.max(np.array(pt_file['valid'][:, 0:2]), axis=1), 0).astype(np.bool_)
                if DBG_MSG:
                    print("Valid array: any true ? {}; all true ? {}".format(np.any(valid_array), np.all(valid_array)))
            else:
                pXd = pX.data
                nonstationary = np.ones(N, dtype=np.bool_)
                for p_ti in range(1, tN):
                    px_f1_i = np.nonzero(np.isnan(pXd[:, p_ti]))[0]
                    px_f0_i = np.nonzero(np.isnan(pXd[:, p_ti-1]))[0]
                    pX_i = np.intersect1d(px_f0_i, px_f1_i)
                    px_f0 = pXd[pX_i, p_ti-1]
                    if DBG_MSG:
                        print("px_f0 - shape: {}".format(px_f0.shape))
                    px_f1 = pXd[pX_i, p_ti]
                    if DBG_MSG:
                        print("px_f1 - shape: {}".format(px_f1.shape))
                    if pX_i.shape[0] > 0:
                        nonstationary[pX_i] = np.logical_and(nonstationary[pX_i], ~np.isclose(px_f0, px_f1).squeeze())
                valid_array = nonstationary
                if DBG_MSG:
                    print("Valid array: any true ? {}; all true ? {}".format(np.any(valid_array), np.all(valid_array)))
                del pXd
            pX_min, pX_max, _, _ = get_data_of_ndarray_nc(pX)
            print("pX - shape: {}; min: {}; max {}".format(pX.shape, pX_min, pX_max))
            pY_min, pY_max, _, _ = get_data_of_ndarray_nc(pY)
            print("pY - shape: {}; min: {}; max {}".format(pY.shape, pY_min, pY_max))
            if is3D and pZ is not None:
                pZ_min, pZ_max, _, _ = get_data_of_ndarray_nc(pZ)
                print("pZ - shape: {}; min: {}; max {}".format(pZ.shape, pZ_min, pZ_max))
            pT_init = pt_file['time'].data
            pT_mins = np.nanmin(pT_init, axis=0)
            pT_maxs = np.nanmax(pT_init, axis=0)
            if DBG_MSG:
                print("Times:\n\tmin = {}\n\tmax = {}\n\tinit: {}".format(pT_mins, pT_maxs, pT_init))
            assert pT_init.shape[1] == pT_mins.shape[0]
            mask_array = valid_array
            for ti in range(pT_init.shape[1]):
                replace_indices = np.isnan(pT_init[:, ti])
                # pT_init[replace_indices, ti] = pT_maxs[ti]  # this ONLY works if there is no delayed start
                pT_init[replace_indices, ti] = pT_mins[ti]  # this ONLY works if there is no delayed start
            if DBG_MSG:
                print("time info from file before baselining: shape = {} type = {} range = ({}, {})".format(pT_init.shape, type(pT_init[0, 0]), np.min(pT_init[0]), np.max(pT_init[0])))
            pT_0 = pT_maxs[0]
            pT = pT_init - pT_0
            pT_dt = convert_timevalue(pT_maxs[1] - pT_maxs[0], pT_0, ns_per_sec, False)
            pT_ft = convert_timearray(pT, pT_dt/60.0,ns_per_sec, False, "pT_ft")  # pt_maxs - pT_0
            reverse_time = (np.all(pT_ft <= np.finfo(pT_ft.dtype).eps) or (np.max(pT_init[0]) - np.min(pT_init[0])) < 0) and (pT_dt < 0)
            if DBG_MSG:
                print("reverse time: {}".format(reverse_time))
            # pT_ft_sec = convert_timearray(pT_mins- pT_0, pT_dt/60.0,ns_per_sec, False, "pT_ft_sec")
            pT_ft_sec = np.arange(0, pT_init.shape[1]*pT_dt, pT_dt) if not reverse_time else np.arange((pT_init.shape[1]-1)*abs(pT_dt), -abs(pT_dt), -abs(pT_dt))
            if DBG_MSG:
                print("time info from file after baselining: \n\tshape = {}".format(pT.shape))
                print("\ttype = {}".format(type(pT[0, 0])))
                print("\trange = {}".format( (np.nanmin(pT), np.nanmax(pT)) ))
                print("\trange sec. = {}".format((np.min(pT_ft_sec), np.max(pT_ft_sec))))
            # ==== time-clippling ==== #
            ti_min = 0
            ti_max = pT_ft_sec.shape[0]-1
            time_clip = False
            # to max before min, otherwise the max-indices shift
            if args.timax is not None:
                time_clip = True
                ti_max = max(0, min(pT_ft_sec.shape[0]-1, args.timax))
            if args.timin is not None:
                time_clip = True
                ti_min = max(0, min(pT.shape[0]-1, args.timin))
            pT_max = max(pT_ft_sec[ti_max], pT_ft_sec[ti_min])
            pT_min = min(pT_ft_sec[ti_max], pT_ft_sec[ti_min])
            del pT_mins
            del pT_maxs
            print("Time after clipping: pT.shape: {}, pT_min: {}, pT_max: {}, ti_min: {}, ti_max: {}, pT_dt: {}, (global) pt_0: {}".format(pT.shape, pT_min, pT_max, ti_min, ti_max, pT_dt, pT_0))

            interpolate_particles = (idt >= 0.0)
            idt = pT_dt if idt < 0.0 else math.copysign(idt, pT_dt)
            iT = pT_ft_sec
            cap_min = pT_ft_sec[ti_min]
            cap_max = pT_ft_sec[ti_max]
            iT_max = np.min(pT_ft_sec)
            iT_min = np.max(pT_ft_sec)
            if interpolate_particles:
                tsteps = int(math.floor((pT_max-pT_min)/idt)) if not reverse_time else int(math.floor((pT_min-pT_max)/idt))
                tsteps = abs(tsteps)
                iT = np.linspace(pT_min, pT_max, tsteps, dtype=np.float64) if not reverse_time else np.linspace(pT_max, pT_min, tsteps, dtype=np.float64)
                ti_min = max(np.min(np.nonzero(iT >= cap_min)[0])-1, 0) if not reverse_time else max(np.min(np.nonzero(iT <= cap_min)[0])-1, 0)
                ti_max = min(np.max(np.nonzero(iT <= cap_max)[0])+1, iT.shape[0]-1) if not reverse_time else min(np.max(np.nonzero(iT >= cap_max)[0])+1, iT.shape[0]-1)
                cap_max = iT[ti_max]
                cap_min = iT[ti_min]
                iT_max = np.max(iT)
                iT_min = np.min(iT)
                print("New time field: t_min = {}, t_max = {}, dt = {}, |T| = {}, ti_min_new = {}, ti_max_new = {}".format(iT_min, iT_max, idt, iT.shape[0], ti_min, ti_max))

            # ==== Attribute fusion ==== #
            pAttr = []
            vAttr = {}
            for attribute in pt_file.keys():
                if attribute not in reserved_attribs:
                    if fnmatch.fnmatch(attribute, "*[_|-|.| ][x|y|z|u|v|w][_|-|.| ]*"):
                        lattr = len(attribute)
                        period_pos = attribute.find('.')
                        uscore_pos = attribute.find('_')
                        hyphen_pos = attribute.find('-')
                        space_pos = attribute.find(' ')
                        lsep_pos = min(period_pos if period_pos > 0 else lattr, uscore_pos if uscore_pos > 0 else lattr,
                                       hyphen_pos if hyphen_pos > 0 else lattr, space_pos if space_pos > 0 else lattr)
                        lsep_char = ''
                        lsep_char = '.' if period_pos == lsep_pos else lsep_char
                        lsep_char = '_' if uscore_pos == lsep_pos else lsep_char
                        lsep_char = '-' if hyphen_pos == lsep_pos else lsep_char
                        lsep_char = ' ' if space_pos == lsep_pos else lsep_char
                        base_attr = attribute[0:lsep_pos]
                        rest_string = attribute[lsep_pos+1, lattr]
                        rattr = len(rest_string)
                        period_pos = rest_string.find('.')
                        uscore_pos = rest_string.find('_')
                        hyphen_pos = rest_string.find('-')
                        space_pos = rest_string.find(' ')
                        rsep_pos = min(period_pos if period_pos > 0 else rattr, uscore_pos if uscore_pos > 0 else rattr,
                                       hyphen_pos if hyphen_pos > 0 else rattr, space_pos if space_pos > 0 else rattr)
                        rsep_char = ''
                        rsep_char = '.' if period_pos == rsep_pos else rsep_char
                        rsep_char = '_' if uscore_pos == rsep_pos else rsep_char
                        rsep_char = '-' if hyphen_pos == rsep_pos else rsep_char
                        rsep_char = ' ' if space_pos == rsep_pos else rsep_char
                        if base_attr not in vAttr.keys():
                            vAttr[base_attr] = {"lsep": lsep_char, "rsep": rsep_char, 'dtype': pt_file[attribute].dtype, "components": [], "attributes": []}
                        component_name = rest_string[0:rattr]
                        vAttr[base_attr]['components'].append(component_name)
                        vAttr[base_attr]['attributes'].append(attribute)
                    pAttr.append(attribute)

            # ==== write transform-clamped particles ==== #
            pX_ft, pY_ft, pZ_ft = None, None, None
            pX_ft_ti0, pY_ft_ti0, pZ_ft_ti0 = None, None, None
            pX_ft_ti1, pY_ft_ti1, pZ_ft_ti1 = None, None, None
            p_tt = 0.0
            p_ti0, p_ti1 = 0, 0
            if not fields_only:
                stored_pt = np.zeros(N, dtype=np.bool_)
                for p_ti in range(ti_min, ti_max+1):
                    pdata_vtk = vtkPolyData()
                    points_vtk = vtkPoints()
                    if interpolate_particles:
                        tx0 = iT_min + float(p_ti) * idt if not reverse_time else iT_max + float(p_ti) * idt
                        # tx0 = pT_min + float(p_ti) * pT_dt if not reverse_time else pT_max + float(p_ti) * pT_dt
                        tx1 = iT_min + float((p_ti+1) % iT.shape[0]) * idt if periodicFlag else iT_min + float(min(p_ti+1, iT.shape[0]-1)) * idt
                        # tx1 = pT_min + float((p_ti+1) % pT_ft_sec.shape[0]) * pT_dt if periodicFlag else pT_min + float(min(p_ti+1, pT_ft_sec.shape[0]-1)) * pT_dt
                        tx1 = (iT_max + float((p_ti + 1) % iT.shape[0]) * idt if periodicFlag else iT_max + float(min(p_ti + 1, iT.shape[0] - 1)) * idt) if reverse_time else tx1
                        # tx1 = (pT_max + float((p_ti + 1) % pT_ft_sec.shape[0]) * pT_dt if periodicFlag else pT_max + float(min(p_ti + 1, pT_ft_sec.shape[0] - 1)) * idt) if reverse_time else tx1
                        if DBG_MSG:
                            print("tx0: {}, tx1: {}".format(tx0, tx1))
                        p_ti0 = time_index_value(tx0, pT_ft_sec, periodicFlag, _ft_dt=pT_dt)
                        p_tt = time_partion_value(tx0, pT_ft_sec, periodicFlag, _ft_dt=pT_dt)
                        p_ti1 = time_index_value(tx1, pT_ft_sec, periodicFlag, _ft_dt=pT_dt)
                        if DBG_MSG:
                            print("p_ti0: {}, p_ti1: {}, p_tt: {}".format(p_ti0, p_ti1, p_tt))
                        pX_ft_ti0 = np.array(pX[:, p_ti0])
                        pX_ft_ti1 = np.array(pX[:, p_ti1])
                        pY_ft_ti0 = np.array(pY[:, p_ti0])
                        pY_ft_ti1 = np.array(pY[:, p_ti1])
                        pZ_ft_ti0 = np.array([0.0, ], dtype=pX.dtype)
                        pZ_ft_ti1 = np.array([0.0, ], dtype=pX.dtype)
                        if is3D and pZ is not None:
                            pZ_ft_ti0 = np.array(pZ[:, p_ti])
                            pZ_ft_ti1 = np.array(pZ[:, p_ti1])
                    else:
                        pX_ft = np.array(pX[:, p_ti])
                        pY_ft = np.array(pY[:, p_ti])
                        pZ_ft = np.array([0.0, ], dtype=pX.dtype)
                        if is3D and pZ is not None:
                            pZ_ft = np.array(pZ[:, p_ti])
                    # pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    pt = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    stored_pt[:] = 0
                    for pi in range(N):
                        if interpolate_particles:
                            pt[0] = (1.0-p_tt) * pX_ft_ti0[pi] + p_tt * pX_ft_ti1[pi]
                            pt[1] = (1.0-p_tt) * pY_ft_ti0[pi] + p_tt * pY_ft_ti1[pi]
                            pt[2] = pZ_ft_ti0[0] if not is3D else (1.0-p_tt) * pZ_ft_ti0[pi] + p_tt * pZ_ft_ti1[pi]
                        else:
                            pt[0] = pX_ft[pi]
                            pt[1] = pY_ft[pi]
                            pt[2] = pZ_ft[0] if not is3D else pZ_ft[pi]
                        if (pt[0] >= fX_ext[0]) and (pt[0] <= fX_ext[1]) and \
                            (pt[1] >= fY_ext[0]) and (pt[1] <= fY_ext[1]) and \
                                ((pt[2] <= fZ_ext[1] and not args.fixZ) or (pt[2] >= fZ_ext[0] and args.fixZ) or not is3D):
                            points_vtk.InsertNextPoint(pt)
                            stored_pt[pi] = 1
                    pdata_vtk.SetPoints(points_vtk)

                    for attribute in pAttr:
                        arrclass = vtkFloatArray if pt_file[attribute].dtype == np.float32 else vtkDoubleArray
                        attr_array_vtk = arrclass()
                        attr_array_vtk.SetNumberOfComponents(1)
                        # attr_array_vtk.SetNumberOfTuples(N)
                        attr_array_vtk.SetName(attribute)
                        pattr_array = None
                        pattr_ti0_array, pattr_ti1_array = None, None
                        if interpolate_particles:
                            pattr_ti0_array = np.array(pt_file[attribute][:, p_ti0])
                            pattr_ti1_array = np.array(pt_file[attribute][:, p_ti1])
                        else:
                            pattr_array = np.array(pt_file[attribute][:, p_ti])
                        for pi in range(N):
                            if stored_pt[pi]:
                                val = 0
                                if interpolate_particles:
                                    val = (1.0-p_tt) * pattr_ti0_array[pi] + p_tt * pattr_ti1_array[pi]
                                else:
                                    val = pattr_array[pi]
                                # attr_array_vtk.SetValue(pi, pattr_array[pi])
                                attr_array_vtk.InsertNextValue(val)
                        pdata_vtk.GetPointData().AddArray(attr_array_vtk)
                        del pattr_array

                    for vattribute in vAttr.keys():
                        arrclass = vtkFloatArray if vAttr[vattribute]['dtype'] == np.float32 else vtkDoubleArray
                        vaDim = len(vAttr[vattribute]['components'])
                        attr_array_vtk = arrclass()
                        attr_array_vtk.SetNumberOfComponents(vaDim)
                        # attr_array_vtk.SetNumberOfTuples(N)
                        attr_array_vtk.SetName(vattribute)
                        pattr_array = None
                        pattr_ti0_array, pattr_ti1_array = None, None
                        darrays = []
                        for va_i in range(vaDim):
                            attribute = vAttr[vattribute]['attributes'][va_i]
                            if interpolate_particles:
                                pattr_ti0_array = np.array(pt_file[attribute][:, p_ti0])
                                pattr_ti1_array = np.array(pt_file[attribute][:, p_ti1])
                                pattr_array = (1.0-p_tt) * pattr_ti0_array + p_tt * pattr_ti1_array
                                darrays.append(pattr_array)
                            else:
                                pattr_array = np.array(pt_file[attribute][:, p_ti])
                                darrays.append(pattr_array)
                        val = np.array([0.0, ] * vaDim, dtype=vAttr[vattribute]['dtype'])
                        for pi in range(N):
                            if stored_pt[pi]:
                                for va_i in range(vaDim):
                                    val[va_i] = darrays[va_i][pi]
                                # attr_array_vtk.SetValue(pi, pattr_array[pi])
                                attr_array_vtk.InsertNextTypedTuple(val)
                        pdata_vtk.GetPointData().AddArray(attr_array_vtk)
                        del pattr_array


                    pdata_file = os.path.join(outdir, "pt_%d.vtp" % (p_ti))
                    pdata_writer = vtkXMLPolyDataWriter()
                    pdata_writer.SetFileName(pdata_file)
                    pdata_writer.SetInputData(pdata_vtk)
                    pdata_writer.Write()

            # ======== INTERPOLATE FIELD ON PARTICLE TIMES ======== #
            xval = np.linspace(start=fX_ext[0], stop=fX_ext[1], num=xsteps, dtype=np.float32)
            print("xval shape: {}".format(xval.shape[0]))
            yval = np.linspace(start=fY_ext[0], stop=fY_ext[1], num=ysteps, dtype=np.float32)
            print("yval shape: {}".format(yval.shape[0]))
            zval = None
            if is3D:
                zval = np.linspace(start=fZ_ext[0], stop=fZ_ext[1], num=zsteps, dtype=np.float32)
                print("zval shape: {}".format(zval.shape[0]))
            else:
                zval = np.array([0.0, ], dtype=np.float32)
            # centers_x = xval[0:-1] + step/2.0
            centers_x = xval
            print("centers_x - shape: {}, min: {}, max: {}".format(centers_x.shape[0], np.min(centers_x), np.max(centers_x)))
            # centers_y = yval[0:-1] + step/2.0
            centers_y = yval
            print("centers_y - shape: {}, min: {}, max: {}".format(centers_y.shape[0], np.min(centers_y), np.max(centers_y)))
            # centers_z = zval[0:-1] + step/2.0
            centers_z = zval
            print("centers_z - shape: {}, min: {}, max: {}".format(centers_z.shape[0], np.min(centers_z), np.max(centers_z)))

            if not particles_only:
                us = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                vs = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                ws = None
                if hasW:
                    ws = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                # us = np.zeros((centers_x.shape[0], centers_y.shape[0], centers_z.shape[0]))
                # vs = np.zeros((centers_x.shape[0], centers_y.shape[0], centers_z.shape[0]))
                # ws = None
                # if hasW:
                #     ws = np.zeros((centers_x.shape[0], centers_y.shape[0], centers_z.shape[0]))

                # us_minmax = [0., 0.]
                # us_statistics = [0., 0.]
                # vs_minmax = [0., 0.]
                # vs_statistics = [0., 0.]
                # if hasW:
                #     ws_minmax = [0., 0.]
                #     ws_statistics = [0., 0.]

                print("Interpolating UVW on a regular-square grid ...")
                total_items = (ti_max+1)-ti_min
                current_item = 0
                for ti in range(ti_min, ti_max+1):
                    tx0 = iT_min + float(ti) * idt if not reverse_time else iT_max + float(ti) * idt
                    if DBG_MSG:
                        print("tx: {}".format(tx0))
                    tx0 = math.fmod(tx0, fT_ext[1]+fT_dt)
                    if DBG_MSG:
                        print("tx fmod fT_max = {}: {}".format(fT_ext[1], tx0))
                    f_ti0 = time_index_value(tx0, fT, periodicFlag, _ft_dt=idt)
                    f_tt = time_partion_value(tx0, fT, periodicFlag, _ft_dt=idt)
                    tx1 = iT_min + float((ti+1) % iT.shape[0]) * idt if periodicFlag else iT_min + float(min(ti+1, iT.shape[0]-1)) * idt
                    tx1 = (iT_max + float((ti + 1) % iT.shape[0]) * idt if periodicFlag else iT_max + float(min(ti + 1, iT.shape[0] - 1)) * idt) if reverse_time else tx1
                    f_ti1 = time_index_value(tx1, fT, periodicFlag, _ft_dt=idt)
                    uvw_ti0 = f_ti0
                    uvw_ti1 = f_ti1
                    if periodicFlag:
                        uvw_ti0 = uvw_ti0 % fT.shape[0]
                        uvw_ti1 = uvw_ti1 % fT.shape[0]
                    else:
                        uvw_ti0 = min(f_ti0, fT.shape[0] - 1)
                        uvw_ti1 = min(f_ti1, fT.shape[0] - 1)
                    if DBG_MSG:
                        print("f_ti0: {}".format(f_ti0))
                        print("f_ti1: {}".format(f_ti1))
                        print("f_tt: {}".format(f_tt))
                        print("uvw_ti0: {}".format(uvw_ti0))
                        print("uvw_ti1: {}".format(uvw_ti1))

                    # ======== OPEN FIELD ======== #
                    fpath_idx_ti0 = fT_fpath_mapping[uvw_ti0][0]
                    local_ti0 = fT_fpath_mapping[uvw_ti0][2]
                    fpath_idx_ti1 = fT_fpath_mapping[uvw_ti1][0]
                    local_ti1 = fT_fpath_mapping[uvw_ti1][2]
                    if DBG_MSG:
                        print("path ti0: {} (local index: {})".format(fpath_idx_ti0, local_ti0))
                        print("path ti1: {} (local index: {})".format(fpath_idx_ti1, local_ti1))
                    uvel_fpath_ti0 = None
                    vvel_fpath_ti0 = None
                    wvel_fpath_ti0 = None
                    uvel_fpath_ti1 = None
                    vvel_fpath_ti1 = None
                    wvel_fpath_ti1 = None
                    if multifile:
                        uvel_fpath_ti0 = uvel_fpath_nc[fpath_idx_ti0]
                        vvel_fpath_ti0 = vvel_fpath_nc[fpath_idx_ti0]
                        if hasW:
                            wvel_fpath_ti0 = wvel_fpath_nc[fpath_idx_ti0]
                        uvel_fpath_ti1 = uvel_fpath_nc[fpath_idx_ti1]
                        vvel_fpath_ti1 = vvel_fpath_nc[fpath_idx_ti1]
                        if hasW:
                            wvel_fpath_ti1 = wvel_fpath_nc[fpath_idx_ti1]
                    else:
                        uvel_fpath_ti0 = uvel_fpath_nc
                        vvel_fpath_ti0 = vvel_fpath_nc
                        if hasW:
                            wvel_fpath_ti0 = wvel_fpath_nc
                        uvel_fpath_ti1 = uvel_fpath_nc
                        vvel_fpath_ti1 = vvel_fpath_nc
                        if hasW:
                            wvel_fpath_ti1 = wvel_fpath_nc
                    if DBG_MSG:
                        print("ti0 - file index: {}, filepath: {}, local ti-index: {}".format(fpath_idx_ti0, uvel_fpath_ti0, local_ti0))
                        print("ti1 - file index: {}, filepath: {}, local ti-index: {}".format(fpath_idx_ti1, uvel_fpath_ti1, local_ti1))
                    # ---- load ti0 ---- #
                    f_u_0 = xr.open_dataset(uvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                    fX0 = f_u_0.variables[xuvar]
                    if resample_x == 1:
                        fX0 = (fX0 / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
                    elif resample_x == 2:
                        fX0 = ((fX0-180.0) / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
                    fY0 = f_u_0.variables[yuvar]
                    if resample_y == 1:
                        fY0 = (fY0 / 90.0) * ((np.pi * polar_b_radius) / 2.0)
                    elif resample_y == 2:
                        fY0 = ((fY0-90.0) / 90.0) * ((np.pi * polar_b_radius) / 2.0)
                    fZ0 = None
                    if is3D:
                        fZ0 = f_u_0.variables[zuvar]
                        if args.fixZ:
                            fZ0 = np.flip(fZ0 * -1.0)
                        else:
                            fZ0 = np.array([0.0, ], dtype=np.float32)
                    if DBG_MSG:
                        print("Loaded XYZ data.")
                    fU0 = f_u_0.variables[uvar]
                    f_v_0 = xr.open_dataset(vvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                    fV0 = f_v_0.variables[vvar]
                    fW0 = None
                    if hasW:
                        f_w_0 = xr.open_dataset(wvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                        fW0 = f_w_0.variables[wvar]
                    # ---- load ti0 ---- #
                    fW1 = None
                    if fpath_idx_ti0 == fpath_idx_ti1 or fpath_idx_ti1 is None:
                        fX1 = fX0
                        fY1 = fY0
                        fZ1 = fZ0
                        fU1 = f_u_0.variables[uvar]
                        fV1 = f_v_0.variables[vvar]
                        if hasW:
                            fW1 = f_w_0.variables[wvar]
                    else:
                        fX1 = None
                        fY1 = None
                        fZ1 = None
                        fU1 = None
                        fV1 = None
                        fW1 = None
                    fU0 = fU0[local_ti0]
                    fV0 = fV0[local_ti0]
                    if hasW:
                        fW0 = fW0[local_ti0]
                    if DBG_MSG:
                        print("fX0 - shape: {}".format(fX0.shape))
                        print("fY0 - shape: {}".format(fY0.shape))
                        print("fZ0 - shape: {}".format(fZ0.shape))
                    if clamp:
                        fX0d = fX0.data
                        fX0_mask_min = fX0d >= fX_ext[0]
                        fX0_mask_max = fX0d <= fX_ext[1]
                        fX0_mask_ids = np.intersect1d(np.nonzero(fX0_mask_min)[0], np.nonzero(fX0_mask_max)[0])
                        xi0_min = np.maximum(np.min(fX0_mask_ids)-2, 0)
                        xi0_max = np.minimum(np.max(fX0_mask_ids)+2, fX0d.shape[0]-1)
                        del fX0d
                        fY0d = fY0.data
                        fY0_mask_min = fY0d >= fY_ext[0]
                        fY0_mask_max = fY0d <= fY_ext[1]
                        fY0_mask_ids = np.intersect1d(np.nonzero(fY0_mask_min)[0], np.nonzero(fY0_mask_max)[0])
                        yi0_min = np.maximum(np.min(fY0_mask_ids)-2, 0)
                        yi0_max = np.minimum(np.max(fY0_mask_ids)+2, fY0d.shape[0]-1)
                        del fY0d
                        if is3D:
                            fZ0d = fZ0.data
                            fZ0_mask_min = fZ0d >= fZ_ext[0]
                            fZ0_mask_max = fZ0d <= fZ_ext[1]
                            fZ0_mask_ids = np.intersect1d(np.nonzero(fZ0_mask_min)[0], np.nonzero(fZ0_mask_max)[0])
                            zi0_min = np.maximum(np.min(fZ0_mask_ids)-2, 0)
                            zi0_max = np.minimum(np.max(fZ0_mask_ids)+2, fZ0d.shape[0]-1)
                            del fZ0d
                        else:
                            zi0_min = 0
                            zi0_max = 0
                        fX0 = fX0[xi0_min:xi0_max+1]
                        fY0 = fY0[yi0_min:yi0_max+1]
                        fZ0 = fZ0[zi0_min:zi0_max+1]
                        fU0 = fU0[zi0_min:zi0_max+1, yi0_min:yi0_max+1, xi0_min:xi0_max+1]
                        fV0 = fV0[zi0_min:zi0_max+1, yi0_min:yi0_max+1, xi0_min:xi0_max+1]
                        if hasW:
                            fW0 = fW0[zi0_min:zi0_max+1, yi0_min:yi0_max+1, xi0_min:xi0_max+1]
                        if DBG_MSG:
                            fX0d = np.array(fX0)
                            print("fX0 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fX0.shape, np.min(fX0d), np.max(fX0d), xi0_min, xi0_max))
                            del fX0d
                            fY0d = np.array(fY0)
                            print("fY0 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fY0.shape, np.min(fY0d), np.max(fY0d), yi0_min, yi0_max))
                            del fY0d
                            fZ0d = np.array(fZ0)
                            print("fZ0 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fZ0.shape, np.min(fZ0d), np.max(fZ0d), zi0_min, zi0_max))
                            del fZ0d
                            print("fU0 (post-clamp) - shape: {}".format(fU0.shape))
                            print("fV0 (post-clamp) - shape: {}".format(fV0.shape))
                            if hasW:
                                print("fW0 (post-clamp) - shape: {}".format(fW0.shape))
                    with np.errstate(invalid='ignore'):
                        fU0 = np.nan_to_num(fU0, nan=0.0)
                        fV0 = np.nan_to_num(fV0, nan=0.0)
                        if hasW:
                            fW0 = np.nan_to_num(fW0, nan=0.0)
                    if is3D and args.fixZ:
                        fU0 = np.flip(fU0, 0)
                        fV0 = np.flip(fV0, 0)
                        if hasW:
                            fW0 = np.flip(fW0, 0)
                    if fU1 is None or fV1 is None or fW1 is None:
                        f_u_1 = xr.open_dataset(uvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                        fU1 = f_u_1.variables[uvar]
                        fX1 = f_u_1.variables[xuvar]
                        if resample_x == 1:
                            fX1 = (fX1 / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
                        elif resample_x == 2:
                            fX1 = ((fX1-180.0) / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
                        fY1 = f_u_1.variables[yuvar]
                        if resample_y == 1:
                            fY1 = (fY1 / 90.0) * ((np.pi * polar_b_radius) / 2.0)
                        elif resample_y == 2:
                            fY1 = ((fY1-90.0) / 90.0) * ((np.pi * polar_b_radius) / 2.0)
                        fZ1 = None
                        if is3D:
                            fZ1 = f_u_1.variables[zuvar]
                            if args.fixZ:
                                fZ1 = np.flip(fZ1 * -1.0)
                            else:
                                fZ1 = np.array([0.0, ], dtype=np.float32)
                        f_v_1 = xr.open_dataset(vvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                        fV1 = f_v_1.variables[vvar]
                        if hasW:
                            f_w_1 = xr.open_dataset(wvel_fpath_ti0, decode_cf=True, engine='netcdf4')
                            fW1 = f_w_1.variables[wvar]
                    fU1 = fU1[local_ti1]
                    fV1 = fV1[local_ti1]
                    if hasW:
                        fW1 = fW1[local_ti1]
                    if DBG_MSG:
                        print("fU1 - shape: {}".format(fU1.shape))
                        print("fV1 - shape: {}".format(fV1.shape))
                        if hasW:
                            print("fW1 - shape: {}".format(fW1.shape))
                    if clamp:
                        fX1d = fX1.data
                        fX1_mask_min = fX1d >= fX_ext[0]
                        fX1_mask_max = fX1d <= fX_ext[1]
                        fX1_mask_ids = np.intersect1d(np.nonzero(fX1_mask_min)[0], np.nonzero(fX1_mask_max)[0])
                        xi1_min = np.maximum(np.min(fX1_mask_ids)-2, 0)
                        xi1_max = np.minimum(np.max(fX1_mask_ids)+2, fX1.shape[0]-1)
                        del fX1d
                        fY1d = fY1.data
                        fY1_mask_min = fY1d >= fY_ext[0]
                        fY1_mask_max = fY1d <= fY_ext[1]
                        fY1_mask_ids = np.intersect1d(np.nonzero(fY1_mask_min)[0], np.nonzero(fY1_mask_max)[0])
                        yi1_min = np.maximum(np.min(fY1_mask_ids)-2, 0)
                        yi1_max = np.minimum(np.max(fY1_mask_ids)+2, fY1.shape[0]-1)
                        del fY1d
                        if is3D:
                            fZ1d = fZ1.data
                            fZ1_mask_min = fZ1d >= fZ_ext[0]
                            fZ1_mask_max = fZ1d <= fZ_ext[1]
                            fZ1_mask_ids = np.intersect1d(np.nonzero(fZ1_mask_min)[0], np.nonzero(fZ1_mask_max)[0])
                            zi1_min = np.maximum(np.min(fZ1_mask_ids)-2, 0)
                            zi1_max = np.minimum(np.max(fZ1_mask_ids)+2, fZ1d.shape[0]-1)
                            del fZ1d
                        else:
                            zi1_min = 0
                            zi1_max = 0
                        fX1 = fX1[xi1_min:xi1_max+1]
                        fY1 = fY1[yi1_min:yi1_max+1]
                        fZ1 = fZ1[zi1_min:zi1_max+1]
                        fU1 = fU1[zi1_min:zi1_max+1, yi1_min:yi1_max+1, xi1_min:xi1_max+1]
                        fV1 = fV1[zi1_min:zi1_max+1, yi1_min:yi1_max+1, xi1_min:xi1_max+1]
                        if hasW:
                            fW1 = fW1[zi1_min:zi1_max+1, yi1_min:yi1_max+1, xi1_min:xi1_max+1]
                        if DBG_MSG:
                            fX1d = np.array(fX1)
                            print("fX1 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fX1.shape, np.min(fX1d), np.max(fX1d), xi1_min, xi1_max))
                            del fX1d
                            fY1d = np.array(fY1)
                            print("fY1 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fY1.shape, np.min(fY1d), np.max(fY1d), yi1_min, yi1_max))
                            del fY1d
                            fZ1d = np.array(fZ1)
                            print("fZ1 (post-clamp) - shape: {}, min: {}, max: {}, imin = {}, imax = {}".format(fZ1.shape, np.min(fZ1d), np.max(fZ1d), zi1_min, zi1_max))
                            del fZ1d
                            print("fU1 (post-clamp) - shape: {}".format(fU1.shape))
                            print("fV1 (post-clamp) - shape: {}".format(fV1.shape))
                            if hasW:
                                print("fW1 (post-clamp) - shape: {}".format(fW1.shape))
                    with np.errstate(invalid='ignore'):
                        fU1 = np.nan_to_num(fU1, nan=0.0)
                        fV1 = np.nan_to_num(fV1, nan=0.0)
                        if hasW:
                            fW1 = np.nan_to_num(fW1, nan=0.0)
                    if is3D and args.fixZ:
                        fU1 = np.flip(fU1, 0)
                        fV1 = np.flip(fV1, 0)
                        if hasW:
                            fW1 = np.flip(fW1, 0)

                    if store_sgrid and is3D:
                        centers_z = np.array(fZ0)
                        del us
                        del vs
                        us = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                        vs = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                        ws = None
                        if hasW:
                            del ws
                            ws = np.zeros((centers_z.shape[0], centers_y.shape[0], centers_x.shape[0]))
                    else:
                        us[:, :, :] = 0
                        vs[:, :, :] = 0
                        if hasW:
                            ws[:, :, :] = 0
                    p_center_z, p_center_y, p_center_x = np.meshgrid(centers_z, centers_y, centers_x, sparse=False, indexing='ij')
                    # p_center_x, p_center_y, p_center_z = np.meshgrid(centers_x, centers_y, centers_z, sparse=False, indexing='ij')
                    gcenters = (p_center_z.flatten(), p_center_y.flatten(), p_center_x.flatten())
                    # gcenters = (p_center_x.flatten(), p_center_y.flatten(), p_center_z.flatten())
                    if DBG_MSG:
                        print("gcenters dims = ({}, {}, {})".format(gcenters[0].shape, gcenters[1].shape, gcenters[2].shape))
                        print("u0 dims = ({}, {}, {})".format(fU0.shape[0], fU0.shape[1], fU0.shape[2]))
                        print("v0 dims = ({}, {}, {})".format(fV0.shape[0], fV0.shape[1], fV0.shape[2]))
                        if hasW:
                            print("w0 dims = ({}, {}, {})".format(fW0.shape[0], fW0.shape[1], fW0.shape[2]))
                    mgrid0 = (fZ0, fY0, fX0)
                    # mgrid0 = (fX0, fY0, fZ0)
                    if DBG_MSG:
                        print("mgrid0 dims = ({}, {}, {})".format(mgrid0[0].shape, mgrid0[1].shape, mgrid0[2].shape))
                    us_local_0 = interpn(mgrid0, fU0.squeeze(), gcenters, method='linear', fill_value=.0)
                    vs_local_0 = interpn(mgrid0, fV0.squeeze(), gcenters, method='linear', fill_value=.0)
                    if hasW:
                        ws_local_0 = interpn(mgrid0, fW0.squeeze(), gcenters, method='linear', fill_value=.0)
                    del mgrid0
                    # print("us_local_0 dims: {}".format(us_local_0.shape))
                    # print("vs_local_0 dims: {}".format(vs_local_0.shape))
                    # print("ws_local_0 dims: {}".format(ws_local_0.shape))
                    mgrid1 = (fZ1, fY1, fX1)
                    us_local_1 = interpn(mgrid1, fU1.squeeze(), gcenters, method='linear', fill_value=.0)
                    vs_local_1 = interpn(mgrid1, fV1.squeeze(), gcenters, method='linear', fill_value=.0)
                    if hasW:
                        ws_local_1 = interpn(mgrid1, fW1.squeeze(), gcenters, method='linear', fill_value=.0)
                    del fU0
                    del fU1
                    del fV0
                    del fV1
                    if hasW:
                        del fW0
                        del fW1
                    del fX0
                    del fX1
                    del fY0
                    del fY1
                    del fZ0
                    del fZ1

                    # us_local = np.reshape(us_local, p_center_y.shape)
                    # vs_local = np.reshape(vs_local, p_center_y.shape)
                    # print("us_local_0 dims: {}".format(us_local_0.shape))
                    # print("vs_local_0 dims: {}".format(vs_local_0.shape))
                    # if hasW:
                    #     print("ws_local_0 dims: {}".format(ws_local_0.shape))
                    # u0 = np.transpose(np.reshape(us_local_0, p_center_x.shape), [2, 1, 0])
                    # u1 = np.transpose(np.reshape(us_local_1, p_center_x.shape), [2, 1, 0])
                    u0 = np.reshape(us_local_0, p_center_x.shape)
                    u1 = np.reshape(us_local_1, p_center_x.shape)
                    # print("u0 dims: {}".format(u0.shape))
                    # print("u1 dims: {}".format(u1.shape))
                    us[:, :, :] = (1.0 - f_tt) * u0 + f_tt * u1
                    del u0
                    del u1
                    # v0 = np.transpose(np.reshape(vs_local_0, p_center_y.shape), (2, 1, 0))
                    # v1 = np.transpose(np.reshape(vs_local_1, p_center_y.shape), (2, 1, 0))
                    v0 = np.reshape(vs_local_0, p_center_y.shape)
                    v1 = np.reshape(vs_local_1, p_center_y.shape)
                    vs[:, :, :] = (1.0 - f_tt) * v0 + f_tt * v1
                    del v0
                    del v1
                    if hasW:
                        # w0 = np.transpose(np.reshape(ws_local_0, p_center_z.shape), (2, 1, 0))
                        # w1 = np.transpose(np.reshape(ws_local_1, p_center_z.shape), (2, 1, 0))
                        w0 = np.reshape(ws_local_0, p_center_z.shape)
                        w1 = np.reshape(ws_local_1, p_center_z.shape)
                        ws[:, :, :] = (1.0 - f_tt) * w0 + f_tt * w1
                        del w0
                        del w1
                    ss = us ** 2 + vs ** 2 + ws ** 2 if hasW else us ** 2 + vs ** 2
                    ss = np.where(ss > 0, np.sqrt(ss), 0)

                    # us_minmax = [min(us_minmax[0], us.min()), max(us_minmax[1], us.max())]
                    # us_statistics[0] += us.mean()
                    # us_statistics[1] += us.std()
                    # vs_minmax = [min(vs_minmax[0], vs.min()), max(vs_minmax[1], vs.max())]
                    # vs_statistics[0] += vs.mean()
                    # vs_statistics[1] += vs.std()
                    # if hasW:
                    #     ws_minmax = [min(ws_minmax[0], ws.min()), max(ws_minmax[1], ws.max())]
                    #     ws_statistics[0] += ws.mean()
                    #     ws_statistics[1] += ws.std()

                    del us_local_0
                    del us_local_1
                    del vs_local_0
                    del vs_local_1
                    if hasW:
                        del ws_local_0
                        del ws_local_1
                    del p_center_z
                    del p_center_y
                    del p_center_x


                    udata_vtk = np_vtk.numpy_to_vtk(num_array=us.ravel(), deep=True, array_type=(vtk.VTK_FLOAT if us.dtype==np.float32 else vtk.VTK_DOUBLE))
                    udata_vtk.SetName("U")
                    vdata_vtk = np_vtk.numpy_to_vtk(num_array=vs.ravel(), deep=True, array_type=(vtk.VTK_FLOAT if vs.dtype==np.float32 else vtk.VTK_DOUBLE))
                    vdata_vtk.SetName("V")
                    wdata_vtk= None
                    if hasW:
                        wdata_vtk = np_vtk.numpy_to_vtk(num_array=ws.ravel(), deep=True, array_type=(vtk.VTK_FLOAT if ws.dtype==np.float32 else vtk.VTK_DOUBLE))
                        wdata_vtk.SetName("W")
                    sdata_vtk = np_vtk.numpy_to_vtk(num_array=ss.ravel(), deep=True, array_type=(vtk.VTK_FLOAT if ss.dtype==np.float32 else vtk.VTK_DOUBLE))
                    sdata_vtk.SetName("S")
                    ss_invalid = np.isclose(ss, 0) & np.isclose(ss, -0)
                    ss_min = np.finfo(ss.dtype).eps if ss_invalid.all() else np.min(ss[~ss_invalid])
                    ss_max = np.finfo(ss.dtype).eps if ss_invalid.all() else np.max(ss[~ss_invalid])
                    # ssi = (np.ones(ss.shape, ss.dtype) * ss_max) - ss
                    # sidata_vtk = np_vtk.numpy_to_vtk(num_array=ssi.ravel(), deep=True, array_type=(vtk.VTK_FLOAT if ss.dtype == np.float32 else vtk.VTK_DOUBLE))
                    # sidata_vtk.SetName("S_inv")

                    s_img = None
                    ss_sgrid = None
                    if store_sgrid:
                        sgrid = vtkStructuredGrid()
                        sgrid.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                        points_vtk = vtkPoints()

                        ss_sgrid = vtkStructuredGrid()
                        ss_sgrid.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                        if DBG_MSG:
                            print("Assemble new grid interpolation ...")
                        assemble_grid_total_items = centers_x.shape[0] * centers_y.shape[0] * centers_z.shape[0]
                        assemble_grid_current_item = 0
                        offset = 0
                        pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        zDim = 1 if not is3D else centers_z.shape[0]
                        for di in range(zDim):
                            kOffset = di * centers_y.shape[0] * centers_x.shape[0]
                            depth = centers_z[di]
                            for lati in range(centers_y.shape[0]):
                                jOffset = lati * centers_x.shape[0]
                                lat = yval[lati]
                                for loni in range(centers_x.shape[0]):
                                    offset = loni + jOffset + kOffset
                                    lon = xval[loni]
                                    pt[0] = lon
                                    pt[1] = lat
                                    pt[2] = depth
                                    points_vtk.InsertNextPoint(pt)
                                    # points_vtk.SetPoint(offset, pt)
                                    assemble_grid_current_item += 1
                            assemble_grid_workdone = assemble_grid_current_item / assemble_grid_total_items
                            if DBG_MSG:
                                print("\rAssemble Grid - Progress: [{0:50s}] {1:.1f}% (offset: {2:d})".format('#' * int(assemble_grid_workdone * 50), assemble_grid_workdone * 100, offset), end="", flush=True)
                        if DBG_MSG:
                            print("")
                            print("New grid assembled.")
                        sgrid.SetPoints(points_vtk)
                        ss_sgrid.SetPoints(points_vtk)

                        sgrid.GetPointData().SetScalars(udata_vtk)
                        sgrid.GetPointData().SetActiveScalars("U")
                        sgrid_file = os.path.join(outdir, "U_%d.vts" % (ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()

                        sgrid.GetPointData().SetScalars(vdata_vtk)
                        sgrid.GetPointData().SetActiveScalars("V")
                        sgrid_file = os.path.join(outdir, "V_%d.vts" % (ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(sgrid_file)
                        writer.SetInputData(sgrid)
                        writer.Write()

                        if hasW:
                            sgrid.GetPointData().SetScalars(wdata_vtk)
                            sgrid.GetPointData().SetActiveScalars("W")
                            sgrid_file = os.path.join(outdir, "W_%d.vts" % (ti))
                            writer = vtkXMLStructuredGridWriter()
                            writer.SetFileName(sgrid_file)
                            writer.SetInputData(sgrid)
                            writer.Write()

                        ss_sgrid.GetPointData().SetScalars(sdata_vtk)
                        ss_sgrid.GetPointData().SetActiveScalars("S")
                        ss_sgrid_file = os.path.join(outdir, "S_%d.vts" % (ti))
                        writer = vtkXMLStructuredGridWriter()
                        writer.SetFileName(ss_sgrid_file)
                        writer.SetInputData(ss_sgrid)
                        writer.Write()
                        # ss_sgrid.GetPointData().SetScalars(sidata_vtk)
                        # ss_sgrid.GetPointData().SetActiveScalars("S_inv")
                    else:
                        u_img = vtkImageData()
                        u_img.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                        u_img.SetOrigin(fX_ext[0], fY_ext[0], fZ_ext[0])
                        u_img.SetSpacing(lateral_gres, lateral_gres, vertical_gres)
                        u_img.GetPointData().SetScalars(udata_vtk)
                        u_img.GetPointData().SetActiveScalars("U")
                        u_vtk_file = os.path.join(outdir, "U_%d.vti" % (ti))
                        u_wrt = vtkXMLImageDataWriter()
                        u_wrt.SetFileName(u_vtk_file)
                        u_wrt.SetInputData(u_img)
                        u_wrt.Write()

                        v_img = vtkImageData()
                        v_img.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                        v_img.SetOrigin(fX_ext[0], fY_ext[0], fZ_ext[0])
                        v_img.SetSpacing(lateral_gres, lateral_gres, vertical_gres)
                        v_img.GetPointData().SetScalars(vdata_vtk)
                        v_vtk_file = os.path.join(outdir, "V_%d.vti" % (ti))
                        v_img.GetPointData().SetActiveScalars("V")
                        v_wrt = vtkXMLImageDataWriter()
                        v_wrt.SetFileName(v_vtk_file)
                        v_wrt.SetInputData(v_img)
                        v_wrt.Write()

                        if hasW:
                            w_img = vtkImageData()
                            w_img.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                            w_img.SetOrigin(fX_ext[0], fY_ext[0], fZ_ext[0])
                            w_img.SetSpacing(lateral_gres, lateral_gres, vertical_gres)
                            w_img.GetPointData().SetScalars(wdata_vtk)
                            w_vtk_file = os.path.join(outdir, "W_%d.vti" % (ti))
                            w_img.GetPointData().SetActiveScalars("V")
                            w_wrt = vtkXMLImageDataWriter()
                            w_wrt.SetFileName(w_vtk_file)
                            w_wrt.SetInputData(w_img)
                            w_wrt.Write()

                        s_img = vtkImageData()
                        s_img.SetDimensions(centers_x.shape[0], centers_y.shape[0], centers_z.shape[0])
                        s_img.SetOrigin(fX_ext[0], fY_ext[0], fZ_ext[0])
                        s_img.SetSpacing(lateral_gres, lateral_gres, vertical_gres)
                        s_img.GetPointData().SetScalars(sdata_vtk)
                        s_vtk_file = os.path.join(outdir, "S_%d.vti" % (ti))
                        s_img.GetPointData().SetActiveScalars("S")
                        s_wrt = vtkXMLImageDataWriter()
                        s_wrt.SetFileName(s_vtk_file)
                        s_wrt.SetInputData(s_img)
                        s_wrt.Write()
                        # s_img.GetPointData().SetScalars(sidata_vtk)
                        # s_img.GetPointData().SetActiveScalars("S_inv")

                    if args.bathymetry:
                        ss_data = ss_sgrid if store_sgrid else s_img
                        # thresh = vtkThresholdPoints()
                        # thresh = vtkImageThreshold()
                        thresh = vtkThreshold()
                        thresh.SetInputData(ss_data)
                        # thresh.ReplaceInOn()
                        # thresh.ReplaceOutOn()
                        # thresh.SetReplaceIn(1)
                        # thresh.SetReplaceOut(0)
                        # thresh.ThresholdByUpper(0.0)
                        # thresh.ThresholdBetween(0, ss_min)
                        thresh.ThresholdBetween(-np.finfo(ss.dtype).eps, np.finfo(ss.dtype).eps)
                        # thresh.AllScalarsOn()
                        # thresh.Update()

                        # contour_vtk = vtkDiscreteMarchingCubes()
                        # # contour_vtk.SetInputData(thresh.GetOutput())
                        # contour_vtk.SetInputConnection(thresh.GetOutputPort())
                        # contour_vtk.GenerateValues(1,0,1)
                        # contour_vtk.SetValue(0, 1)
                        # # contour_vtk.Update()

                        # contour_vtk = vtkContourFilter()
                        # # contour_vtk.SetInputData(ss_data)
                        # contour_vtk.SetInputConnection(thresh.GetOutputPort())
                        # contour_vtk.SetNumberOfContours(1)
                        # # contour_vtk.SetValue(0, ss_max)
                        # contour_vtk.SetValue(0, 0.5)
                        # contour_vtk.ComputeNormalsOn()
                        # contour_vtk.GenerateTrianglesOn()
                        # # contour_vtk.Update()

                        contour_vtk = vtkDataSetSurfaceFilter()
                        contour_vtk.SetPieceInvariant(1)
                        contour_vtk.SetInputConnection(thresh.GetOutputPort())

                        smoother = vtkWindowedSincPolyDataFilter()
                        smoother.SetInputConnection(contour_vtk.GetOutputPort())
                        smoother.SetNumberOfIterations(20)
                        smoother.BoundarySmoothingOff()
                        smoother.FeatureEdgeSmoothingOff()
                        smoother.NonManifoldSmoothingOn()
                        smoother.NormalizeCoordinatesOn()
                        smoother.SetFeatureAngle(120.0)
                        smoother.SetEdgeAngle(90.0)
                        smoother.SetPassBand(0.1)
                        smoother.Update()

                        sdata_file = os.path.join(outdir, "bathymetry_%d.vtp" % (ti, ))
                        sdata_writer = vtkXMLPolyDataWriter()
                        sdata_writer.SetFileName(sdata_file)
                        sdata_writer.SetInputData(smoother.GetOutput())
                        sdata_writer.Write()

                    current_item = ti+1
                    workdone = current_item / total_items
                    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
                print("")
                print("\nFinished UVW-interpolation.")
                del us
                del vs
                if hasW:
                    del ws

            del centers_x
            del centers_y
            del centers_z
            del xval
            del yval
            del zval
        elif "h5" in fileformat:
            pass

