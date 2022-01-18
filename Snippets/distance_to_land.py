"""
This is a function that can calculate and save an array that calculates the distance to the nearest model land cell for each ocean cell. For example,
if you are running a simulation with beaching that requires determining if a particle is within a certain distance to land, sampling a field holding
the output array from this function will allow you to do that.

This code is reasonably fast, but it was written under the assumption that it would run once (with the saved output being loaded afterwards) instead
of this field being calculated again and again during each simulation. As such, it has been optimized to the point that it was fast enough for me, but 
there are undoubtably faster ways of doing this. If you do further optimize this function, please update this code for future users! 
  - As a reference, running on the CMEMS Mediterranean data that is provided on 380x1016 grid, this code ran in a matter of minutes. For HYCOM global data on
    a 3251x4500 grid, this code took a few hours (but then again, that is why you would only run it once...).
    
Within the oceanparcels tutorials (specifically "Tutorial on implementing boundary conditions in an A grid") another approach is presented using numpy.roll
functions to determine the coastline edges. This is undoubtably faster than the approach below, but it does just provide the distance to the nearest model land
cell in the land-adjacent ocean cells. If you need to know the distance to shore for particles in the other ocean cells, the approach below would be more
suitable.
  - OceanParcels tutorial: https://nbviewer.org/github/OceanParcels/parcels/blob/master/parcels/examples/documentation_unstuck_Agrid.ipynb

In addition, this code was written assuming the circulation data is provided on an A grid (e.g. HYCOM, Globcurrent, CMEMS Mediterranean). A modified version
of this code probably works for a C grid as well, but this is left for later users to figure out :) If you do come up with a version of this code (or some other 
approach entirely), please feel free to add it in as well. 

Finally, in order to run this code you need to provide the following input variables:
  - output_name = This is the complete path where you wish for the output file to be saved.
  - grid = This is a masked numpy array from you circulation data. This is necessary to get the land mask, so please make sure the mask is still attached!
  - lon = This is a numpy array containing all the lon coordinates of the circulation data.
  - lat = This is a numpy array containing all the lat coordinates of the circulation data.

Contact person: Victor Onink
"""
import numpy as np
import xarray
import progressbar
import geopy.distance

def create_distance_to_shore(output_name: str, grid: np.array, lon: np.array, lat: np.array):
    # Getting the dimensions of the model grid
    land = grid.mask
    n_lon, n_lat = len(lon), len(lat)

    # Initializing the distance array
    distance = np.zeros(land.shape)

    # Creating a search_memory variable that keeps track of how many cells away land is. That way, when we move from one
    # cell to it's neighbor, then we don't need to check everything again...
    search_memory = 1

    # Looping through all the cells
    for lat_index in progressbar.progressbar(range(land.shape[0])):
        for lon_index in range(land.shape[1]):
            # If the land mask is false, find the nearest land cell
            if not land[lat_index, lon_index]:
                # Reduce the search_memory from the previous cell by two. This saves computational effort because if for
                # example you are in the middle of the Pacific and you know that for the neighboring cell the nearest
                # land cell was 500 cells away, you don't need to check all those cells again for the current cell since
                # you've only moved one cell.
                if search_memory > 2:
                    box_size = search_memory - 2
                else:
                    box_size = 1

                # These lists keep track of all land cells that are encountered in the search
                land_lon, land_lat, land_distance = [], [], []

                # Keep looking for land cells until you encounter at least one.
                while land_lon.__len__() == 0:
                    # First check the top and bottom rows of cells in the search box
                    for lat_step in [-box_size, box_size]:
                        for lon_step in range(-box_size, box_size + 1):
                            # Check if any boundary conditions are being broken
                            if boundary_conditions(n_lat, n_lon, lat_index, lat_step, lon_index, lon_step):
                                # If the cell being checked is land, save the lon and lat indices
                                if land[(lat_index + lat_step) % n_lat, (lon_index + lon_step) % n_lon]:
                                    land_lat.append((lat_index + lat_step) % n_lat)
                                    land_lon.append((lon_index + lon_step) % n_lon)
                    # Then check the left and right columns of cells in the search box.
                    for lat_step in range(-box_size, box_size + 1):
                        for lon_step in [-box_size, box_size]:
                            # Check if any boundary conditions are being broken
                            if boundary_conditions(n_lat, n_lon, lat_index, lat_step, lon_index, lon_step):
                                # If the cell being checked is land, save the lon and lat indices
                                if land[(lat_index + lat_step) % n_lat, (lon_index + lon_step) % n_lon]:
                                    land_lat.append((lat_index + lat_step) % n_lat)
                                    land_lon.append((lon_index + lon_step) % n_lon)
                    # If we don't encounter land cells, we increase the size of the search box by 1.
                    box_size += 1

                # Once we have found land, save the size of the search box in the memory
                search_memory = box_size

                # For all the encountered land cells, determine the distance to the ocean cell in kilometers
                for points in range(land_lon.__len__()):
                    land_distance.append(geopy.distance.distance((lat[lat_index], lon[lon_index]),
                                                       (lat[land_lat[points]], lon[land_lon[points]])).km)

                # Update the distance array with the shortest distance to land in the land_distance list
                distance[lat_index, lon_index] = np.min(land_distance)
            else:
                # If the cell we are looking at is land, set the search_memory to 1
                search_memory = 1

    # Saving the entire distance field
    dset = xarray.Dataset({'distance': xarray.DataArray(distance, coords=[('lat', lat), ('lon', lon)])},
                          coords={'lat': lat, 'lon': lon})
    dset.to_netcdf(output_name)
    
def boundary_conditions(n_lat: int, n_lon: int, lat_index: int, k: int, lon_index: int, m: int, circulation_data: str = 'GLOBAL'):
    if circulation_data == 'GLOBAL':
        # Check that you don't look above or below the North and South Pole. However, it is ok to loop around in the lon direction to account
        # for transitioning from -180 to 180 or from 0 to 360 degrees
        return (lat_index + k) < n_lat & (lat_index + k) > 0
    else:
        # If you are looking at more regional data (e.g. Mediterranean, Gulf of Mexico, Indian Ocean, etc.), then check you remain within the model
        # domain with no periodic boundary conditions
        return ((lat_index + k) < n_lat) & ((lat_index + k) >= 0) & ((lon_index + m) < n_lon) & ((lon_index + m) >= 0)
