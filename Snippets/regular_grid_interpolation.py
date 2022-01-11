# This snippet interpolates a regular-gridded UV field on a (differently-shaped) grid.
# Input:
#   - U and V: a 3D-array with dimensionality/coordinates [time_index][lat_index][lon_index]
#   - fT: 1D array of float times; dimensionality [time_index]
#   - flats: 1D array of float latitudes; dimensionality [lat_index]
#   - flons: 1D array of float longitutes; dimensionality [lon_index]
#   - centers_x: x-direction cell centers of the target grid (or field); dimensionality [new_lon_index]
#   - centers_y: y-direction cell centers of the target grid (or field); dimensionality [new_lat_index]
#   - us and vs: output 3D array with dimensionality [time_index][new_lat_index][new_lon_index]

print("Interpolating UV on a regular-square grid ...")
total_items = fT.shape[1]
for ti in range(fT.shape[1]):
    uv_ti = ti
    if periodicFlag:
        uv_ti = ti % U.shape[0]
    else:
        uv_ti = min(ti, U.shape[0]-1)
    mgrid = (flats, flons)
    p_center_y, p_center_x = np.meshgrid(centers_y, centers_x, sparse=False, indexing='ij')
    gcenters = (p_center_y.flatten(), p_center_x.flatten())
    us_local = interpn(mgrid, U[uv_ti], gcenters, method='linear', fill_value=.0)
    vs_local = interpn(mgrid, V[uv_ti], gcenters, method='linear', fill_value=.0)
    us[:, :] = np.reshape(us_local, p_center_y.shape)
    vs[:, :] = np.reshape(vs_local, p_center_y.shape)

    del us_local
    del vs_local
    del p_center_y
    del p_center_x

    current_item = ti
    workdone = current_item / total_items
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
print("\nFinished UV-interpolation.")
