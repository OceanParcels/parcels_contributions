"""
The kernel below is a modification of the standard RK4 advection kernel, but we include depth dependent horizontal Stokes drift. According to 
Stokes theory, the magnitude of Stokes drift is largest at the ocean surface and with an exponential reduction in magnitude with increasing depth.
However, reanalysis products generally only provide the surface Stokes drift. Using the depth dependence from equation 19 from Breivik et al. (2016),
we can then calculate the horizontal Stokes drift at z = particle.depth (assuming particle.depth > 0, otherwise take the absolute magnitude of the 
particle depth for calculating the st_z correction term).

In order for this code to work, the following fields need to be added to the fieldset (on top of the standard UV fields):
  - fieldset.WP: Data fields containing peak wave period data
  - fieldset.SURF_Z: a fieldset constant that has the depth of the surface data level. This is just in case the surface data level is not exactly at z = 0,
    such as with the CMEMS Mediterranean data. Without including this correction, the Stokes drift at the surface would be underestimated.

Breivik et al. (2016) https://doi.org/10.1016/j.ocemod.2016.01.005

Contact person: Victor Onink
"""
import math

def AdvectionRK4_STOKES(particle, fieldset, time):
  """
  Author: Victor Onink, 25/01/22 for the Stokes drift adaptation, the rest of the RK4 kernel is from the standard parcels package.
  """
  # Determine the peak wave period at the particle position, and then from that the peak wave number
  w_p = (2 * math.pi / fieldset.WP[time, particle.depth, particle.lat, particle.lon])
  k_p = w_p ** 2 / 9.81

  # Calculate the corrected particle depth (in case the ocean surface is not at exactly z = 0)
  z_c = max(particle.depth - fieldset.SURF_Z, 0)

  # Calculate the Stokes depth correction term
  st_z = min(max(math.exp(-2 * k_p * z_c) - math.sqrt(2 * math.pi * k_p * z_c) * math.erfc(2 * k_p * z_c), 0), 1)

  # The rest of the kernel is just standard RK4 advection, but now with Stokes drift added in as well.
  (u1, v1) = fieldset.UV[particle]
  (uS1, vS1) = st_z * fieldset.Ust[time, particle.depth, particle.lat, particle.lon], st_z  * fieldset.Vst[time, particle.depth, particle.lat, particle.lon]
  lon1, lat1 = (particle.lon + (u1 + uS1) * .5 * particle.dt, particle.lat + (v1 + vS1) * .5 * particle.dt)

  (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
  (uS2, vS2) = st_z * fieldset.Ust[time + .5 * particle.dt, particle.depth, lat1, lon1], st_z  * fieldset.Vst[time + .5 * particle.dt, particle.depth, lat1, lon1]
  lon2, lat2 = (particle.lon + (u2 + uS2) * .5 * particle.dt, particle.lat + (v2 + vS2) * .5 * particle.dt)

  (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
  (uS3, vS3) = st_z * fieldset.Ust[time + .5 * particle.dt, particle.depth, lat2, lon2], st_z  * fieldset.Vst[time + .5 * particle.dt, particle.depth, lat2, lon2]    
  lon3, lat3 = (particle.lon + (u3 + uS3) * particle.dt, particle.lat + (v3 + vS3) * particle.dt)

  (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
  (uS4, vS4) = st_z * fieldset.Ust[time + particle.dt, particle.depth, lat3, lon3], st_z  * fieldset.Vst[time + particle.dt, particle.depth, lat3, lon3]

  particle.lon += (u1 + 2 * u2 + 2 * u3 + u4 + uS1 + 2 * uS2 + 2 * uS3 + uS4) / 6. * particle.dt
  particle.lat += (v1 + 2 * v2 + 2 * v3 + v4 + vS1 + 2 * vS2 + 2 * vS3 + vS4) / 6. * particle.dt
