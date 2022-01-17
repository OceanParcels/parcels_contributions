"""
Based on Onink et al. (2022), this is kernel for the KPP turbulent wind mixing parametrization. For the full details of all the theory
incorporated within this parametrization and some model parameter sensitivies, please refer to the paper at https://doi.org/10.5194/gmd-2021-195
or refer to the full wind mixing code at https://github.com/VictorOnink/Wind-Mixing-Diffusion/. In addition, please note that the current kernel does
not apply any surface boundary conditions. Please refer to Onink et al. (2022) for a discussion of which boundary conditions could be appropriate.

The comments within the kernel describe the basic functioning of the kernel. In order for the kernel to work, the following data/constants must be defined
within the fieldset:
  - fieldset.MLD: data fields of the mixed layer depth.
  - fieldset.u10 and fieldset.v10: zonal and meridional wind speeds at the ocean surface in m/s.
  - fieldset.RHO_A = 1.22 kg/m3: a fieldset constant set at the density of air. 
  - fieldset.BETA = 35: the wave age for a fully developed wave state, following Kukulka et al. (2012) https://doi.org/10.1029/2012GL051116
  - fieldset.G = 9.81 m/s2: Acceleration of gravity
  - fieldset.SURF_Z: A fieldset constant set at the shallowest depth level within the 3D circulation data used (in case this is not 0 m).
  - fieldset.VK = 0.4: von Karman constant
  - fieldset.THETA: This is the Langmuir circulation amplification factor. If fieldset.THETA = 1.0, then this neglects any mixing due to Langmuir
                    turbulence. As shown in Onink et al. (2022), higher THETA values lead to deeper mixing, but dynamically calculating THETA values
                    remains an open challenge.
  - fieldset.PHI = 0.9: Stability function in Monin-Obukov boundary layer theory (Boufadel et al., 2019) https://doi.org/10.1029/2019JC015727
  
In the current form, it is assumed that there is a separate kernel which determines the water density at the ocean surface, which is saved within 
particle.surface_density. In addition, it is required to set the particle rise velocity, which is saved in particle.rise_velocity. This rise velocity
could either be calculated dynamically throughout the simulation through buoyancy calculations, or set as just a constant (in this case, it is computationally
cheaper to set it as a fieldset constant).

Contact person: Victor Onink
"""

def KPP_wind_mixing(particle, fieldset, time):
    # Loading the mixed layer depth from the fieldset
    mld = fieldset.MLD[time, fieldset.SURF_Z, particle.lat, particle.lon]

    # Below the MLD there is no wind-driven turbulent diffusion according to KPP theory, so we set both Kz and dKz to zero.
    if particle.depth > mld:
        Kz = 0
        dKz = 0
        
    # Within the MLD we compute the vertical diffusion according to Boufadel et al. (2020) https://doi.org/10.1029/2019JC015727
    else:
        # Calculate the wind speed at the ocean surface
        w_10 = math.sqrt(fieldset.u10[time, fieldset.SURF_Z, particle.lat, temp_lon] ** 2 + \
                         fieldset.v10[time, fieldset.SURF_Z, particle.lat, temp_lon] ** 2)
        
        # Drag coefficient according to Large & Pond (1981) https://doi.org/10.1175/1520-0485(1981)011%3C0324:OOMFMI%3E2.0.CO;2
        C_D = min(max(1.2E-3, 1.0E-3 * (0.49 + 0.065 * w_10)), 2.12E-3)
        
        # Calculate the surface wind stress based on the surface wind speed and the density of air
        tau = C_D * fieldset.RHO_A * w_10 ** 2
        
        # Calcuate the friction velocity of water at the ocean surface using the surface wind stress and the surface water density
        U_W = math.sqrt(tau / particle.surface_density)
        
        # Calcuate the surface roughness z0 following Zhao & Li (2019) https://doi.org/10.1007/s10872-018-0494-9
        z0 = 3.5153e-5 * fieldset.BETA ** (-0.42) * w_10 ** 2 / fieldset.G
        
        # The corrected particle depth, since the depth is not always zero for the surface circulation data
        z_correct = particle.depth - fieldset.SURF_Z
        
        # The diffusion gradient at particle.depth
        C1 = (fieldset.VK * U_W * fieldset.THETA) / (fieldset.PHI * mld ** 2)
        dKz = C1 * (mld - z_correct) * (mld - 3 * z_correct - 2 * z0)
        
        # The KPP profile vertical diffusion, at a depth corrected for the vertical gradient in Kz
        C2 = (fieldset.VK * U_W * fieldset.THETA) / fieldset.PHI
        Kz = C2 * (z_correct + z0) * math.pow(1 - z_correct / mld, 2)
        
    # The Markov-0 vertical transport from Grawe et al. (2012) http://dx.doi.org/10.1007/s10236-012-0523-y
    gradient = dKz * particle.dt
    R = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3) * math.sqrt(2 * Kz)
    rise = particle.rise_velocity * particle.dt
    
    # Update the particle depth
    particle.depth = particle.depth + gradient + R + rise
