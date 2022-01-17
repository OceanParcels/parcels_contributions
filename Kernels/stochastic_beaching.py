"""
This is a stochastic beaching and resuspension kernel based on Onink et al. (2021) https://doi.org/10.1088/1748-9326/abecbd. For full details please refer to this
paper, but a brief summary is provided below.

How it works: If the distance to the nearest model coastline of an adrift particle (particle.beach == 0) is less than some predefined boundary distance 
              (fieldset.Coastal_Boundary), then for a given integration timestep dt the particle has a certain probability of beaching, set by fieldset.p_beach.
              fieldset.p_beach is calculated as exp(- dt / lambda_B), where lambda_B is the beaching timescale in seconds. The implementation below assumes that the
              beaching timescale is a simulation-wide constant, and so to save computational time p_beach is added as a constant to the fieldset. However, if 
              the probability is particle dependent it can be calculated within the kernel. 
              If the random number drawn by ParcelsRandom.uniform(0, 1) is greater than fieldset.p_beach, the particle is considered to have beached and 
              particle.beach == 1.
              
              A beached particle can be resuspended again, with a resuspension probability (fieldset.p_resus) for a given dt. This resuspension probability is 
              defined as exp(- dt / lambda_R), where lambda_R is the resuspension timescale in seconds. If the random number is greater than fieldset.p_resus,
              then the particle is considered to have been resuspended and we set particle.beach == 0.

FieldSet requirements: The following fields/constants are required within your fieldset:
                       - a distance to shore field, which for each ocean cell indicates the distance to the nearest coastal cell
                       - a fieldset.Coastal_Boundary constant, which sets the maximum distance to shore within which beaching is possible.
                       - fieldset.p_beach and fieldset.p_resus, which in turn requires setting beaching and resuspension timescales.

Particle requirements: The particle class must contain a variable "beach", where beach == 0 indicates a particle that is afloat/adrift, and beach == 1 indicates
                       a particle that is beached. 
                       
Contact person: Victor Onink

"""
from parcels import ParcelsRandom

def beaching_kernel(particle, fieldset, time):
    # Beaching
    if particle.beach == 0:
        dist = fieldset.distance2shore[time, particle.depth, particle.lat, particle.lon]
        if dist < fieldset.Coastal_Boundary:
            if ParcelsRandom.uniform(0, 1) > fieldset.p_beach:
                particle.beach = 1
    # Resuspension
    elif particle.beach == 1:
        if ParcelsRandom.uniform(0, 1) > fieldset.p_resus:
            particle.beach = 0
