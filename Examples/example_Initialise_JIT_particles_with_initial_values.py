### This example shows how to create a particle set in JIT mode while still sampling initial conditions of a variable eg. Temperature
### This works even with mulitple releases during a simulation
### This is an example of a solution to the github issue here: https://github.com/OceanParcels/parcels/issues/861
### This code assumes you have already made the fieldset. (This is not a stand-only piece of code)

# First create a particle class which samples values, in this case it is sampling temperature.
# Within this class create the variable you wish to sample (temp) and initialise it with a value of 0.
# Also create a variable (sampled) to identify if it is the particles day of release.

class SampleParticle(JITParticle):         # Define a new particle class
    sampled = Variable('sampled', dtype = np.float32, initial = 0, to_write=False) # variable to identify if it is just released
    temp = Variable('temp', dtype=np.float32, initial=0)  # initialise temperature

# Create the kernel to sample temperature
def SampleTemp(particle, fieldset, time):
    particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]

# Create the kernel to get the initial temperature value for each particle.
def SampleInitial(particle, fieldset, time): 
    if particle.sampled == 0:
         particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]
         particle.sampled = 1
         
# Create your kernel list to run,
# Put the SampleInitial Kernel before the advection kernel to get the initial value of temp before the 1st advection step occurs.
kernels = SampleInitial + pset.Kernel(AdvectionRK4) + SampleTemp

# Excute the code
pset.execute(kernels, 
             dt=delta(minutes=30), 
             output_file=pfile, 
             verbose_progress=True,
             #moviedt=delta(hours=1),
             runtime = runtime,
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
pfile.close()