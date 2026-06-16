'''
NOTE: Currently the chroma packages are commented to allow me to test that the code works on my machine without a Nvidia GPU
Make sure you uncomment them before building the image or else the code will not work
'''

#from PocarChroma.run_manager import primary_generator
#from chroma.sim import Simulation
#from chroma.event import Photons

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#from chroma import gpu
#import pycuda.tools
#from .analysis_manager import analysis_manager


'''
The purpose of this file is to take the old run_manager.py and remove the class structure from it, so that it is just functions.

Goals: 
- Eliminate constants baked into functions
- 
'''

'''
===== Constants and Parameters =====
This section is for constants and parameters that may be selected as defaults
'''

# Default simulation parameters
default_params = {
    'seed': 5020,
    'num_particles': 10000000,  # Total number of photons to simulate, broken up into a certain number of batches
    'max_batch_size': 1000000,  # maximum number of photons to be processed by GPU at once. setting this to greater than two million can cause problems
    'track_return_ct': 500,     # number of photons to return for plotting. Due to reasons of implimentation, this must be smaller than max_batch_size. 
    # This doesn't really matter because the plotting code breaks down after a few thousand photons
    # wavelength of photons in nm

    'wavelength': 175.0,
    # Source parameters:
    'source_type': 'isotropic',     #("beam", 'cone')
    'source_shape': 'point',        #("disk")
    'source_r': 0,                  # The radius of the disk
    'source_axis': 'y',             # The axis along which beam and cone sources point, as well as the normal axis to the disk

    # For the following two parameters, please refer to en.wikipedia.org/wiki/Azimuth 
    # source_axis is considered the zenith
    'beam_azimuth': 0,             # rotation about source_axis for beam momenta axis in degrees
    'beam_elevation': 0,           # angle between beam axis and source axis 
    
    'cone_angle': 0,               # The half-angle of the cone in degrees


    # The following three parameters are related to GPU performance, specify your own with caution
    'n_threads': 64,
    'max_blocks': 1024,
    'propagation_seed': 20000000
}

        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(np.cos(angle / 2), 1, self.num_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = np.cos(phi) * sin_theta
        curr_py = cos_theta
        curr_pz = np.sin(phi) * sin_theta
        if not positive:
            curr_py *= -1
# These are the defualt types of surface interactions and their flags
default_interactions = {
    "RAYLEIGH_SCATTER": 4,
    "REFLECT_DIFFUSE": 5,
    "REFLECT_SPECULAR": 6,
    "SURFACE_REEMIT": 7,
    "SURFACE_TRANSMIT": 8,
    "BULK_REEMIT": 9,
    "CHERENKOV": 10,
    "SCINTILLATION": 11,
    }

# This is a flag that will cause verbose output when true
verbose = True

# vprint will only print if verbose is true
vprint = print if verbose else lambda *a, **k: None



'''
===== Functions =====
'''



'''
init() is a function for stuff that needs to be initialized before any simulations can be run that doesn't fit into other functions
'''
def init():

    return


'''
run_sim() is the function that actually runs a simulation 
'''

def run_sim(
    simulation_name:str,
    geometry,               # geometry object from geometry_manager 
    custom_params:dict,     # These will overwrite any default parameters

    # select the generator function for the photons.
    # if set to "None", it will use the default generator function specified below 
    # otherwise, it is expecting a function that returns a dictionary of numpy arrays with the following keys
    # 'positions'
    # 'directions'
    # 'wavelengths'
    # 'polarizations'
    generator_function = None,

    halt_on_warning = False # if a warning arises, wait for user permission to continue
    ):

    source_center = geometry.get_solid_center(name="source")

    

    return

'''
Generates a batch of photons, runs it through a given number of steps, and returns the results 
'''
def propagate_batch(
    n_photons:int,
    geometry,
    params:dict,
    generator_function,
    source_center
    ):

    # Start with generating photons
    # if generation function isn't specified, use the default
    if generator_function == None:
        vprint('Default photon generator selected')
        generator = photon_gen(
            params= params,
            source_location= source_center,
            n_batch_particles= n_photons
            )
        positions = generator['positions']
        directions = generator['directions']
        wavelengths = generator['wavelengths']
        polarizations = generator['polarizations']

        primary_photons = Photons(positions, directions, polarizations, wavelengths)
    else:
    # if a generator function is specified, use it. Specify your own generator functions at your own risk
    # NOTE: I decided to generate the actual Photons object here because it means that no external programs need to interface with Chroma
        generator = generator_function(n_photons, params)

        try:
            positions = generator['positions']
            directions = generator['directions']
            wavelengths = generator['wavelengths']
            polarizations = generator['polarizations']
        except:
            raise ValueError("Custom-defined generator functions must output a dictionary four of numpy arrays with the following keys: 'positions', 'directions', 'wavelengths', and 'polarizations'")
        primary_photons = Photons(positions, directions, polarizations, wavelengths)

    # initialize GPU photons and GPU Geometry

    gpu_photons = gpu.GPUPhotons(primary_photons)
    gpu_geometry = gpu.GPUGeometry(geometry)

    


    return

'''
append_output_file() appends the run information onto an output file 
'''
def append_output_file():

    return


'''
===== Photon Generation Functions =====
'''

'''
pg_init() contains code that needs to be run for every photon generator
'''
def photon_gen(
    params,
    source_location,
    n_batch_particles,
    ):

    # initialize a random number generator that will be used for all subsequent operations.
    # For a given source configuration, including seed, this will produce a deterministic outcome
    rng = np.random.default_rng(seed=params['seed'])

    # verify that source_axis is correctly formatted
    if not params['source_axis'] in {'x', 'y', 'z'}:
        raise ValueError("params['source_axis'] may only be 'x', 'y' or 'z' ")

    # positions 
    positions = pg_make_positions(n_batch_particles, params, source_location, rng)
    directions = pg_make_directions(n_batch_particles, params, rng)

    # make polarization and wavelength arrays
    polarizations = np.cross(directions, isotropic_sphere_dist(n_batch_particles, rng))
    wavelengths = np.ones(n_batch_particles) * params['wavelength']

    # actually make the photons

    primary_photons = {}
    primary_photons['positions'] = positions
    primary_photons['directions'] = directions
    primary_photons['polarizations'] = polarizations
    primary_photons['wavelengths'] = wavelengths

    return primary_photons

# This code will return an array that contains the starting positions of all of the photons
def pg_make_positions(
    n_batch_particles:int,
    params,
    source_location,
    rng
    ):

    # Creates a set of origins all at the same point 
    if params['source_shape'] == 'point':
        return np.tile(source_location, (n_batch_particles, 1))
    # Creates a set of origins of a given disk
    elif params['source_shape'] == 'disk':
        curr_sqrtr = np.sqrt(rng.uniform(0, params['source_r'], n_batch_particles))
        curr_theta = rng.uniform(0, 2.0 * np.pi, n_batch_particles)
        
        

        curr_x = np.ones(n_batch_particles) * source_location[0]
        curr_y = curr_sqrtr * np.sin(curr_theta) + source_location[1]
        curr_z = curr_sqrtr * np.cos(curr_theta) + source_location[2]

        # Apply the appropriate change of basis and return    
        return change_of_basis(np.vstack((curr_x, curr_y, curr_z)).T, params['source_axis'])
    
    # Raise an error if 'source_shape' is defined improperly
    else:
        raise ValueError("params['source_shape'] may only be 'point' or 'disk' ")


'''
Make the momenta (direction) of the photons 
'''
def pg_make_directions(
    n_batch_particles:int,
    params,
    rng
    ):
    # makes an isotropic source, with uniform distribution over the sphere
    if params['source_type'] == 'isotropic':
        return isotropic_sphere_dist(n_batch_particles, rng)
    elif params['source_type'] == 'beam':

        theta = np.deg2rad(params['beam_elevation'])
        phi = np.deg2rad(params['beam_azimuth'])

        px = np.cos(theta)
        py = np.sin(theta) * np.sin(phi)
        pz = np.sin(theta) * np.cos(phi)
        
        return change_of_basis(np.tile([px, py, pz], (n_batch_particles, 1)), params['source_axis'])
    
    elif params['source_type'] == 'cone':

        phi = rng.uniform(0, 2.0 * np.pi, n_batch_particles)
        cos_theta = rng.uniform(np.cos(np.deg2rad(params['cone_angle'])), 1, n_batch_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = cos_theta
        curr_py = np.cos(phi) * sin_theta
        curr_pz = np.sin(phi) * sin_theta

        return change_of_basis(np.vstack((curr_px, curr_py, curr_pz)).T, params['source_axis'])
    else:
        raise ValueError("Unknown source_type. Valid source types are 'isotropic', 'beam' and 'cone' ")
    

    return

'''
General function that takes data and aligns it's x-axis to an orthogonal axis (y, or z)
'''

def change_of_basis(
    input_data,
    new_axis:str):
    # because this redefines x to the specified axis, if the new_axis is x then it is an identity operation
    if new_axis == 'x':
        return input_data

    # Otherwise, we define an appropriate rotation matrix 
    elif new_axis == 'y':
        rot_mat = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]])
    elif new_axis == 'z':
        rot_mat = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]])
    else: 
        #Raise an error if axis is incorrectly specified
        raise ValueError("new_axis may only be 'x', 'y' or 'z' ")
    
    # and then apply the matrix multiplication
    return input_data @ rot_mat


'''
Returns an isotropic set of (x,y,z) points on the surface of a unit sphere. 
'''
def isotropic_sphere_dist(n_points, rng):
    phi = rng.uniform(0, 2.0 * np.pi, n_points)
    cos_theta = rng.uniform(-1.0, 1.0, n_points)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    curr_px = np.cos(phi) * sin_theta
    curr_py = np.sin(phi) * sin_theta
    curr_pz = cos_theta
    return np.vstack((curr_px, curr_py, curr_pz)).T





'''
WARNING: Below Here I am calling functions for testing purposes. Be sure to remove anything beyond this point before building the image
'''



