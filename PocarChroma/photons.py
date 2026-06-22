from PocarChroma.run_manager import primary_generator
from chroma.sim import Simulation
from chroma.event import Photons

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os

from chroma import gpu
import pycuda.tools
from .analysis_manager import analysis_manager

'''
================ START AUXILLARY FUNCTIONS ===============
'''
# Set this parameter to true to enable more information printed out
test_mode = False

# tprint will print only if test mode is set to True

def init(use_test_mode = False):

    global test_mode
    global tprint

    test_mode = use_test_mode

    tprint = print if test_mode else lambda *a, **k: None
    if test_mode:
        tprint('test_mode active')




'''
================= START PROPAGATION CODE =================
'''




'''
Propagate() only handles propagation of a single "batch" of photons, 
batching is either to be handled by an auxilliary function in this file or something in the analysis file
'''

def propagate(
    photons,    # This should be a chroma Photons object. NOT the photon generator
    geometry,   # This should be a chroma geometry object
    interactions,
    seed = 5555,
    track_return_ct = 0,
    num_steps = 15,

    # The following parameters are highly GPU dependant, change at your own risk
    n_threads = 64,
    max_blocks = 1024,
    
    ):

    # Get number of photons from Photons object
    n_photons = photons.pos.shape[0]
    
    # Print a warning if attempting to propagate a large number of photons
    if n_photons > 2000000:
        print('WARNING: Attempting to propagate more than 2 million photons. This may crash the GPU!')

    # raise an error if more photon tracks are requested than photons simulated
    if track_return_ct > n_photons:
        raise ValueError('More photon tracks requested than photons simulated!')


    
    # initialize some arrays for storing output information

    photon_tracks = np.zeros((num_steps + 1, track_return_ct, 3))
    
    #initialize the first row of photon tracks with initial photon positions
    photon_tracks[0, :, :] = photons.pos[:track_return_ct]



    particle_histories = {
            curr_int: np.zeros(n_photons, dtype=int)
            for curr_int in interactions.keys()
        }

    
    # start a simulation
    sim = Simulation(
            geometry.global_geometry, seed = seed, geant4_processes=0
        ) 
    
    # intialize GPU states
    gpu_photons = gpu.GPUPhotons(photons)
    gpu_geometry = gpu.GPUGeometry(geometry.global_geometry)

    rng_states = gpu.get_rng_states(n_threads * max_blocks, seed=seed)

    # ===== BEGIN RUN LOOP =====


    for i in range(num_steps):

        gpu_photons.propagate(
            gpu_geometry,
            rng_states,
            nthreads_per_block=n_threads,
            max_blocks=max_blocks,
            max_steps=1,
        )

        # Get a chroma Photons object
        photons = gpu_photons.get()

        # Add a new column to tracks
        photon_tracks[i + 1, :, :] = photons.pos[:track_return_ct]

        # This is the update_tallies() function from run_manager
        for key, value in interactions.items():
            curr_tally = (photons.flags & (0x1 << value)).astype(bool).astype(int)
            particle_histories[key] += curr_tally

        # This is reset non-terminal flags from run_manager
        new_flags = photons.flags & 2147479567
        gpu_photons.flags[: n_photons].set(new_flags.astype(np.uint32))
    
    #simulation done, clear GPU cache to save memory
    pycuda.tools.clear_context_caches()

    return photon_tracks, particle_histories

'''
================== END PROPAGATION CODE ==================


================= START FILE SAVING CODE =================
'''


def make_HDF5_file(
    file_path, 
    attributes:dict,


    tracks_shape,    # The shape of the tracks dataset, in standard numpy notation.

    tallies_rows:int,
    # and the tallies columns
    tallies_columns
    ):
    '''
    Makes an HDF5 file with the desired header information

    You can pass anything under 64kb into attributes and it will be saved along with the file. 
    This includes things such as a notes section
    To read attributes, use

    f.attrs.get[<attribute_name>].

    It also creates two datasets, one called tallies and one called tracks

    Note, specifying dataset shape will ensure that file creation goes smoothly.
    The tracks dataset has the shape (<number of steps> + 1, <number of tracks to be saved>, 3)
    You should also specify the total number of rows (photons) in the tallies arrays
    '''
    # I could in theory make it so the HDF5 file is configured to be dynamic, but that would take extra work that doesn't seem worth it right now

    save_dir = file_path.rsplit('/', 1)[0]
    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir, exist_ok=True)


    # Convert tallies_columns into list if supplied as dict
    if isinstance(tallies_columns, dict):
        tallies_columns = list(tallies_columns.keys())
    
    # make the tallies column names into a structured dtype with columns as bools

    tallies_dtype = np.dtype([(name, 'i') for name in tallies_columns])

    # actually make the file
    with h5py.File(file_path, 'w') as f:

        # make the two datasets
        tallies_ds = f.create_dataset(name='tallies',
            shape=(tallies_rows,), 
            dtype=tallies_dtype
            )
        
        # set an attribute that keeps track of next writable row 

        tallies_ds.attrs['next_writable'] = 0
        
        tracks_ds = f.create_dataset('tracks', tracks_shape, dtype='f')

        tracks_ds.attrs['next_writable'] = 0

        for key, value in attributes.items():
            f.attrs[key] = value
    
    print(f'Results file created at {file_path}')

    return

'''
I decided to impliment two writing functions because tallies is writing bools to columns but tracks is writing n by 3 arrays to rows
'''

def tallies_write(
    file_path:str,
    tallies_dict:dict,
):
    '''
    :param file_path: The path to the previously created hdf5 file
    :type file_path: str
    :param tallies_dict: A dictionary of numpy arrays
    :type tallies_dict: dict
    '''
    
    tprint('The tallies dictionary is:')
    tprint(tallies_dict)

    with h5py.File(file_path, 'r+') as f:
        ds = f['tallies']
        next_row = ds.attrs['next_writable']


        # Get the end row by getting the length of the first array in the dict
        end_row = next_row + len(next(iter(tallies_dict.values())))

        for key, value in tallies_dict.items():

            ds[next_row: end_row][key] = value
        
        ds.attrs['next_writable'] = end_row

    return

    

def tracks_write():
    pass

    



'''
================== END FILE SAVING CODE ==================



============== START PHOTON GENERATION CODE ==============
'''


'''
This class is only a wrapper to make working with photon generator functions easier. 
It handles initiating a photon generator, generating photon arrays, and converting those arrays into Chroma Photon objects 
'''
class pg_handler():

    def __init__(
        self,
        generator,
        **generator_args
        ) -> None:
        
        
        self.gen = generator(**generator_args)
        next(self.gen)

    def get_photons(self, 
        n_photons:int,
        return_dict = False):
        
        try:
            photons_dict = self.gen.send(n_photons)
        except:
            return None
        
        if return_dict:
            return photons_dict
        else: 
            pass
        
        try:
            positions = photons_dict['positions']
            directions = photons_dict['directions']
            wavelengths = photons_dict['wavelengths']
            polarizations = photons_dict['polarizations']
        except KeyError:
            raise ValueError('Photon generator not outputting properly formatted dictionaries')
        
        return Photons(positions, directions, polarizations, wavelengths)

'''
This is a function that takes the dict returned by a photon generator and converts it into a Chroma photons object.
'''

def to_photon(photons_dict):
        try:
            positions = photons_dict['positions']
            directions = photons_dict['directions']
            wavelengths = photons_dict['wavelengths']
            polarizations = photons_dict['polarizations']
        except KeyError:
            raise ValueError('photons_dict not properly formatted')
        
        return Photons(positions, directions, polarizations, wavelengths)



'''
Photon generator functions, when the .send method is used, return a dictionary of numpy arrays, with keys of 'positions', 'directions' 'wavelengths' and 'polarizations' 
This one is a reimplementation of the photon generation code from the origin PocarChroma v2 run_manager. 
'''
def photon_generator(
    seed = 5555,
    max_photons = 1000000,
    position_gen_type = 'point',
    direction_gen_type = 'isotropic',
    source_axis = 'y',
    source_location = [0,0,0],
    source_r = None,
    beam_azimuth = None,
    beam_declination = None,
    wavelength = None,
    cone_angle = None
    ):

    # ===== START INITIALIZATION BLOCK =====

    # start with initializing an RNG
    rng = np.random.default_rng(seed = seed)

    # init some dicts for arguments
    position_args = {}
    direction_args = {}

    # make sure that wavelength is a float
    try:
        wavelength = float(wavelength)
    except:
        raise ValueError('Wavelength must be defined as a float in nanometers')
    
    # Set up rotation matrix:
    if source_axis == 'x':
        rot_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
    elif source_axis == 'y':
        rot_mat = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]])
    elif source_axis == 'z':
        rot_mat = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]])
    else:
        raise ValueError('source_axis may be x, y, or z')


    # Set up position generator

    if position_gen_type == 'point':

        position_function = pg_point_source

    elif position_gen_type == 'disk':

        position_function = pg_disk_source

        position_args['rng'] = rng
        position_args['rot_matrix'] = rot_mat

        # Check to make sure that source_r has been defined appropriately
        if  isinstance(source_r, (int, float)):
            position_args['source_r'] = source_r
        else:
            raise ValueError('source_r must be specified as an int or float')
    
    else:
        raise ValueError('Unknown position_gen_type. Currently accepted position_generators are point and disk')


    # Set up direction generation function

    if direction_gen_type == 'isotropic':
        direction_function = pg_isotropic_source

        direction_args['rng'] = rng
    
    elif direction_gen_type == 'beam':

        direction_function = pg_beam_source
        
        direction_args['rot_matrix'] = rot_mat

        # Check to make sure that beam_azimuth has been defined appropriately
        if  isinstance(beam_azimuth, (int, float)):
            direction_args['beam_azimuth'] = beam_azimuth
        else:
            raise ValueError('beam_azimuth must be specified as an int or float')
        
        # Check to make sure that beam_declination has been defined appropriately
        if  isinstance(beam_declination, (int, float)):
            direction_args['beam_declination'] = beam_declination
        else:
            raise ValueError('beam_declination must be specified as an int or float')
    
    elif direction_gen_type == 'cone':

        direction_function = pg_cone_source

        direction_args['rot_matrix'] = rot_mat
        direction_args['rng'] = rng

        # Check to make sure that cone_angle has been defined appropriately
        if  isinstance(cone_angle, (int, float)):
            direction_args['cone_angle'] = cone_angle
        else:
            raise ValueError('cone_angle must be specified as an int or float')

    else:
        raise ValueError('Unknown direction_gen_type. Currently accepted direction generators are isotropic, beam, and cone')

    # ===== END INITIALIZATION BLOCK =====

    # ===== START GENERATOR BLOCK =====
    total_photons = 0
    n_photons = 0
    photon_arrs = None
    break_flag = False
    while True:

        total_photons += n_photons

        # at this point total_photons and n_photons are zero
        n_photons = yield photon_arrs    # after initialization, the generator stops here, waiting for the .send method to provide n_photons

        # we check if this next batch of photons will exceed the total number of photons requested
        if total_photons + n_photons >= max_photons:
            # if that is the case, we restrict the total number of photons requested so as not to exceed max_photons
            n_photons = max_photons - total_photons
            # and then we set break_flag, which will modify behavior later on
            break_flag = True
        
        photon_arrs = {}
        # Then we call functions

        positions = position_function(
            n_photons= n_photons,
            source_location= source_location,
            **position_args)
        
        directions = direction_function(
            n_photons=n_photons,
            **direction_args)
        
        polarizations  = np.cross(directions, pg_isotropic_source(n_photons=n_photons, rng=rng))
        wavelengths = np.ones(n_photons) * wavelength

        photon_arrs['positions'] = positions
        photon_arrs['directions'] = directions
        photon_arrs['polarizations'] = polarizations
        photon_arrs['wavelengths'] = wavelengths

        if break_flag:
            yield photon_arrs
            break
    # ===== END GENERATOR BLOCK =====








'''
============= PHOTON GENERATION SUB-FUNCTIONS ============
Here are a family of sub-functions that are used by the photon generator object
Source functions should take number of particles, a source location, and a kwargs statement
Direction functions should take a number of particles and a kwargs statment
Source kwargs:
    - source_r: Radius of the source 
    - rng: numpy random number generator object
    - rot_matrix: a 3x3 np array to rotate the resulting vectors into the desired orientation using matrix multiplication

Direction kwargs:
    - beam_declination: The angle between source_axis and the beam (in radians)
    - beam_azimuth: The angle of the beam about source_axis (in radians)
    - cone_angle: the angle of the cone (in radians)
    - rng
    - rot_matrix

These functions aren't neccesarily intended to be called by external programs, but they can if you want them to
'''


def pg_point_source(
    n_photons,
    source_location,
    **kwargs
    ):
    return np.tile(source_location, (n_photons, 1))

'''
Creates starting positions for a disk source of a given radius

It does this by creating a disk distribution normal to the x-axis and then rotating to be normal to the appropriate axis
'''
def pg_disk_source(
    n_photons,
    source_location,
    **kwargs
    ):

    # Pull disk_r and rng from kwargs
    disk_r = kwargs['source_r']
    rng = kwargs['rng']
    rot_mat = kwargs['rot_matrix']

    # create 
    curr_sqrtr = np.sqrt(rng.uniform(0, disk_r, n_photons))
    curr_theta = rng.uniform(0, 2.0 * np.pi, n_photons)
    
    curr_x = np.ones(n_photons) * source_location[0]
    curr_y = curr_sqrtr * np.sin(curr_theta) + source_location[1]
    curr_z = curr_sqrtr * np.cos(curr_theta) + source_location[2]

    # make an array of the positions and then rotate it to make row-vectors
    positions = np.vstack((curr_x, curr_y, curr_z)).T
    # rotate the positions into the appropriate reference frame
    return positions @ rot_mat

'''
Make spherically isotropic directions for the photons
'''

def pg_isotropic_source(
    n_photons,
    **kwargs
    ):

    rng = kwargs['rng']
    
    phi = rng.uniform(0, 2.0 * np.pi, n_photons)
    cos_theta = rng.uniform(-1.0, 1.0, n_photons)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    curr_px = np.cos(phi) * sin_theta
    curr_py = np.sin(phi) * sin_theta
    curr_pz = cos_theta
    return np.vstack((curr_px, curr_py, curr_pz)).T

'''
Make a beam of particles in a given direction, as specified by declination and azimuth
'''

def pg_beam_source(
    n_photons,
    **kwargs
    ):
    print(kwargs)
    theta = kwargs['beam_declination']
    phi = kwargs['beam_azimuth']
    rot_mat = kwargs['rot_matrix']

    px = np.cos(theta)
    py = np.sin(theta) * np.sin(phi)
    pz = np.sin(theta) * np.cos(phi)
    directions = np.tile([px, py, pz], (n_photons, 1))
    print(directions)
    return  directions @ rot_mat

'''
Make a cone source, centered about the appropriate axis, with a given angle.
'''

def pg_cone_source(
    n_photons,
    **kwargs
    ):

    rng = kwargs['rng']
    angle = kwargs['cone_angle']
    rot_mat = kwargs['rot_matrix']

    phi = rng.uniform(0, 2.0 * np.pi, n_photons)
    cos_theta = rng.uniform(np.cos(angle), 1, n_photons)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    curr_px = cos_theta
    curr_py = np.cos(phi) * sin_theta
    curr_pz = np.sin(phi) * sin_theta

    return np.vstack((curr_px, curr_py, curr_pz)).T @ rot_mat

'''
=========== END PHOTON GENERATION SUB-FUNCTIONS ==========

=============== END PHOTON GENERATION CODE ===============
'''




