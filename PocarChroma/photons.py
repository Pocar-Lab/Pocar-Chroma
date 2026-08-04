from chroma.sim import Simulation
from chroma.event import Photons

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from enum import Enum, IntEnum
from typing import Optional

from chroma import gpu
import pycuda.tools
from .analysis_manager import analysis_manager


class Shape(Enum):
    POINT = 1
    DISK = 2

class Emission(Enum):
    ISOTROPIC = 1
    BEAM = 2
    CONE = 3

class Axis(Enum):
    X = 1
    Y = 2
    Z = 3

def photon_generator(
    seed = 5555,
    max_photons = 1_000_000,
    batch_size = 1_000_000,
    wavelength = 178.0,
    shape: Shape = Shape.POINT,
    direction: Emission = Emission.ISOTROPIC,
    source_axis: Axis = Axis.Y,
    source_location = [0,0,0],
    source_r: Optional[float] = None,
    beam_azimuth: Optional[float] = None,
    beam_declination: Optional[float] = None,
    cone_angle: Optional[float] = None
    ):
    '''
    Photon generator functions, return initial Chroma Photons object to be propagated.
    '''

    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # init some dicts for arguments
    position_args = {}
    direction_args = {}

    # Set up rotation matrix:
    if source_axis == Axis.X:
            rot_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    elif source_axis == Axis.Y:
            rot_mat = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]])
    elif source_axis == Axis.Z:
            rot_mat = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]])

    # Set up position generator
    if shape == Shape.POINT:
            position_function = pg_point_source
    elif shape == Shape.DISK:
            position_function = pg_disk_source
            position_args['rng'] = rng
            position_args['rot_matrix'] = rot_mat
            position_args['source_r'] = source_r

    if direction == Emission.ISOTROPIC:
            direction_function = pg_isotropic_source
            direction_args['rng'] = rng
    elif direction == Emission.BEAM:
            direction_function = pg_beam_source
            direction_args['rot_matrix'] = rot_mat
            direction_args['beam_azimuth'] = beam_azimuth
            direction_args['beam_declination'] = beam_declination
    elif direction == Emission.CONE:
            direction_function = pg_cone_source
            direction_args['rot_matrix'] = rot_mat
            direction_args['rng'] = rng
            direction_args['cone_angle'] = cone_angle

    total_photons = 0
    while total_photons < max_photons:
        # Check if this next batch of photons will exceed the total number of photons requested
        n_photons = min(batch_size, max_photons - total_photons)

        positions = position_function(
            n_photons=n_photons,
            source_location=source_location,
            **position_args)
        directions = direction_function(
            n_photons=n_photons,
            **direction_args)

        polarizations = np.cross(directions, pg_isotropic_source(n_photons=n_photons, rng=rng))
        wavelengths = np.ones(n_photons) * wavelength

        photons = Photons(positions, directions, polarizations, wavelengths)
        total_photons += n_photons
        yield photons # after initialization, the generator stops here, waiting for the .send method to provide n_photons


### PHOTON GENERATION SUB-FUNCTIONS

# Here are a family of sub-functions that are used by the photon generator object
# Source functions should take number of particles, a source location, and a kwargs statement
# Direction functions should take a number of particles and a kwargs statment
# Source kwargs:
#     - source_r: Radius of the source
#     - rng: numpy random number generator object
#     - rot_matrix: a 3x3 np array to rotate the resulting vectors into the desired orientation using matrix multiplication

# Direction kwargs:
#     - beam_declination: The angle between source_axis and the beam (in radians)
#     - beam_azimuth: The angle of the beam about source_axis (in radians)
#     - cone_angle: the angle of the cone (in radians)
#     - rng
#     - rot_matrix

# These functions aren't neccesarily intended to be called by external programs, but they can if you want them to

def pg_point_source(n_photons, source_location):
    return np.tile(source_location, (n_photons, 1))

def pg_disk_source(n_photons, source_location, source_r, rng, rot_mat):
    curr_sqrtr = np.sqrt(rng.uniform(0, source_r, n_photons))
    curr_theta = rng.uniform(0, 2.0 * np.pi, n_photons)

    curr_x = np.ones(n_photons) * source_location[0]
    curr_y = curr_sqrtr * np.sin(curr_theta) + source_location[1]
    curr_z = curr_sqrtr * np.cos(curr_theta) + source_location[2]

    # make an array of the positions and then rotate it to make row-vectors
    positions = np.vstack((curr_x, curr_y, curr_z)).T
    # rotate the positions into the appropriate reference frame
    return positions @ rot_mat

def pg_isotropic_source(n_photons, rng):
    '''Make spherically isotropic directions for the photons.'''

    phi = rng.uniform(0, 2.0 * np.pi, n_photons)
    cos_theta = rng.uniform(-1.0, 1.0, n_photons)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    curr_px = np.cos(phi) * sin_theta
    curr_py = np.sin(phi) * sin_theta
    curr_pz = cos_theta
    return np.vstack((curr_px, curr_py, curr_pz)).T

def pg_beam_source(n_photons, beam_declination, beam_azimuth, rot_matrix):
    '''Make a beam of particles in a given direction, as specified by declination and azimuth.'''
    px = np.cos(beam_declination)
    py = np.sin(beam_declination) * np.sin(beam_azimuth)
    pz = np.sin(beam_declination) * np.cos(beam_azimuth)
    directions = np.tile([px, py, pz], (n_photons, 1))
    return  directions @ rot_matrix

def pg_cone_source(n_photons, rng, cone_angle, rot_matrix):
    '''Make a cone source, centered about the appropriate axis, with a given angle.'''

    phi = rng.uniform(0, 2.0 * np.pi, n_photons)
    cos_theta = rng.uniform(np.cos(cone_angle), 1, n_photons)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

    curr_px = cos_theta
    curr_py = np.cos(phi) * sin_theta
    curr_pz = np.sin(phi) * sin_theta

    return np.vstack((curr_px, curr_py, curr_pz)).T @ rot_matrix



def propagate(
    photons,    # This should be a chroma Photons object. NOT the photon generator
    geometry,   # This should be a chroma geometry object
    seed = 5555,
    track_return_ct = 0,
    num_steps = 15,

    # The following parameters are highly GPU dependant, change at your own risk
    n_threads = 64,
    max_blocks = 1024,
    ):
    '''Propagates photons through geometry'''

    # Get number of photons from Photons object
    n_photons = photons.pos.shape[0]

    # Print a warning if attempting to propagate a large number of photons
    if n_photons > 2000000:
        print('WARNING: Attempting to propagate more than 2 million photons. This may crash the GPU!')

    # raise an error if more photon tracks are requested than photons simulated
    if track_return_ct > n_photons:
        raise ValueError('More photon tracks requested than photons simulated!')

    # start a simulation
    # TODO what does this do?
    sim = Simulation(geometry.global_geometry, seed=seed, geant4_processes=0)

    # initialize GPU states
    gpu_photons = gpu.GPUPhotons(photons)
    gpu_geometry = gpu.GPUGeometry(geometry.global_geometry)

    rng_states = gpu.get_rng_states(n_threads * max_blocks, seed=seed)

    photon_steps = np.empty(num_steps + 1, dtype=Photons) # Record each step and the initial state
    photon_steps[0] = photons
    for current_step in range(1, num_steps + 1):
        gpu_photons.propagate(
            gpu_geometry,
            rng_states,
            nthreads_per_block=n_threads,
            max_blocks=max_blocks,
            max_steps=1,
        )

        # Get the propagated chroma Photons object
        photons = gpu_photons.get()

        # This is reset non-terminal flags from run_manager
        # 0b1111111111111111111000000001111
        new_flags = photons.flags & 2147479567 # TODO why this number?
        gpu_photons.flags[: n_photons].set(new_flags.astype(np.uint32))

        photon_steps[current_step] = photons

    # simulation done, clear GPU cache to save memory
    pycuda.tools.clear_context_caches()

    return photon_steps


### PARTICLE HISTORIES

class Interaction(IntEnum):
    NO_HIT           = 0x1 << 0
    BULK_ABSORB      = 0x1 << 1
    SURFACE_DETECT   = 0x1 << 2
    SURFACE_ABSORB   = 0x1 << 3
    RAYLEIGH_SCATTER = 0x1 << 4
    REFLECT_DIFFUSE  = 0x1 << 5
    REFLECT_SPECULAR = 0x1 << 6
    SURFACE_REEMIT   = 0x1 << 7
    SURFACE_TRANSMIT = 0x1 << 8
    BULK_REEMIT      = 0x1 << 9
    CHERENKOV        = 0x1 << 10
    SCINTILLATION    = 0x1 << 11
    PREV_ABSORB      = 0x1 << 12
    NAN_ABORT        = 0x1 << 31

def triangles_from_name(geometry_manager, part_name):
    # Find the index of the part_name in solids keys
    part_idx = list(geometry_manager.solids.keys()).index(part_name)
    # Get all triangle indices where solid_id equals part_idx
    triangles = [i for i, val in enumerate(geometry_manager.global_geometry.solid_id)
                   if val == part_idx]
    return np.array(triangles)

class Filter():
    def __init__(self, geometry_manager, interactions, parts=[]):
        """Create photon filter for interaction flags, and collisions with parts if given.
        Run update after each batch of photons is propagated,
        at the end res is a set that contains all the photon indices which passed the filter."""
        self.parts = parts
        if parts: # if part names are given, get a list of triangle indices for all parts
            list_tri = [triangles_from_name(geometry_manager, part) for part in parts]
            self.triangles = np.concatenate(list_tri)
        self.interactions = int(interactions)
        self.batch_num = 0
        self.res = set() # Resulting set of indices of photons which pass filter

    def update(self, photon_steps):
        "Update filter for each photon batch"
        for step in photon_steps:
            interacted = (step.flags & self.interactions) != 0
            if self.parts:
                collision = np.isin(step.last_hit_triangles, self.triangles)
                interacted &= collision
            self.res |= set(np.flatnonzero(interacted) + self.batch_num*len(step.pos))
        self.batch_num += 1
