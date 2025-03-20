#!/usr/bin/env python

from chroma.sim import Simulation
from chroma.event import Photons

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from chroma import gpu
import pycuda.tools
from .analysis_manager import analysis_manager


class run_manager:
    """
    Manages the simulation run, including initializing the simulation, generating primary photons,
    and propagating photons through the geometry.

    :param geometry_manager: The geometry manager that handles the geometry of the simulation.
    :type geometry_manager: GeometryManager
    :param experiment_name: The name of the experiment.
    :type experiment_name: str
    :param random_seed: The random seed for reproducibility.
    :type random_seed: int
    :param num_particles: The number of particles to simulate.
    :type num_particles: int
    :param plots: The plots to be generated during analysis.
    :type plots: list
    :param write: Whether to write output files, defaults to False.
    :type write: boolean
    :param pg: An optional primary generator instance. If None, a new primary generator will be created.
    :type pg: primary_generator, optional
    :param batches: Whether to run simulations in batches, defaults to False.
    :type batches: boolean
    """

    def __init__(
        self,
        geometry_manager,
        experiment_name,
        random_seed,
        num_particles,
        plots,
        write=False,
        pg=None,
        batches = False,
        batch_amount = 2_000_000,
        nthreads_per_block = 64,
        max_blocks = 1024,
        propagation_seed = 20000000,
        
    ):
        self.num_steps = 15
        self.seed = random_seed
        self.gm = geometry_manager
        self.center_pos = self.gm.get_solid_center(name="source")
        self.num_particles = num_particles
        
        #test parameters for gpu optimization, be careful
        self.nthreads = nthreads_per_block
        self.max_blocks = max_blocks
        self.propagation_seed = propagation_seed

        self.given_pg = pg
        self.pg = pg
        self.photon_flags = []
        self.photon_dir = []
        self.photon_pos = []
        self.photon_tracks = []
        self.first_photon_tracks = []
        self.total_photons = []

        self.interactions = {
            "RAYLEIGH_SCATTER": 4,
            "REFLECT_DIFFUSE": 5,
            "REFLECT_SPECULAR": 6,
            "SURFACE_REEMIT": 7,
            "SURFACE_TRANSMIT": 8,
            "BULK_REEMIT": 9,
            "CHERENKOV": 10,
            "SCINTILLATION": 11,
        }

        self.total_particle_histories = {
            curr_int: np.zeros(self.num_particles, dtype=int)
            for curr_int in self.interactions.keys()
        }

        #run the simulation either all at once or in batches depending on the value of 'batches'
        #running more than 2-3 million all at once will cause the program to crash unless your GPU has a lot of vram so be careful
        if not batches or self.num_particles <= 2_000_000:
            self.run_single_simulation(random_seed, num_particles)
            self.ana_man = analysis_manager(
                self.gm,
                experiment_name,
                plots,
                self.photons,
                self.photon_tracks,
                self.seed,
                self.particle_histories,
            )
        else:
            self.run_batches(random_seed, num_particles)
            myPhotons = MyPhotons(photon_pos = self.photon_pos, photon_dir = self.photon_dir, photon_flags = self.photon_flags)
            self.ana_man = analysis_manager(
                self.gm,
                experiment_name,
                plots,
                myPhotons,  #8/6/2024 Changed from self.photons. This was in attempt to incorporate multiple simulations. If needed put it back.
                photon_tracks = self.first_photon_tracks, # photon tracks from just the first part of simulations
                seed = self.seed,
                histories = self.particle_histories,
            )

        
        
    def propagate_photon(self,num_particles):
        """
        Propagates photons through the geometry using the GPU, collecting their positions and interaction histories.

        :param num_particles: The number of particles to simulate.
        :type num_particles: int

        :return: None
        """
        nthreads_per_block = self.nthreads
        max_blocks = self.max_blocks
        seed = self.propagation_seed

        gpu_photons = gpu.GPUPhotons(self.pg.primary_photons)
        gpu_geometry = gpu.GPUGeometry(self.gm.global_geometry)
        self.photon_tracks = np.zeros((self.num_steps + 1, num_particles, 3))
        self.photon_tracks[0, :, :] = self.pg.positions
        rng_states = gpu.get_rng_states(nthreads_per_block * max_blocks, seed=seed)
        self.particle_histories = {
            curr_int: np.zeros(num_particles, dtype=int)
            for curr_int in self.interactions.keys()
        }

        for i in range(self.num_steps):
            # propagate photons
            gpu_photons.propagate(
                gpu_geometry,
                rng_states,
                nthreads_per_block=nthreads_per_block,
                max_blocks=max_blocks,
                max_steps=1,
            )
            # from GPUPhoton get event.Photon object
            self.photons = gpu_photons.get()
            self.photon_tracks[i + 1, :, :] = self.photons.pos
            # collect information from flags about interaction history
            self.update_tallies(self.photons)
            # reset interaction history bits that are nonterminal
            new_flags = self.reset_nonterminal_flags(self.photons.flags)
            gpu_photons.flags[: num_particles].set(new_flags.astype(np.uint32))
        
        #simulation done, clear GPU cache to save memory
        pycuda.tools.clear_context_caches()

    def run_single_simulation(self,seed, batch_size):
        #runs a simulation with a set number of particless at a set seed using the simulation parameters passed into run manager initialization
        self.sim = Simulation(
            self.gm.global_geometry, seed = seed, geant4_processes=0
        )   
        if self.given_pg is None:
            self.pg = primary_generator(
                batch_size, center_pos=self.center_pos
            )
        else:
            self.pg = self.given_pg
        self.propagate_photon(batch_size)
    

    #runs a simulation of a larger number of photons in smaller batches to avoid running out of memory.
    #the seed is incremented after each batch to ensure a different seed but also have the process be repeatable from the same initial seed. 
    # This means that a simulation of 100 million photons at seed 1000 will use seeds 1000-1049, so be careful not to call initial seeds too close together as there may be overlap
    def run_batches(self, seed, num_particles):

        #running more than 2-3 million all at once will cause the program to crash unless your GPU has a lot of vram so be careful
        batch_size = self.batch_amount
        num_sims = math.ceil(num_particles / batch_size)
        
        total_flags = []
        total_dir = []
        total_pos = []
        total_photon_tracks = []
        total_particle_histories = {key: np.zeros(num_particles, dtype=int) for key in self.interactions.keys()}
        
        for i in range(num_sims):
            current_batch_size = min(batch_size, num_particles - i * batch_size)
            
            # Run a single simulation with the current batch
            self.run_single_simulation(seed + i, current_batch_size)
            
            if i == 0:
                self.first_photon_tracks = self.photon_tracks
            total_flags.append(self.photons.flags)
            total_dir.append(self.photons.dir)
            total_pos.append(self.photons.pos)  
            # total_photon_tracks.append(self.photon_tracks) we do not need all photon tracks for tens of millions of photons, we can only plot on the order of thousands.
            
            #update particle histories
            start_idx = i * batch_size
            end_idx = start_idx + current_batch_size
            for key in self.particle_histories.keys():
                total_particle_histories[key][start_idx:end_idx] = self.particle_histories[key]
            
        
        # Combine the photon tracks
        # self.photon_tracks = np.concatenate(total_photon_tracks, axis=1)
        
        #combine the extracted photon data
        self.photon_flags = np.concatenate(total_flags)
        self.photon_dir = np.concatenate(total_dir)
        self.photon_pos = np.concatenate(total_pos)
        
        self.particle_histories = total_particle_histories
        
        # clean up
        del total_flags, total_dir, total_pos, total_photon_tracks

    def reset_nonterminal_flags(self, flag_list):
        """
        Resets non-terminal interaction flags in the given flag list.

        :param flag_list: The list of photon interaction flags.
        :type flag_list: numpy.ndarray
        :return: The updated flag list with non-terminal flags reset.
        :rtype: numpy.ndarray
        """
        # mask = 1111111111111111111000000001111
        mask = 2147479567
        return flag_list & mask

    def update_tallies(self, photons):
        """
        Updates the tally of interactions for the given photons.

        :param photons: The photons whose interaction histories are to be updated.
        :type photons: Photons
        :return: None
        """
        for key, value in self.interactions.items():
            curr_tally = (photons.flags & (0x1 << value)).astype(bool).astype(int)
            self.particle_histories[key] += curr_tally

    
        #a new class to store all the necessary information as photons that we use in analysis manager. this is not stored in te GPU and has less information so it can accomodate a larger amount of photons
class MyPhotons:

    def __init__(self,photon_pos,photon_dir, photon_flags):
        self.pos = photon_pos
        self.dir = photon_dir
        self.flags = photon_flags

    def __len__(self):
        return len(self.flags)

class primary_generator:  # photon generator
    """
    Generates primary photons for the simulation, including their positions, directions, polarization, and wavelengths.

    :param num_particles: The number of particles to generate.
    :type num_particles: int
    :param center_pos: The center position of the photon source.
    :type center_pos: list
    """

    # C++: methods/functions
    # def __init__(self, num_particles, center_pos = [0, 0, 0], delta_placement = 0.0):
    def __init__(self, num_particles, center_pos=[0, 0, 0], r=0, source_type = "isotropic", beam_theta = 0, beam_phi = 0):
        self.num_particles = num_particles
        self.center_pos = center_pos

        # positions
        if r == 0:
            self.positions = self.get_point_source_pos(
                x0=self.center_pos[0], y0=self.center_pos[1], z0=self.center_pos[2]
            )
        else:
            self.positions = self.get_xz_disk_source_pos(
                x0=self.center_pos[0],
                y0=self.center_pos[1],
                z0=self.center_pos[2],
                r=r,
            )

        #directions
        if source_type == "isotropic":
            isotropic_dir = self.get_isotropic_dir()
            self.directions = isotropic_dir
        elif source_type == "beam":
            beam_dir = self.get_beam_dir(beam_theta,beam_phi)
            self.directions = beam_dir

        elif source_type == "y_cone":
            cone_angle = (np.pi / 32.0)
            y_cone_dir = self.get_y_cone_dir(angle = cone_angle, positive = True)
            self.directions = y_cone_dir
        else:
            raise Exception("Unknown Source Type. Currently supported source types are 'isotropic', 'beam', and 'y_cone'.")


        # polarization
        self.polarization = np.cross(self.directions, self.get_isotropic_dir())
        # wavelengths
        self.wavelengths = np.ones(self.num_particles) * 175.0
        # create photons
        self.primary_photons = Photons(
            self.positions, self.directions, self.polarization, self.wavelengths
        )

    def get_point_source_pos(self, x0, y0, z0):
        """
        Generates positions for a point source at the given coordinates.

        :param x0: The x-coordinate of the source.
        :type x0: float
        :param y0: The y-coordinate of the source.
        :type y0: float
        :param z0: The z-coordinate of the source.
        :type z0: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
        return np.tile([x0, y0, z0], (self.num_particles, 1))

    # input center of disk and radius
    def get_xy_disk_source_pos(self, x0, y0, z0, r):
        """
        Generates positions for a disk source in the xy-plane with the given center and radius.

        :param x0: The x-coordinate of the disk center.
        :type x0: float
        :param y0: The y-coordinate of the disk center.
        :type y0: float
        :param z0: The z-coordinate of the disk center.
        :type z0: float
        :param r: The radius of the disk.
        :type r: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
        # pos_array = np.empty(self.num_particles, 3)
        curr_sqrtr = np.sqrt(np.random.uniform(0, r, self.num_particles))
        curr_theta = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        curr_x = curr_sqrtr * np.cos(curr_theta) + x0
        curr_y = curr_sqrtr * np.sin(curr_theta) + y0
        curr_z = np.ones(self.num_particles) * z0
        return np.vstack((curr_x, curr_y, curr_z)).T

    def get_xz_disk_source_pos(self, x0, y0, z0, r):
        """
        Generates positions for a disk source in the xz-plane with the given center and radius.

        :param x0: The x-coordinate of the disk center.
        :type x0: float
        :param y0: The y-coordinate of the disk center.
        :type y0: float
        :param z0: The z-coordinate of the disk center.
        :type z0: float
        :param r: The radius of the disk.
        :type r: float
        :return: The array of positions.
        :rtype: numpy.ndarray
        """
        # pos_array = np.empty(self.num_particles, 3)
        curr_sqrtr = np.sqrt(np.random.uniform(0, r, self.num_particles))
        curr_theta = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        curr_x = curr_sqrtr * np.cos(curr_theta) + x0
        curr_y = np.ones(self.num_particles) * y0
        curr_z = curr_sqrtr * np.sin(curr_theta) + z0
        return np.vstack((curr_x, curr_y, curr_z)).T

    # def get_yz_disk_source_pos(self, x0, y0, z0, r):

    # def get_xy_cylinder_source_pos(self, x0, y0, z0, r, h):

    def get_isotropic_dir(self):
        """
        Generates isotropic directions for the photons.

        :return: The array of directions.
        :rtype: numpy.ndarray
        """
        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(-1.0, 1.0, self.num_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = np.cos(phi) * sin_theta
        curr_py = np.sin(phi) * sin_theta
        curr_pz = cos_theta
        return np.vstack((curr_px, curr_py, curr_pz)).T


    def get_beam_dir(self, theta, phi):
        """
        Generates beam direction based on angles theta and phi. 
        The SiPM has been located in the positive y direction so that is the axis we will use for the beam.
        i.e. a azimuthal angle theta = 0 will correspond to the vector [0,1,0], hitting the detector.

        :param theta: The azimuthal angle of the beam direction.
        :type theta: float
        :param phi: The polar angle of the beam direction.
        :type phi: float
        :return: The array of beam directions.
        :rtype: numpy.ndarray
        """
        px = np.sin(theta) * np.sin(phi)
        py = np.cos(theta)
        pz = np.sin(theta) * np.cos(phi)
        
        return np.tile([px, py, pz], (self.num_particles, 1))
    

    def get_x_cone_dir(self, angle, positive=True):
        """
        Generates directions for a cone in the x direction.

        :param angle: The angle of the cone.
        :type angle: float
        :param positive: Whether the cone is in the positive x direction, defaults to True.
        :type positive: bool, optional
        :return: The array of cone directions.
        :rtype: numpy.ndarray
        """
        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(np.cos(angle / 2), 1, self.num_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = cos_theta
        curr_py = np.cos(phi) * sin_theta
        curr_pz = np.sin(phi) * sin_theta
        if not positive:
            curr_px *= -1
        return np.vstack((curr_px, curr_py, curr_pz)).T

    def get_y_cone_dir(self, angle, positive=True):
        """
        Generates directions for a cone in the y direction.

        :param angle: The angle of the cone.
        :type angle: float
        :param positive: Whether the cone is in the positive y direction, defaults to True.
        :type positive: bool, optional
        :return: The array of cone directions.
        :rtype: numpy.ndarray
        """
        phi = np.random.uniform(0, 2.0 * np.pi, self.num_particles)
        cos_theta = np.random.uniform(np.cos(angle / 2), 1, self.num_particles)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        curr_px = np.cos(phi) * sin_theta
        curr_py = cos_theta
        curr_pz = np.sin(phi) * sin_theta
        if not positive:
            curr_py *= -1

        return np.vstack((curr_px, curr_py, curr_pz)).T

