#!/usr/bin/env python

import sys, getopt
import numpy as np

from PocarChroma.geometry_manager import geometry_manager
from PocarChroma.run_manager import run_manager
from PocarChroma.material_manager import material_manager
from PocarChroma.surface_manager import surface_manager
from PocarChroma.analysis_manager import analysis_manager


import time

def usage():
    print ("=====================================================================")
    print ("  The minimum paramaters the simulation needs are:")
    print ("    (1) '-e' <Str>              name of experiment to be simulated.")
    print ("  Additional options can be chosen:")
    print ("  	(2) '-n' <#>	            number of photons to be simulated.") 
    print ("  	(3) '-s' <#>                choose the seed number")
    print ("    (4) '-p' <Str1,Str2,...>    choose which plots to run")
    print ("=====================================================================")

def main():
    args = sys.argv[1:]
    try:
        opts, args = getopt.getopt(args, "n:s:r:e:p:")
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        usage()
        sys.exit()

    experiment_name = None
    num_particles = 1_000_000
    seed = np.random.randint(0,1000000)
    run_id = 1
    visualize = False
    plots = []
    write = False

    for opt, arg in opts:
        if opt == '-e':
            experiment_name = str(arg)
        elif opt == '-n':
            num_particles = int(arg)
        elif opt == '-s':
            seed = int(arg)
        elif opt == '-p':
            plots = [i.strip() for i in arg.split(',')]


    if not experiment_name:
        print("Please input an experiment name")
        usage()
        sys.exit()

    print('Experiment Name:         ' + experiment_name)
    print('Number of particles:     ' + str(num_particles))
    print('Seed:             ' + str(seed))
    print('Visualize:               ' + str(visualize))
    if(len(plots) > 0):
        print('Plots:                   ' + ', '.join(plots))
    else:
        print('Plots:                   ' + 'None')
    print('Saving Data:             ' + str(write))

    mm = material_manager(experiment_name=experiment_name)
    sm = surface_manager(material_manager = mm, experiment_name = experiment_name)
    gm = geometry_manager(experiment_name=experiment_name,surf_manager = sm)
    photons, photon_tracks, particle_histories = run_manager(geometry_manager=gm, experiment_name=experiment_name, random_seed=seed, num_particles=num_particles,plots=plots)
    am = analysis_manager(
                gm,
                experiment_name,
                plots,
                photons,
                photon_tracks,
                seed,
                particle_histories,
            )
    return am.get_end_time()


if __name__ == '__main__':
	s = time.time()
	e = main()
	print(f'The simulation run time is: {e - s} s')
