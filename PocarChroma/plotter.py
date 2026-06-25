
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# mpl.use("Agg")
from stl import mesh
from matplotlib import colors
from mpl_toolkits import mplot3d
from array import array
import time
import os
import file_handler as fh


def plot_geometry(
    geometry_df,
    axes,
):
    '''
    plots geometry. 
    :param geometry_df: dataframe of geometry (from geometry CSV)
    :type geometry_df: Dataframe
    :param axes: an mpl 3d axes object (optional)
    :type axes: Axes
    '''




    # Get columns from geometry dataframe
    part_name = geometry_df['name']
    stl_names = geometry_df["stl_filepath"]
    colors = geometry_df["color"]
    x_displacement = geometry_df["displacement x"]
    y_displacement = geometry_df["displacement y"]
    z_displacement = geometry_df["displacement z"]

    # iterate through rows
    for (
        curr_part_name,
        curr_filename,
        curr_color,
        current_x_displacement,
        current_y_displacement,
        current_z_displacement,
    ) in zip(part_name, stl_names, colors, x_displacement, y_displacement, z_displacement):

        m = mesh.Mesh.from_file(curr_filename)
        m.translate([
            current_x_displacement,
            current_y_displacement,
            current_z_displacement
        ])

        poly3d = Poly3DCollection(m.vectors)
        poly3d.set_alpha(0.2)
        poly3d.set_edgecolor(None)
        poly3d.set_facecolor(curr_color)
        axes.add_collection3d(poly3d)

    
    scale = m.points.flatten()

    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel("x position (mm)")
    axes.set_ylabel("y position (mm)")
    axes.set_zlabel("z position (mm)")
    return axes




def plot_tracks(
    tracks,
    axes,
    num_tracks_to_plot = 1000,
    color = 'black',
    linewidth = 1
):
    # in the original code, tracks were selected at random to plot if too many tracks were provided, but honestly I don't know why

    if not num_tracks_to_plot > tracks.shape[1]:
        tracks = tracks[:, :num_tracks_to_plot, :]
    else:
        pass

    for i in range(tracks.shape[1]):
        axes.plot(
            tracks[:, i, 0], tracks[:, i,  1], tracks[:, i, 2], color=color, linewidth=linewidth
        )

    return axes








