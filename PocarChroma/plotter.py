
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
    photon_steps,
    axes,
    num_tracks = 1000,
    color = 'black',
    linewidth = 1
):
    # Format photon steps into tracks which can be plotted
    tracks = np.zeros((len(photon_steps), num_tracks, 3))
    for step in range(len(photon_steps)):
        tracks[step, :, :] = photon_steps[step].pos[:num_tracks]

    # Plot all the tracks
    for i in range(num_tracks):
        axes.plot(
            tracks[:, i, 0], tracks[:, i,  1], tracks[:, i, 2],
            # tracks.pos[i][0], tracks.pos[i][1], tracks.pos[i][2],
            color=color, linewidth=linewidth
        )

    return axes


def plot_chroma(geometry=None, tracks=None, tracks_num=1000, tracks_color='black', tracks_linewidth=1):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    axes.view_init(elev=90, azim=-90)
    if geometry is not None:
        plot_geometry(geometry, axes)
    if tracks is not None:
        plot_tracks(tracks, axes, tracks_num, tracks_color, tracks_linewidth)
    plt.show()
