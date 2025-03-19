#!/usr/bin/env python
from chroma.detector import Detector
from chroma.stl import mesh_from_stl
from chroma.geometry import Solid
from chroma import view
from chroma.loader import load_bvh


import pandas as pd
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from .material_manager import MaterialManager
from .surface_manager import SurfaceManager


class GeometryManager:
    """
    Manages the geometry of an experiment by reading component data from a CSV file,
    creating solid objects, and organizing them into a detector geometry.

    Attributes:
        experiment_name (str): String used to identify each experiment.
        mat_manager (material_manager): Instance of the material_manager class.
        surf_manager (surface_manager): Instance of the surface_manager class.
        global_geometry (Detector): The global detector geometry.
        geometry_data_path (str): Path to the CSV file with geometry component data.
        geometry_df (pd.DataFrame): DataFrame containing the geometry component data.
        solids (dict): Dictionary of solid objects.
    """

    def __init__(self, experiment_name, exclude=None, surf_manager = None):
        """
        Initializes the geometry_manager with the given experiment name and run ID.

        Args:
            experiment_name (str): String used to identify each experiment.
        """
        self.exclude = [] if exclude is None else exclude
        self.experiment_name = experiment_name
        self.mat_manager = MaterialManager(self.experiment_name) if surf_manager is None else surf_manager.mat_manager
        self.surf_manager = SurfaceManager(self.mat_manager, self.experiment_name) if surf_manager is None else surf_manager
        self.global_geometry = Detector(self.mat_manager.global_material)

        self.geometry_data_path = f"/workspace/data_files/data/{experiment_name}/geometry_components_{experiment_name}.csv"
        self.build_geometry()

        self.global_geometry.flatten()
        self.global_geometry.bvh = load_bvh(self.global_geometry)

    def build_geometry(self):
        """
        Builds the geometry by reading the CSV file and creating solid objects.
        Adds the solids to the global geometry based on their type.
        """

        # read in the csv file into dataframe
        self.geometry_df = pd.read_csv(self.geometry_data_path)

        # iterate through all geometries and create Solid object, store into dictionary of solids
        self.solids = {}
        for index, row in self.geometry_df.iterrows():
            curr_name = row["name"]

            if curr_name in self.exclude:
                continue

            mesh = mesh_from_stl(
                filename=row["stl_filepath"]
            )  # convert the stl files to mesh used in Chroma?
            inner_mat = self.mat_manager.get_material(row["inner_mat"])
            outer_mat = self.mat_manager.get_material(row["outer_mat"])

            if "killing surface" in self.geometry_df and bool(row["killing surface"]):
                surface = self.surf_manager.get_surface("killing surface")
            else:
                surface = self.surf_manager.get_surface(row["surface"])

            color = int(colors.cnames[row["color"]][1:], 16)

            curr_displacement = (
                row["displacement x"],
                row["displacement y"],
                row["displacement z"],
            )
            self.solids[curr_name] = Solid(
                mesh=mesh,
                material1=inner_mat,
                material2=outer_mat,
                surface=surface,
                color=color,
            )
            # check to see if it is a detecting volume, add to geometry, (pmt and other solid will be treated differently?)
            if row["solid_type"] == "pmt":
                self.global_geometry.add_pmt(
                    pmt=self.solids[curr_name],
                    rotation=None,
                    displacement=curr_displacement,
                )
            elif row["solid_type"] == "solid":
                self.global_geometry.add_solid(
                    solid=self.solids[curr_name],
                    rotation=None,
                    displacement=curr_displacement,
                )

    def get_solid_center(self, name):
        """
        Gets the center of a solid object.

        Args:
            name (str): The name of the solid.

        Returns:
            list: Coordinates of the center of the solid.
        """
        curr_mesh_triangles = self.solids[name].mesh.get_triangle_centers()
        return [
            np.mean(curr_mesh_triangles[:, 0]),
            np.mean(curr_mesh_triangles[:, 1]),
            np.mean(curr_mesh_triangles[:, 2]),
        ]
