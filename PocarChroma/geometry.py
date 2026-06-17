'''
This is a class-less reimplimentation of geometry_manager.py, material_manager.py and surface_manager.py

Changes Compared to geometry_manager.py
- Rather than being handed paths to csv files, has the option to build the geometry from dataframes
- Integrating materials_manager and surface_manager

'''
from chroma.detector import Detector
from chroma.stl import mesh_from_stl
from chroma.geometry import Solid
from chroma import view
from chroma.loader import load_bvh
from chroma.geometry import Material
from chroma.geometry import Surface


import pandas as pd
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from .material_manager import material_manager
from .surface_manager import surface_manager





def build_geometry(
    geo_path:str,   # Path to the geometry csv file
    mat_path:str,   # Path to the materials csv file
    surf_path:str:  # Path to the surfaces csv file

):
    pass

def build_geometry_from_df(
    geo_df,     # geometry dataframe, from csv_file
    mat_df,     # materials dataframe
    surf_df     # surfaces dataframe

):
    pass



def build_materials(
    materials_df    #dataframe of materials
):

    pass

def build_surfaces():


    pass