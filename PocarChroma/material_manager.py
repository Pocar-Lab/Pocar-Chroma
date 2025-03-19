#!/usr/bin/env python

from chroma.geometry import Material

import pandas as pd
import random

# read materials from database
# randomize functionality

class MaterialManager:
	"""
    Manages materials by reading their properties from a CSV file and creating Material objects.
    
    Attributes:
        experiment_name (str): String used to identify each experiment.
        run_id (int): Identifier for the run.
        material_data_path (str): Path to the CSV file with material properties.
        materials_df (pd.DataFrame): DataFrame containing the material properties.
        materials (dict): Dictionary of Material objects.
        material_props (dict): Dictionary of material properties.
        global_material (Material): The global material used in the experiment.
    """
	def __init__(self, experiment_name):
		"""
        Initializes the material_manager with the given experiment name and run ID.
        
        Args:
            experiment_name (str): String used to identify each experiment.
            run_id (int): Identifier for the run.
        """

		self.material_data_path = '/workspace/data_files/data/' + experiment_name + '/bulk_materials_' + experiment_name + '.csv'
	
		self.build_materials()
		self.global_material = self.materials['liquid xenon']

	def add_attributes(self,
				curr_material,
				refractive_index = None,
				absorption_length = None,
				scattering_length = None,
				density = None):
		"""
        Sets the optical properties for a given material.
        
        Args:
            curr_material (Material): The Material object to which attributes are added.
            refractive_index (float): Refractive index of the material.
            absorption_length (float): Absorption length of the material.
            scattering_length (float): Scattering length of the material.
            density (float): Density of the material.
        """
		#set the optical index grabbed from csv file for simulation
		if refractive_index is not None:
			curr_material.set('refractive_index', refractive_index) 
		if absorption_length is not None:
			curr_material.set('absorption_length', absorption_length) 
		if scattering_length is not None:
			curr_material.set('scattering_length', scattering_length)
		if density is not None:
			curr_material.density = density

		# print(refractive_index)
		# return refractive_index

	def build_materials(self):
		"""
        Reads material properties from a CSV file and creates Material objects.
        
        Args:
            run_id (int): Identifier for the run.
        """
		# read in the csv file into dataframe
		self.materials_df = pd.read_csv(self.material_data_path)
		#print(self.materials_df)
		# iterate through all materials and create Material object, store into dictionary of materials
		self.materials = {}
		self.material_props = {}
		properties = self.materials_df.columns
		for index, row in self.materials_df.iterrows():
			curr_name = row['name'] #name of the material		
			self.materials[curr_name] = Material(name = curr_name)	
			self.add_attributes(self.materials[curr_name],
								#refractive_index = r_i,
								#refractive_index = row['refractive_index']+random.uniform(-row['abs(r_i_error)'],row['abs(r_i_error)']), # only used when no surface model is defined
								refractive_index = row['refractive_index'],
								absorption_length = row['absorption_length'],
								scattering_length = row['scattering_length'],
								density = row['density'])

			self.material_props[curr_name] = dict(row)
			self.material_props[curr_name]["eta"] = row["eta"]
			self.material_props[curr_name]["k"] = row["k"]




	def get_material(self, material_name):
		"""
        Retrieves the Material object for a given material name from the created materials dictionary.
        
        Args:
            material_name (str): The name of the material.
        
        Returns:
            Material: The Material object.
        
        Raises:
            Exception: If the material does not exist.
        """
		# check to see if material exists. if not, throw exception
		if material_name in self.materials:
			return self.materials[material_name]
		else:
			raise Exception('Material does not exist: ' + material_name)