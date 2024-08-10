#!/usr/bin/env python

from chroma.geometry import Material

import pandas as pd
import random

# read materials from database
# randomize functionality

class material_manager:
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
	def __init__(self, experiment_name, run_id):
		"""
        Initializes the material_manager with the given experiment name and run ID.
        
        Args:
            experiment_name (str): String used to identify each experiment.
            run_id (int): Identifier for the run.
        """

		self.material_data_path = '/workspace/data_files/data/' + experiment_name + '/bulk_materials_' + experiment_name + '.csv'
	
		self.run_id = run_id
		self.build_materials(run_id)
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

	def build_materials(self,run_id):
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
			# # # below define material properties without referring to the material property csv
			# if curr_name == 'silicon':
			# 	# column = ['Cu eta']
			# 	# copper_eta_ = pd.read_csv(self.savedata_path,usecols=column).to_numpy()
			# 	# copper_eta_all = copper_eta_.flatten()
			# 	# copper_eta = copper_eta_all[self.run_id]
			# 	# row['eta'] = round(random.uniform(0.72553,0.96400),4)
			# 	# row['eta'] = copper_eta
			# 	row['k'] = round(random.uniform(1.7223,2.0872),4)
			# # 	row['k'] = 1.413
			# 	print('silicon k',row['k'])



			# if curr_name == 'liquid xenon':	
			# # # # 	column = ['LXe Index of Refraction']
			# # # # 	LXe_eta_ = pd.read_csv(self.savedata_path,usecols=column).to_numpy()
			# # # # 	LXe_eta_all = LXe_eta_.flatten()
			# # # # 	LXe_eta = LXe_eta_all[self.run_id]
			# # # 	# row['refractive_index'] = round(random.uniform(1.59,1.78),4)
			#  	row['refractive_index'] = 1.60+(run_id)*0.01
			#  	print('LXe',row['refractive_index'])
			#  	# row['refractive_index'] = 1.69
			# # 	#row['refractive_index'] = LXe_eta
				

			# if curr_name == 'silicon':
			# 	row['eta'] = pd.read_csv('/home/chroma/Downloads/Silicon_Refractive_Indices.csv', usecols = ['n']).to_numpy().flatten()[run_id]
			# 	print(row['eta'])
			# 	row['k'] = pd.read_csv('/home/chroma/Downloads/Silicon_Refractive_Indices.csv', usecols = ['k']).to_numpy().flatten()[run_id]
			# 	print(row['k'])

			# if curr_name == 'copper':
			# 	row['eta'] = pd.read_csv('/workspace/data_files/Werner_Cu_Refractive_Indices.csv', usecols = ['n']).to_numpy().flatten()[run_id]
			# 	print('Cu',row['eta'])
			# 	row['k'] = pd.read_csv('/workspace/data_files/Werner_Cu_Refractive_Indices.csv', usecols = ['k']).to_numpy().flatten()[run_id]
			# 	print('Cu',row['k'])

			self.add_attributes(self.materials[curr_name],
								#refractive_index = r_i,
								#refractive_index = row['refractive_index']+random.uniform(-row['abs(r_i_error)'],row['abs(r_i_error)']), # only used when no surface model is defined
								refractive_index = row['refractive_index'],
								absorption_length = row['absorption_length'],
								scattering_length = row['scattering_length'],
								density = row['density'])

			self.material_props[curr_name] = dict(row)
			# print('material props',self.material_props)
			# try to update the real refractive_index to the csv. file
		# print(self.materials)



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
