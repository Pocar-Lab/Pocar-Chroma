#!/usr/bin/env python

from chroma.geometry import Surface
from chroma.geometry import DichroicProps,SiPMEmpiricalProps,ArrayProps2D
import chroma.geometry
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Manage the surface optical model (of the SiPM and other)? Mainly using model 0 which is the standard Fresnel

class SurfaceManager:
    """
    Manages the surfaces used in the simulation, including their properties and construction based on input data.

    :param material_manager: The manager handling material properties.
    :type material_manager: MaterialManager
    :param experiment_name: The name of the experiment.
    :type experiment_name: str
    """

    def __init__(self, material_manager, experiment_name):
        self.surfaces = {}
        self.surface_props = {}
        self.surface_data_path = f"/workspace/data_files/data/{experiment_name}/surface_props_{experiment_name}.csv"
        
        #the sipm reflectivity was determined by another lab experimentally. The lab reported an upperbound and a lower bound for the reflectivites.
        #the FBK.csv file was created by averaging the upper and lower bounds
        self.SiPMAOIref_path = "/workspace/data_files/FBK.csv"
        # self.SiPMAOIref_path = '/workspace/data_files/FBK reflectivity_upper_bound.csv'
        # self.SiPMAOIref_path ='/workspace/data_files/FBK reflectivity_lower_bound.csv'

        self.mat_manager = material_manager
        self.experiment_name = experiment_name
        self.wavelengths = chroma.geometry.standard_wavelengths
        self.num_wavelengths = len(self.wavelengths)
        self.build_surfaces()

    def build_surfaces(self):
        """
        Constructs surface objects based on data read from a CSV file.

        :return: None
        """
        self.surfaces_df = pd.read_csv(self.surface_data_path)

        for index, row in self.surfaces_df.iterrows():
            curr_name = row["name"]
            curr_inner_mat_name = row["inner_mat"]
            curr_outer_mat_name = row["outer_mat"]
            curr_model_id = row["model_id"]
            curr_reflect_specular = row["reflect_specular"]
            curr_reflect_diffuse = row["reflect_diffuse"]
            curr_reflect_lobed = row["reflect_lobed"]
            curr_sigma_alpha = row["sigma_alpha"]

            if curr_model_id == 0:
                curr_surface = Surface(curr_name, model=curr_model_id)
                curr_surface.set("detect", 1.0)

            elif curr_model_id == 3:
                curr_surface = self.create_dichroic_surface(
                    curr_name, curr_inner_mat_name, curr_outer_mat_name
                )
            elif curr_model_id == 4:
                curr_surface = self.create_dielectric_metal_surface(
                    curr_name, curr_inner_mat_name
                )

            # below try to include the SiPM empirical surface
            elif curr_model_id == 5:
                curr_surface = self.SiPMEmpirical_surface(curr_name)

            elif curr_model_id == 6:
                curr_surface = self.create_lobed_surface(curr_name,curr_inner_mat_name)

            # Sili: added on 11/17/2022 to build a killing surface
            elif curr_model_id == 8:
                curr_surface = Surface(curr_name, model=0)
                # Photon will be killed(absorbed) when reaching the surface
                curr_surface.set("absorb", 1)

            # for teflon
            elif curr_model_id == 9:
                curr_surface = Surface(curr_name, model=0)
                # curr_surface.set('absorb',0.05)
            else:
                curr_surface = None

            if curr_surface is not None:
                curr_surface.set("reflect_specular", curr_reflect_specular)
                curr_surface.set("reflect_diffuse", curr_reflect_diffuse)
                curr_surface.set("reflect_lobed", curr_reflect_lobed)
                curr_surface.set("sigma_alpha", curr_sigma_alpha)

            self.surfaces[curr_name] = curr_surface

    def get_surface(self, surface_name):
        """
        Retrieves a surface object by its name.

        :param surface_name: The name of the surface.
        :type surface_name: str
        :return: The surface object.
        :rtype: Surface
        :raises Exception: If the surface does not exist.
        """
        if surface_name == "None":
            return None
        elif surface_name in self.surfaces:
            return self.surfaces[surface_name]
        else:
            raise Exception("Surface does not exist: " + surface_name)

    def create_dielectric_metal_surface(self, name, inner_mat):
        """
        Creates a dielectric-metal surface with given properties.

        :param name: The name of the surface.
        :type name: str
        :param inner_mat: The name of the inner material.
        :type inner_mat: str
        :return: The dielectric-metal surface object.
        :rtype: Surface
        """
        dielectric_metal_surface = Surface(name, model=4)
        eta2 = float(self.mat_manager.material_props[inner_mat]["eta"])
        k2 = float(self.mat_manager.material_props[inner_mat]["k"])
        dielectric_metal_surface.set("eta", eta2)
        dielectric_metal_surface.set("k", k2)

        return dielectric_metal_surface

    def SiPMEmpirical_surface(self, name):
        """
        Creates an empirical SiPM surface based on reflectivity data.

        :param name: The name of the surface.
        :type name: str
        :return: The empirical SiPM surface object.
        :rtype: Surface
        """

        sipmEmpirical_surface = Surface(name, model=5)

        self.SiPMreflctivity = pd.read_csv(self.SiPMAOIref_path)
        ref = []
        rela_PDE = []
        inci_theta_csv = self.SiPMreflctivity["AOI"].tolist()
        inci_theta = np.radians(inci_theta_csv)
        ref_csv = self.SiPMreflctivity["reflectivity"].tolist()

        for i in range(0, len(inci_theta_csv)):
            r_SiPM = np.zeros((self.num_wavelengths, 2), dtype=np.float32)
            r_SiPM[:, 0] = self.wavelengths
            r_SiPM[:, 1] = ref_csv[i]
            ref.append(r_SiPM)
            relative_PDE = np.zeros((self.num_wavelengths, 2), dtype=np.float32)
            relative_PDE[:, 0] = self.wavelengths
            relative_PDE[:, 1] = 1.0
            rela_PDE.append(relative_PDE)

        SiPM_props = SiPMEmpiricalProps(inci_theta, ref, rela_PDE)
        sipmEmpirical_surface.sipmEmpirical_props = SiPM_props

        return sipmEmpirical_surface

    def create_dichroic_surface(self, name, inner_mat, outer_mat):
        """
        Creates a dichroic surface with given properties.

        :param name: The name of the surface.
        :type name: str
        :param inner_mat: The name of the inner material.
        :type inner_mat: str
        :param outer_mat: The name of the outer material.
        :type outer_mat: str
        :return: The dichroic surface object.
        :rtype: Surface
        """
        dichroic_surface = Surface(name, model=3)
        n1 = self.mat_manager.material_props[outer_mat]["refractive_index"]
        # Sili: the eta2 and k2 are randomize with some uncertainty
        eta2 = self.mat_manager.material_props[inner_mat]["eta"] + random.uniform(
            -self.mat_manager.material_props[inner_mat]["abs(eta_error)"],
            self.mat_manager.material_props[inner_mat]["abs(eta_error)"],
        )
        k2 = self.mat_manager.material_props[inner_mat]["k"] + random.uniform(
            -self.mat_manager.material_props[inner_mat]["abs(k_error)"],
            self.mat_manager.material_props[inner_mat]["abs(k_error)"],
        )
        # Sili
        self.get_eta2_k2(eta2, k2)
        theta, reflectance, transmittance = self.calc_R_T(n1, eta2, k2, num_angles=500)
        R = []
        T = []
        for idx in range(len(theta)):
            curr_R = np.zeros((self.num_wavelengths, 2), dtype=np.float32)
            curr_R[:, 0] = self.wavelengths
            # curr_R[:, 1] = reflectance[idx]
            curr_R[:, 1] = 1
            R.append(curr_R)

            curr_T = np.zeros((self.num_wavelengths, 2), dtype=np.float32)
            curr_T[:, 0] = self.wavelengths
            curr_T[:, 1] = 0
            # curr_T[:, 1] = transmittance[idx]
            T.append(curr_T)
        dichroic_props = DichroicProps(theta, R, T)
        dichroic_surface.dichroic_props = dichroic_props
        return dichroic_surface

    def calc_R_T(self, n1, eta2, k2, num_angles):
        """
        Calculates the reflectance and transmittance for a range of incident angles.

        :param n1: The refractive index of the outer material.
        :type n1: float
        :param eta2: The real part of the refractive index of the inner material.
        :type eta2: float
        :param k2: The imaginary part of the refractive index of the inner material.
        :type k2: float
        :param num_angles: The number of angles to calculate.
        :type num_angles: int
        :return: A tuple containing the angles, reflectance, and transmittance.
        :rtype: tuple
        """
        n2 = np.complex(eta2, k2)
        theta_deg = np.linspace(0, 90, num_angles, dtype=np.float32)
        theta = np.radians(theta_deg)

        def r_s(theta):
            numerator = n1 * np.cos(theta) - n2 * np.sqrt(
                1 - np.square(n1 * np.sin(theta) / n2)
            )
            denominator = n1 * np.cos(theta) + n2 * np.sqrt(
                1 - np.square(n1 * np.sin(theta) / n2)
            )
            return np.divide(numerator, denominator)

        def R_s(theta):
            rs = r_s(theta)
            return np.square(np.absolute(rs))

        def r_p(theta):
            numerator = n1 * np.sqrt(
                1 - np.square(n1 * np.sin(theta) / n2)
            ) - n2 * np.cos(theta)
            denominator = n1 * np.sqrt(
                1 - np.square(n1 * np.sin(theta) / n2)
            ) + n2 * np.cos(theta)
            return np.divide(numerator, denominator)

        def R_p(theta):
            rp = r_p(theta)
            return np.square(np.absolute(rp))

        def R(theta):
            return 0.5 * (R_s(theta) + R_p(theta))

        reflectance = R(theta)
        transmittance = np.zeros(len(reflectance), dtype=np.float32)
        # absorption = 1 - reflectance
        return (theta, reflectance, transmittance)

    # Sili:
    def get_eta2_k2(self, eta2, k2):
        """
        Updates and retrieves the eta2 and k2 values.

        :param eta2: The real part of the refractive index of the inner material.
        :type eta2: float
        :param k2: The imaginary part of the refractive index of the inner material.
        :type k2: float
        :return: A tuple containing the updated eta2 and k2 values.
        :rtype: tuple
        """
        self.eta2 = eta2
        self.k2 = k2
        return (self.eta2, self.k2)
 

    def create_lobed_surface(self, name,inner_mat_name):
        curr_surface = Surface(name, model = 6)
        eta2 = float(self.mat_manager.material_props[inner_mat_name]["eta"])
        k2 = float(self.mat_manager.material_props[inner_mat_name]["k"])
        curr_surface.set("eta", eta2)
        curr_surface.set("k", k2)
        curr_surface.num_angles = 0
        
        #2D array initialization
        reflect_array_path = f"/workspace/data_files/data/{self.experiment_name}/{name}/reflect.csv"
        transmit_array_path = f"/workspace/data_files/data/{self.experiment_name}/{name}/transmit.csv"
        detect_array_path = f"/workspace/data_files/data/{self.experiment_name}/{name}/detect.csv"

        reflect_df = transmit_df = detect_df = None

        if os.path.exists(reflect_array_path):
            reflect_df = pd.read_csv(reflect_array_path)
        if os.path.exists(transmit_array_path):
            transmit_df = pd.read_csv(transmit_array_path)
        if os.path.exists(detect_array_path):
            detect_df = pd.read_csv(detect_array_path)
        
        dfs = [reflect_df, transmit_df, detect_df]

        if all(df is None for df in dfs):
            #all CSV files dont exist
            curr_surface.array_props_2D = None
            return curr_surface

        elif any(df is not None for df in dfs) and len({df.shape for df in dfs if df is not None}) == 1:
            numAngles = next(df.shape[0] for df in dfs if df is not None)
            array_props = ArrayProps2D(reflect_df.values.tolist(), reflect_df.values.tolist(), reflect_df.values.tolist())
            curr_surface.array_props_2D = array_props
            curr_surface.num_angles = numAngles
            return curr_surface

        else:
            raise IndexError("CSV files have mismatched dimensions") 




    def overwrite_property(self, surface, property, new_value):
        self.surfaces[surface].set(property, new_value)
