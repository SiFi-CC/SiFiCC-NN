####################################################################################################
#
# Script for mass generating datasets. This script does not contain fancy conditions nor settings.
# Datasets tend to be generated only once and need an easy method for automatic generation.
# Generation of specific datasets are handled in blocks and can be enabled by removing the comments.
# Datasets are automatically generated in the /datasets/ sub-directory.
#
####################################################################################################

import os

from SIFICCNN.data import RootSimulation
from SIFICCNN.utils import parent_directory
#from SIFICCNN.downloader import dSimulation_to_GraphCluster
from SIFICCNN.downloader import dSimulation_to_GraphSiPM

# get current path, go two subdirectories higher
path = parent_directory()

path_root = "~/Master/root_files/"
path_datasets = "/net/scratch_g4rt1/clement/datasets_small/"

####################################################################################################
# Simulation GraphCluster
####################################################################################################

n = 10000 #685891
# files = ["1to1_Cluster_BP0mm_2e10protons_simV3.root"]
#files = ["1to1_Cluster_CONT_2e10protons_simV3.root"]
# NEW FILE 4to1_SiPM
files = ["OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons.root"]

for file in files:
    root_simulation = RootSimulation(path_root + file)
    dSimulation_to_GraphSiPM(root_simulation=root_simulation,                       #Cluster statt SiPM
                                dataset_name=root_simulation.file_name,
                                path="",
                                n=n,
                                coordinate_system="CRACOW",
                                energy_cut=None)
