####################################################################################################
#
# Script for mass generating datasets. This script does not contain fancy conditions nor settings.
# Datasets tend to be generated only once and need an easy method for automatic generation.
# Generation of specific datasets are handled in blocks and can be enabled by removing the comments.
# Datasets are automatically generated in the /datasets/ sub-directory.
#
####################################################################################################

import os
import argparse

from SIFICCNN.data import RootSimulation
from SIFICCNN.utils import parent_directory
#from SIFICCNN.downloader import dSimulation_to_GraphCluster
from SIFICCNN.downloader import dSimulation_to_GraphSiPM

# get current path, go two subdirectories higher
path = parent_directory()

path_root = "~/Master/root_files/"
path_datasets = "/net/scratch_g4rt1/clement/datasets_aachen"

####################################################################################################
# Simulation GraphCluster
####################################################################################################

n = 10000 #685891
# files = ["1to1_Cluster_BP0mm_2e10protons_simV3.root"]
#files = ["1to1_Cluster_CONT_2e10protons_simV3.root"]
# NEW FILE 4to1_SiPM


def main(n, file, coordinate_system):
    root_simulation = RootSimulation(path_root + file)
    dSimulation_to_GraphSiPM(root_simulation=root_simulation,                       #Cluster statt SiPM
                                dataset_name=root_simulation.file_name,
                                path="",
                                n=n,
                                coordinate_system=coordinate_system,
                                energy_cut=None)
    
if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Trainings script ECRNCluster model')
    parser.add_argument("--n", type=int, help="Number of Events")
    parser.add_argument("--coordinates", type=str, help="Coordinate system")
    parser.add_argument("--file", type=str, help="File name")
    args = parser.parse_args()

    n                   = args.n if args.n is not None else None
    coordinate_system   = args.coordinates if args.coordinates is not None else "AACHEN"
    if args.file is not None:
        file = args.file
    else:
        raise ValueError("No File specified!")

    main(n, 
         file,
         coordinate_system
         )
