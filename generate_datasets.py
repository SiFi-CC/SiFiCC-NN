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
path_datasets = "/net/scratch_g4rt1/clement/datasets"

####################################################################################################
# Simulation GraphCluster
####################################################################################################

def main(n, with_neutrons, photon_set, coordinate_system):
    # files = ["1to1_Cluster_BP0mm_2e10protons_simV3.root"]
    #files = ["1to1_Cluster_CONT_2e10protons_simV3.root"]
    # NEW FILE 4to1_SiPM
    if not photon_set:
        files = ["OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons.root"]
    else:
        files = ["OptimisedGeometry_4to1_0mm_4e9protons_simv4.root"]

    for file in files:
        root_simulation = RootSimulation(path_root + file)
        dSimulation_to_GraphSiPM(root_simulation=root_simulation,                       #Cluster statt SiPM
                                    dataset_name=root_simulation.file_name,
                                    path="",
                                    n=n,
                                    coordinate_system=coordinate_system,
                                    energy_cut=None,
                                    with_neutrons=with_neutrons,
                                    photon_set=photon_set)

if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Trainings script ECRNCluster model')
    parser.add_argument("--n", type=int, help="Number of Events")
    parser.add_argument("--coordinates", type=str, help="Coordinate system")
    parser.add_argument("--neutrons", action=argparse.BooleanOptionalAction, help="Dataset consisting of only neutron events or without any neutrons")
    parser.add_argument("--photon_set", action=argparse.BooleanOptionalAction, help="Dataset does not contain any neutrons, only prompt gammas")

    args = parser.parse_args()

    n                   = args.n if args.n is not None else None
    coordinate_system   = args.coordinates if args.coordinates is not None else "CRACOW"
    photon_set          = args.photon_set if args.photon_set is not None else True
    with_neutrons       = args.neutrons if (args.neutrons is not None and not photon_set) else False

    main(n, 
         with_neutrons, 
         photon_set,
         coordinate_system
         )




