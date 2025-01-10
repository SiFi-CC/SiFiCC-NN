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
from SIFICCNN.downloader import dSimulation_to_GraphSiPM

# Get current path, go two subdirectories higher
path = parent_directory()

####################################################################################################
# Simulation GraphCluster
####################################################################################################


def main(n_stop, file, coordinate_system, n_start, neutrons):
    # Initialize RootSimulation with the specified file
    root_simulation = RootSimulation(file)
    
    # Convert simulation data to GraphSiPM format
    dSimulation_to_GraphSiPM(
        root_simulation    = root_simulation,
        dataset_name       = root_simulation.file_name,
        path               = "",
        n_stop             = n_stop,
        coordinate_system  = coordinate_system,
        energy_cut         = None,
        neutrons           = neutrons,
        n_start            = n_start
    )

    

if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Trainings script ECRNCluster model')
    
    # Add arguments to the parser
    parser.add_argument("--n_stop", type=int, help="Stop at Event")
    parser.add_argument("--coordinates", type=str, default="AACHEN", help="Coordinate system")
    parser.add_argument("--file", type=str, required=True, help="File name")
    parser.add_argument("--n_start", type=int, help="Start at Event")
    parser.add_argument("--neutrons", type=int, default=0, help="Set neutron parameters: 1 = on, 2 = filter for neutrons only, 3 = filter for photons only from neutron dataset")
    
    # Parse arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    n_stop = args.n_stop
    coordinate_system = args.coordinates
    n_start = args.n_start
    neutrons = args.neutrons
    file = args.file

    # Call the main function with parsed arguments
    main(n_stop, file, coordinate_system, n_start, neutrons)
