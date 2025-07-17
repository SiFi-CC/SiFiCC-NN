##########################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
##########################################################################

import numpy as np
import os
import argparse
import logging
import matplotlib.pyplot as plt
import awkward as ak

from SIFICCNN.utils import parent_directory
from SIFICCNN.utils.numba import make_all_edges

logging.basicConfig(level=logging.INFO)

def dSimulation_to_GraphSiPMCM(
     root_simulation,
        dataset_name,
        path="",
        coordinate_system="CRACOW",
        energy_cut=None,
        neutrons=0,
        n_start=0,
        n_stop=None,
):

    if path == "":
        path = parent_directory() + "/datasets/"
        path = os.path.join(path, "CMSimGraphSiPM", dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    
    # Name files for splitting using condor:
    name_additions = ""
    if n_start != 0:
        name_additions += "{}-".format(n_start)
    if n_stop is not None:
        name_additions += "{}".format(n_stop)
    if name_additions != "":
        name_additions = name_additions + "_"

    logging.info("Loading root file: {}".format(root_simulation.file_name))
    logging.info("Dataset name: {}".format(dataset_name))
    logging.info("Path: {}".format(path))
    logging.info("Energy Cut: {}".format(energy_cut))
    logging.info("Coordinate system of root file: {}".format(coordinate_system))
    logging.info("Starting to load events")

    # Create lists to store data for each batch of the iteration
    ary_A = []
    ary_graph_indicator = []
    ary_node_attributes = []
    ary_graph_attributes = []
    ary_graph_labels = []
    ary_fibre_positions = []
    ary_fibre_indicator = []
    ary_pe = []
    ary_source_position = []


    for i, batch in enumerate(root_simulation.iterate_events(n_stop=n_stop, n_start=n_start)):

        sipm_data = batch.sipm_hit
        fibre_data = batch.fibre_hit
        cluster_data = batch.cluster_hit
        n_clusters = len(cluster_data["ClusterEnergy"])
        n_nodes = ak.sum(ak.num(sipm_data["SiPMId"]))
        n_edges = ak.sum(ak.num(sipm_data["SiPMId"]) ** 2)
        
        logging.info(f"Total number of graphs to be created: {n_clusters}")
        logging.info(f"Total number of nodes to be created: {n_nodes}")
        logging.info(f"Total number of edges to be created: {n_edges}")

        # counting SiPMs per event
        nodes_per_cluster = ak.num(sipm_data["SiPMId"], axis=1).to_numpy()

        # --- build node attributes ---
        # Map SiPMHit positions to attribute arrays according to coordinate system.
        if coordinate_system.upper() == "CRACOW":
            x = ak.flatten(sipm_data["SiPMPosition"]["z"])
            y = -ak.flatten(sipm_data["SiPMPosition"]["y"])
            z = ak.flatten(sipm_data["SiPMPosition"]["x"])
        else:  # AACHEN
            x = ak.flatten(sipm_data["SiPMPosition"]["x"])
            y = ak.flatten(sipm_data["SiPMPosition"]["y"])
            z = ak.flatten(sipm_data["SiPMPosition"]["z"])
            
        timestamp = ak.flatten(sipm_data["SiPMTimeStamp"])
        photon_count = ak.flatten(sipm_data["SiPMPhotonCount"])

        # Convert each flattened array to NumPy
        x_np = ak.to_numpy(x)
        y_np = ak.to_numpy(y)
        z_np = ak.to_numpy(z)
        timestamp_np = ak.to_numpy(timestamp)
        photon_count_np = ak.to_numpy(photon_count)

        # Stack the arrays column-wise
        batch_node_attributes = np.column_stack((x_np, y_np, z_np, timestamp_np, photon_count_np))
        logging.info("Created node attributes")

        # --- graph indicator: assign each node its event id ---
        # First create a ragged array of cluster indices, then flatten.
        cluster_ids = ak.from_numpy(np.arange(len(sipm_data["SiPMId"])))
        batch_graph_indicator = ak.ravel(ak.broadcast_arrays(ak.local_index(sipm_data["SiPMId"]),
                                                    cluster_ids)[1]).to_numpy()
        
        cluster_ids = ak.from_numpy(np.arange(len(sipm_data["SiPMId"])))
        print("cluster_ids:", cluster_ids)

        local_idx = ak.local_index(sipm_data["SiPMId"])
        print("local_idx:", local_idx)

        broadcasted = ak.broadcast_arrays(local_idx, cluster_ids)
        print("Broadcasted cluster IDs:", broadcasted[1])

        batch_graph_indicator = ak.ravel(broadcasted[1]).to_numpy()
        print("batch_graph_indicator:", batch_graph_indicator)

        logging.info("Created graph indicator")
        
        batch_A = make_all_edges(np.ma.filled(nodes_per_cluster, fill_value=0))
        logging.info("Created adjacency matrix")

        # --- fibre positions ---
        # Assume FibreHit contains FibreId and FibrePosition which are vectorized in each event.
        if coordinate_system.upper() == "CRACOW":
            x = ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["z"]))
            y = -ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["y"]))
            z = ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["x"]))
        else:
            x = ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["x"]))
            y = ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["y"]))
            z = ak.to_numpy(ak.flatten(fibre_data["FibrePosition"]["z"]))

        batch_fibre_positions = np.column_stack((x, y, z))
        logging.info("Created fibre positions")
        # Build fibre indicator (graph id per fibre)
        batch_fibre_indicator = ak.to_numpy(ak.ravel(ak.broadcast_arrays(ak.local_index(fibre_data["FibreId"]),
                                                    cluster_ids)[1]))
        logging.info("Created fibre indicator")
        
        # --- graph-level attributes and labels ---

        logging.info("Calculated graph attributes and labels")

        # Prepare target data
        target_energy = np.ma.filled(ak.to_numpy(cluster_data["ClusterEnergy"]), 0)
        target_fibre_id = np.ma.filled(ak.to_numpy(cluster_data["ClusterFibreId"]), 0)
        if coordinate_system == "CRACOW":
            x = ak.to_numpy(cluster_data["ClusterPosition"]["z"])
            y = -ak.to_numpy(cluster_data["ClusterPosition"]["y"])
            z = ak.to_numpy(cluster_data["ClusterPosition"]["x"])
        else:
            x = ak.to_numpy(cluster_data["ClusterPosition"]["x"])
            y = ak.to_numpy(cluster_data["ClusterPosition"]["y"])
            z = ak.to_numpy(cluster_data["ClusterPosition"]["z"])
        target_position = np.column_stack((x, y, z))
        batch_graph_attributes = np.column_stack((target_energy, target_position, target_fibre_id))
        batch_graph_labels = np.ones(n_clusters, dtype=np.bool_)

        # get source positions
        if coordinate_system.upper() == "CRACOW":
            x = ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["z"])
            y = -ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["y"])
            z = ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["x"])
        else:
            x = ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["x"])
            y = ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["y"])
            z = ak.to_numpy(cluster_data["Cluster_MCPosition_source"]["z"])
        batch_source_position = np.column_stack((x, y, z))

        logging.info("Created graph attributes")

        # Primary energies per event
        batch_pe = np.ma.filled((ak.to_numpy(batch.MCEnergyPrimary)),0) # fill missing values with 0
        
        # Graph labels as boolean (convert tag to int if desired)
        batch_graph_labels = np.ones(n_clusters, dtype=np.bool_) 

        # Before appending the indicators, check last index of the previous batch
        if i > 0:
            batch_graph_indicator += ary_graph_indicator[-1][-1] + 1
            batch_fibre_indicator += ary_fibre_indicator[-1][-1] + 1

        # Append data to lists
        ary_A.append(batch_A)
        ary_graph_indicator.append(batch_graph_indicator)
        ary_node_attributes.append(batch_node_attributes)
        ary_graph_attributes.append(batch_graph_attributes)
        ary_graph_labels.append(batch_graph_labels)
        ary_fibre_positions.append(batch_fibre_positions)
        ary_fibre_indicator.append(batch_fibre_indicator)
        ary_pe.append(batch_pe)
        ary_source_position.append(batch_source_position)

    # Concatenate lists to arrays
    ary_A = np.concatenate(ary_A, axis=0)
    ary_graph_indicator = np.concatenate(ary_graph_indicator, axis=0)
    ary_node_attributes = np.concatenate(ary_node_attributes, axis=0)
    ary_graph_attributes = np.concatenate(ary_graph_attributes, axis=0)
    ary_graph_labels = np.concatenate(ary_graph_labels, axis=0)
    ary_fibre_indicator = np.concatenate(ary_fibre_indicator, axis=0)
    ary_fibre_positions = np.concatenate(ary_fibre_positions, axis=0)
    ary_pe = np.concatenate(ary_pe, axis=0)
    ary_source_position = np.concatenate(ary_source_position, axis=0)


    # Save datasets as .npy files 
    np.save(path + "/" + name_additions + "A.npy", ary_A)
    np.save(path + "/" + name_additions + "graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + name_additions + "node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + name_additions + "graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + name_additions + "graph_pe.npy", ary_pe)
    np.save(
        path + "/" + name_additions + "fibre_indicator.npy", ary_fibre_indicator
    )
    np.save(path + "/" + name_additions + "graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + name_additions + "fibre_positions.npy", ary_fibre_positions)
    np.save(path + "/" + name_additions + "source_positions.npy", ary_source_position)
    logging.info("Datasets saved successfully")
    

if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description="Simulation to GraphSiPM downloader")
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    parser.add_argument(
        "--with_neutrons", action="store_true", help="Include events with neutrons"
    )
    parser.add_argument("--photon_set", action="store_true", help="Include photon set")
    args = parser.parse_args()

    dSimulation_to_GraphSiPMCM(
        root_simulation=args.rf,
        dataset_name=args.name,
        path=args.path if args.path is not None else "",
        n=args.n if args.n is not None else None,
        coordinate_system=args.cs if args.cs is not None else "CRACOW",
        energy_cut=args.ec if args.ec is not None else None,
        with_neutrons=args.with_neutrons,
        photon_set=args.photon_set,
    )
