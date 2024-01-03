####################################################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
####################################################################################################


import numpy as np
import os
import argparse
import sys

from SIFICCNN.utils import TVector3, tVector_list, parent_directory


def dSimulation_to_GraphCluster(root_simulation,
                                dataset_name,
                                path="",
                                n=None,
                                coordinate_system="CRACOW",
                                energy_cut=None):
    """
    Script to generate a datasets in graph basis. Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array size, one to read the
    data. Final data is stored as npy files, separated by their usage.

    Args:
        root_simulation (RootSimulation):   root file container object
        dataset_name (str):                 final name of datasets for storage
        path (str):                         destination path, if not given it will default to
                                            /datasets in parent directory
        n (int or None):                    Number of events sampled from root file,
                                            if None all events are used
        energy_cut (float or None):         Energy cut applied to sum of all cluster energies,
                                            if None, no energy cut is applied
        coordinate_system (str):            Coordinate system of the given root file, everything
                                            will be converted to Aachen coordinate system

    """

    # generate directory and finalize path
    if path == "":
        path = parent_directory() + "/datasets/"
        path = os.path.join(path, "SimGraphCluster", dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    print("Loading root file: {}".format(root_simulation.file_name))
    print("Dataset name: {}".format(dataset_name))
    print("Path: {}".format(path))
    print("Energy Cut: {}".format(energy_cut))
    print("Coordinate system of root file: {}".format(coordinate_system))
    print("\nCounting number of graphs to be created")
    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        idx_scat, idx_abs = event.RecoCluster.sort_clusters_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            continue

        if energy_cut is not None:
            if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                continue
        k_graphs += 1
        n_nodes += len(event.RecoCluster.RecoClusterEntries)
        m_edges += len(event.RecoCluster.RecoClusterEntries) ** 2
    print("Total number of Graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Total number of edges to be created: ", m_edges)
    print("Graph features: {}".format(10))
    print("Graph targets: {}".format(9))

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool)
    ary_node_attributes = np.zeros(shape=(n_nodes, 10), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    ary_edge_attributes = np.zeros(shape=(m_edges, 3), dtype=np.float32)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)

    # main iteration over root file, containing beta coincidence check
    # NOTE:
    # "id" are here used for indexing instead of using the iteration variables i,j,k since some
    # events are skipped due to cuts or filters, therefore more controlled indexing is needed
    graph_id = 0
    node_id = 0
    edge_id = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        # get number of cluster
        n_cluster = int(len(event.RecoCluster.RecoClusterEntries))

        # coincidence check
        idx_scat, idx_abs = event.RecoCluster.sort_clusters_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            continue
        # energy cut if applied
        if energy_cut is not None:
            if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                continue

        # double iteration over every cluster to determine adjacency and edge features
        for j in range(n_cluster):
            for k in range(n_cluster):
                """
                # This is an example of a connection rule for adjacency
                if j in idx_abs and k in idx_scat:
                    continue
                """
                # determine edge attributes
                if j != k:
                    # grab edge features in polar and cartesian representation
                    r, phi, theta = event.RecoCluster.get_edge_features(j, k, cartesian=False)
                else:
                    r, phi, theta = 0, 0, 0
                ary_edge_attributes[edge_id, :] = [r, phi, theta]

                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            # Graph indicator counts up which node belongs to which graph
            ary_graph_indicator[node_id] = graph_id

            # collect node attributes for each node
            # exception for different coordinate systems
            if coordinate_system == "CRACOW":
                attributes = np.array([event.RecoCluster.RecoClusterEntries[j],
                                       event.RecoCluster.RecoClusterTimestamps[j],
                                       event.RecoCluster.RecoClusterEnergies_values[j],
                                       event.RecoCluster.RecoClusterEnergies_uncertainty[j],
                                       event.RecoCluster.RecoClusterPosition[j].z,
                                       -event.RecoCluster.RecoClusterPosition[j].y,
                                       event.RecoCluster.RecoClusterPosition[j].x,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].z,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].y,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].x])
                ary_node_attributes[node_id, :] = attributes
            if coordinate_system == "AACHEN":
                attributes = np.array([event.RecoCluster.RecoClusterEntries[j],
                                       event.RecoCluster.RecoClusterTimestamps[j],
                                       event.RecoCluster.RecoClusterEnergies_values[j],
                                       event.RecoCluster.RecoClusterEnergies_uncertainty[j],
                                       event.RecoCluster.RecoClusterPosition[j].x,
                                       event.RecoCluster.RecoClusterPosition[j].y,
                                       event.RecoCluster.RecoClusterPosition[j].z,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].x,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].y,
                                       event.RecoCluster.RecoClusterPosition_uncertainty[j].z])
                ary_node_attributes[node_id, :] = attributes

            # count up node indexing
            node_id += 1

        # grab target labels and attributes
        event.ph_method = 2
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[graph_id] = distcompton_tag * 1
        ary_pe[graph_id] = event.MCEnergy_Primary
        if coordinate_system == "CRACOW":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                 target_energy_p,
                                                 target_position_e.z,
                                                 -target_position_e.y,
                                                 target_position_e.x,
                                                 target_position_p.z,
                                                 -target_position_p.y,
                                                 target_position_p.x]
            ary_sp[graph_id] = event.MCPosition_source.z
        if coordinate_system == "AACHEN":
            ary_graph_attributes[graph_id, :] = [target_energy_e,
                                                 target_energy_p,
                                                 target_position_e.x,
                                                 target_position_e.y,
                                                 target_position_e.z,
                                                 target_position_p.x,
                                                 target_position_p.y,
                                                 target_position_p.z]
            ary_sp[graph_id] = event.MCPosition_source.x

        # count up graph indexing
        graph_id += 1

    # save up all files
    np.save(path + "/" + "A.npy", ary_A)
    np.save(path + "/" + "graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + "graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + "node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + "graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + "edge_attributes.npy", ary_edge_attributes)
    np.save(path + "/" + "graph_pe.npy", ary_pe)
    np.save(path + "/" + "graph_sp.npy", ary_sp)


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Simulation to GraphCluster downloader')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    parser.add_argument("-ec", type=float, help="Energy cut applied to sum of all cluster energies")
    args = parser.parse_args()

    dSimulation_to_GraphCluster(root_simulation=args.rf,
                                dataset_name=args.name,
                                path=args.path if args.path is not None else "",
                                n=args.n if args.n is not None else None,
                                coordinate_system=args.cs if args.cs is not None else "CRACOW",
                                energy_cut=args.ec if args.ec is not None else None)
