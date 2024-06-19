####################################################################################################
#
# This script converts a SiFi-CC simulation root file to a python readable datasets ready to be
# used for Neural Network training.
#
####################################################################################################


"""
Condition changed manually
"""

import numpy as np
import os
import argparse
import sys

from SIFICCNN.utils import TVector3, tVector_list, parent_directory


def dSimulation_to_GraphSiPM(root_simulation,
                             dataset_name,
                             path="",
                             n=None,
                             coordinate_system="CRACOW",
                             energy_cut=None,
                             with_neutrons=False):
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
    if with_neutrons:
        neutron_key = "Neutrons"
    else:
        neutron_key = "NoNeutrons"
    
    if path == "":
        path = parent_directory() + "/datasets/"
        path = os.path.join(path, "SimGraphSiPM",neutron_key, dataset_name)
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
    k_graphs_NoNeutron = 0
    n_nodes_NoNeutron = 0
    m_edges_NoNeutron = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        if event == None:
            continue
        if (event.MCNPrimaryNeutrons == 0 and not with_neutrons) or (event.MCNPrimaryNeutrons != 0 and with_neutrons):
            idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
            if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
                continue
            """
            # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS
            if energy_cut is not None:
                if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                    continue
            """
            k_graphs_NoNeutron += 1
            n_nodes_NoNeutron += len(event.SiPMHit.SiPMId)
            m_edges_NoNeutron += len(event.SiPMHit.SiPMId) ** 2
    print("Total number of Graphs to be created: ", k_graphs_NoNeutron)
    print("Total number of nodes to be created: ", n_nodes_NoNeutron)
    print("Total number of edges to be created: ", m_edges_NoNeutron)
    print("Graph features: {}".format(10))
    print("Graph targets: {}".format(9))

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A_NoNeutron = np.zeros(shape=(m_edges_NoNeutron, 2), dtype=np.int32)
    ary_graph_indicator_NoNeutron = np.zeros(shape=(n_nodes_NoNeutron,), dtype=np.int32)
    ary_graph_labels_NoNeutron = np.zeros(shape=(k_graphs_NoNeutron,), dtype=np.bool_)
    ary_node_attributes_NoNeutron = np.zeros(shape=(n_nodes_NoNeutron, 5), dtype=np.float32)
    ary_graph_attributes_NoNeutron = np.zeros(shape=(k_graphs_NoNeutron, 8), dtype=np.float32)
    ary_edge_attributes_NoNeutron = np.zeros(shape=(m_edges_NoNeutron, 3), dtype=np.float32)
    # meta data
    ary_pe_NoNeutron = np.zeros(shape=(k_graphs_NoNeutron,), dtype=np.float32)
    ary_sp_NoNeutron = np.zeros(shape=(k_graphs_NoNeutron,), dtype=np.float32)

    # main iteration over root file, containing beta coincidence check
    # NOTE:
    # "id" are here used for indexing instead of using the iteration variables i,j,k since some
    # events are skipped due to cuts or filters, therefore more controlled indexing is needed
    graph_id_NoNeutron = 0
    node_id_NoNeutron = 0
    edge_id_NoNeutron = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        # get number of cluster
        if event == None:
            continue
        if (event.MCNPrimaryNeutrons == 0 and not with_neutrons) or (event.MCNPrimaryNeutrons != 0 and with_neutrons):
            n_sipm = int(len(event.SiPMHit.SiPMId))

            # coincidence check
            idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
            if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
                continue
            """
            # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS   
            # energy cut if applied
            if energy_cut is not None:
                if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                    continue
            """
            # double iteration over every cluster to determine adjacency and edge features
            for j in range(n_sipm):
                for k in range(n_sipm):
                    """
                    # This is an example of a connection rule for adjacency
                    if j in idx_abs and k in idx_scat:
                        continue
                    """
                    # determine edge attributes
                    if j != k:
                        # grab edge features in polar and cartesian representation
                        r, phi, theta = event.SiPMHit.get_edge_features(j, k, cartesian=False)
                    else:
                        r, phi, theta = 0, 0, 0
                    ary_edge_attributes_NoNeutron[edge_id_NoNeutron, :] = [r, phi, theta]

                    ary_A_NoNeutron[edge_id_NoNeutron, :] = [node_id_NoNeutron, node_id_NoNeutron - j + k]
                    edge_id_NoNeutron += 1

                # Graph indicator counts up which node belongs to which graph
                ary_graph_indicator_NoNeutron[node_id_NoNeutron] = graph_id_NoNeutron

                # collect node attributes for each node
                # exception for different coordinate systems
                if coordinate_system == "CRACOW":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                        -event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
                    ary_node_attributes_NoNeutron[node_id_NoNeutron, :] = attributes
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].z,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
                    ary_node_attributes_NoNeutron[node_id_NoNeutron, :] = attributes

                # count up node indexing
                node_id_NoNeutron += 1

            # grab target labels and attributes
            event.ph_method = 2
            distcompton_tag = event.get_distcompton_tag()
            target_energy_e, target_energy_p = event.get_target_energy()
            target_position_e, target_position_p = event.get_target_position()
            ary_graph_labels_NoNeutron[graph_id_NoNeutron] = distcompton_tag * 1
            ary_pe_NoNeutron[graph_id_NoNeutron] = event.MCEnergy_Primary
            if coordinate_system == "CRACOW":
                ary_graph_attributes_NoNeutron[graph_id_NoNeutron, :] = [target_energy_e,
                                                    target_energy_p,
                                                    target_position_e.z,
                                                    -target_position_e.y,
                                                    target_position_e.x,
                                                    target_position_p.z,
                                                    -target_position_p.y,
                                                    target_position_p.x]
                ary_sp_NoNeutron[graph_id_NoNeutron] = event.MCPosition_source.z
            if coordinate_system == "AACHEN":
                ary_graph_attributes_NoNeutron[graph_id_NoNeutron, :] = [target_energy_e,
                                                    target_energy_p,
                                                    target_position_e.x,
                                                    target_position_e.y,
                                                    target_position_e.z,
                                                    target_position_p.x,
                                                    target_position_p.y,
                                                    target_position_p.z]
                ary_sp_NoNeutron[graph_id_NoNeutron] = event.MCPosition_source.x

            # count up graph indexing
            graph_id_NoNeutron += 1

    # save up all files
    np.save(path + "/" + "A.npy", ary_A_NoNeutron)
    np.save(path + "/" + "graph_indicator.npy", ary_graph_indicator_NoNeutron)
    np.save(path + "/" + "graph_labels.npy", ary_graph_labels_NoNeutron)
    np.save(path + "/" + "node_attributes.npy", ary_node_attributes_NoNeutron)
    np.save(path + "/" + "graph_attributes.npy", ary_graph_attributes_NoNeutron)
    np.save(path + "/" + "edge_attributes.npy", ary_edge_attributes_NoNeutron)
    np.save(path + "/" + "graph_pe.npy", ary_pe_NoNeutron)
    np.save(path + "/" + "graph_sp.npy", ary_sp_NoNeutron)


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Simulation to GraphSiPM downloader')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--name", type=str, help="Name of final datasets")
    parser.add_argument("-path", type=str, help="Path to final datasets")
    parser.add_argument("-n", type=int, help="Number of events used")
    parser.add_argument("-cs", type=str, help="Coordinate system of root file")
    # parser.add_argument("-ec", type=float, help="Energy cut applied to sum of all cluster energies")
    args = parser.parse_args()

    dSimulation_to_GraphSiPM(root_simulation=args.rf,
                             dataset_name=args.name,
                             path=args.path if args.path is not None else "",
                             n=args.n if args.n is not None else None,
                             coordinate_system=args.cs if args.cs is not None else "CRACOW",
                             energy_cut=args.ec if args.ec is not None else None)
