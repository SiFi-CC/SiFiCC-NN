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
import matplotlib.pyplot as plt

from SIFICCNN.utils import TVector3, tVector_list, parent_directory


def dSimulation_to_GraphSiPM(root_simulation,
                             dataset_name,
                             path="",
                             n=None,
                             coordinate_system="CRACOW",
                             energy_cut=None,
                             with_neutrons=False,
                             photon_set=True):
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

    #DEBUG
    PrimaryEnergies = list()
    NeutronCount = list()
    def plot_neutrons_vs_energy(NeutronCount, NeutronPrimaryEnergies, key):

        # Create a histogram
        plt.figure(figsize=(8, 6))
        plt.hist2d(NeutronCount, NeutronPrimaryEnergies, bins=30, cmap="Reds")
        plt.title(key+' Count vs Primary Energies')
        plt.xlabel(key+' Count')
        plt.ylabel(key+' Primary Energies')
        plt.colorbar(label='Counts')

        # Show the plot
        plt.savefig("EnergyNeutronHist_"+neutron_key+".png")
        plt.close()

    def plot_primary_energy(NeutronPrimaryEnergies, neutron_key,compton=False):
        if compton:
            end = "Compton"
        else:
            end = ""
        # Create a histogram
        plt.figure(figsize=(8, 6))
        plt.hist(NeutronPrimaryEnergies, bins=np.arange(0,20,0.2))
        plt.title('Primary Energies'+' '+neutron_key+' '+end)
        plt.xlabel('Energies / MeV')
        plt.ylabel('Primary Energies')
        plt.grid()
        plt.xlim(left=0,right=20)

        # Show the plot
        plt.savefig("EnergySpectrum_"+neutron_key+end+".png")
        plt.close()

    def stacked_primary_energy(compton, not_compton, neutron_key):
        plt.figure(figsize=(8, 6))
        
        # Create the histogram with stacking
        plt.hist([compton, not_compton], bins=np.arange(0, 20, 0.2), stacked=True, label=["Compton", "Not Compton"])
        
        plt.title('Primary Energies ' + neutron_key)
        plt.xlabel('Energies / MeV')
        plt.ylabel('Primary Energies')
        plt.grid()
        plt.xlim(left=0, right=20)
        plt.legend()
        
        # Save the plot
        plt.savefig("EnergySpectrumStacked_" + neutron_key + ".png")
        plt.close()

    # generate directory and finalize path
    if with_neutrons:
        neutron_key = "Neutrons"
    else:
        neutron_key = "NoNeutrons"
    if photon_set:
        neutron_key = "OldDataset"
    
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
    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        if event == None:
            continue
        if (event.MCNPrimaryNeutrons == 0 and not with_neutrons) or (event.MCNPrimaryNeutrons != 0 and with_neutrons) or photon_set:
            idx_scat, idx_abs = event.SiPMHit.sort_sipm_by_module()
            if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
                continue
            """
            # DISABLED FOR NOW AS NO RECO AVAILABLE TO FORM ENERGY FROM SIPM HITS
            if energy_cut is not None:
                if sum(event.RecoCluster.RecoClusterEnergies_values) < energy_cut:
                    continue
            """
            k_graphs += 1
            n_nodes += len(event.SiPMHit.SiPMId)
            m_edges += len(event.SiPMHit.SiPMId) ** 2
            if not photon_set:
                NeutronCount.append(event.MCNPrimaryNeutrons)
            PrimaryEnergies.append(event.MCEnergy_Primary)

    if not photon_set:
        plot_neutrons_vs_energy(NeutronCount, PrimaryEnergies, neutron_key)
    plot_primary_energy(PrimaryEnergies, neutron_key)
    print("Total number of Graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Total number of edges to be created: ", m_edges)
    print("Graph features: {}".format(10))
    print("Graph targets: {}".format(9))

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int32)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int32)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool_)
    ary_node_attributes = np.zeros(shape=(n_nodes, 5), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    ary_edge_attributes = np.zeros(shape=(m_edges, 5), dtype=np.float32)
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

    distcompton_tags=list()
    for i, event in enumerate(root_simulation.iterate_events(n=n)):
        # get number of cluster
        if event == None:
            continue
        if (event.MCNPrimaryNeutrons == 0 and not with_neutrons) or (event.MCNPrimaryNeutrons != 0 and with_neutrons) or photon_set:
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
                        r, phi, theta, dt, dPhotonCount = event.SiPMHit.get_edge_features(j, k, cartesian=False)
                    else:
                        r, phi, theta, dt, dPhotonCount = 0, 0, 0, 0, 0 
                    ary_edge_attributes[edge_id, :] = [r, phi, theta, dt, dPhotonCount]

                    ary_A[edge_id, :] = [node_id, node_id - j + k]
                    edge_id += 1

                # Graph indicator counts up which node belongs to which graph
                ary_graph_indicator[node_id] = graph_id

                # collect node attributes for each node
                # exception for different coordinate systems
                if coordinate_system == "CRACOW":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].z,
                                        -event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
                    ary_node_attributes[node_id, :] = attributes
                if coordinate_system == "AACHEN":
                    attributes = np.array([event.SiPMHit.SiPMPosition[j].x,
                                        event.SiPMHit.SiPMPosition[j].y,
                                        event.SiPMHit.SiPMPosition[j].z,
                                        event.SiPMHit.SiPMTimeStamp[j],
                                        event.SiPMHit.SiPMPhotonCount[j]])
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

            distcompton_tags.append(distcompton_tag)

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

    if not photon_set:
        NeutronCount=np.array(NeutronCount)
    PrimaryEnergies=np.array(PrimaryEnergies)
    distcompton_tags=np.array(distcompton_tags)
    ComptonPrimaryEnergies=PrimaryEnergies[distcompton_tags]
    NotComptonPrimaryEnergies=PrimaryEnergies[np.logical_not(distcompton_tags)]
    plot_primary_energy(ComptonPrimaryEnergies, neutron_key,compton=True)
    stacked_primary_energy(ComptonPrimaryEnergies,NotComptonPrimaryEnergies,neutron_key)
    np.save(path + "/" + "ComptonPrimaryEnergies.npy", ComptonPrimaryEnergies)


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
